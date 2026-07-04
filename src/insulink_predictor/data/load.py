"""Real PostgreSQL loader (ROADMAP §2 real loader; resolves §7).

The production schema **has insulin and carbs** (``bolus_entries``) plus pumps and
sensors → this is a **Type-1, insulin-using** population, not the wellness wedge.
Consequences (§7): COB/IOB become primary predictors, and the regulatory line
shifts toward a medical device — surface as "patterns/insights", not dosing.

Confirmed against a real data sample:
- ``recorded_at`` is **unix epoch milliseconds**; ``glucose`` is **mg/dL** (also
  ``user_settings.content.glucose_unit == "mgdl"``).
- CGM is **Dexcom G7** (``sensors.type``) → native 5-min cadence (grid_minutes=5).
- ``bolus_entries.glucose`` is a real SMBG at bolus time → folded into the glucose
  channel to enrich coverage.
- ``events`` are **alerts** (e.g. ``glucose_high``), *derived from glucose*, so they
  are deliberately NOT used as inputs (would be circular / leaky).
- ``sport_measurements.type`` includes ``WEIGHT`` (a user-static attribute, not a
  30-60 min forecasting signal); only heart-rate/steps types feed the grid.

Design: I/O (``connect`` / ``fetch_tables`` / ``inspect``) is separated from pure
transformation (``assemble_raw`` and the unit/timestamp helpers) so the mapping
logic is fully testable without a live database. The loader emits exactly the RAW
shape ``synth.generate`` produces, so ``align`` + the whole pipeline are reused
unchanged.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config, PostgresConfig

# The RAW contract (identical to synth.generate's output) that align() consumes.
RAW_COLUMNS = [
    "user_id",
    "ts_utc",
    "ts_local",
    "glucose_mgdl",
    "meal_flag",
    "carbs_g",
    "insulin_u",
    "steps",
    "activity_flag",
    "hr",
    "weather_temp",
    "daily_steps",
    "daily_distance",
    "isf",
    "icr",
]

# Heuristic type maps for sport_measurements (confirm against `gf db-inspect`).
_HR_TYPES = {"heart_rate", "heartrate", "hr", "bpm", "pulse"}
_STEP_TYPES = {"steps", "step_count", "steps_count", "step"}

MMOL_TO_MGDL = 18.0182


# --------------------------------------------------------------------------- #
# Pure helpers (no DB) — unit / timestamp detection & conversion              #
# --------------------------------------------------------------------------- #
def detect_ts_unit(values: pd.Series | np.ndarray) -> str:
    """Guess whether a bigint epoch column is in 'ms' or 's' from its magnitude."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return "ms"
    med = float(v.median())
    return "ms" if med > 1e11 else "s"  # ~1e12 for ms, ~1e9 for s in this era


def to_datetime_utc(values: pd.Series, ts_unit: str) -> pd.Series:
    """Epoch bigint -> tz-aware UTC datetime (auto-detect unit if requested)."""
    unit = detect_ts_unit(values) if ts_unit == "auto" else ts_unit
    return pd.to_datetime(pd.to_numeric(values, errors="coerce"), unit=unit, utc=True)


def detect_glucose_unit(values: pd.Series | np.ndarray) -> str:
    """mg/dL (~40-400) vs mmol/L (~2-22) from the central tendency."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return "mg/dL"
    return "mmol/L" if float(v.median()) < 30.0 else "mg/dL"


def to_mgdl(values: pd.Series, source_unit: str) -> pd.Series:
    """Convert a glucose series to mg/dL (§7.3). mmol/L → ×18."""
    unit = detect_glucose_unit(values) if source_unit == "auto" else source_unit
    v = pd.to_numeric(values, errors="coerce")
    return v * MMOL_TO_MGDL if unit == "mmol/L" else v


def _plausible_glucose(values: pd.Series) -> np.ndarray:
    """NaN out non-physiological glucose (e.g. the 1.0 no-reading placeholder)."""
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return np.where((v >= 20.0) & (v <= 500.0), v, np.nan)


def _is_intraday(ts, grid_minutes: int) -> bool:
    """True if a measurement stream is intraday (not a daily/sparse aggregate).

    Guards against mapping DAILY aggregates into the 5-min grid: in the real data
    STEPS/DISTANCE/CALORIES arrive once per day (median gap ~24h), so a day-total
    would otherwise be dumped into a single bucket and corrupt the steps features.
    """
    s = pd.Series(pd.to_datetime(ts, utc=True)).sort_values()
    if len(s) < 3:
        return False
    med_gap_min = s.diff().dropna().dt.total_seconds().median() / 60.0
    return med_gap_min <= max(60.0, grid_minutes * 6)


def _expand_intervals(
    df: pd.DataFrame, grid_minutes: int, max_hours: int = 6
) -> pd.DataFrame:
    """Expand [start,end] activity windows to grid-spaced timestamps (bounded)."""
    rows = []
    freq = f"{grid_minutes}min"
    for _, r in df.iterrows():
        start, end = r["_start"], r["_end"]
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        end = min(end, start + pd.Timedelta(hours=max_hours))
        for ts in pd.date_range(start, end, freq=freq):
            rows.append({"user_id": r["user_id"], "ts_utc": ts})
    return pd.DataFrame(rows, columns=["user_id", "ts_utc"])


def _channel_block(user_id, ts_utc, **channels) -> pd.DataFrame:
    """One source's rows: given channels set, all other RAW channels defaulted."""
    ts = pd.Series(pd.to_datetime(ts_utc, utc=True)).reset_index(drop=True)
    n = len(ts)
    block = {
        "user_id": pd.Series(np.asarray(user_id)).astype(str).reset_index(drop=True),
        "ts_utc": ts,
        "glucose_mgdl": np.full(n, np.nan),
        "meal_flag": np.zeros(n, dtype=bool),
        "carbs_g": np.full(n, np.nan),
        "insulin_u": np.full(n, np.nan),
        "steps": np.full(n, np.nan),
        "activity_flag": np.zeros(n, dtype=bool),
        "hr": np.full(n, np.nan),
        "weather_temp": np.full(n, np.nan),
        "daily_steps": np.full(n, np.nan),
        "daily_distance": np.full(n, np.nan),
    }
    block.update({k: np.asarray(v) for k, v in channels.items()})
    return pd.DataFrame(block)


def assemble_raw(tables: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Map fetched tables into the RAW contract frame (pure; align-ready).

    ``tables`` may contain: glucose_entries, bolus_entries, sport_measurements,
    sport_trainings, sport_workouts. Missing tables/channels degrade cleanly.
    """
    pg = cfg.pg
    blocks: list[pd.DataFrame] = []

    # --- glucose (CGM signal) ----------------------------------------------
    g = tables.get("glucose_entries")
    if g is not None and len(g):
        blocks.append(
            _channel_block(
                g["user_id"],
                to_datetime_utc(g["recorded_at"], pg.ts_unit),
                glucose_mgdl=_plausible_glucose(
                    to_mgdl(g["value"], pg.source_glucose_unit)
                ),
            )
        )

    # --- bolus (carbs + insulin => COB/IOB, now primary) -------------------
    # Also carries the pre-bolus SMBG (bolus_entries.glucose) — a real glucose
    # reading that enriches the CGM channel and fills gaps.
    b = tables.get("bolus_entries")
    if b is not None and len(b):
        carbs = pd.to_numeric(b["carbohydrates"], errors="coerce").to_numpy()
        smbg = (
            to_mgdl(b["glucose"], pg.source_glucose_unit)
            if "glucose" in b
            else pd.Series(np.full(len(b), np.nan))
        )
        blocks.append(
            _channel_block(
                b["user_id"],
                to_datetime_utc(b["recorded_at"], pg.ts_unit),
                glucose_mgdl=_plausible_glucose(
                    smbg
                ),  # drops no-reading placeholders (e.g. 1.0)
                carbs_g=np.where(carbs > 0, carbs, np.nan),
                insulin_u=pd.to_numeric(b["insulin"], errors="coerce").to_numpy(),
                meal_flag=(carbs > 0),
            )
        )

    # --- sport_measurements (hr / steps by type, intraday only) ------------
    # In the real schema this table is DAILY aggregates (STEPS/DISTANCE/CALORIES,
    # one row/day) + sporadic WEIGHT — none are an intraday activity stream. The
    # cadence gate below excludes daily aggregates so they don't corrupt the grid;
    # only genuinely intraday hr/steps data (e.g. from a wearable) is mapped.
    sm = tables.get("sport_measurements")
    if sm is not None and len(sm):
        t = sm["type"].astype(str).str.lower()
        ts = to_datetime_utc(sm["recorded_at"], pg.ts_unit)
        val = pd.to_numeric(sm["value"], errors="coerce")
        for channel, typeset in (("hr", _HR_TYPES), ("steps", _STEP_TYPES)):
            mask = t.isin(typeset).to_numpy()
            if mask.any() and _is_intraday(ts[mask], cfg.grid_minutes):
                blocks.append(
                    _channel_block(
                        sm["user_id"].to_numpy()[mask],
                        ts[mask],
                        **{channel: val.to_numpy()[mask]},
                    )
                )

        # daily STEPS/DISTANCE totals -> a per-day activity context, carried on
        # their own daily_* columns (broadcast per day in align, lagged causally to
        # "yesterday" in build_features). These are the DAILY aggregates that the
        # intraday gate above deliberately keeps out of the 5-min `steps` channel.
        for channel, typ in (("daily_steps", "steps"), ("daily_distance", "distance")):
            dmask = (t == typ).to_numpy()
            if dmask.any():
                blocks.append(
                    _channel_block(
                        sm["user_id"].to_numpy()[dmask],
                        ts[dmask],
                        **{channel: val.to_numpy()[dmask]},
                    )
                )

    # --- activity windows (trainings + workouts) ---------------------------
    act_frames = []
    st = tables.get("sport_trainings")
    if st is not None and len(st):
        act_frames.append(
            pd.DataFrame(
                {
                    "user_id": st["user_id"].astype(str),
                    "_start": to_datetime_utc(st["started_at"], pg.ts_unit),
                    "_end": to_datetime_utc(st["ended_at"], pg.ts_unit),
                }
            )
        )
    sw = tables.get("sport_workouts")
    if sw is not None and len(sw):
        start = to_datetime_utc(sw["started_at"], pg.ts_unit)
        act_frames.append(
            pd.DataFrame(
                {
                    "user_id": sw["user_id"].astype(str),
                    "_start": start,
                    "_end": start + pd.Timedelta(minutes=45),
                }  # assume a default session length
            )
        )
    if act_frames:
        expanded = _expand_intervals(
            pd.concat(act_frames, ignore_index=True), cfg.grid_minutes
        )
        if len(expanded):
            blocks.append(
                _channel_block(
                    expanded["user_id"],
                    expanded["ts_utc"],
                    activity_flag=np.ones(len(expanded), bool),
                )
            )

    if not blocks:
        return pd.DataFrame(columns=RAW_COLUMNS)

    raw = pd.concat(blocks, ignore_index=True)
    raw = raw.dropna(subset=["ts_utc"])
    raw["ts_local"] = raw["ts_utc"].dt.tz_convert(pg.local_tz).dt.tz_localize(None)

    # --- per-user therapy settings (ISF/ICR) from user_settings ------------
    settings = _settings_table(tables.get("user_settings"), cfg)
    raw = raw.merge(settings, on="user_id", how="left")
    raw["isf"] = raw["isf"].fillna(cfg.features.default_isf)
    raw["icr"] = raw["icr"].fillna(cfg.features.default_icr)

    return raw[RAW_COLUMNS].sort_values(["user_id", "ts_utc"]).reset_index(drop=True)


def parse_settings(content) -> dict:
    """Extract ISF/ICR from a user_settings.content JSON blob (robust to junk)."""
    try:
        c = json.loads(content) if isinstance(content, str) else (content or {})
    except (ValueError, TypeError):
        c = {}

    def num(key):
        try:
            return float(c.get(key))
        except (TypeError, ValueError):
            return None

    return {"isf": num("bolus_correction_factor"), "icr": num("bolus_carb_factor")}


def _settings_table(us: Optional[pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Per-user ISF/ICR frame from user_settings; empty frame if unavailable."""
    if us is None or not len(us):
        return pd.DataFrame(columns=["user_id", "isf", "icr"])
    rows = []
    for _, r in us.iterrows():
        p = parse_settings(r.get("content"))
        rows.append({"user_id": str(r["user_id"]), "isf": p["isf"], "icr": p["icr"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# I/O layer (SQLAlchemy + psycopg3)                                           #
# --------------------------------------------------------------------------- #
def connect(cfg: Config):
    """Create a SQLAlchemy engine (lazy import so this module loads without a DB)."""
    from sqlalchemy import create_engine

    return create_engine(cfg.pg.url(), pool_pre_ping=True)


def _q(schema: str, name: str) -> str:
    return f'"{schema}"."{name}"'


def fetch_tables(
    cfg: Config, engine=None, since=None, user_ids: Optional[list[str]] = None
) -> dict[str, pd.DataFrame]:
    """Fetch the forecasting-relevant tables into DataFrames.

    ``since`` (a pandas-parseable datetime) filters time-series rows server-side;
    ``user_ids`` restricts to specific users. Compliance filter is pushed down.
    """
    from sqlalchemy import text

    pg = cfg.pg
    engine = engine or connect(cfg)
    ts_unit = (
        detect_ts_unit(_probe_recorded_at(engine, pg))
        if pg.ts_unit == "auto"
        else pg.ts_unit
    )
    since_epoch = None
    if since is not None:
        secs = int(pd.Timestamp(since).timestamp())
        since_epoch = secs * 1000 if ts_unit == "ms" else secs

    def where(time_col: str) -> tuple[str, dict]:
        clauses, params = [], {}
        if pg.only_compliant:
            clauses.append(
                f"user_id IN (SELECT id FROM {_q(pg.db_schema, 'users')} WHERE compliant = true)"
            )
        if since_epoch is not None:
            clauses.append(f"{time_col} >= :since")
            params["since"] = since_epoch
        if user_ids:
            clauses.append("user_id = ANY(:uids)")
            params["uids"] = list(user_ids)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    specs = {
        "glucose_entries": ("user_id, recorded_at, value", "recorded_at"),
        "bolus_entries": (
            "user_id, recorded_at, carbohydrates, insulin, glucose, carbohydrate_ratio, insulin_type",
            "recorded_at",
        ),
        "sport_measurements": ("user_id, recorded_at, type, value", "recorded_at"),
        "sport_trainings": (
            "user_id, started_at, ended_at, type, distance",
            "started_at",
        ),
        "sport_workouts": ("user_id, started_at", "started_at"),
    }
    out: dict[str, pd.DataFrame] = {}
    for name, (cols, time_col) in specs.items():
        clause, params = where(time_col)
        sql = text(f"SELECT {cols} FROM {_q(pg.db_schema, name)}{clause}")
        out[name] = pd.read_sql(sql, engine, params=params)

    # user_settings has no recorded_at → fetch per user (for ISF/ICR therapy features)
    ucl = (
        " WHERE user_id IN (SELECT id FROM %s WHERE compliant = true)"
        % _q(pg.db_schema, "users")
        if pg.only_compliant
        else ""
    )
    out["user_settings"] = pd.read_sql(
        text(f"SELECT user_id, content FROM {_q(pg.db_schema, 'user_settings')}{ucl}"),
        engine,
    )
    return out


def _probe_recorded_at(engine, pg: PostgresConfig) -> pd.Series:
    from sqlalchemy import text

    sql = text(
        f"SELECT recorded_at FROM {_q(pg.db_schema, 'glucose_entries')} LIMIT 2000"
    )
    return pd.read_sql(sql, engine)["recorded_at"]


def load_raw(
    cfg: Config, since=None, user_ids: Optional[list[str]] = None
) -> pd.DataFrame:
    """Connect, fetch, and assemble the RAW frame (ready for ``align``)."""
    engine = connect(cfg)
    tables = fetch_tables(cfg, engine=engine, since=since, user_ids=user_ids)
    return assemble_raw(tables, cfg)


def inspect(cfg: Config, engine=None) -> dict:
    """Surface the unknowns the DDL hides: ts unit, glucose unit, type vocabularies.

    Run this first (``gf db-inspect``) to pin the mappings against real values.
    """
    from sqlalchemy import text

    pg = cfg.pg
    engine = engine or connect(cfg)
    report: dict = {}

    def frame(sql: str, **p) -> pd.DataFrame:
        return pd.read_sql(text(sql), engine, params=p)

    counts = {}
    for name in [
        "glucose_entries",
        "bolus_entries",
        "sport_measurements",
        "sport_trainings",
        "sport_workouts",
        "events",
        "users",
        "user_settings",
        "pumps",
        "sensors",
    ]:
        try:
            counts[name] = int(
                frame(f"SELECT count(*) c FROM {_q(pg.db_schema, name)}")["c"].iloc[0]
            )
        except Exception as exc:  # pragma: no cover - depends on live DB
            counts[name] = f"error: {exc}"
    report["row_counts"] = counts

    ra = _probe_recorded_at(engine, pg)
    unit = detect_ts_unit(ra)
    report["ts_unit_detected"] = unit
    rng = frame(
        f"SELECT min(recorded_at) lo, max(recorded_at) hi FROM {_q(pg.db_schema, 'glucose_entries')}"
    )
    report["glucose_time_range_utc"] = {
        "min": str(to_datetime_utc(rng["lo"], unit).iloc[0]),
        "max": str(to_datetime_utc(rng["hi"], unit).iloc[0]),
    }

    gv = frame(f"SELECT value FROM {_q(pg.db_schema, 'glucose_entries')} LIMIT 5000")[
        "value"
    ]
    report["glucose_unit_detected"] = detect_glucose_unit(gv)
    report["glucose_value_stats"] = {
        "median": round(float(pd.to_numeric(gv).median()), 2),
        "min": round(float(pd.to_numeric(gv).min()), 2),
        "max": round(float(pd.to_numeric(gv).max()), 2),
    }

    cad = frame(
        f"SELECT recorded_at FROM {_q(pg.db_schema, 'glucose_entries')} ORDER BY recorded_at LIMIT 5000"
    )["recorded_at"]
    diffs = (
        to_datetime_utc(cad, unit).sort_values().diff().dropna().dt.total_seconds()
        / 60.0
    )
    report["cgm_cadence_min_median"] = (
        round(float(diffs.median()), 2) if len(diffs) else None
    )

    for tbl, col in [("sport_measurements", "type"), ("events", "type")]:
        try:
            vc = frame(
                f"SELECT {col}, count(*) c FROM {_q(pg.db_schema, tbl)} GROUP BY {col} ORDER BY c DESC LIMIT 40"
            )
            report[f"{tbl}.{col}"] = dict(zip(vc[col].astype(str), vc["c"].astype(int)))
        except Exception as exc:  # pragma: no cover
            report[f"{tbl}.{col}"] = f"error: {exc}"

    for tbl, col in [("user_settings", "content"), ("events", "data")]:
        try:
            s = frame(f"SELECT {col} FROM {_q(pg.db_schema, tbl)} LIMIT 3")[col].astype(
                str
            )
            report[f"{tbl}.{col}_samples"] = [x[:300] for x in s.tolist()]
        except Exception as exc:  # pragma: no cover
            report[f"{tbl}.{col}_samples"] = f"error: {exc}"

    return report
