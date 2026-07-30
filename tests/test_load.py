"""Real-loader transformation logic (no DB): unit/timestamp handling + assembly.

The SQLAlchemy/psycopg I/O is exercised against a live DB by `gf db-inspect` /
`gf load`; here we test the pure mapping that turns fetched tables into the RAW
contract and prove it flows through align() + the schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.load import (
    RAW_COLUMNS,
    assemble_raw,
    detect_glucose_unit,
    detect_ts_unit,
    to_datetime_utc,
    to_mgdl,
)

_MS = 1000
_MIN_MS = 60 * _MS


def _epoch_ms(iso: str) -> int:
    return int(pd.Timestamp(iso).timestamp() * 1000)


def test_detect_ts_unit_ms_vs_s():
    assert detect_ts_unit(pd.Series([_epoch_ms("2024-06-01T00:00:00Z")])) == "ms"
    assert detect_ts_unit(pd.Series([1_717_200_000])) == "s"


def test_to_datetime_utc_ms():
    base = _epoch_ms("2024-06-01T08:00:00Z")
    out = to_datetime_utc(pd.Series([base]), "ms")
    assert str(out.iloc[0]) == "2024-06-01 08:00:00+00:00"


def test_glucose_unit_detection_and_conversion():
    assert detect_glucose_unit(pd.Series([80, 120, 160, 200.0])) == "mg/dL"
    assert detect_glucose_unit(pd.Series([4.5, 6.7, 8.9])) == "mmol/L"
    # mmol -> mg/dL (x18)
    conv = to_mgdl(pd.Series([5.0]), "mmol/L")
    assert abs(conv.iloc[0] - 90.09) < 0.1
    # mg/dL passes through unchanged
    assert to_mgdl(pd.Series([120.0]), "mg/dL").iloc[0] == 120.0


def _sample_tables(n_glucose=180):
    """One user: 15h of 5-min CGM, two boluses, HR samples, one training window."""
    uid = "11111111-1111-1111-1111-111111111111"
    base = _epoch_ms("2024-06-01T06:00:00Z")
    g = pd.DataFrame(
        {
            "user_id": uid,
            "recorded_at": [base + i * 5 * _MIN_MS for i in range(n_glucose)],
            "value": 120 + 30 * np.sin(np.linspace(0, 6, n_glucose)),  # mg/dL
        }
    )
    bolus = pd.DataFrame(
        {
            "user_id": uid,
            "recorded_at": [base + 60 * _MIN_MS, base + 300 * _MIN_MS],
            "carbohydrates": [45.0, 60.0],
            "insulin": [4.5, 6.0],
            "glucose": [140.0, 110.0],
            "carbohydrate_ratio": [10.0, 10.0],
            "insulin_type": ["rapid", "rapid"],
        }
    )
    # Intraday HR + steps every 15 min (recognised as a stream), a daily WEIGHT
    # aggregate (excluded by the cadence gate), and an unknown type (dropped).
    hr_rows = [
        {
            "user_id": uid,
            "recorded_at": base + i * 15 * _MIN_MS,
            "type": "heart_rate",
            "value": 70.0 + i,
        }
        for i in range(12)
    ]
    step_rows = [
        {
            "user_id": uid,
            "recorded_at": base + i * 15 * _MIN_MS,
            "type": "STEPS",
            "value": 100.0 + i,
        }
        for i in range(12)
    ]
    weight_rows = [
        {
            "user_id": uid,
            "recorded_at": base + d * 24 * 60 * _MIN_MS,
            "type": "WEIGHT",
            "value": 73.0,
        }
        for d in range(3)
    ]
    unknown = [
        {"user_id": uid, "recorded_at": base, "type": "unknown_metric", "value": 1.0}
    ]
    sm = pd.DataFrame(hr_rows + step_rows + weight_rows + unknown)
    st = pd.DataFrame(
        {
            "user_id": uid,
            "started_at": [base + 120 * _MIN_MS],
            "ended_at": [base + 160 * _MIN_MS],
            "type": ["run"],
            "distance": [5000.0],
        }
    )
    settings = pd.DataFrame(
        [
            {
                "user_id": uid,
                "content": '{"bolus_correction_factor":"35","bolus_carb_factor":"15"}',
            }
        ]
    )
    return {
        "glucose_entries": g,
        "bolus_entries": bolus,
        "sport_measurements": sm,
        "sport_trainings": st,
        "user_settings": settings,
    }


def test_assemble_raw_produces_contract_columns():
    cfg = Config()
    raw = assemble_raw(_sample_tables(), cfg)
    required = {
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
    }
    assert required.issubset(raw.columns)
    assert raw["ts_utc"].dt.tz is not None  # tz-aware UTC
    assert raw["ts_local"].dt.tz is None  # naive local wall-clock
    # channels landed: 180 CGM readings + 2 bolus SMBG folded in
    assert raw["glucose_mgdl"].notna().sum() == 182
    assert (raw["carbs_g"] > 0).sum() == 2  # two meals
    assert raw["insulin_u"].notna().sum() == 2
    assert raw["hr"].notna().sum() == 12  # intraday HR stream mapped
    assert raw["steps"].notna().sum() == 12  # intraday STEPS mapped
    assert raw["activity_flag"].sum() >= 1  # training expanded to buckets
    # ISF/ICR parsed from user_settings and attached per user
    assert raw["isf"].iloc[0] == 35.0 and raw["icr"].iloc[0] == 15.0


def test_meals_carry_the_doses_when_bolus_entries_is_empty():
    """The production shape: the app logs every dose as a meal row.

    `bolus_entries` has an entity but no controller writing it, so COB/IOB have to
    come from `nutrition_meals`. A meal with carbs raises meal_flag; a pure
    correction dose (carbs 0) still contributes insulin_u but must not read as one.
    """
    cfg = Config()
    tables = _sample_tables()
    uid = tables["glucose_entries"]["user_id"].iloc[0]
    base = int(tables["glucose_entries"]["recorded_at"].iloc[0])
    tables["bolus_entries"] = pd.DataFrame()  # as in prod: nothing writes it
    tables["nutrition_meals"] = pd.DataFrame(
        {
            "user_id": uid,
            "time": [base + 60 * _MIN_MS, base + 300 * _MIN_MS],
            "carbs": [45.0, 0.0],
            "glucose": [140, 0],  # 0 = no reading available when the dose was logged
            "bolus": [4.5, 1.5],
        }
    )
    raw = assemble_raw(tables, cfg)
    assert (raw["carbs_g"] > 0).sum() == 1
    assert raw["meal_flag"].sum() == 1  # the correction dose is not a meal
    assert raw["insulin_u"].notna().sum() == 2  # but it does carry insulin
    assert raw["glucose_mgdl"].notna().sum() == 181  # 180 CGM + the one real reading
    grid = align(raw, cfg)
    assert grid["insulin_u"].notna().sum() >= 2


def test_a_missing_table_is_skipped_not_fatal(monkeypatch):
    """One absent table must not abort the fetch — and kill the nightly retrain.

    Deployments run different schema versions; `assemble_raw` already degrades on
    a missing source, so the fetch has to hand it back missing rather than raise.
    """
    from sqlalchemy.exc import ProgrammingError

    from insulink_predictor.data import load as load_module

    def fake_read_sql(sql, engine, params=None):
        if "nutrition_meals" in str(sql):
            raise ProgrammingError("SELECT ...", {}, Exception("relation missing"))
        return pd.DataFrame({"recorded_at": [_epoch_ms("2024-06-01T06:00:00Z")]})

    monkeypatch.setattr(load_module.pd, "read_sql", fake_read_sql)
    out = load_module.fetch_tables(Config(), engine=object())
    assert "nutrition_meals" not in out  # skipped, no exception
    assert "glucose_entries" in out  # the other tables still came back


def test_daily_aggregate_steps_are_excluded():
    """A daily STEPS total must NOT be mapped into the intraday steps channel."""
    cfg = Config()
    tables = _sample_tables()
    uid = tables["glucose_entries"]["user_id"].iloc[0]
    base = int(tables["glucose_entries"]["recorded_at"].iloc[0])
    daily = pd.DataFrame(
        [
            {
                "user_id": uid,
                "recorded_at": base + d * 24 * 60 * _MIN_MS,
                "type": "STEPS",
                "value": 12000.0,
            }
            for d in range(5)
        ]
    )
    tables["sport_measurements"] = daily  # only daily-cadence STEPS
    raw = assemble_raw(tables, cfg)
    assert (
        raw["steps"].notna().sum() == 0
    )  # daily totals excluded, not dumped into a bucket


def test_pulse_samples_feed_the_hr_channel():
    """HR lives in health_pulse_samples, not sport_measurements (the real schema).

    Multiple samples per 5-min bucket must average, and implausible bpm (a 0-bpm
    dropout) must not drag that average down.
    """
    cfg = Config()
    tables = _sample_tables()
    uid = tables["glucose_entries"]["user_id"].iloc[0]
    base = int(tables["glucose_entries"]["recorded_at"].iloc[0])
    tables["sport_measurements"] = tables["sport_measurements"][
        tables["sport_measurements"]["type"] != "heart_rate"
    ]  # as in prod: no HR type here at all
    tables["health_pulse_samples"] = pd.DataFrame(
        [
            {"user_id": uid, "recorded_at": base + 60_000, "bpm": 60},
            {"user_id": uid, "recorded_at": base + 120_000, "bpm": 80},  # same bucket
            {"user_id": uid, "recorded_at": base + 180_000, "bpm": 0},  # dropout
        ]
    )
    raw = assemble_raw(tables, cfg)
    assert raw["hr"].notna().sum() == 2  # the 0-bpm sample is dropped, not kept
    grid = align(raw, cfg)
    bucket = pd.to_datetime(base, unit="ms", utc=True).round(cfg.grid_freq)
    hr = grid.loc[grid["ts_utc"] == bucket, "hr"].iloc[0]
    assert hr == 70.0  # mean(60, 80) — not 46.7 (would include the 0)


def test_settings_parse():
    from insulink_predictor.data.load import parse_settings

    p = parse_settings('{"bolus_correction_factor":"35","bolus_carb_factor":"15"}')
    assert p == {"isf": 35.0, "icr": 15.0}
    assert parse_settings("not json") == {"isf": None, "icr": None}


def test_assembled_raw_flows_through_align_and_schema():
    cfg = Config()
    raw = assemble_raw(_sample_tables(), cfg)
    grid = align(raw, cfg)  # validates against the pandera contract internally
    # regular 5-min grid, carbs/insulin survived the alignment; ISF/ICR carried
    diffs = grid["ts_utc"].diff().dropna().dt.total_seconds() / 60
    assert (diffs == cfg.grid_minutes).all()
    assert (grid["carbs_g"] > 0).sum() >= 2
    assert grid["insulin_u"].notna().sum() >= 2
    assert grid["activity_flag"].any()
    assert (grid["isf"] == 35.0).all() and (grid["icr"] == 15.0).all()


def test_empty_tables_yield_empty_raw():
    cfg = Config()
    raw = assemble_raw({}, cfg)
    assert raw.empty
    assert list(raw.columns) == RAW_COLUMNS
