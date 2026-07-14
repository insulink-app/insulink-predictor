"""FastAPI inference service for the glucose curve forecast.

Serving is **DB-backed and per-user**. A request carries a ``user_id`` (plus the
horizon); the service pulls that user's full history — glucose, carbs/insulin
(COB/IOB), sport and per-user therapy (ISF/ICR) — straight from PostgreSQL and
forecasts from the latest bucket, so the model sees every available channel.

Models are per user. A background scheduler (see ``serve/training.py``) tunes and
rebuilds each user's model **daily** and writes it to ``MODELS_DIR`` (a mounted
Docker volume). On startup, if ``MODELS_DIR`` holds no models yet, an initial
training run is kicked off in the background (the server still comes up at once).
Serving loads ``MODELS_DIR/<user_id>.joblib`` — hot-reloaded on change. There is
**no global fallback**: a user without a model yet (never trained, or too little
data to tune) gets a 404.

Because the DB is written asynchronously, the freshest CGM value may not have
landed yet. The caller may pass ``readings`` — the latest glucose point(s) —
folded into the glucose channel before alignment so the forecast is anchored to
the true tip rather than a stale DB snapshot.

Run:  ``uv run uvicorn serve.app:app --host 0.0.0.0 --port 8000``
Env: ``DATABASE_URL``/``GF_PG__*`` (DB), ``MODELS_DIR`` (per-user model dir),
``ENABLE_SCHEDULER`` (default on), ``RETRAIN_HOUR`` (local hour, default 3).
Run only ONE instance/worker with the scheduler enabled.

Each curve point carries an uncertainty band (``lo_mgdl``/``hi_mgdl``, the
conformally-calibrated q10/q90). The point forecast is the conditional *mean*, so
it is correctly shrunk toward the middle and on its own almost never calls a low
or a high; the band is what flags them (see ``training.BAND_QUANTILES``). ``risk``
summarises the band against the 70–180 target range.

Contract:
    POST /predict
      {"user_id": "<uuid>", "horizon_min": 30|60,
       "readings": [{"ts": <epoch_ms>, "mgdl": <float>}, ...]}   # readings optional
    -> {"generated_at": <epoch_ms>, "horizon_min": 30,
        "curve": [{"offset_min": 5, "mgdl": 142.0,
                   "lo_mgdl": 121.4, "hi_mgdl": 168.9}, ...],
        "band_quantiles": [0.1, 0.9],
        "risk": {"low": false, "high": true}}
    Models trained before the band existed omit lo/hi, band_quantiles and risk.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
from contextlib import asynccontextmanager, suppress

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from insulink_predictor.config import load_config
from insulink_predictor.data.align import align
from insulink_predictor.data.load import (
    assemble_raw,
    connect,
    fetch_tables,
    to_datetime_utc,
)
from insulink_predictor.features.build import build_features
from insulink_predictor.models.events import forecast_curve, forecast_curve_quantiles
from serve import training

# Surface our INFO logs (scheduler / bootstrap / per-user training) — uvicorn only
# wires handlers for its own loggers, so without this the tune activity is silent.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("insulink.serve")

_GLUCOSE_LO, _GLUCOSE_HI = 10.0, 700.0

# One config for the tune job and serving (unmodified ``load_config()``) => every
# per-user model sees exactly the multi-channel feature set it was trained on.
_CFG = load_config()
_ENGINE = None  # pooled SQLAlchemy engine, created on first request (see _engine)
_user_model_cache: dict[str, tuple[float, dict]] = {}  # user_id -> (mtime, model)

# --- scheduler config -------------------------------------------------------- #
_SCHEDULER_ENABLED = os.environ.get("ENABLE_SCHEDULER", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_RETRAIN_HOUR = int(os.environ.get("RETRAIN_HOUR", "3"))  # local hour of day, 0-23


class Reading(BaseModel):
    ts: int  # epoch milliseconds
    mgdl: float


class PredictRequest(BaseModel):
    user_id: str
    horizon_min: int = 30
    readings: list[Reading] = []  # freshest glucose not yet ingested by the DB


def _engine():
    """Return a pooled DB engine, created lazily so import/startup needs no live DB."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = connect(_CFG)
    return _ENGINE


def _model_for(user_id: str) -> dict | None:
    """The user's own model if present (hot-reloaded on file change), else ``None``.

    A background job writes ``MODELS_DIR/<user_id>.joblib`` atomically; we key the
    cache on mtime so a freshly rebuilt model is picked up without a restart. There
    is no global fallback — an absent file means the user has no model.
    """
    path = training.MODELS_DIR / f"{user_id}.joblib"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None  # no model for this user (not trained / insufficient data)
    cached = _user_model_cache.get(user_id)
    if cached is None or cached[0] != mtime:
        _user_model_cache[user_id] = (mtime, joblib.load(path))
    return _user_model_cache[user_id][1]


def _user_grid(user_id: str, readings: list[Reading]) -> pd.DataFrame | None:
    """Aligned 5-min grid for one user, pulled from the DB (all channels).

    ``readings`` (the caller's freshest glucose) are injected as extra
    ``glucose_entries`` rows so the whole pipeline — unit handling, ISF/ICR merge,
    grid anchoring — treats them exactly like DB glucose. The grid is anchored to
    the glucose span, so a reading newer than the DB tip extends the origin forward.

    The async DB write may already hold some of these readings. To avoid a bucket
    being fed from both sources (``align`` would otherwise average them), any DB
    glucose row whose 5-min bucket a reading also covers is dropped first — the
    fresher request value wins. ``None`` when there is no forecastable data (no CGM).
    """
    tables = fetch_tables(_CFG, engine=_engine(), user_ids=[user_id])
    if readings:
        tip = pd.DataFrame(
            {
                "user_id": user_id,
                "recorded_at": [r.ts for r in readings],  # epoch ms, like the DB
                "value": [float(r.mgdl) for r in readings],
            }
        )
        ge = tables.get("glucose_entries")
        if ge is not None and len(ge):
            freq = _CFG.grid_freq
            unit = _CFG.pg.ts_unit
            reading_buckets = set(
                to_datetime_utc(tip["recorded_at"], unit).dt.round(freq)
            )
            db_buckets = to_datetime_utc(ge["recorded_at"], unit).dt.round(freq)
            ge = ge[~db_buckets.isin(reading_buckets).to_numpy()]  # readings win
            tables["glucose_entries"] = pd.concat([ge, tip], ignore_index=True)
        else:
            tables["glucose_entries"] = tip
    raw = assemble_raw(tables, _CFG)
    if raw.empty:
        return None
    grid = align(raw, _CFG)
    return grid if not grid.empty else None


def forecast(
    user_id: str,
    horizon_min: int,
    readings: list[Reading] | None = None,
    model: dict | None = None,
) -> tuple[list[float], dict[float, list[float]], int]:
    """Forecast a user's mg/dL trajectory ``t+5 … t+horizon`` from their DB history.

    Uses the user's own model (``model`` arg, else resolved via ``_model_for``).
    ``readings`` (optional) are the freshest glucose the async DB write may not hold
    yet; they set the forecast origin. Returns ``(trajectory, band, generated_at_ms)``
    — origin = the latest bucket, ``band`` mapping each quantile to its trajectory
    (empty for a model trained before the band existed). ``([], {}, 0)`` when there
    is no model or no data.
    """
    model = model or _model_for(user_id)
    if model is None:
        return [], {}, 0
    grid = _user_grid(user_id, readings or [])
    if grid is None:
        return [], {}, 0
    feat, _ = build_features(grid, _CFG)
    last = feat.iloc[[-1]].copy()
    for col in model["feature_cols"]:  # reindex safety net: absent -> NaN (LGBM ok)
        if col not in last.columns:
            last[col] = np.nan
    curve = forecast_curve(
        last, model["curve_models"], model["feature_cols"], model["predict_delta"]
    )[0]
    steps = max(1, horizon_min // model["grid_minutes"])

    def clip(c) -> list[float]:
        return [float(np.clip(v, _GLUCOSE_LO, _GLUCOSE_HI)) for v in c[:steps]]

    trajectory = clip(curve)

    band: dict[float, list[float]] = {}
    if model.get("quantile_models"):
        qcurves = forecast_curve_quantiles(
            last,
            model["quantile_models"],
            model["feature_cols"],
            model["predict_delta"],
        )
        offsets = model.get("q_offsets") or {}
        for q, c in qcurves.items():
            band[q] = clip(c[0] + np.asarray(offsets.get(q, 0.0)))
        # Quantile models are fit independently, so they can cross; sort per step
        # to keep lo <= hi.
        qs = sorted(band)
        stacked = np.sort(np.array([band[q] for q in qs]), axis=0)
        band = {q: list(map(float, stacked[i])) for i, q in enumerate(qs)}

    generated_at = int(last["ts_utc"].iloc[0].timestamp() * 1000)
    return trajectory, band, generated_at


# --------------------------------------------------------------------------- #
# Daily retrain scheduler (dependency-free asyncio loop in the app lifespan)   #
# --------------------------------------------------------------------------- #
def _seconds_until_next(hour: int, now: datetime.datetime) -> float:
    """Seconds from ``now`` to the next occurrence of ``hour:00`` local time."""
    nxt = now.replace(hour=hour % 24, minute=0, second=0, microsecond=0)
    if nxt <= now:
        nxt += datetime.timedelta(days=1)
    return (nxt - now).total_seconds()


def _run_retrain(user_id: str | None = None) -> dict:
    """Blocking tune+train (run in a worker thread). One user, or every user."""
    if user_id:
        return {
            "n_users": 1,
            "results": [training.train_user_model(_CFG, user_id, _engine())],
        }
    return training.retrain_all(_CFG, engine=_engine())


def _has_any_model() -> bool:
    """True if at least one per-user model exists on disk."""
    md = training.MODELS_DIR
    return md.exists() and any(md.glob("*.joblib"))


async def _retrain_in_thread(what: str) -> None:
    """Run a full retrain off the event loop and log the summary. Never raises."""
    log.info("starting %s per-user retrain", what)
    try:
        summary = await asyncio.to_thread(_run_retrain)
        log.info(
            "%s retrain done: trained=%s rejected=%s skipped=%s errors=%s",
            what,
            summary.get("trained"),
            summary.get("rejected"),
            summary.get("skipped"),
            summary.get("errors"),
        )
    except Exception:  # a failed run must not kill the caller
        log.exception("%s retrain crashed", what)


async def _daily_retrain_loop() -> None:
    while True:
        delay = _seconds_until_next(_RETRAIN_HOUR, datetime.datetime.now())
        log.info("next per-user retrain in %.0f min", delay / 60)
        await asyncio.sleep(delay)
        await _retrain_in_thread("scheduled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks: list[asyncio.Task] = []
    if _SCHEDULER_ENABLED:
        tasks.append(asyncio.create_task(_daily_retrain_loop()))
        log.info(
            "daily retrain scheduler enabled (RETRAIN_HOUR=%d local)", _RETRAIN_HOUR
        )
        # Cold start: with no models on disk, train them now (in the background, so
        # the server still comes up immediately and serves 404s until they land).
        if not _has_any_model():
            log.info("no per-user models found -> starting initial training")
            tasks.append(asyncio.create_task(_retrain_in_thread("initial")))
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            with suppress(asyncio.CancelledError):
                await t


app = FastAPI(title="insulink-predictor", lifespan=lifespan)


@app.get("/")
def health() -> dict:
    md = training.MODELS_DIR
    n_user_models = len(list(md.glob("*.joblib"))) if md.exists() else 0
    return {
        "status": "ok",
        "user_models": n_user_models,
        "scheduler": _SCHEDULER_ENABLED,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    model = _model_for(req.user_id)
    if model is None:
        raise HTTPException(
            404,
            f"no model for user {req.user_id} (not trained yet / insufficient data)",
        )
    curve, band, generated_at = forecast(
        req.user_id, req.horizon_min, req.readings, model
    )
    if not curve:
        raise HTTPException(404, f"no forecastable data for user {req.user_id}")
    grid = model["grid_minutes"]
    qs = sorted(band)
    lo, hi = (band[qs[0]], band[qs[-1]]) if qs else (None, None)
    point = []
    for i, v in enumerate(curve):
        p = {"offset_min": (i + 1) * grid, "mgdl": round(v, 1)}
        if lo is not None:
            p["lo_mgdl"] = round(lo[i], 1)
            p["hi_mgdl"] = round(hi[i], 1)
        point.append(p)
    return {
        "generated_at": generated_at,
        "horizon_min": req.horizon_min,
        "curve": point,
        "band_quantiles": qs or None,
        # The point curve is the conditional mean and rarely leaves 70-180; the
        # band is what actually flags an approaching low/high.
        "risk": {
            "low": bool(lo is not None and min(lo) < 70),
            "high": bool(hi is not None and max(hi) > 180),
        }
        if qs
        else None,
    }


@app.post("/admin/retrain")
async def admin_retrain(user_id: str | None = None) -> dict:
    """Trigger a tune+train now — one ``user_id`` (fast) or all users (slow)."""
    return await asyncio.to_thread(_run_retrain, user_id)
