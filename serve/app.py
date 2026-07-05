"""FastAPI inference service for the glucose curve forecast.

Loads the model serialized by ``scripts/train_and_save.py`` once at startup and
serves a single endpoint. It is **internal** — the Spring backend
(``insulink-api``) calls it and supplies the readings from its own database; this
service is stateless and never talks to a DB.

Run:  ``uv run uvicorn serve.app:app --host 0.0.0.0 --port 8000``
(``MODEL_PATH`` env overrides the artifact location.)

Contract:
    POST /predict
      {"readings": [{"ts": <epoch_ms>, "mgdl": <float>}, ...], "horizon_min": 30|60}
    -> {"generated_at": <epoch_ms>, "horizon_min": 30,
        "curve": [{"offset_min": 5, "mgdl": 142.0}, ...]}
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from insulink_predictor.config import load_config
from insulink_predictor.data.align import align
from insulink_predictor.features.build import build_features
from insulink_predictor.models.events import forecast_curve

_GLUCOSE_LO, _GLUCOSE_HI = 10.0, 700.0
_MODEL_PATH = Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))

app = FastAPI(title="insulink-predictor")


def _config():
    """Glucose-only config — mirrors ``scripts/train_and_save.py`` (no skew)."""
    cfg = load_config()
    cfg.features.use_carbs = False
    cfg.features.use_insulin = False
    cfg.features.use_hr = False
    cfg.features.use_weather = False
    cfg.features.use_therapy = False
    return cfg


_CFG = _config()
_MODEL = joblib.load(_MODEL_PATH) if _MODEL_PATH.exists() else None


class Reading(BaseModel):
    ts: int  # epoch milliseconds
    mgdl: float


class PredictRequest(BaseModel):
    readings: list[Reading]
    horizon_min: int = 30


def _grid_from_readings(readings: list[Reading]) -> pd.DataFrame:
    """Turn raw readings into the aligned 5-min grid the feature builder expects."""
    ts = pd.to_datetime([r.ts for r in readings], unit="ms", utc=True)
    raw = pd.DataFrame(
        {
            "user_id": "app",
            "ts_utc": ts,
            "ts_local": ts.tz_convert(_CFG.pg.local_tz).tz_localize(None),
            "glucose_mgdl": [float(r.mgdl) for r in readings],
            "steps": 0,
            "meal_flag": False,
            "carbs_g": np.nan,
            "insulin_u": np.nan,
            "activity_flag": False,
            "hr": np.nan,
            "weather_temp": np.nan,
        }
    )
    return align(raw, _CFG)


def forecast(readings: list[Reading], horizon_min: int) -> list[float]:
    """Absolute mg/dL trajectory ``t+5 … t+horizon``, clipped to plausible range."""
    feat, _ = build_features(_grid_from_readings(readings), _CFG)
    last = feat.iloc[[-1]].copy()
    for col in _MODEL["feature_cols"]:  # reindex safety net: absent -> NaN (LGBM ok)
        if col not in last.columns:
            last[col] = np.nan
    curve = forecast_curve(
        last, _MODEL["curve_models"], _MODEL["feature_cols"], _MODEL["predict_delta"]
    )[0]
    steps = max(1, horizon_min // _MODEL["grid_minutes"])
    return [float(np.clip(v, _GLUCOSE_LO, _GLUCOSE_HI)) for v in curve[:steps]]


@app.get("/")
def health() -> dict:
    return {"status": "ok", "model_loaded": _MODEL is not None}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if _MODEL is None:
        raise HTTPException(503, "model not loaded")
    if not req.readings:
        raise HTTPException(400, "no readings")
    grid = _MODEL["grid_minutes"]
    curve = forecast(req.readings, req.horizon_min)
    return {
        "generated_at": req.readings[-1].ts,
        "horizon_min": req.horizon_min,
        "curve": [
            {"offset_min": (i + 1) * grid, "mgdl": round(v, 1)}
            for i, v in enumerate(curve)
        ],
    }
