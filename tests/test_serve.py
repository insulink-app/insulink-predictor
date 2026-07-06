"""Serving-path checks for the per-user, DB-backed forecast endpoint.

No live database and no on-disk artifact: the DB layer is stubbed and a small but
real per-user curve model is built in-process, so these exercise the true feature
build + forecast contract end-to-end."""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import pytest

from insulink_predictor.data.align import align
from insulink_predictor.eval.harness import build_supervised
from insulink_predictor.models.events import build_curve_targets, train_curve_models
from serve import app as serve


def _ramp_grid(n: int = 40, start_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    """Aligned grid for one user: n buckets 5 min apart, glucose 100 -> ~160 mg/dL,
    with a carb+insulin event so the multi-channel (COB/IOB/therapy) path runs."""
    step_ms = 5 * 60 * 1000
    ts = pd.to_datetime([start_ms + i * step_ms for i in range(n)], unit="ms", utc=True)
    carbs = np.full(n, np.nan)
    insulin = np.full(n, np.nan)
    meal = np.zeros(n, dtype=bool)
    carbs[5], insulin[5], meal[5] = 40.0, 4.0, True  # one meal+bolus
    raw = pd.DataFrame(
        {
            "user_id": "u1",
            "ts_utc": ts,
            "ts_local": ts.tz_convert(serve._CFG.pg.local_tz).tz_localize(None),
            "glucose_mgdl": [100.0 + 1.5 * i for i in range(n)],
            "meal_flag": meal,
            "carbs_g": carbs,
            "insulin_u": insulin,
            "steps": 0,
            "activity_flag": False,
            "hr": np.nan,
            "weather_temp": np.nan,
            "daily_steps": np.nan,
            "daily_distance": np.nan,
            "isf": 35.0,
            "icr": 15.0,
        }
    )
    return align(raw, serve._CFG)


@pytest.fixture(scope="module")
def tiny_model() -> dict:
    """A real (small) per-user curve model built from the ramp grid — no artifact."""
    grid = _ramp_grid(60)
    sup, feature_cols = build_supervised(serve._CFG, grid)
    max_step = max(serve._CFG.horizons_steps)
    sup = build_curve_targets(sup, max_step)
    models = train_curve_models(
        sup, feature_cols, max_step, serve._CFG.features.predict_delta
    )
    return {
        "curve_models": models,
        "feature_cols": feature_cols,
        "grid_minutes": serve._CFG.grid_minutes,
        "max_step": max_step,
        "predict_delta": serve._CFG.features.predict_delta,
    }


def test_ramp_grid_is_regular():
    grid = _ramp_grid(24)
    deltas = grid["ts_utc"].diff().dropna().dt.total_seconds().unique()
    assert deltas.tolist() == [300.0]  # exactly 5-min spacing, no gaps
    assert not grid["sensor_gap"].any()


def test_user_grid_folds_in_fresh_readings(monkeypatch):
    """Readings the async DB hasn't ingested still set the forecast origin."""
    monkeypatch.setattr(serve, "_engine", lambda: None)
    monkeypatch.setattr(serve, "fetch_tables", lambda *a, **k: {})  # DB has nothing
    step_ms = 5 * 60 * 1000
    start_ms = 1_700_000_100_000  # on a 5-min boundary (align rounds to nearest)
    readings = [
        serve.Reading(ts=start_ms + i * step_ms, mgdl=100.0 + i) for i in range(30)
    ]
    grid = serve._user_grid("u1", readings)
    assert grid is not None
    last_ms = int(grid["ts_utc"].iloc[-1].timestamp() * 1000)
    assert last_ms == readings[-1].ts  # newest reading is the grid tip


def test_user_grid_readings_win_over_overlapping_db_bucket(monkeypatch):
    """A reading and a DB row in the same 5-min bucket must not be averaged/doubled."""
    monkeypatch.setattr(serve, "_engine", lambda: None)
    step_ms = 5 * 60 * 1000
    start = 1_700_000_100_000  # on a 5-min boundary
    db_ts = [start + i * step_ms for i in range(5)]  # buckets 0..4, all value 100
    ge = pd.DataFrame(
        {"user_id": ["u1"] * 5, "recorded_at": db_ts, "value": [100.0] * 5}
    )
    monkeypatch.setattr(serve, "fetch_tables", lambda *a, **k: {"glucose_entries": ge})

    readings = [
        serve.Reading(ts=start + 4 * step_ms, mgdl=150.0),  # overlaps DB bucket 4
        serve.Reading(ts=start + 5 * step_ms, mgdl=160.0),  # brand-new bucket 5
    ]
    grid = serve._user_grid("u1", readings)

    def _val(bucket_i):
        ts = pd.to_datetime(start + bucket_i * step_ms, unit="ms", utc=True)
        return grid.loc[grid["ts_utc"] == ts, "glucose_mgdl"].iloc[0]

    assert len(grid) == 6  # buckets 0..5, no duplicate row for the overlap
    assert _val(4) == 150.0  # reading wins over the DB's 100 (not mean 125)
    assert _val(5) == 160.0  # fresh reading extends the tip


@pytest.mark.parametrize("horizon", [30, 60])
def test_forecast_shape_and_range(monkeypatch, tiny_model, horizon):
    monkeypatch.setattr(serve, "_user_grid", lambda uid, readings: _ramp_grid(40))
    curve, generated_at = serve.forecast("u1", horizon, model=tiny_model)
    assert len(curve) == horizon // tiny_model["grid_minutes"]
    assert all(np.isfinite(curve))
    assert all(serve._GLUCOSE_LO <= v <= serve._GLUCOSE_HI for v in curve)
    assert generated_at > 0  # origin = latest bucket's epoch ms


def test_predict_endpoint_contract(monkeypatch, tiny_model):
    monkeypatch.setattr(serve, "_model_for", lambda uid: tiny_model)
    monkeypatch.setattr(serve, "_user_grid", lambda uid, readings: _ramp_grid(40))
    resp = serve.predict(serve.PredictRequest(user_id="u1", horizon_min=30))
    assert resp["horizon_min"] == 30
    assert [pt["offset_min"] for pt in resp["curve"]] == [5, 10, 15, 20, 25, 30]
    assert all(
        serve._GLUCOSE_LO <= pt["mgdl"] <= serve._GLUCOSE_HI for pt in resp["curve"]
    )


def test_predict_404_when_user_has_no_data(monkeypatch, tiny_model):
    from fastapi import HTTPException

    monkeypatch.setattr(serve, "_model_for", lambda uid: tiny_model)
    monkeypatch.setattr(serve, "_user_grid", lambda uid, readings: None)
    with pytest.raises(HTTPException) as exc:
        serve.predict(serve.PredictRequest(user_id="u1", horizon_min=30))
    assert exc.value.status_code == 404


def test_predict_404_when_user_has_no_model(monkeypatch):
    """No global fallback: a user without a trained model gets a 404, not a forecast."""
    from fastapi import HTTPException

    monkeypatch.setattr(serve, "_model_for", lambda uid: None)
    with pytest.raises(HTTPException) as exc:
        serve.predict(serve.PredictRequest(user_id="ghost", horizon_min=30))
    assert exc.value.status_code == 404


def test_model_for_loads_per_user_and_hot_reloads(monkeypatch, tmp_path):
    """Absent file -> None (no fallback); present file loads and reloads on change."""
    monkeypatch.setattr(serve.training, "MODELS_DIR", tmp_path)
    serve._user_model_cache.clear()

    assert serve._model_for("nobody") is None  # no model -> None, no fallback

    path = tmp_path / "u1.joblib"
    joblib.dump({"tag": "v1"}, path)
    assert serve._model_for("u1")["tag"] == "v1"

    m1 = path.stat().st_mtime
    joblib.dump({"tag": "v2"}, path)  # background job rebuilt it
    os.utime(path, (m1 + 10, m1 + 10))  # deterministically newer mtime
    assert serve._model_for("u1")["tag"] == "v2"  # hot-reloaded, not stale-cached


def test_has_any_model(monkeypatch, tmp_path):
    monkeypatch.setattr(serve.training, "MODELS_DIR", tmp_path)
    assert serve._has_any_model() is False
    joblib.dump({"tag": "v1"}, tmp_path / "u1.joblib")
    assert serve._has_any_model() is True


def test_lifespan_bootstraps_training_when_empty(monkeypatch, tmp_path):
    """On startup with an empty MODELS_DIR, an initial retrain is kicked off once."""
    import asyncio

    monkeypatch.setattr(serve.training, "MODELS_DIR", tmp_path)  # empty
    monkeypatch.setattr(serve, "_SCHEDULER_ENABLED", True)

    async def _idle_loop():  # don't run the real (sleeping) daily loop
        while True:
            await asyncio.sleep(3600)

    monkeypatch.setattr(serve, "_daily_retrain_loop", _idle_loop)

    calls = []
    monkeypatch.setattr(
        serve,
        "_run_retrain",
        lambda user_id=None: (
            calls.append(user_id) or {"trained": 0, "skipped": 0, "errors": 0}
        ),
    )

    async def run():
        async with serve.lifespan(serve.app):
            for _ in range(100):  # let the background bootstrap thread finish
                if calls:
                    break
                await asyncio.sleep(0.02)

    asyncio.run(run())
    assert calls == [None]  # one full (all-users) retrain triggered


def test_lifespan_skips_bootstrap_when_models_exist(monkeypatch, tmp_path):
    import asyncio

    joblib.dump({"tag": "v1"}, tmp_path / "u1.joblib")  # a model already exists
    monkeypatch.setattr(serve.training, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(serve, "_SCHEDULER_ENABLED", True)

    async def _idle_loop():
        while True:
            await asyncio.sleep(3600)

    monkeypatch.setattr(serve, "_daily_retrain_loop", _idle_loop)
    calls = []
    monkeypatch.setattr(
        serve, "_run_retrain", lambda user_id=None: calls.append(user_id) or {}
    )

    async def run():
        async with serve.lifespan(serve.app):
            await asyncio.sleep(0.1)

    asyncio.run(run())
    assert calls == []  # models present -> no bootstrap training
