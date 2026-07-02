"""Phase 2 DoD — LightGBM must beat persistence (skill > 0) at every horizon."""

from __future__ import annotations

import pytest

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.synth import generate
from insulink_predictor.eval.harness import run_lgbm_eval


@pytest.fixture(scope="module")
def lgbm_result():
    cfg = Config(synth={"n_users": 4, "days": 12, "seed": 3}, mlflow={"enabled": False})
    grid = align(generate(cfg), cfg)
    return run_lgbm_eval(cfg, df=grid, write=False)


def test_lgbm_beats_persistence_every_horizon(lgbm_result):
    for _, r in lgbm_result["lgbm_metrics"].iterrows():
        assert r["skill"] > 0, (
            f"skill@{r['horizon_min']}min = {r['skill']} (must be > 0)"
        )


def test_lgbm_rmse_below_persistence(lgbm_result):
    lg = lgbm_result["lgbm_metrics"].set_index("horizon_min")["rmse"]
    pe = lgbm_result["persistence_metrics"].set_index("horizon_min")["rmse"]
    assert (lg < pe).all()


def test_context_features_drive_delta_and_weather_is_not_top(lgbm_result):
    # The model predicts the CHANGE over persistence, so context/dynamics features
    # (time-since-meal, circadian, rate) drive it — not the absolute level.
    fi = lgbm_result["feature_importance"].set_index("feature")["gain_pct"]
    drivers = {
        "time_since_meal",
        "time_since_activity",
        "rate_short",
        "rate_long",
        "hour_sin",
        "hour_cos",
        "roll30_max",
        "roll60_max",
        "roll120_max",
        "cob_glucose",
        "carb_activity",
        "ins_activity",
        "time_since_bolus",
        "accel",
        # horizon-specific physiological forecast: the expected mg/dL change over
        # the next h min from carbs/insulin on board — directly targets the delta,
        # so it is expected to rank at or near the top.
        "carb_delta_6",
        "carb_delta_12",
        "ins_delta_6",
        "ins_delta_12",
        # per-user circadian baseline / deviation
        "tod_baseline",
        "tod_dev",
    }
    assert fi.index[0] in drivers, f"top feature {fi.index[0]} not an expected driver"
    assert "weather_now" not in set(fi.index[:3])  # measured, not believed (§3)
