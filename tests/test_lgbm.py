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
        assert r["skill"] > 0, f"skill@{r['horizon_min']}min = {r['skill']} (must be > 0)"


def test_lgbm_rmse_below_persistence(lgbm_result):
    lg = lgbm_result["lgbm_metrics"].set_index("horizon_min")["rmse"]
    pe = lgbm_result["persistence_metrics"].set_index("horizon_min")["rmse"]
    assert (lg < pe).all()


def test_current_glucose_dominates_and_weather_is_low(lgbm_result):
    fi = lgbm_result["feature_importance"].set_index("feature")["gain_pct"]
    assert fi["lag_0"] == fi.max()             # current glucose is the top feature
    assert fi.get("weather_now", 0.0) < 5.0    # weather is a minor feature (§3)
