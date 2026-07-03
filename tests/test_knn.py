"""k-NN baseline + head-to-head comparison harness.

k-NN is not required to win (that is the empirical question the comparison
answers); these tests assert the wiring is sound — it runs behind the shared
pred_fn interface, produces finite predictions, and honours the delta target —
and that LGBM stays the reference that beats persistence.
"""

from __future__ import annotations

import numpy as np
import pytest

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.synth import generate
from insulink_predictor.eval.harness import (
    build_supervised,
    run_curve,
    run_model_comparison,
)
from insulink_predictor.eval.split import chronological_split
from insulink_predictor.models.knn import make_knn_pred_fn, train_knn


@pytest.fixture(scope="module")
def comparison():
    cfg = Config(synth={"n_users": 4, "days": 12, "seed": 3}, mlflow={"enabled": False})
    grid = align(generate(cfg), cfg)
    return run_model_comparison(cfg, df=grid, k=50, write=False)


def test_comparison_has_all_four_models(comparison):
    models = set(comparison["comparison"]["model"])
    assert {"persistence", "lgbm"} <= models
    assert any(m.startswith("knn(") for m in models)
    assert any(m.startswith("knn_wtd(") for m in models)


def test_all_skills_finite(comparison):
    assert np.isfinite(comparison["comparison"]["skill"].to_numpy()).all()


def test_lgbm_remains_the_reference(comparison):
    # LGBM must still beat persistence at every horizon (the Phase-2 gate).
    lg = comparison["comparison"]
    lg = lg[lg["model"] == "lgbm"]
    assert (lg["skill"] > 0).all()


def test_knn_predicts_finite_absolute_glucose():
    # A trained k-NN must return absolute mg/dL (delta added back), all finite on
    # valid rows — proving the delta round-trip and imputation handle NaNs.
    cfg = Config(synth={"n_users": 3, "days": 10, "seed": 5}, mlflow={"enabled": False})
    grid = align(generate(cfg), cfg)
    sup, cols = build_supervised(cfg, grid)
    train, test = chronological_split(sup, cfg.split.test_fraction)
    models = train_knn(train, cols, cfg, k=30)
    pred_fn = make_knn_pred_fn(models, cols, cfg.features.predict_delta)
    for h in cfg.horizons_steps:
        valid = test[f"valid_{h}"].to_numpy()
        preds = np.asarray(pred_fn(test, h))
        assert np.isfinite(preds[valid]).all()
        # sane physiological range — not echoing raw deltas
        assert (preds[valid] > 20).all() and (preds[valid] < 600).all()


def test_curve_runs_on_knn_with_empirical_band(tmp_path):
    # gf curve --model knn: trajectory forecast + a band read off the neighbor
    # set (no separately-trained quantile models), enclosing the median.
    cfg = Config(synth={"n_users": 4, "days": 14, "seed": 7}, mlflow={"enabled": False})
    grid = align(generate(cfg), cfg)
    # tmp out — never pollute reports/ with synthetic-data diagrams.
    res = run_curve(
        cfg,
        df=grid,
        horizon_min=60,
        model="knn",
        knn_k=50,
        band=True,
        out=tmp_path / "c.png",
    )
    assert res["examples"], "expected at least one post-meal forecast example"
    for ex in res["examples"]:
        lo = np.asarray(ex["lower"])
        hi = np.asarray(ex["upper"])
        med = np.asarray(ex["predicted"])
        assert np.isfinite(lo).all() and np.isfinite(hi).all()
        assert (lo <= med + 1e-6).all() and (med <= hi + 1e-6).all()
