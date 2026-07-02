"""Phase 4 DoD — personalized beats global per-user; cold-start degrades gracefully."""

from __future__ import annotations

import numpy as np
import pytest

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.synth import generate
from insulink_predictor.eval.harness import run_personalize_eval


@pytest.fixture(scope="module")
def personalize_result():
    cfg = Config(
        synth={"n_users": 6, "days": 16, "seed": 7},
        split={"test_fraction": 0.2, "heldout_users": ["user_5"]},
        mlflow={"enabled": False},
    )
    grid = align(generate(cfg), cfg)
    return cfg, run_personalize_eval(cfg, df=grid, write=False)


def test_personalized_beats_global_per_user(personalize_result):
    _, res = personalize_result
    s = res["summary"]
    assert (s["personalized_mean_skill"] > s["global_mean_skill"]).all(), s


def test_majority_of_users_improve(personalize_result):
    _, res = personalize_result
    for _, r in res["summary"].iterrows():
        assert r["users_improved"] >= r["n_users"] / 2


def test_coldstart_is_graceful(personalize_result):
    _, res = personalize_result
    assert res["coldstart"]["graceful"] is True
    assert res["coldstart"]["n_heldout_users"] == 1


def test_coldstart_user_has_no_residual_and_predicts_finite(personalize_result):
    cfg, res = personalize_result
    pm = res["personalized"]
    heldout = res["heldout"]
    for h in cfg.horizons_steps:
        # a brand-new user has no per-user residual model -> falls back to the base
        assert all(uid not in pm.residual_models[h] for uid in heldout["user_id"].unique())
        valid = heldout[f"valid_{h}"].to_numpy()
        preds = pm.predict(heldout, h)
        assert np.isfinite(preds[valid]).all()
