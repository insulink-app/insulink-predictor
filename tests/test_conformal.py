"""Conformal recalibration of the curve uncertainty band.

Raw quantile GBMs are over-confident on limited CGM data; the split-conformal
offsets snap the band to nominal coverage. These tests check the mechanics: the
offsets have the right shape and monotonic ordering, and `gf curve --conformal`
still produces a valid (non-crossing) band.
"""

from __future__ import annotations

import numpy as np

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.synth import generate
from insulink_predictor.eval.harness import run_curve
from insulink_predictor.eval.split import chronological_split
from insulink_predictor.models.events import (
    build_curve_targets,
    conformal_offsets,
    train_quantile_curve_models,
)
from insulink_predictor.eval.harness import build_supervised


def _cfg():
    return Config(
        synth={"n_users": 4, "days": 16, "seed": 5}, mlflow={"enabled": False}
    )


def test_offsets_shape_and_order():
    cfg = _cfg()
    grid = align(generate(cfg), cfg)
    sup, cols = build_supervised(cfg, grid)
    max_step = max(cfg.horizons_steps)
    sup = build_curve_targets(sup, max_step)
    proper, calib = chronological_split(sup, 0.25)
    qs = (0.1, 0.5, 0.9)
    qm = train_quantile_curve_models(
        proper, cols, max_step, qs, cfg.features.predict_delta
    )
    off = conformal_offsets(qm, calib, cols, max_step, cfg.features.predict_delta)
    assert set(off) == set(qs)
    for q in qs:
        assert off[q].shape == (max_step,)
        assert np.isfinite(off[q]).all()
    # lower-quantile offset should not exceed the upper-quantile offset (per step)
    assert (off[0.1] <= off[0.9] + 1e-9).all()


def test_curve_conformal_band_valid():
    cfg = _cfg()
    grid = align(generate(cfg), cfg)
    res = run_curve(cfg, df=grid, horizon_min=60, band=True, conformal=True, out=None)
    assert res["examples"], "expected at least one forecast example"
    for ex in res["examples"]:
        lo = np.asarray(ex["lower"])
        hi = np.asarray(ex["upper"])
        assert np.isfinite(lo).all() and np.isfinite(hi).all()
        assert (lo <= hi + 1e-6).all()
