"""Recency weighting: old rows fade, they are not cut off (ModelConfig)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from insulink_predictor.config import load_config
from insulink_predictor.models.lgbm import recency_weight, sample_weight


def _rows(days: list[float]) -> pd.DataFrame:
    """Rows aged ``days`` before the newest one."""
    newest = pd.Timestamp("2026-09-01", tz="UTC")
    return pd.DataFrame(
        {"ts_utc": [newest - pd.Timedelta(days=age) for age in sorted(days, reverse=True)]}
    )


def test_no_half_life_leaves_every_row_equal():
    cfg = load_config()
    assert cfg.model.recency_half_life_days == 0.0  # committed default is off
    assert recency_weight(_rows([0, 30, 300]), cfg) is None


def test_one_half_life_back_weighs_half():
    cfg = load_config()
    cfg.model.recency_half_life_days = 30.0
    weights = recency_weight(_rows([0, 30, 60]), cfg)
    assert np.allclose(weights, [0.25, 0.5, 1.0])


def test_age_is_measured_against_the_newest_row_not_now():
    """A walk-forward fold must weigh its own training data the way production does."""
    cfg = load_config()
    cfg.model.recency_half_life_days = 30.0
    old_fold = _rows([0, 30])
    old_fold["ts_utc"] = old_fold["ts_utc"] - pd.Timedelta(days=400)
    assert np.allclose(recency_weight(old_fold, cfg), [0.5, 1.0])


def test_excursion_and_recency_multiply_when_both_are_on():
    cfg = load_config()
    cfg.model.recency_half_life_days = 30.0
    cfg.model.excursion_weight_alpha = 1.0
    rows = _rows([0, 30])
    delta = np.array([10.0, -10.0])
    combined = sample_weight(rows, delta, cfg)
    assert np.allclose(combined / recency_weight(rows, cfg), combined[0] / 0.5)
    assert (combined > 0).all()
