"""Phase 1 — metrics, especially the skill-score north star."""

from __future__ import annotations

import numpy as np

from glucose_forecast.eval.metrics import mae, rmse, skill_score


def test_rmse_zero_when_perfect():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_known_value():
    # errors 3 and 4 -> sqrt((9+16)/2) = sqrt(12.5)
    assert abs(rmse([0, 0], [3, 4]) - np.sqrt(12.5)) < 1e-9


def test_mae_known_value():
    assert mae([0, 0, 0], [1, 2, 3]) == 2.0


def test_skill_persistence_vs_itself_is_zero():
    # DoD: persistence measured against itself scores exactly 0.
    r = rmse([100, 110, 120], [98, 112, 119])
    assert skill_score(r, r) == 0.0


def test_skill_perfect_model_is_one():
    assert skill_score(0.0, 10.0) == 1.0


def test_skill_worse_than_persistence_is_negative():
    assert skill_score(20.0, 10.0) == -1.0


def test_metrics_ignore_nan_pairs():
    assert rmse([1.0, np.nan, 3.0], [1.0, 99.0, 3.0]) == 0.0
