"""Phase 1 — error-grid zone percentages are well-formed and sane."""

from __future__ import annotations

import numpy as np

from glucose_forecast.eval.error_grid import (
    clarke_zone_pct,
    parkes_zone_pct,
    unsafe_fraction,
)


def test_perfect_prediction_is_all_zone_a():
    y = np.array([70, 90, 120, 180, 250, 60, 300.0])
    pz = parkes_zone_pct(y, y)
    assert pz["A"] == 100.0
    assert unsafe_fraction(pz) == 0.0


def test_parkes_zone_percentages_sum_to_100():
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 300, 500)
    pred = y + rng.normal(0, 25, 500)
    pz = parkes_zone_pct(y, pred)
    assert abs(sum(pz.values()) - 100.0) < 1e-6


def test_clarke_zone_percentages_sum_to_100():
    rng = np.random.default_rng(1)
    y = rng.uniform(50, 300, 500)
    pred = y + rng.normal(0, 25, 500)
    cz = clarke_zone_pct(y, pred)
    assert abs(sum(cz.values()) - 100.0) < 1e-6


def test_large_errors_produce_unsafe_zones():
    # wildly wrong predictions must land some points outside A/B.
    y = np.array([80, 80, 80, 300, 300, 300.0])
    pred = np.array([300, 300, 300, 40, 40, 40.0])
    assert unsafe_fraction(parkes_zone_pct(y, pred)) > 0.0
