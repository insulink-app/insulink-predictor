"""Serving-path checks: the grid/feature assembly and (if a model is present)
the end-to-end forecast contract. Runs without a DB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from serve import app as serve


def _ramp_readings(n: int = 30, start_ms: int = 1_700_000_000_000) -> list:
    """n readings, 5 min apart, rising 100 -> ~145 mg/dL."""
    step_ms = 5 * 60 * 1000
    return [
        serve.Reading(ts=start_ms + i * step_ms, mgdl=100.0 + 1.5 * i) for i in range(n)
    ]


def test_grid_is_regular_and_matches_glucose():
    grid = serve._grid_from_readings(_ramp_readings(24))
    deltas = grid["ts_utc"].diff().dropna().dt.total_seconds().unique()
    assert deltas.tolist() == [300.0]  # exactly 5-min spacing, no gaps
    assert grid["glucose_mgdl"].iloc[0] == 100.0
    assert not grid["sensor_gap"].any()


@pytest.mark.skipif(
    not Path(serve._MODEL_PATH).exists(), reason="no trained model artifact"
)
@pytest.mark.parametrize("horizon", [30, 60])
def test_forecast_shape_and_range(horizon):
    curve = serve.forecast(_ramp_readings(36), horizon)
    assert len(curve) == horizon // serve._MODEL["grid_minutes"]
    assert all(np.isfinite(curve))
    assert all(serve._GLUCOSE_LO <= v <= serve._GLUCOSE_HI for v in curve)
