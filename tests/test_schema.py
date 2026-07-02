"""Phase 0 — the pandera Data Contract accepts valid grids and rejects violations."""

from __future__ import annotations

import pandera.errors
import pytest

from insulink_predictor.data.schema import validate


def test_valid_grid_passes(grid):
    # align() already validates, but assert the public entry point too.
    validate(grid)


def test_glucose_nan_outside_gap_is_rejected(grid):
    bad = grid.copy()
    # a NaN glucose where sensor_gap is False violates the core invariant.
    idx = bad.index[~bad["sensor_gap"]][0]
    bad.loc[idx, "glucose_mgdl"] = float("nan")
    bad.loc[idx, "sensor_gap"] = False
    with pytest.raises(pandera.errors.SchemaError):
        validate(bad)


def test_negative_steps_rejected(grid):
    bad = grid.copy()
    bad.loc[bad.index[0], "steps"] = -1
    with pytest.raises(pandera.errors.SchemaError):
        validate(bad)


def test_out_of_range_glucose_rejected(grid):
    bad = grid.copy()
    bad.loc[bad.index[0], "glucose_mgdl"] = 5000.0  # implausible
    with pytest.raises(pandera.errors.SchemaError):
        validate(bad)


def test_missing_required_column_rejected(grid):
    bad = grid.drop(columns=["meal_flag"])
    with pytest.raises(pandera.errors.SchemaError):
        validate(bad)
