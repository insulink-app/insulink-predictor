"""Phase 0 DoD — alignment yields a provably regular grid; gaps handled per §6."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glucose_forecast.config import Config
from glucose_forecast.data.align import align


def _raw_at_minutes(minutes: list[int]) -> pd.DataFrame:
    """Build a raw frame at exact grid minutes with glucose = 100 + minute."""
    ts = pd.Timestamp("2025-01-06 00:00:00", tz="UTC") + pd.to_timedelta(minutes, unit="min")
    n = len(minutes)
    return pd.DataFrame(
        {
            "user_id": "u",
            "ts_utc": ts,
            "ts_local": ts.tz_localize(None),  # tz offset 0
            "glucose_mgdl": [100.0 + m for m in minutes],
            "meal_flag": [False] * n,
            "carbs_g": [np.nan] * n,
            "insulin_u": [np.nan] * n,
            "steps": [0.0] * n,
            "activity_flag": [False] * n,
            "hr": [70.0] * n,
            "weather_temp": [10.0] * n,
        }
    )


def test_regular_grid_per_user(grid, cfg: Config):
    """Every user's timestamps step by exactly grid_minutes — no unmarked holes."""
    for _, g in grid.groupby("user_id"):
        diffs = g["ts_utc"].diff().dropna().dt.total_seconds() / 60
        assert (diffs == cfg.grid_minutes).all()


def test_glucose_nan_implies_gap(grid):
    assert (grid.loc[grid["glucose_mgdl"].isna(), "sensor_gap"]).all()


def test_short_gap_interpolated_long_gap_left_nan():
    cfg = Config(grid_minutes=5, max_interp_gap_min=15)  # interpolate <= 3 steps
    # present: 0,5,10, [skip 15,20 => short], 25,30,35,40, [skip 45..65 => long], 70,75
    present = [0, 5, 10, 25, 30, 35, 40, 70, 75]
    out = align(_raw_at_minutes(present), cfg).set_index("ts_utc")

    base = pd.Timestamp("2025-01-06 00:00:00", tz="UTC")

    def at(minute: int):
        return out.loc[base + pd.Timedelta(minutes=minute)]

    # regular grid spans 0..75 with no missing timestamps
    diffs = out.index.to_series().diff().dropna().dt.total_seconds() / 60
    assert (diffs == 5).all()

    # short gap (2 buckets = 10 min): interpolated, value on the line, still a gap
    assert at(15)["sensor_gap"] and not np.isnan(at(15)["glucose_mgdl"])
    assert abs(at(15)["glucose_mgdl"] - 115.0) < 1e-6
    assert abs(at(20)["glucose_mgdl"] - 120.0) < 1e-6

    # long gap (5 buckets = 25 min > 15): NaN, marked as gap, NOT interpolated
    for minute in (45, 50, 55, 60, 65):
        assert at(minute)["sensor_gap"] and np.isnan(at(minute)["glucose_mgdl"])

    # real readings are not marked as gaps
    for minute in present:
        assert not at(minute)["sensor_gap"]
        assert abs(at(minute)["glucose_mgdl"] - (100.0 + minute)) < 1e-6
