"""Align irregular raw readings onto a regular per-user grid (ROADMAP §3).

Contract produced:
- Per user, a gap-free regular index at ``grid_minutes`` (no missing timestamps).
- ``sensor_gap`` marks every bucket that had no real reading.
- Short gaps (≤ ``max_interp_gap_min``) are linearly interpolated; longer gaps
  stay NaN and excluded downstream (§6 — never over-interpolate large gaps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .schema import validate


def _align_user(sub: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sub = sub.sort_values("ts_utc")
    freq = cfg.grid_freq

    # per-user local offset, recovered from raw (ts_local is naive wall-clock)
    offset = sub["ts_local"].iloc[0] - sub["ts_utc"].iloc[0].tz_convert(
        "UTC"
    ).tz_localize(None)

    # Bucket each reading to the *nearest* grid point (not floor): CGM timestamps
    # drift ±jitter around the nominal cadence, so rounding keeps a reading in its
    # intended bucket instead of spilling into the previous one.
    bucket = sub["ts_utc"].dt.round(freq)
    grid = pd.date_range(bucket.min(), bucket.max(), freq=freq, tz="UTC")

    g = sub.assign(_bucket=bucket).groupby("_bucket")
    agg = pd.DataFrame(
        {
            "glucose_mgdl": g["glucose_mgdl"].mean(),
            "steps": g["steps"].sum(),
            "meal_flag": g["meal_flag"].any(),
            "carbs_g": g["carbs_g"].sum(min_count=1),
            "insulin_u": g["insulin_u"].sum(min_count=1),
            "activity_flag": g["activity_flag"].any(),
            "hr": g["hr"].mean(),
            "weather_temp": g["weather_temp"].mean(),
        }
    ).reindex(grid)

    # a bucket with no reading is a sensor gap (marked before any interpolation)
    sensor_gap = agg["glucose_mgdl"].isna()

    # interpolate glucose only across short gap runs; long gaps remain NaN
    glucose = agg["glucose_mgdl"]
    isna = glucose.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    short = isna & (run_len <= cfg.max_interp_steps)
    interp = glucose.interpolate(method="linear", limit_area="inside")
    glucose = glucose.where(~short, interp)

    out = pd.DataFrame(
        {
            "user_id": sub["user_id"].iloc[0],
            "ts_utc": grid,
            "ts_local": (grid.tz_localize(None) + offset),
            "glucose_mgdl": glucose.to_numpy(),
            # empty buckets reindex to NaN in the (object) bool cols; .eq(True)
            # maps NaN→False without the deprecated fillna-downcast path.
            "meal_flag": agg["meal_flag"].eq(True).to_numpy(),
            "carbs_g": agg["carbs_g"].to_numpy(),
            "insulin_u": agg["insulin_u"].to_numpy(),
            "steps": agg["steps"].fillna(0).round().astype("int64").to_numpy(),
            "activity_flag": agg["activity_flag"].eq(True).to_numpy(),
            "hr": agg["hr"].to_numpy(),
            "weather_temp": agg["weather_temp"].to_numpy(),
            "sensor_gap": sensor_gap.to_numpy(),
        }
    )
    return out


def align(raw: pd.DataFrame, cfg: Config, *, do_validate: bool = True) -> pd.DataFrame:
    """Resample irregular raw data to a regular per-user grid; validate the contract."""
    frames = [_align_user(sub, cfg) for _, sub in raw.groupby("user_id", sort=True)]
    grid = pd.concat(frames, ignore_index=True)
    grid = grid.sort_values(["user_id", "ts_utc"]).reset_index(drop=True)
    if do_validate:
        grid = validate(grid)
    return grid
