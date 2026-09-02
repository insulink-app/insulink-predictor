"""Align irregular raw readings onto a regular per-user grid (ROADMAP §3).

Contract produced:
- Per user, a gap-free regular index at ``grid_minutes`` (no missing timestamps).
- ``sensor_gap`` marks every bucket that had no real reading.
- Short gaps (≤ ``max_interp_gap_min``) are linearly interpolated; longer gaps
  stay NaN and excluded downstream (§6 — never over-interpolate large gaps).
"""

from __future__ import annotations

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
    # Anchor the grid to the glucose signal's span. Auxiliary channels (e.g. daily
    # activity, which can predate the CGM) must not stretch the grid into
    # glucose-less time; readings outside the CGM span aren't forecastable anyway.
    gb = bucket[sub["glucose_mgdl"].notna().to_numpy()]
    lo, hi = (gb.min(), gb.max()) if len(gb) else (bucket.min(), bucket.max())
    grid = pd.date_range(lo, hi, freq=freq, tz="UTC")

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

    # Basal is summed per bucket, not meaned: it is units that entered the body in
    # that stretch, exactly like a bolus. Optional, so a dataset without a pump
    # (or the synthetic one) simply has no such column.
    if "basal_u" in sub.columns:
        agg["basal_u"] = g["basal_u"].sum(min_count=1).reindex(grid)

    # A pod activation is a rare pulse, so the bucket it lands in is all that is
    # needed; build_features turns it into time_since_pod. Optional, like basal.
    if "pod_flag" in sub.columns:
        agg["pod_flag"] = g["pod_flag"].any().reindex(grid)

    # A bucket's position is the mean of its fixes — five minutes of standing still
    # produces a handful of jittering fixes around one spot, and their mean is that
    # spot. ponytail: a plain mean is wrong across the ±180° meridian; nobody
    # forecasts glucose there, and a fix-count-weighted circular mean is the fix.
    for col in ("lat", "lon"):
        if col in sub.columns:
            agg[col] = g[col].mean().reindex(grid)

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
    if "basal_u" in agg.columns:
        out["basal_u"] = agg["basal_u"].to_numpy()
    if "pod_flag" in agg.columns:
        out["pod_flag"] = agg["pod_flag"].eq(True).to_numpy()
    for col in ("lat", "lon"):
        if col in agg.columns:
            out[col] = agg[col].to_numpy()

    # Carry per-user therapy settings through unchanged (constant per user).
    for col in ("isf", "icr"):
        if col in sub.columns:
            out[col] = sub[col].iloc[0]

    # Daily activity totals: one value per day -> broadcast to every bucket of that
    # local date (max over the date; NaN for days with no measurement). Lagged to
    # "yesterday" causally in build_features; degrades cleanly when absent.
    for col in ("daily_steps", "daily_distance"):
        if col in sub.columns:
            out[col] = g[col].max().reindex(grid).to_numpy()
            out[col] = out.groupby(out["ts_local"].dt.date)[col].transform("max")
    return out


def align(raw: pd.DataFrame, cfg: Config, *, do_validate: bool = True) -> pd.DataFrame:
    """Resample irregular raw data to a regular per-user grid; validate the contract."""
    # Skip users with no CGM (e.g. an empty duplicate account) — there is nothing
    # to forecast, and their auxiliary rows would only inflate the grid with gaps.
    groups = list(raw.groupby("user_id", sort=True))
    with_cgm = [(k, sub) for k, sub in groups if sub["glucose_mgdl"].notna().any()]
    frames = [_align_user(sub, cfg) for _, sub in (with_cgm or groups)]
    grid = pd.concat(frames, ignore_index=True)
    grid = grid.sort_values(["user_id", "ts_utc"]).reset_index(drop=True)
    if do_validate:
        grid = validate(grid)
    return grid
