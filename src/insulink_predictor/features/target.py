"""Supervised target construction (shared by every model).

Single source of truth for the ``(t → t+h)`` pairs. Targets are built **per
user** so no future value ever crosses a user boundary, and rows are marked
invalid when either endpoint falls in a sensor gap (§6 — gaps are excluded from
prediction, never predicted over).
"""

from __future__ import annotations

import pandas as pd


def build_targets(df: pd.DataFrame, horizons_steps: list[int]) -> pd.DataFrame:
    """Add ``y_{h}`` (future glucose) and ``valid_{h}`` (usable row) per horizon.

    For each horizon ``h`` (in grid steps):
    - ``y_{h}`` = glucose at ``t + h`` within the same user (NaN past series end).
    - ``valid_{h}`` = current point is real (not a gap) **and** the target point
      is real (not a gap, not missing).
    """
    df = df.sort_values(["user_id", "ts_utc"]).reset_index(drop=True).copy()
    grp = df.groupby("user_id", sort=False)

    for h in horizons_steps:
        df[f"y_{h}"] = grp["glucose_mgdl"].shift(-h)
        # sensor_gap at t+h; beyond the series end counts as invalid.
        fut_gap = grp["sensor_gap"].shift(-h)
        df[f"valid_{h}"] = (
            df["glucose_mgdl"].notna()
            & ~df["sensor_gap"]
            & df[f"y_{h}"].notna()
            & (fut_gap == False)  # noqa: E712 — NaN (past end) must fail, so no `~`
        )
    return df
