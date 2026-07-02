"""Data Contract (ROADMAP §3) enforced with pandera.

The aligned grid is a regular 5-min-per-user table. This schema is the single
gate every DataFrame passes through before it is trusted downstream.

Key invariant: ``glucose_mgdl`` may be NaN **only** inside a marked sensor gap
(§6 — large CGM gaps are excluded, never over-interpolated).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

# Physiologically implausible glucose is a data error, not a reading.
_GLUCOSE_LO, _GLUCOSE_HI = 10.0, 700.0


def _glucose_nan_only_in_gap(df: pd.DataFrame) -> pd.Series:
    """Row-wise: a NaN glucose is allowed only where ``sensor_gap`` is True."""
    return df["glucose_mgdl"].notna() | df["sensor_gap"]


GRID_SCHEMA = pa.DataFrameSchema(
    columns={
        "user_id": pa.Column(str, nullable=False),
        # Grid anchor — timezone-aware UTC.
        "ts_utc": pa.Column("datetime64[ns, UTC]", nullable=False),
        # Local wall-clock (tz stripped) for circadian features (§3).
        "ts_local": pa.Column("datetime64[ns]", nullable=False),
        "glucose_mgdl": pa.Column(
            float,
            checks=pa.Check.in_range(_GLUCOSE_LO, _GLUCOSE_HI),
            nullable=True,  # only inside sensor gaps (enforced by the df-level check)
        ),
        "meal_flag": pa.Column(bool, nullable=False),
        "carbs_g": pa.Column(
            float, checks=pa.Check.ge(0), nullable=True, required=False
        ),
        "insulin_u": pa.Column(
            float, checks=pa.Check.ge(0), nullable=True, required=False
        ),
        "steps": pa.Column("int64", checks=pa.Check.ge(0), nullable=False),
        "activity_flag": pa.Column(bool, nullable=False, required=False),
        "hr": pa.Column(float, checks=pa.Check.ge(0), nullable=True, required=False),
        "weather_temp": pa.Column(float, nullable=True, required=False),
        "sensor_gap": pa.Column(bool, nullable=False),
    },
    checks=pa.Check(
        _glucose_nan_only_in_gap,
        error="glucose_mgdl may be NaN only where sensor_gap is True",
    ),
    strict=False,  # tolerate extra derived columns
    coerce=True,
    ordered=False,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate against the grid contract; raises ``pandera.errors.SchemaError``."""
    return GRID_SCHEMA.validate(df)
