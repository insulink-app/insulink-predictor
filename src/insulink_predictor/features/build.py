"""Strictly causal feature builder (ROADMAP §Phase 2 + §6).

Every feature at time ``t`` depends **only** on data at times ``≤ t``. This is
the single most important invariant in the project — the leakage test perturbs
future values and asserts these features do not move.

Notes on causality:
- Lags use ``shift(+k)`` (past values). Targets (elsewhere) use ``shift(-h)``.
- Rolling windows are trailing and *include* ``t`` — "now" is not the future.
- ``time_since_*`` forward-fills past event timestamps only.
- COB/IOB are causal IIR accumulators (only past + current carbs/insulin).
- Optional channels (carbs, insulin, hr, weather) degrade cleanly when absent
  or all-NaN, so the same builder serves insulin and non-insulin users.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..config import Config


def _has_channel(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and bool(df[col].notna().any())


def _time_since(df: pd.DataFrame, flag_col: str) -> pd.Series:
    """Minutes since the most recent True in ``flag_col`` (per user, causal)."""
    marked_ts = df["ts_utc"].where(df[flag_col].astype(bool))
    last = marked_ts.groupby(df["user_id"]).ffill()
    return (df["ts_utc"] - last).dt.total_seconds() / 60.0


def _decay_accumulate(
    df: pd.DataFrame, col: str, tau_min: float, grid_min: int
) -> pd.Series:
    """Causal exponential accumulator y[n] = x[n] + decay·y[n-1] (COB/IOB proxy)."""
    decay = math.exp(-grid_min / tau_min)
    x = df[col].fillna(0.0)
    return x.groupby(df["user_id"], sort=False).transform(
        lambda s: pd.Series(lfilter([1.0], [1.0, -decay], s.to_numpy()), index=s.index)
    )


def build_features(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(df_with_features, feature_cols)``. Input is the aligned grid."""
    fc = cfg.features
    grid = cfg.grid_minutes
    df = df.sort_values(["user_id", "ts_utc"]).reset_index(drop=True).copy()
    g = df.groupby("user_id", sort=False)["glucose_mgdl"]
    cols: list[str] = []

    # --- glucose lags (past) ------------------------------------------------
    for lag_min in fc.glucose_lags_min:
        k = lag_min // grid
        name = f"lag_{lag_min}"
        df[name] = g.shift(k)
        cols.append(name)

    # --- change rates & acceleration (past differences) ---------------------
    df["rate_short"] = (df["lag_0"] - df["lag_5"]) / 5.0  # mg/dL per min
    df["rate_long"] = (df["lag_0"] - df["lag_30"]) / 30.0
    df["accel"] = (df["lag_0"] - 2 * df["lag_5"] + df["lag_10"]) / (5.0**2)
    cols += ["rate_short", "rate_long", "accel"]

    # --- trailing rolling stats (include t => causal) -----------------------
    for w_min in fc.roll_windows_min:
        w = max(1, w_min // grid)
        for stat in ("mean", "std", "min", "max"):
            name = f"roll{w_min}_{stat}"
            df[name] = g.transform(
                lambda s, w=w, stat=stat: getattr(s.rolling(w, min_periods=1), stat)()
            )
            cols.append(name)

    # --- time since events --------------------------------------------------
    df["time_since_meal"] = _time_since(df, "meal_flag")
    cols.append("time_since_meal")
    if _has_channel(df, "activity_flag"):
        df["time_since_activity"] = _time_since(df, "activity_flag")
        cols.append("time_since_activity")

    # --- circadian (from local wall-clock) ----------------------------------
    hour = df["ts_local"].dt.hour + df["ts_local"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["is_weekend"] = (df["ts_local"].dt.dayofweek >= 5).astype(int)
    cols += ["hour_sin", "hour_cos", "is_weekend"]

    # --- activity / steps (trailing sums) -----------------------------------
    gs = df.groupby("user_id", sort=False)["steps"]
    for w_min in fc.steps_windows_min:
        w = max(1, w_min // grid)
        name = f"steps_{w_min}"
        df[name] = gs.transform(lambda s, w=w: s.rolling(w, min_periods=1).sum())
        cols.append(name)
    if _has_channel(df, "activity_flag"):
        df["workout_flag"] = df["activity_flag"].astype(int)
        cols.append("workout_flag")

    # --- optional context channels (degrade cleanly) ------------------------
    if fc.use_carbs and _has_channel(df, "carbs_g"):
        df["cob"] = _decay_accumulate(df, "carbs_g", fc.cob_tau_min, grid)
        cols.append("cob")
    if fc.use_insulin and _has_channel(df, "insulin_u"):
        df["iob"] = _decay_accumulate(df, "insulin_u", fc.iob_tau_min, grid)
        cols.append("iob")
    if fc.use_hr and _has_channel(df, "hr"):
        df["hr_now"] = df["hr"]
        cols.append("hr_now")
    if fc.use_weather and _has_channel(df, "weather_temp"):
        df["weather_now"] = df["weather_temp"]
        cols.append("weather_now")

    return df, cols
