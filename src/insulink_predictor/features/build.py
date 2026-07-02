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


def _biexp_response(
    rise_min: float, decay_min: float, grid_min: int, length_min: int = 240
) -> np.ndarray:
    """Bi-exponential glucose response kernel (level), peak normalized to 1.0."""
    t = np.arange(0, length_min, grid_min)
    k = np.exp(-t / decay_min) - np.exp(-t / rise_min)
    peak = k.max()
    return k / peak if peak > 0 else k


def _causal_conv(df: pd.DataFrame, col: str, kernel: np.ndarray) -> pd.Series:
    """Causal convolution of a per-bucket dose series with ``kernel`` (past doses)."""
    x = df[col].fillna(0.0)
    if kernel.size == 0:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return x.groupby(df["user_id"], sort=False).transform(
        lambda s: pd.Series(np.convolve(s.to_numpy(), kernel)[: len(s)], index=s.index)
    )


def _activity(
    df: pd.DataFrame, col: str, tau_min: float, grid_min: int, length_min: int = 360
) -> pd.Series:
    """Causal activity signal: past doses convolved with t·exp(−t/τ) (peak at τ).

    Unlike the IOB/COB *stock*, this is the *rate of action* now — the near-term
    glucose pressure that drives the change over the next 30–60 min.
    """
    n = max(2, length_min // grid_min)
    t = np.arange(n) * grid_min
    ker = t * np.exp(-t / tau_min)
    ker = ker / ker.sum() if ker.sum() > 0 else ker
    x = df[col].fillna(0.0)
    return x.groupby(df["user_id"], sort=False).transform(
        lambda s: pd.Series(np.convolve(s.to_numpy(), ker)[: len(s)], index=s.index)
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
    if _has_channel(df, "insulin_u"):
        df["_bolus_flag"] = df["insulin_u"].fillna(0.0) > 0
        df["time_since_bolus"] = _time_since(df, "_bolus_flag")
        cols.append("time_since_bolus")

    # --- circadian (from local wall-clock) ----------------------------------
    hour = df["ts_local"].dt.hour + df["ts_local"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["is_weekend"] = (df["ts_local"].dt.dayofweek >= 5).astype(int)
    df["day_of_week"] = df["ts_local"].dt.dayofweek
    cols += ["hour_sin", "hour_cos", "is_weekend", "day_of_week"]

    # --- per-user circadian baseline (causal, past-only) --------------------
    # Expanding mean glucose at this user's local time-of-day bin, using only
    # earlier rows (shift(1)), plus the current deviation from it. Captures the
    # per-user circadian phase that shared hour_sin/cos cannot (it averages out
    # across users). Strictly causal: a future value never enters a past bin mean.
    if fc.use_tod_baseline:
        bin_min = max(1, fc.tod_bin_min)
        df["_tod_bin"] = (
            df["ts_local"].dt.hour * 60 + df["ts_local"].dt.minute
        ) // bin_min
        base = df.groupby(["user_id", "_tod_bin"], sort=False)[
            "glucose_mgdl"
        ].transform(lambda s: s.expanding().mean().shift(1))
        df["tod_baseline"] = base
        df["tod_dev"] = df["glucose_mgdl"] - base
        df.drop(columns="_tod_bin", inplace=True)
        cols += ["tod_baseline", "tod_dev"]

    # --- horizon-specific physiological forecast (causal) -------------------
    # Expected glucose change over the next h steps from doses already on board:
    # conv(dose, kernel[h:]) − conv(dose, kernel) = effect at t+h minus effect now,
    # both using only past doses. Directly estimates the delta target's meal/insulin
    # component, per horizon.
    if fc.use_physio_delta:
        specs = [
            ("carbs_g", fc.carb_resp_rise_min, fc.carb_resp_decay_min, +1.0, "carb"),
            ("insulin_u", fc.ins_resp_rise_min, fc.ins_resp_decay_min, -1.0, "ins"),
        ]
        for col, rise, decay, sign, tag in specs:
            if not _has_channel(df, col):
                continue
            k = _biexp_response(rise, decay, grid)
            now = _causal_conv(df, col, k)
            for h in cfg.horizons_steps:
                name = f"{tag}_delta_{h}"
                df[name] = sign * (_causal_conv(df, col, k[h:]) - now)
                cols.append(name)

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

    # --- therapy-scaled features: COB/IOB in glucose-equivalent mg/dL --------
    # ISF (mg/dL per U) and ICR (g per U) come from user_settings; CSF = ISF/ICR
    # (mg/dL per g). These hand the model each user's sensitivity directly, so a
    # global model no longer has to average over very different responders.
    # Only the time-varying glucose-equivalent effects are added here. The raw
    # ISF/ICR constants are user-identity and belong to personalization
    # conditioning (Phase 4a), not the shared base feature set.
    if fc.use_therapy and _has_channel(df, "isf") and _has_channel(df, "icr"):
        csf = df["isf"] / df["icr"]
        if "cob" in df.columns:
            df["cob_glucose"] = df["cob"] * csf  # expected mg/dL rise still on board
            cols.append("cob_glucose")
        if "iob" in df.columns:
            df["iob_glucose"] = (
                df["iob"] * df["isf"]
            )  # expected mg/dL drop still on board
            cols.append("iob_glucose")
        # Activity (rate of action) in glucose units — the near-term pressure that
        # drives the *change*, which IOB/COB stock misses.
        if _has_channel(df, "insulin_u"):
            act = _activity(df, "insulin_u", fc.ins_activity_tau_min, grid)
            df["ins_activity"] = act * df["isf"]
            cols.append("ins_activity")
        if _has_channel(df, "carbs_g"):
            act = _activity(df, "carbs_g", fc.carb_activity_tau_min, grid)
            df["carb_activity"] = act * csf
            cols.append("carb_activity")

    return df, cols
