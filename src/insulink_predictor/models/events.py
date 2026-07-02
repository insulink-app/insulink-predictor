"""Event detection + multi-output trajectory forecast (ROADMAP §Phase 3).

Be sharp where persistence fails — around events (meals, glucose rises, activity).
At a trigger the model forecasts the whole curve ``t+5 … t+60`` (one LightGBM per
step, direct, reusing the Phase-2 causal features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .lgbm import _make_regressor


def detect_events(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Flag causal triggers: meal logged / glucose rise over threshold / activity."""
    out = df.sort_values(["user_id", "ts_utc"]).reset_index(drop=True).copy()

    out["event_meal"] = (
        out["meal_flag"].astype(bool) if cfg.event.meal_trigger else False
    )

    if "rate_short" in out.columns:
        rate = out["rate_short"]
    else:  # (g_t - g_{t-1}) / grid_minutes, per user
        prev = out.groupby("user_id", sort=False)["glucose_mgdl"].shift(1)
        rate = (out["glucose_mgdl"] - prev) / cfg.grid_minutes
    out["event_rise"] = (rate >= cfg.event.glucose_rate_threshold).fillna(False)

    if cfg.event.activity_trigger and "activity_flag" in out.columns:
        out["event_activity"] = out["activity_flag"].astype(bool)
    else:
        out["event_activity"] = False

    out["event_any"] = out[["event_meal", "event_rise", "event_activity"]].any(axis=1)
    return out


def build_curve_targets(df: pd.DataFrame, max_step: int) -> pd.DataFrame:
    """Add per-step future glucose ``cy_{k}`` and validity ``cvalid_{k}`` (k=1..max)."""
    df = df.sort_values(["user_id", "ts_utc"]).reset_index(drop=True).copy()
    g = df.groupby("user_id", sort=False)["glucose_mgdl"]
    gap = df.groupby("user_id", sort=False)["sensor_gap"]
    base_ok = df["glucose_mgdl"].notna() & ~df["sensor_gap"]
    for k in range(1, max_step + 1):
        df[f"cy_{k}"] = g.shift(-k)
        df[f"cvalid_{k}"] = base_ok & df[f"cy_{k}"].notna() & (gap.shift(-k) == False)  # noqa: E712
    return df


def train_curve_models(
    train: pd.DataFrame,
    feature_cols: list[str],
    max_step: int,
    predict_delta: bool = True,
) -> dict[int, object]:
    """Train one LightGBM per step ``1..max_step`` on valid rows.

    With ``predict_delta`` each step's target is the change over persistence
    (cy_k − g_t), added back in ``forecast_curve``.
    """
    models: dict[int, object] = {}
    for k in range(1, max_step + 1):
        mask = train[f"cvalid_{k}"].to_numpy()
        y = train.loc[mask, f"cy_{k}"]
        if predict_delta:
            y = y - train.loc[mask, "glucose_mgdl"]
        models[k] = _make_regressor().fit(train.loc[mask, feature_cols], y)
    return models


def forecast_curve(
    df: pd.DataFrame,
    models: dict[int, object],
    feature_cols: list[str],
    predict_delta: bool = True,
) -> np.ndarray:
    """Return the predicted trajectory in absolute mg/dL, shape ``(n_rows, max_step)``."""
    X = df[feature_cols]
    steps = sorted(models)
    preds = np.column_stack([models[k].predict(X) for k in steps])
    if predict_delta:
        preds = preds + df["glucose_mgdl"].to_numpy()[:, None]
    return preds
