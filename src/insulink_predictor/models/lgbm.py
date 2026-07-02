"""Direct multi-horizon LightGBM (ROADMAP §Phase 2).

One model **per horizon** (direct, not recursive — recursive is forbidden in v1
because errors accumulate). LightGBM eats mixed features, is tiny, and is
on-device-capable. Trained deterministically so results are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ..config import Config

_SEED = 42


def _make_regressor() -> LGBMRegressor:
    # Regularized to curb over-reaction on quiet periods (real CGM is noisy). This
    # setting improves both the synthetic DoDs and real per-user skill vs the
    # lighter default; see diagnostics in the per-user analysis.
    return LGBMRegressor(
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=16,
        min_child_samples=120,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=4.0,
        random_state=_SEED,
        n_jobs=1,  # deterministic
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )


def train_lgbm(
    train: pd.DataFrame, feature_cols: list[str], cfg: Config
) -> dict[int, LGBMRegressor]:
    """Train one LGBM per horizon on the valid rows of the training split.

    With ``cfg.features.predict_delta`` the target is the change over persistence
    (y_{t+h} − g_t); persistence is added back at inference.
    """
    delta = cfg.features.predict_delta
    models: dict[int, LGBMRegressor] = {}
    for h in cfg.horizons_steps:
        mask = train[f"valid_{h}"].to_numpy()
        X = train.loc[mask, feature_cols]
        y = train.loc[mask, f"y_{h}"]
        if delta:
            y = y - train.loc[mask, "glucose_mgdl"]
        model = _make_regressor()
        model.fit(X, y)
        models[h] = model
    return models


def make_pred_fn(
    models: dict[int, LGBMRegressor],
    feature_cols: list[str],
    predict_delta: bool = True,
):
    """Wrap trained models into a harness-compatible ``pred_fn(df, h)`` (absolute mg/dL)."""

    def pred_fn(df: pd.DataFrame, horizon_steps: int) -> np.ndarray:
        pred = models[horizon_steps].predict(df[feature_cols])
        if predict_delta:
            pred = pred + df["glucose_mgdl"].to_numpy()
        return pred

    return pred_fn


def feature_importance(
    models: dict[int, LGBMRegressor], feature_cols: list[str]
) -> pd.DataFrame:
    """Mean gain-importance across horizons, normalized to percent (desc)."""
    rows = []
    for h, model in models.items():
        gains = model.booster_.feature_importance(importance_type="gain")
        for col, gain in zip(feature_cols, gains):
            rows.append({"feature": col, "gain": float(gain)})
    agg = pd.DataFrame(rows).groupby("feature")["gain"].mean()
    total = agg.sum()
    pct = (agg / total * 100.0) if total > 0 else agg
    return (
        pct.round(3)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"gain": "gain_pct"})
    )
