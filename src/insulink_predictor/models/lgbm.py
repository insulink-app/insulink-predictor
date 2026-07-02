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

# Physiological sign of each therapy channel's effect on the (delta) target.
# +1: raising the feature can only raise the forecast; −1: only lower it. Applied
# as LightGBM monotone constraints when cfg.model.use_monotone is set. Momentum
# features (rate_*) are deliberately left unconstrained — mean-reversion makes
# their sign genuinely non-monotone.
_MONOTONE_SIGNS = {
    "cob": +1,
    "cob_glucose": +1,
    "carb_activity": +1,
    "iob": -1,
    "iob_glucose": -1,
    "ins_activity": -1,
}


def _monotone_vector(feature_cols: list[str]) -> list[int]:
    return [_MONOTONE_SIGNS.get(c, 0) for c in feature_cols]


def _make_regressor(
    cfg: Config | None = None, feature_cols: list[str] | None = None
) -> LGBMRegressor:
    # Regularized to curb over-reaction on quiet periods (real CGM is noisy). This
    # setting improves both the synthetic DoDs and real per-user skill vs the
    # lighter default; see diagnostics in the per-user analysis.
    reg = LGBMRegressor(
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
    if cfg is not None:
        m = cfg.model
        if m.objective == "huber":
            reg.set_params(objective="huber", alpha=m.huber_delta)
        if m.use_monotone and feature_cols is not None:
            reg.set_params(monotone_constraints=_monotone_vector(feature_cols))
    return reg


def _excursion_weight(delta: np.ndarray, cfg: Config) -> np.ndarray | None:
    """Sample weights that emphasize large excursions (|Δ| over persistence)."""
    alpha = cfg.model.excursion_weight_alpha
    if alpha <= 0:
        return None
    scale = float(np.std(delta)) or 1.0
    return 1.0 + alpha * np.minimum(
        np.abs(delta) / scale, cfg.model.excursion_weight_cap
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
        excursion = (
            train.loc[mask, f"y_{h}"] - train.loc[mask, "glucose_mgdl"]
        ).to_numpy()
        y = excursion if delta else train.loc[mask, f"y_{h}"].to_numpy()
        model = _make_regressor(cfg, feature_cols)
        model.fit(X, y, sample_weight=_excursion_weight(excursion, cfg))
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
