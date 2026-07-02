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


# Determinism/reproducibility flags shared by every LGBM we build.
_BASE_PARAMS = dict(
    subsample_freq=1,
    random_state=_SEED,
    n_jobs=1,  # deterministic
    deterministic=True,
    force_col_wise=True,
    verbose=-1,
)

# Default (horizon-agnostic) config. Regularized to curb over-reaction on quiet
# periods (real CGM is noisy). Used by the curve/quantile/personalize paths and
# as the fallback for any horizon without a tuned config.
_DEFAULT_PARAMS = dict(
    n_estimators=350,
    learning_rate=0.04,
    num_leaves=16,
    min_child_samples=120,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=4.0,
)

# Per-horizon configs from an Optuna/TPE search (120 trials/horizon) on real
# per-user data: chosen on a chronological validation slice, then confirmed by
# 5-fold walk-forward (tuned beat the default at 60min in 5/5 folds, +0.013 mean
# skill; at 30min in 4/5, +0.007). Both favour a slow learning rate with many
# trees, but the horizons want genuinely different capacity. Tuned on a single
# user's data — revisit when multi-user data lands.
_TUNED_PARAMS: dict[int, dict] = {
    30: dict(
        n_estimators=480,
        learning_rate=0.0079,
        num_leaves=44,
        min_child_samples=68,
        min_child_weight=0.005,
        reg_lambda=0.319,
        reg_alpha=0.003,
        subsample=0.79,
        colsample_bytree=0.91,
        max_depth=16,
        min_split_gain=0.299,
    ),
    60: dict(
        n_estimators=666,
        learning_rate=0.0068,
        num_leaves=218,
        min_child_samples=47,
        min_child_weight=0.456,
        reg_lambda=0.009,
        reg_alpha=32.448,
        subsample=0.50,
        colsample_bytree=0.93,
        max_depth=6,
        min_split_gain=0.144,
    ),
}


def _make_regressor(
    cfg: Config | None = None,
    feature_cols: list[str] | None = None,
    horizon_min: int | None = None,
) -> LGBMRegressor:
    """Deterministic LGBM. Uses the per-horizon TPE-tuned config when
    ``horizon_min`` has one (30/60); otherwise the regularized default. ``cfg``
    overrides (huber objective, monotone therapy constraints) are layered on top."""
    params = _TUNED_PARAMS.get(horizon_min, _DEFAULT_PARAMS)
    reg = LGBMRegressor(**_BASE_PARAMS, **params)
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
        model = _make_regressor(cfg, feature_cols, h * cfg.grid_minutes)
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
