"""Personalization (ROADMAP §Phase 4) — the moat + retention story.

Two mechanisms, both leakage-safe (user-level stats come from TRAIN only):

- **4a global-conditioned:** one global LightGBM given per-user static features
  (mean glucose, variability, typical rise). One model, no cold-start.
- **4b residual-stacking:** the conditioned base + a tiny per-user residual model
  that learns only the delta. A brand-new user has no residual model → residual
  is exactly 0 → predictions fall back to the global base (graceful cold-start,
  gradual generic→personal transition as data accumulates).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ..config import Config
from .lgbm import _make_regressor

# Enough per-user history before a residual model is worth fitting. The product
# trigger is ~2-4 weeks (§Phase 4); here we require a modest floor so the
# mechanism engages in eval while still degrading gracefully below it.
MIN_RESIDUAL_ROWS = 200

STATIC_COLS = ["user_mean_glucose", "user_std_glucose", "user_typ_rise"]


def _small_regressor() -> LGBMRegressor:
    """A deliberately tiny model — it only learns the per-user residual."""
    return LGBMRegressor(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=80,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
    )


def user_static_features(train: pd.DataFrame) -> pd.DataFrame:
    """Per-user summary stats from TRAIN only (no leakage): the 4a conditioning.

    Per-user therapy settings (ISF/ICR) are pure identity attributes, so this is
    their proper home — conditioning the personalized model, not the shared base.
    """
    rising = train["rate_short"].where(train["rate_short"] > 0)
    g = train.groupby("user_id")
    cols = {
        "user_mean_glucose": g["glucose_mgdl"].mean(),
        "user_std_glucose": g["glucose_mgdl"].std(),
        "user_typ_rise": rising.groupby(train["user_id"]).mean(),
    }
    for therapy in ("isf", "icr"):
        if therapy in train.columns:
            cols[f"user_{therapy}"] = g[
                therapy
            ].first()  # distinct name: grid carries isf/icr too
    return pd.DataFrame(cols).reset_index()


def static_cols(static: pd.DataFrame) -> list[str]:
    """The conditioning feature columns present in a static table."""
    return [c for c in static.columns if c != "user_id"]


def add_static(df: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Left-merge per-user static features (NaN for unseen/cold-start users)."""
    return df.merge(static, on="user_id", how="left")


def train_per_horizon(
    train: pd.DataFrame, cols: list[str], cfg: Config
) -> dict[int, LGBMRegressor]:
    delta = cfg.features.predict_delta
    models: dict[int, LGBMRegressor] = {}
    for h in cfg.horizons_steps:
        mask = train[f"valid_{h}"].to_numpy()
        y = train.loc[mask, f"y_{h}"]
        if delta:
            y = y - train.loc[mask, "glucose_mgdl"]
        models[h] = _make_regressor().fit(train.loc[mask, cols], y)
    return models


class PersonalizedModel:
    """Conditioned global base (4a) + per-user residual models (4b)."""

    def __init__(
        self,
        cond_models,
        residual_models,
        feature_cols,
        static_cols,
        static_table,
        predict_delta: bool = True,
    ):
        self.cond_models = cond_models
        self.residual_models = residual_models  # {h: {user_id: model}}
        self.feature_cols = feature_cols
        self.static_cols = static_cols
        self.static_table = static_table
        self.predict_delta = predict_delta

    def predict(self, df: pd.DataFrame, horizon_steps: int) -> np.ndarray:
        d = df.reset_index(drop=True)
        merged = d.merge(self.static_table, on="user_id", how="left")  # preserves order
        base = (
            self.cond_models[horizon_steps]
            .predict(merged[self.feature_cols + self.static_cols])
            .astype(float)
        )
        if self.predict_delta:
            base = base + d["glucose_mgdl"].to_numpy()  # delta → absolute
        # add the per-user (absolute) residual where a model exists; cold-start → +0
        for uid, rmodel in self.residual_models.get(horizon_steps, {}).items():
            m = (d["user_id"] == uid).to_numpy()
            if m.any():
                base[m] = base[m] + rmodel.predict(d.loc[m, self.feature_cols])
        return base


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def train_residuals(train_c, cond_models, feature_cols, static_cols, cfg) -> dict:
    """Fit a tiny residual model per (horizon, user), keeping it only if it helps.

    Each user's residual is validated on a held-out chronological slice of their
    own training data; if it doesn't beat the base there, it is dropped (that user
    stays global). This makes personalization non-harmful by construction — a
    residual is never applied unless it demonstrably improves that user.
    """
    delta = cfg.features.predict_delta
    residual_models: dict[int, dict] = {}
    for h in cfg.horizons_steps:
        base = cond_models[h].predict(train_c[feature_cols + static_cols]).astype(float)
        if delta:
            base = base + train_c["glucose_mgdl"].to_numpy()  # absolute base
        tc = train_c.assign(_base=base, _resid=train_c[f"y_{h}"] - base)
        residual_models[h] = {}
        for uid, gdf in tc.groupby("user_id"):
            g = gdf[gdf[f"valid_{h}"].to_numpy()]
            if len(g) < MIN_RESIDUAL_ROWS:
                continue  # not enough history yet → stay global (graceful)
            cut = int(len(g) * 0.8)
            fit, val = g.iloc[:cut], g.iloc[cut:]
            if len(val) < 30:
                continue
            trial = _small_regressor().fit(fit[feature_cols], fit["_resid"])
            yv, basev = val[f"y_{h}"].to_numpy(), val["_base"].to_numpy()
            if _rmse(yv, basev + trial.predict(val[feature_cols])) < _rmse(yv, basev):
                # helps on the user's own holdout → refit on all their data
                residual_models[h][uid] = _small_regressor().fit(
                    g[feature_cols], g["_resid"]
                )
    return residual_models
