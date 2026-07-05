"""Auto-tune LGBM per horizon to the training data.

Random search over a wide space, each trial early-stopped on a chronological
validation slice carved from ``train`` (leakage-free). Returns the best
hyperparameters per horizon (minutes). Used by ``train_lgbm`` when
``cfg.model.auto_tune`` is set, so the model re-tunes to whatever data is loaded
instead of carrying the frozen ``_TUNED_PARAMS`` constants.

Kept import-independent of ``lgbm`` (which imports this) to avoid a cycle: trials
use a self-contained fast base; ``train_lgbm`` layers ``_BASE_PARAMS`` + cfg
overrides onto the returned params for the final deterministic models.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor

from ..config import Config
from ..eval.metrics import rmse, skill_score
from ..eval.split import chronological_split

_SEED = 42
# Fast base for tuning trials (n_jobs=-1; bit-determinism not needed to *rank*).
_TRIAL_BASE = dict(
    subsample_freq=1, random_state=_SEED, n_jobs=-1, force_col_wise=True, verbose=-1
)


def _sample(rng) -> dict:
    def lg(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    return dict(
        n_estimators=3000,  # capped by early stopping
        num_leaves=int(round(lg(8, 256))),
        learning_rate=lg(0.005, 0.2),
        min_child_samples=int(round(lg(5, 300))),
        reg_lambda=lg(1e-3, 100),
        reg_alpha=lg(1e-3, 50),
        subsample=float(rng.uniform(0.5, 1.0)),
        colsample_bytree=float(rng.uniform(0.4, 1.0)),
        max_depth=int(rng.integers(3, 17)),
        min_split_gain=float(rng.uniform(0.0, 0.5)),
    )


def _xy(d, cols, h):
    m = d[f"valid_{h}"].to_numpy()
    return d.loc[m, cols], (d.loc[m, f"y_{h}"] - d.loc[m, "glucose_mgdl"]).to_numpy()


def tune_lgbm(
    train,
    feature_cols: list[str],
    cfg: Config,
    seed: int = _SEED,
    verbose: bool = False,
) -> dict[int, dict | None]:
    """Best LGBM params per horizon-min, chosen on a val slice of ``train``.

    Returns ``{horizon_min: params}``; a horizon maps to ``None`` if it has too
    little valid data to tune (caller falls back to the frozen params).
    """
    rng = np.random.default_rng(seed)
    ptr, val = chronological_split(train, cfg.model.tune_val_fraction)
    out: dict[int, dict | None] = {}
    for h in cfg.horizons_steps:
        hm = h * cfg.grid_minutes
        Xtr, ytr = _xy(ptr, feature_cols, h)
        Xv, yv = _xy(val, feature_cols, h)
        if len(yv) < 50 or len(ytr) < 50:  # not enough to tune -> frozen fallback
            out[hm] = None
            continue
        ref = rmse(yv, np.zeros_like(yv))
        best_sk, best_p = -9.9, None
        for _ in range(cfg.model.tune_trials):
            p = _sample(rng)
            m = LGBMRegressor(**_TRIAL_BASE, **p)
            m.fit(
                Xtr,
                ytr,
                eval_set=[(Xv, yv)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            sk = skill_score(rmse(yv, m.predict(Xv)), ref)
            if sk > best_sk:
                p["n_estimators"] = int(m.best_iteration_ or p["n_estimators"])
                best_sk, best_p = sk, p
        out[hm] = best_p
        if verbose and best_p is not None:
            print(
                f"  tuned LGBM @{hm}min: val-skill={best_sk:+.4f} "
                f"(leaves={best_p['num_leaves']}, lr={best_p['learning_rate']:.4f}, "
                f"n_iter={best_p['n_estimators']})"
            )
    return out
