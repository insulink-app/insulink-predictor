"""Tune LGBM and k-NN to the loaded dataset before evaluating.

Both tuners select on a chronological **validation** slice carved from the given
training data (never the test set), so the CDF scripts can re-tune per dataset
without leaking. Returns per-horizon settings keyed by horizon-minutes.

- LGBM  : random search over a wide space, early-stopped on val (no optuna dep).
- k-NN  : best tricube neighborhood ``k`` (euclidean metric + tricube kernel is the
          established best recipe; ``k`` is the parameter that moves with the data).

Then ``build_lgbm_pred_fn`` / ``build_knn_pred_fn`` train on a (possibly larger)
training set with those settings and return a ``pred_fn(df, h) -> abs mg/dL``.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from insulink_predictor.eval.metrics import rmse, skill_score
from insulink_predictor.eval.split import chronological_split
from insulink_predictor.models.lgbm import _BASE_PARAMS

EPS = 1e-9
KNN_GRID = (50, 100, 200, 300, 500, 800, 1200)
LGBM_TRIALS = 40
# deterministic base for FINAL models; a faster variant for tuning trials.
_FAST = {**_BASE_PARAMS, "n_jobs": -1, "deterministic": False}


def _delta_xy(d, cols, h):
    """Feature matrix + delta-over-persistence target on the valid rows."""
    m = d[f"valid_{h}"].to_numpy()
    y = (d.loc[m, f"y_{h}"] - d.loc[m, "glucose_mgdl"]).to_numpy()
    return d.loc[m, cols], y


def _logint(rng, lo, hi):
    return int(round(float(np.exp(rng.uniform(np.log(lo), np.log(hi))))))


def _logunif(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


# --- LGBM --------------------------------------------------------------------
def tune_lgbm(train, cols, cfg, n_trials=LGBM_TRIALS, seed=42, verbose=True):
    """Per-horizon LGBM params via random search, chosen on a val slice."""
    rng = np.random.default_rng(seed)
    ptr, val = chronological_split(train, 0.25)
    out = {}
    for h in cfg.horizons_steps:
        hm = h * cfg.grid_minutes
        Xtr, ytr = _delta_xy(ptr, cols, h)
        Xv, yv = _delta_xy(val, cols, h)
        ref = rmse(yv, np.zeros_like(yv))
        best = (-9.9, None)
        for _ in range(n_trials):
            p = dict(
                n_estimators=3000,
                num_leaves=_logint(rng, 8, 256),
                learning_rate=_logunif(rng, 0.005, 0.2),
                min_child_samples=_logint(rng, 5, 300),
                reg_lambda=_logunif(rng, 1e-3, 100),
                reg_alpha=_logunif(rng, 1e-3, 50),
                subsample=float(rng.uniform(0.5, 1.0)),
                colsample_bytree=float(rng.uniform(0.4, 1.0)),
                max_depth=int(rng.integers(3, 17)),
                min_split_gain=float(rng.uniform(0.0, 0.5)),
            )
            m = LGBMRegressor(**_FAST, **p)
            m.fit(
                Xtr,
                ytr,
                eval_set=[(Xv, yv)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            sv = skill_score(rmse(yv, m.predict(Xv)), ref)
            if sv > best[0]:
                bp = dict(p)
                bp["n_estimators"] = int(m.best_iteration_ or p["n_estimators"])
                best = (sv, bp)
        out[hm] = best[1]
        if verbose:
            print(
                f"  tuned LGBM @{hm}min: val-skill={best[0]:+.4f} "
                f"(leaves={best[1]['num_leaves']}, lr={best[1]['learning_rate']:.4f}, "
                f"n_iter={best[1]['n_estimators']})"
            )
    return out


def build_lgbm_pred_fn(train, cols, cfg, params_by_hm):
    """Train per-horizon LGBM with the tuned params; return pred_fn(df, h)."""
    delta = cfg.features.predict_delta
    models = {}
    for h in cfg.horizons_steps:
        hm = h * cfg.grid_minutes
        X, y = _delta_xy(train, cols, h)
        models[h] = LGBMRegressor(**_BASE_PARAMS, **params_by_hm[hm]).fit(X, y)

    def pred_fn(df, h):
        p = models[h].predict(df[cols])
        return p + df["glucose_mgdl"].to_numpy() if delta else p

    return pred_fn


# --- k-NN (euclidean + tricube; tune k) --------------------------------------
def tune_knn(train, cols, cfg, ks=KNN_GRID, verbose=True):
    """Per-horizon best tricube neighborhood ``k``, chosen on a val slice."""
    ptr, val = chronological_split(train, 0.25)
    out = {}
    for h in cfg.horizons_steps:
        hm = h * cfg.grid_minutes
        Xtr, ytr = _delta_xy(ptr, cols, h)
        Xv, yv = _delta_xy(val, cols, h)
        im = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
        sc = StandardScaler().fit(im.transform(Xtr))
        kmax = min(max(ks), len(Xtr) - 1)
        nn = NearestNeighbors(n_neighbors=kmax).fit(sc.transform(im.transform(Xtr)))
        dist, idx = nn.kneighbors(sc.transform(im.transform(Xv)))
        ref = rmse(yv, np.zeros_like(yv))
        best = (-9.9, ks[0])
        for k in ks:
            if k > kmax:
                continue
            d, yy = dist[:, :k], ytr[idx[:, :k]]
            u = d / np.maximum(d[:, -1:], EPS)
            w = np.clip(1 - u**3, 0, None) ** 3
            pred = (w * yy).sum(1) / np.maximum(w.sum(1), EPS)
            sv = skill_score(rmse(yv, pred), ref)
            if sv > best[0]:
                best = (sv, k)
        out[hm] = best[1]
        if verbose:
            print(f"  tuned k-NN @{hm}min: val-skill={best[0]:+.4f} (k={best[1]})")
    return out


def build_knn_pred_fn(train, cols, cfg, ks_by_hm):
    """Fit per-horizon tricube k-NN with the tuned k; return pred_fn(df, h)."""
    delta = cfg.features.predict_delta
    fitted = {}
    for h in cfg.horizons_steps:
        hm = h * cfg.grid_minutes
        Xtr, ytr = _delta_xy(train, cols, h)
        im = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
        sc = StandardScaler().fit(im.transform(Xtr))
        k = min(ks_by_hm[hm], len(Xtr) - 1)
        nn = NearestNeighbors(n_neighbors=k).fit(sc.transform(im.transform(Xtr)))
        fitted[h] = (im, sc, nn, ytr)

    def pred_fn(df, h):
        im, sc, nn, ytr = fitted[h]
        dist, idx = nn.kneighbors(sc.transform(im.transform(df[cols])))
        u = dist / np.maximum(dist[:, -1:], EPS)
        w = np.clip(1 - u**3, 0, None) ** 3
        d = (w * ytr[idx]).sum(1) / np.maximum(w.sum(1), EPS)
        return d + df["glucose_mgdl"].to_numpy() if delta else d

    return pred_fn
