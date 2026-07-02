"""Walk-forward backtesting — trustworthy skill with error bars.

A single 80/20 chronological split (``chronological_split``) yields one skill
number whose noise — across users and across the particular cut point — is
invisible. This module runs **rolling-origin, expanding-window** evaluation:
several chronological folds, each training on all data *before* its test window
and testing on the next contiguous window. Reporting **mean ± std across folds**
turns "skill = 0.17" into "skill = 0.17 ± 0.02", so a modeling change is only
believed when it moves the mean by more than the fold-to-fold noise.

Two guardrails keep it honest:

- **Per user, always chronological** (§6). Folds tile each user's own timeline;
  train always precedes test within a user.
- **Embargo.** ``max(horizons_steps)`` rows are dropped between each train window
  and its test window (per user), so no training *target* (which looks ``h`` steps
  ahead) reaches into the test window's feature span. The single-split harness
  omits this; here it matters because folds abut.

Models are compared on **identical folds and the identical persistence
denominator**, so the per-fold skill difference between two models is *paired* —
far less noisy than comparing two independent mean±std summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..config import Config
from .metrics import rmse, skill_score

# fit_predict(train_df, feature_cols, cfg) -> pred_fn(test_df, horizon_steps) -> np.ndarray
FitPredict = Callable[[pd.DataFrame, list, Config], Callable]


def walk_forward_masks(
    df: pd.DataFrame, n_folds: int, test_span: float, embargo: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Boolean (train_mask, test_mask) per fold over ``df`` rows (per-user aware).

    The last ``test_span`` fraction of each user's timeline is tiled into
    ``n_folds`` contiguous test windows. Fold ``i`` trains on everything up to
    ``embargo`` rows before its test window starts (expanding window).
    """
    if not 0.0 < test_span < 1.0:
        raise ValueError("test_span must be in (0, 1)")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    df = df.sort_values(["user_id", "ts_utc"])
    label = df.index.to_numpy()  # index labels back into the caller's frame
    pos = df.groupby("user_id", sort=True).cumcount().to_numpy()
    n = df.groupby("user_id", sort=True)["user_id"].transform("size").to_numpy()

    folds = []
    train_start_frac = 1.0 - test_span
    step = test_span / n_folds
    for i in range(n_folds):
        lo = np.round((train_start_frac + i * step) * n).astype(int)
        hi = np.round((train_start_frac + (i + 1) * step) * n).astype(int)
        test = (pos >= lo) & (pos < hi)
        train = pos < (lo - embargo)
        folds.append((label[train], label[test]))
    return folds


def backtest_models(
    cfg: Config,
    models: dict[str, FitPredict],
    df: pd.DataFrame | None = None,
    sup: pd.DataFrame | None = None,
    feature_cols: list | None = None,
    n_folds: int = 5,
    test_span: float = 0.5,
) -> pd.DataFrame:
    """Per-fold skill for each named model on identical folds.

    Returns a tidy frame: ``model, fold, horizon_min, n, rmse, rmse_persistence,
    skill``. Every model in ``models`` sees the same train/test split and is scored
    against the same per-fold persistence RMSE (paired comparison).
    """
    from ..models.baseline import persistence_predict

    if sup is None:
        from .harness import build_supervised

        sup, feature_cols = build_supervised(cfg, df)
    sup = sup.sort_values(["user_id", "ts_utc"]).reset_index(drop=True)

    embargo = max(cfg.horizons_steps)
    folds = walk_forward_masks(sup, n_folds, test_span, embargo)

    rows = []
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        train = sup.loc[tr_idx]
        test = sup.loc[te_idx]
        # persistence denominator, computed once per fold and shared by all models
        ref = {}
        for h in cfg.horizons_steps:
            m = test[f"valid_{h}"].to_numpy()
            if m.sum() == 0:
                continue
            yt = test[f"y_{h}"].to_numpy()[m]
            ref[h] = rmse(yt, persistence_predict(test, h)[m])

        for name, fit_predict in models.items():
            pred_fn = fit_predict(train, feature_cols, cfg)
            for h in cfg.horizons_steps:
                mask = test[f"valid_{h}"].to_numpy()
                if mask.sum() == 0 or h not in ref:
                    continue
                yt = test[f"y_{h}"].to_numpy()[mask]
                yp = np.asarray(pred_fn(test, h))[mask]
                r = rmse(yt, yp)
                rows.append(
                    {
                        "model": name,
                        "fold": fold_i,
                        "horizon_min": h * cfg.grid_minutes,
                        "n": int(mask.sum()),
                        "rmse": round(r, 3),
                        "rmse_persistence": round(ref[h], 3),
                        "skill": round(skill_score(r, ref[h]), 4),
                    }
                )
    return pd.DataFrame(rows)


def summarize(perfold: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std skill across folds, with the worst fold, per (model, horizon)."""
    g = perfold.groupby(["model", "horizon_min"])["skill"]
    out = g.agg(mean="mean", std="std", worst="min", folds="count").reset_index()
    out["mean"] = out["mean"].round(4)
    out["std"] = out["std"].round(4)
    out["worst"] = out["worst"].round(4)
    return out


def run_backtest(
    cfg: Config,
    df: pd.DataFrame | None = None,
    n_folds: int = 5,
    test_span: float = 0.5,
    write: bool = True,
) -> dict:
    """Walk-forward skill for the committed LightGBM model — the trustworthy number.

    Trains the direct multi-horizon model on each expanding fold and reports skill
    mean ± std (and worst fold) vs persistence. Written to ``reports/backtest_*``.
    """
    from ..models.lgbm import make_pred_fn, train_lgbm
    from .harness import build_supervised
    from .reporting import mlflow_run, write_table

    sup, feature_cols = build_supervised(cfg, df)

    def lgbm(train, cols, _cfg):
        return make_pred_fn(
            train_lgbm(train, cols, cfg), cols, cfg.features.predict_delta
        )

    perfold = backtest_models(
        cfg,
        {"lgbm": lgbm},
        sup=sup,
        feature_cols=feature_cols,
        n_folds=n_folds,
        test_span=test_span,
    )
    summary = summarize(perfold)

    if write:
        reports = Path(cfg.paths.reports_dir)
        write_table(perfold, reports / "backtest_per_fold.csv")
        write_table(summary, reports / "backtest_summary.csv")
        with mlflow_run(cfg, "backtest") as log:
            log.params({"model": "lgbm", "n_folds": n_folds, "test_span": test_span})
            for _, r in summary.iterrows():
                hm = int(r["horizon_min"])
                log.metrics(
                    {f"skill_mean_h{hm}": r["mean"], f"skill_std_h{hm}": r["std"]}
                )

    return {"per_fold": perfold, "summary": summary, "feature_cols": feature_cols}


def paired_delta(perfold: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Per-fold paired skill lift of each model over ``baseline``.

    Because both models ran on the identical folds, the per-fold difference cancels
    the shared fold-to-fold noise — the mean lift is significant when it clears its
    own (small) std. Reports mean lift, its std, and how many folds improved.
    """
    base = (
        perfold[perfold["model"] == baseline]
        .set_index(["fold", "horizon_min"])["skill"]
        .rename("base")
    )
    rows = []
    for name, gdf in perfold.groupby("model"):
        if name == baseline:
            continue
        j = gdf.set_index(["fold", "horizon_min"])["skill"].to_frame("var").join(base)
        j = j.dropna()
        for h_min, hg in j.groupby(level="horizon_min"):
            d = (hg["var"] - hg["base"]).to_numpy()
            rows.append(
                {
                    "model": name,
                    "vs": baseline,
                    "horizon_min": h_min,
                    "mean_lift": round(float(d.mean()), 4),
                    "lift_std": round(float(d.std()), 4),
                    "folds_improved": int((d > 0).sum()),
                    "n_folds": int(d.size),
                }
            )
    return pd.DataFrame(rows)
