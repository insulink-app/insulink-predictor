"""Post-meal absolute-error CDF (walk-forward OOF): where the tail actually is.

Restricts the pooled out-of-sample predictions to rows <=60 min after a logged
meal and plots the error CDF for persistence / tuned LGBM / best k-NN, per
horizon. Also reports post-meal skill vs persistence (persistence denominator
recomputed on the post-meal subset, so it is a fair post-meal comparison).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from insulink_predictor.config import load_config
from insulink_predictor.eval.outputs import diagram_path
from insulink_predictor.data.align import align
from insulink_predictor.data.load import load_raw
from insulink_predictor.eval.backtest import walk_forward_masks
from insulink_predictor.eval.harness import build_supervised
from insulink_predictor.eval.metrics import rmse, skill_score
from insulink_predictor.models.baseline import persistence_predict
from insulink_predictor.models.lgbm import make_pred_fn, train_lgbm

EPS = 1e-9
N_FOLDS, TEST_SPAN, POST_MIN = 10, 0.6, 60
KNN_K = {30: 300, 60: 800}
MODELS = ["persistence", "LGBM (tuned)", "k-NN (eucl+tricube)"]
COLORS = {"persistence": "#7f7f7f", "LGBM (tuned)": "#0072B2", "k-NN (eucl+tricube)": "#D55E00"}
STYLES = {"persistence": (0, (5, 2)), "LGBM (tuned)": "-", "k-NN (eucl+tricube)": "-"}

cfg = load_config()
sup, cols = build_supervised(cfg, align(load_raw(cfg), cfg))
sup = sup.sort_values(["user_id", "ts_utc"]).reset_index(drop=True)
folds = walk_forward_masks(sup, N_FOLDS, TEST_SPAN, embargo=max(cfg.horizons_steps))


def knn_delta(train, test, h, k):
    mtr = train[f"valid_{h}"].to_numpy()
    Xtr = train.loc[mtr, cols]
    ytr = (train.loc[mtr, f"y_{h}"] - train.loc[mtr, "glucose_mgdl"]).to_numpy()
    im = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(im.transform(Xtr))
    tf = lambda X: sc.transform(im.transform(X))
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr) - 1), metric="minkowski", p=2).fit(tf(Xtr))
    dist, idx = nn.kneighbors(tf(test[cols]))
    u = dist / np.maximum(dist[:, -1:], EPS)
    w = np.clip(1 - u**3, 0, None) ** 3
    return (w * ytr[idx]).sum(1) / np.maximum(w.sum(1), EPS)


keys = ["y", "tsm"] + MODELS
oof = {h: {k: [] for k in keys} for h in cfg.horizons_steps}
for tr_idx, te_idx in folds:
    train, test = sup.loc[tr_idx], sup.loc[te_idx]
    lp = make_pred_fn(train_lgbm(train, cols, cfg), cols, cfg.features.predict_delta)
    for h in cfg.horizons_steps:
        m = test[f"valid_{h}"].to_numpy()
        if m.sum() == 0:
            continue
        g = test["glucose_mgdl"].to_numpy()
        oof[h]["y"].append(test[f"y_{h}"].to_numpy()[m])
        oof[h]["tsm"].append(test["time_since_meal"].to_numpy()[m])
        oof[h]["persistence"].append(persistence_predict(test, h)[m])
        oof[h]["LGBM (tuned)"].append(lp(test, h)[m])
        oof[h]["k-NN (eucl+tricube)"].append((knn_delta(train, test, h, KNN_K[h * cfg.grid_minutes]) + g)[m])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
rows = []
for ax, h in zip(axes, cfg.horizons_steps):
    hm = h * cfg.grid_minutes
    yt = np.concatenate(oof[h]["y"])
    tsm = np.concatenate(oof[h]["tsm"])
    post = np.isfinite(tsm) & (tsm <= POST_MIN)
    yt_p = yt[post]
    persist_p = np.concatenate(oof[h]["persistence"])[post]
    for name in MODELS:
        yp = np.concatenate(oof[h][name])[post]
        ae = np.sort(np.abs(yt_p - yp))
        cdf = np.arange(1, len(ae) + 1) / len(ae)
        ax.plot(ae, cdf, color=COLORS[name], lw=2, linestyle=STYLES[name], label=name)
        sk = skill_score(rmse(yt_p, yp), rmse(yt_p, persist_p))
        rows.append((hm, name, int(post.sum()), float(np.median(np.abs(yt_p - yp))),
                     float(np.percentile(np.abs(yt_p - yp), 90)), sk))
    ax.set_xlim(0, float(np.percentile(np.abs(yt_p - persist_p), 98)))
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#d9d9d9", lw=0.8, zorder=0)
    ax.set_title(f"{hm}-min horizon  (n={int(post.sum())} post-meal points)", fontsize=11)
    ax.set_xlabel("absolute error  |pred − actual|  (mg/dL)")
    ax.grid(True, alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("cumulative fraction of predictions ≤ x")
axes[0].legend(loc="lower right", frameon=False, fontsize=10)
fig.suptitle(f"Post-meal absolute-error CDF (≤{POST_MIN} min after a logged meal) — walk-forward OOF",
             fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = diagram_path(cfg.paths.reports_dir, "cdf", "post_meal", "lgbm-tuned_knn-tricube")
fig.savefig(out, dpi=130)
print(f"-> {out}\n")
print(f"{'horizon':>7} {'model':<22} {'n':>6} {'median':>7} {'P90':>7} {'skill':>8}")
for hm, name, n, med, p90, sk in rows:
    print(f"{hm:>7} {name:<22} {n:>6} {med:>7.1f} {p90:>7.1f} {sk:>+8.4f}")
