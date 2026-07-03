"""Walk-forward OOF absolute-error CDF: every point predicted from its PAST.

The methodologically-sound realisation of "evaluate every point out-of-sample"
for an autocorrelated time series (literal LOO leaks via temporal neighbours).
Expanding, embargoed folds (embargo = max horizon) tile the tail 60% of the data;
each fold trains tuned LGBM and best k-NN (eucl+tricube) on its past and predicts
its window. Errors are pooled across folds -> CDF over ~15k out-of-sample points.
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
N_FOLDS, TEST_SPAN = 10, 0.6
KNN_K = {30: 300, 60: 800}
COLORS = {"persistence": "#7f7f7f", "LGBM (tuned)": "#0072B2", "k-NN (eucl+tricube)": "#D55E00"}
STYLES = {"persistence": (0, (5, 2)), "LGBM (tuned)": "-", "k-NN (eucl+tricube)": "-"}

cfg = load_config()
sup, cols = build_supervised(cfg, align(load_raw(cfg), cfg))
sup = sup.sort_values(["user_id", "ts_utc"]).reset_index(drop=True)
folds = walk_forward_masks(sup, N_FOLDS, TEST_SPAN, embargo=max(cfg.horizons_steps))
print(f"all data: {len(sup)} rows; {N_FOLDS} expanding folds over tail {TEST_SPAN:.0%}")


def knn_delta(train, test, h, k):
    mtr = train[f"valid_{h}"].to_numpy()
    Xtr = train.loc[mtr, cols]
    ytr = (train.loc[mtr, f"y_{h}"] - train.loc[mtr, "glucose_mgdl"]).to_numpy()
    im = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(im.transform(Xtr))
    tf = lambda X: sc.transform(im.transform(X))
    kk = min(k, len(Xtr) - 1)
    nn = NearestNeighbors(n_neighbors=kk, metric="minkowski", p=2).fit(tf(Xtr))
    dist, idx = nn.kneighbors(tf(test[cols]))
    u = dist / np.maximum(dist[:, -1:], EPS)
    w = np.clip(1 - u**3, 0, None) ** 3
    return (w * ytr[idx]).sum(1) / np.maximum(w.sum(1), EPS)


# collect pooled OOF (actual, pred) per horizon
oof = {h: {m: [] for m in ["y", "persistence", "LGBM (tuned)", "k-NN (eucl+tricube)"]}
       for h in cfg.horizons_steps}
for i, (tr_idx, te_idx) in enumerate(folds):
    train, test = sup.loc[tr_idx], sup.loc[te_idx]
    lgbm = train_lgbm(train, cols, cfg)
    lp = make_pred_fn(lgbm, cols, cfg.features.predict_delta)
    for h in cfg.horizons_steps:
        m = test[f"valid_{h}"].to_numpy()
        if m.sum() == 0:
            continue
        g = test["glucose_mgdl"].to_numpy()
        oof[h]["y"].append(test[f"y_{h}"].to_numpy()[m])
        oof[h]["persistence"].append(persistence_predict(test, h)[m])
        oof[h]["LGBM (tuned)"].append(lp(test, h)[m])
        oof[h]["k-NN (eucl+tricube)"].append(
            (knn_delta(train, test, h, KNN_K[h * cfg.grid_minutes]) + g)[m]
        )
    print(f"  fold {i}: train={len(train)} test={len(test)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
summary = []
for ax, h in zip(axes, cfg.horizons_steps):
    hm = h * cfg.grid_minutes
    yt = np.concatenate(oof[h]["y"])
    persist = np.concatenate(oof[h]["persistence"])
    for name in ["persistence", "LGBM (tuned)", "k-NN (eucl+tricube)"]:
        yp = np.concatenate(oof[h][name])
        ae = np.sort(np.abs(yt - yp))
        cdf = np.arange(1, len(ae) + 1) / len(ae)
        ax.plot(ae, cdf, color=COLORS[name], lw=2, linestyle=STYLES[name], label=name)
        sk = skill_score(rmse(yt, yp), rmse(yt, persist))
        summary.append((hm, name, float(np.median(np.abs(yt - yp))),
                        float(np.percentile(np.abs(yt - yp), 90)), sk))
    ax.set_xlim(0, float(np.percentile(np.abs(yt - persist), 98)))
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#d9d9d9", lw=0.8, zorder=0)
    ax.set_title(f"{hm}-min horizon  (n={len(yt)} OOF points)", fontsize=11)
    ax.set_xlabel("absolute error  |pred − actual|  (mg/dL)")
    ax.grid(True, alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("cumulative fraction of predictions ≤ x")
axes[0].legend(loc="lower right", frameon=False, fontsize=10)
fig.suptitle("Absolute-error CDF — walk-forward out-of-sample over all data (up & left = better)",
             fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = diagram_path(cfg.paths.reports_dir, "cdf", "walk_forward_oof", "lgbm-tuned_knn-tricube")
fig.savefig(out, dpi=130)
print(f"\n-> {out}")

print(f"\n{'horizon':>7} {'model':<22} {'median AE':>10} {'P90 AE':>8} {'skill':>8}")
for hm, name, med, p90, sk in summary:
    print(f"{hm:>7} {name:<22} {med:>10.2f} {p90:>8.2f} {sk:>+8.4f}")
