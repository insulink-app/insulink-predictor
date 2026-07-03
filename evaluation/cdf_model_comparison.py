"""Absolute-error CDF: tuned LGBM vs best k-NN vs persistence on ALL DB data.

Full dataset, chronological 80/20 split; both learners trained on the train
portion, errors measured out-of-sample on the test tail. A CDF that rises higher
and further left = more predictions with small error = better.
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
from insulink_predictor.eval.harness import build_supervised
from insulink_predictor.eval.split import chronological_split
from insulink_predictor.models.baseline import persistence_predict
from insulink_predictor.models.lgbm import make_pred_fn, train_lgbm

EPS = 1e-9
KNN_K = {30: 300, 60: 800}  # tuned tricube neighborhood per horizon

# Okabe-Ito colorblind-safe: gray reference, blue, vermillion.
COLORS = {"persistence": "#7f7f7f", "LGBM (tuned)": "#0072B2", "k-NN (eucl+tricube)": "#D55E00"}
STYLES = {"persistence": (0, (5, 2)), "LGBM (tuned)": "-", "k-NN (eucl+tricube)": "-"}

cfg = load_config()
sup, cols = build_supervised(cfg, align(load_raw(cfg), cfg))
train, test = chronological_split(sup, cfg.split.test_fraction)
print(f"all data: {len(sup)} rows, {sup['user_id'].nunique()} user(s); "
      f"train={len(train)} test={len(test)}")

lgbm = train_lgbm(train, cols, cfg)
lgbm_pred = make_pred_fn(lgbm, cols, cfg.features.predict_delta)


def knn_predict(h, k):
    mtr = train[f"valid_{h}"].to_numpy()
    Xtr = train.loc[mtr, cols]
    ytr = (train.loc[mtr, f"y_{h}"] - train.loc[mtr, "glucose_mgdl"]).to_numpy()
    im = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(im.transform(Xtr))
    trans = lambda X: sc.transform(im.transform(X))
    nn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(trans(Xtr))
    dist, idx = nn.kneighbors(trans(test[cols]))
    u = dist / np.maximum(dist[:, -1:], EPS)
    w = np.clip(1 - u**3, 0, None) ** 3  # tricube
    delta = (w * ytr[idx]).sum(1) / np.maximum(w.sum(1), EPS)
    return delta + test["glucose_mgdl"].to_numpy()


fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
summary = []
for ax, h in zip(axes, cfg.horizons_steps):
    hm = h * cfg.grid_minutes
    m = test[f"valid_{h}"].to_numpy()
    yt = test.loc[m, f"y_{h}"].to_numpy()
    preds = {
        "persistence": persistence_predict(test, h)[m],
        "LGBM (tuned)": lgbm_pred(test, h)[m],
        "k-NN (eucl+tricube)": knn_predict(h, KNN_K[hm])[m],
    }
    for name, yp in preds.items():
        ae = np.abs(yt - yp)
        ae = np.sort(ae[np.isfinite(ae)])
        cdf = np.arange(1, len(ae) + 1) / len(ae)
        ax.plot(ae, cdf, color=COLORS[name], lw=2, linestyle=STYLES[name], label=name)
        summary.append((hm, name, float(np.median(ae)), float(np.percentile(ae, 90))))
    xcap = float(np.percentile(np.abs(yt - preds["persistence"]), 98))
    ax.set_xlim(0, xcap)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#d9d9d9", lw=0.8, zorder=0)  # median reference
    ax.set_title(f"{hm}-min horizon  (n={int(m.sum())})", fontsize=11)
    ax.set_xlabel("absolute error  |pred − actual|  (mg/dL)")
    ax.grid(True, alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("cumulative fraction of predictions ≤ x")
axes[0].legend(loc="lower right", frameon=False, fontsize=10)
fig.suptitle("Absolute-error CDF — out-of-sample test on full DB data  (up & left = better)",
             fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = diagram_path(cfg.paths.reports_dir, "cdf", "model_comparison", "lgbm-tuned_knn-tricube")
fig.savefig(out, dpi=130)
print(f"\n-> {out}")

print(f"\n{'horizon':>7} {'model':<22} {'median AE':>10} {'P90 AE':>8}")
for hm, name, med, p90 in summary:
    print(f"{hm:>7} {name:<22} {med:>10.2f} {p90:>8.2f}")
