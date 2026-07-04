"""Absolute-error CDF: LGBM vs k-NN vs persistence on ALL DB data.

Both learners are **re-tuned to the loaded dataset** (LGBM params + k-NN k, chosen
on a validation slice of the training data) before anything is computed, then
trained on the full training split and scored out-of-sample on the test tail.
A CDF that rises higher and further left = more predictions with small error.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from _tuning import build_knn_pred_fn, build_lgbm_pred_fn, tune_knn, tune_lgbm

from insulink_predictor.config import load_config
from insulink_predictor.data.align import align
from insulink_predictor.data.load import load_raw
from insulink_predictor.eval.harness import build_supervised
from insulink_predictor.eval.outputs import diagram_path
from insulink_predictor.eval.split import chronological_split
from insulink_predictor.models.baseline import persistence_predict

# Okabe-Ito colorblind-safe: gray reference, blue, vermillion.
COLORS = {
    "persistence": "#7f7f7f",
    "LGBM (tuned)": "#0072B2",
    "k-NN (tuned)": "#D55E00",
}
STYLES = {"persistence": (0, (5, 2)), "LGBM (tuned)": "-", "k-NN (tuned)": "-"}

cfg = load_config()
sup, cols = build_supervised(cfg, align(load_raw(cfg), cfg))
train, test = chronological_split(sup, cfg.split.test_fraction)
print(
    f"all data: {len(sup)} rows, {sup['user_id'].nunique()} user(s); "
    f"train={len(train)} test={len(test)}"
)

print("tuning to dataset (validation slice of train)...")
lgbm_params = tune_lgbm(train, cols, cfg)
knn_ks = tune_knn(train, cols, cfg)
lgbm_pred = build_lgbm_pred_fn(train, cols, cfg, lgbm_params)
knn_pred = build_knn_pred_fn(train, cols, cfg, knn_ks)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
summary = []
for ax, h in zip(axes, cfg.horizons_steps):
    hm = h * cfg.grid_minutes
    m = test[f"valid_{h}"].to_numpy()
    yt = test.loc[m, f"y_{h}"].to_numpy()
    preds = {
        "persistence": persistence_predict(test, h)[m],
        "LGBM (tuned)": lgbm_pred(test, h)[m],
        "k-NN (tuned)": knn_pred(test, h)[m],
    }
    for name, yp in preds.items():
        ae = np.sort(np.abs(yt - yp))
        cdf = np.arange(1, len(ae) + 1) / len(ae)
        ax.plot(ae, cdf, color=COLORS[name], lw=2, linestyle=STYLES[name], label=name)
        summary.append(
            (
                hm,
                name,
                float(np.median(np.abs(yt - yp))),
                float(np.percentile(np.abs(yt - yp), 90)),
            )
        )
    xcap = float(np.percentile(np.abs(yt - preds["persistence"]), 98))
    ax.set_xlim(0, xcap)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#d9d9d9", lw=0.8, zorder=0)
    ax.set_title(
        f"{hm}-min horizon  (n={int(m.sum())}, k-NN k={knn_ks[hm]})", fontsize=11
    )
    ax.set_xlabel("absolute error  |pred − actual|  (mg/dL)")
    ax.grid(True, alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("cumulative fraction of predictions ≤ x")
axes[0].legend(loc="lower right", frameon=False, fontsize=10)
fig.suptitle(
    "Absolute-error CDF — auto-tuned LGBM vs k-NN, out-of-sample (up & left = better)",
    fontweight="bold",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = diagram_path(cfg.paths.reports_dir, "cdf", "model_comparison", "auto-tuned")
fig.savefig(out, dpi=130)
print(f"\n-> {out}")

print(f"\n{'horizon':>7} {'model':<16} {'median AE':>10} {'P90 AE':>8}")
for hm, name, med, p90 in summary:
    print(f"{hm:>7} {name:<16} {med:>10.2f} {p90:>8.2f}")
