"""Post-meal absolute-error CDF (walk-forward OOF): where the tail actually is.

Both learners are tuned once to the dataset on the earliest (leakage-free) slice,
then walk-forward OOF predictions are pooled and restricted to rows <=60 min after
a logged meal. Reports post-meal skill vs persistence (denominator recomputed on
the post-meal subset, so it is a fair post-meal comparison).
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
from insulink_predictor.eval.backtest import walk_forward_masks
from insulink_predictor.eval.harness import build_supervised
from insulink_predictor.eval.metrics import rmse, skill_score
from insulink_predictor.eval.outputs import diagram_path
from insulink_predictor.eval.split import chronological_split
from insulink_predictor.models.baseline import persistence_predict

N_FOLDS, TEST_SPAN, POST_MIN = 10, 0.6, 60
MODELS = ["persistence", "LGBM (tuned)", "k-NN (tuned)"]
COLORS = {
    "persistence": "#7f7f7f",
    "LGBM (tuned)": "#0072B2",
    "k-NN (tuned)": "#D55E00",
}
STYLES = {"persistence": (0, (5, 2)), "LGBM (tuned)": "-", "k-NN (tuned)": "-"}

cfg = load_config()
sup, cols = build_supervised(cfg, align(load_raw(cfg), cfg))
sup = sup.sort_values(["user_id", "ts_utc"]).reset_index(drop=True)
folds = walk_forward_masks(sup, N_FOLDS, TEST_SPAN, embargo=max(cfg.horizons_steps))

tune_train, _ = chronological_split(sup, TEST_SPAN)  # earliest slice, before test folds
print("tuning to dataset (leakage-free early slice)...")
lgbm_params = tune_lgbm(tune_train, cols, cfg)
knn_ks = tune_knn(tune_train, cols, cfg)

oof = {h: {k: [] for k in ["y", "tsm"] + MODELS} for h in cfg.horizons_steps}
for tr_idx, te_idx in folds:
    train, test = sup.loc[tr_idx], sup.loc[te_idx]
    lp = build_lgbm_pred_fn(train, cols, cfg, lgbm_params)
    kp = build_knn_pred_fn(train, cols, cfg, knn_ks)
    for h in cfg.horizons_steps:
        m = test[f"valid_{h}"].to_numpy()
        if m.sum() == 0:
            continue
        oof[h]["y"].append(test[f"y_{h}"].to_numpy()[m])
        oof[h]["tsm"].append(test["time_since_meal"].to_numpy()[m])
        oof[h]["persistence"].append(persistence_predict(test, h)[m])
        oof[h]["LGBM (tuned)"].append(lp(test, h)[m])
        oof[h]["k-NN (tuned)"].append(kp(test, h)[m])

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
        rows.append(
            (
                hm,
                name,
                int(post.sum()),
                float(np.median(np.abs(yt_p - yp))),
                float(np.percentile(np.abs(yt_p - yp), 90)),
                sk,
            )
        )
    ax.set_xlim(0, float(np.percentile(np.abs(yt_p - persist_p), 98)))
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#d9d9d9", lw=0.8, zorder=0)
    ax.set_title(
        f"{hm}-min, post-meal  (n={int(post.sum())}, k-NN k={knn_ks[hm]})", fontsize=11
    )
    ax.set_xlabel("absolute error  |pred − actual|  (mg/dL)")
    ax.grid(True, alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("cumulative fraction of predictions ≤ x")
axes[0].legend(loc="lower right", frameon=False, fontsize=10)
fig.suptitle(
    f"Post-meal error CDF (≤{POST_MIN} min after a meal) — walk-forward OOF, auto-tuned",
    fontweight="bold",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = diagram_path(cfg.paths.reports_dir, "cdf", "post_meal", "auto-tuned")
fig.savefig(out, dpi=130)
print(f"-> {out}\n")
print(f"{'horizon':>7} {'model':<16} {'n':>6} {'median':>7} {'P90':>7} {'skill':>8}")
for hm, name, n, med, p90, sk in rows:
    print(f"{hm:>7} {name:<16} {n:>6} {med:>7.1f} {p90:>7.1f} {sk:>+8.4f}")
