"""Recursive 5-min stepping vs direct multi-horizon (matched features).

Both learners see the SAME reduced, roll-forward-reconstructable feature set
(glucose lags + rate/accel/rolling + clock + time-since-meal), so this isolates
the *architecture* question: does chaining six easy 5-min steps beat one 30-min
jump? A sanity check asserts the rollout reconstructs features identically to the
pipeline at step 0 before any result is trusted. Single chronological split.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from insulink_predictor.config import load_config
from insulink_predictor.eval.outputs import diagram_path
from insulink_predictor.data.align import align
from insulink_predictor.data.load import load_raw
from insulink_predictor.eval.metrics import rmse, skill_score
from insulink_predictor.features.build import build_features
from insulink_predictor.features.target import build_targets
from insulink_predictor.models.lgbm import _make_regressor

cfg = load_config()
grid = align(load_raw(cfg), cfg)
feat, _ = build_features(grid, cfg)
sup = (
    build_targets(feat, [1, 6, 12])
    .sort_values(["user_id", "ts_utc"])
    .reset_index(drop=True)
)
assert sup["user_id"].nunique() == 1, "rollout seeding assumes one contiguous user"

LAGS = cfg.features.glucose_lags_min  # [0,5,10,15,30,45,60]
LSTEP = [m // cfg.grid_minutes for m in LAGS]  # [0,1,2,3,6,9,12]
ROLLW = [m // cfg.grid_minutes for m in cfg.features.roll_windows_min]  # [6,12]
FEATS = (
    [f"lag_{m}" for m in LAGS]
    + ["rate_short", "rate_long", "accel"]
    + [
        f"roll{m}_{s}"
        for m in cfg.features.roll_windows_min
        for s in ("mean", "std", "min", "max")
    ]
    + ["hour_sin", "hour_cos", "is_weekend", "day_of_week", "time_since_meal"]
)
MAXBACK = max(max(LSTEP), max(ROLLW))  # history needed = 12 steps

g_all = sup["glucose_mgdl"].to_numpy()
cut = int(len(sup) * (1.0 - cfg.split.test_fraction))


def reconstruct(H, s_elapsed, hour0, dow0, weekend0, tsm0):
    """Build the FEATS matrix (n, len(FEATS)) from glucose buffer H (n, >=13)."""
    col = {}
    for m, k in zip(LAGS, LSTEP):
        col[f"lag_{m}"] = H[:, -1 - k]
    col["rate_short"] = (H[:, -1] - H[:, -2]) / 5.0
    col["rate_long"] = (H[:, -1] - H[:, -1 - LSTEP[LAGS.index(30)]]) / 30.0
    col["accel"] = (H[:, -1] - 2 * H[:, -2] + H[:, -3]) / 25.0
    for m, w in zip(cfg.features.roll_windows_min, ROLLW):
        win = H[:, -w:]
        col[f"roll{m}_mean"] = win.mean(1)
        col[f"roll{m}_std"] = win.std(1, ddof=1)
        col[f"roll{m}_min"] = win.min(1)
        col[f"roll{m}_max"] = win.max(1)
    hour = (hour0 + 5.0 * s_elapsed / 60.0) % 24.0
    col["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    col["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    col["is_weekend"] = weekend0
    col["day_of_week"] = dow0
    col["time_since_meal"] = tsm0 + 5.0 * s_elapsed
    return np.column_stack([col[f] for f in FEATS])


# --- train: 5-min recursive model + direct 30/60 models, matched features ----
train = sup.iloc[:cut]


def fit(hstep):
    m = train[f"valid_{hstep}"].to_numpy()
    X = train.loc[m, FEATS]
    y = (train.loc[m, f"y_{hstep}"] - train.loc[m, "glucose_mgdl"]).to_numpy()
    return _make_regressor().fit(X, y)


m1, m6, m12 = fit(1), fit(6), fit(12)

# --- test anchors: need MAXBACK history; take the test split -----------------
pos = np.arange(cut, len(sup))
pos = pos[pos >= MAXBACK]
test = sup.iloc[pos].reset_index(drop=True)
H0 = g_all[pos[:, None] + np.arange(-MAXBACK, 1)]  # (n, 13) contiguous t-60..t
hour0 = (test["ts_local"].dt.hour + test["ts_local"].dt.minute / 60.0).to_numpy()
dow0 = test["day_of_week"].to_numpy()
wk0 = test["is_weekend"].to_numpy()
tsm0 = test["time_since_meal"].to_numpy()
g0 = test["glucose_mgdl"].to_numpy()

# --- SANITY: reconstruction at step 0 must match the pipeline features -------
R0 = reconstruct(H0, 0, hour0, dow0, wk0, tsm0)
P0 = test[FEATS].to_numpy()
both = np.isfinite(R0) & np.isfinite(P0)
maxdiff = np.nanmax(np.abs(R0[both] - P0[both]))
print(f"sanity: max |recon - pipeline| over {both.sum()} cells = {maxdiff:.2e}")
assert maxdiff < 1e-6, "feature reconstruction mismatch — rollout not trustworthy"

# --- recursive rollout (assume no new meal/insulin; exogenous decays away) ---
H = H0.copy()
preds_rec = {}
for step in range(1, 13):
    X = reconstruct(H, step - 1, hour0, dow0, wk0, tsm0)
    gnext = H[:, -1] + m1.predict(X)
    H = np.column_stack([H, gnext])
    if step in (6, 12):
        preds_rec[step] = gnext

# --- direct predictions -------------------------------------------------------
preds_dir = {6: m6.predict(P0) + g0, 12: m12.predict(P0) + g0}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
COL = {"persistence": "#7f7f7f", "direct": "#0072B2", "recursive 5-min": "#009E73"}
STY = {"persistence": (0, (5, 2)), "direct": "-", "recursive 5-min": "-"}
print(
    f"\n{'horizon':>7} {'window':>9} {'model':<16} {'median':>7} {'P90':>7} {'skill':>8}"
)
for ax, hstep in zip(axes, (6, 12)):
    hm = hstep * cfg.grid_minutes
    y_true = test[f"y_{hstep}"].to_numpy()
    valid = test[f"valid_{hstep}"].to_numpy()
    tsm = test["time_since_meal"].to_numpy()
    post = valid & np.isfinite(tsm) & (tsm <= 60)
    series = {
        "persistence": g0,
        "direct": preds_dir[hstep],
        "recursive 5-min": preds_rec[hstep],
    }
    for win, mask in (("all", valid), ("post_meal", post)):
        for name, yp in series.items():
            ae = np.abs(y_true[mask] - yp[mask])
            sk = skill_score(rmse(y_true[mask], yp[mask]), rmse(y_true[mask], g0[mask]))
            print(
                f"{hm:>7} {win:>9} {name:<16} {np.median(ae):>7.1f} {np.percentile(ae, 90):>7.1f} {sk:>+8.4f}"
            )
    # post-meal CDF
    for name, yp in series.items():
        ae = np.sort(np.abs(y_true[post] - yp[post]))
        ax.plot(
            ae,
            np.arange(1, len(ae) + 1) / len(ae),
            color=COL[name],
            lw=2,
            linestyle=STY[name],
            label=name,
        )
    ax.set_xlim(0, float(np.percentile(np.abs(y_true[post] - g0[post]), 98)))
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#d9d9d9", lw=0.8, zorder=0)
    ax.set_title(f"{hm}-min horizon, post-meal  (n={int(post.sum())})", fontsize=11)
    ax.set_xlabel("absolute error  |pred − actual|  (mg/dL)")
    ax.grid(True, alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    print()
axes[0].set_ylabel("cumulative fraction of predictions ≤ x")
axes[0].legend(loc="lower right", frameon=False, fontsize=10)
fig.suptitle(
    "Post-meal error CDF — recursive 5-min stepping vs direct (matched features)",
    fontweight="bold",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = diagram_path(cfg.paths.reports_dir, "cdf", "recursive_vs_direct", "reduced-feats")
fig.savefig(out, dpi=130)
print(f"-> {out}")
