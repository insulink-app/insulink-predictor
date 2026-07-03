"""Recursive 5-min vs direct — FULL features, walk-forward OOF, post-meal.

(a) OOF across embargoed folds; (c) full feature set. Glucose-derived features
come from the predicted path; every non-glucose feature (COB/IOB/activity/
time-since/clock) is read from its real deterministic value at t+s — valid with
zero leakage because we restrict to anchors with NO new meal/bolus in the horizon
(so those features depend only on inputs <= t). Continuous exogenous (steps/hr/
workout) are rolled forward with the realistic no-future assumption. A step-0
sanity assert confirms the assembled feature row matches the pipeline.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from insulink_predictor.config import load_config
from insulink_predictor.eval.outputs import diagram_path
from insulink_predictor.data.align import align
from insulink_predictor.data.load import load_raw
from insulink_predictor.eval.backtest import walk_forward_masks
from insulink_predictor.eval.metrics import rmse, skill_score
from insulink_predictor.features.build import build_features
from insulink_predictor.features.target import build_targets
from insulink_predictor.models.lgbm import _make_regressor

cfg = load_config()
grid = align(load_raw(cfg), cfg)
feat, cols = build_features(grid, cfg)
sup = build_targets(feat, [1, 6, 12]).sort_values(["user_id", "ts_utc"]).reset_index(drop=True)
feat = feat.sort_values(["user_id", "ts_utc"]).reset_index(drop=True)
assert sup["user_id"].nunique() == 1
print(f"full feature set ({len(cols)}): {cols}")

LAGS = cfg.features.glucose_lags_min
LSTEP = [m // cfg.grid_minutes for m in LAGS]
ROLLW = [m // cfg.grid_minutes for m in cfg.features.roll_windows_min]
GLUC = ([f"lag_{m}" for m in LAGS] + ["rate_short", "rate_long", "accel"]
        + [f"roll{m}_{s}" for m in cfg.features.roll_windows_min for s in ("mean", "std", "min", "max")])
GLUC = [c for c in GLUC if c in cols]
STEPW = {f"steps_{m}": m // cfg.grid_minutes for m in cfg.features.steps_windows_min if f"steps_{m}" in cols}
CONT = list(STEPW) + [c for c in ("hr_now", "workout_flag", "time_since_activity") if c in cols]
MAXBACK = max(max(LSTEP), max(ROLLW), 1)
CIDX = {c: i for i, c in enumerate(cols)}

FA = feat[cols].to_numpy()  # real feature matrix aligned to grid rows
g_all = sup["glucose_mgdl"].to_numpy()
steps_all = grid.sort_values(["user_id", "ts_utc"])["steps"].to_numpy() if "steps" in grid else None
step_cs = np.concatenate([[0], np.cumsum(np.nan_to_num(steps_all))]) if steps_all is not None else None
gr = grid.sort_values(["user_id", "ts_utc"]).reset_index(drop=True)
future_event = ((gr.get("meal_flag", pd.Series(False, index=gr.index)).astype(bool))
                | (gr.get("insulin_u", pd.Series(0.0, index=gr.index)).fillna(0) > 0)
                | (gr.get("carbs_g", pd.Series(0.0, index=gr.index)).fillna(0) > 0)).to_numpy()


def gluc_override(X, H):
    for m, k in zip(LAGS, LSTEP):
        if f"lag_{m}" in CIDX:
            X[:, CIDX[f"lag_{m}"]] = H[:, -1 - k]
    if "rate_short" in CIDX:
        X[:, CIDX["rate_short"]] = (H[:, -1] - H[:, -2]) / 5.0
    if "rate_long" in CIDX:
        X[:, CIDX["rate_long"]] = (H[:, -1] - H[:, -1 - LSTEP[LAGS.index(30)]]) / 30.0
    if "accel" in CIDX:
        X[:, CIDX["accel"]] = (H[:, -1] - 2 * H[:, -2] + H[:, -3]) / 25.0
    for m, w in zip(cfg.features.roll_windows_min, ROLLW):
        win = H[:, -w:]
        for st, fn in (("mean", win.mean(1)), ("std", win.std(1, ddof=1)), ("min", win.min(1)), ("max", win.max(1))):
            if f"roll{m}_{st}" in CIDX:
                X[:, CIDX[f"roll{m}_{st}"]] = fn


def cont_override(X, anchors, s):
    for c, w in STEPW.items():
        lo = np.clip(anchors + s - w + 1, 0, None)
        hi = anchors  # future steps (>p) treated as 0
        X[:, CIDX[c]] = step_cs[hi + 1] - step_cs[lo]
    if "hr_now" in CIDX:
        X[:, CIDX["hr_now"]] = FA[anchors, CIDX["hr_now"]]  # hold anchor
    if "workout_flag" in CIDX and s > 0:
        X[:, CIDX["workout_flag"]] = 0.0
    if "time_since_activity" in CIDX:
        X[:, CIDX["time_since_activity"]] = FA[anchors, CIDX["time_since_activity"]] + 5.0 * s


folds = walk_forward_masks(sup, 10, 0.6, embargo=max(cfg.horizons_steps))
pool = {6: {"y": [], "pers": [], "direct": [], "rec": []},
        12: {"y": [], "pers": [], "direct": [], "rec": []}}
sanity_max = 0.0
for tr_idx, te_idx in folds:
    train = sup.iloc[tr_idx]

    def fit(h):
        m = train[f"valid_{h}"].to_numpy()
        y = (train.loc[m, f"y_{h}"] - train.loc[m, "glucose_mgdl"]).to_numpy()
        return _make_regressor().fit(train.loc[m, cols], y)

    m1, m6, m12 = fit(1), fit(6), fit(12)
    te = np.asarray(te_idx)
    tsm = sup["time_since_meal"].to_numpy()
    ok = (sup["valid_6"].to_numpy() & sup["valid_12"].to_numpy()
          & np.isfinite(tsm) & (tsm <= 60))
    ok &= np.arange(len(sup)) >= MAXBACK
    anchors = te[ok[te]]
    # exclude anchors with a new meal/bolus/carb anywhere in (p, p+12]
    win = np.array([future_event[a + 1:a + 13].any() for a in anchors])
    anchors = anchors[~win]
    if len(anchors) == 0:
        continue

    H = g_all[anchors[:, None] + np.arange(-MAXBACK, 1)].astype(float)
    for s in range(12):
        X = FA[anchors + s].copy()
        gluc_override(X, H)
        cont_override(X, anchors, s)
        if s == 0:
            sanity_max = max(sanity_max, np.nanmax(np.abs(X - FA[anchors])))
        d = m1.predict(pd.DataFrame(X, columns=cols))
        H = np.column_stack([H, H[:, -1] + d])
        if s + 1 == 6:
            rec6 = H[:, -1]
        if s + 1 == 12:
            rec12 = H[:, -1]

    g0 = g_all[anchors]
    Xa = pd.DataFrame(FA[anchors], columns=cols)
    for h, rec in ((6, rec6), (12, rec12)):
        pool[h]["y"].append(sup[f"y_{h}"].to_numpy()[anchors])
        pool[h]["pers"].append(g0)
        pool[h]["direct"].append((m6 if h == 6 else m12).predict(Xa) + g0)
        pool[h]["rec"].append(rec)

print(f"sanity: max |assembled - pipeline| at step 0 = {sanity_max:.2e}")
assert sanity_max < 1e-6, "feature assembly mismatch"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
COL = {"persistence": "#7f7f7f", "direct": "#0072B2", "recursive 5-min": "#009E73"}
STY = {"persistence": (0, (5, 2)), "direct": "-", "recursive 5-min": "-"}
print(f"\n{'horizon':>7} {'model':<16} {'n':>6} {'median':>7} {'P90':>7} {'skill':>8}")
for ax, h in zip(axes, (6, 12)):
    hm = h * cfg.grid_minutes
    yt = np.concatenate(pool[h]["y"])
    g0 = np.concatenate(pool[h]["pers"])
    series = {"persistence": g0, "direct": np.concatenate(pool[h]["direct"]),
              "recursive 5-min": np.concatenate(pool[h]["rec"])}
    for name, yp in series.items():
        ae = np.abs(yt - yp)
        sk = skill_score(rmse(yt, yp), rmse(yt, g0))
        print(f"{hm:>7} {name:<16} {len(yt):>6} {np.median(ae):>7.1f} {np.percentile(ae, 90):>7.1f} {sk:>+8.4f}")
        aes = np.sort(ae)
        ax.plot(aes, np.arange(1, len(aes) + 1) / len(aes), color=COL[name], lw=2, linestyle=STY[name], label=name)
    ax.set_xlim(0, float(np.percentile(np.abs(yt - g0), 98)))
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="#d9d9d9", lw=0.8, zorder=0)
    ax.set_title(f"{hm}-min, post-meal (no new meal/bolus in window)  n={len(yt)}", fontsize=10)
    ax.set_xlabel("absolute error  |pred − actual|  (mg/dL)")
    ax.grid(True, alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("cumulative fraction ≤ x")
axes[0].legend(loc="lower right", frameon=False, fontsize=10)
fig.suptitle("Recursive 5-min vs direct — full features, walk-forward OOF, post-meal",
             fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = diagram_path(cfg.paths.reports_dir, "cdf", "recursive_full_oof", "full-feats")
fig.savefig(out, dpi=130)
print(f"-> {out}")
