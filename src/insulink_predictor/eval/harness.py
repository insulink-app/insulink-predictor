"""Evaluation harness (ROADMAP §Phase 1). The north-star wiring.

A *prediction function* has signature ``pred_fn(test_df, horizon_steps) -> np.ndarray``
aligned to ``test_df`` rows. Persistence is one; Phase 2's LightGBM is another.
Everything is scored only on ``valid_{h}`` rows and compared to persistence via
the skill-score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..config import Config
from ..features.target import build_targets
from ..models.baseline import persistence_predict
from .error_grid import clarke_zone_pct, parkes_zone_pct, plot_parkes, unsafe_fraction
from .metrics import mae, rmse, skill_score
from .reporting import mlflow_run, plot_curves, plot_feature_importance, write_table
from .split import chronological_split, heldout_user_split

PredFn = Callable[[pd.DataFrame, int], np.ndarray]


def load_grid(cfg: Config) -> pd.DataFrame:
    """Load the aligned grid parquet, regenerating from synth if it is absent."""
    path = cfg.paths.data_dir / "processed" / "grid.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from ..data.align import align
    from ..data.synth import generate

    return align(generate(cfg), cfg)


def persistence_ref_rmse(
    test: pd.DataFrame, horizons_steps: list[int]
) -> dict[int, float]:
    """Per-horizon persistence RMSE — the denominator of every skill-score."""
    ref = {}
    for h in horizons_steps:
        mask = test[f"valid_{h}"].to_numpy()
        yt = test[f"y_{h}"].to_numpy()[mask]
        yp = persistence_predict(test, h)[mask]
        ref[h] = rmse(yt, yp)
    return ref


def score_predictions(
    test: pd.DataFrame,
    cfg: Config,
    pred_fn: PredFn,
    ref_rmse: dict[int, float] | None = None,
) -> pd.DataFrame:
    """RMSE/MAE/skill per horizon on valid test rows."""
    rows = []
    for h in cfg.horizons_steps:
        mask = test[f"valid_{h}"].to_numpy()
        yt = test[f"y_{h}"].to_numpy()[mask]
        yp = np.asarray(pred_fn(test, h))[mask]
        r, a = rmse(yt, yp), mae(yt, yp)
        ss = skill_score(r, ref_rmse[h]) if ref_rmse is not None else 0.0
        rows.append(
            {
                "horizon_min": h * cfg.grid_minutes,
                "n": int(mask.sum()),
                "rmse": round(r, 3),
                "mae": round(a, 3),
                "skill": round(ss, 4),
            }
        )
    return pd.DataFrame(rows)


def error_grid_table(
    test: pd.DataFrame, cfg: Config, pred_fn: PredFn, prefix: str, plot: bool = True
) -> pd.DataFrame:
    """Parkes (+ Clarke A/B) zone percentages per horizon; optional plots."""
    reports = Path(cfg.paths.reports_dir)
    rows = []
    for h in cfg.horizons_steps:
        mask = test[f"valid_{h}"].to_numpy()
        yt = test[f"y_{h}"].to_numpy()[mask]
        yp = np.asarray(pred_fn(test, h))[mask]
        pz = parkes_zone_pct(yt, yp)
        cz = clarke_zone_pct(yt, yp)
        if plot:
            plot_parkes(
                yt, yp, reports / f"{prefix}_parkes_h{h * cfg.grid_minutes}.png"
            )
        rows.append(
            {
                "horizon_min": h * cfg.grid_minutes,
                **{f"parkes_{z}": pz[z] for z in "ABCDE"},
                "parkes_unsafe_CDE": unsafe_fraction(pz),
                "clarke_A": cz["A"],
                "clarke_B": cz["B"],
            }
        )
    return pd.DataFrame(rows)


def run_baseline_eval(
    cfg: Config, df: pd.DataFrame | None = None, write: bool = True
) -> dict:
    """Phase 1 DoD: persistence report (RMSE/MAE @30/@60 + Parkes zones)."""
    df = df if df is not None else load_grid(cfg)
    sup = build_targets(df, cfg.horizons_steps)
    _, test = chronological_split(sup, cfg.split.test_fraction)

    ref = persistence_ref_rmse(test, cfg.horizons_steps)
    metrics = score_predictions(test, cfg, persistence_predict, ref_rmse=ref)
    metrics.insert(0, "model", "persistence")
    grids = error_grid_table(test, cfg, persistence_predict, "persistence", plot=write)

    if write:
        reports = Path(cfg.paths.reports_dir)
        write_table(metrics, reports / "persistence_metrics.csv")
        write_table(grids, reports / "persistence_error_grid.csv")
        with mlflow_run(cfg, "persistence") as log:
            log.params(
                {"model": "persistence", "test_fraction": cfg.split.test_fraction}
            )
            for _, r in metrics.iterrows():
                hm = int(r["horizon_min"])
                log.metrics({f"rmse_h{hm}": r["rmse"], f"mae_h{hm}": r["mae"]})

    return {"metrics": metrics, "error_grid": grids, "test": test, "ref_rmse": ref}


def build_supervised(cfg: Config, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Grid -> causal features -> supervised targets. Returns (frame, feature_cols)."""
    from ..features.build import build_features

    df = df if df is not None else load_grid(cfg)
    feat, feature_cols = build_features(df, cfg)
    sup = build_targets(feat, cfg.horizons_steps)
    return sup, feature_cols


def run_lgbm_eval(cfg: Config, df: pd.DataFrame | None = None, write: bool = True) -> dict:
    """Phase 2 DoD: LightGBM vs persistence — must beat it (skill > 0) per horizon."""
    from ..models.lgbm import feature_importance, make_pred_fn, train_lgbm

    sup, feature_cols = build_supervised(cfg, df)
    train, test = chronological_split(sup, cfg.split.test_fraction)

    models = train_lgbm(train, feature_cols, cfg)
    pred_fn = make_pred_fn(models, feature_cols)

    ref = persistence_ref_rmse(test, cfg.horizons_steps)
    lgbm_m = score_predictions(test, cfg, pred_fn, ref_rmse=ref)
    lgbm_m.insert(0, "model", "lgbm")
    pers_m = score_predictions(test, cfg, persistence_predict, ref_rmse=ref)
    pers_m.insert(0, "model", "persistence")
    comparison = pd.concat([pers_m, lgbm_m], ignore_index=True)
    fi = feature_importance(models, feature_cols)

    if write:
        reports = Path(cfg.paths.reports_dir)
        write_table(comparison, reports / "model_comparison.csv")
        write_table(fi, reports / "feature_importance.csv")
        plot_feature_importance(fi, reports / "feature_importance.png")
        error_grid_table(test, cfg, pred_fn, "lgbm", plot=True)
        with mlflow_run(cfg, "lgbm") as log:
            log.params({"model": "lgbm", "n_features": len(feature_cols),
                        "test_fraction": cfg.split.test_fraction})
            for _, r in lgbm_m.iterrows():
                hm = int(r["horizon_min"])
                log.metrics({f"rmse_h{hm}": r["rmse"], f"skill_h{hm}": r["skill"]})
            log.artifact(reports / "feature_importance.png")

    return {
        "comparison": comparison,
        "lgbm_metrics": lgbm_m,
        "persistence_metrics": pers_m,
        "feature_importance": fi,
        "models": models,
        "feature_cols": feature_cols,
        "test": test,
        "ref_rmse": ref,
    }


def subset_skill(test: pd.DataFrame, cfg: Config, pred_fn: PredFn, mask: np.ndarray) -> pd.DataFrame:
    """Skill of a model vs persistence, restricted to ``mask`` rows.

    Crucially the persistence denominator is recomputed **on the same subset** —
    so post-event skill is measured against persistence's own post-event error.
    """
    rows = []
    for h in cfg.horizons_steps:
        valid = test[f"valid_{h}"].to_numpy() & mask
        yt = test[f"y_{h}"].to_numpy()[valid]
        yp = np.asarray(pred_fn(test, h))[valid]
        r_model = rmse(yt, yp)
        r_pers = rmse(yt, persistence_predict(test, h)[valid])
        rows.append(
            {
                "horizon_min": h * cfg.grid_minutes,
                "n": int(valid.sum()),
                "rmse": round(r_model, 3),
                "rmse_persistence": round(r_pers, 3),
                "skill": round(skill_score(r_model, r_pers), 4),
            }
        )
    return pd.DataFrame(rows)


def _curve_examples(test, curve_models, feature_cols, cfg, max_step=None, n=3, at=None) -> list[dict]:
    """Forecast trajectories with a fully-observed future.

    Selection: post-meal events by default, or the bucket nearest ``at`` (a
    timestamp) if given. ``max_step`` sets the curve length (defaults to the
    longest configured horizon).
    """
    from ..models.events import forecast_curve

    max_step = max_step or max(cfg.horizons_steps)
    has_future = test[f"cvalid_{max_step}"].to_numpy()

    if at is not None:
        at_ts = pd.Timestamp(at)
        at_ts = at_ts.tz_localize("UTC") if at_ts.tz is None else at_ts.tz_convert("UTC")
        cand = test[has_future]
        if cand.empty:
            return []
        pos = (cand["ts_utc"] - at_ts).abs().to_numpy().argmin()
        picks = [int(np.flatnonzero(has_future)[pos])]
    else:
        ok = test["event_meal"].to_numpy() & has_future
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return []
        picks = idx[np.linspace(0, idx.size - 1, num=min(n, idx.size)).astype(int)]

    minutes = [k * cfg.grid_minutes for k in range(1, max_step + 1)]
    examples = []
    for j in picks:
        row = test.iloc[[j]]
        pred = forecast_curve(row, curve_models, feature_cols)[0][:max_step]
        actual = [float(test.iloc[j][f"cy_{k}"]) for k in range(1, max_step + 1)]
        ts = pd.Timestamp(test.iloc[j]["ts_local"]).strftime("%a %d.%m %H:%M")
        tag = "meal" if bool(test.iloc[j].get("event_meal", False)) else "t"
        uid_short = str(test.iloc[j]["user_id"])[:8]  # keep titles short so they don't overlap
        examples.append(
            {
                "title": f"{uid_short}… · {tag} @ {ts}",
                "g0": float(test.iloc[j]["glucose_mgdl"]),
                "minutes": minutes,
                "predicted": [float(x) for x in pred],
                "actual": actual,
            }
        )
    return examples


def _curve_metrics(test, cfg, curve_models, feature_cols, max_step) -> pd.DataFrame:
    """Skill/RMSE vs persistence over the forecast window, at the curve horizon.

    Reported over all out-of-sample rows and, separately, over post-meal rows
    (where the plotted curves anchor and persistence fails hardest).
    """
    from ..models.events import forecast_curve

    yhat = forecast_curve(test, curve_models, feature_cols)[:, max_step - 1]
    yt = test[f"cy_{max_step}"].to_numpy()
    g_t = test["glucose_mgdl"].to_numpy()  # persistence prediction
    valid = test[f"cvalid_{max_step}"].to_numpy()
    tsm = test["time_since_meal"].to_numpy()
    post = valid & np.isfinite(tsm) & (tsm <= cfg.event.post_event_window_min)

    rows = []
    for name, mask in (("all", valid), ("post_meal", post)):
        if mask.sum() == 0:
            continue
        r_m = rmse(yt[mask], yhat[mask])
        r_p = rmse(yt[mask], g_t[mask])
        rows.append(
            {
                "window": name,
                "horizon_min": max_step * cfg.grid_minutes,
                "n": int(mask.sum()),
                "rmse": round(r_m, 2),
                "rmse_persistence": round(r_p, 2),
                "skill": round(skill_score(r_m, r_p), 4),
            }
        )
    return pd.DataFrame(rows)


def run_curve(cfg, df=None, horizon_min=60, n=3, at=None, out=None, metrics=False) -> dict:
    """Train curve models on the early data and plot forecast trajectories.

    Forecasts are out-of-sample: the chronological tail (test split) is where the
    curves are drawn, after training on the earlier portion. With ``metrics``, also
    score the whole forecast window vs persistence at the horizon.
    """
    from ..models.events import build_curve_targets, detect_events, train_curve_models

    sup, feature_cols = build_supervised(cfg, df)
    max_step = max(1, horizon_min // cfg.grid_minutes)
    sup = build_curve_targets(sup, max_step)
    sup = detect_events(sup, cfg)

    train, test = chronological_split(sup, cfg.split.test_fraction)
    if train[f"cvalid_{max_step}"].sum() < 50 or len(test) <= max_step:
        return {"examples": [], "n_train": len(train), "n_test": len(test), "out": None,
                "metrics": None, "reason": "not enough data to train/forecast this range"}

    curve_models = train_curve_models(train, feature_cols, max_step)
    examples = _curve_examples(test, curve_models, feature_cols, cfg, max_step=max_step, n=n, at=at)

    out = Path(out) if out else Path(cfg.paths.reports_dir) / f"curves_{horizon_min}min.png"
    if examples:
        plot_curves(examples, out)
    metrics_df = _curve_metrics(test, cfg, curve_models, feature_cols, max_step) if metrics else None
    return {"examples": examples, "n_train": int(len(train)), "n_test": int(len(test)),
            "out": out, "metrics": metrics_df}


def run_event_eval(cfg: Config, df: pd.DataFrame | None = None, write: bool = True) -> dict:
    """Phase 3 DoD: post-meal skill must exceed the global skill; plot the curve."""
    from ..models.events import build_curve_targets, detect_events, train_curve_models
    from ..models.lgbm import make_pred_fn

    sup, feature_cols = build_supervised(cfg, df)
    max_step = max(cfg.horizons_steps)
    sup = build_curve_targets(sup, max_step)
    sup = detect_events(sup, cfg)

    train, test = chronological_split(sup, cfg.split.test_fraction)
    curve_models = train_curve_models(train, feature_cols, max_step)
    pred_fn = make_pred_fn(curve_models, feature_cols)  # keyed by step; covers horizons

    tsm = test["time_since_meal"].to_numpy()
    post_meal = np.isfinite(tsm) & (tsm <= cfg.event.post_event_window_min)
    global_mask = np.ones(len(test), dtype=bool)

    g = subset_skill(test, cfg, pred_fn, global_mask)
    g.insert(0, "window", "global")
    p = subset_skill(test, cfg, pred_fn, post_meal)
    p.insert(0, "window", "post_meal")
    comparison = pd.concat([g, p], ignore_index=True)

    examples = _curve_examples(test, curve_models, feature_cols, cfg)

    if write:
        reports = Path(cfg.paths.reports_dir)
        write_table(comparison, reports / "event_skill.csv")
        if examples:
            plot_curves(examples, reports / "event_curves.png")
        with mlflow_run(cfg, "events") as log:
            log.params({"model": "lgbm-curve", "post_event_window_min": cfg.event.post_event_window_min})
            for _, r in comparison.iterrows():
                log.metrics({f"skill_{r['window']}_h{int(r['horizon_min'])}": r["skill"]})

    return {
        "comparison": comparison,
        "global_skill": g,
        "post_meal_skill": p,
        "curve_models": curve_models,
        "feature_cols": feature_cols,
        "test": test,
        "examples": examples,
    }


def per_user_skill(test: pd.DataFrame, cfg: Config, pred_fn: PredFn, model_name: str) -> pd.DataFrame:
    """Per-user, per-horizon RMSE and skill vs persistence."""
    rows = []
    for uid, gdf in test.groupby("user_id"):
        preds = {h: np.asarray(pred_fn(gdf, h)) for h in cfg.horizons_steps}
        for h in cfg.horizons_steps:
            valid = gdf[f"valid_{h}"].to_numpy()
            yt = gdf.loc[valid, f"y_{h}"].to_numpy()
            yp = preds[h][valid]
            r_model = rmse(yt, yp)
            r_pers = rmse(yt, gdf.loc[valid, "glucose_mgdl"].to_numpy())
            rows.append(
                {
                    "model": model_name,
                    "user_id": uid,
                    "horizon_min": h * cfg.grid_minutes,
                    "rmse": round(r_model, 3),
                    "skill": round(skill_score(r_model, r_pers), 4),
                }
            )
    return pd.DataFrame(rows)


def run_personalize_eval(cfg: Config, df: pd.DataFrame | None = None, write: bool = True) -> dict:
    """Phase 4 DoD: personalized beats global per-user; cold-start degrades gracefully."""
    from ..models.personalize import (
        PersonalizedModel,
        add_static,
        static_cols,
        train_per_horizon,
        train_residuals,
        user_static_features,
    )

    sup, feature_cols = build_supervised(cfg, df)
    seen, heldout = heldout_user_split(sup, cfg.split.heldout_users)
    train, test = chronological_split(seen, cfg.split.test_fraction)

    # global (feature-only) baseline
    global_models = train_per_horizon(train, feature_cols, cfg)

    # 4a conditioned + 4b residual (ISF/ICR enter here, as user-static conditioning)
    static = user_static_features(train)
    scols = static_cols(static)
    train_c = add_static(train, static)
    cond_models = train_per_horizon(train_c, feature_cols + scols, cfg)
    residual_models = train_residuals(train_c, cond_models, feature_cols, scols, cfg)
    personalized = PersonalizedModel(cond_models, residual_models, feature_cols, scols, static)

    def global_pred(d, h):
        return global_models[h].predict(d[feature_cols])

    gu = per_user_skill(test, cfg, global_pred, "global")
    pu = per_user_skill(test, cfg, personalized.predict, "personalized")
    per_user = pd.concat([gu, pu], ignore_index=True)

    # summary: mean per-user skill + how many users improved, per horizon
    rows = []
    for h_min in sorted(gu["horizon_min"].unique()):
        gh = gu[gu["horizon_min"] == h_min].set_index("user_id")["skill"]
        ph = pu[pu["horizon_min"] == h_min].set_index("user_id")["skill"]
        improved = int((ph > gh).sum())
        rows.append(
            {
                "horizon_min": h_min,
                "global_mean_skill": round(gh.mean(), 4),
                "personalized_mean_skill": round(ph.mean(), 4),
                "users_improved": improved,
                "n_users": int(gh.size),
            }
        )
    summary = pd.DataFrame(rows)

    # cold-start: personalized pipeline must run on held-out users without crashing
    coldstart = {"n_heldout_users": int(heldout["user_id"].nunique()) if len(heldout) else 0}
    if len(heldout) > 0:
        finite_ok = True
        for h in cfg.horizons_steps:
            valid = heldout[f"valid_{h}"].to_numpy()
            preds = personalized.predict(heldout, h)
            finite_ok &= bool(np.isfinite(preds[valid]).all())
            # cold-start users have no residual model -> base only
            assert all(uid not in residual_models[h] for uid in heldout["user_id"].unique())
        coldstart["graceful"] = finite_ok

    if write:
        reports = Path(cfg.paths.reports_dir)
        write_table(summary, reports / "personalization_summary.csv")
        write_table(per_user, reports / "personalization_per_user.csv")
        with mlflow_run(cfg, "personalize") as log:
            log.params({"min_residual_rows": 200, "heldout": ",".join(cfg.split.heldout_users)})
            for _, r in summary.iterrows():
                hm = int(r["horizon_min"])
                log.metrics(
                    {
                        f"global_skill_h{hm}": r["global_mean_skill"],
                        f"personalized_skill_h{hm}": r["personalized_mean_skill"],
                    }
                )

    return {
        "summary": summary,
        "per_user": per_user,
        "coldstart": coldstart,
        "personalized": personalized,
        "global_models": global_models,
        "feature_cols": feature_cols,
        "heldout": heldout,
        "test": test,
    }
