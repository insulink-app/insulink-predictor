"""Background per-user tuning job with a champion/challenger promotion guard.

For each eligible user it pulls the full DB history, builds the causal features,
**auto-tunes** LightGBM per horizon, and trains the per-step curve models — but a
freshly built model only *replaces* the served one if it passes a promotion gate:

    1. Split the user's data chronologically: earlier = train, tail = holdout.
    2. Tune + train the CHALLENGER on the train split (it never sees the holdout).
    3. Re-fit the INCUMBENT's stored params on the same train split — a fair A/B,
       both judged on the same holdout, no leakage from the incumbent's history.
    4. Promote iff the challenger beats persistence on the holdout (skill floor)
       AND does not regress against the incumbent (within a tolerance).
    5. On promotion, refit the challenger's params on ALL data and write it
       atomically to ``<MODELS_DIR>/<user_id>.joblib`` (the dir serving reads).

A rejected challenger leaves the incumbent in place; a user with too little CGM to
tune is skipped (they keep falling back to a 404 in ``serve/app.py``).

Invoked by the FastAPI scheduler (daily / on startup) and by ``POST /admin/retrain``.
``MODELS_DIR`` is an env-overridable path — in Docker a mounted volume.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import joblib

from insulink_predictor.config import Config, load_config
from insulink_predictor.data.align import align
from insulink_predictor.data.load import (
    assemble_raw,
    connect,
    distinct_user_ids,
    fetch_tables,
)
from insulink_predictor.eval.harness import build_supervised
from insulink_predictor.eval.metrics import rmse, skill_score
from insulink_predictor.eval.split import chronological_split
from insulink_predictor.models.events import (
    build_curve_targets,
    conformal_offsets,
    train_curve_models,
    train_quantile_curve_models,
)
from insulink_predictor.models.tuning import tune_lgbm

log = logging.getLogger("insulink.training")

# Where per-user models are written. A locally-mounted Docker volume in prod.
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "artifacts/models"))
# Minimum curve-valid rows before we bother tuning a user (below this the tuner's
# validation slice is too thin to be meaningful; the user keeps the global model).
MIN_TRAIN_ROWS = int(os.environ.get("MIN_TRAIN_ROWS", "500"))  # ~1.7 days @ 5-min

# --- promotion gate knobs ---------------------------------------------------- #
# A challenger must beat persistence by more than this skill on every evaluable
# horizon (0.0 = must simply beat persistence).
PROMOTE_MIN_SKILL = float(os.environ.get("PROMOTE_MIN_SKILL", "0.0"))
# Allowed skill drop vs the incumbent before it counts as a regression (absorbs
# holdout noise so a statistically-equal retrain isn't blocked).
PROMOTE_TOLERANCE = float(os.environ.get("PROMOTE_TOLERANCE", "0.02"))
# Min valid holdout rows at a horizon before its skill number is trusted.
MIN_EVAL_ROWS = int(os.environ.get("MIN_EVAL_ROWS", "30"))

# Uncertainty band served alongside the point curve. An L2 point forecast is the
# conditional MEAN, so it is *correctly* shrunk toward the middle and almost never
# calls <70 / >180 (measured on the real user @60 min: 1.9% / 4.3% recall). The
# extremes live in the band, not in the point: the same rows at q10/q90 recall
# 47.5% / 50.2% of them, at no cost to the point forecast's RMSE.
BAND_QUANTILES = (0.1, 0.9)


def _params_by_step(
    tuned: dict[int, dict | None], grid_minutes: int, max_step: int
) -> dict[int, dict | None]:
    """Map each curve step ``k`` (1..max_step) to its nearest tuned horizon's params.

    ``tune_lgbm`` tunes per horizon (e.g. 30/60 min); the curve has one model per
    5-min step, so steps ≤ ~30 min take the 30-min params, later steps the 60-min
    params. A horizon that was too thin to tune maps to ``None`` -> _DEFAULT_PARAMS.
    """
    horizons_min = sorted(tuned)
    out: dict[int, dict | None] = {}
    for k in range(1, max_step + 1):
        step_min = k * grid_minutes
        nearest = min(horizons_min, key=lambda hm: abs(hm - step_min))
        out[k] = tuned.get(nearest)
    return out


def _atomic_dump(obj, path: Path) -> None:
    """Serialize ``obj`` to ``path`` atomically (write temp + ``os.replace``).

    Serving may read the file concurrently; an atomic rename means it never sees a
    half-written model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(obj, tmp)
        os.replace(tmp, path)  # atomic on the same filesystem
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _curve_skill(
    models: dict[int, object],
    feature_cols: list[str],
    holdout,
    cfg: Config,
    predict_delta: bool,
) -> dict[int, float | None]:
    """Per-horizon skill vs persistence of ``models`` on ``holdout``.

    ``None`` for a horizon with fewer than ``MIN_EVAL_ROWS`` valid rows to judge.
    """
    out: dict[int, float | None] = {}
    for h in cfg.horizons_steps:
        hm = h * cfg.grid_minutes
        mask = holdout[f"cvalid_{h}"].to_numpy()
        if int(mask.sum()) < MIN_EVAL_ROWS:
            out[hm] = None
            continue
        g = holdout.loc[mask, "glucose_mgdl"].to_numpy()
        ytrue = holdout.loc[mask, f"cy_{h}"].to_numpy()
        yhat = models[h].predict(holdout.loc[mask, feature_cols])
        if predict_delta:
            yhat = yhat + g  # models emit the delta over persistence
        # persistence prediction at t+h is just g_t
        out[hm] = round(skill_score(rmse(ytrue, yhat), rmse(ytrue, g)), 4)
    return out


def _evaluate_config(
    train,
    holdout,
    feature_cols: list[str],
    max_step: int,
    params_by_step: dict[int, dict | None] | None,
    cfg: Config,
) -> dict[int, float | None]:
    """Train curve models on ``train`` with ``params_by_step`` and score on ``holdout``."""
    models = train_curve_models(
        train, feature_cols, max_step, cfg.features.predict_delta, params_by_step
    )
    return _curve_skill(models, feature_cols, holdout, cfg, cfg.features.predict_delta)


def _incumbent_params(models_dir: Path, user_id: str) -> dict[int, dict | None] | None:
    """The currently-served model's stored per-step params (for a fair A/B), or None."""
    path = models_dir / f"{user_id}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path).get("params_by_step")
    except Exception:  # unreadable/legacy artifact -> no fair comparison possible
        return None


def _promotion_decision(
    chal_skill: dict[int, float | None],
    inc_skill: dict[int, float | None] | None,
    has_incumbent: bool,
) -> tuple[bool, str]:
    """Decide whether the challenger may replace the incumbent, with a reason."""
    evaluable = {h: s for h, s in chal_skill.items() if s is not None}
    if not evaluable:
        if has_incumbent:
            return False, "holdout too small to evaluate; kept incumbent"
        return True, "holdout too small to evaluate; promoted first model ungated"

    below = {h: s for h, s in evaluable.items() if s <= PROMOTE_MIN_SKILL}
    if below:
        return False, f"below persistence floor {PROMOTE_MIN_SKILL}: {below}"

    if inc_skill is not None:
        regressions = {
            h: (round(s, 4), round(inc_skill[h], 4))
            for h, s in evaluable.items()
            if inc_skill.get(h) is not None and s < inc_skill[h] - PROMOTE_TOLERANCE
        }
        if regressions:
            return False, (
                f"regression vs incumbent (tol {PROMOTE_TOLERANCE}) "
                f"[challenger, incumbent]: {regressions}"
            )
    return True, "ok"


def train_user_model(
    cfg: Config, user_id: str, engine, models_dir: Path | None = None
) -> dict:
    """Tune, gate (champion/challenger), and — if it passes — promote one user's model."""
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    raw = assemble_raw(fetch_tables(cfg, engine=engine, user_ids=[user_id]), cfg)
    if raw.empty:
        return {"user_id": user_id, "status": "skipped", "reason": "no data"}

    grid = align(raw, cfg)
    sup, feature_cols = build_supervised(cfg, grid)
    max_step = max(cfg.horizons_steps)
    sup = build_curve_targets(sup, max_step)

    n_valid = int(sup[f"cvalid_{max_step}"].sum())
    if n_valid < MIN_TRAIN_ROWS:
        return {
            "user_id": user_id,
            "status": "skipped",
            "reason": f"only {n_valid} valid rows (< {MIN_TRAIN_ROWS})",
        }

    # --- champion/challenger on a held-out tail (challenger never sees it) ------
    train, holdout = chronological_split(sup, cfg.split.test_fraction)
    tuned = tune_lgbm(train, feature_cols, cfg)
    pbs = _params_by_step(tuned, cfg.grid_minutes, max_step)
    chal_skill = _evaluate_config(train, holdout, feature_cols, max_step, pbs, cfg)

    prev_pbs = _incumbent_params(models_dir, user_id)
    inc_skill = (
        _evaluate_config(train, holdout, feature_cols, max_step, prev_pbs, cfg)
        if prev_pbs is not None
        else None
    )

    promote, reason = _promotion_decision(chal_skill, inc_skill, prev_pbs is not None)
    if not promote:
        log.info("user %s NOT promoted (%s)", user_id, reason)
        return {
            "user_id": user_id,
            "status": "rejected",
            "reason": reason,
            "challenger_skill": chal_skill,
            "incumbent_skill": inc_skill,
        }

    # --- promote: refit the winning params on ALL data, then write atomically --
    models = train_curve_models(
        sup, feature_cols, max_step, cfg.features.predict_delta, params_by_step=pbs
    )
    # Uncertainty band. Fit on `train` only and conformally recalibrated on the
    # untouched `holdout`, so the offsets measure real coverage rather than the
    # band's own training residuals. ponytail: default params (untuned) and only
    # the outer quantiles — tune them if the band's coverage drifts off nominal.
    qmodels = train_quantile_curve_models(
        train, feature_cols, max_step, BAND_QUANTILES, cfg.features.predict_delta
    )
    q_offsets = conformal_offsets(
        qmodels, holdout, feature_cols, max_step, cfg.features.predict_delta
    )
    payload = {
        "curve_models": models,
        "quantile_models": qmodels,
        "q_offsets": q_offsets,
        "band_quantiles": list(BAND_QUANTILES),
        "feature_cols": feature_cols,
        "grid_minutes": cfg.grid_minutes,
        "max_step": max_step,
        "predict_delta": cfg.features.predict_delta,
        "params_by_step": pbs,  # read back as the incumbent config next run
        "meta": {
            "user_id": user_id,
            "n_valid_rows": n_valid,
            "tuned_horizons": {hm: (p is not None) for hm, p in tuned.items()},
            "data_range": [str(grid["ts_utc"].min()), str(grid["ts_utc"].max())],
            "eval_skill": chal_skill,
            "prev_skill": inc_skill,
        },
    }
    _atomic_dump(payload, models_dir / f"{user_id}.joblib")
    log.info(
        "promoted user %s (%d valid rows, holdout skill %s) -> %s",
        user_id,
        n_valid,
        chal_skill,
        models_dir,
    )
    return {
        "user_id": user_id,
        "status": "trained",
        "n_valid_rows": n_valid,
        "eval_skill": chal_skill,
        "prev_skill": inc_skill,
        "reason": reason,
    }


def retrain_all(
    cfg: Config | None = None, engine=None, models_dir: Path | None = None
) -> dict:
    """Tune + gate + promote every eligible user. Isolates per-user failures."""
    cfg = cfg or load_config()
    engine = engine or connect(cfg)
    users = distinct_user_ids(cfg, engine)
    log.info("retrain_all: %d candidate user(s)", len(users))

    results: list[dict] = []
    for u in users:
        try:
            results.append(train_user_model(cfg, u, engine, models_dir))
        except Exception as exc:  # one bad user must not abort the whole run
            log.exception("retrain failed for user %s", u)
            results.append({"user_id": u, "status": "error", "reason": str(exc)})

    summary = {
        "n_users": len(users),
        "trained": sum(r["status"] == "trained" for r in results),
        "rejected": sum(r["status"] == "rejected" for r in results),
        "skipped": sum(r["status"] == "skipped" for r in results),
        "errors": sum(r["status"] == "error" for r in results),
        "results": results,
    }
    log.info(
        "retrain_all done: trained=%(trained)d rejected=%(rejected)d "
        "skipped=%(skipped)d errors=%(errors)d",
        summary,
    )
    return summary
