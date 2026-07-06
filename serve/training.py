"""Background per-user tuning job.

For each eligible user it pulls the full DB history, builds the causal features,
**auto-tunes** LightGBM hyperparameters (per horizon), trains the per-step curve
models with those params, and writes them **atomically** to
``<MODELS_DIR>/<user_id>.joblib`` — the directory serving hot-reloads from. Users
with too little CGM to tune are skipped (they keep falling back to the global
model in ``serve/app.py``).

Invoked by the FastAPI scheduler (daily) and by ``POST /admin/retrain``.
``MODELS_DIR`` is an env-overridable path — in Docker it is a mounted volume so
the models survive restarts and never live inside the image.
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
from insulink_predictor.models.events import build_curve_targets, train_curve_models
from insulink_predictor.models.tuning import tune_lgbm

log = logging.getLogger("insulink.training")

# Where per-user models are written. A locally-mounted Docker volume in prod.
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "artifacts/models"))
# Minimum curve-valid rows before we bother tuning a user (below this the tuner's
# validation slice is too thin to be meaningful; the user keeps the global model).
MIN_TRAIN_ROWS = int(os.environ.get("MIN_TRAIN_ROWS", "500"))  # ~1.7 days @ 5-min


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


def train_user_model(
    cfg: Config, user_id: str, engine, models_dir: Path | None = None
) -> dict:
    """Tune + train one user's curve model and persist it. Returns a status dict."""
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

    tuned = tune_lgbm(sup, feature_cols, cfg)
    pbs = _params_by_step(tuned, cfg.grid_minutes, max_step)
    models = train_curve_models(
        sup, feature_cols, max_step, cfg.features.predict_delta, params_by_step=pbs
    )

    payload = {
        "curve_models": models,
        "feature_cols": feature_cols,
        "grid_minutes": cfg.grid_minutes,
        "max_step": max_step,
        "predict_delta": cfg.features.predict_delta,
        "meta": {
            "user_id": user_id,
            "n_valid_rows": n_valid,
            "tuned_horizons": {hm: (p is not None) for hm, p in tuned.items()},
            "data_range": [str(grid["ts_utc"].min()), str(grid["ts_utc"].max())],
        },
    }
    _atomic_dump(payload, models_dir / f"{user_id}.joblib")
    log.info("trained user %s (%d valid rows) -> %s", user_id, n_valid, models_dir)
    return {"user_id": user_id, "status": "trained", "n_valid_rows": n_valid}


def retrain_all(
    cfg: Config | None = None, engine=None, models_dir: Path | None = None
) -> dict:
    """Tune + train every eligible user. Isolates per-user failures. Returns a summary."""
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
        "skipped": sum(r["status"] == "skipped" for r in results),
        "errors": sum(r["status"] == "error" for r in results),
        "results": results,
    }
    log.info(
        "retrain_all done: trained=%(trained)d skipped=%(skipped)d errors=%(errors)d",
        summary,
    )
    return summary
