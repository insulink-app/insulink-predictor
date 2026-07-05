"""Train the production curve model on the real-data grid and serialize it.

Unlike the ``gf`` CLI (which retrains per run and throws the model away), this
persists the per-step LightGBM curve models plus the exact feature-column order
that the serving API (``serve/app.py``) must reproduce. v1 is **glucose-only**:
the optional carb/insulin/hr/weather channels are disabled so the trained
feature set matches glucose-only inference input (no train/serve skew).

Run:  ``uv run python scripts/train_and_save.py``  (writes ``artifacts/model.joblib``).
Re-run whenever ``data/processed/grid.parquet`` is refreshed (``gf load``).
"""

from __future__ import annotations

from pathlib import Path

import joblib

from insulink_predictor.config import Config, load_config
from insulink_predictor.eval.harness import build_supervised, load_grid
from insulink_predictor.models.events import build_curve_targets, train_curve_models

ARTIFACT = Path("artifacts/model.joblib")


def glucose_only_config() -> Config:
    """Config with the optional non-glucose channels disabled (v1 scope).

    Shared contract with ``serve/app.py``: both build features from glucose
    alone, so the served model sees exactly the columns it was trained on.
    """
    cfg = load_config()
    cfg.features.use_carbs = False
    cfg.features.use_insulin = False
    cfg.features.use_hr = False
    cfg.features.use_weather = False
    cfg.features.use_therapy = False
    return cfg


def main() -> None:
    cfg = glucose_only_config()
    sup, feature_cols = build_supervised(cfg, load_grid(cfg))
    max_step = max(cfg.horizons_steps)
    sup = build_curve_targets(sup, max_step)
    models = train_curve_models(sup, feature_cols, max_step, cfg.features.predict_delta)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "curve_models": models,
            "feature_cols": feature_cols,
            "grid_minutes": cfg.grid_minutes,
            "max_step": max_step,
            "predict_delta": cfg.features.predict_delta,
        },
        ARTIFACT,
    )
    print(
        f"saved {len(models)} step-models, {len(feature_cols)} features "
        f"(max_step={max_step}) -> {ARTIFACT}"
    )


if __name__ == "__main__":
    main()
