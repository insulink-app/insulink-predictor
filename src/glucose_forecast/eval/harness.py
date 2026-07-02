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
from .reporting import mlflow_run, write_table
from .split import chronological_split

PredFn = Callable[[pd.DataFrame, int], np.ndarray]


def load_grid(cfg: Config) -> pd.DataFrame:
    """Load the aligned grid parquet, regenerating from synth if it is absent."""
    path = cfg.paths.data_dir / "processed" / "grid.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from ..data.align import align
    from ..data.synth import generate

    return align(generate(cfg), cfg)


def persistence_ref_rmse(test: pd.DataFrame, horizons_steps: list[int]) -> dict[int, float]:
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
            plot_parkes(yt, yp, reports / f"{prefix}_parkes_h{h * cfg.grid_minutes}.png")
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


def run_baseline_eval(cfg: Config, df: pd.DataFrame | None = None, write: bool = True) -> dict:
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
            log.params({"model": "persistence", "test_fraction": cfg.split.test_fraction})
            for _, r in metrics.iterrows():
                hm = int(r["horizon_min"])
                log.metrics({f"rmse_h{hm}": r["rmse"], f"mae_h{hm}": r["mae"]})

    return {"metrics": metrics, "error_grid": grids, "test": test, "ref_rmse": ref}
