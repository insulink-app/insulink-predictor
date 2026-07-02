"""Metrics (ROADMAP §Phase 1). Skill-score is the north star.

``skill = 1 − RMSE_model / RMSE_persistence``: 0 means "no better than
persistence", >0 means genuinely better, <0 means worse. Every model is judged
by this per horizon.
"""

from __future__ import annotations

import numpy as np


def _clean(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def rmse(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _clean(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def skill_score(rmse_model: float, rmse_persistence: float) -> float:
    """1 − RMSE_model / RMSE_persistence. Persistence vs itself → 0."""
    if rmse_persistence == 0:
        return 0.0
    return float(1.0 - rmse_model / rmse_persistence)
