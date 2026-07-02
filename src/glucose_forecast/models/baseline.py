"""Persistence baseline (ROADMAP §Phase 1) — the thing every model must beat.

``ŷ_{t+h} = g_t``: the best guess in the resting state is "flat stays flat".
It is horizon-independent (the same current value predicts every horizon).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_predict(df: pd.DataFrame, horizon_steps: int) -> np.ndarray:
    """Predict ``g_t`` for every row (independent of the horizon)."""
    return df["glucose_mgdl"].to_numpy(dtype=float)
