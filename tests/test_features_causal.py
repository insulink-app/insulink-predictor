"""Phase 2 DoD — the leakage test (§6). Features must be strictly causal.

Method: build features, corrupt every input channel strictly *after* a cutoff
``t0``, rebuild, and assert features for rows ``≤ t0`` are byte-identical. Any
dependence on the future would move them. A second test proves the method bites
by injecting a deliberately future-looking column and showing it *does* change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.synth import generate
from insulink_predictor.features.build import build_features


def _grid_and_cfg():
    cfg = Config(synth={"n_users": 2, "days": 6, "seed": 5})
    return align(generate(cfg), cfg), cfg


def _corrupt_future(df: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    fut = (out["ts_utc"] > t0).to_numpy()
    n = int(fut.sum())
    rng = np.random.default_rng(0)
    out.loc[fut, "glucose_mgdl"] = rng.uniform(40, 400, n)
    out.loc[fut, "carbs_g"] = rng.uniform(0, 100, n)
    out.loc[fut, "insulin_u"] = rng.uniform(0, 10, n)
    out.loc[fut, "steps"] = rng.integers(0, 500, n)
    out.loc[fut, "hr"] = rng.uniform(50, 180, n)
    out.loc[fut, "weather_temp"] = rng.uniform(-10, 40, n)
    out.loc[fut, "meal_flag"] = ~out.loc[fut, "meal_flag"]
    out.loc[fut, "activity_flag"] = ~out.loc[fut, "activity_flag"]
    return out


def test_features_are_causal_no_future_leak():
    grid, cfg = _grid_and_cfg()
    feat1, cols = build_features(grid, cfg)
    t0 = grid["ts_utc"].sort_values().iloc[len(grid) // 2]

    feat2, cols2 = build_features(_corrupt_future(grid, t0), cfg)
    assert cols == cols2  # corruption must not change the feature *set*

    a = feat1.loc[feat1["ts_utc"] <= t0, cols].reset_index(drop=True)
    b = feat2.loc[feat2["ts_utc"] <= t0, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)  # past features unchanged by future edits


def test_leakage_test_detects_an_injected_leak():
    """Sanity: a future-looking feature IS caught by the same perturbation."""
    grid, cfg = _grid_and_cfg()
    t0 = grid["ts_utc"].sort_values().iloc[len(grid) // 2]

    def build_with_leak(df):
        ordered = df.sort_values(["user_id", "ts_utc"]).reset_index(drop=True)
        feat, _ = build_features(df, cfg)  # same ordering as `ordered`
        leak = ordered.groupby("user_id")["glucose_mgdl"].shift(-6)  # future value
        return feat.assign(LEAK=leak.to_numpy())

    f1 = build_with_leak(grid)
    f2 = build_with_leak(_corrupt_future(grid, t0))

    leak1 = f1.loc[f1["ts_utc"] <= t0, "LEAK"].reset_index(drop=True)
    leak2 = f2.loc[f2["ts_utc"] <= t0, "LEAK"].reset_index(drop=True)
    # rows just below t0 read glucose beyond t0 -> the leak must be detectable
    assert not leak1.equals(leak2)
