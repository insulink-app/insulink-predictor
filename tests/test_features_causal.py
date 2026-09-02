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
    # Optional channels, present only on the all-channels grid below.
    for col in ("basal_u", "daily_steps", "daily_distance"):
        if col in out.columns:
            out.loc[fut, col] = rng.uniform(0, 20, n)
    if "pod_flag" in out.columns:
        out.loc[fut, "pod_flag"] = ~out.loc[fut, "pod_flag"]
    if "lat" in out.columns:
        out.loc[fut, "lat"] = rng.uniform(-60, 60, n)
        out.loc[fut, "lon"] = rng.uniform(-120, 120, n)
    return out


def _all_channels_grid_and_cfg():
    """A grid carrying every optional channel, with every optional flag ON.

    The default config leaves the candidate channels off, so the test above proves
    causality only for the committed feature set. Anything a flag can switch on
    has to clear the same bar before it can ever be enabled.
    """
    cfg = Config(
        synth={"n_users": 2, "days": 6, "seed": 5},
        features={
            "use_hr_dynamics": True,
            "use_tod_baseline": True,
            "use_physio_delta": True,
            "use_daily_activity": True,
            "use_gps": True,
        },
    )
    grid, _ = align(generate(cfg), cfg), None
    rng = np.random.default_rng(3)
    n = len(grid)
    # Channels the synthetic generator does not produce: a pump's drip and pod
    # swaps, and the daily activity totals.
    grid["basal_u"] = 0.05
    grid["pod_flag"] = False
    grid.loc[grid.index % 864 == 0, "pod_flag"] = True  # ~every 3 days
    grid["daily_steps"] = rng.uniform(2000, 15000, n)
    grid["daily_distance"] = rng.uniform(1, 12, n)
    # A life spent mostly in one spot, with occasional trips elsewhere.
    grid["lat"] = 51.0 + rng.normal(0, 0.0004, n)
    grid["lon"] = 7.0 + rng.normal(0, 0.0004, n)
    trip = (grid.index // 200) % 7 == 0
    grid.loc[trip, "lat"] += 0.4
    return grid, cfg


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


def test_features_are_causal_with_every_optional_channel_on():
    grid, cfg = _all_channels_grid_and_cfg()
    feat1, cols = build_features(grid, cfg)
    # The flags must actually have produced their features, or this proves nothing.
    for expected in (
        "hr_excess",
        "hr_activity",
        "time_since_pod",
        "tod_baseline",
        "gps_speed",
        "gps_settled_min",
        "place_familiarity",
    ):
        assert expected in cols, expected
    t0 = grid["ts_utc"].sort_values().iloc[len(grid) // 2]

    feat2, cols2 = build_features(_corrupt_future(grid, t0), cfg)
    assert cols == cols2

    a = feat1.loc[feat1["ts_utc"] <= t0, cols].reset_index(drop=True)
    b = feat2.loc[feat2["ts_utc"] <= t0, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
