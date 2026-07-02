"""Phase 1 — supervised targets shift correctly and never leak across users/gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from insulink_predictor.features.target import build_targets


def _toy(n_per_user=6):
    rows = []
    base = pd.Timestamp("2025-01-06", tz="UTC")
    for uid, offset in [("a", 100.0), ("b", 200.0)]:
        for i in range(n_per_user):
            rows.append(
                {
                    "user_id": uid,
                    "ts_utc": base + pd.Timedelta(minutes=5 * i),
                    "glucose_mgdl": offset + i,  # strictly increasing per user
                    "sensor_gap": False,
                }
            )
    return pd.DataFrame(rows)


def test_target_is_future_value_within_user():
    df = build_targets(_toy(), horizons_steps=[1, 2])
    a = df[df["user_id"] == "a"].reset_index(drop=True)
    # y_1 at row i is glucose at row i+1
    assert a.loc[0, "y_1"] == a.loc[1, "glucose_mgdl"]
    assert a.loc[0, "y_2"] == a.loc[2, "glucose_mgdl"]


def test_no_cross_user_leakage():
    df = build_targets(_toy(), horizons_steps=[1])
    # last row of user 'a' must NOT borrow user 'b's first value
    a = df[df["user_id"] == "a"]
    assert np.isnan(a["y_1"].iloc[-1])
    assert not a["valid_1"].iloc[-1]


def test_gap_endpoints_are_invalid():
    df = _toy()
    # mark one interior point as a gap (glucose NaN)
    gi = df.index[3]
    df.loc[gi, "sensor_gap"] = True
    df.loc[gi, "glucose_mgdl"] = np.nan
    out = build_targets(df, horizons_steps=[1])
    # the gap row itself is invalid, and the row whose target is the gap is invalid
    assert not out.loc[gi, "valid_1"]
    assert not out.loc[gi - 1, "valid_1"]
