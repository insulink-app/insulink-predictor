"""Pod age as a forecasting input — the pump's own signal.

A pod is a cannula sitting in one spot for three days, and the spot absorbs worse
as it ages, so the same units act more slowly late in a pod's life. The ``pumps``
table is the only place that knows when a spot was fresh. These tests pin the
three steps that carry it: the loader turns each activation into a pulse, align
keeps the pulse in its bucket, and the feature builder turns it into an age.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.load import _pod_block, assemble_raw
from insulink_predictor.features.build import build_features

_EPOCH = 1_700_000_000_000  # unix ms


def _raw(minutes: list[int], **channels) -> pd.DataFrame:
    """A raw frame on the 5-minute grid with glucose present throughout."""
    ts = pd.Timestamp("2025-01-06 00:00:00", tz="UTC") + pd.to_timedelta(
        minutes, unit="min"
    )
    n = len(minutes)
    frame = pd.DataFrame(
        {
            "user_id": "u",
            "ts_utc": ts,
            "ts_local": ts.tz_localize(None),
            "glucose_mgdl": [120.0] * n,
            "meal_flag": [False] * n,
            "carbs_g": [np.nan] * n,
            "insulin_u": [np.nan] * n,
            "steps": [0.0] * n,
            "activity_flag": [False] * n,
            "hr": [70.0] * n,
            "weather_temp": [10.0] * n,
        }
    )
    for name, values in channels.items():
        frame[name] = values
    return frame


def test_pod_block_marks_each_activation() -> None:
    cfg = Config()
    frame = pd.DataFrame(
        {"user_id": ["u", "u"], "registered_at": [_EPOCH, _EPOCH + 259_200_000]}
    )
    block = _pod_block(frame, cfg.pg)
    assert block is not None
    assert block["pod_flag"].tolist() == [True, True]
    # A pod activation is not insulin, not a meal, and not a glucose reading.
    assert block["insulin_u"].isna().all()
    assert block["glucose_mgdl"].isna().all()
    assert not block["meal_flag"].any()


def test_pod_block_absent_table_degrades() -> None:
    cfg = Config()
    assert _pod_block(None, cfg.pg) is None
    assert _pod_block(pd.DataFrame(), cfg.pg) is None


def test_assemble_raw_carries_pod_activations() -> None:
    cfg = Config()
    tables = {
        "glucose_entries": pd.DataFrame(
            {"user_id": ["u"], "recorded_at": [_EPOCH], "value": [120.0]}
        ),
        "pumps": pd.DataFrame({"user_id": ["u"], "registered_at": [_EPOCH]}),
    }
    raw = assemble_raw(tables, cfg)
    assert raw["pod_flag"].sum() == 1


def test_time_since_pod_counts_up_and_resets_on_the_next_pod() -> None:
    cfg = Config()
    minutes = list(range(0, 120, 5))
    pods = [False] * len(minutes)
    pods[2] = True  # first pod at t=10 min
    pods[14] = True  # swapped at t=70 min
    grid = align(_raw(minutes, pod_flag=pods), cfg)
    features, cols = build_features(grid, cfg)

    assert "time_since_pod" in cols
    age = features["time_since_pod"]
    assert age.iloc[:2].isna().all(), "no pod was known yet"
    assert age.iloc[2] == 0.0
    assert age.iloc[13] == 55.0  # ageing through the first pod
    assert age.iloc[14] == 0.0, "a fresh pod resets the age"
    assert age.iloc[-1] == 45.0


def test_no_pump_means_no_pod_feature() -> None:
    """A user without a pump must get exactly the features they got before."""
    cfg = Config()
    minutes = list(range(0, 60, 5))
    _, cols = build_features(align(_raw(minutes), cfg), cfg)
    assert "time_since_pod" not in cols
    # An all-False column (table present, user has no pod) is the same non-signal.
    pods = [False] * len(minutes)
    _, cols = build_features(align(_raw(minutes, pod_flag=pods), cfg), cfg)
    assert "time_since_pod" not in cols
