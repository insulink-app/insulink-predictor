"""The pump's basal drip as a forecasting input.

Boluses already reach the model as meal rows. Basal does not, and on a pump it is
often about half a day's insulin — so insulin on board built from boluses alone is
built from half the insulin. These tests pin the two decisions that make adding it
safe rather than harmful:

1. It rides its OWN raw channel (``basal_u``), never folded into ``insulin_u``.
   ``insulin_u`` drives the discrete-dose features (``_bolus_flag``,
   ``time_since_bolus``); a continuous drip in there would make "a bolus happened"
   true in every bucket.
2. The two ARE summed where summing is correct — insulin on board and its rate of
   action — because basal and bolus insulin are the same molecule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.load import _basal_block, assemble_raw
from insulink_predictor.features.build import build_features


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
            "isf": [40.0] * n,
            "icr": [10.0] * n,
        }
    )
    for name, values in channels.items():
        frame[name] = values
    return frame


# --------------------------------------------------------------------------- #
# The loader maps basal_entries onto its own channel                          #
# --------------------------------------------------------------------------- #
def test_basal_block_maps_units_onto_its_own_channel() -> None:
    cfg = Config()
    frame = pd.DataFrame(
        {
            "user_id": ["u", "u"],
            "recorded_at": [1_700_000_000_000, 1_700_000_900_000],
            "insulin": [0.24, 0.24],
        }
    )
    block = _basal_block(frame, cfg.pg)
    assert block is not None
    assert list(block["basal_u"]) == [0.24, 0.24]
    # It must not leak into the bolus channel or raise a meal.
    assert block["insulin_u"].isna().all()
    assert block["carbs_g"].isna().all()
    assert not block["meal_flag"].any()


def test_basal_block_ignores_non_positive_units() -> None:
    cfg = Config()
    frame = pd.DataFrame(
        {
            "user_id": ["u", "u", "u"],
            "recorded_at": [1, 2, 3],
            "insulin": [0.0, -1.0, 0.5],
        }
    )
    block = _basal_block(frame, cfg.pg)
    assert block is not None
    assert block["basal_u"].isna().tolist() == [True, True, False]


def test_basal_block_absent_table_degrades() -> None:
    cfg = Config()
    assert _basal_block(None, cfg.pg) is None
    assert _basal_block(pd.DataFrame(), cfg.pg) is None


def test_assemble_raw_keeps_basal_and_boluses_apart() -> None:
    cfg = Config()
    tables = {
        "nutrition_meals": pd.DataFrame(
            {
                "user_id": ["u"],
                "time": [1_700_000_000_000],
                "carbs": [40.0],
                "glucose": [140.0],
                "bolus": [4.0],
            }
        ),
        "basal_entries": pd.DataFrame(
            {
                "user_id": ["u"],
                "recorded_at": [1_700_000_900_000],
                "insulin": [0.25],
            }
        ),
    }
    raw = assemble_raw(tables, cfg)
    assert raw["insulin_u"].max() == 4.0
    assert raw["basal_u"].max() == 0.25
    # The bolus row carries no basal and the basal row carries no bolus.
    assert raw.loc[raw["basal_u"].notna(), "insulin_u"].isna().all()


# --------------------------------------------------------------------------- #
# align() sums basal per bucket, like a dose, not means it like a rate         #
# --------------------------------------------------------------------------- #
def test_align_sums_basal_within_a_bucket() -> None:
    cfg = Config()
    # Two basal windows landing in the same 5-minute bucket.
    raw = _raw([0, 1, 5], basal_u=[0.1, 0.2, 0.3])
    grid = align(raw, cfg)
    assert grid["basal_u"].iloc[0] == 0.30000000000000004 or np.isclose(
        grid["basal_u"].iloc[0], 0.3
    )
    assert np.isclose(grid["basal_u"].iloc[1], 0.3)


def test_align_without_basal_still_works() -> None:
    cfg = Config()
    grid = align(_raw([0, 5, 10]), cfg)
    assert "basal_u" not in grid.columns or grid["basal_u"].isna().all()
    assert len(grid) == 3


# --------------------------------------------------------------------------- #
# The features: IOB includes basal, the bolus edges do not                    #
# --------------------------------------------------------------------------- #
def _grid_with(minutes: list[int], **channels) -> pd.DataFrame:
    return align(_raw(minutes, **channels), Config())


def test_iob_counts_basal() -> None:
    cfg = Config()
    minutes = list(range(0, 120, 5))
    # A steady drip and nothing else.
    drip = [0.05] * len(minutes)
    with_basal, with_cols = build_features(_grid_with(minutes, basal_u=drip), cfg)
    without, without_cols = build_features(_grid_with(minutes), cfg)
    # Basal alone is enough to put insulin on board.
    assert "iob" in with_cols
    assert with_basal["iob"].iloc[-1] > 0
    # With no insulin of any kind there is no IOB feature at all, as before.
    assert "iob" not in without_cols


def test_basal_does_not_raise_the_bolus_flag() -> None:
    """The regression this design exists to prevent."""
    cfg = Config()
    minutes = list(range(0, 120, 5))
    drip = [0.05] * len(minutes)
    features, cols = build_features(_grid_with(minutes, basal_u=drip), cfg)
    # No bolus was ever given, so the bolus-edge feature must not exist at all —
    # and certainly must not read as "a bolus just happened" in every bucket.
    assert "time_since_bolus" not in cols
    assert "_bolus_flag" not in features.columns
    # The basal is still on board, though.
    assert features["iob"].iloc[-1] > 0


def test_a_real_bolus_still_marks_its_edge_alongside_basal() -> None:
    cfg = Config()
    minutes = list(range(0, 120, 5))
    drip = [0.05] * len(minutes)
    boluses = [np.nan] * len(minutes)
    boluses[6] = 3.0
    features, _ = build_features(
        _grid_with(minutes, basal_u=drip, insulin_u=boluses), cfg
    )
    since = features["time_since_bolus"]
    assert since.iloc[:6].isna().all(), "no bolus had happened yet"
    assert since.iloc[6] == 0.0
    assert since.iloc[-1] > 0.0


def test_basal_now_exposes_the_current_rate() -> None:
    cfg = Config()
    minutes = list(range(0, 60, 5))
    drip = [0.05] * len(minutes)
    # A temp-basal reduction to zero in the second half.
    drip[6:] = [0.0] * (len(minutes) - 6)
    features, cols = build_features(_grid_with(minutes, basal_u=drip), cfg)
    assert "basal_now" in cols
    assert features["basal_now"].iloc[0] > 0
    assert features["basal_now"].iloc[-1] == 0


def test_insulin_activity_counts_basal() -> None:
    cfg = Config()
    minutes = list(range(0, 240, 5))
    drip = [0.05] * len(minutes)
    with_basal, with_cols = build_features(_grid_with(minutes, basal_u=drip), cfg)
    _, without_cols = build_features(_grid_with(minutes), cfg)
    assert "ins_activity" in with_cols, "basal must drive insulin activity too"
    assert with_basal["ins_activity"].abs().iloc[-1] > 0
    assert "ins_activity" not in without_cols


def test_features_without_a_pump_are_unchanged() -> None:
    """A user with no pump must get exactly the features they got before."""
    cfg = Config()
    minutes = list(range(0, 120, 5))
    boluses = [np.nan] * len(minutes)
    boluses[4] = 5.0
    baseline, cols = build_features(_grid_with(minutes, insulin_u=boluses), cfg)
    assert "basal_now" not in cols
    assert baseline["iob"].iloc[-1] > 0
    assert baseline["time_since_bolus"].iloc[4] == 0.0
