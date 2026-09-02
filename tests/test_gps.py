"""GPS as a forecasting input — movement and place.

The app's background sampler writes ``location_entries`` all day, which makes it
the only *continuous* movement signal this deployment has: the intraday ``steps``
channel is empty and logged workouts are rare, so everyday movement reaches the
model through here or not at all.

Two things are derived from it. Movement — speed, trailing distance, and how long
the user has been settled. And place familiarity — the share of this user's own
past spent where they are now, which is deliberately unsupervised: no home address
is ever declared and no coordinate is ever a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.load import _location_block, assemble_raw
from insulink_predictor.features.build import build_features

_EPOCH = 1_700_000_000_000  # unix ms
_HOME = (51.0, 7.0)


def _cfg(gps: bool = True) -> Config:
    return Config(features={"use_gps": gps})


def _grid(lat: list[float], lon: list[float]) -> pd.DataFrame:
    minutes = list(range(0, 5 * len(lat), 5))
    ts = pd.Timestamp("2025-01-06 00:00:00", tz="UTC") + pd.to_timedelta(
        minutes, unit="min"
    )
    raw = pd.DataFrame(
        {
            "user_id": "u",
            "ts_utc": ts,
            "ts_local": ts.tz_localize(None),
            "glucose_mgdl": [120.0] * len(lat),
            "meal_flag": False,
            "carbs_g": np.nan,
            "insulin_u": np.nan,
            "steps": 0.0,
            "activity_flag": False,
            "hr": np.nan,
            "weather_temp": np.nan,
            "lat": lat,
            "lon": lon,
        }
    )
    return align(raw, Config())


def _still(n: int) -> tuple[list[float], list[float]]:
    return [_HOME[0]] * n, [_HOME[1]] * n


# --------------------------------------------------------------------------- #
# The loader maps location_entries and refuses broken fixes                   #
# --------------------------------------------------------------------------- #
def test_location_block_maps_coordinates() -> None:
    frame = pd.DataFrame(
        {
            "user_id": ["u", "u"],
            "recorded_at": [_EPOCH, _EPOCH + 60_000],
            "latitude": [51.0, 51.001],
            "longitude": [7.0, 7.001],
        }
    )
    block = _location_block(frame, Config().pg)
    assert block is not None
    assert block["lat"].tolist() == [51.0, 51.001]
    # A position is not a reading, a dose or a meal.
    assert block["glucose_mgdl"].isna().all()
    assert not block["meal_flag"].any()


def test_location_block_drops_broken_fixes() -> None:
    """A 0/0 placeholder or an out-of-range value would invent movement."""
    frame = pd.DataFrame(
        {
            "user_id": ["u"] * 3,
            "recorded_at": [1, 2, 3],
            "latitude": [0.0, 999.0, 51.0],
            "longitude": [0.0, 7.0, 7.0],
        }
    )
    block = _location_block(frame, Config().pg)
    assert block is not None
    assert block["lat"].isna().tolist() == [True, True, False]


def test_location_block_absent_table_degrades() -> None:
    assert _location_block(None, Config().pg) is None
    assert _location_block(pd.DataFrame(), Config().pg) is None


def test_assemble_raw_carries_positions() -> None:
    tables = {
        "glucose_entries": pd.DataFrame(
            {"user_id": ["u"], "recorded_at": [_EPOCH], "value": [120.0]}
        ),
        "location_entries": pd.DataFrame(
            {
                "user_id": ["u"],
                "recorded_at": [_EPOCH],
                "latitude": [51.0],
                "longitude": [7.0],
            }
        ),
    }
    raw = assemble_raw(tables, Config())
    assert raw["lat"].max() == 51.0


# --------------------------------------------------------------------------- #
# Movement                                                                    #
# --------------------------------------------------------------------------- #
def test_speed_is_zero_at_rest_and_rises_on_the_move() -> None:
    lat, lon = _still(20)
    walk = [_HOME[0] + 0.004 * i for i in range(1, 13)]  # ~89 m/min, a walk
    features, cols = build_features(_grid(lat + walk, lon + [_HOME[1]] * 12), _cfg())
    assert "gps_speed" in cols

    speed = features["gps_speed"]
    assert np.isnan(speed.iloc[0]), "no previous fix to measure against"
    assert speed.iloc[10] == 0.0, "standing still is zero speed, not missing data"
    assert 60.0 < speed.iloc[25] < 120.0, "a walk reads as walking pace"


def test_trailing_distance_accumulates_a_walk() -> None:
    lat, lon = _still(10)
    walk = [_HOME[0] + 0.004 * i for i in range(1, 13)]
    features, cols = build_features(_grid(lat + walk, lon + [_HOME[1]] * 12), _cfg())
    assert "gps_dist_30" in cols and "gps_dist_60" in cols
    assert features["gps_dist_30"].iloc[9] == 0.0
    assert features["gps_dist_30"].iloc[-1] > 2000.0
    assert features["gps_dist_60"].iloc[-1] >= features["gps_dist_30"].iloc[-1]


def test_settled_minutes_count_up_after_the_walk_ends() -> None:
    """Movement raises insulin sensitivity for hours; "stopped 20 min ago" is a
    different state from "has been sitting all afternoon"."""
    lat, lon = _still(10)
    walk = [_HOME[0] + 0.004 * i for i in range(1, 13)]
    rest = [_HOME[0] + 0.004 * 12] * 24
    features, _ = build_features(
        _grid(lat + walk + rest, lon + [_HOME[1]] * 36), _cfg()
    )
    settled = features["gps_settled_min"]
    assert settled.iloc[15] == 0.0, "still walking"
    assert settled.iloc[-1] > 60.0, "long since settled"
    assert settled.is_monotonic_increasing is False  # resets while moving


def test_gps_speed_is_missing_not_zero_without_a_fix() -> None:
    lat, lon = _still(20)
    lat[10:14] = [np.nan] * 4
    lon[10:14] = [np.nan] * 4
    features, _ = build_features(_grid(lat, lon), _cfg())
    assert features["gps_speed"].iloc[11:14].isna().all()


# --------------------------------------------------------------------------- #
# Place — unsupervised, no home address anywhere                              #
# --------------------------------------------------------------------------- #
def test_familiarity_is_high_where_the_user_lives_and_low_somewhere_new() -> None:
    lat, lon = _still(90)
    away_lat = [_HOME[0] + 0.5] * 10  # ~55 km away, a place never visited before
    features, cols = build_features(
        _grid(lat + away_lat, lon + [_HOME[1]] * 10), _cfg()
    )
    assert "place_familiarity" in cols
    familiarity = features["place_familiarity"]
    assert familiarity.iloc[89] > 0.95, "the spot they spend their life in"
    assert familiarity.iloc[92] < 0.05, "somewhere they have never been"
    # It is a share, so it can never leave [0, 1] — no slow drift a tree could
    # mistake for a time trend.
    assert familiarity.dropna().between(0.0, 1.0).all()


def test_gps_jitter_does_not_split_one_place_in_two() -> None:
    rng = np.random.default_rng(0)
    n = 80
    lat = list(_HOME[0] + rng.normal(0, 0.0003, n))  # ~30 m of GPS noise
    lon = list(_HOME[1] + rng.normal(0, 0.0003, n))
    features, _ = build_features(_grid(lat, lon), _cfg())
    assert features["place_familiarity"].iloc[-1] > 0.8


# --------------------------------------------------------------------------- #
# Degrading cleanly                                                           #
# --------------------------------------------------------------------------- #
def test_flag_off_produces_no_gps_features() -> None:
    lat, lon = _still(20)
    _, cols = build_features(_grid(lat, lon), _cfg(gps=False))
    assert not [c for c in cols if c.startswith(("gps_", "place_"))]


def test_a_user_without_location_gets_no_gps_features() -> None:
    _, cols = build_features(_grid([np.nan] * 20, [np.nan] * 20), _cfg())
    assert not [c for c in cols if c.startswith(("gps_", "place_"))]


def test_familiarity_forgets_a_place_left_behind() -> None:
    """Somebody who moves house stops being a stranger in their own street.

    Also the reason the window is bounded at all: serving reads a slice of the DB,
    so an unbounded count would mean one thing in training and another at
    inference.
    """
    cfg = Config(features={"use_gps": True, "place_window_days": 1})
    per_day = 1440 // cfg.grid_minutes
    old_home, new_home = _still(per_day * 2)
    moved_lat = [_HOME[0] + 0.5] * (per_day * 2)
    features, _ = build_features(
        _grid(old_home + moved_lat, new_home + [_HOME[1]] * per_day * 2), cfg
    )
    familiarity = features["place_familiarity"]
    assert familiarity.iloc[per_day * 2 - 1] > 0.95, "the old place, while living there"
    assert familiarity.iloc[per_day * 2 + 5] < 0.05, "the new place, on arrival"
    assert familiarity.iloc[-1] > 0.95, "the new place, once it is a day old"
