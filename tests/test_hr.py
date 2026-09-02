"""Heart rate as a forecasting input.

The real deployment has no intraday step count and a near-empty activity flag, so
the worn band's pulse (``health_pulse_samples``) is the only intraday activity
signal there is. A bare bpm carries little on its own — 70 is rest for one person
and effort for another — so ``use_hr_dynamics`` derives two channels from it:
how far above this user's OWN rest the heart is now, and the accumulated exercise
pressure that outlasts the elevation (glucose keeps falling after the heart has
come back down).

These tests pin that both are causal in shape and that a user with no band at all
still gets exactly the feature set they got before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.features.build import build_features


def _grid(hr: list[float]) -> pd.DataFrame:
    minutes = list(range(0, 5 * len(hr), 5))
    ts = pd.Timestamp("2025-01-06 00:00:00", tz="UTC") + pd.to_timedelta(
        minutes, unit="min"
    )
    raw = pd.DataFrame(
        {
            "user_id": "u",
            "ts_utc": ts,
            "ts_local": ts.tz_localize(None),
            "glucose_mgdl": [120.0] * len(hr),
            "meal_flag": False,
            "carbs_g": np.nan,
            "insulin_u": np.nan,
            "steps": 0.0,
            "activity_flag": False,
            "hr": hr,
            "weather_temp": 10.0,
        }
    )
    return align(raw, Config())


def _cfg(dynamics: bool) -> Config:
    return Config(features={"use_hr_dynamics": dynamics})


def test_dynamics_off_leaves_only_the_bare_bpm() -> None:
    """The default must reproduce the committed feature set exactly."""
    _, cols = build_features(_grid([70.0] * 40), _cfg(False))
    assert "hr_now" in cols
    assert "hr_excess" not in cols and "hr_activity" not in cols


def test_excess_is_zero_at_rest_and_rises_with_effort() -> None:
    cfg = _cfg(True)
    resting = [70.0] * 60
    bout = resting + [140.0] * 12 + resting
    features, cols = build_features(_grid(bout), cfg)
    assert "hr_excess" in cols and "hr_activity" in cols

    excess = features["hr_excess"]
    assert (excess.dropna() >= 0).all(), "excess is never negative"
    assert excess.iloc[50] == 0.0, "sitting still is not effort"
    assert excess.iloc[65] > 50.0, "a 140 bpm bout is effort"


def test_activity_outlasts_the_elevation() -> None:
    """The point of the feature: the pressure is still there once the heart is not."""
    cfg = _cfg(True)
    bout = [70.0] * 60 + [140.0] * 12 + [70.0] * 40
    features, _ = build_features(_grid(bout), cfg)

    activity = features["hr_activity"].to_numpy()
    assert activity[59] == 0.0, "nothing had happened yet"
    after_bout = activity[75]  # 15 min after the heart came back down
    assert after_bout > 0.0
    assert features["hr_excess"].iloc[75] == 0.0, "the heart is back at rest"
    assert activity[-1] < after_bout, "and it decays away again"


def test_a_user_without_a_band_gets_no_hr_features() -> None:
    cfg = _cfg(True)
    _, cols = build_features(_grid([np.nan] * 40), cfg)
    assert not [c for c in cols if c.startswith("hr")]
