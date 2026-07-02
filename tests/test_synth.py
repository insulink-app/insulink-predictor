"""Phase 0 — synthetic generator is reproducible, multi-user and irregular."""

from __future__ import annotations

import pandas as pd

from insulink_predictor.config import Config
from insulink_predictor.data.synth import generate

_REQUIRED = {
    "user_id",
    "ts_utc",
    "ts_local",
    "glucose_mgdl",
    "meal_flag",
    "carbs_g",
    "insulin_u",
    "steps",
    "activity_flag",
    "hr",
    "weather_temp",
}


def test_reproducible(cfg: Config):
    a = generate(cfg)
    b = generate(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_seed_changes_output(cfg: Config):
    other = Config(synth={**cfg.synth.model_dump(), "seed": cfg.synth.seed + 1})
    assert not generate(cfg).equals(generate(other))


def test_multiple_users_and_columns(cfg: Config, raw):
    assert raw["user_id"].nunique() == cfg.synth.n_users
    assert _REQUIRED.issubset(raw.columns)


def test_raw_is_irregular(raw):
    # raw is pre-alignment: cadence must NOT be a clean regular grid.
    one = raw[raw["user_id"] == "user_0"].sort_values("ts_utc")
    diffs = one["ts_utc"].diff().dropna().dt.total_seconds() / 60
    assert diffs.nunique() > 1  # jitter + gaps => varied spacing


def test_optional_insulin_degrades(cfg: Config, raw):
    # some users log insulin, some don't (all-NaN) — exercises optional-feature path.
    all_nan = {u: g["insulin_u"].isna().all() for u, g in raw.groupby("user_id")}
    assert any(all_nan.values()) and not all(all_nan.values())
