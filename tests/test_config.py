"""Phase 0 — config loads from YAML and derived values are correct."""

from __future__ import annotations

from glucose_forecast.config import Config, load_config


def test_yaml_loads_and_defaults():
    cfg = load_config()  # reads config/config.yaml
    assert cfg.glucose_unit == "mg/dL"
    assert cfg.grid_minutes == 5
    assert cfg.horizons_min == [30, 60]


def test_horizons_steps_derived():
    cfg = Config()  # defaults
    # 30 min / 5 min = 6 steps, 60 / 5 = 12 steps
    assert cfg.horizons_steps == [6, 12]


def test_grid_freq_and_interp_steps():
    cfg = Config(grid_minutes=5, max_interp_gap_min=15)
    assert cfg.grid_freq == "5min"
    assert cfg.max_interp_steps == 3


def test_unit_and_interval_are_configurable_not_hardcoded():
    # §7: unit & interval come from config, so alternatives must be expressible.
    cfg = Config(grid_minutes=15, horizons_min=[30, 60, 90])
    assert cfg.horizons_steps == [2, 4, 6]
    assert cfg.grid_freq == "15min"
