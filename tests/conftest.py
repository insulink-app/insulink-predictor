"""Shared pytest fixtures — small, fast synthetic data for the whole suite."""

from __future__ import annotations

import pytest

from glucose_forecast.config import Config, load_config


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Small config for fast tests (few users, few days)."""
    return Config(synth={"n_users": 3, "days": 7, "seed": 0})


@pytest.fixture(scope="session")
def raw(cfg: Config):
    from glucose_forecast.data.synth import generate

    return generate(cfg)


@pytest.fixture(scope="session")
def grid(cfg: Config, raw):
    from glucose_forecast.data.align import align

    return align(raw, cfg)


@pytest.fixture(scope="session")
def full_cfg() -> Config:
    """The real project config (config/config.yaml)."""
    return load_config()
