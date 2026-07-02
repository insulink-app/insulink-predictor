"""Typed configuration (ROADMAP §1: pydantic-settings + YAML, no magic).

The YAML file is the single source of truth for units, grid resolution and
feature flags. Nothing unit- or interval-related is hardcoded elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


class FeatureConfig(BaseModel):
    glucose_lags_min: list[int] = [0, 5, 10, 15, 30, 45, 60]
    roll_windows_min: list[int] = [30, 60]
    steps_windows_min: list[int] = [15, 30, 60]
    use_carbs: bool = True
    use_insulin: bool = True
    use_hr: bool = True
    use_weather: bool = True
    cob_tau_min: float = 45.0
    iob_tau_min: float = 55.0


class EventConfig(BaseModel):
    meal_trigger: bool = True
    glucose_rate_threshold: float = 1.5
    activity_trigger: bool = True
    post_event_window_min: int = 60


class SynthConfig(BaseModel):
    n_users: int = 6
    days: int = 21
    seed: int = 42
    meals_per_day: int = 3
    insulin_user_fraction: float = 0.5
    dropout_prob: float = 0.02
    gap_events_per_day: float = 0.5


class SplitConfig(BaseModel):
    test_fraction: float = 0.2
    heldout_users: list[str] = Field(default_factory=list)


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")


class MLflowConfig(BaseModel):
    tracking_uri: str = "file:./mlruns"
    experiment: str = "insulink-predictor"
    enabled: bool = True


class Config(BaseSettings):
    """Top-level config. Loaded from YAML; env vars (prefix ``GF_``) may override."""

    glucose_unit: str = "mg/dL"
    grid_minutes: int = 5
    horizons_min: list[int] = [30, 60]
    max_interp_gap_min: int = 15

    features: FeatureConfig = FeatureConfig()
    event: EventConfig = EventConfig()
    synth: SynthConfig = SynthConfig()
    split: SplitConfig = SplitConfig()
    paths: PathsConfig = PathsConfig()
    mlflow: MLflowConfig = MLflowConfig()

    model_config = SettingsConfigDict(
        env_prefix="GF_", env_nested_delimiter="__", extra="ignore"
    )

    # --- derived ------------------------------------------------------------
    @property
    def horizons_steps(self) -> list[int]:
        """Horizons expressed in grid steps (e.g. 30/5 = 6, 60/5 = 12)."""
        return [h // self.grid_minutes for h in self.horizons_min]

    @property
    def grid_freq(self) -> str:
        """Pandas offset alias for the grid (e.g. ``"5min"``)."""
        return f"{self.grid_minutes}min"

    @property
    def max_interp_steps(self) -> int:
        """Longest gap (in steps) we linearly interpolate; longer stays a hole."""
        return self.max_interp_gap_min // self.grid_minutes


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load a :class:`Config` from YAML. Missing file → defaults."""
    path = Path(path)
    data: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    return Config(**data)
