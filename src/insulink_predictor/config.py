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
    glucose_lags_min: list[int] = [0, 5, 10, 15, 30, 45, 60, 90, 120]
    roll_windows_min: list[int] = [30, 60, 120]
    steps_windows_min: list[int] = [15, 30, 60]
    use_carbs: bool = True
    use_insulin: bool = True
    use_hr: bool = True
    use_weather: bool = True
    cob_tau_min: float = 45.0
    iob_tau_min: float = 55.0
    # Insulin/carb ACTIVITY (rate of action) kernels: t·exp(−t/τ), peak at τ.
    # Activity drives the near-term glucose change; IOB/COB stock does not.
    ins_activity_tau_min: float = 55.0
    carb_activity_tau_min: float = 35.0
    # Therapy features: express COB/IOB in glucose-equivalent mg/dL using each
    # user's ISF (correction factor) and ICR (carb ratio) from user_settings.
    use_therapy: bool = True
    default_isf: float = 40.0  # mg/dL per 1U, fallback when a user has no setting
    default_icr: float = 12.0  # g carbs per 1U, fallback

    # Model the CHANGE over persistence (target = y_{t+h} − g_t) instead of the
    # absolute level. The regularized learner shrinks the delta toward ~0 when
    # there's no signal, so quiet periods fall back to persistence instead of
    # adding noise; deviations are reserved for real excursions.
    predict_delta: bool = True


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


class PostgresConfig(BaseModel):
    """Connection + ingestion settings for the real PostgreSQL source (§7).

    Secrets never live in YAML. Provide credentials via env: either a full
    ``GF_PG__DSN`` / ``DATABASE_URL``, or ``GF_PG__PASSWORD`` etc.
    """

    dsn: str | None = None  # full SQLAlchemy URL; overrides the parts below
    host: str = "localhost"
    port: int = 5432
    database: str = "insulink"
    user: str = "postgres"
    password: str = ""
    sslmode: str = "prefer"
    db_schema: str = "public"

    # Confirmed source conventions (kept overridable; "auto" falls back to detection):
    # recorded_at is unix epoch milliseconds, glucose is mg/dL.
    ts_unit: str = "ms"  # ms | s | auto
    source_glucose_unit: str = "mg/dL"  # mg/dL | mmol/L | auto
    local_tz: str = (
        "Europe/Berlin"  # for ts_local (circadian); per-user tz is future work
    )
    only_compliant: bool = True  # users.compliant filter

    def url(self) -> str:
        """Build a SQLAlchemy URL from ``dsn``/``DATABASE_URL`` or the parts."""
        import os

        dsn = self.dsn or os.environ.get("DATABASE_URL")
        if dsn:
            # normalise the common postgres:// prefix to the psycopg3 driver
            return dsn.replace("postgresql://", "postgresql+psycopg://").replace(
                "postgres://", "postgresql+psycopg://"
            )
        pw = self.password or os.environ.get("GF_PG__PASSWORD", "")
        return (
            f"postgresql+psycopg://{self.user}:{pw}@{self.host}:{self.port}/"
            f"{self.database}?sslmode={self.sslmode}"
        )


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
    pg: PostgresConfig = PostgresConfig()

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
