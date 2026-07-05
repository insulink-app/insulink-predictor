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

    # Per-user circadian baseline: the causal (past-only) expanding mean glucose at
    # this user's local time-of-day bin, plus the current deviation from it. The
    # shared hour_sin/cos can only fit the *average* circadian phase; each user
    # peaks at a different hour, so that structure averages out globally (see
    # feature_importance_notes). This hands the global model each user's own
    # time-of-day norm — the per-user circadian moat, available before any
    # per-user model exists.
    #
    # OFF by default: lifts synth skill (+0.008 @30, 5/5 folds) but slightly HURTS
    # the real single-user data (−0.004, 2/5 folds) — synth's clean sinusoidal
    # per-user circadian doesn't match one noisy real user. Re-test with
    # `gf backtest --source db` on a multi-user real set before enabling.
    use_tod_baseline: bool = False
    tod_bin_min: int = 30  # width of the time-of-day bin (minutes)

    # Horizon-specific physiological forecast: the expected glucose *change* over
    # the next h minutes from carbs/insulin already on board, obtained by
    # forward-integrating a bi-exponential response kernel over currently-known
    # doses (strictly causal — only past doses enter). Unlike cob/iob (stock now)
    # and *_activity (rate now), this speaks the delta target's language directly:
    # "how much rise/drop is still coming in the next 30/60 min." Emitted as shared
    # columns carb_delta_<h>/ins_delta_<h> for every horizon; both horizon models
    # see all of them (the tree picks the relevant lead).
    #
    # OFF by default: top feature and biggest post-meal lift on synth (+0.005…),
    # but slightly HURTS the real single-user data (−0.005 @60, 0–1/5 folds) — the
    # fixed bi-exp kernel matches synth's generator but not real absorption. Fit the
    # kernel to real data (or learn the response) before enabling.
    use_physio_delta: bool = False
    carb_resp_rise_min: float = 20.0
    carb_resp_decay_min: float = 90.0
    ins_resp_rise_min: float = 20.0
    ins_resp_decay_min: float = 70.0

    # Daily activity context: yesterday's *completed* step/distance totals (plus a
    # 3-day average), broadcast onto every bucket and strictly causal. Exercise
    # raises insulin sensitivity for 24-48h, so an active prior day blunts the
    # post-meal excursion. Consumes the DB daily STEPS/DISTANCE aggregates;
    # degrades cleanly (feature absent) when they are (e.g. synth).
    #
    # OFF by default: paired walk-forward on the real DB user shows it HELPS the
    # post-meal window (+0.002…+0.007, 8/10 folds @30) but slightly HURTS the
    # overall window (−0.001…−0.005) — extra variance on the 98% non-post-meal
    # rows. Net ~neutral. Enable when optimizing post-meal specifically, and
    # re-test with `gf backtest` on multi-user data.
    use_daily_activity: bool = False

    # Model the CHANGE over persistence (target = y_{t+h} − g_t) instead of the
    # absolute level. The regularized learner shrinks the delta toward ~0 when
    # there's no signal, so quiet periods fall back to persistence instead of
    # adding noise; deviations are reserved for real excursions.
    predict_delta: bool = True


class ModelConfig(BaseModel):
    """Learner-side knobs, A/B-able through walk-forward backtesting.

    All default to *off*, so the committed defaults reproduce the Phase-2 model.
    Paired walk-forward comparison on the synthetic population showed **no lift**
    for any of these (excursion-weighting actively hurt — the delta target already
    handles excursions and up-weighting them causes over-reaction), so they stay
    off. They are kept as ready-to-retest levers for **real** data, where the noise
    and robustness trade-offs differ (huber/monotone are safety-oriented). Re-run
    ``gf backtest`` on real data before enabling any of them.
    """

    # Emphasize excursion rows in training. Skill is RMSE-based and persistence
    # RMSE is dominated by large post-meal/activity swings, so winning those rows
    # disproportionately raises skill. weight = 1 + alpha·min(|Δ|/scale, cap),
    # Δ = y_{t+h} − g_t. alpha=0 disables; scale = std(Δ) on the training rows.
    excursion_weight_alpha: float = 0.0
    excursion_weight_cap: float = 3.0

    # Physiologically-signed monotone constraints on the therapy channels (with the
    # delta target): more carbs-on-board can only raise the forecast, more
    # insulin-on-board can only lower it. Curbs over-fitting on excursions and is a
    # safety property for the error grid.
    use_monotone: bool = False

    # Point-forecast objective. "huber" is robust to CGM artifacts (calibration
    # jumps, compression lows) but discounts the large excursions skill rewards —
    # so it is a genuine trade-off, decided by backtest, not assumed.
    objective: str = "l2"  # l2 | huber
    huber_delta: float = 8.0  # mg/dL; residual scale where huber turns linear

    # Re-tune LGBM hyperparameters to the training data on every train_lgbm call
    # (random search + early stopping on a chronological validation slice), instead
    # of the frozen `_TUNED_PARAMS`. Adapts to whatever data is loaded; costs a
    # tuning pass per train (per fold in `gf backtest`). Set False for the fast
    # frozen-param path (used by the test suite for speed + determinism).
    auto_tune: bool = True
    tune_trials: int = 30
    tune_val_fraction: float = 0.25


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
    model: ModelConfig = ModelConfig()
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
