# Glucose Forecasting Engine

Event-triggered glucose forecasting. The thesis (ROADMAP §0): **don't predict
glucose — beat Persistence** (`ŷ_{t+h} = g_t`). Value lives in the *excursions*
(post-meal rises, activity dips); every model is scored by a **skill-score**
`= 1 − RMSE_model / RMSE_persistence` and must beat persistence per horizon.

Positioning is **Wellness / "patterns & insights"**, not a medical device.
Product language is "patterns/insights", never "dosing prediction".

See [ROADMAP.md](ROADMAP.md) for the binding, phase-gated plan.

## Setup

```bash
uv sync
# macOS only: LightGBM needs the OpenMP runtime
brew install libomp
```

## Pipeline (`gf`)

```bash
uv run gf synth       # synthetic data -> align -> validate -> data/processed/grid.parquet
uv run gf features    # strictly-causal feature matrix
uv run gf eval        # LightGBM vs persistence (skill-score + Parkes error grid)
uv run gf eval-events        # post-meal vs global skill + 60-min trajectory plot
uv run gf eval-personalize   # personalized vs global per-user + cold-start check
```

### Real data (PostgreSQL)

Credentials come from the environment — never hardcoded or committed:

```bash
export DATABASE_URL="postgresql://user:pass@host:5432/insulink"   # or GF_PG__* vars
uv run gf db-inspect  # probe: ts unit, glucose unit, CGM cadence, type vocabularies
uv run gf load        # fetch -> align -> validate -> data/processed/grid.parquet
uv run gf load --since 2024-01-01
```

The schema has insulin + carbs → a **Type-1 insulin population**, so COB/IOB are
primary features (§7). They live in `nutrition_meals` (the app logs every dose as a
meal row: `carbs`, `bolus`, `glucose`); `bolus_entries` is the Dexcom-shaped table
and no controller writes it, so it is empty. Both are read, and a table the
deployment does not have is skipped with a warning.
`recorded_at` is unix-ms, `glucose` is mg/dL.
Run `gf db-inspect` to pin the `sport_measurements` / `events` type mappings.

## Tests

```bash
uv run pytest
```

## Layout

```
config/config.yaml              # units, grid, horizons, feature flags (single source of truth)
src/insulink_predictor/
  config.py                     # typed config (pydantic-settings + YAML)
  data/{schema,synth,align,load} # contract, generator, regular-grid alignment, real loader (stub)
  features/                     # strictly-causal feature builder (Phase 2)
  models/                       # persistence, lgbm, events, personalization
  eval/                         # chronological split, metrics, error grid, reporting
  cli.py                        # gf entrypoint
tests/                          # one module per phase
```

## Guardrails (ROADMAP §6, non-negotiable)

- Never a random split — always chronological (autocorrelation ⇒ leakage).
- Strict feature causality; a leakage test must exist and stay green.
- Persistence-gate: no phase is "done" unless its model beats persistence per horizon.
- No deep learning before Phase 5.
- Large CGM gaps are marked and excluded, never over-interpolated.
- A red error-grid zone is an incident, not a data point.
