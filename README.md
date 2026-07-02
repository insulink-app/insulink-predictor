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
uv run gf synth       # generate synthetic data -> align -> validate -> data/processed/grid.parquet
```

Later phases add `gf features`, `gf train-*`, `gf eval*`.

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
