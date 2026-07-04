# Trustworthy skill + what actually moves it (walk-forward backtest)

## Why this exists

`gf eval` reports one 80/20 chronological split — a **single draw**. Its test
window is the last 20 % of each user, an unusually hard stretch, so the number
reads low *and* is noisy: it can flip a real change into an apparent regression at
one horizon from split luck alone.

`gf backtest` runs **rolling-origin, expanding-window** evaluation: N folds per
user, each training on all earlier data and testing on the next contiguous slice,
with a `max(horizon)`-row **embargo** so no training target reaches into a test
window. It reports **skill mean ± std and the worst fold**, on synth or real data
(`--source db`). A change is believed only when its *paired* per-fold lift (same
folds, same persistence denominator) clears the fold-to-fold noise.

## Trustworthy numbers (5 folds, expanding)

| | single-split `gf eval` (synth) | walk-forward synth | walk-forward **real DB** |
|---|---|---|---|
| skill @30 | 0.175 | 0.203 ± 0.028 | **0.214 ± 0.036** |
| skill @60 | 0.128 | 0.168 ± 0.037 | **0.226 ± 0.065** |
| worst fold | — | 0.174 / 0.125 | 0.183 / 0.157 |

Real data = one T1 user, ~26k 5-min buckets, Apr–Jul 2026, ISF 35 / ICR 15. The
model **beats persistence on every fold and horizon** on real data, and skill is
*higher* at 60 min than 30 min (persistence degrades faster there on real CGM).

## The headline lesson: synth gains did NOT transfer

Two causal features were prototyped and, on synth, looked like clear wins. On the
**real** data they slightly HURT — so they are **off by default**.

| change | synth Δskill @30/@60 | **real DB** Δskill @30/@60 | default |
|---|---|---|---|
| per-user time-of-day baseline (`tod_*`) | +0.008 / +0.002 (5/5, 3/5) | **−0.004 / −0.002** (2/5, 1/5) | **off** |
| + horizon physio forecast (`*_delta_h`) | +0.011 / +0.006 (4/5, 4/5) | **−0.004 / −0.005** (0/5, 1/5) | **off** |

Why they overfit synth:
- `physio_delta` forward-integrates a fixed bi-exponential kernel — which *is*
  synth's generative kernel, so on synth it reconstructs the answer key. Real
  absorption differs, so the fixed kernel adds noise.
- `tod_baseline` shines on synth's clean per-user sinusoidal circadian; with one
  noisy real user its expanding time-of-day mean is mostly noise.

This is the entire point of backtesting on real data: **a synth win is a
hypothesis, not a result.** Re-enable only after (a) more real users and (b)
fitting/learning the response kernel from real data.

## Also rejected (on synth — never promoted)

| change | synth Δskill @30/@60 | verdict |
|---|---|---|
| excursion sample-weighting (α=1) | −0.026 / −0.033 (0/5) | rejected — over-reacts; delta target already handles excursions |
| monotone therapy constraints | −0.001 / −0.004 | off (safety lever for real data) |
| huber objective | −0.000 / −0.010 | off (tightens variance, not skill) |

## Current committed model

The **original Phase-2 feature set** (no `tod_*`, no `*_delta_h`) is best on real
data and is the default. All the above are wired as off-by-default
`cfg.features.*` / `cfg.model.*` levers, ready to re-test.

## Reproduce

```bash
uv run gf backtest                              # synth, mean ± std
uv run --env-file .env gf backtest --source db  # real data
uv run --env-file .env gf backtest --source db --folds 8
```

A/B comparisons use `eval.backtest.backtest_models` + `paired_delta` (see the
module docstring) with variants built by toggling `cfg.features.*`.
