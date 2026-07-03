# Feature-Importance Findings (Phase 2)

Source: `feature_importance.csv` / `.png` (mean gain across horizons @30/@60),
full config (6 users, 21 days). LightGBM vs persistence on the chronological
test split:

| model | @30 RMSE | @60 RMSE | skill@30 | skill@60 |
|---|---|---|---|---|
| persistence | 7.90 | 11.08 | 0.000 | 0.000 |
| **lgbm** | **6.78** | **10.07** | **+0.141** | **+0.091** |

## Which context features actually carry (ROADMAP §Phase 2 expectation)

- **Recent glucose dominates:** `lag_0` (current value) ≈ 51% gain, then
  `roll30_max`, `lag_5` — the model is persistence *plus* the short-term
  trajectory shape.
- **Context that matters, as predicted:** `time_since_meal` (~3.9%), `cob`
  (carbs-on-board, ~3.0%) and `hour_sin`/`hour_cos` (time-of-day, ~2.3%/2.0%)
  are the strongest non-glucose signals — i.e. **time-since-meal and time-of-day
  are high**, exactly the roadmap's hypothesis. `iob` and `time_since_activity`
  add smaller but real contributions.
- **Weather is low, as predicted:** `weather_now` sits at ~1.2% (rank ~10) —
  measured, not believed (§3).
- **Therapy-scaled carbs help:** `cob_glucose` (COB × CSF, the expected mg/dL rise
  still on board, using each user's ISF/ICR from `user_settings`) ranks ~4th
  (~7%) — above raw `cob`. See the therapy A/B below.

Note: the *global* importance of the hour features is modest because each user's
circadian rhythm peaks at a different phase — a shared model can only capture the
average. That residual per-user circadian structure is exactly what Phase 4's
per-user models exploit (see `personalization_summary.csv`).

## Therapy features (user_settings ISF/ICR) — A/B

Wiring COB/IOB into glucose-equivalent mg/dL (`cob_glucose = cob·ISF/ICR`,
`iob_glucose = iob·ISF`) gives a **small, consistent lift at 30 min**
(~+0.005…+0.015 skill across seeds) and is roughly neutral at 60 min on the
synthetic population (ISF 25–55, ICR 8–18). It is enabled by default
(`features.use_therapy`) because it is physiologically principled and should help
more on real data, where per-user sensitivity varies more widely. The raw ISF/ICR
constants are per-user *identity*, so they live in **Phase-4 conditioning**
(`user_isf`/`user_icr` static features), not the shared base — keeping the
global-vs-personalized comparison clean.

## Takeaway

The skill comes from combining persistence with **short-term rate/shape** and
**meal/circadian context** — the excursion-relevant features. This is the wedge
the roadmap bets on (§0): value lives around events, not in the flat resting
state.
