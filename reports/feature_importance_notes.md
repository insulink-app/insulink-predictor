# Feature-Importance Findings (Phase 2)

Source: `feature_importance.csv` / `.png` (mean gain across horizons @30/@60),
full config (6 users, 21 days). LightGBM vs persistence on the chronological
test split:

| model | @30 RMSE | @60 RMSE | skill@30 | skill@60 |
|---|---|---|---|---|
| persistence | 9.37 | 13.32 | 0.000 | 0.000 |
| **lgbm** | **7.42** | **11.25** | **+0.208** | **+0.156** |

## Which context features actually carry (ROADMAP §Phase 2 expectation)

- **Recent glucose dominates:** `lag_0` (current value) ≈ 45% gain, then
  `roll30_max`, `lag_5`, `roll30_mean` — the model is persistence *plus* the
  short-term trajectory shape.
- **Context that matters, as predicted:** `cob` (carbs-on-board, ~4.5%),
  `hour_sin` (time-of-day / circadian, ~4.3%) and `time_since_meal` (~3.8%) are
  the strongest non-glucose signals — i.e. **time-of-day and time-since-meal are
  high**, exactly the roadmap's hypothesis. `iob` and `time_since_activity` add
  smaller but real contributions.
- **Weather is low, as predicted:** `weather_now` sits at ~1.1% (rank ~12) —
  measured, not believed (§3). It is kept as an optional feature but carries
  little signal in this data.

## Takeaway

The skill comes from combining persistence with **short-term rate/shape** and
**meal/circadian context** — the excursion-relevant features. This is the wedge
the roadmap bets on (§0): value lives around events, not in the flat resting
state.
