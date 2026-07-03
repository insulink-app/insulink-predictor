# evaluation/ — standalone analysis scripts

Ad-hoc evaluation scripts that render **error-CDF diagrams** comparing models on
the real DB data. They are *not* part of the `insulink_predictor` package or the
test suite — run them by hand when you want the plots. Each writes a PNG to
`reports/` and prints a summary table.

## Running

They read the real DB (via `insulink_predictor.data.load.load_raw`), so pass the
env file with the DB credentials, from the repo root:

```bash
uv run --env-file .env python evaluation/cdf_model_comparison.py
```

An empirical CDF of absolute error reads: **up-and-to-the-left = better** (more
predictions with small error). Persistence is the reference; a model whose curve
sits above/left of it is beating persistence across the whole distribution, not
just on average.

## Scripts

| Script | Output PNG | What it shows |
|---|---|---|
| `cdf_model_comparison.py` | `reports/error_cdf_knn_vs_lgbm.png` | Persistence vs tuned LGBM vs best k-NN (euclidean + tricube), single chronological 80/20 split, @30 & @60. |
| `cdf_walk_forward_oof.py` | `reports/error_cdf_oof.png` | Same three models but **walk-forward out-of-sample** — every point predicted from its past (10 embargoed expanding folds, ~3× the test coverage). |
| `cdf_post_meal.py` | `reports/error_cdf_postmeal.png` | Walk-forward OOF CDF restricted to **post-meal** rows (≤60 min after a logged meal) — the hard excursion tail — plus per-model post-meal skill. |
| `cdf_recursive_vs_direct.py` | `reports/error_cdf_recursive.png` | Recursive 5-min stepping vs direct multi-horizon, **matched reduced features**, single split. Includes a step-0 feature-reconstruction sanity assert. |
| `cdf_recursive_full_oof.py` | `reports/error_cdf_recursive_full_oof.png` | Recursive vs direct with the **full feature set**, walk-forward OOF, post-meal (anchors with no new meal/bolus in the horizon). |

## Notes

- **k-NN neighborhood** (`KNN_K`) and **fold settings** (`N_FOLDS`, `TEST_SPAN`)
  are constants at the top of each script — edit to taste.
- The colors are the Okabe–Ito colorblind-safe palette (persistence = gray,
  LGBM = blue, k-NN/recursive = vermillion/green).
- To run on synthetic instead of DB, replace `align(load_raw(cfg), cfg)` with
  `align(generate(cfg), cfg)` (`from insulink_predictor.data.synth import generate`)
  and drop `--env-file .env`.
- The canonical skill-with-error-bars number is still `gf backtest --source db`;
  these CDFs are for seeing the *shape* of the error distribution.
