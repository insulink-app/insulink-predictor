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

## Output layout

Every diagram is written under `reports/<category>/` with a self-describing,
timestamped filename via `insulink_predictor.eval.outputs.diagram_path`:

```
reports/cdf/<name>_<model-params>_<YYYYmmdd-HHMMSS>.png
```

so repeated runs accumulate as versioned files instead of overwriting. (Tables —
CSV/MD — stay at the `reports/` root; only image diagrams are foldered.)

## Auto-tuning

The three model-comparison CDFs **re-tune LGBM and k-NN to the loaded dataset**
before computing anything (`evaluation/_tuning.py`):

- **LGBM** — random search (`LGBM_TRIALS`, default 40) over a wide space, each trial
  early-stopped on a chronological **validation** slice of the training data.
- **k-NN** — best tricube neighborhood `k` from `KNN_GRID` (euclidean metric +
  tricube kernel is the established recipe; `k` is what moves with the data).

Selection is always on a validation slice, never the test set. In the walk-forward
scripts, tuning runs **once** on the earliest slice (before any test fold, so no
leakage), then each fold trains with those settings. Tune the search budget via the
constants at the top of `_tuning.py`.

## Scripts

All CDF scripts write to `reports/cdf/`:

| Script | Diagram name | What it shows |
|---|---|---|
| `cdf_model_comparison.py` | `model_comparison` | Persistence vs **auto-tuned** LGBM vs **auto-tuned** k-NN, single chronological 80/20 split, @30 & @60. |
| `cdf_walk_forward_oof.py` | `walk_forward_oof` | Same three models but **walk-forward out-of-sample** — every point predicted from its past (10 embargoed expanding folds); tuned once on the early slice. |
| `cdf_post_meal.py` | `post_meal` | Walk-forward OOF CDF restricted to **post-meal** rows (≤60 min after a logged meal) — the hard excursion tail — plus per-model post-meal skill. |
| `cdf_recursive_vs_direct.py` | `recursive_vs_direct` | Recursive 5-min stepping vs direct multi-horizon, **matched reduced features**, single split. Uses matched *default* LGBM params (an architecture control, not a tuning bake-off) + a step-0 feature-reconstruction sanity assert. |
| `cdf_recursive_full_oof.py` | `recursive_full_oof` | Recursive vs direct with the **full feature set**, walk-forward OOF, post-meal (anchors with no new meal/bolus). Matched default params. |

## Notes

- **k-NN neighborhood** (`KNN_K`) and **fold settings** (`N_FOLDS`, `TEST_SPAN`)
  are constants at the top of each script — edit to taste.
- The colors are the Okabe–Ito colorblind-safe palette (persistence = gray,
  LGBM = blue, k-NN/recursive = vermillion/green).
- These run on **real DB data only** — like the rest of the pipeline, there is no
  synthetic-data path.
- The canonical skill-with-error-bars number is still `gf backtest`; these CDFs
  are for seeing the *shape* of the error distribution.
