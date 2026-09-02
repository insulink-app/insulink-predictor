"""Paired ablation of the optional channels on the real DB — the switch-on test.

Several channels ship OFF because they could not be shown to help *yet*
(`use_hr_dynamics`, `use_gps`, `use_tod_baseline`, `use_physio_delta`,
`use_daily_activity`). Each carries the same instruction: re-measure, and flip it
when it wins. This is that measurement, so the instruction is a command rather
than a promise.

Every arm runs on **identical folds** against the **identical persistence
denominator**, so the per-fold difference is paired and cancels the fold-to-fold
noise that swamps these effects (they are worth thousandths of skill; the
fold-to-fold spread is hundredths).

Two windows are reported, because they answer different questions:

- **era** — only the stretch where the channel actually has data. "Is there signal
  in this channel at all?"
- **production** — the whole history, which is what `serve/training.py` trains on.
  "Does the signal survive being diluted by the years before the channel existed?"

A channel is worth switching on when it wins the PRODUCTION table. Winning only
the era table means the signal is real but the training set is not ready yet — for
heart rate that resolves itself as the band accumulates history; see
`reports/tables/wearable_channel_notes.md`.

    uv run --env-file .env python evaluation/channel_ablation.py
    uv run --env-file .env python evaluation/channel_ablation.py hr        # only HR arms
"""

from __future__ import annotations

import sys

import pandas as pd

from insulink_predictor.config import load_config
from insulink_predictor.data.align import align
from insulink_predictor.data.load import load_raw
from insulink_predictor.eval.backtest import backtest_models, paired_delta, summarize
from insulink_predictor.eval.harness import build_supervised
from insulink_predictor.models.lgbm import make_pred_fn, train_lgbm

N_FOLDS = 8
BASELINE = "committed"

# name -> the feature flags it overrides, relative to the committed config.
ARMS: dict[str, dict] = {
    "committed": {},
    "no_hr": {"use_hr": False},
    "hr_dynamics": {"use_hr_dynamics": True},
    "gps": {"use_gps": True},
    "hr_dynamics+gps": {"use_hr_dynamics": True, "use_gps": True},
}

# Which channel each arm needs, so the "era" window can be found from the data.
NEEDS = {"hr_dynamics": "hr", "gps": "lat", "hr_dynamics+gps": "lat", "no_hr": "hr"}


def era_start(grid: pd.DataFrame, arms: list[str]) -> pd.Timestamp:
    """Earliest timestamp at which every channel under test already reports."""
    starts = [
        grid.loc[grid[col].notna(), "ts_utc"].min()
        for arm in arms
        if (col := NEEDS.get(arm)) and col in grid.columns and grid[col].notna().any()
    ]
    return max(starts) if starts else grid["ts_utc"].min()


def run(grid: pd.DataFrame, arms: list[str], test_span: float, label: str) -> None:
    cfg = load_config()
    cfg.model.auto_tune = False  # frozen params: deterministic, and the arms are paired

    def lgbm(train, cols, _cfg):
        return make_pred_fn(
            train_lgbm(train, cols, cfg), cols, cfg.features.predict_delta
        )

    frames = []
    for arm in arms:
        base = load_config().features.model_dump()
        cfg.features = type(cfg.features)(**{**base, **ARMS[arm]})
        supervised, feature_cols = build_supervised(cfg, grid)
        frames.append(
            backtest_models(
                cfg,
                {arm: lgbm},
                sup=supervised,
                feature_cols=feature_cols,
                n_folds=N_FOLDS,
                test_span=test_span,
            )
        )
        print(f"  {arm:18s} {len(feature_cols)} features")

    per_fold = pd.concat(frames, ignore_index=True)
    print(f"\n--- {label}: skill vs persistence ---")
    print(summarize(per_fold).to_string(index=False))
    print(f"\n--- {label}: paired lift over '{BASELINE}' ---")
    print(paired_delta(per_fold, BASELINE).to_string(index=False))
    print()


def main() -> None:
    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    arms = [a for a in ARMS if a == BASELINE or not wanted or wanted in a]

    cfg = load_config()
    grid = align(load_raw(cfg), cfg)
    print(f"grid: {len(grid)} buckets, {grid['user_id'].nunique()} user(s)")
    for col in ("hr", "lat"):
        if col in grid.columns:
            print(f"  {col}: {grid[col].notna().mean():.1%} of all buckets")

    start = era_start(grid, arms)
    era = grid[grid["ts_utc"] >= start].reset_index(drop=True)
    print(f"\narms: {arms}\nera starts {start} ({len(era)} buckets)\n")

    run(era, arms, 0.5, "ERA (is there signal at all?)")
    # Same tail, but every arm trains on the whole history the way production does.
    run(
        grid, arms, len(era) / len(grid) * 0.5, "PRODUCTION (does it survive dilution?)"
    )


if __name__ == "__main__":
    main()
