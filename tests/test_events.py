"""Phase 3 DoD — post-meal skill beats global skill; a 60-min curve is produced."""

from __future__ import annotations

import numpy as np
import pytest

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.synth import generate
from insulink_predictor.eval.harness import run_event_eval
from insulink_predictor.models.events import detect_events, forecast_curve


@pytest.fixture(scope="module")
def event_result():
    cfg = Config(synth={"n_users": 4, "days": 12, "seed": 3}, mlflow={"enabled": False})
    grid = align(generate(cfg), cfg)
    return cfg, run_event_eval(cfg, df=grid, write=False)


def test_post_meal_skill_exceeds_global(event_result):
    _, res = event_result
    g = res["global_skill"].set_index("horizon_min")["skill"]
    p = res["post_meal_skill"].set_index("horizon_min")["skill"]
    for h in g.index:
        assert p[h] > g[h], f"@{h}min: post_meal {p[h]} not > global {g[h]}"


def test_post_meal_still_beats_persistence(event_result):
    _, res = event_result
    assert (res["post_meal_skill"]["skill"] > 0).all()


def test_curve_is_60_minutes_and_well_formed(event_result):
    cfg, res = event_result
    assert len(res["examples"]) >= 1
    ex = res["examples"][0]
    max_step = max(cfg.horizons_steps)
    assert len(ex["predicted"]) == max_step
    assert len(ex["minutes"]) == len(ex["predicted"]) == len(ex["actual"])
    assert ex["minutes"][-1] == 60  # forecasts the full next hour


def test_forecast_curve_shape(event_result):
    cfg, res = event_result
    test = res["test"]
    curve = forecast_curve(test.head(10), res["curve_models"], res["feature_cols"])
    assert curve.shape == (10, max(cfg.horizons_steps))


def test_curve_band_brackets_median():
    from insulink_predictor.eval.harness import run_curve

    cfg = Config(
        synth={"n_users": 3, "days": 12, "seed": 2},
        split={"test_fraction": 0.2},
        mlflow={"enabled": False},
    )
    grid = align(generate(cfg), cfg)
    res = run_curve(cfg, df=grid, horizon_min=30, n=2, band=True, lo=0.1, hi=0.9)
    assert res["examples"], "expected at least one banded curve"
    for ex in res["examples"]:
        lower = np.array(ex["lower"])
        upper = np.array(ex["upper"])
        pred = np.array(ex["predicted"])  # median
        assert (lower <= upper + 1e-6).all()              # no quantile crossing
        assert (lower <= pred + 1e-6).all() and (pred <= upper + 1e-6).all()
        assert (upper - lower).mean() > 0                 # band has real width


def test_detect_events_flags_meals():
    cfg = Config(synth={"n_users": 2, "days": 5, "seed": 1})
    grid = align(generate(cfg), cfg)
    ev = detect_events(grid, cfg)
    assert ev.loc[ev["meal_flag"], "event_meal"].all()
    assert ev["event_any"].sum() >= ev["event_meal"].sum()
