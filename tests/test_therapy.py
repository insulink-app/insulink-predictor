"""Therapy features: glucose-equivalent COB/IOB from user_settings ISF/ICR.

These express carbs/insulin "on board" in mg/dL using each user's sensitivity
(CSF = ISF/ICR). Structural tests here; the prediction lift is measured in the
model A/B (documented in reports)."""

from __future__ import annotations

import numpy as np

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.synth import generate
from insulink_predictor.features.build import build_features


def _grid():
    cfg = Config(synth={"n_users": 2, "days": 6, "seed": 5})
    return align(generate(cfg), cfg), cfg


def test_therapy_features_present_when_isf_icr_available():
    grid, cfg = _grid()
    assert "isf" in grid.columns and "icr" in grid.columns
    feat, cols = build_features(grid, cfg)
    assert "cob_glucose" in cols and "iob_glucose" in cols


def test_cob_glucose_equals_cob_times_csf():
    grid, cfg = _grid()
    feat, _ = build_features(grid, cfg)
    csf = feat["isf"] / feat["icr"]
    assert np.allclose(feat["cob_glucose"], feat["cob"] * csf, equal_nan=True)


def test_therapy_degrades_when_settings_absent():
    grid, cfg = _grid()
    stripped = grid.drop(columns=["isf", "icr"])  # dataset without therapy settings
    feat, cols = build_features(stripped, cfg)
    assert (
        "cob_glucose" not in cols and "iob_glucose" not in cols
    )  # no crash, just absent


def test_flag_disables_therapy_features():
    grid, cfg = _grid()
    cfg2 = Config(synth=cfg.synth.model_dump(), features={"use_therapy": False})
    _, cols = build_features(grid, cfg2)
    assert "cob_glucose" not in cols
