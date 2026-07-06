"""Per-user tune job checks — the DB layer is stubbed, so no live database and no
(slow) tuning runs here: they exercise the mapping, the thin-data guard, and the
atomic write."""

from __future__ import annotations

import joblib
import pandas as pd

from serve import training


def test_params_by_step_maps_to_nearest_horizon():
    tuned = {30: {"num_leaves": 44}, 60: {"num_leaves": 218}}
    pbs = training._params_by_step(tuned, grid_minutes=5, max_step=12)
    assert pbs[1] is tuned[30]  # 5 min  -> 30
    assert pbs[6] is tuned[30]  # 30 min -> 30
    assert pbs[9] is tuned[30]  # 45 min ties -> 30 (first sorted)
    assert pbs[10] is tuned[60]  # 50 min -> 60
    assert pbs[12] is tuned[60]  # 60 min -> 60


def test_params_by_step_passes_through_untunable_horizon():
    pbs = training._params_by_step(
        {30: {"a": 1}, 60: None}, grid_minutes=5, max_step=12
    )
    assert pbs[12] is None  # too thin to tune -> falls back to _DEFAULT_PARAMS


def test_atomic_dump_roundtrip(tmp_path):
    p = tmp_path / "sub" / "m.joblib"
    training._atomic_dump({"x": 1}, p)
    assert p.exists() and joblib.load(p) == {"x": 1}
    assert not list(p.parent.glob("*.tmp"))  # temp cleaned up


def test_retrain_all_skips_thin_users(monkeypatch, tmp_path):
    """A user with too little CGM is skipped (not tuned) and nothing is written."""
    cfg = training.load_config()
    monkeypatch.setattr(training, "distinct_user_ids", lambda *a, **k: ["u1"])
    step_ms = 5 * 60 * 1000
    ge = pd.DataFrame(
        {
            "user_id": ["u1"] * 60,
            "recorded_at": [1_700_000_100_000 + i * step_ms for i in range(60)],
            "value": [110.0 + (i % 10) for i in range(60)],
        }
    )
    monkeypatch.setattr(
        training, "fetch_tables", lambda *a, **k: {"glucose_entries": ge}
    )

    summary = training.retrain_all(cfg, engine=object(), models_dir=tmp_path)

    assert summary["n_users"] == 1
    assert summary["skipped"] == 1 and summary["trained"] == 0
    assert summary["results"][0]["status"] == "skipped"
    assert not list(tmp_path.glob("*.joblib"))  # nothing promoted
