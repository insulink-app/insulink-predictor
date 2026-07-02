"""Real-loader transformation logic (no DB): unit/timestamp handling + assembly.

The SQLAlchemy/psycopg I/O is exercised against a live DB by `gf db-inspect` /
`gf load`; here we test the pure mapping that turns fetched tables into the RAW
contract and prove it flows through align() + the schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insulink_predictor.config import Config
from insulink_predictor.data.align import align
from insulink_predictor.data.load import (
    assemble_raw,
    detect_glucose_unit,
    detect_ts_unit,
    to_datetime_utc,
    to_mgdl,
)

_MS = 1000
_MIN_MS = 60 * _MS


def _epoch_ms(iso: str) -> int:
    return int(pd.Timestamp(iso).timestamp() * 1000)


def test_detect_ts_unit_ms_vs_s():
    assert detect_ts_unit(pd.Series([_epoch_ms("2024-06-01T00:00:00Z")])) == "ms"
    assert detect_ts_unit(pd.Series([1_717_200_000])) == "s"


def test_to_datetime_utc_ms():
    base = _epoch_ms("2024-06-01T08:00:00Z")
    out = to_datetime_utc(pd.Series([base]), "ms")
    assert str(out.iloc[0]) == "2024-06-01 08:00:00+00:00"


def test_glucose_unit_detection_and_conversion():
    assert detect_glucose_unit(pd.Series([80, 120, 160, 200.0])) == "mg/dL"
    assert detect_glucose_unit(pd.Series([4.5, 6.7, 8.9])) == "mmol/L"
    # mmol -> mg/dL (x18)
    conv = to_mgdl(pd.Series([5.0]), "mmol/L")
    assert abs(conv.iloc[0] - 90.09) < 0.1
    # mg/dL passes through unchanged
    assert to_mgdl(pd.Series([120.0]), "mg/dL").iloc[0] == 120.0


def _sample_tables(n_glucose=180):
    """One user: 15h of 5-min CGM, two boluses, HR samples, one training window."""
    uid = "11111111-1111-1111-1111-111111111111"
    base = _epoch_ms("2024-06-01T06:00:00Z")
    g = pd.DataFrame(
        {
            "user_id": uid,
            "recorded_at": [base + i * 5 * _MIN_MS for i in range(n_glucose)],
            "value": 120 + 30 * np.sin(np.linspace(0, 6, n_glucose)),  # mg/dL
        }
    )
    bolus = pd.DataFrame(
        {
            "user_id": uid,
            "recorded_at": [base + 60 * _MIN_MS, base + 300 * _MIN_MS],
            "carbohydrates": [45.0, 60.0],
            "insulin": [4.5, 6.0],
            "glucose": [140.0, 110.0],
            "carbohydrate_ratio": [10.0, 10.0],
            "insulin_type": ["rapid", "rapid"],
        }
    )
    sm = pd.DataFrame(
        {
            "user_id": uid,
            "recorded_at": [base + i * 30 * _MIN_MS for i in range(6)],
            "type": ["heart_rate", "steps", "heart_rate", "unknown_metric", "steps", "heart_rate"],
            "value": [72.0, 400.0, 88.0, 1.0, 250.0, 95.0],
        }
    )
    st = pd.DataFrame(
        {
            "user_id": uid,
            "started_at": [base + 120 * _MIN_MS],
            "ended_at": [base + 160 * _MIN_MS],
            "type": ["run"],
            "distance": [5000.0],
        }
    )
    return {"glucose_entries": g, "bolus_entries": bolus, "sport_measurements": sm, "sport_trainings": st}


def test_assemble_raw_produces_contract_columns():
    cfg = Config()
    raw = assemble_raw(_sample_tables(), cfg)
    required = {
        "user_id", "ts_utc", "ts_local", "glucose_mgdl", "meal_flag",
        "carbs_g", "insulin_u", "steps", "activity_flag", "hr", "weather_temp",
    }
    assert required.issubset(raw.columns)
    assert raw["ts_utc"].dt.tz is not None            # tz-aware UTC
    assert raw["ts_local"].dt.tz is None              # naive local wall-clock
    # channels landed: 180 CGM readings + 2 bolus SMBG folded in
    assert raw["glucose_mgdl"].notna().sum() == 182
    assert (raw["carbs_g"] > 0).sum() == 2            # two meals
    assert raw["insulin_u"].notna().sum() == 2
    assert raw["hr"].notna().sum() == 3               # 3 heart_rate rows
    assert raw["activity_flag"].sum() >= 1            # training expanded to buckets
    # the unknown_metric type is dropped (not hr/steps)
    assert raw["hr"].notna().sum() + raw["steps"].notna().sum() == 5


def test_assembled_raw_flows_through_align_and_schema():
    cfg = Config()
    raw = assemble_raw(_sample_tables(), cfg)
    grid = align(raw, cfg)  # validates against the pandera contract internally
    # regular 5-min grid, carbs/insulin/hr survived the alignment
    diffs = grid["ts_utc"].diff().dropna().dt.total_seconds() / 60
    assert (diffs == cfg.grid_minutes).all()
    assert (grid["carbs_g"] > 0).sum() >= 2
    assert grid["insulin_u"].notna().sum() >= 2
    assert grid["activity_flag"].any()


def test_empty_tables_yield_empty_raw():
    cfg = Config()
    raw = assemble_raw({}, cfg)
    assert raw.empty
    assert list(raw.columns) == [
        "user_id", "ts_utc", "ts_local", "glucose_mgdl", "meal_flag",
        "carbs_g", "insulin_u", "steps", "activity_flag", "hr", "weather_temp",
    ]
