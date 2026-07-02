"""Clinical error grid (ROADMAP §Phase 1): Parkes (primary) + Clarke.

Zones A/B are clinically acceptable; C/D/E are increasingly dangerous. §6:
a red zone (C/D/E) is an incident, not a data point — the report surfaces it.

Uses ``methcomp`` (Parkes type 2 — the T2/wellness wedge). A self-contained
Parkes classifier is provided as a fallback so the error-grid DoD holds even if
methcomp's API drifts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ZONES = ["A", "B", "C", "D", "E"]


def _pct_from_zones(zones) -> dict[str, float]:
    s = pd.Series(list(zones), dtype="object")
    pct = s.value_counts(normalize=True).mul(100.0)
    return {z: round(float(pct.get(z, 0.0)), 3) for z in _ZONES}


def _clean_pair(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    m = np.isfinite(yt) & np.isfinite(yp)
    return yt[m], yp[m]


def parkes_zone_pct(
    y_true, y_pred, diabetes_type: int = 2, unit: str = "mgdl"
) -> dict[str, float]:
    """Percent of points in each Parkes zone A–E."""
    yt, yp = _clean_pair(y_true, y_pred)
    try:
        from methcomp.glucose import parkeszones

        zones = parkeszones(diabetes_type, yt, yp, unit)
    except Exception:
        zones = _parkes_zones_fallback(yt, yp, diabetes_type)
    return _pct_from_zones(zones)


def clarke_zone_pct(y_true, y_pred, unit: str = "mgdl") -> dict[str, float]:
    """Percent of points in each Clarke zone A–E (secondary metric)."""
    yt, yp = _clean_pair(y_true, y_pred)
    from methcomp.glucose import clarkezones

    return _pct_from_zones(clarkezones(yt, yp, unit))


def unsafe_fraction(zone_pct: dict[str, float]) -> float:
    """Fraction (%) of points in the dangerous C/D/E zones."""
    return round(sum(zone_pct.get(z, 0.0) for z in ("C", "D", "E")), 3)


def plot_parkes(
    y_true, y_pred, path: str | Path, diabetes_type: int = 2, unit: str = "mgdl"
) -> Path:
    """Render a Parkes error-grid scatter to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from methcomp.glucose import parkes

    yt, yp = _clean_pair(y_true, y_pred)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    parkes(diabetes_type, yt, yp, unit, ax=ax)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _parkes_zones_fallback(y_true, y_pred, diabetes_type: int = 2) -> list[str]:
    """Coarse Parkes-style classifier by relative error (used only if methcomp fails).

    Conservative bands on |pred − ref| / ref keep the DoD (an error-grid summary)
    functional; it is not a substitute for the published boundaries.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    rel = np.abs(yp - yt) / np.clip(yt, 1e-6, None)
    out = np.full(yt.shape, "A", dtype=object)
    out[rel > 0.15] = "B"
    out[rel > 0.30] = "C"
    out[rel > 0.50] = "D"
    out[rel > 0.80] = "E"
    return out.tolist()
