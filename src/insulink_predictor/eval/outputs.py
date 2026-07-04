"""Structured output paths for report **diagrams** (images).

Diagrams are organised as ``reports/<category>/<name>[_<params>]_<timestamp>.<ext>``
so every run keeps a versioned, self-describing file instead of overwriting the
last one. ``params`` is a short slug identifying the model / key settings
(e.g. ``"lgbm-tuned"``, ``"knn-k300"``, ``"10fold"``).

Tables (CSV/MD/JSON) deliberately stay at the ``reports/`` root — this helper is
for image diagrams only.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def diagram_path(
    reports_dir,
    category: str,
    name: str,
    params: str = "",
    ext: str = "png",
) -> Path:
    """Return ``reports/<category>/<name>[_<params>]_<YYYYmmdd-HHMMSS>.<ext>``.

    Creates the category subfolder if needed. The timestamp is local wall-clock so
    repeated runs of the same diagram accumulate as versioned files.
    """
    folder = Path(reports_dir) / category
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = "_".join(p for p in (name, params, stamp) if p)
    return folder / f"{slug}.{ext}"


def table_path(reports_dir, name: str, ext: str = "csv") -> Path:
    """Return ``reports/tables/<name>.<ext>`` (canonical latest table, no timestamp).

    Tables are the current metrics, not versioned snapshots, so they keep fixed
    names. ``write_table`` writes both the ``.csv`` and a sibling ``.md`` here.
    """
    folder = Path(reports_dir) / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.{ext}"
