"""Report writing + MLflow logging (ROADMAP §Phase 1).

Tables go to ``reports/`` as CSV **and** Markdown (human-readable evidence for the
DoD). MLflow logging is best-effort: a missing/broken tracker never fails a run
(tests must not need a server).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from ..config import Config


def _to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub Markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a metrics table as CSV and a sibling ``.md`` for easy reading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    path.with_suffix(".md").write_text(_to_markdown(df), encoding="utf-8")
    return path


@contextmanager
def mlflow_run(cfg: Config, run_name: str):
    """Best-effort MLflow run. Yields a logger with no-op fallbacks."""

    class _Logger:
        def __init__(self, active: bool):
            self.active = active

        def params(self, params: dict):
            if not self.active:
                return
            try:
                import mlflow

                mlflow.log_params(params)
            except Exception:
                self.active = False

        def metrics(self, metrics: dict):
            if not self.active:
                return
            try:
                import mlflow

                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
            except Exception:
                self.active = False

        def artifact(self, path: str | Path):
            if not self.active:
                return
            try:
                import mlflow

                mlflow.log_artifact(str(path))
            except Exception:
                self.active = False

    if not cfg.mlflow.enabled:
        yield _Logger(False)
        return

    try:
        import mlflow

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment)
        with mlflow.start_run(run_name=run_name):
            yield _Logger(True)
    except Exception:
        # Tracker unavailable — degrade to a no-op logger, don't fail the run.
        yield _Logger(False)
