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


def plot_feature_importance(fi: pd.DataFrame, path: str | Path, top: int = 20) -> Path:
    """Horizontal bar chart of the top-N features by gain%."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top_fi = fi.head(top).iloc[::-1]  # largest at the top of the chart
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top_fi))))
    ax.barh(top_fi["feature"], top_fi["gain_pct"], color="#3b7dd8")
    ax.set_xlabel("gain importance (%)")
    ax.set_title("LightGBM feature importance (mean across horizons)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_curves(examples: list[dict], path: str | Path) -> Path:
    """Plot predicted vs actual post-event trajectories.

    Each example: ``{title, minutes, predicted, actual}`` (minutes/predicted/actual
    are equal-length sequences over the forecast horizon).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, len(examples))
    horizon = int(examples[0]["minutes"][-1]) if examples else 0
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.8), squeeze=False, constrained_layout=True)
    for ax, ex in zip(axes[0], examples):
        m = [0, *ex["minutes"]]
        ax.plot(m, [ex["g0"], *ex["actual"]], "o-", color="#222", label="actual", ms=3)
        ax.plot(m, [ex["g0"], *ex["predicted"]], "s--", color="#d8543b", label="forecast", ms=3)
        ax.axhline(ex["g0"], color="#999", ls=":", lw=1, label="persistence")
        ax.set_title(ex["title"], fontsize=8)
        ax.set_xlabel("minutes ahead")
        ax.set_ylabel("glucose (mg/dL)")
        ax.legend(fontsize=7)
    fig.suptitle(f"{horizon}-min glucose forecast — predicted vs actual vs persistence", fontsize=11)
    fig.savefig(path, dpi=120)
    plt.close(fig)
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

                mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items() if v is not None}
                )
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
