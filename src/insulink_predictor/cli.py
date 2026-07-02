"""``gf`` — one Typer command per pipeline step (ROADMAP §2).

Phase 0 ships ``gf synth``. Later phases add ``features``, ``train-*`` and
``eval`` commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import load_config

app = typer.Typer(help="Glucose forecasting pipeline (gf).", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Glucose forecasting pipeline. Run a sub-command, e.g. ``gf synth``."""
    # Presence of a callback keeps commands as sub-commands even when only one
    # exists yet (Typer would otherwise collapse a lone command to the top level).


@app.command()
def synth(
    config: Path = typer.Option(
        Path("config/config.yaml"), help="Path to config.yaml."
    ),
    out: Optional[Path] = typer.Option(
        None, help="Output parquet (default: data/processed/grid.parquet)."
    ),
) -> None:
    """Generate synthetic data, align to the grid, validate the contract, write parquet."""
    from .data.align import align
    from .data.synth import generate

    cfg = load_config(config)
    raw = generate(cfg)
    grid = align(raw, cfg)  # validates against the schema internally

    out = out or cfg.paths.data_dir / "processed" / "grid.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(out)

    gaps = int(grid["sensor_gap"].sum())
    typer.echo(
        f"synth: {grid['user_id'].nunique()} users, {len(grid)} rows "
        f"({gaps} gap buckets, {gaps / len(grid):.1%}) -> {out}"
    )


@app.command()
def features(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config.yaml."),
    out: Optional[Path] = typer.Option(None, help="Output parquet (default: data/processed/features.parquet)."),
) -> None:
    """Build the strictly-causal feature matrix from the aligned grid."""
    from .eval.harness import build_supervised

    cfg = load_config(config)
    sup, cols = build_supervised(cfg)
    out = out or cfg.paths.data_dir / "processed" / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    sup.to_parquet(out)
    typer.echo(f"features: {len(sup)} rows, {len(cols)} causal features -> {out}")


@app.command(name="train-lgbm")
def train_lgbm_cmd(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config.yaml."),
) -> None:
    """Train LightGBM (direct multi-horizon) and evaluate vs persistence."""
    _run_eval(config)


@app.command()
def eval(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config.yaml."),
    baseline_only: bool = typer.Option(False, help="Evaluate only the persistence baseline."),
) -> None:
    """Evaluate LightGBM vs persistence: skill-score per horizon + Parkes error grid."""
    if baseline_only:
        from .eval.harness import run_baseline_eval

        cfg = load_config(config)
        res = run_baseline_eval(cfg, write=True)
        typer.echo("Persistence baseline (test split):")
        typer.echo(res["metrics"].to_string(index=False))
        typer.echo(f"\nReports written to {cfg.paths.reports_dir}/")
    else:
        _run_eval(config)


def _run_eval(config: Path) -> None:
    from .eval.harness import run_lgbm_eval

    cfg = load_config(config)
    res = run_lgbm_eval(cfg, write=True)
    typer.echo("Model comparison (test split):")
    typer.echo(res["comparison"].to_string(index=False))
    skills = {int(r["horizon_min"]): r["skill"] for _, r in res["lgbm_metrics"].iterrows()}
    verdict = "PASS" if all(v > 0 for v in skills.values()) else "FAIL"
    typer.echo(f"\nLGBM skill vs persistence: {skills}  ->  {verdict} (need > 0 each)")
    typer.echo(f"Reports + feature importance written to {cfg.paths.reports_dir}/")


@app.command(name="train-events")
def train_events_cmd(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config.yaml."),
) -> None:
    """Train the event-triggered curve models and evaluate post-event windows."""
    _run_event_eval(config)


@app.command(name="eval-events")
def eval_events_cmd(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config.yaml."),
) -> None:
    """Evaluate post-meal skill vs global skill; plot 60-min trajectory forecasts."""
    _run_event_eval(config)


def _run_event_eval(config: Path) -> None:
    from .eval.harness import run_event_eval

    cfg = load_config(config)
    res = run_event_eval(cfg, write=True)
    typer.echo("Event-triggered skill (global vs post-meal):")
    typer.echo(res["comparison"].to_string(index=False))
    g = res["global_skill"].set_index("horizon_min")["skill"]
    p = res["post_meal_skill"].set_index("horizon_min")["skill"]
    verdict = "PASS" if all(p[h] > g[h] for h in g.index) else "FAIL"
    typer.echo(f"\nPost-meal skill exceeds global skill? {verdict}")
    typer.echo(f"Curve forecast plot -> {cfg.paths.reports_dir}/event_curves.png")


@app.command(name="train-personalize")
def train_personalize_cmd(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config.yaml."),
) -> None:
    """Train global-conditioned + per-user residual models and evaluate per user."""
    _run_personalize_eval(config)


@app.command(name="eval-personalize")
def eval_personalize_cmd(
    config: Path = typer.Option(Path("config/config.yaml"), help="Path to config.yaml."),
) -> None:
    """Compare personalized vs global per-user skill; check cold-start on held-out users."""
    _run_personalize_eval(config)


def _run_personalize_eval(config: Path) -> None:
    from .eval.harness import run_personalize_eval

    cfg = load_config(config)
    res = run_personalize_eval(cfg, write=True)
    s = res["summary"]
    typer.echo("Personalization (per-user mean skill):")
    typer.echo(s.to_string(index=False))
    verdict = "PASS" if (s["personalized_mean_skill"] > s["global_mean_skill"]).all() else "FAIL"
    cs = res["coldstart"]
    typer.echo(f"\nPersonalized beats global (per-user)? {verdict}")
    typer.echo(f"Cold-start on {cs.get('n_heldout_users', 0)} held-out user(s) graceful? "
               f"{cs.get('graceful', 'n/a')}")
    typer.echo(f"Reports written to {cfg.paths.reports_dir}/")


if __name__ == "__main__":  # pragma: no cover
    app()
