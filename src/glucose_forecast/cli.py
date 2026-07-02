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


if __name__ == "__main__":  # pragma: no cover
    app()
