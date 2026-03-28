from __future__ import annotations

import typer

from mrrpropy import __version__

app = typer.Typer(
    help="Utilities for the mrrpropy scientific processing package.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)
