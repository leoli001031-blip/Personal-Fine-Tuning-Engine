"""Workspace initialization command."""

from __future__ import annotations

from pathlib import Path

import typer


DEFAULT_WORKSPACE = "user_default"
DEFAULT_HOME = ".pfe"


def _init_runtime_dirs(home: Path, workspace: str) -> list[Path]:
    paths = [
        home / "data",
        home / "adapters" / workspace,
        home / "cache",
        home / "logs",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def register_init_command(app: typer.Typer) -> None:
    """Attach the workspace initialization command to the root CLI app."""

    @app.command("init")
    def init(
        workspace: str = typer.Option(
            DEFAULT_WORKSPACE,
            "--workspace",
            help="Workspace name for adapter artifacts.",
        ),
        base_model: str = typer.Option(
            ...,
            "--base-model",
            help="Local model path or model id used as the initial base model.",
        ),
        home: Path = typer.Option(
            Path(DEFAULT_HOME),
            "--home",
            help="PFE runtime directory to create.",
        ),
    ) -> None:
        """Create a local .pfe workspace and default config."""

        from pfe_core.config import PFEConfig

        resolved_home = home.expanduser()
        _init_runtime_dirs(resolved_home, workspace)

        config = PFEConfig()
        config.model.base_model = base_model
        config_path = config.save(home=resolved_home)

        typer.echo("PFE workspace initialized")
        typer.echo(f"config path: {config_path}")
        typer.echo(f"workspace:   {workspace}")
        typer.echo(f"base model:  {base_model}")
        typer.echo("")
        typer.echo("Next steps:")
        typer.echo(f"  pfe doctor --workspace {workspace}")
        typer.echo(f"  pfe next --workspace {workspace}")
        typer.echo(f"  pfe serve --port 8921 --workspace {workspace} --live")


__all__ = ["register_init_command"]
