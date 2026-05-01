"""Root-level utility commands."""

from __future__ import annotations

import time
import webbrowser
from typing import Optional

import typer

from . import formatters_matrix
from .utility_command_deps import UtilityCommandDeps


def register_basic_utility_commands(app: typer.Typer, deps: UtilityCommandDeps) -> None:
    @app.command("doctor")
    def doctor(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        base_model: Optional[str] = typer.Option(
            None,
            "--base-model",
            help="Override base model path or model id for local model checks.",
        ),
    ) -> None:
        """Show strict_local readiness signals for trainer, model, export, and adapter state."""

        typer.echo(deps.format_doctor(workspace=workspace, base_model=base_model))

    @app.command("dashboard")
    def dashboard(
        port: int = typer.Option(8921, "--port", min=1, max=65535, help="Server port to connect to."),
        host: str = typer.Option("127.0.0.1", "--host", help="Server host."),
        open_browser: bool = typer.Option(True, "--open/--no-open", help="Open dashboard in browser."),
    ) -> None:
        """Launch the PFE observability dashboard in a web browser."""

        dashboard_url = f"http://{host}:{port}/dashboard"

        typer.echo("PFE Observability Dashboard")
        typer.echo(f"URL: {dashboard_url}")

        if open_browser:
            typer.echo("Opening browser...")
            webbrowser.open(dashboard_url)
        else:
            typer.echo("Use --open to launch browser automatically.")

    @app.command("boot")
    def boot() -> None:
        """Display PFE boot sequence with ZC logo."""

        from .pixel_logo import render_boot_banner, render_commands_matrix, render_loading_sequence

        typer.echo(render_boot_banner(version="2.0.0"))

        steps = [
            "Loading adapter store...",
            "Initializing trainer service...",
            "Mounting signal collector...",
            "Establishing daemon connection...",
            "Calibrating neural weights...",
        ]

        for index, step in enumerate(steps, 1):
            typer.echo(
                f"{formatters_matrix.MatrixColors.GREEN}  "
                f"{render_loading_sequence(index, len(steps))}{formatters_matrix.MatrixColors.RESET} {step}"
            )
            time.sleep(0.15)

        typer.echo("")
        typer.echo(
            f"{formatters_matrix.MatrixColors.GREEN_BRIGHT}{formatters_matrix.MatrixColors.BOLD}  "
            f">> ALL SYSTEMS OPERATIONAL <<{formatters_matrix.MatrixColors.RESET}"
        )
        typer.echo("")
        typer.echo(render_commands_matrix())


__all__ = ["register_basic_utility_commands"]
