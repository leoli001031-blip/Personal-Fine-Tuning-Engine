"""Typer commands for serving, console, and status surfaces."""

from __future__ import annotations

import typer

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_command import register_console_command
from .runtime_serve_command import register_serve_command
from .runtime_status_command import register_status_command


def register_runtime_commands(app: typer.Typer, deps: RuntimeCommandDeps) -> None:
    """Attach serve, console, and status commands to the root CLI app."""

    register_serve_command(app, deps)
    register_console_command(app, deps)
    register_status_command(app, deps)
