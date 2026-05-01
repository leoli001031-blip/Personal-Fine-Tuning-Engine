"""Train queue daemon command registration."""

from __future__ import annotations

import typer

from .operations_command_deps import OperationsCommandDeps
from .operations_daemon_control_commands import register_daemon_control_commands
from .operations_daemon_monitor_commands import register_daemon_monitor_commands


def register_daemon_commands(*, daemon_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach train queue daemon commands to the daemon sub-app."""
    register_daemon_control_commands(daemon_app=daemon_app, deps=deps)
    register_daemon_monitor_commands(daemon_app=daemon_app, deps=deps)


__all__ = ["register_daemon_commands"]
