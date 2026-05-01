"""Train queue daemon control command registration."""

from __future__ import annotations

import typer

from .operations_command_deps import OperationsCommandDeps
from .operations_daemon_history_commands import register_daemon_history_command
from .operations_daemon_recovery_commands import register_daemon_recovery_commands
from .operations_daemon_state_commands import register_daemon_state_commands


def register_daemon_control_commands(*, daemon_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach daemon control commands to the daemon sub-app."""

    register_daemon_state_commands(daemon_app=daemon_app, deps=deps)
    register_daemon_recovery_commands(daemon_app=daemon_app, deps=deps)
    register_daemon_history_command(daemon_app=daemon_app, deps=deps)


__all__ = ["register_daemon_control_commands"]
