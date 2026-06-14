"""Auto-train trigger command registration."""

from __future__ import annotations

import typer

from .operations_command_deps import OperationsCommandDeps
from .operations_trigger_control_commands import register_trigger_control_commands
from .operations_trigger_queue_commands import register_trigger_queue_commands
from .operations_trigger_status_commands import register_trigger_status_commands
from .operations_trigger_worker_commands import register_trigger_worker_commands


def register_trigger_commands(*, trigger_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach auto-train trigger commands to the trigger sub-app."""
    register_trigger_control_commands(trigger_app=trigger_app, deps=deps)
    register_trigger_queue_commands(trigger_app=trigger_app, deps=deps)
    register_trigger_worker_commands(trigger_app=trigger_app, deps=deps)
    register_trigger_status_commands(trigger_app=trigger_app, deps=deps)


__all__ = ["register_trigger_commands"]
