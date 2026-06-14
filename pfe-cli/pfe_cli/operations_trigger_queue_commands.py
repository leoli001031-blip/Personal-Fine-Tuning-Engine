"""Train queue commands under the auto-train trigger app."""

from __future__ import annotations

import typer

from .operations_command_deps import OperationsCommandDeps
from .operations_trigger_queue_history_commands import register_trigger_queue_history_command
from .operations_trigger_queue_processing_commands import register_trigger_queue_processing_commands
from .operations_trigger_queue_review_commands import register_trigger_queue_review_commands


def register_trigger_queue_commands(*, trigger_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach queue processing, review, and history commands."""

    register_trigger_queue_processing_commands(trigger_app=trigger_app, deps=deps)
    register_trigger_queue_review_commands(trigger_app=trigger_app, deps=deps)
    register_trigger_queue_history_command(trigger_app=trigger_app, deps=deps)


__all__ = ["register_trigger_queue_commands"]
