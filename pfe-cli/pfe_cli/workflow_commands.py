"""Typer commands for generate, distill, and eval workflows."""

from __future__ import annotations

import typer

from .workflow_command_deps import WorkflowCommandDeps
from .workflow_distill_command import register_distill_command
from .workflow_eval_command import register_eval_command
from .workflow_generate_command import register_generate_command


def register_workflow_commands(app: typer.Typer, deps: WorkflowCommandDeps) -> None:
    """Attach generation, distillation, and evaluation commands."""

    register_generate_command(app, deps)
    register_distill_command(app, deps)
    register_eval_command(app, deps)


__all__ = ["WorkflowCommandDeps", "register_workflow_commands"]
