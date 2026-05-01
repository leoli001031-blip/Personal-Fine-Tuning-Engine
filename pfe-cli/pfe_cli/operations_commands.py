"""Typer command groups for operational PFE workflows."""

from __future__ import annotations

import typer

from .operations_candidate_commands import register_candidate_commands
from .operations_collect_commands import register_collect_commands
from .operations_command_deps import OperationsCommandDeps
from .operations_daemon_commands import register_daemon_commands
from .operations_eval_trigger_commands import register_eval_trigger_commands
from .operations_trigger_commands import register_trigger_commands


def register_operations_commands(
    *,
    trigger_app: typer.Typer,
    daemon_app: typer.Typer,
    candidate_app: typer.Typer,
    eval_trigger_app: typer.Typer,
    collect_app: typer.Typer,
    deps: OperationsCommandDeps,
) -> None:
    """Attach operations command groups to their Typer sub-apps."""

    register_trigger_commands(trigger_app=trigger_app, deps=deps)
    register_daemon_commands(daemon_app=daemon_app, deps=deps)
    register_candidate_commands(candidate_app=candidate_app, deps=deps)
    register_eval_trigger_commands(eval_trigger_app=eval_trigger_app, deps=deps)
    register_collect_commands(collect_app=collect_app, deps=deps)
