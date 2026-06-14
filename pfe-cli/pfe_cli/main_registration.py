"""Command registration wiring for the Typer entrypoint."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import typer

from .main_registration_operations import register_main_operations_commands
from .main_registration_runtime import register_main_runtime_commands
from .main_registration_training import register_main_training_commands
from .main_registration_utility import register_main_utility_commands
from .main_registration_workflow import register_main_workflow_commands


def register_main_commands(
    *,
    app: typer.Typer,
    trigger_app: typer.Typer,
    daemon_app: typer.Typer,
    candidate_app: typer.Typer,
    eval_trigger_app: typer.Typer,
    collect_app: typer.Typer,
    symbols: MutableMapping[str, Any],
) -> None:
    """Attach all CLI command groups using the main module's mutable symbol table."""

    register_main_operations_commands(
        trigger_app=trigger_app,
        daemon_app=daemon_app,
        candidate_app=candidate_app,
        eval_trigger_app=eval_trigger_app,
        collect_app=collect_app,
        symbols=symbols,
    )
    register_main_training_commands(app, symbols)
    register_main_workflow_commands(app, symbols)
    register_main_runtime_commands(app, symbols)
    register_main_utility_commands(app, symbols)
