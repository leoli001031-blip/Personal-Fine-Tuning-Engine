"""Typer command registration for local training workflows."""

from __future__ import annotations

import typer

from .training_command_deps import TrainingCommandDeps, validate_backend_or_exit
from .training_dpo_command import register_dpo_command
from .training_train_command import register_train_command


def _validate_backend_option(backend: str, *, train_type: str) -> str:
    return validate_backend_or_exit(backend, train_type=train_type)


def register_training_commands(app: typer.Typer, deps: TrainingCommandDeps) -> None:
    """Attach training commands to the root CLI app."""

    register_train_command(app, deps)
    register_dpo_command(app, deps)


__all__ = ["TrainingCommandDeps", "_validate_backend_option", "register_training_commands"]
