"""Dependency contract and validation for training CLI commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import typer

from .training_controls import validate_train_backend_option


@dataclass(frozen=True)
class TrainingCommandDeps:
    """Runtime hooks supplied by the main CLI module."""

    load_service: Callable[..., Any | None]
    run_placeholder: Callable[[str], None]
    resolve_handler: Callable[..., Any | None]
    run_handler: Callable[..., None]
    format_train_preview: Callable[..., str]
    format_train_result: Callable[..., str]
    record_train_cli_state: Callable[..., None]


def validate_backend_or_exit(backend: str, *, train_type: str) -> str:
    try:
        return validate_train_backend_option(backend, train_type=train_type)
    except ValueError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)


__all__ = ["TrainingCommandDeps", "validate_backend_or_exit"]
