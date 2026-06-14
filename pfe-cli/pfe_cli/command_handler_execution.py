"""Handler execution and user-facing error handling for CLI commands."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import typer


def friendly_exception_message(exc: Exception) -> str | None:
    """Return a concise message for known domain errors."""

    name = exc.__class__.__name__.lower()
    if "trainingerror" in name:
        return f"Training failed: {exc}"
    if "adaptererror" in name:
        return f"Adapter error: {exc}"
    if "evaluationerror" in name:
        return f"Evaluation failed: {exc}"
    if "servererror" in name:
        return f"Server error: {exc}"
    if "pipelineerror" in name:
        return f"Pipeline error: {exc}"
    return None


@dataclass(frozen=True)
class CommandExecutionDeps:
    """Runtime hooks supplied by the main CLI module."""

    coerce_mapping: Callable[[Any], dict[str, Any] | None]
    friendly_exception_message: Callable[[Exception], str | None]


def run_handler(
    command_name: str,
    handler: Callable[..., Any],
    formatter: Callable[[Any], str] | None = None,
    on_result: Callable[[Any], None] | None = None,
    *,
    deps: CommandExecutionDeps,
    **kwargs: Any,
) -> None:
    """Execute a handler with short domain-error messages and full propagation for unknown bugs."""

    try:
        result = handler(**kwargs)
    except typer.Exit:
        raise
    except Exception as exc:
        friendly = deps.friendly_exception_message(exc)
        if friendly is not None:
            typer.secho(friendly, err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        raise

    if result is not None:
        if on_result is not None:
            try:
                on_result(result)
            except Exception:
                pass
        typer.echo(formatter(result) if formatter is not None else result)


def run_handler_json(
    command_name: str,
    handler: Callable[..., Any],
    *,
    deps: CommandExecutionDeps,
    **kwargs: Any,
) -> None:
    try:
        result = handler(**kwargs)
    except typer.Exit:
        raise
    except Exception as exc:
        friendly = deps.friendly_exception_message(exc)
        if friendly is not None:
            typer.secho(friendly, err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        raise

    mapping = deps.coerce_mapping(result)
    payload: Any = mapping if mapping is not None else result
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def run_placeholder(command_name: str) -> None:
    typer.echo(
        f"[pfe] {command_name}: command is not available in the current environment. "
        "Some PFE surfaces are still bootstrap-oriented or require optional services to be resolved."
    )


__all__ = [
    "CommandExecutionDeps",
    "friendly_exception_message",
    "run_handler",
    "run_handler_json",
    "run_placeholder",
]
