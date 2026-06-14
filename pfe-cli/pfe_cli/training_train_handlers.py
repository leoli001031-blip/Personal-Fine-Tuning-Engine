"""Handler resolution and kwargs for the SFT train CLI command."""

from __future__ import annotations

from typing import Any

import typer

from .training_command_deps import TrainingCommandDeps


def train_handler_kwargs(
    *,
    method: str,
    epochs: int,
    base_model: str | None,
    train_type: str,
    workspace: str | None,
    dry_run: bool,
    real_local: bool,
    backend_hint: str | None,
) -> dict[str, Any]:
    handler_kwargs: dict[str, Any] = {
        "method": method,
        "epochs": epochs,
        "base_model": base_model,
        "train_type": train_type,
        "workspace": workspace,
        "dry_run": dry_run,
        "real_local": real_local,
    }
    if backend_hint is not None:
        handler_kwargs["backend"] = backend_hint
    return handler_kwargs


def incremental_handler_kwargs(
    *,
    base_adapter: str,
    method: str,
    epochs: int,
    train_type: str,
    workspace: str | None,
    dry_run: bool,
    real_local: bool,
    backend_hint: str | None,
) -> dict[str, Any]:
    handler_kwargs: dict[str, Any] = {
        "base_adapter": base_adapter,
        "method": method,
        "epochs": epochs,
        "train_type": train_type,
        "workspace": workspace,
        "dry_run": dry_run,
        "real_local": real_local,
    }
    if backend_hint is not None:
        handler_kwargs["backend"] = backend_hint
    return handler_kwargs


def resolve_train_handler(
    *,
    deps: TrainingCommandDeps,
    service: Any,
    incremental: bool,
    base_adapter: str | None,
) -> tuple[Any | None, str | None]:
    handler = deps.resolve_handler(service, "train_result", "train")
    if not incremental:
        return handler, None

    incremental_handler = deps.resolve_handler(service, "train_incremental")
    if incremental_handler is None:
        typer.secho(
            "Incremental training is unavailable because no train_incremental handler is registered.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    if not base_adapter:
        typer.secho("Incremental training requires --base-adapter.", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    return incremental_handler, base_adapter


__all__ = [
    "incremental_handler_kwargs",
    "resolve_train_handler",
    "train_handler_kwargs",
]
