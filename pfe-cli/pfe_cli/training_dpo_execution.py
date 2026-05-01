"""DPO training command execution helpers."""

from __future__ import annotations

from typing import Any

import typer

from .training_command_deps import TrainingCommandDeps
from .training_controls import real_training_env


def preview_dpo_dataset(
    *,
    deps: TrainingCommandDeps,
    service: Any,
    workspace: str | None,
    min_confidence: float,
    train: bool,
) -> bool:
    preview_handler = deps.resolve_handler(service, "build_dpo_dataset")
    if preview_handler:
        result = preview_handler(workspace=workspace, min_confidence=min_confidence)
        typer.echo("DPO Dataset Preview:")
        typer.echo(f"  Estimated pairs: {result.get('num_pairs', 0)}")
        typer.echo(f"  Min confidence: {result.get('min_confidence', min_confidence)}")
        typer.echo(f"  Signal statistics: {result.get('statistics', {})}")
        if not train:
            typer.echo("\nUse --train to execute DPO training.")
            return True
        return False

    typer.secho("DPO preview not available.", err=True, fg=typer.colors.YELLOW)
    return not train


def dpo_handler_kwargs(
    *,
    method: str,
    epochs: int,
    base_model: str | None,
    base_adapter: str | None,
    min_confidence: float,
    workspace: str | None,
    dry_run: bool,
    real_local: bool,
    backend_hint: str | None,
) -> dict[str, Any]:
    handler_kwargs: dict[str, Any] = {
        "method": method,
        "epochs": epochs,
        "base_model": base_model,
        "workspace": workspace,
        "min_confidence": min_confidence,
        "dry_run": dry_run,
        "real_local": real_local,
    }
    if backend_hint is not None:
        handler_kwargs["backend"] = backend_hint
    if base_adapter:
        handler_kwargs["base_adapter_path"] = base_adapter
    return handler_kwargs


def run_dpo_training(
    *,
    deps: TrainingCommandDeps,
    service: Any,
    method: str,
    epochs: int,
    base_model: str | None,
    base_adapter: str | None,
    min_confidence: float,
    workspace: str | None,
    dry_run: bool,
    real_local: bool,
    backend_hint: str | None,
) -> None:
    handler = deps.resolve_handler(service, "train_dpo")
    if handler is None:
        typer.secho(
            "DPO training is unavailable. Ensure pfe_core is installed with trl support.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    mode_label = "dry-run" if dry_run else "real" if real_local else "planned"
    typer.echo(
        f"Starting DPO training (method={method}, epochs={epochs}, "
        f"min_confidence={min_confidence}, mode={mode_label})..."
    )
    if base_adapter:
        typer.echo(f"  Base SFT adapter: {base_adapter}")
    with real_training_env(real_local=real_local):
        deps.run_handler(
            "dpo",
            handler,
            formatter=lambda result: deps.format_train_result(result, workspace=workspace or "user_default"),
            on_result=lambda result: deps.record_train_cli_state(result, workspace=workspace or "user_default"),
            **dpo_handler_kwargs(
                method=method,
                epochs=epochs,
                base_model=base_model,
                base_adapter=base_adapter,
                min_confidence=min_confidence,
                workspace=workspace,
                dry_run=dry_run,
                real_local=real_local,
                backend_hint=backend_hint,
            ),
        )


__all__ = ["dpo_handler_kwargs", "preview_dpo_dataset", "run_dpo_training"]
