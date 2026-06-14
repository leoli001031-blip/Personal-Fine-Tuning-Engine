"""DPO train command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .training_command_deps import TrainingCommandDeps, validate_backend_or_exit
from .training_dpo_execution import preview_dpo_dataset, run_dpo_training
from .training_service_resolution import training_service


def register_dpo_command(app: typer.Typer, deps: TrainingCommandDeps) -> None:
    @app.command("dpo")
    def dpo_train(
        method: str = typer.Option("qlora", "--method", help="Training method, e.g. lora or qlora."),
        epochs: int = typer.Option(3, "--epochs", min=1, help="Training epochs."),
        base_model: Optional[str] = typer.Option(None, "--base-model", help="Base model id or local path."),
        base_adapter: Optional[str] = typer.Option(None, "--base-adapter", help="Parent SFT adapter for incremental DPO training."),
        min_confidence: float = typer.Option(
            0.4,
            "--min-confidence",
            min=0.0,
            max=1.0,
            help="Minimum signal confidence for DPO pairs.",
        ),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        backend: str = typer.Option("dpo", "--backend", help="Training backend for DPO: dpo, peft, mock_local, or auto."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Build the DPO training plan without launching real training."),
        real_local: bool = typer.Option(False, "--real-local", help="Enable real local DPO training for this invocation."),
        preview: bool = typer.Option(False, "--preview", help="Preview DPO dataset without training."),
        train: bool = typer.Option(False, "--train", help="Execute real DPO training (default is preview mode)."),
    ) -> None:
        """Train using Direct Preference Optimization (DPO)."""

        backend = validate_backend_or_exit(backend, train_type="dpo")
        service = training_service(deps)
        if service is None:
            deps.run_placeholder("dpo")
            return

        if preview or (not train and not dry_run):
            preview_done = preview_dpo_dataset(
                deps=deps,
                service=service,
                workspace=workspace,
                min_confidence=min_confidence,
                train=train,
            )
            if preview_done:
                return

        run_dpo_training(
            deps=deps,
            service=service,
            method=method,
            epochs=epochs,
            base_model=base_model,
            base_adapter=base_adapter,
            min_confidence=min_confidence,
            workspace=workspace,
            dry_run=dry_run,
            real_local=real_local,
            backend_hint=None if backend == "auto" else backend,
        )


__all__ = ["register_dpo_command"]
