"""SFT/incremental train command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .training_command_deps import TrainingCommandDeps, validate_backend_or_exit
from .training_controls import real_training_env
from .training_service_resolution import training_service
from .training_train_handlers import incremental_handler_kwargs, resolve_train_handler, train_handler_kwargs


def register_train_command(app: typer.Typer, deps: TrainingCommandDeps) -> None:
    @app.command("train")
    def train(
        method: str = typer.Option("qlora", "--method", help="Training method, e.g. lora or qlora."),
        epochs: int = typer.Option(3, "--epochs", min=1, help="Training epochs."),
        base_model: Optional[str] = typer.Option(None, "--base-model", help="Base model id or local path."),
        incremental: bool = typer.Option(False, "--incremental", help="Continue training from an existing adapter."),
        base_adapter: Optional[str] = typer.Option(
            None,
            "--base-adapter",
            help="Parent adapter version or path for incremental training.",
        ),
        train_type: str = typer.Option("sft", "--train-type", help="Training type, e.g. sft or dpo."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        backend: str = typer.Option("auto", "--backend", help="Training backend: auto, mock_local, peft, dpo, unsloth, mlx."),
        preview: bool = typer.Option(False, "--preview", help="Show the training plan and exit without creating artifacts."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Plan/materialize training without launching real training."),
        real_local: bool = typer.Option(False, "--real-local", help="Enable real local training for this invocation."),
    ) -> None:
        """Train an adapter. The trainer service decides backend selection and artifact export."""

        backend = validate_backend_or_exit(backend, train_type=train_type)
        backend_hint = None if backend == "auto" else backend
        service = training_service(deps)
        if service is None:
            deps.run_placeholder("train")
            return

        handler, resolved_base_adapter = resolve_train_handler(
            deps=deps,
            service=service,
            incremental=incremental,
            base_adapter=base_adapter,
        )
        if handler is None:
            deps.run_placeholder("train")
            return

        if incremental:
            handler_kwargs = incremental_handler_kwargs(
                base_adapter=str(resolved_base_adapter),
                method=method,
                epochs=epochs,
                train_type=train_type,
                workspace=workspace,
                dry_run=dry_run,
                real_local=real_local,
                backend_hint=backend_hint,
            )
        else:
            handler_kwargs = train_handler_kwargs(
                method=method,
                epochs=epochs,
                base_model=base_model,
                train_type=train_type,
                workspace=workspace,
                dry_run=dry_run,
                real_local=real_local,
                backend_hint=backend_hint,
            )

        preview_text = deps.format_train_preview(
            method=method,
            epochs=epochs,
            base_model=base_model,
            train_type=train_type,
            workspace=workspace,
            snapshot_workspace=workspace or "user_default",
            backend_hint=backend_hint,
            dry_run=dry_run,
            real_local=real_local,
        )
        typer.echo(preview_text)
        if preview:
            return

        with real_training_env(real_local=real_local):
            deps.run_handler(
                "train",
                handler,
                formatter=lambda result: deps.format_train_result(result, workspace=workspace or "user_default"),
                on_result=lambda result: deps.record_train_cli_state(result, workspace=workspace or "user_default"),
                **handler_kwargs,
            )


__all__ = ["register_train_command"]
