"""Train queue processing command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps, run_simple_status_command


def register_trigger_queue_processing_commands(*, trigger_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    @trigger_app.command("process-next")
    def trigger_process_next(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Process the next queued auto-train item when queue mode is deferred."""

        run_simple_status_command(
            deps,
            command_name="trigger process-next",
            handler_name="process_next_train_queue",
            workspace=workspace,
        )

    @trigger_app.command("process-batch")
    def trigger_process_batch(
        limit: int = typer.Option(5, "--limit", min=1, help="Maximum queued items to process in one batch."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Process up to N queued auto-train items when queue mode is deferred."""

        run_simple_status_command(
            deps,
            command_name="trigger process-batch",
            handler_name="process_train_queue_batch",
            workspace=workspace,
            limit=limit,
        )

    @trigger_app.command("process-until-idle")
    def trigger_process_until_idle(
        max_iterations: int = typer.Option(
            10,
            "--max-iterations",
            min=1,
            help="Maximum queued items to process before stopping.",
        ),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Process queued auto-train items until the queue drains or the iteration cap is reached."""

        run_simple_status_command(
            deps,
            command_name="trigger process-until-idle",
            handler_name="process_train_queue_until_idle",
            workspace=workspace,
            max_iterations=max_iterations,
        )


__all__ = ["register_trigger_queue_processing_commands"]
