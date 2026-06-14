"""Train queue history command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps, pipeline_service


def register_trigger_queue_history_command(*, trigger_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    @trigger_app.command("queue-history")
    def trigger_queue_history(
        job_id: Optional[str] = typer.Option(None, "--job-id", help="Specific queue job id."),
        limit: int = typer.Option(10, "--limit", min=1, help="Maximum history entries to show."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show train queue history for the latest job or a specific queue job."""

        service = pipeline_service(deps)
        if service is None:
            deps.run_placeholder("trigger queue-history")
            return

        handler = deps.resolve_handler(service, "train_queue_history")
        if handler is None:
            deps.run_placeholder("trigger queue-history")
            return

        deps.run_handler(
            "trigger queue-history",
            handler,
            formatter=deps.format_train_queue_history,
            workspace=workspace,
            job_id=job_id,
            limit=limit,
        )


__all__ = ["register_trigger_queue_history_command"]
