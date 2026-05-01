"""Train queue manual review command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps, run_simple_status_command


def register_trigger_queue_review_commands(*, trigger_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    @trigger_app.command("approve-next")
    def trigger_approve_next(
        note: Optional[str] = typer.Option(None, "--note", help="Optional approval note for audit trail."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Approve the next queued item that is waiting for manual confirmation."""

        run_simple_status_command(
            deps,
            command_name="trigger approve-next",
            handler_name="approve_next_train_queue",
            workspace=workspace,
            note=note,
        )

    @trigger_app.command("reject-next")
    def trigger_reject_next(
        note: Optional[str] = typer.Option(None, "--note", help="Optional rejection note for audit trail."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Reject the next queued item that is waiting for manual confirmation."""

        run_simple_status_command(
            deps,
            command_name="trigger reject-next",
            handler_name="reject_next_train_queue",
            workspace=workspace,
            note=note,
        )


__all__ = ["register_trigger_queue_review_commands"]
