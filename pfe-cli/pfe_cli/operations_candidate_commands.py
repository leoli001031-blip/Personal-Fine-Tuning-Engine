"""Candidate lifecycle command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps, pipeline_service, run_simple_status_command


def register_candidate_commands(*, candidate_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach candidate lifecycle commands to the candidate sub-app."""

    @candidate_app.command("promote")
    def candidate_promote(
        note: Optional[str] = typer.Option(None, "--note", help="Optional promotion note for audit trail."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Promote the current candidate adapter to latest promoted."""

        run_simple_status_command(
            deps,
            command_name="candidate promote",
            handler_name="promote_candidate",
            workspace=workspace,
            note=note,
        )

    @candidate_app.command("archive")
    def candidate_archive(
        note: Optional[str] = typer.Option(None, "--note", help="Optional archive note for audit trail."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Archive the current candidate adapter without changing latest promoted."""

        run_simple_status_command(
            deps,
            command_name="candidate archive",
            handler_name="archive_candidate",
            workspace=workspace,
            note=note,
        )

    @candidate_app.command("history")
    def candidate_history(
        limit: int = typer.Option(10, "--limit", min=1, help="Maximum candidate history entries to show."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show candidate lifecycle history for the workspace."""

        service = pipeline_service(deps)
        if service is None:
            deps.run_placeholder("candidate history")
            return

        handler = deps.resolve_handler(service, "candidate_history")
        if handler is None:
            deps.run_placeholder("candidate history")
            return

        deps.run_handler(
            "candidate history",
            handler,
            formatter=deps.format_candidate_history,
            workspace=workspace,
            limit=limit,
        )

    @candidate_app.command("timeline")
    def candidate_timeline(
        limit: int = typer.Option(10, "--limit", min=1, help="Maximum candidate timeline entries to show."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show candidate lifecycle timeline for the workspace."""

        service = pipeline_service(deps)
        if service is None:
            deps.run_placeholder("candidate timeline")
            return

        handler = deps.resolve_handler(service, "candidate_timeline")
        if handler is None:
            deps.run_placeholder("candidate timeline")
            return

        deps.run_handler(
            "candidate timeline",
            handler,
            formatter=deps.format_candidate_timeline,
            workspace=workspace,
            limit=limit,
        )
