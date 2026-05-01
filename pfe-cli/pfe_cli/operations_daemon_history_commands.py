"""Daemon history command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps, pipeline_service


def register_daemon_history_command(*, daemon_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    @daemon_app.command("history")
    def daemon_history(
        limit: int = typer.Option(10, "--limit", min=1, help="Maximum daemon history entries to show."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show the background worker daemon control history."""

        service = pipeline_service(deps)
        if service is not None:
            handler = deps.resolve_handler(service, "train_queue_daemon_history", "daemon_history")
            if handler is not None:
                deps.run_handler(
                    "daemon history",
                    handler,
                    formatter=deps.format_train_queue_daemon_history,
                    workspace=workspace,
                    limit=limit,
                )
                return

        state = deps.read_train_queue_daemon_state(workspace) or {"workspace": workspace or "user_default"}
        history = list(state.get("history") or [])
        payload = {
            "workspace": state.get("workspace") or workspace or "user_default",
            "count": len(history),
            "last_event": state.get("last_event"),
            "last_reason": state.get("last_reason"),
            "items": history[-max(1, int(limit or 10)) :],
        }
        typer.echo(deps.format_train_queue_daemon_history(payload))


__all__ = ["register_daemon_history_command"]
