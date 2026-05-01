"""Daemon status/start/stop command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps, pipeline_service


def register_daemon_state_commands(*, daemon_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    @daemon_app.command("status")
    def daemon_status(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show the current background worker daemon control state."""

        service = pipeline_service(deps)
        if service is not None:
            handler = deps.resolve_handler(service, "train_queue_daemon_status", "daemon_status", "get_daemon_status")
            if handler is not None:
                deps.run_handler(
                    "daemon status",
                    handler,
                    formatter=deps.format_train_queue_daemon_status,
                    workspace=workspace,
                )
                return

        typer.echo(
            deps.format_train_queue_daemon_status(
                deps.read_train_queue_daemon_state(workspace)
                or {"workspace": workspace or "user_default", "command_status": "absent"}
            )
        )

    @daemon_app.command("start")
    def daemon_start(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Request the background worker daemon to start."""

        service = pipeline_service(deps)
        if service is not None:
            handler = deps.resolve_handler(service, "start_train_queue_daemon", "run_train_queue_daemon", "daemon_start")
            if handler is not None:
                deps.run_handler(
                    "daemon start",
                    handler,
                    formatter=deps.format_train_queue_daemon_status,
                    workspace=workspace,
                )
                return

        typer.echo(
            deps.format_train_queue_daemon_status(
                deps.update_train_queue_daemon_state(
                    workspace=workspace,
                    desired_state="running",
                    event="start_requested",
                    reason="cli_requested",
                )
            )
        )

    @daemon_app.command("stop")
    def daemon_stop(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Request the background worker daemon to stop."""

        service = pipeline_service(deps)
        if service is not None:
            handler = deps.resolve_handler(service, "stop_train_queue_daemon", "daemon_stop")
            if handler is not None:
                deps.run_handler(
                    "daemon stop",
                    handler,
                    formatter=deps.format_train_queue_daemon_status,
                    workspace=workspace,
                )
                return

        typer.echo(
            deps.format_train_queue_daemon_status(
                deps.update_train_queue_daemon_state(
                    workspace=workspace,
                    desired_state="stopped",
                    event="stop_requested",
                    reason="cli_requested",
                )
            )
        )


__all__ = ["register_daemon_state_commands"]
