"""Daemon recovery and reliability alert command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps
from .operations_daemon_monitor_runner import run_daemon_monitor_handler


def register_daemon_monitor_reliability_commands(*, daemon_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    @daemon_app.command("force-recovery")
    def daemon_force_recovery(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        reason: Optional[str] = typer.Option(None, "--reason", help="Optional reason for forced recovery."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    ) -> None:
        """Force daemon recovery with reset restart policy (bypasses backoff)."""

        run_daemon_monitor_handler(
            deps,
            command_name="daemon force-recovery",
            handler_names=("force_recovery", "force_daemon_recovery"),
            formatter=deps.format_train_queue_daemon_status,
            json_output=json_output,
            unavailable_message="Force recovery not available.",
            workspace=workspace,
            reason=reason,
        )

    @daemon_app.command("alerts")
    def daemon_alerts(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        level: Optional[str] = typer.Option(None, "--level", help="Filter by level (critical, error, warning, attention, info)."),
        scope: Optional[str] = typer.Option(None, "--scope", help="Filter by scope (daemon, runner, task, system)."),
        limit: int = typer.Option(10, "--limit", min=1, help="Maximum alerts to show."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    ) -> None:
        """Show reliability alerts for monitoring."""

        run_daemon_monitor_handler(
            deps,
            command_name="daemon alerts",
            handler_names=("get_reliability_alerts",),
            formatter=deps.format_daemon_alerts,
            json_output=json_output,
            unavailable_message="Alerts check not available.",
            workspace=workspace,
            level=level,
            scope=scope,
            limit=limit,
        )


__all__ = ["register_daemon_monitor_reliability_commands"]
