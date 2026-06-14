"""Daemon status-oriented monitoring command registration."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps
from .operations_daemon_monitor_runner import run_daemon_monitor_handler


def register_daemon_monitor_status_commands(*, daemon_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    @daemon_app.command("health")
    def daemon_health(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    ) -> None:
        """Show comprehensive health status for daemon and runner."""

        run_daemon_monitor_handler(
            deps,
            command_name="daemon health",
            handler_names=("get_health_status", "get_daemon_health_status"),
            formatter=deps.format_daemon_health_status,
            json_output=json_output,
            unavailable_message="Health status check not available.",
            workspace=workspace,
        )

    @daemon_app.command("heartbeat")
    def daemon_heartbeat(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    ) -> None:
        """Show heartbeat status for daemon and runner."""

        run_daemon_monitor_handler(
            deps,
            command_name="daemon heartbeat",
            handler_names=("get_heartbeat_status", "get_daemon_heartbeat_status"),
            formatter=deps.format_daemon_heartbeat_status,
            json_output=json_output,
            unavailable_message="Heartbeat status check not available.",
            workspace=workspace,
        )

    @daemon_app.command("lease")
    def daemon_lease(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    ) -> None:
        """Show lease status for task execution."""

        run_daemon_monitor_handler(
            deps,
            command_name="daemon lease",
            handler_names=("get_lease_status", "get_runner_lease_status"),
            formatter=deps.format_daemon_lease_status,
            json_output=json_output,
            unavailable_message="Lease status check not available.",
            workspace=workspace,
        )

    @daemon_app.command("check-stale")
    def daemon_check_stale(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        takeover: bool = typer.Option(False, "--takeover", help="Attempt to take over stale locks."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    ) -> None:
        """Check if daemon or runner is stale and optionally trigger takeover."""

        run_daemon_monitor_handler(
            deps,
            command_name="daemon check-stale",
            handler_names=("check_stale_status", "check_daemon_stale"),
            formatter=deps.format_daemon_stale_check,
            json_output=json_output,
            unavailable_message="Stale check not available.",
            workspace=workspace,
            takeover=takeover,
        )


__all__ = ["register_daemon_monitor_status_commands"]
