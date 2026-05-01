"""Auto-train trigger control commands."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps, run_simple_status_command


def register_trigger_control_commands(*, trigger_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach trigger reset, retry, enable, and disable commands."""

    @trigger_app.command("reset")
    def trigger_reset(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Clear persisted auto-train cooldown/backoff state for the workspace."""

        run_simple_status_command(
            deps,
            command_name="trigger reset",
            handler_name="reset_auto_train_trigger",
            workspace=workspace,
        )

    @trigger_app.command("retry")
    def trigger_retry(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Re-check the auto-train trigger and run it if all current gates pass."""

        run_simple_status_command(
            deps,
            command_name="trigger retry",
            handler_name="retry_auto_train_trigger",
            workspace=workspace,
        )

    @trigger_app.command("enable")
    def trigger_enable(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Enable auto-train trigger for the workspace."""

        run_simple_status_command(
            deps,
            command_name="trigger enable",
            handler_name="enable_auto_train_trigger",
            workspace=workspace,
        )

    @trigger_app.command("disable")
    def trigger_disable(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Disable auto-train trigger for the workspace."""

        run_simple_status_command(
            deps,
            command_name="trigger disable",
            handler_name="disable_auto_train_trigger",
            workspace=workspace,
        )


__all__ = ["register_trigger_control_commands"]
