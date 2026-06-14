"""Auto-train trigger control commands."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps, run_simple_status_command


def register_trigger_control_commands(*, trigger_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach trigger reset, retry, enable, disable, and configure commands."""

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

    @trigger_app.command("configure")
    def trigger_configure(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        enable: bool = typer.Option(False, "--enable", help="Enable auto-train trigger."),
        disable: bool = typer.Option(False, "--disable", help="Disable auto-train trigger."),
        min_new_samples: Optional[int] = typer.Option(
            None,
            "--min-new-samples",
            min=1,
            help="Minimum new curated signal samples required before auto-train can run.",
        ),
        max_interval_days: Optional[int] = typer.Option(
            None,
            "--max-interval-days",
            min=0,
            help="Maximum days between training runs before the interval gate is considered elapsed.",
        ),
        queue_mode: Optional[str] = typer.Option(None, "--queue-mode", help="Queue mode: inline or deferred."),
        require_confirmation: Optional[bool] = typer.Option(
            None,
            "--require-confirmation/--no-require-confirmation",
            help="Require manual queue approval before deferred jobs can run.",
        ),
        epochs: Optional[int] = typer.Option(None, "--epochs", min=1, help="Default epochs for auto-train jobs."),
        backend: Optional[str] = typer.Option(None, "--backend", help="Default trainer backend for auto-train jobs."),
    ) -> None:
        """Persist auto-train trigger thresholds and queue execution mode."""

        if enable and disable:
            raise typer.BadParameter("--enable and --disable cannot be used together")

        enabled: bool | None = None
        if enable:
            enabled = True
        elif disable:
            enabled = False

        run_simple_status_command(
            deps,
            command_name="trigger configure",
            handler_name="configure_auto_train_trigger",
            workspace=workspace,
            enabled=enabled,
            min_new_samples=min_new_samples,
            max_interval_days=max_interval_days,
            queue_mode=queue_mode,
            require_queue_confirmation=require_confirmation,
            epochs=epochs,
            backend=backend,
        )


__all__ = ["register_trigger_control_commands"]
