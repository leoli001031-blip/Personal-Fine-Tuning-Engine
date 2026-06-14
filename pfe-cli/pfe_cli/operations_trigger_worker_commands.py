"""Train queue worker commands under the auto-train trigger app."""

from __future__ import annotations

from typing import Optional

import typer

from .operations_command_deps import OperationsCommandDeps, pipeline_service, run_simple_status_command


def register_trigger_worker_commands(*, trigger_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach bounded worker loop and worker runner commands."""

    @trigger_app.command("run-worker-loop")
    def trigger_run_worker_loop(
        max_cycles: int = typer.Option(10, "--max-cycles", min=1, help="Maximum worker loop cycles before stopping."),
        idle_rounds: int = typer.Option(1, "--idle-rounds", min=1, help="Stop after this many idle polling rounds."),
        poll_interval_seconds: float = typer.Option(
            0.0,
            "--poll-interval-seconds",
            min=0.0,
            help="Sleep between loop cycles in seconds.",
        ),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Run the train queue worker loop for a bounded number of cycles."""

        run_simple_status_command(
            deps,
            command_name="trigger run-worker-loop",
            handler_name="run_train_queue_worker_loop",
            workspace=workspace,
            max_cycles=max_cycles,
            idle_rounds=idle_rounds,
            poll_interval_seconds=poll_interval_seconds,
        )

    @trigger_app.command("run-worker-runner")
    def trigger_run_worker_runner(
        max_seconds: float = typer.Option(30.0, "--max-seconds", min=0.1, help="Maximum runner duration in seconds."),
        idle_sleep_seconds: float = typer.Option(1.0, "--idle-sleep-seconds", min=0.0, help="Sleep duration between idle polls."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Run the long-poll train queue worker runner for a bounded duration."""

        run_simple_status_command(
            deps,
            command_name="trigger run-worker-runner",
            handler_name="run_train_queue_worker_runner",
            workspace=workspace,
            max_seconds=max_seconds,
            idle_sleep_seconds=idle_sleep_seconds,
        )

    @trigger_app.command("stop-worker-runner")
    def trigger_stop_worker_runner(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Request the long-poll train queue worker runner to stop."""

        run_simple_status_command(
            deps,
            command_name="trigger stop-worker-runner",
            handler_name="stop_train_queue_worker_runner",
            workspace=workspace,
        )

    @trigger_app.command("worker-runner-history")
    def trigger_worker_runner_history(
        limit: int = typer.Option(10, "--limit", min=1, help="Maximum worker runner history entries to show."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show worker runner lifecycle history for the workspace."""

        service = pipeline_service(deps)
        if service is None:
            deps.run_placeholder("trigger worker-runner-history")
            return

        handler = deps.resolve_handler(service, "train_queue_worker_runner_history")
        if handler is None:
            deps.run_placeholder("trigger worker-runner-history")
            return

        deps.run_handler(
            "trigger worker-runner-history",
            handler,
            formatter=deps.format_worker_runner_history,
            workspace=workspace,
            limit=limit,
        )


__all__ = ["register_trigger_worker_commands"]
