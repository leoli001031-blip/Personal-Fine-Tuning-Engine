"""Daemon recover/restart command registration."""

from __future__ import annotations

from typing import Any, Optional

import typer

from .operations_command_deps import OperationsCommandDeps, handler_accepts_note, pipeline_service


def _run_recovery_handler(
    *,
    deps: OperationsCommandDeps,
    handler: Any,
    command_name: str,
    workspace: str | None,
    note: str | None,
) -> None:
    handler_kwargs: dict[str, Any] = {"workspace": workspace}
    if handler_accepts_note(handler):
        handler_kwargs["note"] = note
    deps.run_handler(
        command_name,
        handler,
        formatter=deps.format_train_queue_daemon_status,
        **handler_kwargs,
    )


def _echo_recovery_payload(
    *,
    deps: OperationsCommandDeps,
    workspace: str | None,
    action: str,
    note: str | None,
) -> None:
    typer.echo(
        deps.format_train_queue_daemon_status(
            deps.daemon_recovery_payload(
                workspace=workspace,
                action=action,
                note=note,
                reason="cli_requested",
            )
        )
    )


def register_daemon_recovery_commands(*, daemon_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    @daemon_app.command("recover")
    def daemon_recover(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        note: Optional[str] = typer.Option(None, "--note", help="Optional recovery note."),
    ) -> None:
        """Request daemon recovery with local restart policy bookkeeping."""

        service = pipeline_service(deps)
        if service is not None:
            handler = deps.resolve_handler(service, "recover_train_queue_daemon", "daemon_recover")
            if handler is not None:
                _run_recovery_handler(
                    deps=deps,
                    handler=handler,
                    command_name="daemon recover",
                    workspace=workspace,
                    note=note,
                )
                return

        _echo_recovery_payload(deps=deps, workspace=workspace, action="recover", note=note)

    @daemon_app.command("restart")
    def daemon_restart(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        note: Optional[str] = typer.Option(None, "--note", help="Optional restart note."),
    ) -> None:
        """Request daemon restart with local retry bookkeeping."""

        service = pipeline_service(deps)
        if service is not None:
            handler = deps.resolve_handler(service, "restart_train_queue_daemon", "daemon_restart")
            if handler is not None:
                _run_recovery_handler(
                    deps=deps,
                    handler=handler,
                    command_name="daemon restart",
                    workspace=workspace,
                    note=note,
                )
                return

        _echo_recovery_payload(deps=deps, workspace=workspace, action="restart", note=note)


__all__ = ["register_daemon_recovery_commands"]
