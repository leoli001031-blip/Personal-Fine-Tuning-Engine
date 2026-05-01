"""Daemon and runner console action routing."""

from __future__ import annotations

from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext


def route_console_daemon_action(ctx: ConsoleRouteContext) -> ConsoleCommandResult | None:
    normalized = ctx.normalized
    deps = ctx.deps
    service = ctx.service
    workspace = ctx.workspace

    if normalized in {"recover daemon", "daemon recover"}:
        handler = deps.resolve_handler(service, "recover_train_queue_daemon", "daemon_recover")
        if handler is None:
            return "Daemon recovery is unavailable.", "daemon-recover-unavailable", None
        result = handler(workspace=workspace, note=None)
        return deps.format_train_queue_daemon_status(result), "daemon-recover", None
    if normalized in {"restart daemon", "daemon restart"}:
        handler = deps.resolve_handler(service, "restart_train_queue_daemon", "daemon_restart")
        if handler is None:
            return "Daemon restart is unavailable.", "daemon-restart-unavailable", None
        result = handler(workspace=workspace, note=None)
        return deps.format_train_queue_daemon_status(result), "daemon-restart", None
    if normalized in {"stop daemon", "daemon stop"}:
        handler = deps.resolve_handler(service, "stop_train_queue_daemon", "daemon_stop")
        if handler is None:
            return "Daemon stop is unavailable.", "daemon-stop-unavailable", None
        result = handler(workspace=workspace, note=None)
        return deps.format_train_queue_daemon_status(result), "daemon-stop", None
    if normalized in {"start daemon", "daemon start"}:
        handler = deps.resolve_handler(service, "start_train_queue_daemon", "daemon_start")
        if handler is None:
            return "Daemon start is unavailable.", "daemon-start-unavailable", None
        result = handler(workspace=workspace, note=None)
        return deps.format_train_queue_daemon_status(result), "daemon-start", None
    if normalized in {"stop runner", "runner stop"}:
        handler = deps.resolve_handler(service, "stop_train_queue_worker_runner", "runner_stop")
        if handler is None:
            return "Runner stop is unavailable.", "runner-stop-unavailable", None
        result = handler(workspace=workspace)
        return deps.format_worker_runner_status(result), "runner-stop", None

    return None


__all__ = ["route_console_daemon_action"]
