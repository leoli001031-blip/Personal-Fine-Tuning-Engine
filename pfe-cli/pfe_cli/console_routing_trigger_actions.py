"""Auto-train trigger console action routing."""

from __future__ import annotations

from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext


def route_console_trigger_action(ctx: ConsoleRouteContext) -> ConsoleCommandResult | None:
    normalized = ctx.normalized
    deps = ctx.deps
    service = ctx.service
    workspace = ctx.workspace

    if normalized in {"retry", "trigger-train", "trigger train"}:
        handler = deps.resolve_handler(service, "retry_auto_train_trigger")
        if handler is None:
            return "Retry action is unavailable.", "retry-unavailable", None
        result = handler(workspace=workspace)
        return deps.format_status(result, workspace=workspace), "retry-trigger", None
    if normalized == "reset":
        handler = deps.resolve_handler(service, "reset_auto_train_trigger")
        if handler is None:
            return "Reset action is unavailable.", "reset-unavailable", None
        result = handler(workspace=workspace)
        return deps.format_status(result, workspace=workspace), "reset-trigger", None

    return None


__all__ = ["route_console_trigger_action"]
