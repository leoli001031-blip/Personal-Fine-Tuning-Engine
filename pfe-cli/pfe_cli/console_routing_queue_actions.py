"""Train queue console action routing."""

from __future__ import annotations

from .console_routing_action_helpers import batch_limit, command_suffix
from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext


def route_console_queue_action(ctx: ConsoleRouteContext) -> ConsoleCommandResult | None:
    normalized = ctx.normalized
    deps = ctx.deps
    service = ctx.service
    workspace = ctx.workspace

    if normalized.startswith("approve"):
        handler = deps.resolve_handler(service, "approve_next_train_queue", "approve_train_queue_item")
        if handler is None:
            return "Approve action is unavailable.", "approve-unavailable", None
        result = handler(workspace=workspace, note=command_suffix(normalized))
        return deps.format_status(result, workspace=workspace), "approve-next", None
    if normalized.startswith("reject"):
        handler = deps.resolve_handler(service, "reject_next_train_queue", "reject_train_queue_item")
        if handler is None:
            return "Reject action is unavailable.", "reject-unavailable", None
        result = handler(workspace=workspace, note=command_suffix(normalized))
        return deps.format_status(result, workspace=workspace), "reject-next", None
    if normalized in {"process", "process next", "next"}:
        handler = deps.resolve_handler(service, "process_next_train_queue")
        if handler is None:
            return "Queue processing is unavailable.", "process-unavailable", None
        result = handler(workspace=workspace)
        return deps.format_status(result, workspace=workspace), "process-next", None

    return None


def route_console_queue_batch_action(ctx: ConsoleRouteContext) -> ConsoleCommandResult | None:
    normalized = ctx.normalized
    deps = ctx.deps
    service = ctx.service
    workspace = ctx.workspace

    if normalized in {"batch", "process batch"}:
        handler = deps.resolve_handler(service, "process_train_queue_batch")
        if handler is None:
            return "Queue batch processing is unavailable.", "batch-unavailable", None
        result = handler(limit=batch_limit(normalized))
        return deps.format_status(result, workspace=workspace), "process-batch", None
    if normalized in {"until-idle", "process until-idle", "process idle"}:
        handler = deps.resolve_handler(service, "process_train_queue_until_idle")
        if handler is None:
            return "Queue until-idle processing is unavailable.", "until-idle-unavailable", None
        result = handler()
        return deps.format_status(result, workspace=workspace), "process-until-idle", None

    return None


__all__ = ["route_console_queue_action", "route_console_queue_batch_action"]
