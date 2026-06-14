"""Candidate and adapter-oriented console action routing."""

from __future__ import annotations

from .console_routing_action_helpers import command_suffix
from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext


def route_console_candidate_action(ctx: ConsoleRouteContext) -> ConsoleCommandResult | None:
    normalized = ctx.normalized
    deps = ctx.deps
    service = ctx.service
    workspace = ctx.workspace

    if normalized.startswith("promote"):
        handler = deps.resolve_handler(service, "promote_candidate")
        if handler is None:
            return "Candidate promote is unavailable.", "candidate-promote-unavailable", None
        result = handler(workspace=workspace, note=command_suffix(normalized))
        return deps.format_status(result, workspace=workspace), "candidate-promote", None
    if normalized.startswith("archive"):
        handler = deps.resolve_handler(service, "archive_candidate")
        if handler is None:
            return "Candidate archive is unavailable.", "candidate-archive-unavailable", None
        result = handler(workspace=workspace, note=command_suffix(normalized))
        return deps.format_status(result, workspace=workspace), "candidate-archive", None
    if normalized.startswith("rollback"):
        handler = deps.resolve_handler(service, "rollback_candidate", "rollback_adapter")
        if handler is None:
            return "Rollback action is unavailable.", "rollback-unavailable", None
        result = handler(workspace=workspace, version=command_suffix(normalized))
        return deps.format_status(result, workspace=workspace), "candidate-rollback", None
    if normalized in {"list", "adapter list", "adapters"}:
        handler = deps.resolve_handler(service, "list_versions")
        if handler is not None:
            result = handler(workspace=workspace, limit=20)
            lines = deps.format_lifecycle_summary(result)
            return "\n".join(lines or ["No adapters found."]), "adapter-list", None
        return "Adapter listing is unavailable.", "adapter-list-unavailable", None

    return None


__all__ = ["route_console_candidate_action"]
