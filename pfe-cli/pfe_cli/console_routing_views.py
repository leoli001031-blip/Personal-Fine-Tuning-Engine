"""Read-only console slash-command routing."""

from __future__ import annotations

from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext
from .console_routing_summaries import (
    console_candidate_summary_text,
    console_daemon_summary_text,
    console_queue_summary_text,
    console_runner_summary_text,
)


def route_console_view_command(ctx: ConsoleRouteContext) -> ConsoleCommandResult | None:
    """Route candidate, queue, runner, and daemon view commands."""
    normalized = ctx.normalized
    deps = ctx.deps
    payload = ctx.payload
    workspace = ctx.workspace
    service = ctx.service

    if normalized in {"candidate history", "candidate-history", "cand hist"}:
        handler = deps.resolve_handler(service, "candidate_history")
        if handler is not None:
            result = handler(workspace=workspace, limit=10)
            return deps.format_candidate_history(result), "candidate-history", None
        history = payload.get("candidate_history") or {}
        return deps.format_candidate_history(history), "candidate-history", None
    if normalized in {"candidate summary", "candidate compact", "cand", "cand sum"}:
        handler = deps.resolve_handler(service, "candidate_timeline")
        timeline_result = handler(workspace=workspace, limit=5) if handler is not None else None
        return console_candidate_summary_text(payload, timeline=timeline_result, deps=deps), "candidate-summary", None
    if normalized in {"candidate", "candidate timeline", "candidate-timeline", "cand tl"}:
        handler = deps.resolve_handler(service, "candidate_timeline")
        if handler is not None:
            result = handler(workspace=workspace, limit=5)
            return deps.format_candidate_timeline(result), "candidate", None
        timeline = payload.get("candidate_timeline") or payload.get("operations_console", {}).get("candidate")
        return deps.format_candidate_timeline(timeline or {}), "candidate", None
    if normalized in {"queue", "queue summary", "queue compact", "queue sum", "qs"}:
        handler = deps.resolve_handler(service, "train_queue_history")
        result = handler(workspace=workspace, limit=5) if handler is not None else None
        history = deps.coerce_mapping(result) if result is not None else None
        return console_queue_summary_text(payload, history=history, deps=deps), "queue-summary", None
    if normalized in {"queue history", "queue-history", "queue hist"}:
        handler = deps.resolve_handler(service, "train_queue_history")
        if handler is not None:
            result = handler(workspace=workspace, limit=10)
            return deps.format_train_queue_history(result), "queue-history", None
        history = payload.get("train_queue") or {}
        return deps.format_train_queue_history(history), "queue-history", None
    if normalized in {"runner", "runner summary", "runner compact", "runner sum", "rs"}:
        handler = deps.resolve_handler(service, "train_queue_worker_runner_history")
        result = handler(workspace=workspace, limit=5) if handler is not None else None
        history = deps.coerce_mapping(result) if result is not None else None
        return console_runner_summary_text(payload, history=history, deps=deps), "runner-summary", None
    if normalized in {"runner timeline", "runner-timeline", "runner tl"}:
        timeline = payload.get("runner_timeline") or payload.get("operations_console", {}).get("runner_timeline") or {}
        return deps.format_runner_timeline_summary(timeline), "runner-timeline", None
    if normalized in {"runner history", "runner-history", "runner hist"}:
        handler = deps.resolve_handler(service, "train_queue_worker_runner_history")
        if handler is not None:
            result = handler(workspace=workspace, limit=10)
            return deps.format_worker_runner_history(result), "runner-history", None
        history = payload.get("runner_timeline") or {}
        return deps.format_runner_timeline_summary(history), "runner-history", None
    if normalized in {"daemon history", "daemon-history", "daemon hist"}:
        handler = deps.resolve_handler(service, "train_queue_daemon_history", "daemon_history")
        if handler is not None:
            result = handler(workspace=workspace, limit=10)
            return deps.format_train_queue_daemon_history(result), "daemon-history", None
        return (
            deps.format_train_queue_daemon_history(
                deps.read_train_queue_daemon_state(workspace) or {"workspace": workspace or "user_default", "history": []}
            ),
            "daemon-history",
            None,
        )
    if normalized in {"daemon summary", "daemon compact", "daemon sum", "ds"}:
        handler = deps.resolve_handler(service, "train_queue_daemon_status", "daemon_status", "get_daemon_status")
        if handler is not None:
            result = handler(workspace=workspace)
            return console_daemon_summary_text(result, deps=deps), "daemon-summary", None
        return (
            console_daemon_summary_text(
                deps.read_train_queue_daemon_state(workspace) or {"workspace": workspace or "user_default", "command_status": "absent"},
                deps=deps,
            ),
            "daemon-summary",
            None,
        )
    if normalized in {"daemon timeline", "daemon-timeline", "daemon tl"}:
        timeline = payload.get("daemon_timeline") or payload.get("operations_console", {}).get("daemon_timeline") or {}
        return deps.format_daemon_timeline_summary(timeline), "daemon-timeline", None
    if normalized == "daemon":
        handler = deps.resolve_handler(service, "train_queue_daemon_status", "daemon_status", "get_daemon_status")
        if handler is not None:
            result = handler(workspace=workspace)
            return deps.format_train_queue_daemon_status(result), "daemon", None
        return (
            deps.format_train_queue_daemon_status(
                deps.read_train_queue_daemon_state(workspace) or {"workspace": workspace or "user_default", "command_status": "absent"}
            ),
            "daemon",
            None,
        )

    return None


__all__ = ["route_console_view_command"]
