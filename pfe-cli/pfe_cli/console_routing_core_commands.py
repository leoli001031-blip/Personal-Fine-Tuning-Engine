"""Core console slash-command routing."""

from __future__ import annotations

from .console_routing_context import ConsoleCommandResult, ConsoleRouteContext
from .console_routing_summaries import (
    console_gate_summary_text,
    console_runtime_summary_text,
    console_trigger_summary_text,
)


def route_console_core_command(ctx: ConsoleRouteContext) -> ConsoleCommandResult | None:
    """Route mode, feedback, status, and operations summary commands."""
    normalized = ctx.normalized
    deps = ctx.deps

    if normalized in {"quit", "exit"}:
        return None, "quit", None
    if normalized == "help":
        return deps.console_help_text(), "help", None
    if normalized == "chat":
        return None, "mode:chat", None
    if normalized in {"cmd", "command"}:
        return None, "mode:command", None
    if normalized.startswith("mode "):
        selected_mode = normalized.split(" ", 1)[1].strip()
        if selected_mode in {"chat", "command"}:
            return None, f"mode:{selected_mode}", None
        return "Unknown mode. Use /mode chat or /mode command.", "unknown", None
    if normalized in {"like", "good", "yes"}:
        if ctx.last_interaction is None:
            return "No previous interaction to accept. Start a chat first.", "like", None
        _submit_feedback(ctx, action="continue")
        return "Accepted previous assistant response.", "like", None
    if normalized in {"dislike", "bad", "no"}:
        if ctx.last_interaction is None:
            return "No previous interaction to reject. Start a chat first.", "dislike", None
        _submit_feedback(ctx, action="delete")
        return "Rejected previous assistant response.", "dislike", None
    if normalized in {"again", "redo"}:
        if ctx.last_interaction is None:
            return "No previous interaction to regenerate. Start a chat first.", "again", None
        _submit_feedback(ctx, action="regenerate")
        return "Requested regeneration for previous response.", "again", {"regenerate": True}
    if normalized.startswith("fix "):
        if ctx.last_interaction is None:
            return "No previous interaction to edit. Start a chat first.", "fix", None
        edited_text = normalized[4:].strip()
        if not edited_text:
            return "Usage: /fix <corrected text>", "fix", None
        _submit_feedback(ctx, action="edit", edited_text=edited_text)
        return f"Submitted edit: {edited_text}", "fix", {"edited_text": edited_text}
    if normalized in {"status compact", "status summary", "sum"}:
        return deps.console_status_compact_text(ctx.payload, workspace=ctx.current_workspace or ctx.workspace), "status-compact", None
    if normalized == "status":
        return deps.format_status(dict(ctx.payload), workspace=ctx.workspace), "status", None
    if normalized in {"ops dashboard", "ops summary", "ops dash", "dash"}:
        dashboard_lines = deps.format_operations_dashboard(ctx.payload.get("operations_dashboard")) or ["operations dashboard: none"]
        return "\n".join(dashboard_lines), "ops-dashboard", None
    if normalized in {"ops alerts", "alerts"}:
        alert_lines = deps.format_operations_alert_surface(
            {
                "operations_alerts": ctx.payload.get("operations_alerts"),
                "operations_health": ctx.payload.get("operations_health"),
                "operations_recovery": ctx.payload.get("operations_recovery"),
                "operations_next_actions": ctx.payload.get("operations_next_actions"),
                "operations_dashboard": ctx.payload.get("operations_dashboard"),
                "operations_alert_policy": ctx.payload.get("operations_alert_policy"),
                "operations_console": ctx.payload.get("operations_console"),
                "operations_overview": ctx.payload.get("operations_overview"),
            }
        ) or ["operations alerts: none"]
        return "\n".join(alert_lines), "ops-alerts", None
    if normalized in {"ops policy", "ops pol", "alert policy", "policy"}:
        policy_lines = deps.format_operations_alert_policy(ctx.payload.get("operations_alert_policy")) or ["operations alert policy: none"]
        return "\n".join(policy_lines), "ops-policy", None
    if normalized in {"trigger", "trigger summary", "trig"}:
        return console_trigger_summary_text(ctx.payload, deps=deps), "trigger-summary", None
    if normalized in {"gate", "gate summary", "gates"}:
        return console_gate_summary_text(ctx.payload, deps=deps), "gate-summary", None
    if normalized in {"runtime", "runtime summary", "stability", "rt"}:
        return console_runtime_summary_text(ctx.payload, deps=deps), "runtime-summary", None
    if normalized in {"event stream", "ops event-stream", "ops events", "event"}:
        stream_lines = deps.format_operations_event_stream(ctx.payload.get("operations_event_stream")) or ["operations event stream: none"]
        return "\n".join(stream_lines), "event-stream", None
    if normalized in {"ops", "os"}:
        ops_lines = deps.format_operations_console_digest(dict(ctx.payload)) or ["operations console digest: none"]
        return "\n".join(ops_lines), "ops", None

    return None


def _submit_feedback(ctx: ConsoleRouteContext, *, action: str, edited_text: str | None = None) -> None:
    last_interaction = ctx.last_interaction or {}
    kwargs = {
        "workspace": ctx.workspace or ctx.current_workspace or "user_default",
        "session_id": last_interaction.get("session_id", ""),
        "request_id": last_interaction.get("request_id", ""),
        "user_message": last_interaction.get("user_message", ""),
        "assistant_message": last_interaction.get("assistant_message", ""),
        "response_time_seconds": last_interaction.get("response_time_seconds", 0.0),
        "adapter_version": last_interaction.get("adapter_version", ctx.adapter),
        "action": action,
    }
    if edited_text is not None:
        kwargs["edited_text"] = edited_text
    ctx.deps.console_submit_feedback(**kwargs)


__all__ = ["route_console_core_command"]
