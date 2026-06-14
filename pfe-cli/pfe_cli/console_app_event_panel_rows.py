"""Row renderers for the Rich event stream panel."""

from __future__ import annotations

from rich.text import Text

from .console_app_badges import (
    _action_badge,
    _event_runtime_badges,
    _focus_badge,
    _severity_badge,
    _trigger_category_badge,
    _trigger_category_for_reason,
)
from .console_app_data import _value, _yes_no
from .console_app_event_context import EventPanelContext


def severity_text(ctx: EventPanelContext) -> Text:
    severity_text = Text(f"{ctx.stream_severity} ")
    severity_text.append_text(_severity_badge(ctx.stream_severity))
    return severity_text


def focus_text(ctx: EventPanelContext) -> Text:
    focus_text = Text(f"{ctx.focus_value} ")
    focus_text.append_text(_focus_badge(ctx.focus_value, severity=ctx.stream_severity))
    trigger_focus_category = _trigger_category_for_reason(
        ctx.focus_value,
        fallback=ctx.trigger_blocked_category,
    )
    if trigger_focus_category and ctx.latest_source in {"trigger", "ops", "queue"}:
        focus_text.append(" ", style="dim")
        focus_text.append_text(_trigger_category_badge(trigger_focus_category))
    return focus_text


def status_text(ctx: EventPanelContext) -> Text:
    status_text = Text()
    status_text.append(_value(ctx.stream, "status", default="healthy"), style="white")
    status_text.append(" ", style="dim")
    status_text.append(f"attn={_yes_no(ctx.stream.get('attention_needed'))}", style="dim")
    if ctx.action_value not in {"", "observe_and_monitor", "none"}:
        status_text.append(" ", style="dim")
        status_text.append_text(_action_badge(ctx.action_value, priority=ctx.priority_value))
    return status_text


def source_text(ctx: EventPanelContext) -> Text:
    source_text = Text()
    source_text.append(ctx.latest_source, style="white")
    source_text.append(" ", style="dim")
    source_text.append(f"alerts={_value(ctx.stream, 'alert_count', default='0')}", style="dim")
    trigger_source_category = _trigger_category_for_reason(
        ctx.focus_value or ctx.latest_reason,
        fallback=ctx.trigger_blocked_category,
    )
    if trigger_source_category and ctx.latest_source in {"trigger", "ops", "queue"}:
        source_text.append(" ", style="dim")
        source_text.append_text(_trigger_category_badge(trigger_source_category))
    runtime_badges = _event_runtime_badges(source=ctx.latest_source, reason=ctx.focus_value or ctx.latest_reason)
    if runtime_badges:
        source_text.append(" ", style="dim")
        source_text.append_text(runtime_badges[0])
    return source_text


__all__ = ["focus_text", "severity_text", "source_text", "status_text"]
