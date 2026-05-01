"""Content construction for the Rich event stream panel."""

from __future__ import annotations

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .console_app_badges import (
    _event_runtime_badges,
    _focus_badge,
    _severity_badge,
    _trigger_category_badge,
    _trigger_category_for_reason,
)
from .console_app_data import _compact_text, _mapping, _prefer_inspection_summary_for_generic_monitor, _value
from .console_app_event_context import EventPanelContext
from .console_app_event_panel_rows import focus_text, severity_text, source_text, status_text


def build_event_panel_content(ctx: EventPanelContext) -> Group:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Sev", severity_text(ctx))
    table.add_row("St", status_text(ctx))
    table.add_row("Src", source_text(ctx))
    table.add_row("F", focus_text(ctx))
    if _prefer_inspection_summary_for_generic_monitor(
        focus=ctx.normalized_focus,
        summary_source="",
        inspection_summary=ctx.inspection_summary,
    ):
        if ctx.inspection_summary:
            table.add_row("I", _compact_text(ctx.inspection_summary, max_len=42))
    return Group(table, Text("R", style="bold dim"), *_recent_lines(ctx))


def _recent_lines(ctx: EventPanelContext) -> list[Text]:
    recent_lines = [_recent_line(item, ctx=ctx) for item in ctx.items[:4]]
    if not recent_lines:
        recent_lines.append(Text("- none", style="dim"))
    return recent_lines


def _recent_line(item: object, *, ctx: EventPanelContext) -> Text:
    mapping = _mapping(item)
    source = _value(mapping, "source", default="ops")
    event = _value(mapping, "event", default="none")
    severity = _value(mapping, "severity", default="stable")
    reason = _value(mapping, "reason", default="none")
    line = Text("- ", style="dim")
    line.append(_compact_text(source, max_len=8), style="bold")
    line.append(":", style="dim")
    line.append(_compact_text(event, max_len=10), style="white")
    line.append(" ", style="dim")
    line.append_text(_severity_badge(severity))
    line.append(" ", style="dim")
    line.append_text(_focus_badge(reason, severity=severity))
    trigger_item_category = _trigger_category_for_reason(
        reason,
        fallback=ctx.trigger_blocked_category if source in {"trigger", "ops", "queue"} else "",
    )
    if trigger_item_category and source in {"trigger", "ops", "queue"}:
        line.append(" ", style="dim")
        line.append_text(_trigger_category_badge(trigger_item_category))
    for badge in _event_runtime_badges(source=source, reason=reason):
        line.append(" ", style="dim")
        line.append_text(badge)
    return line


__all__ = ["build_event_panel_content"]
