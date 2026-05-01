"""Rich text cells for the operations panel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.text import Text

from .console_app_badges import _action_badge, _focus_badge, _trigger_category_badge
from .console_app_data import _compact_text
from .console_app_guidance import _payload_command_guidance
from .console_app_operations_context import OperationsPanelContext


def _focus_text(ctx: OperationsPanelContext) -> Text:
    focus_text = Text(f"{_compact_text(ctx.focus_value, max_len=20)} ")
    focus_text.append_text(_focus_badge(ctx.focus_value, severity=ctx.severity))
    return focus_text


def _action_text(ctx: OperationsPanelContext) -> Text:
    action_text = Text(f"{_compact_text(ctx.action_value, max_len=20)} ")
    action_text.append_text(_action_badge(ctx.action_value, priority=ctx.priority_value))
    return action_text


def _trigger_text(ctx: OperationsPanelContext) -> Text:
    trigger_text = Text()
    trigger_text.append_text(_trigger_category_badge(ctx.trigger_blocked_category))
    trigger_text.append(" | ", style="dim")
    trigger_text.append(_compact_text(ctx.trigger_blocked_reason, max_len=18), style="white")
    trigger_text.append(" | ", style="dim")
    trigger_text.append(_compact_text(ctx.trigger_blocked_action, max_len=22), style="white")
    return trigger_text


def _action_command_text(payload: Mapping[str, Any], ctx: OperationsPanelContext) -> tuple[Text, Text]:
    primary_cmd, secondary_cmd = _payload_command_guidance(payload, ctx.focus_value)
    do_text = Text()
    do_text.append_text(_action_badge(ctx.action_value, priority=ctx.priority_value))
    do_text.append(" ", style="dim")
    do_text.append(_compact_text(primary_cmd, max_len=20), style="bold cyan")
    return do_text, Text(_compact_text(secondary_cmd, max_len=22), style="dim")


__all__ = ["_action_command_text", "_action_text", "_focus_text", "_trigger_text"]
