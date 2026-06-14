"""Help and conversation panels for the Rich operations console."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

from .console_app_data import _compact_text, _mapping, _payload_focus, _value
from .console_app_guidance import _payload_command_guidance


def _chat_help_panel(payload: Mapping[str, Any], *, interactive: bool = False) -> RenderableType:
    latest = _mapping(payload.get("latest_adapter"))
    queue = _mapping(payload.get("train_queue"))
    console = _mapping(payload.get("operations_console"))
    overview = _mapping(payload.get("operations_overview"))
    focus = _payload_focus(payload)
    primary_cmd, secondary_cmd = _payload_command_guidance(payload, focus)
    normalized_focus = str(focus or "").strip().lower()
    trigger_threshold = _mapping(console.get("trigger_threshold_summary")) or _mapping(overview.get("trigger_threshold_summary"))
    trigger_blocked_summary = _value(
        console,
        "trigger_blocked_summary",
        default=_value(overview, "trigger_blocked_summary", default=""),
    )
    trigger_ready = False
    min_samples = trigger_threshold.get("min_new_samples")
    eligible_samples = trigger_threshold.get("effective_eligible_train_samples")
    if eligible_samples is None:
        eligible_samples = trigger_threshold.get("eligible_signal_train_samples")
    if min_samples is not None and eligible_samples is not None:
        try:
            trigger_ready = float(eligible_samples) >= float(min_samples)
        except (ValueError, TypeError):
            trigger_ready = False
    header_parts = [f"lat={latest.get('version') or 'none'}", f"q={queue.get('count', 0)}"]
    if normalized_focus in {
        "insufficient_new_signal_samples",
        "holdout_not_ready",
        "cooldown_active",
        "failure_backoff_active",
    }:
        header_parts.append(
            "gate="
            + _compact_text(
                _value(trigger_threshold, "summary_line", default=trigger_blocked_summary or "trigger gate active"),
                max_len=28,
            )
        )
    elif trigger_ready and normalized_focus in {
        "policy_requires_auto_evaluate",
        "trigger_ready",
        "awaiting_signal_trigger",
    }:
        header_parts.append(f"ready={eligible_samples}/{min_samples}")
    lines = [" | ".join(header_parts), f"do: {primary_cmd}", f"see: {secondary_cmd}"]
    if trigger_ready and normalized_focus in {
        "policy_requires_auto_evaluate",
        "trigger_ready",
        "awaiting_signal_trigger",
    }:
        lines.append("action: /do trigger-train")
    if interactive:
        lines.extend(
            [
                "chat=text | cmd=/do /see",
                "slash=/help /cmd /quit",
            ]
        )
    return Panel(Text("\n".join(lines)), border_style="green")


def _conversation_panel(session_messages: Sequence[Mapping[str, Any]] | None = None) -> RenderableType:
    items = list(session_messages or [])
    lines: list[str] = []
    for item in items[-8:]:
        role = str(item.get("role") or "system")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        label = "user" if role == "user" else "assistant" if role == "assistant" else role
        lines.append(f"{label}> {content}")
    if not lines:
        lines.append("No conversation yet.")
    return Panel(Text("\n\n".join(lines)), title="Conversation", border_style="blue")


__all__ = ["_chat_help_panel", "_conversation_panel"]
