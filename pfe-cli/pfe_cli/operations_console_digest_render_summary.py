"""Render the top summary line for operations console digests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_formatting_deps import OperationsFormattingDeps


def append_console_digest_summary_line(
    lines: list[str],
    console: Mapping[str, Any],
    *,
    deps: OperationsFormattingDeps,
) -> None:
    """Append the compact operations console digest summary line."""
    digest_parts: list[str] = []
    attention_needed = console.get("attention_needed")
    if attention_needed is not None:
        digest_parts.append(f"attention_needed={deps.format_scalar(attention_needed)}")
    attention_reason = console.get("attention_reason")
    if attention_reason is not None:
        digest_parts.append(f"attention_reason={deps.format_scalar(attention_reason)}")
    current_focus = console.get("current_focus")
    if current_focus is not None:
        digest_parts.append(f"current_focus={deps.format_scalar(current_focus)}")
    required_action = console.get("required_action")
    if required_action is not None:
        digest_parts.append(f"required_action={deps.format_scalar(required_action)}")
    summary_line = console.get("summary_line")
    if summary_line:
        digest_parts.append(f"summary={deps.format_scalar(summary_line)}")
    inspection_summary_line = console.get("inspection_summary_line")
    if inspection_summary_line and inspection_summary_line != summary_line:
        digest_parts.append(f"inspection_summary={deps.format_scalar(inspection_summary_line)}")
    next_actions = console.get("next_actions")
    if next_actions:
        digest_parts.append(f"next_actions={deps.format_scalar(next_actions)}")
    last_recovery_event = console.get("last_recovery_event")
    if last_recovery_event is not None:
        digest_parts.append(f"last_recovery_event={deps.format_scalar(last_recovery_event)}")
    if digest_parts:
        lines.append("  " + " | ".join(digest_parts))


__all__ = ["append_console_digest_summary_line"]
