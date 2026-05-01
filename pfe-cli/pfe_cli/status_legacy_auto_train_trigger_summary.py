"""Legacy auto-train trigger summary formatting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .status_legacy_auto_train_trigger_fields import TRIGGER_KEYS, append_scalar_parts


def append_trigger_summary_lines(lines: list[str], auto_trigger: Mapping[str, Any], *, deps: Any) -> None:
    trigger_parts: list[str] = []
    for key in ("enabled", "state", "ready"):
        value = auto_trigger.get(key)
        if value is not None:
            trigger_parts.append(f"{key}={deps.format_scalar(value)}")
    reason = auto_trigger.get("reason")
    if reason:
        trigger_parts.append(f"reason={deps.format_scalar(reason)}")
    append_scalar_parts(trigger_parts, auto_trigger, TRIGGER_KEYS, deps=deps)

    blocked_reasons = auto_trigger.get("blocked_reasons")
    if blocked_reasons:
        if isinstance(blocked_reasons, Sequence) and not isinstance(blocked_reasons, (str, bytes, bytearray)):
            trigger_parts.append(f"blocked_reasons={list(blocked_reasons)!r}")
        else:
            trigger_parts.append(f"blocked_reasons={deps.format_scalar(blocked_reasons)}")
    if trigger_parts:
        lines.append("auto train trigger: " + " | ".join(trigger_parts))


__all__ = ["append_trigger_summary_lines"]
