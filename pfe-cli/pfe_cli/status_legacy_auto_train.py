"""Legacy plain-text auto-train status formatting."""

from __future__ import annotations

from typing import Any

from .status_legacy_auto_train_action import append_legacy_auto_train_action_lines
from .status_legacy_auto_train_trigger import append_legacy_auto_train_trigger_lines


def append_legacy_auto_train_lines(
    lines: list[str],
    mapping: dict[str, Any],
    *,
    deps: Any,
) -> None:
    """Append legacy auto-train trigger and action lines."""

    append_legacy_auto_train_action_lines(lines, mapping, deps=deps)
    append_legacy_auto_train_trigger_lines(lines, mapping, deps=deps)


__all__ = ["append_legacy_auto_train_lines"]
