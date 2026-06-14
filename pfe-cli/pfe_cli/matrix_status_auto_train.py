"""Auto-train status sections for Matrix terminal status output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_status_auto_train_action import append_auto_train_action_section
from .matrix_status_auto_train_trigger import append_auto_train_trigger_section


def append_auto_train_status_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append auto-train trigger and action status boxes."""
    append_auto_train_trigger_section(lines, mapping)
    append_auto_train_action_section(lines, mapping)


__all__ = ["append_auto_train_status_sections"]
