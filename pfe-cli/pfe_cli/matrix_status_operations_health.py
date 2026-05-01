"""Operations health status sections for Matrix terminal output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping, _coerce_sequence_of_scalars
from .terminal_theme import draw_box, format_key_value


def append_operations_health_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append operations health, recovery, and next action boxes."""
    operations_health = _coerce_mapping(mapping.get("operations_health"))
    if operations_health:
        oh_content = []
        for key, value in operations_health.items():
            if value is not None:
                oh_content.append(format_key_value(key.replace("_", " "), value))
        if oh_content:
            lines.append(draw_box("OPERATIONS HEALTH", oh_content))
            lines.append("")

    operations_recovery = _coerce_mapping(mapping.get("operations_recovery"))
    if operations_recovery:
        or_content = []
        for key, value in operations_recovery.items():
            if value is not None:
                or_content.append(format_key_value(key.replace("_", " "), value))
        if or_content:
            lines.append(draw_box("OPERATIONS RECOVERY", or_content))
            lines.append("")

    operations_next_actions = _coerce_sequence_of_scalars(mapping.get("operations_next_actions"))
    if operations_next_actions:
        lines.append(draw_box("NEXT ACTIONS", [", ".join(str(a) for a in operations_next_actions)]))
        lines.append("")


__all__ = ["append_operations_health_sections"]
