"""Operations status sections for Matrix terminal status output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_status_operation_timelines import append_operation_timeline_sections
from .matrix_status_operations_console import append_operations_console_sections
from .matrix_status_operations_health import append_operations_health_sections
from .matrix_status_operations_overview import append_operations_overview_sections


def append_operations_status_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append operations-related status boxes to the provided line buffer."""
    append_operations_overview_sections(lines, mapping)
    append_operations_console_sections(lines, mapping)
    append_operation_timeline_sections(lines, mapping)
    append_operations_health_sections(lines, mapping)


__all__ = ["append_operations_status_sections"]
