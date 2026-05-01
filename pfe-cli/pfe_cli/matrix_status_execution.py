"""Execution and system health sections for Matrix terminal status output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_status_execution_sections import (
    append_export_toolchain_box,
    append_real_execution_box,
    append_system_health_box,
)


def append_execution_status_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append execution, export, and system health status boxes."""
    append_real_execution_box(lines, mapping)
    append_export_toolchain_box(lines, mapping)
    append_system_health_box(lines, mapping)


__all__ = ["append_execution_status_sections"]
