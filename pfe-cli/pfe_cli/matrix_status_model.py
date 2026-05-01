"""Model and training status sections for Matrix terminal status output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_status_adapter import append_adapter_lifecycle_section
from .matrix_status_capabilities import append_capability_status_sections
from .matrix_status_current_training import append_current_training_section
from .matrix_status_signal import append_signal_readiness_section


def append_model_status_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append model, signal, and current training status boxes."""
    append_adapter_lifecycle_section(lines, mapping)
    append_capability_status_sections(lines, mapping)
    append_signal_readiness_section(lines, mapping)
    append_current_training_section(lines, mapping)


__all__ = ["append_model_status_sections"]
