"""Current training status section for Matrix terminal output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping
from .terminal_theme import draw_box, format_key_value, progress_bar, status_badge


def append_current_training_section(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append current training status box."""
    current_training = _coerce_mapping(mapping.get("current_training"))
    if current_training:
        training_content = []
        status = current_training.get("status", "idle")
        version = current_training.get("version", "n/a")

        training_content.append(format_key_value("status", status_badge(status)))
        training_content.append(format_key_value("version", version))

        epochs = current_training.get("epochs", 0)
        current_epoch = current_training.get("current_epoch", 0)
        if epochs > 0:
            training_content.append(format_key_value("progress", progress_bar(current_epoch, epochs)))

        lines.append(draw_box("TRAINING STATUS", training_content))
        lines.append("")


__all__ = ["append_current_training_section"]
