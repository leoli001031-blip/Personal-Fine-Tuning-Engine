"""Matrix Green Terminal formatters for PFE CLI output.

Transforms standard PFE CLI output into Terminal Hacker aesthetic.
"""

from __future__ import annotations

from typing import Any

from .matrix_formatting_common import _coerce_mapping
from .matrix_status_execution import append_execution_status_sections
from .matrix_status_model import append_model_status_sections
from .matrix_status_operations import append_operations_status_sections
from .matrix_status_training_control import append_training_control_status_sections
from .terminal_theme import (
    MatrixColors,
    draw_header,
    draw_separator,
)


def format_status_matrix(result: Any, *, workspace: str | None = None) -> str:
    """Format status output in Matrix Green terminal style."""
    lines = []

    # Header
    ws_text = workspace or "default"
    lines.append(draw_header(f"PFE STATUS // WORKSPACE: {ws_text}"))

    mapping = _coerce_mapping(result)
    if mapping is None:
        lines.append(f"{MatrixColors.RED}ERROR: Unable to parse status data{MatrixColors.RESET}")
        return "\n".join(lines)

    append_model_status_sections(lines, mapping)
    append_training_control_status_sections(lines, mapping)
    append_execution_status_sections(lines, mapping)
    append_operations_status_sections(lines, mapping)

    # Footer
    lines.append(draw_separator())
    guided_workspace = workspace or "user_default"
    lines.append(f"{MatrixColors.GREEN_DIM}> Next: pfe next --workspace {guided_workspace}{MatrixColors.RESET}")
    lines.append(f"{MatrixColors.GREEN_DIM}> PFE v2.0 // Matrix Terminal Interface{MatrixColors.RESET}")

    return "\n".join(lines)


__all__ = ["format_status_matrix"]
