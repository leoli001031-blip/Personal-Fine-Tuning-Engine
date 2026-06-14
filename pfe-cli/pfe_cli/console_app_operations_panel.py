"""Operations summary panel for the Rich operations console."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.console import RenderableType
from rich.panel import Panel

from .console_app_operations_context import build_operations_panel_context
from .console_app_operations_panel_rows import build_operations_panel_table


def _operations_panel(payload: Mapping[str, Any]) -> RenderableType:
    context = build_operations_panel_context(payload)
    return Panel(build_operations_panel_table(payload, context), border_style="yellow")


__all__ = ["_operations_panel"]
