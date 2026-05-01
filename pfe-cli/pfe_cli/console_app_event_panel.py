"""Event stream panel for the Rich operations console."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.console import RenderableType
from rich.panel import Panel

from .console_app_event_context import build_event_panel_context
from .console_app_event_panel_content import build_event_panel_content


def _event_stream_panel(payload: Mapping[str, Any]) -> RenderableType:
    context = build_event_panel_context(payload)
    return Panel(build_event_panel_content(context), border_style="magenta")


__all__ = ["_event_stream_panel"]
