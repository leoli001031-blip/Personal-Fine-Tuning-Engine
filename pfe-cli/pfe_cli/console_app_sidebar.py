"""Sidebar assembly for the Rich operations console."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from .console_app_badges import _ops_badge, _ops_state_badge, _prompt_badge, _section_label, _severity_badge
from .console_app_data import _mapping, _value
from .console_app_panels import _chat_help_panel, _event_stream_panel, _operations_panel
from .console_app_prompt import _sidebar_snapshot_text


def build_sidebar_panel(
    payload: Mapping[str, Any],
    *,
    interactive: bool,
    ops_refresh_state: str | None,
    ops_age_seconds: float | None,
    refresh_seconds: float | None,
    severity: Any,
) -> Panel:
    sidebar_header = Text("Operations Sidebar ", style="bold")
    sidebar_header.append_text(_ops_badge(ops_refresh_state, severity=severity))
    if ops_age_seconds is not None:
        sidebar_header.append(f" {ops_age_seconds:.1f}s", style="dim")
    return Panel(
        _sidebar_group(
            payload,
            interactive=interactive,
            ops_refresh_state=ops_refresh_state,
            ops_age_seconds=ops_age_seconds,
            refresh_seconds=refresh_seconds,
            severity=severity,
        ),
        title=sidebar_header,
        border_style="bright_black",
    )


def _sidebar_group(
    payload: Mapping[str, Any],
    *,
    interactive: bool,
    ops_refresh_state: str | None,
    ops_age_seconds: float | None,
    refresh_seconds: float | None,
    severity: Any,
) -> Group:
    event_stream = _event_stream_panel(payload)
    event_stream_title = Text("Operations Event Stream ", style="bold")
    event_stream_title.append_text(_ops_badge(ops_refresh_state, severity=severity))
    event_stream.title = event_stream_title

    help_panel = _chat_help_panel(payload, interactive=interactive)
    help_title = Text("Help ", style="bold")
    help_title.append_text(_ops_state_badge(ops_refresh_state))
    help_panel.title = help_title

    return Group(
        _section_label("Snapshot", badge=_ops_badge(ops_refresh_state, severity=severity)),
        _sidebar_snapshot_text(
            ops_refresh_state=ops_refresh_state,
            ops_age_seconds=ops_age_seconds,
            refresh_seconds=refresh_seconds,
        ),
        _section_label("Operations", badge=_ops_badge(ops_refresh_state, severity=severity)),
        _operations_panel(payload),
        _section_label(
            "Event Stream",
            badge=_severity_badge(
                _value(_mapping(payload.get("operations_event_stream")), "severity", default="stable")
            ),
        ),
        event_stream,
        _section_label("Help", badge=_prompt_badge("shortcuts", "bold black on bright_white")),
        help_panel,
    )


__all__ = ["build_sidebar_panel"]
