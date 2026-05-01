"""Main body layout assembly for the Rich operations console."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.layout import Layout

from .console_app_panels import _conversation_panel
from .console_app_sidebar import build_sidebar_panel


def build_console_body_layout(
    payload: Mapping[str, Any],
    *,
    session_messages: Sequence[Mapping[str, Any]] | None,
    interactive: bool,
    ops_refresh_state: str | None,
    ops_age_seconds: float | None,
    refresh_seconds: float | None,
    severity: Any,
) -> Layout:
    body = Layout(name="body")
    body.split_row(
        Layout(_conversation_panel(session_messages), name="transcript", ratio=3),
        Layout(
            build_sidebar_panel(
                payload,
                interactive=interactive,
                ops_refresh_state=ops_refresh_state,
                ops_age_seconds=ops_age_seconds,
                refresh_seconds=refresh_seconds,
                severity=severity,
            ),
            name="sidebar",
            size=54,
        ),
    )
    return body


__all__ = ["build_console_body_layout"]
