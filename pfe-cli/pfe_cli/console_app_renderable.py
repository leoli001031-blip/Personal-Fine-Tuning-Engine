"""Rich renderable builder for the operations console."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import RenderableType
from rich.layout import Layout

from .console_app_body_layout import build_console_body_layout
from .console_app_data import _mapping, _value
from .console_app_panels import _status_header
from .console_app_prompt import _footer_digest, _prompt_panel
from .console_app_prompt_rules import _prompt_context_focus


def build_console_renderable(
    payload: Mapping[str, Any],
    *,
    workspace: str | None = None,
    session_messages: Sequence[Mapping[str, Any]] | None = None,
    interactive: bool = False,
    feedback: str | None = None,
    mode: str = "chat",
    prompt_label: str = "chat>",
    model: str | None = None,
    adapter: str | None = None,
    real_local: bool = False,
    refresh_seconds: float | None = None,
    input_active: bool = False,
    input_text: str | None = None,
    input_cursor: int | None = None,
    shortcut_hint: str | None = None,
    ops_refresh_state: str | None = None,
    ops_age_seconds: float | None = None,
) -> RenderableType:
    header = _status_header(payload, workspace=workspace)
    dashboard = _mapping(payload.get("operations_dashboard"))
    severity = _value(dashboard, "severity", default="stable")
    focus = _prompt_context_focus(payload)
    body = build_console_body_layout(
        payload,
        session_messages=session_messages,
        interactive=interactive,
        ops_refresh_state=ops_refresh_state,
        ops_age_seconds=ops_age_seconds,
        refresh_seconds=refresh_seconds,
        severity=severity,
    )
    lower = _prompt_panel(
        feedback=feedback,
        mode=mode,
        prompt_label=prompt_label,
        model=model,
        adapter=adapter,
        real_local=real_local,
        refresh_seconds=refresh_seconds,
        input_active=input_active,
        input_text=input_text,
        input_cursor=input_cursor,
        shortcut_hint=shortcut_hint,
        ops_refresh_state=ops_refresh_state,
        ops_age_seconds=ops_age_seconds,
        focus=focus,
        payload=payload,
    )
    footer = _footer_digest(
        payload,
        interactive=interactive,
        mode=mode,
        ops_refresh_state=ops_refresh_state,
        ops_age_seconds=ops_age_seconds,
    )
    layout = Layout(name="root")
    layout.split_column(
        Layout(header, name="header", size=5),
        Layout(body, name="main", ratio=1),
        Layout(lower, name="prompt", size=3),
        Layout(footer, name="footer", size=1),
    )
    return layout


__all__ = ["build_console_renderable"]
