"""Console snapshot printing helper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import Console

from .console_app_renderable import build_console_renderable


def render_console_snapshot(
    payload: Mapping[str, Any],
    *,
    workspace: str | None = None,
    console: Console | None = None,
    clear: bool = False,
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
) -> None:
    target = console or Console()
    if clear:
        target.clear()
    target.print(
        build_console_renderable(
            payload,
            workspace=workspace,
            session_messages=session_messages,
            interactive=interactive,
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
        )
    )


__all__ = ["render_console_snapshot"]
