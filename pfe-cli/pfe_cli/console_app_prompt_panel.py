"""Prompt input panel rendering for the Rich operations console."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.box import HEAVY
from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

from .console_app_prompt_panel_input import append_prompt_input_segment
from .console_app_prompt_panel_segments import (
    append_prompt_guidance_segments,
    append_prompt_runtime_segments,
    append_prompt_status_segments,
)
from .console_app_prompt_panel_state import build_prompt_panel_state


def _prompt_panel(
    *,
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
    focus: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> RenderableType:
    state = build_prompt_panel_state(
        feedback=feedback,
        mode=mode,
        input_active=input_active,
        input_text=input_text,
        input_cursor=input_cursor,
        shortcut_hint=shortcut_hint,
        focus=focus,
        payload=payload,
    )
    bar = Text()
    append_prompt_input_segment(bar, prompt_label=prompt_label, input_active=input_active, state=state)
    append_prompt_status_segments(
        bar,
        feedback=feedback,
        mode=mode,
        model=model,
        adapter=adapter,
        real_local=real_local,
        refresh_seconds=refresh_seconds,
        input_active=input_active,
        state=state,
    )
    append_prompt_guidance_segments(bar, mode=mode, focus=focus, payload=payload, state=state)
    append_prompt_runtime_segments(bar, ops_refresh_state=ops_refresh_state, focus=focus, payload=payload)
    return Panel(bar, border_style="white", box=HEAVY)


__all__ = ["_prompt_panel"]
