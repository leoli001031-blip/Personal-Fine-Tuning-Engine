"""State preparation for the Rich console prompt panel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .console_app_prompt_rules import _prompt_placeholder


@dataclass(frozen=True)
class PromptPanelState:
    preview: str
    raw_input: str
    cursor_index: int
    placeholder: str
    recent_action: str
    effective_hint: str


def build_prompt_panel_state(
    *,
    feedback: str | None,
    mode: str,
    input_active: bool,
    input_text: str | None,
    input_cursor: int | None,
    shortcut_hint: str | None,
    focus: str | None,
    payload: Mapping[str, Any] | None,
) -> PromptPanelState:
    preview = (input_text or "").strip()
    raw_input = str(input_text or "")
    cursor_index = max(0, min(int(input_cursor if input_cursor is not None else len(raw_input)), len(raw_input)))
    if len(preview) > 36:
        preview = preview[:33] + "..."
    if input_active and not preview:
        placeholder = _prompt_placeholder(mode, focus=focus, payload=payload)
    else:
        placeholder = ""

    recent_action = (feedback or "").strip()
    if len(recent_action) > 42:
        recent_action = recent_action[:39] + "..."

    effective_hint = shortcut_hint
    if effective_hint is None:
        effective_hint = "Enter,/help,^C" if mode == "chat" else "/status,/candidate,/daemon,/chat"

    return PromptPanelState(
        preview=preview,
        raw_input=raw_input,
        cursor_index=cursor_index,
        placeholder=placeholder,
        recent_action=recent_action,
        effective_hint=effective_hint,
    )


__all__ = ["PromptPanelState", "build_prompt_panel_state"]
