"""Guidance segment rendering for the Rich console prompt panel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.markup import escape
from rich.text import Text

from .console_app_badges import _trigger_category_badge
from .console_app_prompt_panel_state import PromptPanelState
from .console_app_prompt_rules import (
    _prompt_action_guidance,
    _prompt_ctx_digest,
    _prompt_feedback_digest,
    _prompt_mode_help,
    _prompt_target_hint,
    _prompt_trigger_category,
)


def append_prompt_guidance_segments(
    bar: Text,
    *,
    mode: str,
    focus: str | None,
    payload: Mapping[str, Any] | None,
    state: PromptPanelState,
) -> None:
    bar.append(" ", style="dim")
    bar.append("o=", style="dim")
    bar.append(_prompt_target_hint(mode, focus=focus, payload=payload), style="bold cyan")

    ctx_digest = _prompt_ctx_digest(focus)
    if ctx_digest:
        bar.append(" ", style="dim")
        bar.append("x=", style="dim")
        bar.append(ctx_digest, style="bold white")

    trigger_category = _prompt_trigger_category(focus, payload=payload)
    if trigger_category:
        bar.append(" ", style="dim")
        bar.append("tg=", style="dim")
        bar.append_text(_trigger_category_badge(trigger_category))

    primary_action_hint, secondary_action_hint = _prompt_action_guidance(
        mode,
        focus=focus,
        shortcut_hint=state.effective_hint,
        payload=payload,
    )
    if primary_action_hint:
        bar.append(" ", style="dim")
        bar.append("d=", style="dim")
        bar.append(escape(primary_action_hint), style="bold cyan")
    if secondary_action_hint:
        bar.append(" ", style="dim")
        bar.append("s=", style="dim")
        bar.append(escape(secondary_action_hint), style="dim")

    recent_digest = _prompt_feedback_digest(state.recent_action)
    if recent_digest and recent_digest != "idle":
        bar.append(" ", style="dim")
        bar.append(f"l={escape(recent_digest)}", style="bold yellow")

    mode_help = _prompt_mode_help(mode, focus=focus, payload=payload)
    bar.append(" ", style="dim")
    bar.append("?=", style="dim")
    bar.append(mode_help, style="bold cyan")


__all__ = ["append_prompt_guidance_segments"]
