"""Footer digest rendering for the Rich operations console."""

from __future__ import annotations

from rich.text import Text

from .console_app_badges import _action_badge, _focus_badge, _trigger_category_badge
from .console_app_data import _compact_text
from .console_app_prompt_footer_state import FooterDigestState


def render_footer_digest(state: FooterDigestState, *, ops_age_seconds: float | None = None) -> Text:
    line = Text()
    append_focus_segment(line, state)
    line.append(" · ", style="dim")
    append_action_segment(line, state)
    if state.runtime_mode:
        return line

    append_standard_footer_segments(line, state, ops_age_seconds=ops_age_seconds)
    return line


def append_focus_segment(line: Text, state: FooterDigestState) -> None:
    line.append("f=", style="dim")
    if state.runtime_mode:
        line.append(state.full_focus, style="white")
        line.append(" ", style="dim")
        line.append_text(_focus_badge(state.full_focus, severity=state.severity))
        return

    line.append(_compact_text(state.focus, max_len=16), style="white")
    line.append(" ", style="dim")
    line.append_text(_focus_badge(state.focus, severity=state.severity))


def append_action_segment(line: Text, state: FooterDigestState) -> None:
    line.append("a=", style="dim")
    if state.runtime_mode:
        line.append(state.full_action, style="white")
        line.append(" ", style="dim")
        line.append_text(_action_badge(state.full_action, priority=state.action_priority))
        return

    line.append(state.action, style="white")
    line.append(" ", style="dim")
    line.append_text(_action_badge(state.action, priority=state.action_priority))


def append_standard_footer_segments(
    line: Text,
    state: FooterDigestState,
    *,
    ops_age_seconds: float | None = None,
) -> None:
    line.append(" · ", style="dim")
    line.append("h=", style="dim")
    line.append(state.handling, style="cyan")
    if state.trigger_category:
        line.append(" · ", style="dim")
        line.append("tg=", style="dim")
        line.append_text(_trigger_category_badge(state.trigger_category))
    for badge in state.runtime_badges[:2]:
        line.append(" · ", style="dim")
        line.append_text(badge)
    line.append(" · ", style="dim")
    line.append("o=", style="dim")
    line.append_text(state.status_badge)
    line.append(" · ", style="dim")
    line.append("d=", style="dim")
    line.append(state.primary_action_hint, style="bold cyan")
    line.append(" · ", style="dim")
    line.append("s=", style="dim")
    line.append(state.secondary_action_hint, style="dim")
    if ops_age_seconds is not None:
        line.append(" · ", style="dim")
        line.append(f"t={ops_age_seconds:.1f}s", style="dim")


__all__ = ["render_footer_digest"]
