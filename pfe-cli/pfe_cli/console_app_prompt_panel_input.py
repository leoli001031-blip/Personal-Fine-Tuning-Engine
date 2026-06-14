"""Input text segment rendering for the Rich console prompt panel."""

from __future__ import annotations

from rich.markup import escape
from rich.text import Text

from .console_app_prompt_panel_state import PromptPanelState


def append_prompt_input_segment(
    bar: Text,
    *,
    prompt_label: str,
    input_active: bool,
    state: PromptPanelState,
) -> None:
    bar.append("> ", style="bold cyan")
    bar.append(prompt_label, style="bold cyan")
    bar.append(" ")
    if input_active and state.raw_input:
        before = state.raw_input[: state.cursor_index]
        current = state.raw_input[state.cursor_index : state.cursor_index + 1] or " "
        after = state.raw_input[state.cursor_index + 1 :] if state.cursor_index < len(state.raw_input) else ""
        if before:
            bar.append(escape(before), style="white")
        bar.append(current, style="black on white")
        if after:
            bar.append(escape(after), style="white")
    elif state.preview:
        bar.append(escape(state.preview), style="white")
    elif state.placeholder:
        bar.append(state.placeholder, style="dim italic")
    else:
        bar.append("...", style="dim")


__all__ = ["append_prompt_input_segment"]
