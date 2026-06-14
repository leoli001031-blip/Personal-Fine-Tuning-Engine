"""Status segment rendering for the Rich console prompt panel."""

from __future__ import annotations

from rich.text import Text

from .console_app_badges import _prompt_badge
from .console_app_prompt_badges import _activity_badge, _edit_state_badge, _prompt_state_badge
from .console_app_prompt_panel_state import PromptPanelState
from .console_app_prompt_rules import _prompt_adapter_digest, _prompt_model_digest


def append_prompt_status_segments(
    bar: Text,
    *,
    feedback: str | None,
    mode: str,
    model: str | None,
    adapter: str | None,
    real_local: bool,
    refresh_seconds: float | None,
    input_active: bool,
    state: PromptPanelState,
) -> None:
    bar.append("  ", style="dim")
    bar.append_text(_prompt_badge(mode, "bold white on dark_green" if mode == "chat" else "bold white on dark_blue"))
    bar.append(" ")
    bar.append_text(
        _prompt_badge("real-local", "bold white on dark_magenta")
        if real_local
        else _prompt_badge("local", "bold black on bright_white")
    )
    bar.append(" ")
    bar.append_text(_prompt_state_badge(mode=mode, input_active=input_active, preview=state.preview, feedback=feedback))
    bar.append(" ")
    bar.append_text(
        _edit_state_badge(
            input_active=input_active,
            raw_input=state.raw_input,
            cursor_index=state.cursor_index,
        )
    )
    bar.append(" ")
    bar.append_text(_activity_badge(mode=mode, input_active=input_active, feedback=feedback))
    bar.append("  ", style="dim")
    bar.append(f"m={_prompt_model_digest(model)}", style="dim")
    bar.append(" ", style="dim")
    bar.append(f"a={_prompt_adapter_digest(adapter)}", style="dim")
    if refresh_seconds is not None:
        bar.append(" ", style="dim")
        bar.append(f"r={refresh_seconds:.1f}s", style="dim")
    if input_active:
        bar.append(" ", style="dim")
        bar.append(f"c={state.cursor_index}", style="dim")
        bar.append("/", style="dim")
        bar.append(str(len(state.raw_input)), style="dim")


__all__ = ["append_prompt_status_segments"]
