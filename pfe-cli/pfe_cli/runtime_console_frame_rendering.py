"""Frame rendering for the interactive runtime console."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .console_app import build_console_renderable
from .runtime_command_deps import RuntimeCommandDeps


def interactive_prompt_label(mode_name: str) -> str:
    return "cmd>" if mode_name == "command" else "chat>"


def render_console_frame(
    live: Any,
    deps: RuntimeCommandDeps,
    *,
    payload: Mapping[str, Any],
    workspace: str | None,
    transcript: Sequence[Mapping[str, str]],
    feedback: str | None,
    mode_name: str,
    model: str,
    adapter: str,
    real_local: bool,
    refresh_seconds: float,
    input_active: bool,
    input_text: str = "",
    input_cursor: int | None = None,
    shortcut_hint: str | None = None,
    ops_refresh_state: str | None = None,
    ops_age_seconds: float | None = None,
) -> None:
    live.update(
        build_console_renderable(
            payload,
            workspace=workspace,
            session_messages=transcript,
            interactive=True,
            feedback=feedback,
            mode=mode_name,
            prompt_label=interactive_prompt_label(mode_name),
            model=model,
            adapter=adapter,
            real_local=real_local,
            refresh_seconds=refresh_seconds,
            input_active=input_active,
            input_text=input_text,
            input_cursor=input_cursor,
            shortcut_hint=shortcut_hint if shortcut_hint is not None else deps.console_shortcut_hint(mode_name, payload),
            ops_refresh_state=ops_refresh_state,
            ops_age_seconds=ops_age_seconds,
        )
    )
    live.refresh()


__all__ = ["interactive_prompt_label", "render_console_frame"]
