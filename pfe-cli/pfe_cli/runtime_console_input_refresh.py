"""State-aware input refresh helper for the interactive runtime console."""

from __future__ import annotations

from typing import Any

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_rendering import refresh_input_console
from .runtime_console_state import RuntimeConsoleState


def refresh_runtime_console_input(
    live: Any,
    deps: RuntimeCommandDeps,
    input_text: str = "",
    *,
    handler: Any,
    state: RuntimeConsoleState,
    input_cursor: int | None = None,
    current_feedback: str | None = None,
    force_sidebar: bool = False,
) -> None:
    state.payload, state.last_sidebar_refresh_at, state.ops_refresh_state = refresh_input_console(
        live,
        deps,
        handler=handler,
        payload=state.payload,
        workspace=state.workspace,
        transcript=state.transcript,
        feedback=current_feedback if current_feedback is not None else state.feedback,
        mode_name=state.mode_name,
        model=state.model,
        adapter=state.adapter,
        real_local=state.real_local,
        refresh_seconds=state.refresh_seconds,
        last_sidebar_refresh_at=state.last_sidebar_refresh_at,
        input_text=input_text,
        input_cursor=input_cursor,
        force_sidebar=force_sidebar,
    )


__all__ = ["refresh_runtime_console_input"]
