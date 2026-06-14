"""Single-iteration helpers for the interactive runtime console loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import typer

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_input_refresh import refresh_runtime_console_input
from .runtime_console_loop_actions import ConsoleLoopAction, resolve_console_loop_action
from .runtime_console_rendering import interactive_prompt_label
from .runtime_console_state import RuntimeConsoleState, remember_console_input


@dataclass(frozen=True)
class ConsoleReadResult:
    exit_requested: bool = False
    message: str | None = None


def read_console_loop_message(
    live: Any,
    console_ui: Any,
    deps: RuntimeCommandDeps,
    *,
    handler: Any,
    state: RuntimeConsoleState,
) -> ConsoleReadResult:
    prompt_label = interactive_prompt_label(state.mode_name)
    refresh_runtime_console_input(
        live,
        deps,
        handler=handler,
        state=state,
        input_cursor=0,
        current_feedback=state.feedback,
        force_sidebar=True,
    )
    try:
        user_text = deps.console_read_input(
            prompt_label,
            refresh_seconds=state.refresh_seconds,
            refresh_callback=lambda current, cursor: refresh_runtime_console_input(
                live,
                deps,
                current,
                handler=handler,
                state=state,
                input_cursor=cursor,
                current_feedback=state.feedback,
            ),
            history=state.input_history,
        )
    except (typer.Abort, EOFError, KeyboardInterrupt):
        typer.echo("Exiting PFE Console.")
        return ConsoleReadResult(exit_requested=True)

    console_ui.print("")
    message = str(user_text or "").strip()
    if not message:
        state.feedback = "empty input"
        return ConsoleReadResult()

    remember_console_input(state, message)
    return ConsoleReadResult(message=message)


__all__ = [
    "ConsoleLoopAction",
    "ConsoleReadResult",
    "read_console_loop_message",
    "resolve_console_loop_action",
]
