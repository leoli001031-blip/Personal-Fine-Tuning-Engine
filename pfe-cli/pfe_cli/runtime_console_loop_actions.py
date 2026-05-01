"""Action resolution for one interactive runtime console loop iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_command_flow import handle_console_command
from .runtime_console_input_refresh import refresh_runtime_console_input
from .runtime_console_state import RuntimeConsoleState, console_command_input


@dataclass(frozen=True)
class ConsoleLoopAction:
    exit_requested: bool = False
    chat_message: str | None = None
    regenerate_mode: bool = False


def resolve_console_loop_action(
    live: Any,
    deps: RuntimeCommandDeps,
    *,
    service: Any,
    handler: Any,
    state: RuntimeConsoleState,
    message: str,
    previous_real_local: str | None,
) -> ConsoleLoopAction:
    command_input = console_command_input(state, message)
    if not command_input.startswith("/"):
        return ConsoleLoopAction(chat_message=message)

    state.feedback = f"running /{command_input[1:].split()[0]}"
    refresh_runtime_console_input(
        live,
        deps,
        message,
        handler=handler,
        state=state,
        input_cursor=len(message),
        current_feedback=state.feedback,
        force_sidebar=False,
    )
    command_decision = handle_console_command(
        deps=deps,
        service=service,
        handler=handler,
        state=state,
        command_input=command_input,
        previous_real_local=previous_real_local,
    )
    if command_decision.exit_requested:
        return ConsoleLoopAction(exit_requested=True)
    if command_decision.regenerate_message is None:
        return ConsoleLoopAction()
    return ConsoleLoopAction(chat_message=command_decision.regenerate_message, regenerate_mode=True)


__all__ = ["ConsoleLoopAction", "resolve_console_loop_action"]
