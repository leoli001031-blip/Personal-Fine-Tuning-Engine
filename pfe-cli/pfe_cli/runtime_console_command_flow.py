"""Slash-command side effects for the interactive runtime console."""

from __future__ import annotations

from typing import Any

import typer

from .runtime_console_command_actions import (
    RuntimeConsoleCommandDecision,
    apply_edited_response,
    apply_session_updates,
    prepare_regeneration,
)
from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_state import RuntimeConsoleState


def handle_console_command(
    *,
    deps: RuntimeCommandDeps,
    service: Any,
    handler: Any,
    state: RuntimeConsoleState,
    command_input: str,
    previous_real_local: str | None,
) -> RuntimeConsoleCommandDecision:
    command_output, action, updates = deps.console_command_output(
        command_input[1:],
        payload=state.payload,
        workspace=state.workspace,
        service=service,
        current_workspace=state.workspace,
        mode=state.mode_name,
        model=state.model,
        adapter=state.adapter,
        temperature=state.temperature,
        max_tokens=state.max_tokens,
        real_local=state.real_local,
        refresh_seconds=state.refresh_seconds,
        last_interaction=state.last_interaction,
    )
    if action == "quit":
        typer.echo("Exiting PFE Console.")
        return RuntimeConsoleCommandDecision(exit_requested=True)
    if action == "clear":
        state.transcript.clear()
        state.chat_messages.clear()
        state.feedback = "console transcript cleared"
        return RuntimeConsoleCommandDecision()
    if action == "mode:chat":
        state.mode_name = "chat"
        state.feedback = "switched to chat mode"
        return RuntimeConsoleCommandDecision()
    if action == "mode:command":
        state.mode_name = "command"
        state.feedback = "switched to command mode"
        return RuntimeConsoleCommandDecision()
    if action == "fix" or (updates and "edited_text" in updates):
        apply_edited_response(
            deps=deps,
            state=state,
            command_output=command_output,
            edited_text=(updates or {}).get("edited_text", ""),
        )
        return RuntimeConsoleCommandDecision()
    if action == "again" or (updates and updates.get("regenerate")):
        return prepare_regeneration(deps=deps, state=state, command_output=command_output)
    if updates:
        apply_session_updates(
            deps=deps,
            handler=handler,
            state=state,
            command_output=command_output,
            action=action,
            updates=updates,
            previous_real_local=previous_real_local,
        )
        return RuntimeConsoleCommandDecision()

    if command_output:
        deps.append_console_line(state.transcript, role="system", content=command_output)
    state.feedback = f"handled /{action}"
    return RuntimeConsoleCommandDecision()


__all__ = ["RuntimeConsoleCommandDecision", "handle_console_command"]
