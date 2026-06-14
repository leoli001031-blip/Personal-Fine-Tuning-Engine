"""State mutations used by runtime console slash commands."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_state import RuntimeConsoleState, set_real_local_env


@dataclass(frozen=True)
class RuntimeConsoleCommandDecision:
    exit_requested: bool = False
    regenerate_message: str | None = None


def apply_edited_response(
    *,
    deps: RuntimeCommandDeps,
    state: RuntimeConsoleState,
    command_output: str | None,
    edited_text: str,
) -> None:
    if edited_text:
        if state.chat_messages and state.chat_messages[-1]["role"] == "assistant":
            state.chat_messages[-1]["content"] = edited_text
        if state.transcript and state.transcript[-1].get("role") == "assistant":
            state.transcript[-1]["content"] = edited_text
        if state.last_interaction is not None:
            state.last_interaction["assistant_message"] = edited_text
    if command_output:
        deps.append_console_line(state.transcript, role="system", content=command_output)
    state.feedback = "response edited"


def prepare_regeneration(
    *,
    deps: RuntimeCommandDeps,
    state: RuntimeConsoleState,
    command_output: str | None,
) -> RuntimeConsoleCommandDecision:
    if state.chat_messages and state.chat_messages[-1]["role"] == "assistant":
        state.chat_messages.pop()
    regenerate_message = None
    if state.last_interaction is not None:
        regenerate_message = state.last_interaction.get("user_message", "")
    if command_output:
        deps.append_console_line(state.transcript, role="system", content=command_output)
    if regenerate_message is None:
        state.feedback = "nothing to regenerate"
        return RuntimeConsoleCommandDecision()
    return RuntimeConsoleCommandDecision(regenerate_message=str(regenerate_message))


def apply_session_updates(
    *,
    deps: RuntimeCommandDeps,
    handler: Any,
    state: RuntimeConsoleState,
    command_output: str | None,
    action: str,
    updates: dict[str, Any],
    previous_real_local: str | None,
) -> None:
    if "workspace" in updates:
        state.workspace = str(updates["workspace"])
    if "model" in updates:
        state.model = str(updates["model"])
    if "adapter" in updates:
        state.adapter = str(updates["adapter"])
    if "temperature" in updates:
        state.temperature = float(updates["temperature"])
    if "max_tokens" in updates:
        state.max_tokens = updates["max_tokens"]
    if "real_local" in updates:
        state.real_local = bool(updates["real_local"])
        set_real_local_env(state.real_local, previous=previous_real_local)
    if "refresh_seconds" in updates:
        state.refresh_seconds = float(updates["refresh_seconds"])
    state.payload = deps.console_snapshot_payload(handler, workspace=state.workspace)
    state.last_sidebar_refresh_at = time.monotonic()
    state.ops_refresh_state = "live"
    if command_output:
        deps.append_console_line(state.transcript, role="system", content=command_output)
    state.feedback = f"handled /{action}"


__all__ = [
    "RuntimeConsoleCommandDecision",
    "apply_edited_response",
    "apply_session_updates",
    "prepare_regeneration",
]
