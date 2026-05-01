"""Transcript and interaction state helpers for runtime console chat."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_state import RuntimeConsoleState


def append_chat_unavailable(
    *,
    deps: RuntimeCommandDeps,
    state: RuntimeConsoleState,
    message: str,
) -> None:
    deps.append_console_line(state.transcript, role="user", content=message)
    deps.append_console_line(
        state.transcript,
        role="assistant",
        content="Interactive chat is unavailable because no chat_completion handler is registered.",
    )
    state.feedback = "chat handler unavailable"


def prepare_user_message(
    *,
    deps: RuntimeCommandDeps,
    state: RuntimeConsoleState,
    message: str,
    regenerate_mode: bool,
) -> None:
    if regenerate_mode:
        if not state.chat_messages or state.chat_messages[-1]["role"] != "user":
            state.chat_messages.append({"role": "user", "content": message})
        return

    if state.mode_name == "chat" and state.last_interaction is not None:
        deps.console_submit_feedback(
            workspace=state.workspace,
            session_id=state.last_interaction.get("session_id", state.session_id),
            request_id=state.last_interaction.get("request_id", ""),
            user_message=state.last_interaction.get("user_message", ""),
            assistant_message=state.last_interaction.get("assistant_message", ""),
            response_time_seconds=state.last_interaction.get("response_time_seconds", 0.0),
            adapter_version=state.last_interaction.get("adapter_version", state.adapter),
            action="continue",
        )
    deps.append_console_line(state.transcript, role="user", content=message)
    state.chat_messages.append({"role": "user", "content": message})


def append_assistant_response(
    *,
    deps: RuntimeCommandDeps,
    state: RuntimeConsoleState,
    message: str,
    chat_response: Any,
    started_at: float,
) -> None:
    assistant_text = deps.console_chat_text(chat_response) or "(empty response)"
    state.chat_messages.append({"role": "assistant", "content": assistant_text})
    deps.append_console_line(state.transcript, role="assistant", content=assistant_text)
    latency_seconds = time.monotonic() - started_at
    state.last_interaction = {
        "session_id": state.session_id,
        "request_id": f"req-{uuid4().hex[:12]}",
        "user_message": message,
        "assistant_message": assistant_text,
        "response_time_seconds": latency_seconds,
        "adapter_version": state.adapter,
    }
    state.feedback = f"assistant replied ({len(assistant_text)} chars in {latency_seconds:.1f}s)"


__all__ = [
    "append_assistant_response",
    "append_chat_unavailable",
    "prepare_user_message",
]
