"""One chat turn for the interactive runtime console."""

from __future__ import annotations

import time
from typing import Any

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_chat import start_chat_worker, wait_for_chat_worker
from .runtime_console_chat_messages import (
    append_assistant_response,
    append_chat_unavailable,
    prepare_user_message,
)
from .runtime_console_chat_render import render_completed_chat
from .runtime_console_state import RuntimeConsoleState


def run_console_chat_turn(
    *,
    live: Any,
    deps: RuntimeCommandDeps,
    handler: Any,
    chat_handler: Any,
    state: RuntimeConsoleState,
    message: str,
    regenerate_mode: bool,
) -> None:
    if chat_handler is None:
        append_chat_unavailable(deps=deps, state=state, message=message)
        return

    prepare_user_message(deps=deps, state=state, message=message, regenerate_mode=regenerate_mode)
    effective_max_tokens = state.max_tokens or 96
    state.feedback = "assistant generating..."
    started_at = time.monotonic()
    worker, response_holder = start_chat_worker(
        chat_handler=chat_handler,
        chat_messages=state.chat_messages,
        model=state.model,
        adapter=state.adapter,
        temperature=state.temperature,
        effective_max_tokens=effective_max_tokens,
        real_local=state.real_local,
        session_id=state.session_id,
        workspace=state.workspace,
    )
    chat_response, state.payload, state.last_sidebar_refresh_at, state.ops_refresh_state = wait_for_chat_worker(
        worker=worker,
        response_holder=response_holder,
        live=live,
        deps=deps,
        handler=handler,
        payload=state.payload,
        workspace=state.workspace,
        transcript=state.transcript,
        mode_name=state.mode_name,
        model=state.model,
        adapter=state.adapter,
        real_local=state.real_local,
        refresh_seconds=state.refresh_seconds,
        message=message,
        started_at=started_at,
        last_sidebar_refresh_at=state.last_sidebar_refresh_at,
    )
    append_assistant_response(deps=deps, state=state, message=message, chat_response=chat_response, started_at=started_at)
    render_completed_chat(live=live, deps=deps, handler=handler, state=state)


__all__ = ["run_console_chat_turn"]
