"""Interactive console loop for runtime CLI surfaces."""

from __future__ import annotations

import os
import time
from typing import Any

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_chat_flow import run_console_chat_turn
from .runtime_console_loop import read_console_loop_message, resolve_console_loop_action
from .runtime_console_state import (
    RuntimeConsoleState,
    restore_real_local_env,
    set_real_local_env,
)


def run_interactive_console(
    *,
    deps: RuntimeCommandDeps,
    service: Any,
    handler: Any,
    workspace: str | None,
    model: str,
    adapter: str,
    temperature: float,
    max_tokens: int | None,
    real_local: bool,
    refresh_seconds: float,
) -> None:
    """Run the Rich-backed interactive console loop."""

    from rich.console import Console as RichConsole
    from rich.live import Live

    console_ui = RichConsole()
    chat_handler = deps.resolve_handler(service, "chat_completion")
    state = RuntimeConsoleState(
        workspace=workspace,
        model=model,
        adapter=adapter,
        temperature=temperature,
        max_tokens=max_tokens,
        real_local=real_local,
        refresh_seconds=refresh_seconds,
    )

    previous_real_local = os.environ.get("PFE_ENABLE_REAL_LOCAL_INFERENCE")
    set_real_local_env(state.real_local, previous=previous_real_local)
    try:
        with Live(console=console_ui, auto_refresh=False, screen=False) as live:
            state.payload = deps.console_snapshot_payload(handler, workspace=state.workspace)
            state.last_sidebar_refresh_at = time.monotonic()
            state.ops_refresh_state = "live"

            while True:
                read_result = read_console_loop_message(
                    live,
                    console_ui,
                    deps,
                    handler=handler,
                    state=state,
                )
                if read_result.exit_requested:
                    break
                if read_result.message is None:
                    continue

                action = resolve_console_loop_action(
                    live,
                    deps,
                    service=service,
                    handler=handler,
                    state=state,
                    message=read_result.message,
                    previous_real_local=previous_real_local,
                )
                if action.exit_requested:
                    break
                if action.chat_message is None:
                    continue

                run_console_chat_turn(
                    live=live,
                    deps=deps,
                    handler=handler,
                    chat_handler=chat_handler,
                    state=state,
                    message=action.chat_message,
                    regenerate_mode=action.regenerate_mode,
                )
    finally:
        restore_real_local_env(previous_real_local)


__all__ = ["run_interactive_console"]
