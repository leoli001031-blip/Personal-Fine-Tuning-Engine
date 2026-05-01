"""Rendering refresh for completed runtime console chat turns."""

from __future__ import annotations

import time
from typing import Any

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_rendering import render_console_frame
from .runtime_console_state import RuntimeConsoleState


def render_completed_chat(
    *,
    live: Any,
    deps: RuntimeCommandDeps,
    handler: Any,
    state: RuntimeConsoleState,
) -> None:
    state.payload = deps.console_snapshot_payload(handler, workspace=state.workspace)
    render_console_frame(
        live,
        deps,
        payload=state.payload,
        workspace=state.workspace,
        transcript=state.transcript,
        feedback=state.feedback,
        mode_name=state.mode_name,
        model=state.model,
        adapter=state.adapter,
        real_local=state.real_local,
        refresh_seconds=state.refresh_seconds,
        input_active=False,
        input_cursor=0,
        shortcut_hint=deps.console_shortcut_hint(state.mode_name, state.payload),
        ops_refresh_state="live",
        ops_age_seconds=0.0,
    )
    state.last_sidebar_refresh_at = time.monotonic()
    state.ops_refresh_state = "live"


__all__ = ["render_completed_chat"]
