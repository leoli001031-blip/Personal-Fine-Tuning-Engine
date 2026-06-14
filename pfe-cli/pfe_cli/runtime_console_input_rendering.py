"""Input refresh rendering for the interactive runtime console."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_frame_rendering import render_console_frame


def refresh_input_console(
    live: Any,
    deps: RuntimeCommandDeps,
    *,
    handler: Any,
    payload: dict[str, Any],
    workspace: str | None,
    transcript: Sequence[Mapping[str, str]],
    feedback: str | None,
    mode_name: str,
    model: str,
    adapter: str,
    real_local: bool,
    refresh_seconds: float,
    last_sidebar_refresh_at: float,
    input_text: str = "",
    input_cursor: int | None = None,
    force_sidebar: bool = False,
) -> tuple[dict[str, Any], float, str]:
    now = _monotonic()
    should_refresh_sidebar = force_sidebar or (now - last_sidebar_refresh_at >= refresh_seconds)
    if should_refresh_sidebar:
        render_console_frame(
            live,
            deps,
            payload=payload,
            workspace=workspace,
            transcript=transcript,
            feedback=feedback,
            mode_name=mode_name,
            model=model,
            adapter=adapter,
            real_local=real_local,
            refresh_seconds=refresh_seconds,
            input_active=True,
            input_text=input_text,
            input_cursor=input_cursor,
            ops_refresh_state="syncing",
            ops_age_seconds=max(0.0, now - last_sidebar_refresh_at),
        )
        payload = deps.console_snapshot_payload(handler, workspace=workspace)
        last_sidebar_refresh_at = now
        ops_refresh_state = "live"
    else:
        ops_refresh_state = "cached"
    ops_age_seconds = max(0.0, now - last_sidebar_refresh_at)
    render_console_frame(
        live,
        deps,
        payload=payload,
        workspace=workspace,
        transcript=transcript,
        feedback=feedback,
        mode_name=mode_name,
        model=model,
        adapter=adapter,
        real_local=real_local,
        refresh_seconds=refresh_seconds,
        input_active=True,
        input_text=input_text,
        input_cursor=input_cursor,
        ops_refresh_state=ops_refresh_state,
        ops_age_seconds=ops_age_seconds,
    )
    return payload, last_sidebar_refresh_at, ops_refresh_state


def _monotonic() -> float:
    import time

    return time.monotonic()


__all__ = ["refresh_input_console"]
