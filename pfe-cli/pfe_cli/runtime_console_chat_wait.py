"""Polling and live refresh while the runtime console chat worker runs."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_rendering import render_console_frame


def wait_for_chat_worker(
    *,
    worker: threading.Thread,
    response_holder: dict[str, Any],
    live: Any,
    deps: RuntimeCommandDeps,
    handler: Any,
    payload: dict[str, Any],
    workspace: str | None,
    transcript: Sequence[Mapping[str, str]],
    mode_name: str,
    model: str,
    adapter: str,
    real_local: bool,
    refresh_seconds: float,
    message: str,
    started_at: float,
    last_sidebar_refresh_at: float,
) -> tuple[Any, dict[str, Any], float, str]:
    wait_seconds = min(max(refresh_seconds / 2.0, 0.2), 0.5)
    ops_refresh_state = "live"

    while worker.is_alive():
        now = time.monotonic()
        should_refresh_sidebar = now - last_sidebar_refresh_at >= refresh_seconds
        if should_refresh_sidebar:
            render_console_frame(
                live,
                deps,
                payload=payload,
                workspace=workspace,
                transcript=transcript,
                feedback=(
                    f"assistant generating... {time.monotonic() - started_at:.1f}s | "
                    f"mode={mode_name} | model={model} | adapter={adapter}"
                ),
                mode_name=mode_name,
                model=model,
                adapter=adapter,
                real_local=real_local,
                refresh_seconds=refresh_seconds,
                input_active=False,
                input_text=message,
                input_cursor=len(message),
                shortcut_hint="wait,^C",
                ops_refresh_state="syncing",
                ops_age_seconds=max(0.0, now - last_sidebar_refresh_at),
            )
            refreshed_payload = deps.console_snapshot_payload(handler, workspace=workspace)
            payload = refreshed_payload
            last_sidebar_refresh_at = now
            ops_refresh_state = "live"
        else:
            refreshed_payload = payload
            ops_refresh_state = "cached"
        ops_age_seconds = max(0.0, now - last_sidebar_refresh_at)
        elapsed = time.monotonic() - started_at
        refresh_feedback = f"assistant generating... {elapsed:.1f}s | mode={mode_name} | model={model} | adapter={adapter}"
        render_console_frame(
            live,
            deps,
            payload=refreshed_payload,
            workspace=workspace,
            transcript=transcript,
            feedback=refresh_feedback,
            mode_name=mode_name,
            model=model,
            adapter=adapter,
            real_local=real_local,
            refresh_seconds=refresh_seconds,
            input_active=False,
            input_text=message,
            input_cursor=len(message),
            shortcut_hint="wait,^C",
            ops_refresh_state=ops_refresh_state,
            ops_age_seconds=ops_age_seconds,
        )
        worker.join(timeout=wait_seconds)

    if "error" in response_holder:
        raise response_holder["error"]
    return response_holder.get("result"), payload, last_sidebar_refresh_at, ops_refresh_state


__all__ = ["wait_for_chat_worker"]
