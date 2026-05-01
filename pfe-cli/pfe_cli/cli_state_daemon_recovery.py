"""Train queue daemon recovery payload bookkeeping."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .cli_state_daemon_store import read_train_queue_daemon_state, write_train_queue_daemon_state
from .cli_state_deps import CLIStateDeps


def daemon_recovery_payload(
    *,
    workspace: str | None = None,
    action: str,
    note: str | None = None,
    reason: str | None = None,
    deps: CLIStateDeps,
) -> dict[str, Any]:
    state = read_train_queue_daemon_state(workspace, deps=deps) or {}
    history = list(state.get("history") or [])
    timestamp = datetime.now(timezone.utc).isoformat()
    restart_attempts = int(state.get("restart_attempts", 0) or 0)
    if action == "restart":
        restart_attempts += 1
    recovery_state = "restarting" if action == "restart" else "recovering"
    state.update(
        {
            "workspace": workspace or "user_default",
            "desired_state": "running",
            "requested_action": action,
            "command_status": "requested",
            "active": True,
            "observed_state": recovery_state,
            "recovery_state": recovery_state,
            "auto_restart_enabled": True,
            "restart_attempts": restart_attempts,
            "restart_backoff_seconds": float(state.get("restart_backoff_seconds", 30.0) or 30.0),
            "next_restart_after": timestamp,
            "last_requested_at": timestamp,
            "last_requested_by": "pfe-cli",
            "last_reason": reason or "cli_requested",
            "last_recovery_reason": reason or "cli_requested",
        }
    )
    history.append(
        {
            "timestamp": timestamp,
            "event": f"{action}_requested",
            "reason": reason or "cli_requested",
            "note": note,
        }
    )
    state["history"] = history[-20:]
    state["history_count"] = len(history)
    write_train_queue_daemon_state(workspace, state, deps=deps)
    return state


__all__ = ["daemon_recovery_payload"]
