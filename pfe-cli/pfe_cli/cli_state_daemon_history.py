"""Train queue daemon command history bookkeeping."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .cli_state_daemon_store import (
    read_train_queue_daemon_state,
    train_queue_daemon_state_path,
    write_train_queue_daemon_state,
)
from .cli_state_deps import CLIStateDeps


def record_train_queue_daemon_history(
    *,
    workspace: str | None = None,
    event: str,
    reason: str | None = None,
    note: str | None = None,
    deps: CLIStateDeps,
) -> dict[str, Any]:
    payload = read_train_queue_daemon_state(workspace, deps=deps) or {}
    history = list(payload.get("history") or [])
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
    }
    if reason is not None:
        entry["reason"] = str(reason)
    if note is not None:
        entry["note"] = str(note)
    history.append(entry)

    payload.update(
        {
            "workspace": workspace or "user_default",
            "history": history[-20:],
            "history_count": len(history),
            "last_event": str(event),
            "last_reason": str(reason) if reason is not None else payload.get("last_reason"),
            "last_requested_at": entry["timestamp"],
            "last_requested_by": "pfe-cli",
        }
    )
    write_train_queue_daemon_state(workspace, payload, deps=deps)
    return payload


def update_train_queue_daemon_state(
    *,
    workspace: str | None = None,
    desired_state: str,
    event: str,
    reason: str | None = None,
    note: str | None = None,
    observed_state: str | None = None,
    extra: dict[str, Any] | None = None,
    deps: CLIStateDeps,
) -> dict[str, Any]:
    payload = record_train_queue_daemon_history(
        workspace=workspace,
        event=event,
        reason=reason,
        note=note,
        deps=deps,
    )
    payload.update(
        {
            "desired_state": desired_state,
            "requested_action": event.replace("_requested", ""),
            "command_status": "requested",
            "observed_state": observed_state or payload.get("observed_state") or "unknown",
            "state_path": str(train_queue_daemon_state_path(workspace, deps=deps)),
            "active": desired_state == "running",
        }
    )
    if extra:
        payload.update(dict(extra))
    write_train_queue_daemon_state(workspace, payload, deps=deps)
    return payload


__all__ = ["record_train_queue_daemon_history", "update_train_queue_daemon_state"]
