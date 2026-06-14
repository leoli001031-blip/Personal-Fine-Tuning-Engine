"""Worker runner and daemon history/summary helpers for PipelineService."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def runner_history_entry(
    *,
    event: str,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    entry = {
        "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        "event": str(event),
    }
    if reason:
        entry["reason"] = str(reason)
    if metadata:
        entry["metadata"] = dict(metadata)
    return entry


def append_runner_history(
    *,
    payload: Mapping[str, Any],
    event: str,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_metadata_note: bool = False,
    history_limit: int = 20,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    updated = dict(payload)
    history = list(updated.get("history") or [])
    entry = runner_history_entry(
        event=event,
        reason=reason,
        metadata=metadata,
        timestamp=timestamp,
    )
    if include_metadata_note and isinstance(metadata, Mapping) and metadata.get("note") is not None:
        entry["note"] = str(metadata["note"])
    history.append(entry)
    updated["history"] = history[-max(1, int(history_limit or 20)) :]
    updated["history_count"] = len(history)
    return updated


def worker_runner_history_payload(
    *,
    payload: Mapping[str, Any],
    workspace: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    bounded_limit = max(1, int(limit or 10))
    history = list(payload.get("history") or [])
    latest = history[-1] if history else {}
    return {
        "workspace": workspace or "user_default",
        "count": len(history),
        "limit": bounded_limit,
        "last_event": latest.get("event"),
        "last_reason": latest.get("reason"),
        "items": history[-bounded_limit:],
    }


def daemon_history_payload(
    *,
    payload: Mapping[str, Any],
    workspace: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    bounded_limit = max(1, int(limit or 10))
    history = list(payload.get("history") or [])
    latest = history[-1] if history else {}
    return {
        "workspace": workspace or "user_default",
        "count": len(history),
        "limit": bounded_limit,
        "last_event": latest.get("event"),
        "last_reason": latest.get("reason"),
        "latest_timestamp": latest.get("timestamp"),
        "items": history[-bounded_limit:],
    }


def runner_timeline_summary(
    *,
    history_payload: Mapping[str, Any],
    limit: int = 5,
) -> dict[str, Any]:
    bounded_limit = max(1, int(limit or 5))
    items = [dict(item) for item in list(history_payload.get("items") or [])]
    takeover_items = [
        item
        for item in items
        if str(item.get("reason") or "").startswith("stale_lock_takeover")
    ]
    latest = items[-1] if items else {}
    last_takeover = takeover_items[-1] if takeover_items else {}
    return {
        "count": history_payload.get("count", 0),
        "latest_timestamp": ((latest or {}).get("timestamp") or None),
        "last_event": history_payload.get("last_event"),
        "last_reason": history_payload.get("last_reason"),
        "takeover_event_count": len(takeover_items),
        "last_takeover_event": last_takeover.get("event"),
        "last_takeover_reason": last_takeover.get("reason"),
        "recent_takeover_events": [
            {
                "timestamp": item.get("timestamp"),
                "event": item.get("event"),
                "reason": item.get("reason"),
                "note": item.get("note"),
            }
            for item in takeover_items[-bounded_limit:]
        ],
        "recent_events": [
            {
                "timestamp": item.get("timestamp"),
                "event": item.get("event"),
                "reason": item.get("reason"),
                "note": item.get("note"),
            }
            for item in items[-bounded_limit:]
        ],
        "latest": latest,
    }


def daemon_timeline_summary(
    *,
    history_payload: Mapping[str, Any],
    limit: int = 5,
) -> dict[str, Any]:
    bounded_limit = max(1, int(limit or 5))
    items = [dict(item) for item in list(history_payload.get("items") or [])]
    recovery_events = {
        "recover_requested",
        "restart_requested",
        "recover_blocked",
        "stale_lock_takeover",
        "start_requested",
    }
    recovery_items = [item for item in items if str(item.get("event") or "") in recovery_events]
    latest = items[-1] if items else {}
    last_recovery = recovery_items[-1] if recovery_items else {}
    return {
        "count": history_payload.get("count", 0),
        "latest_timestamp": history_payload.get("latest_timestamp"),
        "last_event": history_payload.get("last_event"),
        "last_reason": history_payload.get("last_reason"),
        "recovery_event_count": len(recovery_items),
        "last_recovery_event": last_recovery.get("event"),
        "last_recovery_reason": last_recovery.get("reason"),
        "last_recovery_note": last_recovery.get("note"),
        "recent_recovery_events": [
            {
                "timestamp": item.get("timestamp"),
                "event": item.get("event"),
                "reason": item.get("reason"),
                "note": item.get("note"),
            }
            for item in recovery_items[-bounded_limit:]
        ],
        "recent_events": [
            {
                "timestamp": item.get("timestamp"),
                "event": item.get("event"),
                "reason": item.get("reason"),
                "note": item.get("note"),
            }
            for item in items[-bounded_limit:]
        ],
        "latest": latest,
    }


def worker_summary(
    *,
    payload: Mapping[str, Any],
    state_path: str | Path,
    workspace: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    stale_after_seconds = None
    lease_expires_at = None
    lock_state = "idle"
    if bool(payload.get("active", False)):
        lock_state = "active"
        heartbeat = parse_iso_datetime(payload.get("last_heartbeat_at"))
        if heartbeat is not None:
            if heartbeat.tzinfo is None:
                heartbeat = heartbeat.replace(tzinfo=timezone.utc)
            max_seconds = float(payload.get("max_seconds", 30.0) or 30.0)
            stale_after_seconds = max(5.0, max_seconds * 2.0)
            lease_expires_at = (heartbeat + timedelta(seconds=stale_after_seconds)).isoformat()
            if (current_time - heartbeat).total_seconds() > stale_after_seconds:
                lock_state = "stale"
    history = list(payload.get("history") or [])
    latest = history[-1] if history else {}
    return {
        "state_path": str(state_path),
        "active": bool(payload.get("active", False)),
        "lock_state": lock_state,
        "stale_after_seconds": stale_after_seconds,
        "lease_expires_at": lease_expires_at,
        "stop_requested": bool(payload.get("stop_requested", False)),
        "pid": payload.get("pid"),
        "started_at": payload.get("started_at"),
        "last_heartbeat_at": payload.get("last_heartbeat_at"),
        "last_completed_at": payload.get("last_completed_at"),
        "loop_cycles": int(payload.get("loop_cycles", 0) or 0),
        "processed_count": int(payload.get("processed_count", 0) or 0),
        "failed_count": int(payload.get("failed_count", 0) or 0),
        "stopped_reason": payload.get("stopped_reason"),
        "last_action": payload.get("last_action"),
        "history_count": int(payload.get("history_count", 0) or 0),
        "last_event": latest.get("event"),
        "last_event_reason": latest.get("reason"),
        "max_seconds": payload.get("max_seconds"),
        "idle_sleep_seconds": payload.get("idle_sleep_seconds"),
    }


def _trigger_value(trigger: Any, key: str) -> Any:
    return getattr(trigger, key)


def _payload_float(payload: Mapping[str, Any], key: str, default: Any) -> float:
    return float(payload.get(key, default) or default)


def _payload_int(payload: Mapping[str, Any], key: str, default: Any) -> int:
    return int(payload.get(key, default) or default)


def daemon_summary(
    *,
    payload: Mapping[str, Any],
    state_path: str | Path,
    trigger: Any,
    pid_exists: Callable[[Any], bool],
    workspace: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    active = bool(payload.get("active", False))
    pid = payload.get("pid")
    lock_state = "active" if active else "idle"
    heartbeat = parse_iso_datetime(payload.get("last_heartbeat_at"))
    heartbeat_interval_seconds = _payload_float(
        payload,
        "heartbeat_interval_seconds",
        _trigger_value(trigger, "queue_daemon_heartbeat_interval_seconds"),
    )
    lease_timeout_seconds = _payload_float(
        payload,
        "lease_timeout_seconds",
        _trigger_value(trigger, "queue_daemon_lease_timeout_seconds"),
    )
    heartbeat_timeout_seconds = _payload_float(
        payload,
        "heartbeat_timeout_seconds",
        lease_timeout_seconds,
    )
    heartbeat_age_seconds = None
    lease_expires_at = None
    lease_state = "idle"
    heartbeat_state = "idle"
    current_time = now or datetime.now(timezone.utc)
    if active:
        if heartbeat is not None:
            if heartbeat.tzinfo is None:
                heartbeat = heartbeat.replace(tzinfo=timezone.utc)
            heartbeat_age_seconds = max(0.0, round((current_time - heartbeat).total_seconds(), 3))
            effective_lease_timeout = max(5.0, lease_timeout_seconds)
            fresh_threshold_seconds = max(heartbeat_interval_seconds * 3.0, 5.0)
            lease_expires_at = (heartbeat + timedelta(seconds=effective_lease_timeout)).isoformat()
            if heartbeat_age_seconds <= fresh_threshold_seconds:
                heartbeat_state = "fresh"
            elif heartbeat_age_seconds <= effective_lease_timeout:
                heartbeat_state = "delayed"
            else:
                heartbeat_state = "stale"
            if heartbeat_age_seconds <= max(effective_lease_timeout * 0.5, heartbeat_interval_seconds * 2.0):
                lease_state = "valid"
            elif heartbeat_age_seconds <= effective_lease_timeout:
                lease_state = "expiring"
            else:
                lease_state = "expired"
            if heartbeat_age_seconds > effective_lease_timeout or not pid_exists(pid):
                lock_state = "stale"
        elif not pid_exists(pid):
            lock_state = "stale"
            heartbeat_state = "stale"
            lease_state = "expired"
    history = list(payload.get("history") or [])
    latest = history[-1] if history else {}
    restart_attempts = int(payload.get("restart_attempts", 0) or 0)
    max_restart_attempts = _payload_int(
        payload,
        "max_restart_attempts",
        _trigger_value(trigger, "queue_daemon_max_restart_attempts"),
    )
    restart_backoff_seconds = _payload_float(
        payload,
        "restart_backoff_seconds",
        _trigger_value(trigger, "queue_daemon_restart_backoff_seconds"),
    )
    next_restart_after = parse_iso_datetime(payload.get("next_restart_after"))
    if next_restart_after is not None and next_restart_after.tzinfo is None:
        next_restart_after = next_restart_after.replace(tzinfo=timezone.utc)
    backoff_remaining_seconds = None
    if next_restart_after is not None and current_time < next_restart_after:
        backoff_remaining_seconds = round((next_restart_after - current_time).total_seconds(), 3)
    if backoff_remaining_seconds is not None:
        restart_policy_state = "backoff"
    elif restart_attempts >= max_restart_attempts:
        restart_policy_state = "capped"
    else:
        restart_policy_state = "ready"
    desired_state = payload.get("desired_state") or "stopped"
    recovery_needed = bool(
        desired_state == "running"
        and lock_state in {"idle", "stale"}
        and not bool(payload.get("stop_requested", False))
    )
    can_recover = recovery_needed and backoff_remaining_seconds is None and restart_attempts < max_restart_attempts
    recovery_reason = None
    if recovery_needed:
        if lock_state == "stale":
            recovery_reason = "daemon_stale"
        elif backoff_remaining_seconds is not None:
            recovery_reason = "restart_backoff_active"
        elif restart_attempts >= max_restart_attempts:
            recovery_reason = "restart_attempt_limit_reached"
        else:
            recovery_reason = "daemon_inactive"
    requested_action = str(payload.get("requested_action") or "")
    command_status = str(payload.get("command_status") or "")
    observed_state = payload.get("observed_state") or ("running" if active else "stopped")
    if requested_action == "restart" and command_status == "spawned":
        observed_state = "restarting"
        recovery_state = "restarting"
    elif requested_action == "recover" and command_status == "spawned":
        observed_state = "recovering"
        recovery_state = "recovering"
    elif active and lock_state == "active":
        recovery_state = "healthy"
    elif recovery_needed and can_recover:
        recovery_state = "recoverable"
    elif recovery_needed:
        recovery_state = "blocked"
    else:
        recovery_state = "idle"
    if active and lock_state == "active":
        health_state = "healthy"
    elif lock_state == "stale":
        health_state = "stale"
    elif recovery_state in {"restarting", "recovering"}:
        health_state = "recovering"
    elif recovery_needed and not can_recover:
        health_state = "blocked"
    else:
        health_state = "stopped"
    if requested_action == "recover" and command_status == "spawned":
        recovery_action = (
            "auto_recover"
            if str(payload.get("last_requested_by") or "") == "auto_recovery"
            else "manual_recover"
        )
    elif requested_action == "restart" and command_status == "spawned":
        recovery_action = "restart_required"
    elif recovery_needed and can_recover:
        recovery_action = (
            "auto_recover"
            if bool(payload.get("auto_recover_enabled", _trigger_value(trigger, "queue_daemon_auto_recover")))
            else "manual_recover"
        )
    else:
        recovery_action = "none"
    return {
        "workspace": workspace or "user_default",
        "state_path": str(state_path),
        "desired_state": desired_state,
        "observed_state": observed_state,
        "requested_action": payload.get("requested_action"),
        "command_status": payload.get("command_status") or ("running" if active else "idle"),
        "active": active,
        "lock_state": lock_state,
        "pid": pid,
        "started_at": payload.get("started_at"),
        "last_heartbeat_at": payload.get("last_heartbeat_at"),
        "last_completed_at": payload.get("last_completed_at"),
        "last_requested_at": payload.get("last_requested_at"),
        "last_requested_by": payload.get("last_requested_by"),
        "stop_requested": bool(payload.get("stop_requested", False)),
        "auto_recover_enabled": bool(
            payload.get("auto_recover_enabled", _trigger_value(trigger, "queue_daemon_auto_recover"))
        ),
        "heartbeat_interval_seconds": heartbeat_interval_seconds,
        "lease_timeout_seconds": lease_timeout_seconds,
        "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
        "health_state": health_state,
        "lease_state": lease_state,
        "heartbeat_state": heartbeat_state,
        "restart_policy_state": restart_policy_state,
        "recovery_action": recovery_action,
        "lease_expires_at": lease_expires_at,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "history_count": int(payload.get("history_count", 0) or 0),
        "last_event": latest.get("event"),
        "last_reason": latest.get("reason"),
        "latest_timestamp": latest.get("timestamp"),
        "log_path": payload.get("log_path"),
        "runner_max_seconds": payload.get("runner_max_seconds"),
        "idle_sleep_seconds": payload.get("idle_sleep_seconds"),
        "takeover": bool(payload.get("takeover", False)),
        "previous_pid": payload.get("previous_pid"),
        "auto_restart_enabled": bool(
            payload.get("auto_restart_enabled", _trigger_value(trigger, "queue_daemon_auto_restart"))
        ),
        "restart_attempts": restart_attempts,
        "max_restart_attempts": max_restart_attempts,
        "restart_backoff_seconds": restart_backoff_seconds,
        "next_restart_after": next_restart_after.isoformat() if next_restart_after is not None else None,
        "backoff_remaining_seconds": backoff_remaining_seconds,
        "auto_recovery_count": int(payload.get("auto_recovery_count", 0) or 0),
        "last_auto_recovery_at": payload.get("last_auto_recovery_at"),
        "last_auto_recovery_reason": payload.get("last_auto_recovery_reason"),
        "recovery_needed": recovery_needed,
        "can_recover": can_recover,
        "recovery_reason": recovery_reason,
        "recovery_state": recovery_state,
        "recovery_mode": "restart_policy",
        "recovery_attempts": restart_attempts,
        "recovery_backoff_seconds": restart_backoff_seconds,
        "recovery_next_retry_at": next_restart_after.isoformat() if next_restart_after is not None else None,
    }


__all__ = [
    "append_runner_history",
    "daemon_history_payload",
    "daemon_summary",
    "daemon_timeline_summary",
    "parse_iso_datetime",
    "runner_history_entry",
    "runner_timeline_summary",
    "worker_runner_history_payload",
    "worker_summary",
]
