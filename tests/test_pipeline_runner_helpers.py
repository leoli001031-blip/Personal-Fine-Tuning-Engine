from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import os

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.pipeline_runner import (  # noqa: E402
    append_runner_history,
    daemon_history_payload,
    daemon_summary,
    daemon_timeline_summary,
    runner_timeline_summary,
    worker_runner_history_payload,
    worker_summary,
)


def _trigger() -> SimpleNamespace:
    return SimpleNamespace(
        queue_daemon_auto_recover=True,
        queue_daemon_auto_restart=True,
        queue_daemon_heartbeat_interval_seconds=2.0,
        queue_daemon_lease_timeout_seconds=15.0,
        queue_daemon_max_restart_attempts=3,
        queue_daemon_restart_backoff_seconds=15.0,
    )


def test_worker_runner_history_and_timeline_helpers() -> None:
    first_at = datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc)
    second_at = first_at + timedelta(seconds=1)
    payload = append_runner_history(
        payload={},
        event="started",
        reason="stale_lock_takeover",
        metadata={"note": "worker note stays in metadata"},
        timestamp=first_at,
    )
    payload = append_runner_history(
        payload=payload,
        event="completed",
        reason="idle_exit",
        timestamp=second_at,
    )

    history = worker_runner_history_payload(payload=payload, workspace="lab", limit=1)
    timeline = runner_timeline_summary(history_payload=worker_runner_history_payload(payload=payload), limit=5)

    assert history["workspace"] == "lab"
    assert history["count"] == 2
    assert history["items"][0]["event"] == "completed"
    assert timeline["takeover_event_count"] == 1
    assert timeline["last_takeover_event"] == "started"
    assert timeline["latest_timestamp"] == second_at.isoformat()
    assert "note" not in payload["history"][0]


def test_daemon_history_and_recovery_timeline_helpers() -> None:
    first_at = datetime(2026, 3, 26, 11, 0, tzinfo=timezone.utc)
    payload = append_runner_history(
        payload={},
        event="start_requested",
        reason="daemon_start_requested",
        metadata={"note": "boot"},
        include_metadata_note=True,
        timestamp=first_at,
    )
    payload = append_runner_history(
        payload=payload,
        event="recover_requested",
        reason="daemon_stale",
        metadata={"note": "auto_recovery"},
        include_metadata_note=True,
        timestamp=first_at + timedelta(seconds=30),
    )
    payload = append_runner_history(
        payload=payload,
        event="completed",
        reason="idle_exit",
        include_metadata_note=True,
        timestamp=first_at + timedelta(seconds=60),
    )

    history = daemon_history_payload(payload=payload, workspace="lab", limit=10)
    timeline = daemon_timeline_summary(history_payload=history, limit=5)

    assert history["last_event"] == "completed"
    assert timeline["recovery_event_count"] == 2
    assert timeline["last_recovery_event"] == "recover_requested"
    assert timeline["last_recovery_reason"] == "daemon_stale"
    assert timeline["last_recovery_note"] == "auto_recovery"
    assert timeline["recent_recovery_events"][0]["note"] == "boot"


def test_worker_summary_marks_stale_lock_from_heartbeat_age() -> None:
    heartbeat = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
    summary = worker_summary(
        payload={
            "active": True,
            "last_heartbeat_at": heartbeat.isoformat(),
            "max_seconds": 30.0,
            "history": [{"event": "started", "reason": "run_worker_runner"}],
            "history_count": 1,
        },
        state_path="/tmp/worker.json",
        workspace="lab",
        now=heartbeat + timedelta(seconds=61),
    )

    assert summary["lock_state"] == "stale"
    assert summary["stale_after_seconds"] == 60.0
    assert summary["lease_expires_at"] == (heartbeat + timedelta(seconds=60)).isoformat()
    assert summary["last_event"] == "started"
    assert summary["last_event_reason"] == "run_worker_runner"


def test_daemon_summary_reports_stale_backoff_and_recovery_block() -> None:
    now = datetime(2026, 3, 26, 12, 0, tzinfo=timezone.utc)
    heartbeat = now - timedelta(seconds=30)
    retry_at = now + timedelta(seconds=20)
    summary = daemon_summary(
        payload={
            "workspace": "lab",
            "desired_state": "running",
            "observed_state": "running",
            "command_status": "running",
            "active": True,
            "pid": 999999,
            "last_heartbeat_at": heartbeat.isoformat(),
            "heartbeat_interval_seconds": 2.0,
            "lease_timeout_seconds": 15.0,
            "auto_recover_enabled": True,
            "auto_restart_enabled": True,
            "restart_attempts": 1,
            "max_restart_attempts": 3,
            "restart_backoff_seconds": 15.0,
            "next_restart_after": retry_at.isoformat(),
        },
        state_path="/tmp/daemon.json",
        trigger=_trigger(),
        pid_exists=lambda _pid: False,
        workspace="lab",
        now=now,
    )

    assert summary["lock_state"] == "stale"
    assert summary["health_state"] == "stale"
    assert summary["heartbeat_state"] == "stale"
    assert summary["lease_state"] == "expired"
    assert summary["restart_policy_state"] == "backoff"
    assert summary["recovery_needed"] is True
    assert summary["can_recover"] is False
    assert summary["recovery_reason"] == "daemon_stale"
    assert summary["backoff_remaining_seconds"] == 20.0
