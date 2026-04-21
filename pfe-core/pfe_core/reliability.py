"""Reliability mechanisms for PFE daemon and runner.

This module implements:
- Heartbeat tracking and stale runner detection
- Lease management for task execution
- Restart policy with exponential backoff
- Recovery checkpoint management
- Alert generation and tracking
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .models import (
    AlertEvent,
    AlertLevel,
    AlertThreshold,
    DeadLetterEntry,
    LeaseState,
    RecoveryAction,
    RecoveryCheckpoint,
    RestartPolicy,
    RunnerHeartbeat,
    RunnerState,
    TaskExecutionMetadata,
    TaskLease,
    TaskState,
    parse_utc_datetime,
    utc_now,
)
from .storage import resolve_home, write_json


class _ReliabilityJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for reliability data types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def _serialize_for_json(obj: Any) -> Any:
    """Recursively serialize dataclasses for JSON."""
    if is_dataclass(obj):
        result = {}
        for key, value in asdict(obj).items():
            result[key] = _serialize_for_json(value)
        return result
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _serialize_for_json(value) for key, value in obj.items()}
    return obj


def _write_json_with_dataclass(path: Path, obj: Any) -> None:
    """Write a dataclass object to JSON file with proper serialization."""
    serialized = _serialize_for_json(obj)
    path.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_datetime(value: Any) -> Any:
    """Parse ISO format datetime string back to datetime object."""
    if isinstance(value, str):
        try:
            return parse_utc_datetime(value)
        except ValueError:
            return value
    return value


def _deserialize_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Deserialize dict to dataclass with datetime parsing."""
    # Parse datetime fields
    parsed_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            parsed_data[key] = _parse_datetime(value)
        elif isinstance(value, list):
            # Check if list items might be datetime strings
            parsed_list = []
            for item in value:
                if isinstance(item, str):
                    parsed_list.append(_parse_datetime(item))
                else:
                    parsed_list.append(item)
            parsed_data[key] = parsed_list
        else:
            parsed_data[key] = value

    # Handle Enum fields - get the enum type from the class annotations
    from typing import get_type_hints, get_origin, get_args
    try:
        type_hints = get_type_hints(cls)
        for key, hint in type_hints.items():
            if key not in parsed_data:
                continue
            origin = get_origin(hint)
            if origin is not None:
                # Handle Optional[EnumType]
                args = get_args(hint)
                if len(args) == 2 and type(None) in args:
                    enum_type = args[0] if args[1] is type(None) else args[1]
                    if isinstance(enum_type, type) and issubclass(enum_type, Enum):
                        value = parsed_data[key]
                        if value is not None and isinstance(value, str):
                            parsed_data[key] = enum_type(value)
                continue
            if isinstance(hint, type) and issubclass(hint, Enum):
                value = parsed_data[key]
                if isinstance(value, str):
                    parsed_data[key] = hint(value)
    except Exception:
        pass  # Fall back to raw values

    return cls(**parsed_data)


class HeartbeatManager:
    """Manages runner heartbeats and detects stale runners."""

    def __init__(
        self,
        workspace: str = "user_default",
        lease_timeout_seconds: float = 60.0,
        stale_threshold_multiplier: float = 2.0,
    ):
        self.workspace = workspace
        self.lease_timeout_seconds = lease_timeout_seconds
        self.stale_threshold_multiplier = stale_threshold_multiplier
        self._state_dir = resolve_home() / "data" / "reliability"
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _heartbeat_path(self, runner_id: str) -> Path:
        return self._state_dir / f"heartbeat_{self.workspace}_{runner_id}.json"

    def _runners_index_path(self) -> Path:
        return self._state_dir / f"runners_index_{self.workspace}.json"

    def record_heartbeat(self, heartbeat: RunnerHeartbeat) -> None:
        """Record a heartbeat from a runner."""
        heartbeat.workspace = self.workspace
        path = self._heartbeat_path(heartbeat.runner_id)
        _write_json_with_dataclass(path, heartbeat)
        self._update_runners_index(heartbeat.runner_id, heartbeat.job_id)

    def get_heartbeat(self, runner_id: str) -> Optional[RunnerHeartbeat]:
        """Get the latest heartbeat for a runner."""
        path = self._heartbeat_path(runner_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _deserialize_dataclass(RunnerHeartbeat, data)
        except Exception:
            return None

    def get_runner_state(self, runner_id: str) -> RunnerState:
        """Determine the current state of a runner."""
        heartbeat = self.get_heartbeat(runner_id)
        if heartbeat is None:
            return RunnerState.IDLE

        age_seconds = (utc_now() - heartbeat.timestamp).total_seconds()
        stale_threshold = self.lease_timeout_seconds * self.stale_threshold_multiplier

        if age_seconds > stale_threshold:
            return RunnerState.STALE
        if age_seconds > self.lease_timeout_seconds:
            return RunnerState.DELAYED
        return RunnerState.HEALTHY

    def is_runner_healthy(self, runner_id: str) -> bool:
        """Check if a runner is healthy."""
        return self.get_runner_state(runner_id) == RunnerState.HEALTHY

    def is_runner_stale(self, runner_id: str) -> bool:
        """Check if a runner has gone stale."""
        return self.get_runner_state(runner_id) == RunnerState.STALE

    def list_active_runners(self) -> list[str]:
        """List all runners with recent heartbeats."""
        index_path = self._runners_index_path()
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            runners = data.get("runners", [])
            # Filter to only active (non-stale) runners
            return [r for r in runners if not self.is_runner_stale(r)]
        except Exception:
            return []

    def _update_runners_index(self, runner_id: str, job_id: Optional[str]) -> None:
        """Update the index of known runners."""
        index_path = self._runners_index_path()
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"runners": [], "jobs": {}}

        if runner_id not in data["runners"]:
            data["runners"].append(runner_id)
        if job_id:
            data["jobs"][runner_id] = job_id

        write_json(index_path, data)

    def _prune_runners_index(self, removed_runner_ids: set[str]) -> None:
        """Remove runner ids from the persisted index after heartbeat cleanup."""
        if not removed_runner_ids:
            return

        index_path = self._runners_index_path()
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return

        runners = [runner_id for runner_id in list(data.get("runners", [])) if runner_id not in removed_runner_ids]
        jobs = {
            runner_id: job_id
            for runner_id, job_id in dict(data.get("jobs", {})).items()
            if runner_id not in removed_runner_ids
        }
        write_json(index_path, {"runners": runners, "jobs": jobs})

    def cleanup_stale_runners(self, max_age_seconds: float = 3600) -> list[str]:
        """Remove heartbeats for runners that have been stale for too long."""
        removed = []
        stale_threshold = utc_now() - timedelta(seconds=max_age_seconds)

        for path in self._state_dir.glob(f"heartbeat_{self.workspace}_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                timestamp = parse_utc_datetime(data.get("timestamp"))
                if timestamp < stale_threshold:
                    path.unlink()
                    runner_id = str(data.get("runner_id") or "")
                    if runner_id:
                        removed.append(runner_id)
            except Exception:
                continue

        self._prune_runners_index(set(removed))
        return removed

    def detect_stalled_jobs(self) -> list[dict[str, Any]]:
        """Detect jobs assigned to stale runners."""
        stalled = []
        index_path = self._runners_index_path()

        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            for runner_id, job_id in data.get("jobs", {}).items():
                if self.is_runner_stale(runner_id):
                    stalled.append({
                        "runner_id": runner_id,
                        "job_id": job_id,
                        "detected_at": utc_now().isoformat(),
                    })
        except Exception:
            pass

        return stalled


class LeaseManager:
    """Manages task leases to prevent multiple workers competing."""

    def __init__(
        self,
        workspace: str = "user_default",
        default_lease_timeout_seconds: float = 60.0,
        warning_threshold_ratio: float = 0.75,
    ):
        self.workspace = workspace
        self.default_lease_timeout_seconds = default_lease_timeout_seconds
        self.warning_threshold_ratio = warning_threshold_ratio
        self._state_dir = resolve_home() / "data" / "reliability"
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def _lease_path(self, lease_id: str) -> Path:
        return self._state_dir / f"lease_{self.workspace}_{lease_id}.json"

    def _job_lease_index_path(self) -> Path:
        return self._state_dir / f"job_leases_{self.workspace}.json"

    def acquire_lease(
        self,
        job_id: str,
        runner_id: str,
        lease_timeout_seconds: Optional[float] = None,
    ) -> Optional[TaskLease]:
        """Acquire a lease for a job. Returns None if lease cannot be acquired."""
        # Check if job already has a valid lease
        existing_lease = self.get_active_lease_for_job(job_id)
        if existing_lease is not None and existing_lease.is_valid():
            if existing_lease.runner_id != runner_id:
                return None  # Another runner holds the lease
            # Same runner, return existing lease
            return existing_lease

        timeout = lease_timeout_seconds or self.default_lease_timeout_seconds
        warning_threshold = timeout * self.warning_threshold_ratio

        lease = TaskLease(
            job_id=job_id,
            runner_id=runner_id,
            workspace=self.workspace,
            lease_timeout_seconds=timeout,
            warning_threshold_seconds=warning_threshold,
            expires_at=utc_now() + timedelta(seconds=timeout),
        )

        self._persist_lease(lease)
        self._update_job_lease_index(job_id, lease.lease_id)
        return lease

    def renew_lease(
        self,
        lease_id: str,
        extension_seconds: Optional[float] = None,
    ) -> Optional[TaskLease]:
        """Renew an existing lease."""
        lease = self.get_lease(lease_id)
        if lease is None:
            return None
        if not lease.is_valid():
            return None

        lease.renew(extension_seconds)
        self._persist_lease(lease)
        return lease

    def release_lease(self, lease_id: str) -> bool:
        """Release a lease (task completed or failed)."""
        lease = self.get_lease(lease_id)
        if lease is None:
            return False

        lease.release()
        self._persist_lease(lease)
        return True

    def get_lease(self, lease_id: str) -> Optional[TaskLease]:
        """Get a lease by ID."""
        path = self._lease_path(lease_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _deserialize_dataclass(TaskLease, data)
        except Exception:
            return None

    def get_active_lease_for_job(self, job_id: str) -> Optional[TaskLease]:
        """Get the active lease for a job, if any."""
        index_path = self._job_lease_index_path()
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            lease_id = data.get("jobs", {}).get(job_id)
            if lease_id:
                lease = self.get_lease(lease_id)
                if lease and lease.is_valid():
                    return lease
        except Exception:
            pass
        return None

    def is_job_leased(self, job_id: str) -> bool:
        """Check if a job has an active lease."""
        lease = self.get_active_lease_for_job(job_id)
        return lease is not None and lease.is_valid()

    def list_expired_leases(self) -> list[TaskLease]:
        """List all expired leases."""
        expired = []
        for path in self._state_dir.glob(f"lease_{self.workspace}_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                lease = TaskLease(**data)
                if not lease.is_valid() and lease.state != LeaseState.RELEASED:
                    expired.append(lease)
            except Exception:
                continue
        return expired

    def cleanup_expired_leases(self, max_age_seconds: float = 3600) -> int:
        """Clean up old expired leases. Returns count cleaned."""
        cleaned = 0
        cutoff = utc_now() - timedelta(seconds=max_age_seconds)

        for lease in self.list_expired_leases():
            if lease.expires_at < cutoff:
                path = self._lease_path(lease.lease_id)
                try:
                    path.unlink()
                    cleaned += 1
                except Exception:
                    continue
        return cleaned

    def _persist_lease(self, lease: TaskLease) -> None:
        """Persist lease to disk."""
        path = self._lease_path(lease.lease_id)
        _write_json_with_dataclass(path, lease)

    def _update_job_lease_index(self, job_id: str, lease_id: str) -> None:
        """Update the job-to-lease mapping."""
        index_path = self._job_lease_index_path()
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"jobs": {}}

        data["jobs"][job_id] = lease_id
        write_json(index_path, data)


class CheckpointManager:
    """Manages recovery checkpoints for training jobs."""

    def __init__(self, workspace: str = "user_default"):
        self.workspace = workspace
        self._checkpoint_dir = resolve_home() / "data" / "checkpoints" / workspace
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        return self._checkpoint_dir / f"{checkpoint_id}.json"

    def save_checkpoint(self, checkpoint: RecoveryCheckpoint) -> Path:
        """Save a recovery checkpoint."""
        checkpoint.workspace = self.workspace
        path = self._checkpoint_path(checkpoint.checkpoint_id)
        _write_json_with_dataclass(path, checkpoint)
        return path

    def get_checkpoint(self, checkpoint_id: str) -> Optional[RecoveryCheckpoint]:
        """Get a checkpoint by ID."""
        path = self._checkpoint_path(checkpoint_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _deserialize_dataclass(RecoveryCheckpoint, data)
        except Exception:
            return None

    def get_latest_checkpoint_for_job(self, job_id: str) -> Optional[RecoveryCheckpoint]:
        """Get the most recent checkpoint for a job."""
        checkpoints = []
        for path in self._checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("job_id") == job_id:
                    checkpoint = RecoveryCheckpoint(**data)
                    checkpoints.append(checkpoint)
            except Exception:
                continue

        if not checkpoints:
            return None

        return max(checkpoints, key=lambda c: c.created_at)

    def list_checkpoints_for_job(self, job_id: str) -> list[RecoveryCheckpoint]:
        """List all checkpoints for a job."""
        checkpoints = []
        for path in self._checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("job_id") == job_id:
                    checkpoints.append(RecoveryCheckpoint(**data))
            except Exception:
                continue
        return sorted(checkpoints, key=lambda c: c.created_at, reverse=True)

    def can_resume_from_checkpoint(
        self,
        checkpoint_id: str,
        max_resume_attempts: int = 3,
    ) -> tuple[bool, Optional[str]]:
        """Check if checkpoint can be used for recovery."""
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            return False, "Checkpoint not found"

        if not checkpoint.can_resume(max_resume_attempts):
            if checkpoint.resume_attempt_count >= max_resume_attempts:
                return False, f"Max resume attempts ({max_resume_attempts}) exceeded"
            return False, "Checkpoint not resumable"

        if checkpoint.model_state_path and not Path(checkpoint.model_state_path).exists():
            return False, "Model state file not found"

        return True, None

    def increment_resume_attempt(self, checkpoint_id: str) -> Optional[RecoveryCheckpoint]:
        """Increment the resume attempt counter for a checkpoint."""
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            return None

        checkpoint.resume_attempt_count += 1
        checkpoint.last_resume_at = utc_now()
        self.save_checkpoint(checkpoint)
        return checkpoint

    def cleanup_old_checkpoints(self, job_id: str, keep_count: int = 3) -> int:
        """Clean up old checkpoints for a job, keeping only the most recent."""
        checkpoints = self.list_checkpoints_for_job(job_id)
        if len(checkpoints) <= keep_count:
            return 0

        removed = 0
        for checkpoint in checkpoints[keep_count:]:
            path = self._checkpoint_path(checkpoint.checkpoint_id)
            try:
                path.unlink()
                removed += 1
            except Exception:
                continue
        return removed


class DeadLetterQueue:
    """Manages permanently failed tasks for manual inspection."""

    def __init__(self, workspace: str = "user_default"):
        self.workspace = workspace
        self._dlq_dir = resolve_home() / "data" / "dead_letter"
        self._dlq_dir.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, entry_id: str) -> Path:
        return self._dlq_dir / f"{self.workspace}_{entry_id}.json"

    def add_entry(self, entry: DeadLetterEntry) -> Path:
        """Add an entry to the dead letter queue."""
        entry.workspace = self.workspace
        path = self._entry_path(entry.entry_id)
        _write_json_with_dataclass(path, entry)
        return path

    def get_entry(self, entry_id: str) -> Optional[DeadLetterEntry]:
        """Get a dead letter entry by ID."""
        path = self._entry_path(entry_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _deserialize_dataclass(DeadLetterEntry, data)
        except Exception:
            return None

    def list_entries(
        self,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ) -> list[DeadLetterEntry]:
        """List dead letter entries."""
        entries = []
        for path in self._dlq_dir.glob(f"{self.workspace}_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entry = DeadLetterEntry(**data)
                if resolved is None or entry.resolved == resolved:
                    entries.append(entry)
            except Exception:
                continue

        entries.sort(key=lambda e: e.failed_at, reverse=True)
        return entries[:limit]

    def resolve_entry(
        self,
        entry_id: str,
        action: str,
        note: Optional[str] = None,
    ) -> bool:
        """Mark a dead letter entry as resolved."""
        entry = self.get_entry(entry_id)
        if entry is None:
            return False

        entry.resolved = True
        entry.resolved_at = utc_now()
        entry.resolution_action = action
        if note:
            entry.resolution_note = note
        self.add_entry(entry)
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the dead letter queue."""
        all_entries = self.list_entries(limit=10000)
        resolved = [e for e in all_entries if e.resolved]
        unresolved = [e for e in all_entries if not e.resolved]

        return {
            "total": len(all_entries),
            "resolved": len(resolved),
            "unresolved": len(unresolved),
            "by_category": self._count_by_category(unresolved),
        }

    def _count_by_category(self, entries: list[DeadLetterEntry]) -> dict[str, int]:
        """Count entries by failure category."""
        counts: dict[str, int] = {}
        for entry in entries:
            cat = entry.failure_category or "unknown"
            counts[cat] = counts.get(cat, 0) + 1
        return counts


class AlertManager:
    """Manages alert generation and tracking."""

    def __init__(
        self,
        workspace: str = "user_default",
        thresholds: Optional[AlertThreshold] = None,
    ):
        self.workspace = workspace
        self.thresholds = thresholds or AlertThreshold()
        self._alerts_dir = resolve_home() / "data" / "alerts"
        self._alerts_dir.mkdir(parents=True, exist_ok=True)

    def _alert_path(self, alert_id: str) -> Path:
        return self._alerts_dir / f"{self.workspace}_{alert_id}.json"

    def create_alert(
        self,
        level: AlertLevel,
        scope: str,
        reason: str,
        message: str,
        job_id: Optional[str] = None,
        runner_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> AlertEvent:
        """Create a new alert."""
        alert = AlertEvent(
            level=level,
            scope=scope,
            reason=reason,
            message=message,
            job_id=job_id,
            runner_id=runner_id,
            workspace=self.workspace,
            context=context or {},
        )
        self._persist_alert(alert)
        return alert

    def get_alert(self, alert_id: str) -> Optional[AlertEvent]:
        """Get an alert by ID."""
        path = self._alert_path(alert_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _deserialize_dataclass(AlertEvent, data)
        except Exception:
            return None

    def list_alerts(
        self,
        level: Optional[AlertLevel] = None,
        scope: Optional[str] = None,
        resolved: Optional[bool] = False,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> list[AlertEvent]:
        """List alerts with optional filtering."""
        alerts = []
        for path in self._alerts_dir.glob(f"{self.workspace}_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                alert = _deserialize_dataclass(AlertEvent, data)

                if level is not None and alert.level != level:
                    continue
                if scope is not None and alert.scope != scope:
                    continue
                if resolved is not None and alert.resolved != resolved:
                    continue
                if acknowledged is not None and alert.acknowledged != acknowledged:
                    continue

                alerts.append(alert)
            except Exception:
                continue

        # Handle level comparison - level could be string or enum
        def _level_key(a: AlertEvent):
            level_val = a.level.value if isinstance(a.level, Enum) else a.level
            # Map level strings to numeric priority for sorting
            priority = {"critical": 4, "error": 3, "warning": 2, "attention": 1, "info": 0}
            return (priority.get(str(level_val).lower(), 0), a.timestamp)

        alerts.sort(key=_level_key, reverse=True)
        return alerts[:limit]

    def acknowledge_alert(self, alert_id: str, by: str) -> bool:
        """Acknowledge an alert."""
        alert = self.get_alert(alert_id)
        if alert is None:
            return False

        alert.acknowledge(by)
        self._persist_alert(alert)
        return True

    def resolve_alert(self, alert_id: str, note: Optional[str] = None) -> bool:
        """Resolve an alert."""
        alert = self.get_alert(alert_id)
        if alert is None:
            return False

        alert.resolve(note)
        self._persist_alert(alert)
        return True

    def check_consecutive_failures(self, count: int, job_id: Optional[str] = None) -> Optional[AlertEvent]:
        """Check if consecutive failures should trigger an alert."""
        level = self.thresholds.check_consecutive_failures(count)
        if level in {AlertLevel.INFO, AlertLevel.ATTENTION}:
            return None

        return self.create_alert(
            level=level,
            scope="task",
            reason="consecutive_failures",
            message=f"Task has {count} consecutive failures",
            job_id=job_id,
            context={"consecutive_failures": count, "threshold": self.thresholds.consecutive_failures_warning},
        )

    def check_heartbeat_delay(
        self,
        delay_seconds: float,
        runner_id: str,
        job_id: Optional[str] = None,
    ) -> Optional[AlertEvent]:
        """Check if heartbeat delay should trigger an alert."""
        level = self.thresholds.check_heartbeat_delay(delay_seconds)
        if level == AlertLevel.INFO:
            return None

        return self.create_alert(
            level=level,
            scope="runner",
            reason="heartbeat_delay",
            message=f"Runner heartbeat delayed by {delay_seconds:.1f}s",
            runner_id=runner_id,
            job_id=job_id,
            context={"delay_seconds": delay_seconds},
        )

    def check_task_stall(
        self,
        stall_seconds: float,
        job_id: str,
        runner_id: Optional[str] = None,
    ) -> Optional[AlertEvent]:
        """Check if task stall should trigger an alert."""
        level = self.thresholds.check_task_stall(stall_seconds)
        if level == AlertLevel.INFO:
            return None

        return self.create_alert(
            level=level,
            scope="task",
            reason="task_stall",
            message=f"Task stalled for {stall_seconds:.1f}s",
            job_id=job_id,
            runner_id=runner_id,
            context={"stall_seconds": stall_seconds},
        )

    def get_active_alerts_summary(self) -> dict[str, Any]:
        """Get summary of active (unresolved) alerts."""
        alerts = self.list_alerts(resolved=False, limit=1000)

        by_level: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        for alert in alerts:
            level_key = alert.level.value
            by_level[level_key] = by_level.get(level_key, 0) + 1
            by_scope[alert.scope] = by_scope.get(alert.scope, 0) + 1

        return {
            "total_active": len(alerts),
            "by_level": by_level,
            "by_scope": by_scope,
            "critical_count": by_level.get("critical", 0),
            "error_count": by_level.get("error", 0),
            "warning_count": by_level.get("warning", 0),
            "latest_alert": alerts[0].timestamp.isoformat() if alerts else None,
        }

    def _persist_alert(self, alert: AlertEvent) -> None:
        """Persist alert to disk."""
        path = self._alert_path(alert.alert_id)
        _write_json_with_dataclass(path, alert)


class ReliabilityService:
    """High-level service coordinating all reliability mechanisms."""

    def __init__(
        self,
        workspace: str = "user_default",
        lease_timeout_seconds: float = 60.0,
        restart_policy: Optional[RestartPolicy] = None,
        alert_thresholds: Optional[AlertThreshold] = None,
    ):
        self.workspace = workspace
        self.heartbeat = HeartbeatManager(workspace, lease_timeout_seconds)
        self.lease = LeaseManager(workspace, lease_timeout_seconds)
        self.checkpoint = CheckpointManager(workspace)
        self.dlq = DeadLetterQueue(workspace)
        self.alerts = AlertManager(workspace, alert_thresholds)
        self.restart_policy = restart_policy or RestartPolicy()

    def process_heartbeat(self, heartbeat: RunnerHeartbeat) -> Optional[AlertEvent]:
        """Process a runner heartbeat and generate alerts if needed."""
        self.heartbeat.record_heartbeat(heartbeat)

        # Check for heartbeat delays
        age_seconds = (utc_now() - heartbeat.timestamp).total_seconds()
        if heartbeat.runner_id:
            return self.alerts.check_heartbeat_delay(
                age_seconds, heartbeat.runner_id, heartbeat.job_id
            )
        return None

    def acquire_task_lease(
        self,
        job_id: str,
        runner_id: str,
        lease_timeout_seconds: Optional[float] = None,
    ) -> Optional[TaskLease]:
        """Acquire a lease for task execution."""
        return self.lease.acquire_lease(job_id, runner_id, lease_timeout_seconds)

    def save_recovery_checkpoint(self, checkpoint: RecoveryCheckpoint) -> Path:
        """Save a recovery checkpoint."""
        return self.checkpoint.save_checkpoint(checkpoint)

    def handle_task_failure(
        self,
        job_id: str,
        reason: str,
        metadata: TaskExecutionMetadata,
        error_details: Optional[dict[str, Any]] = None,
    ) -> tuple[RecoveryAction, Optional[AlertEvent]]:
        """Handle a task failure and determine recovery action."""
        metadata.record_failure(reason, error_details)

        # Check consecutive failures
        alert = self.alerts.check_consecutive_failures(
            metadata.consecutive_failures, job_id
        )

        # Determine recovery action
        if metadata.should_retry():
            # Check if we have a checkpoint to resume from
            if metadata.recovery_checkpoint_id:
                can_resume, _ = self.checkpoint.can_resume_from_checkpoint(
                    metadata.recovery_checkpoint_id
                )
                if can_resume:
                    return RecoveryAction.RESUME, alert

            return RecoveryAction.RETRY, alert

        # Max retries exceeded - move to dead letter
        entry = DeadLetterEntry(
            job_id=job_id,
            failure_reason=reason,
            error_details=error_details or {},
            retry_count=metadata.retry_count,
            original_task={"job_id": job_id, "workspace": self.workspace},
        )
        self.dlq.add_entry(entry)

        return RecoveryAction.DEAD_LETTER, alert

    def should_restart_daemon(self) -> tuple[bool, Optional[float]]:
        """Check if daemon restart is allowed under current policy."""
        return self.restart_policy.should_restart()

    def record_daemon_failure(self) -> float:
        """Record a daemon failure and return backoff delay."""
        return self.restart_policy.record_failure()

    def record_daemon_success(self) -> None:
        """Record a successful daemon execution."""
        self.restart_policy.record_success()

    def get_health_summary(self) -> dict[str, Any]:
        """Get overall health summary for operations console."""
        return {
            "workspace": self.workspace,
            "active_runners": len(self.heartbeat.list_active_runners()),
            "stalled_jobs": len(self.heartbeat.detect_stalled_jobs()),
            "expired_leases": len(self.lease.list_expired_leases()),
            "dead_letter_stats": self.dlq.get_stats(),
            "alerts_summary": self.alerts.get_active_alerts_summary(),
            "restart_policy": {
                "current_attempt": self.restart_policy.current_attempt,
                "max_attempts": self.restart_policy.max_restart_attempts,
                "can_restart": self.restart_policy.should_restart()[0],
            },
        }
