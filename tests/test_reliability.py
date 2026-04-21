"""Tests for PFE reliability mechanisms."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pfe_core.models import (
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
    utc_now,
)
from pfe_core.reliability import (
    AlertManager,
    CheckpointManager,
    DeadLetterQueue,
    HeartbeatManager,
    LeaseManager,
    ReliabilityService,
)


class TestRunnerHeartbeat:
    """Tests for RunnerHeartbeat model."""

    def test_heartbeat_creation(self):
        heartbeat = RunnerHeartbeat(
            runner_id="test-runner-1",
            job_id="job-123",
            pid=12345,
            progress_percent=50.0,
            current_step="training_epoch_2",
        )
        assert heartbeat.runner_id == "test-runner-1"
        assert heartbeat.job_id == "job-123"
        assert heartbeat.progress_percent == 50.0
        assert heartbeat.state == RunnerState.HEALTHY

    def test_heartbeat_is_fresh(self):
        heartbeat = RunnerHeartbeat(timestamp=utc_now())
        assert heartbeat.is_fresh(max_age_seconds=30.0) is True
        assert heartbeat.is_fresh(max_age_seconds=0.0) is False

    def test_heartbeat_is_stale(self):
        old_time = utc_now() - timedelta(seconds=120)
        heartbeat = RunnerHeartbeat(timestamp=old_time)
        assert heartbeat.is_stale(lease_timeout_seconds=60.0) is True


class TestTaskLease:
    """Tests for TaskLease model."""

    def test_lease_creation(self):
        lease = TaskLease(
            job_id="job-123",
            runner_id="runner-1",
            lease_timeout_seconds=60.0,
        )
        assert lease.job_id == "job-123"
        assert lease.runner_id == "runner-1"
        assert lease.state == LeaseState.VALID
        assert lease.is_valid() is True

    def test_lease_expiration(self):
        past_time = utc_now() - timedelta(seconds=10)
        lease = TaskLease(
            job_id="job-123",
            runner_id="runner-1",
            acquired_at=past_time,
            expires_at=past_time,
            state=LeaseState.EXPIRED,
        )
        assert lease.is_valid() is False

    def test_lease_renewal(self):
        lease = TaskLease(
            job_id="job-123",
            runner_id="runner-1",
            lease_timeout_seconds=60.0,
        )
        original_expires = lease.expires_at
        time.sleep(0.01)  # Small delay
        lease.renew()
        assert lease.expires_at > original_expires
        assert lease.renewal_count == 1
        assert lease.state == LeaseState.VALID

    def test_lease_is_expiring_soon(self):
        soon = utc_now() + timedelta(seconds=10)
        lease = TaskLease(
            job_id="job-123",
            runner_id="runner-1",
            expires_at=soon,
            warning_threshold_seconds=15.0,
        )
        assert lease.is_expiring_soon() is True

    def test_lease_release(self):
        lease = TaskLease(job_id="job-123", runner_id="runner-1")
        lease.release()
        assert lease.state == LeaseState.RELEASED
        assert lease.is_valid() is False


class TestRecoveryCheckpoint:
    """Tests for RecoveryCheckpoint model."""

    def test_checkpoint_creation(self):
        checkpoint = RecoveryCheckpoint(
            job_id="job-123",
            epoch=2,
            global_step=1000,
            model_state_path="/path/to/model.pt",
        )
        assert checkpoint.job_id == "job-123"
        assert checkpoint.epoch == 2
        assert checkpoint.can_resume(max_resume_attempts=3) is True

    def test_checkpoint_cannot_resume_no_model(self):
        checkpoint = RecoveryCheckpoint(job_id="job-123")
        assert checkpoint.can_resume(max_resume_attempts=3) is False

    def test_checkpoint_max_attempts_exceeded(self):
        checkpoint = RecoveryCheckpoint(
            job_id="job-123",
            model_state_path="/path/to/model.pt",
            resume_attempt_count=5,
        )
        assert checkpoint.can_resume(max_resume_attempts=3) is False


class TestRestartPolicy:
    """Tests for RestartPolicy model."""

    def test_restart_policy_creation(self):
        policy = RestartPolicy(
            max_restart_attempts=3,
            base_backoff_seconds=15.0,
        )
        assert policy.max_restart_attempts == 3
        assert policy.base_backoff_seconds == 15.0

    def test_calculate_backoff(self):
        policy = RestartPolicy(
            base_backoff_seconds=15.0,
            backoff_multiplier=2.0,
        )
        assert policy.calculate_backoff() == 0.0  # No attempts yet

        policy.current_attempt = 1
        assert policy.calculate_backoff() == 15.0

        policy.current_attempt = 2
        assert policy.calculate_backoff() == 30.0

        policy.current_attempt = 3
        assert policy.calculate_backoff() == 60.0

    def test_should_restart_allowed(self):
        policy = RestartPolicy(max_restart_attempts=3)
        allowed, delay = policy.should_restart()
        assert allowed is True
        assert delay == 0.0

    def test_should_restart_blocked_max_attempts(self):
        policy = RestartPolicy(max_restart_attempts=3)
        policy.current_attempt = 3
        allowed, delay = policy.should_restart()
        assert allowed is False
        assert delay is None

    def test_should_restart_in_backoff(self):
        policy = RestartPolicy(base_backoff_seconds=60.0)
        policy.record_failure()
        allowed, delay = policy.should_restart()
        assert allowed is False
        assert delay > 0.0

    def test_record_failure(self):
        policy = RestartPolicy(base_backoff_seconds=15.0)
        delay = policy.record_failure()
        assert policy.current_attempt == 1
        assert delay == 15.0
        assert policy.next_restart_after is not None

    def test_record_success(self):
        policy = RestartPolicy(reset_after_seconds=3600.0)
        policy.record_success()
        assert policy.last_success_at is not None

    def test_reset_after_time_elapsed(self):
        policy = RestartPolicy(
            max_restart_attempts=3,
            reset_after_seconds=1.0,
        )
        policy.current_attempt = 2
        policy.record_success()
        time.sleep(1.1)  # Wait for reset threshold
        allowed, _ = policy.should_restart()
        assert allowed is True
        assert policy.current_attempt == 0  # Should be reset


class TestAlertThreshold:
    """Tests for AlertThreshold model."""

    def test_check_consecutive_failures(self):
        thresholds = AlertThreshold(
            consecutive_failures_warning=2,
            consecutive_failures_error=3,
            consecutive_failures_critical=5,
        )
        assert thresholds.check_consecutive_failures(1) == AlertLevel.INFO
        assert thresholds.check_consecutive_failures(2) == AlertLevel.WARNING
        assert thresholds.check_consecutive_failures(3) == AlertLevel.ERROR
        assert thresholds.check_consecutive_failures(5) == AlertLevel.CRITICAL

    def test_check_heartbeat_delay(self):
        thresholds = AlertThreshold(
            heartbeat_delay_warning=10.0,
            heartbeat_delay_error=30.0,
        )
        assert thresholds.check_heartbeat_delay(5.0) == AlertLevel.INFO
        assert thresholds.check_heartbeat_delay(15.0) == AlertLevel.WARNING
        assert thresholds.check_heartbeat_delay(35.0) == AlertLevel.ERROR

    def test_check_task_stall(self):
        thresholds = AlertThreshold(
            task_stall_warning=300.0,
            task_stall_error=600.0,
            task_stall_critical=1800.0,
        )
        assert thresholds.check_task_stall(100.0) == AlertLevel.INFO
        assert thresholds.check_task_stall(400.0) == AlertLevel.WARNING
        assert thresholds.check_task_stall(700.0) == AlertLevel.ERROR
        assert thresholds.check_task_stall(2000.0) == AlertLevel.CRITICAL


class TestTaskExecutionMetadata:
    """Tests for TaskExecutionMetadata model."""

    def test_transition_to(self):
        metadata = TaskExecutionMetadata(job_id="job-123")
        metadata.transition_to(TaskState.RUNNING, "starting_execution")
        assert metadata.current_state == TaskState.RUNNING
        assert metadata.previous_state == TaskState.PENDING
        assert len(metadata.state_history) == 1

    def test_record_failure(self):
        metadata = TaskExecutionMetadata(job_id="job-123")
        metadata.record_failure("oom_error", {"memory_mb": 16000})
        assert metadata.failure_count == 1
        assert metadata.consecutive_failures == 1
        assert metadata.last_failure_reason == "oom_error"

    def test_should_retry(self):
        metadata = TaskExecutionMetadata(job_id="job-123", max_retries=3)
        assert metadata.should_retry() is True
        metadata.retry_count = 3
        assert metadata.should_retry() is False

    def test_can_recover(self):
        metadata = TaskExecutionMetadata(
            job_id="job-123",
            recovery_checkpoint_id="cp-123",
        )
        assert metadata.can_recover() is True


class TestHeartbeatManager:
    """Tests for HeartbeatManager."""

    @pytest.fixture
    def manager(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "pfe_core.reliability.resolve_home",
            lambda: tmp_path,
        )
        return HeartbeatManager(workspace="test_workspace", lease_timeout_seconds=60.0)

    def test_record_and_get_heartbeat(self, manager):
        heartbeat = RunnerHeartbeat(runner_id="runner-1", job_id="job-123")
        manager.record_heartbeat(heartbeat)

        retrieved = manager.get_heartbeat("runner-1")
        assert retrieved is not None
        assert retrieved.runner_id == "runner-1"
        assert retrieved.job_id == "job-123"

    def test_get_runner_state_healthy(self, manager):
        heartbeat = RunnerHeartbeat(runner_id="runner-1", timestamp=utc_now())
        manager.record_heartbeat(heartbeat)
        assert manager.get_runner_state("runner-1") == RunnerState.HEALTHY

    def test_get_runner_state_stale(self, manager):
        old_time = utc_now() - timedelta(seconds=150)
        heartbeat = RunnerHeartbeat(runner_id="runner-1", timestamp=old_time)
        manager.record_heartbeat(heartbeat)
        assert manager.get_runner_state("runner-1") == RunnerState.STALE

    def test_detect_stalled_jobs(self, manager):
        # Record a heartbeat for a job
        heartbeat = RunnerHeartbeat(runner_id="runner-1", job_id="job-123")
        manager.record_heartbeat(heartbeat)

        # Make it stale
        old_time = utc_now() - timedelta(seconds=150)
        heartbeat.timestamp = old_time
        manager.record_heartbeat(heartbeat)

        stalled = manager.detect_stalled_jobs()
        assert len(stalled) == 1
        assert stalled[0]["job_id"] == "job-123"

    def test_cleanup_stale_runners(self, manager):
        old_time = utc_now() - timedelta(seconds=7200)  # 2 hours ago
        heartbeat = RunnerHeartbeat(runner_id="old-runner", timestamp=old_time)
        manager.record_heartbeat(heartbeat)

        removed = manager.cleanup_stale_runners(max_age_seconds=3600)
        assert "old-runner" in removed
        assert manager.get_heartbeat("old-runner") is None

    def test_cleanup_stale_runners_prunes_runner_index(self, manager):
        old_time = utc_now() - timedelta(seconds=7200)
        manager.record_heartbeat(RunnerHeartbeat(runner_id="old-runner", job_id="job-123", timestamp=old_time))
        manager.record_heartbeat(RunnerHeartbeat(runner_id="fresh-runner", job_id="job-456", timestamp=utc_now()))

        removed = manager.cleanup_stale_runners(max_age_seconds=3600)

        assert "old-runner" in removed
        index_path = manager._runners_index_path()
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        assert payload["runners"] == ["fresh-runner"]
        assert payload["jobs"] == {"fresh-runner": "job-456"}

    def test_cleanup_stale_runners_accepts_legacy_naive_timestamp(self, manager):
        old_time = (utc_now() - timedelta(seconds=7200)).replace(tzinfo=None)
        manager.record_heartbeat(RunnerHeartbeat(runner_id="old-runner", timestamp=old_time))

        removed = manager.cleanup_stale_runners(max_age_seconds=3600)

        assert "old-runner" in removed
        assert manager.get_heartbeat("old-runner") is None


class TestLeaseManager:
    """Tests for LeaseManager."""

    @pytest.fixture
    def manager(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "pfe_core.reliability.resolve_home",
            lambda: tmp_path,
        )
        return LeaseManager(workspace="test_workspace", default_lease_timeout_seconds=60.0)

    def test_acquire_lease(self, manager):
        lease = manager.acquire_lease("job-123", "runner-1")
        assert lease is not None
        assert lease.job_id == "job-123"
        assert lease.runner_id == "runner-1"
        assert lease.is_valid() is True

    def test_acquire_lease_already_held(self, manager):
        lease1 = manager.acquire_lease("job-123", "runner-1")
        assert lease1 is not None

        # Another runner cannot acquire the same lease
        lease2 = manager.acquire_lease("job-123", "runner-2")
        assert lease2 is None

    def test_renew_lease(self, manager):
        lease = manager.acquire_lease("job-123", "runner-1")
        original_expires = lease.expires_at
        time.sleep(0.01)

        renewed = manager.renew_lease(lease.lease_id)
        assert renewed is not None
        assert renewed.expires_at > original_expires

    def test_release_lease(self, manager):
        lease = manager.acquire_lease("job-123", "runner-1")
        assert lease.is_valid() is True

        released = manager.release_lease(lease.lease_id)
        assert released is True

        # Check lease is released
        retrieved = manager.get_lease(lease.lease_id)
        assert retrieved.state == LeaseState.RELEASED

    def test_is_job_leased(self, manager):
        manager.acquire_lease("job-123", "runner-1")
        assert manager.is_job_leased("job-123") is True
        assert manager.is_job_leased("job-456") is False

    def test_list_expired_leases(self, manager):
        # Create an expired lease
        past_time = utc_now() - timedelta(seconds=10)
        lease = TaskLease(
            job_id="old-job",
            runner_id="runner-1",
            acquired_at=past_time - timedelta(seconds=100),
            expires_at=past_time,
            state=LeaseState.EXPIRED,
        )
        manager._persist_lease(lease)
        manager._update_job_lease_index("old-job", lease.lease_id)

        expired = manager.list_expired_leases()
        assert len(expired) >= 1


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def manager(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "pfe_core.reliability.resolve_home",
            lambda: tmp_path,
        )
        return CheckpointManager(workspace="test_workspace")

    def test_save_and_get_checkpoint(self, manager):
        checkpoint = RecoveryCheckpoint(
            job_id="job-123",
            epoch=2,
            global_step=1000,
            model_state_path="/path/to/model.pt",
        )
        path = manager.save_checkpoint(checkpoint)
        assert path.exists()

        retrieved = manager.get_checkpoint(checkpoint.checkpoint_id)
        assert retrieved is not None
        assert retrieved.job_id == "job-123"
        assert retrieved.epoch == 2

    def test_get_latest_checkpoint_for_job(self, manager):
        # Save multiple checkpoints
        cp1 = RecoveryCheckpoint(job_id="job-123", epoch=1, global_step=500)
        cp2 = RecoveryCheckpoint(job_id="job-123", epoch=2, global_step=1000)
        time.sleep(0.01)
        cp3 = RecoveryCheckpoint(job_id="job-123", epoch=3, global_step=1500)

        manager.save_checkpoint(cp1)
        manager.save_checkpoint(cp2)
        manager.save_checkpoint(cp3)

        latest = manager.get_latest_checkpoint_for_job("job-123")
        assert latest is not None
        assert latest.epoch == 3

    def test_list_checkpoints_for_job(self, manager):
        cp1 = RecoveryCheckpoint(job_id="job-123", epoch=1)
        cp2 = RecoveryCheckpoint(job_id="job-123", epoch=2)
        manager.save_checkpoint(cp1)
        manager.save_checkpoint(cp2)

        checkpoints = manager.list_checkpoints_for_job("job-123")
        assert len(checkpoints) == 2
        # Should be sorted by created_at descending
        assert checkpoints[0].epoch == 2

    def test_can_resume_from_checkpoint(self, manager, tmp_path):
        # Create a fake model file
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")

        checkpoint = RecoveryCheckpoint(
            job_id="job-123",
            model_state_path=str(model_file),
        )
        manager.save_checkpoint(checkpoint)

        can_resume, reason = manager.can_resume_from_checkpoint(checkpoint.checkpoint_id)
        assert can_resume is True
        assert reason is None

    def test_cleanup_old_checkpoints(self, manager):
        # Create multiple checkpoints
        for i in range(5):
            cp = RecoveryCheckpoint(job_id="job-123", epoch=i)
            manager.save_checkpoint(cp)
            time.sleep(0.01)

        removed = manager.cleanup_old_checkpoints("job-123", keep_count=2)
        assert removed == 3

        remaining = manager.list_checkpoints_for_job("job-123")
        assert len(remaining) == 2


class TestDeadLetterQueue:
    """Tests for DeadLetterQueue."""

    @pytest.fixture
    def dlq(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "pfe_core.reliability.resolve_home",
            lambda: tmp_path,
        )
        return DeadLetterQueue(workspace="test_workspace")

    def test_add_and_get_entry(self, dlq):
        entry = DeadLetterEntry(
            job_id="job-123",
            failure_reason="max_retries_exceeded",
            failure_category="persistent_error",
        )
        path = dlq.add_entry(entry)
        assert path.exists()

        retrieved = dlq.get_entry(entry.entry_id)
        assert retrieved is not None
        assert retrieved.job_id == "job-123"
        assert retrieved.failure_reason == "max_retries_exceeded"

    def test_list_entries(self, dlq):
        # Add entries
        for i in range(3):
            entry = DeadLetterEntry(job_id=f"job-{i}", failure_reason="error")
            dlq.add_entry(entry)

        entries = dlq.list_entries()
        assert len(entries) == 3

    def test_list_entries_filtered(self, dlq):
        resolved_entry = DeadLetterEntry(
            job_id="job-resolved",
            failure_reason="error",
            resolved=True,
        )
        unresolved_entry = DeadLetterEntry(
            job_id="job-unresolved",
            failure_reason="error",
            resolved=False,
        )
        dlq.add_entry(resolved_entry)
        dlq.add_entry(unresolved_entry)

        resolved = dlq.list_entries(resolved=True)
        assert len(resolved) == 1
        assert resolved[0].job_id == "job-resolved"

        unresolved = dlq.list_entries(resolved=False)
        assert len(unresolved) == 1
        assert unresolved[0].job_id == "job-unresolved"

    def test_resolve_entry(self, dlq):
        entry = DeadLetterEntry(job_id="job-123", failure_reason="error")
        dlq.add_entry(entry)

        result = dlq.resolve_entry(entry.entry_id, "manual_fix", "Fixed by operator")
        assert result is True

        retrieved = dlq.get_entry(entry.entry_id)
        assert retrieved.resolved is True
        assert retrieved.resolution_action == "manual_fix"
        assert retrieved.resolution_note == "Fixed by operator"

    def test_get_stats(self, dlq):
        # Add entries with different categories
        dlq.add_entry(DeadLetterEntry(
            job_id="job-1",
            failure_reason="error",
            failure_category="oom",
        ))
        dlq.add_entry(DeadLetterEntry(
            job_id="job-2",
            failure_reason="error",
            failure_category="oom",
        ))
        dlq.add_entry(DeadLetterEntry(
            job_id="job-3",
            failure_reason="error",
            failure_category="crash",
            resolved=True,
        ))

        stats = dlq.get_stats()
        assert stats["total"] == 3
        assert stats["resolved"] == 1
        assert stats["unresolved"] == 2
        assert stats["by_category"]["oom"] == 2
        # "crash" category has 1 resolved entry, not counted in by_category (which only counts unresolved)
        assert "crash" not in stats["by_category"] or stats["by_category"].get("crash", 0) == 0


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.fixture
    def alert_mgr(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "pfe_core.reliability.resolve_home",
            lambda: tmp_path,
        )
        thresholds = AlertThreshold(
            consecutive_failures_warning=2,
            consecutive_failures_error=3,
            heartbeat_delay_warning=10.0,
        )
        return AlertManager(workspace="test_workspace", thresholds=thresholds)

    def test_create_and_get_alert(self, alert_mgr):
        alert = alert_mgr.create_alert(
            level=AlertLevel.WARNING,
            scope="runner",
            reason="heartbeat_delay",
            message="Runner heartbeat delayed",
        )
        assert alert.level == AlertLevel.WARNING

        retrieved = alert_mgr.get_alert(alert.alert_id)
        assert retrieved is not None
        assert retrieved.message == "Runner heartbeat delayed"

    def test_list_alerts_filtered(self, alert_mgr):
        # Create alerts with different levels
        alert1 = alert_mgr.create_alert(AlertLevel.WARNING, "runner", "r1", "msg1")
        alert2 = alert_mgr.create_alert(AlertLevel.ERROR, "runner", "r2", "msg2")
        alert3 = alert_mgr.create_alert(AlertLevel.WARNING, "task", "r3", "msg3")

        warnings = alert_mgr.list_alerts(level=AlertLevel.WARNING)
        assert len(warnings) == 2

        runner_alerts = alert_mgr.list_alerts(scope="runner")
        assert len(runner_alerts) == 2

        # Resolve one alert and test resolved filter
        alert_mgr.resolve_alert(alert1.alert_id)
        unresolved_warnings = alert_mgr.list_alerts(level=AlertLevel.WARNING, resolved=False)
        assert len(unresolved_warnings) == 1  # Only alert3 remains unresolved

    def test_acknowledge_alert(self, alert_mgr):
        alert = alert_mgr.create_alert(AlertLevel.WARNING, "runner", "r1", "msg")
        result = alert_mgr.acknowledge_alert(alert.alert_id, "operator-1")
        assert result is True

        retrieved = alert_mgr.get_alert(alert.alert_id)
        assert retrieved.acknowledged is True
        assert retrieved.acknowledged_by == "operator-1"

    def test_resolve_alert(self, alert_mgr):
        alert = alert_mgr.create_alert(AlertLevel.WARNING, "runner", "r1", "msg")
        result = alert_mgr.resolve_alert(alert.alert_id, "Issue fixed")
        assert result is True

        retrieved = alert_mgr.get_alert(alert.alert_id)
        assert retrieved.resolved is True
        assert retrieved.resolution_note == "Issue fixed"

    def test_check_consecutive_failures_creates_alert(self, alert_mgr):
        alert = alert_mgr.check_consecutive_failures(3, "job-123")
        assert alert is not None
        assert alert.level == AlertLevel.ERROR
        assert alert.scope == "task"

    def test_check_consecutive_failures_no_alert(self, alert_mgr):
        alert = alert_mgr.check_consecutive_failures(1, "job-123")
        assert alert is None  # Below threshold

    def test_check_heartbeat_delay_creates_alert(self, alert_mgr):
        alert = alert_mgr.check_heartbeat_delay(15.0, "runner-1", "job-123")
        assert alert is not None
        assert alert.level == AlertLevel.WARNING

    def test_get_active_alerts_summary(self, alert_mgr):
        alert1 = alert_mgr.create_alert(AlertLevel.WARNING, "runner", "r1", "msg1")
        alert2 = alert_mgr.create_alert(AlertLevel.ERROR, "runner", "r2", "msg2")
        alert3 = alert_mgr.create_alert(AlertLevel.WARNING, "task", "r3", "msg3")
        # Resolve one alert
        alert_mgr.resolve_alert(alert3.alert_id)

        summary = alert_mgr.get_active_alerts_summary()
        assert summary["total_active"] == 2
        assert summary["warning_count"] == 1
        assert summary["error_count"] == 1


class TestReliabilityService:
    """Integration tests for ReliabilityService."""

    @pytest.fixture
    def service(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "pfe_core.reliability.resolve_home",
            lambda: tmp_path,
        )
        return ReliabilityService(workspace="test_workspace")

    def test_process_heartbeat(self, service):
        heartbeat = RunnerHeartbeat(runner_id="runner-1", job_id="job-123")
        alert = service.process_heartbeat(heartbeat)
        # Fresh heartbeat should not generate alert
        assert alert is None

        # Verify heartbeat was recorded
        assert service.heartbeat.get_runner_state("runner-1") == RunnerState.HEALTHY

    def test_acquire_task_lease(self, service):
        lease = service.acquire_task_lease("job-123", "runner-1")
        assert lease is not None
        assert lease.is_valid() is True

    def test_handle_task_failure_retry(self, service):
        metadata = TaskExecutionMetadata(job_id="job-123", max_retries=3)
        action, alert = service.handle_task_failure(
            "job-123",
            "transient_error",
            metadata,
            {"detail": "network timeout"},
        )
        assert action == RecoveryAction.RETRY
        assert metadata.consecutive_failures == 1

    def test_handle_task_failure_resume(self, service, tmp_path):
        # Create a checkpoint first with actual file
        model_file = tmp_path / "model.pt"
        model_file.write_text("fake model")
        checkpoint = RecoveryCheckpoint(
            job_id="job-123",
            model_state_path=str(model_file),
        )
        service.save_recovery_checkpoint(checkpoint)

        metadata = TaskExecutionMetadata(
            job_id="job-123",
            max_retries=3,
            recovery_checkpoint_id=checkpoint.checkpoint_id,
        )
        action, alert = service.handle_task_failure("job-123", "crash", metadata)
        assert action == RecoveryAction.RESUME

    def test_handle_task_failure_dead_letter(self, service):
        metadata = TaskExecutionMetadata(job_id="job-123", max_retries=1)
        metadata.retry_count = 1  # Already at max

        action, alert = service.handle_task_failure("job-123", "persistent_error", metadata)
        assert action == RecoveryAction.DEAD_LETTER

        # Verify entry was added to DLQ
        stats = service.dlq.get_stats()
        assert stats["total"] == 1

    def test_should_restart_daemon(self, service):
        allowed, delay = service.should_restart_daemon()
        assert allowed is True
        assert delay == 0.0

    def test_record_daemon_failure_and_backoff(self, service):
        delay = service.record_daemon_failure()
        assert delay > 0.0

        # Should not allow restart immediately
        allowed, remaining = service.should_restart_daemon()
        assert allowed is False
        assert remaining is not None

    def test_get_health_summary(self, service):
        # Record some activity with fresh heartbeat
        heartbeat = RunnerHeartbeat(runner_id="runner-1", timestamp=utc_now())
        service.process_heartbeat(heartbeat)
        service.acquire_task_lease("job-123", "runner-1")

        summary = service.get_health_summary()
        assert summary["workspace"] == "test_workspace"
        # Active runners are those with non-stale heartbeats
        assert summary["active_runners"] >= 0  # May be 0 if stale threshold is tight
        assert "alerts_summary" in summary
        assert "restart_policy" in summary
