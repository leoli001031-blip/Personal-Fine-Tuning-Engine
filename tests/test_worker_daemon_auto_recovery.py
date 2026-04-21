from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.config import PFEConfig
from pfe_core.pipeline import PipelineService


class _DummyProcess:
    def __init__(self, pid: int = 45678) -> None:
        self.pid = pid


class WorkerDaemonAutoRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("PFE_HOME")
        self.pfe_home = Path(self.tempdir.name) / ".pfe"
        os.environ["PFE_HOME"] = str(self.pfe_home)

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = self.previous_home
        self.tempdir.cleanup()

    def _service(self) -> PipelineService:
        return PipelineService()

    def test_daemon_status_auto_recovers_stale_daemon_when_enabled(self) -> None:
        service = self._service()
        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        service._persist_train_queue_daemon_state(
            {
                "workspace": "user_default",
                "desired_state": "running",
                "observed_state": "running",
                "command_status": "running",
                "active": True,
                "pid": 999999,
                "started_at": stale_time,
                "last_heartbeat_at": stale_time,
                "auto_restart_enabled": True,
                "restart_attempts": 0,
                "max_restart_attempts": 3,
                "restart_backoff_seconds": 15.0,
            },
            workspace="user_default",
        )

        with patch("pfe_core.pipeline.subprocess.Popen", return_value=_DummyProcess(pid=65432)), patch.object(
            PipelineService,
            "_pid_exists",
            return_value=False,
        ):
            summary = service.train_queue_daemon_status(workspace="user_default")

        self.assertEqual(summary["requested_action"], "recover")
        self.assertEqual(summary["command_status"], "spawned")
        self.assertEqual(summary["last_requested_by"], "auto_recovery")
        self.assertEqual(summary["auto_recovery_count"], 1)
        self.assertTrue(summary["auto_recover_enabled"])
        self.assertEqual(summary["recovery_state"], "recovering")
        self.assertEqual(summary["recovery_action"], "auto_recover")
        history = service.train_queue_daemon_history(workspace="user_default", limit=10)
        self.assertEqual(history["last_event"], "recover_requested")
        self.assertEqual(history["last_reason"], "daemon_stale")

    def test_daemon_status_respects_auto_recover_disabled_config(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.queue_daemon_auto_recover = False
        config.save(home=self.pfe_home)
        service = self._service()
        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        service._persist_train_queue_daemon_state(
            {
                "workspace": "user_default",
                "desired_state": "running",
                "observed_state": "running",
                "command_status": "running",
                "active": True,
                "pid": 999999,
                "started_at": stale_time,
                "last_heartbeat_at": stale_time,
                "auto_restart_enabled": True,
                "restart_attempts": 0,
                "max_restart_attempts": 3,
                "restart_backoff_seconds": 15.0,
            },
            workspace="user_default",
        )

        with patch("pfe_core.pipeline.subprocess.Popen") as popen_mock, patch.object(
            PipelineService,
            "_pid_exists",
            return_value=False,
        ):
            summary = service.train_queue_daemon_status(workspace="user_default")

        popen_mock.assert_not_called()
        self.assertEqual(summary["lock_state"], "stale")
        self.assertTrue(summary["recovery_needed"])
        self.assertTrue(summary["can_recover"])
        self.assertFalse(summary["auto_recover_enabled"])
        self.assertEqual(summary["auto_recovery_count"], 0)
        self.assertEqual(summary["health_state"], "stale")
        self.assertEqual(summary["lease_state"], "expired")
        self.assertEqual(summary["heartbeat_state"], "stale")
        self.assertEqual(summary["restart_policy_state"], "ready")
        self.assertEqual(summary["recovery_action"], "manual_recover")

    def test_start_daemon_uses_configured_heartbeat_and_lease_settings(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.queue_daemon_heartbeat_interval_seconds = 1.5
        config.trainer.trigger.queue_daemon_lease_timeout_seconds = 12.0
        config.save(home=self.pfe_home)
        service = self._service()

        with patch("pfe_core.pipeline.subprocess.Popen", return_value=_DummyProcess(pid=55555)) as popen_mock, patch.object(
            PipelineService,
            "_pid_exists",
            return_value=True,
        ):
            summary = service.start_train_queue_daemon(workspace="user_default")

        command = popen_mock.call_args.args[0]
        self.assertIn("--heartbeat-interval-seconds", command)
        self.assertIn("--lease-timeout-seconds", command)
        self.assertEqual(summary["heartbeat_interval_seconds"], 1.5)
        self.assertEqual(summary["lease_timeout_seconds"], 12.0)
        self.assertTrue(summary["auto_recover_enabled"])

    def test_daemon_status_surfaces_atomic_health_states(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.queue_daemon_auto_recover = False
        config.trainer.trigger.queue_daemon_heartbeat_interval_seconds = 2.0
        config.trainer.trigger.queue_daemon_lease_timeout_seconds = 15.0
        config.save(home=self.pfe_home)
        service = self._service()
        heartbeat = (datetime.now(timezone.utc) - timedelta(seconds=8)).isoformat()
        service._persist_train_queue_daemon_state(
            {
                "workspace": "user_default",
                "desired_state": "running",
                "observed_state": "running",
                "command_status": "running",
                "active": True,
                "pid": os.getpid(),
                "started_at": heartbeat,
                "last_heartbeat_at": heartbeat,
                "auto_restart_enabled": True,
                "auto_recover_enabled": False,
                "restart_attempts": 1,
                "max_restart_attempts": 3,
                "restart_backoff_seconds": 15.0,
            },
            workspace="user_default",
        )

        with patch.object(PipelineService, "_pid_exists", return_value=True):
            summary = service.train_queue_daemon_status(workspace="user_default")

        self.assertEqual(summary["lock_state"], "active")
        self.assertEqual(summary["health_state"], "healthy")
        self.assertEqual(summary["lease_state"], "expiring")
        self.assertEqual(summary["heartbeat_state"], "delayed")
        self.assertEqual(summary["restart_policy_state"], "ready")
        self.assertEqual(summary["recovery_action"], "none")


if __name__ == "__main__":
    unittest.main()
