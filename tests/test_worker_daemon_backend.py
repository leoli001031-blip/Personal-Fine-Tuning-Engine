from __future__ import annotations

import asyncio
import os
import signal
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

from pfe_core.pipeline import PipelineService
from pfe_core.reliability import ReliabilityService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_core.worker_daemon import run_worker_daemon
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class _DummyProcess:
    def __init__(self, pid: int = 43210) -> None:
        self.pid = pid


class WorkerDaemonBackendTests(unittest.TestCase):
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

    def _app(self, service: PipelineService):
        return create_app(
            ServiceBundle(
                inference=InferenceServiceAdapter(service),
                pipeline=PipelineServiceAdapter(service),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
            )
        )

    def test_start_and_stop_worker_daemon_persist_runtime_state(self) -> None:
        service = self._service()
        with patch("pfe_core.pipeline.subprocess.Popen", return_value=_DummyProcess()), patch.object(
            PipelineService,
            "_pid_exists",
            return_value=True,
        ), patch("pfe_core.pipeline.os.kill") as kill_mock:
            started = service.start_train_queue_daemon(workspace="user_default")
            self.assertEqual(started["desired_state"], "running")
            self.assertEqual(started["requested_action"], "start")
            self.assertEqual(started["command_status"], "spawned")
            self.assertTrue(started["active"])
            self.assertEqual(started["lock_state"], "active")
            self.assertGreaterEqual(started["history_count"], 1)

            history = service.train_queue_daemon_history(workspace="user_default", limit=10)
            self.assertEqual(history["last_event"], "start_requested")

            stopped = service.stop_train_queue_daemon(workspace="user_default")
            self.assertEqual(stopped["desired_state"], "stopped")
            self.assertEqual(stopped["requested_action"], "stop")
            self.assertIn(stopped["command_status"], {"requested", "signaled"})
            kill_mock.assert_called()

    def test_http_surface_exposes_worker_daemon_controls(self) -> None:
        service = self._service()
        app = self._app(service)

        async def scenario() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
            with patch("pfe_core.pipeline.subprocess.Popen", return_value=_DummyProcess()), patch.object(
                PipelineService,
                "_pid_exists",
                return_value=True,
            ), patch("pfe_core.pipeline.os.kill"):
                start_result = await smoke_test_request(app, path="/pfe/auto-train/start-worker-daemon", method="POST")
                status_result = await smoke_test_request(app, path="/pfe/auto-train/worker-daemon", method="GET")
                stop_result = await smoke_test_request(app, path="/pfe/auto-train/stop-worker-daemon", method="POST")
            return start_result["body"], status_result["body"], stop_result["body"]

        start_body, status_body, stop_body = asyncio.run(scenario())
        self.assertEqual(start_body["desired_state"], "running")
        self.assertEqual(status_body["requested_action"], "start")
        self.assertEqual(stop_body["desired_state"], "stopped")

    def test_http_surface_exposes_recover_and_restart_controls(self) -> None:
        service = self._service()
        app = self._app(service)
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

        async def scenario() -> tuple[dict[str, object], dict[str, object]]:
            with patch("pfe_core.pipeline.subprocess.Popen", return_value=_DummyProcess(pid=65432)), patch.object(
                PipelineService,
                "_pid_exists",
                return_value=False,
            ), patch("pfe_core.pipeline.os.kill"):
                recover_result = await smoke_test_request(app, path="/pfe/auto-train/recover-worker-daemon", method="POST")
                restart_result = await smoke_test_request(app, path="/pfe/auto-train/restart-worker-daemon", method="POST")
            return recover_result["body"], restart_result["body"]

        recover_body, restart_body = asyncio.run(scenario())
        self.assertEqual(recover_body["requested_action"], "recover")
        self.assertEqual(restart_body["requested_action"], "restart")

    def test_recover_worker_daemon_restarts_when_stale(self) -> None:
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
        with patch("pfe_core.pipeline.subprocess.Popen", return_value=_DummyProcess(pid=54321)), patch.object(
            PipelineService,
            "_pid_exists",
            return_value=False,
        ):
            recovered = service.recover_train_queue_daemon(workspace="user_default")
        self.assertEqual(recovered["requested_action"], "recover")
        self.assertEqual(recovered["command_status"], "spawned")
        self.assertTrue(recovered["can_recover"] is False or recovered["active"])
        history = service.train_queue_daemon_history(workspace="user_default", limit=10)
        self.assertEqual(history["last_event"], "recover_requested")

    def test_recover_worker_daemon_blocks_when_backoff_is_active(self) -> None:
        service = self._service()
        now = datetime.now(timezone.utc)
        service._persist_train_queue_daemon_state(
            {
                "workspace": "user_default",
                "desired_state": "running",
                "observed_state": "stopped",
                "command_status": "completed",
                "active": False,
                "pid": None,
                "auto_restart_enabled": True,
                "restart_attempts": 1,
                "max_restart_attempts": 3,
                "restart_backoff_seconds": 15.0,
                "next_restart_after": (now + timedelta(seconds=30)).isoformat(),
            },
            workspace="user_default",
        )
        recovered = service.recover_train_queue_daemon(workspace="user_default")
        self.assertEqual(recovered["recovery_reason"], "restart_backoff_active")
        history = service.train_queue_daemon_history(workspace="user_default", limit=10)
        self.assertEqual(history["last_event"], "recover_blocked")

    def test_daemon_status_keeps_active_lock_when_heartbeat_is_fresh(self) -> None:
        service = self._service()
        now = datetime.now(timezone.utc).isoformat()
        service._persist_train_queue_daemon_state(
            {
                "workspace": "user_default",
                "desired_state": "running",
                "observed_state": "running",
                "requested_action": "recover",
                "command_status": "running",
                "active": True,
                "pid": 7772,
                "last_heartbeat_at": now,
                "lease_renewed_at": now,
                "heartbeat_interval_seconds": 2.0,
                "lease_timeout_seconds": 15.0,
                "heartbeat_timeout_seconds": 15.0,
                "auto_recover_enabled": True,
                "auto_restart_enabled": True,
            },
            workspace="user_default",
        )

        with patch.object(PipelineService, "_pid_exists", return_value=True):
            status = service.train_queue_daemon_status(workspace="user_default")

        self.assertEqual(status["heartbeat_state"], "fresh")
        self.assertEqual(status["lock_state"], "active")
        self.assertEqual(status["health_state"], "healthy")
        self.assertFalse(status["can_recover"])

    def test_run_worker_daemon_reuses_a_stable_runner_id_for_heartbeats(self) -> None:
        captured_runner_ids: list[str] = []
        loop_counter = {"count": 0}

        def capture_heartbeat(_self: ReliabilityService, heartbeat) -> None:
            captured_runner_ids.append(heartbeat.runner_id)

        def stop_after_two_loops(
            service_self: PipelineService,
            *,
            workspace: str | None = None,
            max_seconds: float = 0.0,
            idle_sleep_seconds: float = 0.0,
        ) -> dict[str, object]:
            del service_self, workspace, max_seconds, idle_sleep_seconds
            loop_counter["count"] += 1
            if loop_counter["count"] >= 2:
                os.kill(os.getpid(), signal.SIGINT)
            return {"auto_train_trigger_action": {"processed_count": 0, "failed_count": 0}}

        with patch.object(ReliabilityService, "process_heartbeat", autospec=True, side_effect=capture_heartbeat), patch.object(
            PipelineService,
            "run_train_queue_worker_runner",
            autospec=True,
            side_effect=stop_after_two_loops,
        ):
            exit_code = run_worker_daemon(
                workspace="user_default",
                runner_max_seconds=0.1,
                idle_sleep_seconds=0.0,
                heartbeat_interval_seconds=0.1,
                lease_timeout_seconds=5.0,
                enable_reliability=True,
            )

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(len(captured_runner_ids), 2)
        self.assertEqual(len(set(captured_runner_ids)), 1)


if __name__ == "__main__":
    unittest.main()
