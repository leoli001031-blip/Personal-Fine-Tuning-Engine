from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli.main import _format_status
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class WorkerRunnerSurfaceTests(unittest.TestCase):
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

    def test_worker_runner_processes_queue_and_surfaces_status(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-runner-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": "20260326-001",
            },
            workspace="user_default",
        )

        snapshot = service.run_train_queue_worker_runner(max_seconds=0.1, idle_sleep_seconds=0.0)
        self.assertEqual(snapshot["auto_train_trigger_action"]["action"], "run_worker_runner")
        self.assertGreaterEqual(snapshot["auto_train_trigger_action"]["processed_count"], 1)
        self.assertIn("worker_runner", snapshot["train_queue"])
        self.assertFalse(snapshot["train_queue"]["worker_runner"]["active"])

        text = _format_status(snapshot, workspace="user_default")
        self.assertIn("worker runner:", text)
        self.assertIn("processed count=", text)
        history = service.train_queue_worker_runner_history(limit=10)
        self.assertGreaterEqual(history["count"], 2)
        self.assertEqual(history["items"][0]["event"], "started")
        self.assertEqual(history["items"][-1]["event"], "completed")

    def test_stop_worker_runner_sets_stop_requested_state(self) -> None:
        service = self._service()
        stop_snapshot = service.stop_train_queue_worker_runner()
        self.assertEqual(stop_snapshot["auto_train_trigger_action"]["action"], "stop_worker_runner")
        self.assertIn(stop_snapshot["auto_train_trigger_action"]["status"], {"requested", "noop"})
        runner = service.train_queue_worker_runner_status()
        self.assertTrue(runner["stop_requested"])
        history = service.train_queue_worker_runner_history(limit=10)
        self.assertEqual(history["last_event"], "stop_requested")

    def test_worker_runner_reentry_is_blocked_when_lock_is_active(self) -> None:
        service = self._service()
        now = datetime.now(timezone.utc).isoformat()
        service._persist_train_queue_worker_state(
            {
                "active": True,
                "stop_requested": False,
                "pid": os.getpid(),
                "started_at": now,
                "last_heartbeat_at": now,
                "last_completed_at": None,
                "loop_cycles": 3,
                "processed_count": 2,
                "failed_count": 0,
                "stopped_reason": None,
                "last_action": "run_worker_runner",
                "max_seconds": 30.0,
                "idle_sleep_seconds": 1.0,
            },
            workspace="user_default",
        )

        snapshot = service.run_train_queue_worker_runner(max_seconds=0.1, idle_sleep_seconds=0.0)
        self.assertEqual(snapshot["auto_train_trigger_action"]["action"], "run_worker_runner")
        self.assertEqual(snapshot["auto_train_trigger_action"]["status"], "blocked")
        self.assertEqual(snapshot["auto_train_trigger_action"]["reason"], "runner_already_active")
        self.assertEqual(snapshot["train_queue"]["worker_runner"]["lock_state"], "active")
        history = service.train_queue_worker_runner_history(limit=10)
        self.assertEqual(history["last_event"], "blocked_reentry")

    def test_stale_worker_lock_is_marked_and_can_be_taken_over(self) -> None:
        service = self._service()
        stale_heartbeat = "2026-03-20T10:00:00+00:00"
        service._persist_train_queue_worker_state(
            {
                "active": True,
                "stop_requested": False,
                "pid": 99999,
                "started_at": stale_heartbeat,
                "last_heartbeat_at": stale_heartbeat,
                "last_completed_at": None,
                "loop_cycles": 1,
                "processed_count": 0,
                "failed_count": 0,
                "stopped_reason": None,
                "last_action": "run_worker_runner",
                "max_seconds": 30.0,
                "idle_sleep_seconds": 1.0,
            },
            workspace="user_default",
        )
        service._append_train_queue_item(
            {
                "job_id": "job-stale-takeover-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": "20260326-003",
            },
            workspace="user_default",
        )

        before = service.train_queue_worker_runner_status()
        self.assertEqual(before["lock_state"], "stale")
        self.assertIsNotNone(before["lease_expires_at"])

        snapshot = service.run_train_queue_worker_runner(max_seconds=0.1, idle_sleep_seconds=0.0)
        self.assertEqual(snapshot["auto_train_trigger_action"]["action"], "run_worker_runner")
        self.assertTrue(snapshot["auto_train_trigger_action"]["reason"].startswith("stale_lock_takeover"))
        self.assertIn("worker_runner", snapshot["train_queue"])
        self.assertEqual(snapshot["train_queue"]["worker_runner"]["lock_state"], "idle")
        app = self._app(service)

        async def status_scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        status_body = asyncio.run(status_scenario())
        runner_timeline = status_body["runner_timeline"]
        self.assertGreaterEqual(runner_timeline["takeover_event_count"], 1)
        self.assertEqual(runner_timeline["last_takeover_event"], "started")
        self.assertEqual(runner_timeline["last_takeover_reason"], "stale_lock_takeover")
        self.assertEqual(runner_timeline["recent_anomaly_reason"], "stale_lock_takeover")
        history = service.train_queue_worker_runner_history(limit=10)
        self.assertEqual(history["items"][-2]["reason"], "stale_lock_takeover")
        self.assertEqual(history["items"][-1]["event"], "completed")

    def test_http_and_frontend_surface_worker_runner_controls(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-runner-http-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": "20260326-002",
            },
            workspace="user_default",
        )
        app = self._app(service)

        async def http_scenario() -> tuple[dict[str, object], dict[str, object], str]:
            run_result = await smoke_test_request(
                app,
                path="/pfe/auto-train/run-worker-runner",
                method="POST",
                query_params={"max_seconds": "0.1", "idle_sleep_seconds": "0"},
            )
            status_result = await smoke_test_request(app, path="/pfe/auto-train/worker-runner", method="GET")
            timeline_result = await smoke_test_request(app, path="/pfe/status", method="GET")
            root_result = await smoke_test_request(app, path="/", method="GET")
            return run_result["body"], status_result["body"], timeline_result["body"], root_result["text"]

        run_body, status_body, timeline_body, root_text = asyncio.run(http_scenario())
        self.assertEqual(run_body["action"], "run_worker_runner")
        self.assertIn(run_body["status"], {"completed", "idle"})
        self.assertIn("processed_count", run_body)
        self.assertIn("active", status_body)
        self.assertIn("runner_timeline", timeline_body)
        self.assertIn("operations_runner_timeline", timeline_body)
        self.assertIn("runner_timeline", timeline_body["operations_console"])
        self.assertIn("Worker Runner", root_text)
        self.assertIn("/pfe/auto-train/run-worker-runner?max_seconds=5&idle_sleep_seconds=0", root_text)
        self.assertIn("/pfe/auto-train/stop-worker-runner", root_text)

    def test_http_surface_exposes_worker_runner_history(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-runner-history-http-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": "20260326-004",
            },
            workspace="user_default",
        )
        service.run_train_queue_worker_runner(max_seconds=0.1, idle_sleep_seconds=0.0)
        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/auto-train/worker-runner/history", method="GET", query_params={"limit": "10"})
            return result["body"]

        body = asyncio.run(scenario())
        self.assertGreaterEqual(body["count"], 2)
        self.assertEqual(body["last_event"], "completed")

    def test_http_status_embeds_runner_timeline_in_operations_console(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-runner-status-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": "20260326-005",
            },
            workspace="user_default",
        )
        service.run_train_queue_worker_runner(max_seconds=0.1, idle_sleep_seconds=0.0)
        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        runner_timeline = body["operations_console"]["runner_timeline"]
        self.assertGreaterEqual(runner_timeline["count"], 2)
        self.assertEqual(runner_timeline["last_event"], "completed")
        self.assertIn("timelines", body["operations_console"])
        self.assertEqual(body["operations_console"]["timelines"]["runner"]["last_event"], "completed")


if __name__ == "__main__":
    unittest.main()
