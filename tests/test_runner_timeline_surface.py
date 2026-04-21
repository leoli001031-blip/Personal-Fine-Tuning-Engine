from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli.main import _format_status
from pfe_core.pipeline import PipelineService
from tests.matrix_test_compat import strip_ansi
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class RunnerTimelineSurfaceTests(unittest.TestCase):
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

    def test_http_status_exposes_runner_timeline_summary(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-runner-timeline-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": "20260326-600",
            },
            workspace="user_default",
        )
        service.run_train_queue_worker_runner(max_seconds=0.1, idle_sleep_seconds=0.0)
        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        runner_timeline = body["runner_timeline"]
        self.assertGreaterEqual(runner_timeline["count"], 1)
        self.assertIn(runner_timeline["current_lock_state"], {"idle", "active", "stale"})
        self.assertIn("summary_line", runner_timeline)
        self.assertIn("takeover_event_count=0", runner_timeline["summary_line"])
        self.assertIn("recent_events", runner_timeline)
        self.assertIn("operations_runner_timeline", body)
        self.assertEqual(body["operations_runner_timeline"]["count"], runner_timeline["count"])
        self.assertIn("runner_timeline", body["operations_console"])

    def test_cli_status_formats_runner_timeline_summary(self) -> None:
        payload = {
            "workspace": "user_default",
            "runner_timeline": {
                "count": 3,
                "last_event": "completed",
                "last_reason": "idle_exit",
                "takeover_event_count": 1,
                "last_takeover_event": "stale_lock_takeover",
                "last_takeover_reason": "runner_already_active",
                "current_active": False,
                "current_lock_state": "idle",
                "current_stop_requested": False,
                "current_lease_expires_at": "2026-03-26T12:00:00+00:00",
                "latest_timestamp": "2026-03-26T11:58:00+00:00",
                "recent_anomaly_reason": "runner_already_active",
                "recent_events": [
                    {"timestamp": "2026-03-26T11:56:00+00:00", "event": "started", "reason": "run_worker_runner"},
                    {"timestamp": "2026-03-26T11:57:00+00:00", "event": "processed", "reason": "queued_job"},
                    {"timestamp": "2026-03-26T11:58:00+00:00", "event": "completed", "reason": "idle_exit"},
                ],
                "recent_takeover_events": [
                    {
                        "timestamp": "2026-03-26T11:57:30+00:00",
                        "event": "stale_lock_takeover",
                        "reason": "runner_already_active",
                        "note": "manual_takeover",
                    }
                ],
            },
            "operations_console": {
                "attention_needed": False,
                "summary_line": "candidate-stage=promoted",
                "runner_timeline": {
                    "count": 3,
                    "last_event": "completed",
                    "last_reason": "idle_exit",
                    "takeover_event_count": 1,
                    "last_takeover_event": "stale_lock_takeover",
                    "last_takeover_reason": "runner_already_active",
                    "recent_anomaly_reason": "runner_already_active",
                    "current_active": False,
                    "current_lock_state": "idle",
                    "current_stop_requested": False,
                    "current_lease_expires_at": "2026-03-26T12:00:00+00:00",
                    "latest_timestamp": "2026-03-26T11:58:00+00:00",
                },
            },
        }

        text = _format_status(payload, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ RUNNER TIMELINE ]", clean)
        self.assertIn("count:", clean)
        self.assertIn("last event:", clean)
        self.assertIn("completed", clean)
        self.assertIn("last reason:", clean)
        self.assertIn("idle_exit", clean)
        self.assertIn("takeover event count:", clean)
        self.assertIn("last takeover event:", clean)
        self.assertIn("stale_lock_takeover", clean)
        self.assertIn("last takeover reason:", clean)
        self.assertIn("runner_already_active", clean)
        self.assertIn("recent anomaly reason:", clean)
        self.assertIn("current lock state:", clean)
        self.assertIn("idle", clean)
        self.assertIn("current stop requested:", clean)
        self.assertIn("event=completed", clean)
        self.assertIn("event=stale_lock_takeover", clean)


if __name__ == "__main__":
    unittest.main()
