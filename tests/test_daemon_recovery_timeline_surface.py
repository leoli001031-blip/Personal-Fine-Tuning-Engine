from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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


class DaemonRecoveryTimelineSurfaceTests(unittest.TestCase):
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

    def _seed_daemon_history(self, service: PipelineService) -> None:
        service._append_train_queue_daemon_history(
            workspace="user_default",
            event="start_requested",
            reason="daemon_start_requested",
            metadata={"note": "boot"},
        )
        service._append_train_queue_daemon_history(
            workspace="user_default",
            event="recover_requested",
            reason="daemon_stale",
            metadata={"note": "auto_recovery"},
        )
        service._append_train_queue_daemon_history(
            workspace="user_default",
            event="completed",
            reason="idle_exit",
            metadata={"note": "daemon_idle"},
        )

    def test_http_status_exposes_daemon_recovery_timeline_summary(self) -> None:
        service = self._service()
        self._seed_daemon_history(service)
        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        daemon_timeline = body["daemon_timeline"]
        self.assertEqual(daemon_timeline["count"], 3)
        self.assertEqual(daemon_timeline["recovery_event_count"], 2)
        self.assertEqual(daemon_timeline["last_recovery_event"], "recover_requested")
        self.assertEqual(daemon_timeline["last_recovery_reason"], "daemon_stale")
        self.assertEqual(daemon_timeline["last_recovery_note"], "auto_recovery")
        self.assertEqual(daemon_timeline["recent_anomaly_reason"], "daemon_stale")
        self.assertGreaterEqual(len(daemon_timeline["recent_events"]), 3)
        self.assertGreaterEqual(len(daemon_timeline["recent_recovery_events"]), 2)
        self.assertIn("daemon_timeline", body["metadata"])
        self.assertEqual(body["operations_daemon_timeline"]["count"], 3)

    def test_cli_status_formats_daemon_recovery_timeline_summary(self) -> None:
        payload = {
            "workspace": "user_default",
            "daemon_timeline": {
                "count": 3,
                "recovery_event_count": 2,
                "last_event": "completed",
                "last_reason": "idle_exit",
                "last_recovery_event": "recover_requested",
                "last_recovery_reason": "daemon_stale",
                "last_recovery_note": "auto_recovery",
                "recent_anomaly_reason": "daemon_stale",
                "latest_timestamp": "2026-03-26T11:01:00+00:00",
                "recent_recovery_events": [
                    {
                        "timestamp": "2026-03-26T11:00:00+00:00",
                        "event": "start_requested",
                        "reason": "daemon_start_requested",
                        "note": "boot",
                    },
                    {
                        "timestamp": "2026-03-26T11:00:30+00:00",
                        "event": "recover_requested",
                        "reason": "daemon_stale",
                        "note": "auto_recovery",
                    },
                ],
            },
            "operations_console": {
                "attention_needed": True,
                "summary_line": "daemon=stale",
                "daemon_timeline": {
                    "count": 3,
                    "recovery_event_count": 2,
                    "last_event": "completed",
                    "last_reason": "idle_exit",
                    "last_recovery_event": "recover_requested",
                    "last_recovery_reason": "daemon_stale",
                    "last_recovery_note": "auto_recovery",
                    "latest_timestamp": "2026-03-26T11:01:00+00:00",
                },
            },
        }

        text = _format_status(payload, workspace="user_default")
        from tests.matrix_test_compat import strip_ansi
        clean = strip_ansi(text)
        self.assertIn("DAEMON TIMELINE", clean)
        self.assertIn("count:", clean)
        self.assertIn("3", clean)
        self.assertIn("recovery event count:", clean)
        self.assertIn("2", clean)
        self.assertIn("last recovery event:", clean)
        self.assertIn("recover_requested", clean)
        self.assertIn("last recovery reason:", clean)
        self.assertIn("daemon_stale", clean)
        self.assertIn("last recovery note:", clean)
        self.assertIn("auto_recovery", clean)
        self.assertIn("recent anomaly reason:", clean)
        self.assertIn("recent recovery events:", clean)
        self.assertIn("start_requested", clean)
        self.assertIn("recover_requested", clean)


if __name__ == "__main__":
    unittest.main()
