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
from tests.matrix_test_compat import strip_ansi
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class DaemonHealthSurfaceTests(unittest.TestCase):
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

    def test_http_status_exposes_daemon_atomic_health_surface(self) -> None:
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
                "auto_restart_enabled": False,
                "auto_recover_enabled": False,
                "restart_attempts": 0,
                "max_restart_attempts": 3,
                "restart_backoff_seconds": 15.0,
            },
            workspace="user_default",
        )
        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        health = body["operations_health"]
        recovery = body["operations_recovery"]
        console_daemon = body["operations_console"]["daemon"]
        self.assertEqual(health["health_state"], "stale")
        self.assertEqual(health["lease_state"], "expired")
        self.assertEqual(health["heartbeat_state"], "stale")
        self.assertIn(health["restart_policy_state"], {"ready", "manual_only", "auto_restart_disabled"})
        self.assertIn("recovery_action", health)
        self.assertIn("daemon_recovery_action", recovery)
        self.assertEqual(console_daemon["health_state"], "stale")
        self.assertEqual(console_daemon["lease_state"], "expired")
        self.assertEqual(console_daemon["heartbeat_state"], "stale")
        self.assertIn(console_daemon["restart_policy_state"], {"ready", "manual_only", "auto_restart_disabled"})
        self.assertEqual(console_daemon["recovery_action"], "manual_recover")
        self.assertTrue(body["operations_next_actions"])
        self.assertIn("recover_worker_daemon", body["operations_next_actions"])

    def test_cli_status_formats_daemon_atomic_health_surface(self) -> None:
        payload = {
            "workspace": "user_default",
            "operations_alerts": [
                {"reason": "daemon_stale", "detail": "daemon lease expired", "severity": "warning"},
            ],
            "operations_health": {
                "status": "attention",
                "health_state": "stale",
                "lease_state": "expired",
                "heartbeat_state": "expired",
                "restart_policy_state": "manual_only",
                "recovery_action": "recover_worker_daemon",
                "daemon_lock_state": "stale",
                "runner_lock_state": "idle",
                "candidate_state": "pending_eval",
                "queue_state": "awaiting_confirmation",
            },
            "operations_recovery": {
                "daemon_recovery_needed": True,
                "daemon_recovery_reason": "daemon_stale",
                "daemon_recovery_state": "recoverable",
                "daemon_recovery_action": "recover_worker_daemon",
            },
            "operations_next_actions": ["recover_worker_daemon"],
            "operations_console": {
                "attention_needed": True,
                "summary_line": "daemon=stale",
                "next_actions": ["recover_worker_daemon"],
            },
        }

        text = _format_status(payload, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("health state:", clean)
        self.assertIn("stale", clean)
        self.assertIn("daemon lock state:", clean)
        self.assertIn("lease state:", clean)
        self.assertIn("expired", clean)
        self.assertIn("heartbeat state:", clean)
        self.assertIn("restart policy state:", clean)
        self.assertIn("manual_only", clean)
        self.assertIn("recovery action:", clean)
        self.assertIn("recover_worker_daemon", clean)
        self.assertIn("daemon recovery needed:", clean)
        self.assertIn("daemon recovery reason:", clean)
        self.assertIn("daemon_stale", clean)
        self.assertIn("daemon recovery action:", clean)


if __name__ == "__main__":
    unittest.main()
