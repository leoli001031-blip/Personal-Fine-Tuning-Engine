from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.config import PFEConfig
from pfe_core.pipeline import PipelineService
from pfe_server.app import build_serve_plan, smoke_test_request


class AutoTriggerHttpControlsTests(unittest.TestCase):
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

    def _seed_ready_service(self) -> PipelineService:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.failure_backoff_minutes = 15
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        return service

    def _signal_payload(self, event_id: str, request_id: str, session_id: str) -> dict[str, object]:
        return {
            "event_id": event_id,
            "request_id": request_id,
            "session_id": session_id,
            "source_event_id": f"{event_id}-source",
            "source_event_ids": [f"{event_id}-source", event_id],
            "event_type": "accept",
            "user_input": "帮我推进今天的任务",
            "model_output": "先定一个最小动作。",
            "user_action": {"type": "accept"},
        }

    def _seed_failure(self) -> None:
        service = self._seed_ready_service()
        with patch.object(service, "train_result", side_effect=RuntimeError("trainer boom")):
            service.signal(self._signal_payload("evt-http-failure", "req-http-failure", "sess-http-failure"))

    def _app(self):
        plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
        return plan.app

    def test_status_http_smoke_surfaces_auto_train_status_schema(self) -> None:
        self._seed_failure()
        app = self._app()

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        self.assertIn("auto_train_trigger", body)
        trigger = body["auto_train_trigger"]
        self.assertTrue(trigger["enabled"])
        self.assertIn("state", trigger)
        self.assertIn("ready", trigger)
        self.assertIn("blocked_reasons", trigger)
        self.assertIn("last_result", trigger)
        self.assertIn("last_result_summary", trigger)
        self.assertIn("failure_backoff_elapsed", trigger)
        self.assertIn("consecutive_failures", trigger)
        self.assertIn("queue_dedup_scope", trigger)
        self.assertIn("queue_priority_policy", trigger)
        self.assertIn("metadata", body)
        self.assertIn("auto_train_trigger", body["metadata"])
        self.assertEqual(body["metadata"]["auto_train_trigger"]["state"], trigger["state"])

    def test_retry_http_smoke_returns_action_schema_and_keeps_backoff_blocked(self) -> None:
        self._seed_failure()
        app = self._app()

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/auto-train/retry", method="POST")
            return result["body"]

        body = asyncio.run(scenario())
        self.assertEqual(body["action"], "retry")
        self.assertEqual(body["status"], "blocked")
        self.assertIn("auto_train_trigger", body)
        self.assertIn("metadata", body)
        self.assertIn("auto_train_trigger_action", body["metadata"])
        self.assertEqual(body["metadata"]["auto_train_trigger_action"]["action"], "retry")
        self.assertEqual(body["auto_train_trigger"]["state"], "blocked")
        self.assertFalse(body["triggered"])
        self.assertIn("failure_backoff_active", body["auto_train_trigger"]["blocked_reasons"])

    def test_reset_http_smoke_returns_action_schema_and_restores_ready_status(self) -> None:
        self._seed_failure()
        app = self._app()

        async def reset_scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/auto-train/reset", method="POST")
            return result["body"]

        reset_body = asyncio.run(reset_scenario())
        self.assertEqual(reset_body["action"], "reset")
        self.assertEqual(reset_body["status"], "completed")
        self.assertIn("auto_train_trigger", reset_body)
        self.assertTrue(reset_body["auto_train_trigger"]["ready"])
        self.assertIn("metadata", reset_body)
        self.assertIn("auto_train_trigger_action", reset_body["metadata"])
        self.assertEqual(reset_body["metadata"]["auto_train_trigger_action"]["action"], "reset")

        async def status_scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        status_body = asyncio.run(status_scenario())
        self.assertTrue(status_body["auto_train_trigger"]["ready"])
        self.assertNotIn("failure_backoff_active", status_body["auto_train_trigger"]["blocked_reasons"])


if __name__ == "__main__":
    unittest.main()
