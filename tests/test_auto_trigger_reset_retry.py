from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
import asyncio
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli.main import _format_status
from pfe_core.config import PFEConfig
from pfe_core.pipeline import PipelineService
from pfe_server.app import build_serve_plan, smoke_test_request
from tests.matrix_test_compat import strip_ansi


class AutoTriggerResetRetryTests(unittest.TestCase):
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
            service.signal(self._signal_payload("evt-failure", "req-failure", "sess-failure"))

    def test_manual_reset_clears_persisted_failure_backoff_state(self) -> None:
        self._seed_failure()
        state_path = PipelineService._auto_trigger_state_path(workspace="user_default")
        self.assertTrue(state_path.exists())

        fresh_service = PipelineService()
        status = fresh_service.reset_auto_train_trigger()
        trigger = status["auto_train_trigger"]

        self.assertTrue(trigger["enabled"])
        self.assertTrue(trigger["ready"])
        self.assertEqual(trigger["state"], "ready")
        self.assertTrue(trigger["failure_backoff_elapsed"])
        self.assertNotIn("failure_backoff_active", trigger["blocked_reasons"])
        self.assertEqual(trigger["consecutive_failures"], 0)
        self.assertEqual(status["auto_train_trigger_action"]["action"], "reset")
        self.assertEqual(status["auto_train_trigger_action"]["status"], "completed")

        text = _format_status(status, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ AUTO TRAIN ACTION ]", clean)
        self.assertIn("action:", clean)
        self.assertIn("reset", clean)
        self.assertIn("status:", clean)
        self.assertIn("completed", clean)
        self.assertIn("reason:", clean)
        self.assertIn("state_cleared", clean)
        self.assertIn("[ AUTO TRAIN TRIGGER ]", clean)
        self.assertIn("enabled:", clean)
        self.assertIn("state:", clean)
        self.assertIn("ready:", clean)
        self.assertNotIn("failure_backoff_active", clean)

    def test_retry_while_backoff_active_returns_clear_blocked_status(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.failure_backoff_minutes = 15
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)

        with patch.object(service, "train_result", side_effect=RuntimeError("trainer boom")):
            service.signal(self._signal_payload("evt-retry-failure", "req-retry-failure", "sess-retry-failure"))

        with patch.object(service, "train_result") as train_result_mock:
            retry_result = service.retry_auto_train_trigger()

        auto_train = retry_result["auto_train_trigger"]
        train_result_mock.assert_not_called()
        self.assertFalse(auto_train["ready"])
        self.assertFalse(retry_result["auto_train_trigger_action"]["triggered"])
        self.assertIn("failure_backoff_active", auto_train["blocked_reasons"])
        self.assertEqual(auto_train["reason"], "failure_backoff_active")
        self.assertEqual(auto_train["state"], "blocked")
        self.assertIn("train_failed", auto_train["last_result_summary"])
        self.assertEqual(retry_result["auto_train_trigger_action"]["action"], "retry")
        self.assertEqual(retry_result["auto_train_trigger_action"]["status"], "blocked")

    def test_server_status_surfaces_reset_and_retry_changes(self) -> None:
        self._seed_failure()
        plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
        app = plan.app

        async def before_reset() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        before_body = asyncio.run(before_reset())
        self.assertIn("failure_backoff_active", before_body["auto_train_trigger"]["blocked_reasons"])

        async def run_retry() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/auto-train/retry", method="POST")
            return result["body"]

        retry_body = asyncio.run(run_retry())
        self.assertEqual(retry_body["action"], "retry")
        self.assertEqual(retry_body["status"], "blocked")
        self.assertIn("failure_backoff_active", retry_body["auto_train_trigger"]["blocked_reasons"])

        async def run_reset() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/auto-train/reset", method="POST")
            return result["body"]

        reset_body = asyncio.run(run_reset())
        self.assertEqual(reset_body["action"], "reset")
        self.assertEqual(reset_body["status"], "completed")
        self.assertTrue(reset_body["auto_train_trigger"]["ready"])

        async def after_reset() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        after_body = asyncio.run(after_reset())
        self.assertTrue(after_body["auto_train_trigger"]["ready"])
        self.assertNotIn("failure_backoff_active", after_body["auto_train_trigger"]["blocked_reasons"])
        self.assertEqual(after_body["metadata"]["auto_train_trigger"]["state"], after_body["auto_train_trigger"]["state"])


if __name__ == "__main__":
    unittest.main()
