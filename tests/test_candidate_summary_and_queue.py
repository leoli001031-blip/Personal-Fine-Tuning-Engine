from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pfe_cli.main import _format_status
from pfe_core.config import PFEConfig
from pfe_core.pipeline import PipelineService
from pfe_server.app import build_serve_plan, smoke_test_request
from tests.matrix_test_compat import strip_ansi

class CandidateSummaryAndQueueTests(unittest.TestCase):
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

    def test_status_surfaces_candidate_summary_when_pending_eval_exists(self) -> None:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        service.train_result(method="mock_local", epochs=1, base_model="base", workspace="user_default", backend="mock_local")

        status = service.status()
        candidate = status["candidate_summary"]
        self.assertEqual(candidate["candidate_state"], "pending_eval")
        self.assertEqual(candidate["pending_eval_count"], 1)
        self.assertFalse(candidate["candidate_needs_promotion"])
        self.assertEqual(candidate["promotion_gate_reason"], "eval_required")
        self.assertEqual(candidate["promotion_gate_action"], "run_eval")

        text = _format_status(status, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ CANDIDATE SUMMARY ]", clean)
        self.assertIn("candidate version:", clean)
        self.assertIn("candidate state:         pending_eval", clean)

    def test_status_surfaces_train_queue_after_auto_train_attempt(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)

        class _FakeTrainResult:
            version = "20260325-777"
            metrics = {"state": "pending_eval", "num_fresh_samples": 1, "num_replay_samples": 0}

        with patch.object(service, "train_result", return_value=_FakeTrainResult()):
            service.signal(
                {
                    "event_id": "evt-queue-1",
                    "request_id": "req-queue-1",
                    "session_id": "sess-queue-1",
                    "source_event_id": "evt-source-queue-1",
                    "source_event_ids": ["evt-source-queue-1", "evt-queue-1"],
                    "event_type": "accept",
                    "user_input": "帮我推进下一步",
                    "model_output": "先做最小动作。",
                    "user_action": {"type": "accept"},
                }
            )

        status = service.status()
        queue = status["train_queue"]
        self.assertEqual(queue["count"], 1)
        self.assertEqual(queue["counts"]["completed"], 1)
        self.assertEqual(queue["last_item"]["adapter_version"], "20260325-777")

        text = _format_status(status, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ TRAIN QUEUE ]", clean)
        self.assertIn("count:                   1", clean)
        self.assertIn("states:                  completed:1", clean)

    def test_server_status_surfaces_candidate_summary_and_queue(self) -> None:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        service.train_result(method="mock_local", epochs=1, base_model="base", workspace="user_default", backend="mock_local")

        plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
        app = plan.app

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        self.assertIn("candidate_summary", body)
        self.assertIn("train_queue", body)
        self.assertIn("candidate_summary", body["metadata"])
        self.assertIn("train_queue", body["metadata"])

if __name__ == "__main__":
    unittest.main()
