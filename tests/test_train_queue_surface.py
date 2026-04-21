from __future__ import annotations

import asyncio
import importlib
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

from pfe_cli.main import _format_status
from pfe_core.config import PFEConfig
from pfe_core.pipeline import PipelineService
from tests.matrix_test_compat import strip_ansi
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class _FakeAdapterStore:
    def __init__(self, rows: list[dict[str, object]], latest_version: str | None):
        self.rows = [dict(row) for row in rows]
        self.latest_version = latest_version

    def list_version_records(self, limit: int = 100) -> list[dict[str, object]]:
        del limit
        return [dict(row) for row in self.rows]

    def current_latest_version(self) -> str | None:
        return self.latest_version


class TrainQueueSurfaceTests(unittest.TestCase):
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

    def _seed_signal_ready_service(self) -> PipelineService:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
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

    def test_queue_state_machine_tracks_queued_running_completed_and_failed(self) -> None:
        service = self._service()

        queued = service._append_train_queue_item(
            {
                "job_id": "job-queued-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
            },
            workspace="user_default",
        )
        self.assertEqual(queued["state"], "queued")

        queued_snapshot = service.status()["train_queue"]
        self.assertEqual(queued_snapshot["count"], 1)
        self.assertEqual(queued_snapshot["counts"]["queued"], 1)
        self.assertEqual(queued_snapshot["current"]["state"], "queued")

        running = service._update_train_queue_item("job-queued-1", {"state": "running"}, workspace="user_default")
        self.assertEqual(running["state"], "running")

        running_snapshot = service.status()["train_queue"]
        self.assertEqual(running_snapshot["counts"]["running"], 1)
        self.assertEqual(running_snapshot["current"]["state"], "running")

        completed = service._update_train_queue_item("job-queued-1", {"state": "completed"}, workspace="user_default")
        self.assertEqual(completed["state"], "completed")

        service._append_train_queue_item(
            {
                "job_id": "job-failed-1",
                "state": "failed",
                "workspace": "user_default",
                "source": "signal_auto_train",
            },
            workspace="user_default",
        )
        snapshot = service.status()["train_queue"]
        self.assertEqual(snapshot["count"], 2)
        self.assertEqual(snapshot["counts"]["completed"], 1)
        self.assertEqual(snapshot["counts"]["failed"], 1)
        self.assertEqual(snapshot["last_item"]["state"], "failed")
        self.assertIn("history", snapshot["last_item"])
        self.assertGreaterEqual(snapshot["history_summary"]["transition_count"], 4)
        self.assertIn(snapshot["history_summary"]["last_transition"]["event"], {"enqueued", "failed", "updated"})

    def test_queue_prioritizes_higher_priority_queued_item(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-low",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "priority": 10,
                "priority_source": "policy:fifo",
                "queue_dedup_scope": "base_model",
            },
            workspace="user_default",
        )
        service._append_train_queue_item(
            {
                "job_id": "job-high",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "priority": 100,
                "priority_source": "policy:promotion_bias:auto_promote",
                "queue_dedup_scope": "train_config",
            },
            workspace="user_default",
        )

        snapshot = service.status()["train_queue"]
        self.assertEqual(snapshot["current"]["job_id"], "job-high")
        self.assertEqual(snapshot["max_priority"], 100)
        self.assertEqual(snapshot["current"]["priority_source"], "policy:promotion_bias:auto_promote")
        self.assertEqual(snapshot["current"]["queue_dedup_scope"], "train_config")
        self.assertEqual(snapshot["policy_summary"]["current_priority_source"], "policy:promotion_bias:auto_promote")
        self.assertEqual(snapshot["policy_summary"]["current_dedup_scope"], "train_config")
        self.assertIn("policy:promotion_bias:auto_promote", snapshot["policy_summary"]["priority_sources"])
        self.assertIn("train_config", snapshot["policy_summary"]["dedup_scopes"])
        self.assertEqual(snapshot["review_policy_summary"]["review_mode"], "auto_queue")
        self.assertEqual(snapshot["review_policy_summary"]["queue_entry_mode"], "queued")
        self.assertEqual(snapshot["review_policy_summary"]["next_action"], "process_next_queue_item")

        text = _format_status(service.status(), workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("max priority:", clean)
        self.assertIn("100", clean)
        self.assertIn("queue review mode:", clean)
        self.assertIn("auto_queue", clean)

    def test_queue_prioritizes_earlier_item_when_priority_ties(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-early",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "priority": 50,
                "triggered_at": "2026-03-25T10:00:00+00:00",
            },
            workspace="user_default",
        )
        service._append_train_queue_item(
            {
                "job_id": "job-late",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "priority": 50,
                "triggered_at": "2026-03-25T10:05:00+00:00",
            },
            workspace="user_default",
        )

        snapshot = service.status()["train_queue"]
        self.assertEqual(snapshot["current"]["job_id"], "job-early")
        self.assertEqual(snapshot["max_priority"], 50)

        text = _format_status(service.status(), workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("max priority:", clean)
        self.assertIn("50", clean)
        self.assertIn("current:", clean)
        self.assertIn("job-early", clean)

    def test_status_and_http_expose_train_queue_and_candidate_summary(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {"job_id": "job-1", "state": "queued", "workspace": "user_default", "source": "signal_auto_train"},
            workspace="user_default",
        )
        service._update_train_queue_item("job-1", {"state": "running"}, workspace="user_default")

        fake_rows = [
            {
                "version": "20260325-003",
                "state": "promoted",
                "adapter_dir": str(self.pfe_home / "adapters" / "user_default" / "20260325-003"),
                "artifact_format": "gguf_merged",
                "base_model": "Qwen/Qwen3-4B",
                "num_samples": 12,
            },
            {
                "version": "20260325-002",
                "state": "pending_eval",
                "adapter_dir": str(self.pfe_home / "adapters" / "user_default" / "20260325-002"),
                "artifact_format": "peft_lora",
                "base_model": "Qwen/Qwen3-4B",
                "num_samples": 10,
            },
            {
                "version": "20260325-001",
                "state": "training",
                "adapter_dir": str(self.pfe_home / "adapters" / "user_default" / "20260325-001"),
                "artifact_format": "peft_lora",
                "base_model": "Qwen/Qwen3-4B",
                "num_samples": 8,
            },
            {
                "version": "20260324-999",
                "state": "failed_eval",
                "adapter_dir": str(self.pfe_home / "adapters" / "user_default" / "20260324-999"),
                "artifact_format": "peft_lora",
                "base_model": "Qwen/Qwen3-4B",
                "num_samples": 6,
            },
        ]
        fake_store = _FakeAdapterStore(fake_rows, "20260325-003")
        with patch("pfe_core.pipeline.create_adapter_store", return_value=fake_store), patch(
            "pfe_core.adapter_store.create_adapter_store",
            return_value=fake_store,
        ):
            status = service.status()
            self.assertIn("candidate_summary", status)
            self.assertIn("train_queue", status)
            self.assertIn("operations_overview", status)
            candidate_summary = status["candidate_summary"]
            train_queue = status["train_queue"]
            operations_overview = status["operations_overview"]
            self.assertEqual(candidate_summary["latest_promoted_version"], "20260325-003")
            self.assertEqual(candidate_summary["candidate_version"], "20260325-002")
            self.assertEqual(candidate_summary["candidate_state"], "pending_eval")
            self.assertEqual(candidate_summary["pending_eval_count"], 1)
            self.assertEqual(candidate_summary["training_count"], 1)
            self.assertEqual(candidate_summary["failed_eval_count"], 1)
            self.assertEqual(train_queue["count"], 1)
            self.assertEqual(train_queue["counts"]["running"], 1)
            self.assertEqual(train_queue["current"]["state"], "running")
            self.assertIn("history_summary", train_queue)
            self.assertIn("transition_count", train_queue["history_summary"])
            self.assertTrue(operations_overview["attention_needed"])
            self.assertEqual(operations_overview["attention_reason"], "candidate_ready_for_promotion")
            self.assertIn("candidate=20260325-002:pending_eval", operations_overview["summary_line"])

            status_text = _format_status(status, workspace="user_default")
            clean_status = strip_ansi(status_text)
            self.assertIn("[ OPERATIONS ]", clean_status)
            self.assertIn("candidate_ready_for_promotion", clean_status)

            app = create_app(
                ServiceBundle(
                    inference=InferenceServiceAdapter(service),
                    pipeline=PipelineServiceAdapter(service),
                    security=ServerSecurityConfig(),
                    provider="core",
                    workspace=str(self.pfe_home),
                )
            )

            async def scenario() -> dict[str, object]:
                result = await smoke_test_request(app, path="/pfe/status", method="GET")
                return result["body"]

            body = asyncio.run(scenario())
            self.assertEqual(body["candidate_summary"]["candidate_version"], "20260325-002")
            self.assertEqual(body["train_queue"]["counts"]["running"], 1)
            self.assertEqual(body["operations_overview"]["attention_reason"], "candidate_ready_for_promotion")
            self.assertIn("history_summary", body["train_queue"])
            self.assertEqual(body["metadata"]["pipeline"]["candidate_summary"]["candidate_version"], "20260325-002")
            self.assertEqual(body["metadata"]["pipeline"]["train_queue"]["counts"]["running"], 1)
            self.assertEqual(body["metadata"]["operations_overview"]["attention_reason"], "candidate_ready_for_promotion")

    def test_auto_train_enqueue_result_is_visible_in_queue_status(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = self._seed_signal_ready_service()

        class _FakeTrainResult:
            version = "20260325-010"
            metrics = {
                "state": "pending_eval",
                "num_fresh_samples": 1,
                "num_replay_samples": 0,
            }

        with patch.object(service, "train_result", return_value=_FakeTrainResult()):
            result = service.signal(
                self._signal_payload("evt-queue-auto", "req-queue-auto", "sess-queue-auto")
            )

        auto_train = result["auto_train"]
        self.assertTrue(auto_train["triggered"])
        self.assertIn("queue_job_id", auto_train)

        status = service.status()
        queue = status["train_queue"]
        self.assertEqual(queue["count"], 1)
        self.assertEqual(queue["counts"]["completed"], 1)
        self.assertEqual(queue["last_item"]["job_id"], auto_train["queue_job_id"])
        self.assertEqual(queue["last_item"]["state"], "completed")
        self.assertEqual(queue["last_item"]["source"], "signal_auto_train")
        self.assertEqual(queue["last_item"]["adapter_version"], "20260325-010")
        self.assertIn("candidate_summary", status)


if __name__ == "__main__":
    unittest.main()
