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

from pfe_cli.main import _format_status
from pfe_core.config import PFEConfig
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig
from tests.matrix_test_compat import strip_ansi


class DeferredQueueExecutorSurfaceTests(unittest.TestCase):
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

    def _configure_deferred_trigger(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.queue_mode = "deferred"
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

    def _signal_payload(self, event_id: str) -> dict[str, object]:
        return {
            "event_id": event_id,
            "request_id": f"req-{event_id}",
            "session_id": f"sess-{event_id}",
            "source_event_id": f"{event_id}-source",
            "source_event_ids": [f"{event_id}-source", event_id],
            "event_type": "accept",
            "user_input": "帮我推进今天的任务",
            "model_output": "先定一个最小动作。",
            "user_action": {"type": "accept"},
        }

    def test_process_next_result_surfaces_through_cli_and_http(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-process-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": "20260325-401",
            },
            workspace="user_default",
        )
        service._update_train_queue_item(
            "job-process-1",
            {"state": "running", "updated_at": "2026-03-25T10:00:00+00:00"},
            workspace="user_default",
        )
        service._update_train_queue_item(
            "job-process-1",
            {"state": "completed", "updated_at": "2026-03-25T10:01:00+00:00"},
            workspace="user_default",
        )

        status = service.status()
        self.assertIn("train_queue", status)
        self.assertEqual(status["train_queue"]["count"], 1)
        self.assertEqual(status["train_queue"]["current"], {})
        self.assertEqual(status["train_queue"]["last_item"]["state"], "completed")

        status_text = _format_status(status, workspace="user_default")
        clean = strip_ansi(status_text)
        self.assertIn("[ TRAIN QUEUE ]", clean)
        self.assertIn("count:                   1", clean)
        self.assertIn("last:                    job-process-1 | completed | 20260325-401", clean)

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
        self.assertIn("train_queue", body)
        self.assertEqual(body["train_queue"]["last_item"]["state"], "completed")
        self.assertEqual(body["metadata"]["pipeline"]["train_queue"]["last_item"]["state"], "completed")

    def test_deferred_signal_should_enqueue_without_direct_train(self) -> None:
        self._configure_deferred_trigger()
        service = self._service()
        service.generate(scenario="life-coach", style="温和", num_samples=8)

        class _FakeTrainResult:
            version = "20260325-888"
            metrics = {
                "state": "pending_eval",
                "num_fresh_samples": 1,
                "num_replay_samples": 0,
            }

        ready_trigger = {
            "enabled": True,
            "state": "ready",
            "ready": True,
            "reason": "ready",
            "blocked_reasons": [],
            "train_type": "sft",
            "method": "qlora",
            "epochs": 1,
            "base_model": "Qwen/Qwen3-4B",
            "min_new_samples": 1,
            "max_interval_days": 0,
            "min_trigger_interval_minutes": 0,
            "failure_backoff_minutes": 30,
            "queue_dedup_scope": "train_config",
            "queue_priority_policy": "source_default",
            "auto_evaluate": False,
            "auto_promote": False,
            "eval_num_samples": 3,
            "eligible_signal_train_samples": 1,
            "eligible_signal_sample_ids": ["smp-deferred-1"],
            "holdout_ready": True,
            "days_since_last_training": None,
            "interval_elapsed": True,
            "cooldown_elapsed": True,
            "cooldown_remaining_minutes": None,
            "failure_backoff_elapsed": True,
            "failure_backoff_remaining_minutes": None,
            "last_attempted_at": None,
            "last_completed_at": None,
            "last_success_at": None,
            "last_failure_at": None,
            "consecutive_failures": 0,
            "recent_training_version": None,
        }

        with patch.object(service, "_auto_train_trigger_status", return_value=ready_trigger), patch.object(
            service,
            "train_result",
            return_value=_FakeTrainResult(),
        ) as train_result_mock:
            result = service.signal(self._signal_payload("evt-deferred-1"))

        train_result_mock.assert_not_called()
        auto_train = result["auto_train"]
        self.assertFalse(auto_train["triggered"])
        self.assertEqual(service.status()["train_queue"]["count"], 1)
        self.assertEqual(service.status()["train_queue"]["counts"]["queued"], 1)
        self.assertEqual(service.status()["train_queue"]["current"]["state"], "queued")
        self.assertEqual(service.status()["train_queue"]["current"]["priority_source"], "policy:source_default")
        self.assertEqual(service.status()["train_queue"]["current"]["queue_dedup_scope"], "train_config")
        self.assertEqual(service.status()["train_queue"]["policy_summary"]["current_priority_source"], "policy:source_default")

    def test_frontend_surface_exposes_process_batch_control(self) -> None:
        app = create_app(
            ServiceBundle(
                inference=InferenceServiceAdapter(self._service()),
                pipeline=PipelineServiceAdapter(self._service()),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
            )
        )

        async def scenario() -> str:
            result = await smoke_test_request(app, path="/", method="GET")
            return result["text"]

        text = asyncio.run(scenario())
        clean = strip_ansi(text)
        self.assertIn("批量处理队列", clean)
        self.assertIn("/pfe/auto-train/process-batch?limit=5", clean)
        self.assertIn("处理队列直到空闲", clean)
        self.assertIn("运行 Worker Loop", clean)
        self.assertIn("Train Queue", clean)
        self.assertIn("max-priority", clean)
        self.assertIn("current-dedup", clean)
        self.assertIn("current-priority", clean)
        self.assertIn("/pfe/auto-train/process-until-idle?max_iterations=10", clean)
        self.assertIn("/pfe/auto-train/run-worker-loop?max_cycles=10&idle_rounds=1&poll_interval_seconds=0", clean)
        self.assertIn("批准下一条待确认任务", clean)
        self.assertIn("/pfe/auto-train/approve-next", clean)
        self.assertIn("拒绝下一条待确认任务", clean)
        self.assertIn("/pfe/auto-train/reject-next", clean)

    def test_http_worker_loop_surface_returns_loop_summary(self) -> None:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-worker-http-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "dedup_key": "worker-http-1",
                "priority": 100,
            },
            workspace="user_default",
        )

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
            result = await smoke_test_request(
                app,
                path="/pfe/auto-train/run-worker-loop",
                method="POST",
                query_params={"max_cycles": 3, "idle_rounds": 1, "poll_interval_seconds": 0},
            )
            return result["body"]

        class _FakeTrainResult:
            version = "20260325-991"
            metrics = {"state": "pending_eval", "num_fresh_samples": 1, "num_replay_samples": 0}

        with patch.object(service, "train_result", return_value=_FakeTrainResult()):
            body = asyncio.run(scenario())

        self.assertEqual(body["action"], "run_worker_loop")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["processed_count"], 1)
        self.assertEqual(body["max_cycles"], 3)
        self.assertEqual(body["loop_cycles"], 2)
        self.assertEqual(body["idle_rounds"], 1)
        self.assertEqual(body["stopped_reason"], "idle")
        self.assertIn("history_summary", body["metadata"]["snapshot"]["train_queue"])
        self.assertGreaterEqual(body["metadata"]["snapshot"]["train_queue"]["history_summary"]["transition_count"], 2)
        self.assertIn("history", body["metadata"]["snapshot"]["train_queue"]["last_item"])

    def test_http_confirmation_controls_surface_action_schema(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.queue_mode = "deferred"
        config.trainer.trigger.require_queue_confirmation = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.save(home=self.pfe_home)

        service = self._service()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        service.signal(self._signal_payload("evt-confirm-http"))

        app = create_app(
            ServiceBundle(
                inference=InferenceServiceAdapter(service),
                pipeline=PipelineServiceAdapter(service),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
            )
        )

        async def approve_scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/auto-train/approve-next", method="POST")
            return result["body"]

        async def reject_scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/auto-train/reject-next", method="POST")
            return result["body"]

        approve_body = asyncio.run(approve_scenario())
        self.assertEqual(approve_body["action"], "approve_next")
        self.assertEqual(approve_body["status"], "completed")
        self.assertEqual(approve_body["confirmation_reason"], "manual_review_required_by_policy")

        service._append_train_queue_item(
            {
                "job_id": "job-http-confirmation-2",
                "state": "awaiting_confirmation",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "confirmation_required": True,
                "confirmation_reason": "manual_review_required_by_policy",
                "dedup_key": "manual-http-confirmation-2",
                "priority": 100,
            },
            workspace="user_default",
        )
        reject_body = asyncio.run(reject_scenario())
        self.assertEqual(reject_body["action"], "reject_next")
        self.assertEqual(reject_body["status"], "completed")
        self.assertEqual(reject_body["rejection_reason"], "manual_reject")

    def test_status_surface_includes_process_until_idle_summary(self) -> None:
        service = self._service()
        config = PFEConfig.load(home=self.pfe_home)
        config.trainer.trigger.enabled = True
        config.trainer.trigger.queue_mode = "deferred"
        config.trainer.trigger.queue_process_until_idle_max = 2
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)
        service._append_train_queue_item(
            {
                "job_id": "job-idle-surface-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "dedup_key": "idle-surface-1",
            },
            workspace="user_default",
        )
        service._append_train_queue_item(
            {
                "job_id": "job-idle-surface-2",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "dedup_key": "idle-surface-2",
            },
            workspace="user_default",
        )

        class _FakeTrainResult:
            def __init__(self, version: str) -> None:
                self.version = version
                self.metrics = {"state": "pending_eval", "num_fresh_samples": 1, "num_replay_samples": 0}

        with patch.object(service, "train_result", side_effect=[_FakeTrainResult("20260325-920"), _FakeTrainResult("20260325-921")]):
            snapshot = service.process_train_queue_until_idle()

        text = _format_status(snapshot, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ AUTO TRAIN ACTION ]", clean)
        self.assertIn("action:                  process_until_idle", clean)
        self.assertIn("status:                  completed", clean)
        self.assertIn("processed count:         2", clean)
        self.assertIn("max iterations:          2", clean)

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
        self.assertEqual(body["metadata"]["auto_train_trigger_action"]["action"], "process_until_idle")
        self.assertEqual(body["metadata"]["auto_train_trigger_action"]["processed_count"], 2)
        self.assertEqual(body["metadata"]["auto_train_trigger_action"]["remaining_queued"], 0)


if __name__ == "__main__":
    unittest.main()
