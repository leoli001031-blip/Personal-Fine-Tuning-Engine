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


class DeferredQueueExecutorTests(unittest.TestCase):
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

    def _seed_service(self) -> PipelineService:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.queue_mode = "deferred"
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        return service

    @staticmethod
    def _signal_payload() -> dict[str, object]:
        return {
            "event_id": "evt-deferred-1",
            "request_id": "req-deferred-1",
            "session_id": "sess-deferred-1",
            "source_event_id": "evt-source-deferred-1",
            "source_event_ids": ["evt-source-deferred-1", "evt-deferred-1"],
            "event_type": "accept",
            "user_input": "帮我安排今天",
            "model_output": "先写下一件最小动作。",
            "user_action": {"type": "accept"},
        }

    def test_signal_in_deferred_mode_only_enqueues(self) -> None:
        service = self._seed_service()

        with patch.object(service, "train_result") as train_result_mock:
            result = service.signal(self._signal_payload())

        auto_train = result["auto_train"]
        train_result_mock.assert_not_called()
        self.assertFalse(auto_train["triggered"])
        self.assertTrue(auto_train["enqueued"])
        self.assertEqual(auto_train["state"], "queued")
        self.assertEqual(auto_train["reason"], "enqueued")
        self.assertEqual(auto_train["queue_dedup_scope"], "train_config")
        self.assertEqual(auto_train["queue_priority_policy"], "source_default")
        self.assertIn("queue_job_id", auto_train)

        status = service.status()
        self.assertEqual(status["train_queue"]["counts"]["queued"], 1)
        self.assertEqual(status["train_queue"]["current"]["state"], "queued")
        self.assertEqual(status["train_queue"]["current"]["priority_source"], "policy:source_default")
        self.assertEqual(status["train_queue"]["current"]["queue_dedup_scope"], "train_config")

        text = _format_status(status, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ AUTO TRAIN TRIGGER ]", clean)
        self.assertIn("queue mode:              deferred", clean)
        self.assertIn("queue dedup scope:       train_config", clean)
        self.assertIn("queue priority policy:   source_default", clean)
        self.assertIn("[ TRAIN QUEUE ]", clean)
        self.assertIn("count:                   1", clean)
        self.assertIn("states:                  queued:1", clean)

    def test_process_next_executes_queued_item(self) -> None:
        service = self._seed_service()
        service.signal(self._signal_payload())

        class _FakeTrainResult:
            version = "20260325-888"
            metrics = {"state": "pending_eval", "num_fresh_samples": 1, "num_replay_samples": 0}

        with patch.object(service, "train_result", return_value=_FakeTrainResult()):
            snapshot = service.process_next_train_queue()

        action = snapshot["auto_train_trigger_action"]
        queue = snapshot["train_queue"]
        self.assertEqual(action["action"], "process_next")
        self.assertEqual(action["status"], "triggered")
        self.assertTrue(action["triggered"])
        self.assertEqual(action["triggered_version"], "20260325-888")
        self.assertEqual(queue["counts"]["completed"], 1)
        self.assertEqual(queue["last_item"]["adapter_version"], "20260325-888")

    def test_process_next_http_surface_returns_action_schema(self) -> None:
        service = self._seed_service()
        service.signal(self._signal_payload())
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
            result = await smoke_test_request(app, path="/pfe/auto-train/process-next", method="POST")
            return result["body"]

        class _FakeTrainResult:
            version = "20260325-889"
            metrics = {"state": "pending_eval", "num_fresh_samples": 1, "num_replay_samples": 0}

        with patch.object(service, "train_result", return_value=_FakeTrainResult()) as train_result_mock:
            body = asyncio.run(scenario())

        train_result_mock.assert_called_once()
        self.assertEqual(body["action"], "process_next")
        self.assertEqual(body["status"], "triggered")
        self.assertEqual(body["triggered_version"], "20260325-889")

    def test_process_batch_executes_multiple_queued_items(self) -> None:
        service = self._seed_service()
        service._append_train_queue_item(
            {
                "job_id": "job-batch-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
            },
            workspace="user_default",
        )
        service._append_train_queue_item(
            {
                "job_id": "job-batch-2",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
            },
            workspace="user_default",
        )

        class _FakeTrainResult:
            def __init__(self, version: str) -> None:
                self.version = version
                self.metrics = {"state": "pending_eval", "num_fresh_samples": 1, "num_replay_samples": 0}

        with patch.object(service, "train_result", side_effect=[_FakeTrainResult("20260325-901"), _FakeTrainResult("20260325-902")]):
            snapshot = service.process_train_queue_batch(limit=2)

        action = snapshot["auto_train_trigger_action"]
        queue = snapshot["train_queue"]
        self.assertEqual(action["action"], "process_batch")
        self.assertEqual(action["status"], "completed")
        self.assertEqual(action["processed_count"], 2)
        self.assertEqual(action["completed_count"], 2)
        self.assertEqual(action["failed_count"], 0)
        self.assertEqual(queue["counts"]["completed"], 2)
        self.assertEqual(queue["count"], 2)

    def test_process_batch_http_surface_returns_batch_action_schema(self) -> None:
        service = self._seed_service()
        service._append_train_queue_item(
            {
                "job_id": "job-http-batch-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
            },
            workspace="user_default",
        )
        service._append_train_queue_item(
            {
                "job_id": "job-http-batch-2",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
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
            result = await smoke_test_request(app, path="/pfe/auto-train/process-batch", method="POST", query_params={"limit": 2})
            return result["body"]

        class _FakeTrainResult:
            def __init__(self, version: str) -> None:
                self.version = version
                self.metrics = {"state": "pending_eval", "num_fresh_samples": 1, "num_replay_samples": 0}

        with patch.object(service, "train_result", side_effect=[_FakeTrainResult("20260325-903"), _FakeTrainResult("20260325-904")]):
            body = asyncio.run(scenario())

        self.assertEqual(body["action"], "process_batch")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["processed_count"], 2)
        self.assertEqual(body["completed_count"], 2)
        self.assertEqual(body["failed_count"], 0)

    def test_process_until_idle_drains_queue(self) -> None:
        service = self._seed_service()
        for index in range(3):
            service._append_train_queue_item(
                {
                    "job_id": f"job-until-idle-{index}",
                    "state": "queued",
                    "workspace": "user_default",
                    "source": "signal_auto_train",
                    "requested_base_model": "Qwen/Qwen3-4B",
                    "requested_method": "qlora",
                    "requested_train_type": "sft",
                },
                workspace="user_default",
            )

        class _FakeTrainResult:
            def __init__(self, version: str) -> None:
                self.version = version
                self.metrics = {"state": "pending_eval", "num_fresh_samples": 1, "num_replay_samples": 0}

        with patch.object(
            service,
            "train_result",
            side_effect=[_FakeTrainResult("20260325-911"), _FakeTrainResult("20260325-912"), _FakeTrainResult("20260325-913")],
        ):
            snapshot = service.process_train_queue_until_idle(max_iterations=10)

        action = snapshot["auto_train_trigger_action"]
        queue = snapshot["train_queue"]
        self.assertEqual(action["action"], "process_until_idle")
        self.assertEqual(action["status"], "completed")
        self.assertEqual(action["processed_count"], 3)
        self.assertTrue(action["drained"])
        self.assertEqual(action["remaining_queued"], 0)
        self.assertEqual(queue["counts"]["completed"], 3)

    def test_run_worker_loop_processes_queue_until_idle(self) -> None:
        service = self._seed_service()
        service._append_train_queue_item(
            {
                "job_id": "job-worker-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "dedup_key": "worker-1",
                "priority": 100,
            },
            workspace="user_default",
        )
        service._append_train_queue_item(
            {
                "job_id": "job-worker-2",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "dedup_key": "worker-2",
                "priority": 90,
            },
            workspace="user_default",
        )

        class _FakeTrainResult:
            def __init__(self, version: str) -> None:
                self.version = version
                self.metrics = {"state": "pending_eval", "num_fresh_samples": 1, "num_replay_samples": 0}

        with patch.object(service, "train_result", side_effect=[_FakeTrainResult("20260325-921"), _FakeTrainResult("20260325-922")]):
            snapshot = service.run_train_queue_worker_loop(max_cycles=5, idle_rounds=1, poll_interval_seconds=0.0)

        action = snapshot["auto_train_trigger_action"]
        queue = snapshot["train_queue"]
        self.assertEqual(action["action"], "run_worker_loop")
        self.assertEqual(action["status"], "completed")
        self.assertEqual(action["processed_count"], 2)
        self.assertEqual(action["loop_cycles"], 3)
        self.assertEqual(action["idle_rounds"], 1)
        self.assertEqual(action["stopped_reason"], "idle")
        self.assertTrue(action["drained"])
        self.assertEqual(queue["counts"]["completed"], 2)

    def test_signal_dedups_when_matching_queued_job_exists(self) -> None:
        service = self._seed_service()
        service._append_train_queue_item(
            {
                "job_id": "job-dedup-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "dedup_key": "train_config|Qwen/Qwen3-4B|qlora|sft|eval=False|promote=False",
                "priority": 100,
            },
            workspace="user_default",
        )

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
            "queue_mode": "deferred",
            "queue_process_batch_size": 5,
            "queue_process_until_idle_max": 10,
            "auto_evaluate": False,
            "auto_promote": False,
            "eval_num_samples": 3,
            "eligible_signal_train_samples": 1,
            "eligible_signal_sample_ids": ["smp-dedup-1"],
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

        with patch.object(service, "_auto_train_trigger_status", return_value=ready_trigger), patch.object(service, "train_result") as train_result_mock:
            result = service.signal(self._signal_payload())

        train_result_mock.assert_not_called()
        auto_train = result["auto_train"]
        self.assertFalse(auto_train["enqueued"])
        self.assertEqual(auto_train["reason"], "queue_duplicate")
        self.assertEqual(auto_train["queue_job_id"], "job-dedup-1")
        self.assertEqual(auto_train["queue_dedup_scope"], "train_config")
        self.assertEqual(auto_train["queue_priority_policy"], "source_default")
        self.assertEqual(service.status()["train_queue"]["count"], 1)

    def test_signal_dedup_scope_ignores_completed_items_but_keeps_live_scope(self) -> None:
        service = self._seed_service()
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
            "queue_mode": "deferred",
            "queue_dedup_scope": "train_config",
            "queue_priority_policy": "source_default",
            "queue_process_batch_size": 5,
            "queue_process_until_idle_max": 10,
            "auto_evaluate": False,
            "auto_promote": False,
            "eval_num_samples": 3,
            "eligible_signal_train_samples": 1,
            "eligible_signal_sample_ids": ["smp-dedup-completed-1"],
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
            "last_result_summary": "idle",
            "last_result": {},
        }
        dedup_key = service._auto_train_queue_dedup_key(ready_trigger)
        service._append_train_queue_item(
            {
                "job_id": "job-dedup-completed-1",
                "state": "completed",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "dedup_key": dedup_key,
                "priority": 100,
            },
            workspace="user_default",
        )

        with patch.object(service, "_auto_train_trigger_status", return_value=ready_trigger), patch.object(
            service,
            "train_result",
        ) as train_result_mock:
            result = service.signal(self._signal_payload())

        train_result_mock.assert_not_called()
        auto_train = result["auto_train"]
        self.assertTrue(auto_train["enqueued"])
        self.assertEqual(auto_train["reason"], "enqueued")
        self.assertEqual(service.status()["train_queue"]["count"], 2)
        self.assertEqual(service.status()["train_queue"]["counts"]["completed"], 1)
        self.assertEqual(service.status()["train_queue"]["counts"]["queued"], 1)

    def test_signal_requires_manual_confirmation_when_enabled(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.queue_mode = "deferred"
        config.trainer.trigger.require_queue_confirmation = True
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)

        with patch.object(service, "train_result") as train_result_mock:
            result = service.signal(self._signal_payload())

        train_result_mock.assert_not_called()
        auto_train = result["auto_train"]
        self.assertEqual(auto_train["state"], "awaiting_confirmation")
        self.assertEqual(auto_train["reason"], "awaiting_confirmation")
        queue = service.status()["train_queue"]
        self.assertEqual(queue["counts"]["awaiting_confirmation"], 1)
        self.assertEqual(queue["last_item"]["confirmation_required"], True)
        self.assertEqual(queue["last_item"]["confirmation_reason"], "manual_review_required_by_policy")
        self.assertEqual(queue["last_item"]["history"][-1]["event"], "enqueued")
        self.assertEqual(queue["confirmation_summary"]["awaiting_confirmation_count"], 1)
        self.assertEqual(queue["confirmation_summary"]["next_confirmation_reason"], "manual_review_required_by_policy")
        self.assertEqual(queue["review_policy_summary"]["review_mode"], "manual_review")
        self.assertEqual(queue["review_policy_summary"]["queue_entry_mode"], "awaiting_confirmation")
        self.assertTrue(queue["review_policy_summary"]["review_required_by_policy"])
        self.assertTrue(queue["review_policy_summary"]["review_required_now"])
        self.assertEqual(queue["review_policy_summary"]["next_action"], "review_queue_confirmation")
        self.assertEqual(queue["review_policy_summary"]["review_reason"], "manual_review_required_by_policy")

    def test_approve_next_moves_item_to_queued_and_reject_next_marks_rejected(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.queue_mode = "deferred"
        config.trainer.trigger.require_queue_confirmation = True
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        service.signal(self._signal_payload())

        approved = service.approve_next_train_queue()
        self.assertEqual(approved["auto_train_trigger_action"]["action"], "approve_next")
        self.assertEqual(approved["auto_train_trigger_action"]["status"], "completed")
        self.assertEqual(approved["auto_train_trigger_action"]["confirmation_reason"], "manual_review_required_by_policy")
        self.assertEqual(approved["auto_train_trigger_action"]["approval_reason"], "manual_approve_next")
        self.assertEqual(approved["train_queue"]["counts"]["queued"], 1)
        self.assertEqual(approved["train_queue"]["confirmation_summary"]["awaiting_confirmation_count"], 0)
        self.assertEqual(approved["train_queue"]["last_transition"]["event"], "approved")
        self.assertEqual(approved["train_queue"]["history_summary"]["last_reason"], "confirmation_approved")

        service._append_train_queue_item(
            {
                "job_id": "job-awaiting-confirmation-2",
                "state": "awaiting_confirmation",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "confirmation_required": True,
                "confirmation_reason": "manual_review_required_by_policy",
                "dedup_key": "manual-confirmation-2",
                "priority": 100,
            },
            workspace="user_default",
        )
        rejected = service.reject_next_train_queue()
        self.assertEqual(rejected["auto_train_trigger_action"]["action"], "reject_next")
        self.assertEqual(rejected["auto_train_trigger_action"]["status"], "completed")
        self.assertEqual(rejected["auto_train_trigger_action"]["confirmation_reason"], "manual_review_required_by_policy")
        self.assertEqual(rejected["auto_train_trigger_action"]["rejection_reason"], "manual_reject")
        self.assertEqual(rejected["train_queue"]["counts"]["rejected"], 1)
        self.assertEqual(rejected["train_queue"]["last_transition"]["event"], "rejected")
        self.assertEqual(rejected["train_queue"]["history_summary"]["last_reason"], "confirmation_rejected")

    def test_process_until_idle_drains_queued_items_and_honors_configured_limit(self) -> None:
        service = self._seed_service()
        config = PFEConfig.load(home=self.pfe_home)
        config.trainer.trigger.queue_process_until_idle_max = 1
        config.save(home=self.pfe_home)
        service._append_train_queue_item(
            {
                "job_id": "job-idle-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "dedup_key": "idle-1",
            },
            workspace="user_default",
        )
        service._append_train_queue_item(
            {
                "job_id": "job-idle-2",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "requested_base_model": "Qwen/Qwen3-4B",
                "requested_method": "qlora",
                "requested_train_type": "sft",
                "dedup_key": "idle-2",
            },
            workspace="user_default",
        )

        class _FakeTrainResult:
            def __init__(self, version: str) -> None:
                self.version = version
                self.metrics = {"state": "pending_eval", "num_fresh_samples": 1, "num_replay_samples": 0}

        with patch.object(service, "train_result", side_effect=[_FakeTrainResult("20260325-911")]):
            snapshot = service.process_train_queue_until_idle()

        action = snapshot["auto_train_trigger_action"]
        queue = snapshot["train_queue"]
        self.assertEqual(action["action"], "process_until_idle")
        self.assertEqual(action["status"], "completed")
        self.assertEqual(action["processed_count"], 1)
        self.assertEqual(action["max_iterations"], 1)
        self.assertEqual(queue["counts"]["completed"], 1)
        self.assertEqual(queue["counts"]["queued"], 1)
        self.assertEqual(queue["count"], 2)

    def test_queue_dedup_key_finds_existing_work_item(self) -> None:
        service = self._seed_service()
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
            "queue_mode": "deferred",
            "queue_dedup_scope": "train_config",
            "queue_priority_policy": "source_default",
            "queue_process_batch_size": 5,
            "queue_process_until_idle_max": 10,
            "auto_evaluate": False,
            "auto_promote": False,
            "eval_num_samples": 3,
            "eligible_signal_train_samples": 1,
            "eligible_signal_sample_ids": ["smp-dedup-1"],
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
            "last_result_summary": "idle",
            "last_result": {},
        }
        dedup_key = service._auto_train_queue_dedup_key(ready_trigger)
        service._append_train_queue_item(
            {
                "job_id": "job-dedup-1",
                "state": "completed",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "dedup_key": dedup_key,
            },
            workspace="user_default",
        )

        dedup_key = service._auto_train_queue_dedup_key(ready_trigger)
        existing = service._find_train_queue_item_by_dedup_key(dedup_key)
        self.assertIsNone(existing)
        existing = service._find_train_queue_item_by_dedup_key(dedup_key, states=("completed",))
        self.assertIsNotNone(existing)
        self.assertEqual(existing["job_id"], "job-dedup-1")
        self.assertEqual(existing["dedup_key"], dedup_key)
        self.assertEqual(service.status()["train_queue"]["count"], 1)

    def test_queue_dedup_scope_and_priority_policy_are_explicit(self) -> None:
        service = self._seed_service()
        trigger_status = {
            "base_model": "Qwen/Qwen3-4B",
            "method": "qlora",
            "train_type": "sft",
            "auto_evaluate": True,
            "auto_promote": True,
        }

        self.assertEqual(
            service._auto_train_queue_dedup_key(trigger_status, dedup_scope="workspace", workspace="alpha"),
            "workspace:alpha",
        )
        self.assertEqual(
            service._auto_train_queue_dedup_key(trigger_status, dedup_scope="base_model", workspace="alpha"),
            "base_model:Qwen/Qwen3-4B",
        )
        self.assertEqual(
            service._auto_train_queue_dedup_key(trigger_status, dedup_scope="train_config", workspace="alpha"),
            "train_config|Qwen/Qwen3-4B|qlora|sft|eval=True|promote=True",
        )

        fifo_priority = service._queue_priority(trigger_status=trigger_status, source="signal_auto_train", policy="fifo")
        self.assertEqual(fifo_priority, (0, "policy:fifo"))
        source_priority = service._queue_priority(trigger_status=trigger_status, source="signal_auto_train", policy="source_default")
        self.assertEqual(source_priority, (100, "policy:source_default"))
        biased_priority = service._queue_priority(trigger_status=trigger_status, source="signal_auto_train", policy="promotion_bias")
        self.assertEqual(biased_priority, (120, "policy:promotion_bias:auto_promote"))


if __name__ == "__main__":
    unittest.main()
