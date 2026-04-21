from __future__ import annotations

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
from pfe_core.db.sqlite import save_samples
from pfe_core.pipeline import PipelineService


class AutoTriggerPhase1Tests(unittest.TestCase):
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

    def test_signal_auto_train_stays_blocked_by_default(self) -> None:
        PFEConfig().save(home=self.pfe_home)
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)

        result = service.signal(
            {
                "event_id": "evt-auto-1",
                "request_id": "req-auto-1",
                "session_id": "sess-auto-1",
                "source_event_id": "evt-source-1",
                "source_event_ids": ["evt-source-1", "evt-auto-1"],
                "event_type": "accept",
                "user_input": "我今天有点乱",
                "model_output": "先做一件最小的事。",
                "user_action": {"type": "accept"},
            }
        )

        auto_train = result["auto_train"]
        self.assertFalse(auto_train["enabled"])
        self.assertFalse(auto_train["triggered"])
        self.assertIn("trigger_disabled", auto_train["blocked_reasons"])

    def test_signal_auto_train_triggers_when_enabled_and_threshold_met(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)

        class _FakeTrainResult:
            version = "20260325-999"
            metrics = {
                "state": "pending_eval",
                "num_fresh_samples": 1,
                "num_replay_samples": 0,
            }

        with patch.object(service, "train_result", return_value=_FakeTrainResult()):
            result = service.signal(
                {
                    "event_id": "evt-auto-2",
                    "request_id": "req-auto-2",
                    "session_id": "sess-auto-2",
                    "source_event_id": "evt-source-2",
                    "source_event_ids": ["evt-source-2", "evt-auto-2"],
                    "event_type": "accept",
                    "user_input": "帮我安排今天的事情",
                    "model_output": "先写下三个最重要的任务。",
                    "user_action": {"type": "accept"},
                }
            )

        auto_train = result["auto_train"]
        self.assertTrue(auto_train["enabled"])
        self.assertTrue(auto_train["ready"])
        self.assertTrue(auto_train["triggered"])
        self.assertEqual(auto_train["triggered_version"], "20260325-999")
        self.assertEqual(auto_train["reason"], "triggered")
        self.assertFalse(auto_train["eval_triggered"])
        self.assertFalse(auto_train["promote_triggered"])

    def test_signal_auto_train_can_trigger_eval_without_promote(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.auto_evaluate = True
        config.trainer.trigger.auto_promote = False
        config.trainer.trigger.eval_num_samples = 2
        config.trainer.epochs = 1
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)

        class _FakeTrainResult:
            version = "20260325-998"
            metrics = {
                "state": "pending_eval",
                "num_fresh_samples": 1,
                "num_replay_samples": 0,
            }

        with patch.object(service, "train_result", return_value=_FakeTrainResult()), patch.object(
            service,
            "evaluate",
            return_value='{"recommendation":"deploy","comparison":"improved"}',
        ) as evaluate_mock:
            result = service.signal(
                {
                    "event_id": "evt-auto-3",
                    "request_id": "req-auto-3",
                    "session_id": "sess-auto-3",
                    "source_event_id": "evt-source-3",
                    "source_event_ids": ["evt-source-3", "evt-auto-3"],
                    "event_type": "accept",
                    "user_input": "帮我梳理这件事",
                    "model_output": "先把问题拆成两层。",
                    "user_action": {"type": "accept"},
                }
            )

        auto_train = result["auto_train"]
        evaluate_mock.assert_called_once()
        self.assertTrue(auto_train["triggered"])
        self.assertTrue(auto_train["eval_triggered"])
        self.assertFalse(auto_train["promote_triggered"])
        self.assertEqual(auto_train["eval_recommendation"], "deploy")
        self.assertEqual(auto_train["eval_comparison"], "improved")

    def test_signal_auto_train_can_trigger_promote_after_deploy_eval(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 1
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.auto_evaluate = True
        config.trainer.trigger.auto_promote = True
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)

        class _FakeTrainResult:
            version = "20260325-997"
            metrics = {
                "state": "pending_eval",
                "num_fresh_samples": 1,
                "num_replay_samples": 0,
            }

        class _FakeStore:
            def list_version_records(self, limit: int | None = None) -> list[dict[str, object]]:
                return []

            def promote(self, version: str) -> str:
                return version

        with patch.object(service, "train_result", return_value=_FakeTrainResult()), patch.object(
            service,
            "evaluate",
            return_value='{"recommendation":"deploy","comparison":"improved"}',
        ), patch("pfe_core.pipeline.create_adapter_store", return_value=_FakeStore()):
            result = service.signal(
                {
                    "event_id": "evt-auto-4",
                    "request_id": "req-auto-4",
                    "session_id": "sess-auto-4",
                    "source_event_id": "evt-source-4",
                    "source_event_ids": ["evt-source-4", "evt-auto-4"],
                    "event_type": "accept",
                    "user_input": "帮我整理下一步",
                    "model_output": "先写下一个最小动作。",
                    "user_action": {"type": "accept"},
                }
            )

        auto_train = result["auto_train"]
        self.assertTrue(auto_train["triggered"])
        self.assertTrue(auto_train["eval_triggered"])
        self.assertTrue(auto_train["promote_triggered"])
        self.assertEqual(auto_train["promoted_version"], "20260325-997")
        self.assertEqual(auto_train["triggered_state"], "promoted")

    def test_signal_auto_train_failure_sets_backoff_and_failure_summary(self) -> None:
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
            result = service.signal(
                {
                    "event_id": "evt-auto-5",
                    "request_id": "req-auto-5",
                    "session_id": "sess-auto-5",
                    "source_event_id": "evt-source-5",
                    "source_event_ids": ["evt-source-5", "evt-auto-5"],
                    "event_type": "accept",
                    "user_input": "帮我推进今天的任务",
                    "model_output": "先定一个最小动作。",
                    "user_action": {"type": "accept"},
                }
            )

        auto_train = result["auto_train"]
        self.assertTrue(auto_train["triggered"])
        self.assertEqual(auto_train["state"], "failed")
        self.assertEqual(auto_train["reason"], "train_failed")
        self.assertEqual(auto_train["error_stage"], "train")
        self.assertIn("trainer boom", auto_train["error"])
        self.assertIn("train_failed", auto_train["last_result_summary"])

        status = service.status()
        trigger = status["auto_train_trigger"]
        self.assertFalse(trigger["ready"])
        self.assertIn("failure_backoff_active", trigger["blocked_reasons"])
        self.assertEqual(trigger["consecutive_failures"], 1)
        self.assertFalse(trigger["failure_backoff_elapsed"])
        self.assertIsNotNone(trigger["failure_backoff_remaining_minutes"])
        self.assertIn("train_failed", trigger["last_result_summary"])

    def test_auto_train_trigger_surfaces_manual_review_queue_gate(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 0
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.queue_mode = "deferred"
        config.trainer.trigger.require_queue_confirmation = True
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        service._append_train_queue_item(
            {
                "job_id": "job-review-gate-1",
                "state": "awaiting_confirmation",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "confirmation_required": True,
                "confirmation_reason": "manual_review_required_by_policy",
            },
            workspace="user_default",
        )

        trigger = service.status()["auto_train_trigger"]
        self.assertFalse(trigger["ready"])
        self.assertIn("queue_pending_review", trigger["blocked_reasons"])
        self.assertEqual(trigger["blocked_reasons"][0], "queue_pending_review")
        self.assertEqual(trigger["queue_gate_reason"], "queue_pending_review")
        self.assertEqual(trigger["queue_gate_action"], "review_queue_confirmation")
        self.assertEqual(trigger["queue_review_mode"], "manual_review")
        self.assertEqual(trigger["blocked_primary_reason"], "queue_pending_review")
        self.assertEqual(trigger["blocked_primary_action"], "review_queue_confirmation")
        self.assertEqual(trigger["blocked_primary_category"], "queue")
        self.assertIn("waiting for queue confirmation review", trigger["blocked_summary"])

    def test_auto_train_trigger_surfaces_queued_execution_gate(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 0
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.queue_mode = "deferred"
        config.trainer.trigger.require_queue_confirmation = False
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        service._append_train_queue_item(
            {
                "job_id": "job-review-gate-2",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "priority": 100,
            },
            workspace="user_default",
        )

        trigger = service.status()["auto_train_trigger"]
        self.assertFalse(trigger["ready"])
        self.assertIn("queue_waiting_execution", trigger["blocked_reasons"])
        self.assertEqual(trigger["blocked_reasons"][0], "queue_waiting_execution")
        self.assertEqual(trigger["queue_gate_reason"], "queue_waiting_execution")
        self.assertEqual(trigger["queue_gate_action"], "process_next_queue_item")
        self.assertEqual(trigger["queue_review_mode"], "auto_queue")
        self.assertEqual(trigger["blocked_primary_reason"], "queue_waiting_execution")
        self.assertEqual(trigger["blocked_primary_action"], "process_next_queue_item")
        self.assertEqual(trigger["blocked_primary_category"], "queue")

    def test_auto_train_trigger_prioritizes_backoff_over_sample_shortage(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 50
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.failure_backoff_minutes = 30
        config.save(home=self.pfe_home)

        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        service._persist_auto_trigger_state(
            {
                "workspace": "user_default",
                "last_failure_at": "2026-03-29T12:00:00+00:00",
                "failure_backoff_until": "2999-01-01T00:00:00+00:00",
                "consecutive_failures": 1,
            },
            workspace="user_default",
        )

        trigger = service.status()["auto_train_trigger"]
        self.assertFalse(trigger["ready"])
        self.assertIn("failure_backoff_active", trigger["blocked_reasons"])
        self.assertIn("insufficient_new_signal_samples", trigger["blocked_reasons"])
        self.assertEqual(trigger["blocked_primary_reason"], "failure_backoff_active")
        self.assertEqual(trigger["blocked_primary_action"], "wait_for_failure_backoff")
        self.assertEqual(trigger["blocked_primary_category"], "recovery")

    def test_auto_train_trigger_uses_dpo_preference_pairs_for_dpo_train_type(self) -> None:
        config = PFEConfig()
        config.trainer.train_type = "dpo"
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 2
        config.trainer.trigger.max_interval_days = 0
        config.save(home=self.pfe_home)

        def _sample(
            sample_id: str,
            *,
            sample_type: str,
            dataset_split: str,
            chosen: str,
            rejected: str | None = None,
        ) -> dict[str, object]:
            return {
                "sample_id": sample_id,
                "sample_type": sample_type,
                "instruction": f"instruction-{sample_id}",
                "chosen": chosen,
                "rejected": rejected,
                "score": 0.9,
                "source": "signal",
                "source_event_ids": [sample_id],
                "source_adapter_version": "20260331-001",
                "created_at": "2026-04-01T00:00:00+00:00",
                "used_in_version": None,
                "metadata": {"dataset_split": dataset_split},
            }

        save_samples(
            [
                _sample("sft-1", sample_type="sft", dataset_split="train", chosen="chosen-sft-1"),
                _sample("sft-2", sample_type="sft", dataset_split="train", chosen="chosen-sft-2"),
                _sample("sft-3", sample_type="sft", dataset_split="train", chosen="chosen-sft-3"),
                _sample("sft-4", sample_type="sft", dataset_split="train", chosen="chosen-sft-4"),
                _sample("sft-val", sample_type="sft", dataset_split="val", chosen="chosen-sft-val"),
            ],
            home=self.pfe_home,
        )

        service = PipelineService()
        trigger = service.status()["auto_train_trigger"]
        self.assertFalse(trigger["ready"])
        self.assertEqual(trigger["blocked_primary_reason"], "insufficient_dpo_preference_pairs")
        self.assertEqual(trigger["blocked_primary_action"], "collect_more_dpo_preference_pairs")
        self.assertEqual(trigger["blocked_primary_category"], "data")
        self.assertEqual(trigger["threshold_summary"]["train_type"], "dpo")
        self.assertEqual(trigger["threshold_summary"]["threshold_label"], "dpo_pairs")
        self.assertIn("dpo_pairs=0/2", trigger["threshold_summary"]["summary_line"])
        self.assertIn("DPO preference pairs", trigger["blocked_summary"])
        self.assertGreaterEqual(trigger["eligible_signal_train_samples"], 4)
        self.assertEqual(trigger["eligible_dpo_preference_pairs"], 0)

        save_samples(
            [
                _sample(
                    "dpo-1",
                    sample_type="dpo",
                    dataset_split="train",
                    chosen="chosen-dpo-1",
                    rejected="rejected-dpo-1",
                ),
                _sample(
                    "dpo-2",
                    sample_type="dpo",
                    dataset_split="train",
                    chosen="chosen-dpo-2",
                    rejected="rejected-dpo-2",
                ),
            ],
            home=self.pfe_home,
        )

        ready_trigger = service.status()["auto_train_trigger"]
        self.assertTrue(ready_trigger["ready"])
        self.assertEqual(ready_trigger["eligible_dpo_preference_pairs"], 2)

    def test_auto_train_trigger_weights_preference_reinforced_signal_samples(self) -> None:
        config = PFEConfig()
        config.trainer.trigger.enabled = True
        config.trainer.trigger.min_new_samples = 4
        config.trainer.trigger.max_interval_days = 0
        config.trainer.trigger.preference_reinforced_sample_weight = 2.0
        config.save(home=self.pfe_home)

        def _sample(sample_id: str, *, reinforced: bool = False) -> dict[str, object]:
            metadata: dict[str, object] = {"dataset_split": "train"}
            if reinforced:
                metadata["explicit_response_preference_reinforced"] = True
                metadata["training_signal_category"] = "preference_reinforced"
            return {
                "sample_id": sample_id,
                "sample_type": "sft",
                "instruction": f"instruction-{sample_id}",
                "chosen": f"chosen-{sample_id}",
                "rejected": None,
                "score": 0.9,
                "source": "signal",
                "source_event_ids": [sample_id],
                "source_adapter_version": "20260416-001",
                "created_at": "2026-04-16T00:00:00+00:00",
                "used_in_version": None,
                "metadata": metadata,
            }

        save_samples(
            [
                _sample("sig-1"),
                _sample("sig-2"),
                _sample("sig-pref", reinforced=True),
                {
                    "sample_id": "sig-val",
                    "sample_type": "sft",
                    "instruction": "instruction-sig-val",
                    "chosen": "chosen-sig-val",
                    "rejected": None,
                    "score": 0.9,
                    "source": "signal",
                    "source_event_ids": ["sig-val"],
                    "source_adapter_version": "20260416-001",
                    "created_at": "2026-04-16T00:00:00+00:00",
                    "used_in_version": None,
                    "metadata": {"dataset_split": "val"},
                },
            ],
            home=self.pfe_home,
        )

        service = PipelineService()
        trigger = service.status()["auto_train_trigger"]
        self.assertTrue(trigger["ready"])
        self.assertEqual(trigger["eligible_signal_train_samples"], 3)
        self.assertEqual(trigger["preference_reinforced_signal_train_samples"], 1)
        self.assertEqual(trigger["preference_reinforced_train_samples"], 1)
        self.assertEqual(trigger["preference_reinforced_sample_weight"], 2.0)
        self.assertEqual(trigger["effective_signal_train_samples"], 4.0)
        self.assertEqual(trigger["effective_eligible_train_samples"], 4.0)
        self.assertEqual(trigger["threshold_summary"]["effective_eligible_train_samples"], 4.0)
        self.assertEqual(trigger["threshold_summary"]["preference_reinforced_train_samples"], 1)
        self.assertIn("effective=4/4", trigger["threshold_summary"]["summary_line"])
        self.assertIn("reinforced=1", trigger["threshold_summary"]["summary_line"])


if __name__ == "__main__":
    unittest.main()
