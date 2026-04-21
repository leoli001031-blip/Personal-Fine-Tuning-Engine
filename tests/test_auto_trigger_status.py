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
from pfe_core.config import PFEConfig
from pfe_server.app import build_serve_plan, smoke_test_request
from tests.matrix_test_compat import strip_ansi


class AutoTriggerStatusTests(unittest.TestCase):
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

    def test_cli_status_surfaces_auto_trigger_readiness_and_last_result(self) -> None:
        payload = {
            "home": str(self.pfe_home),
            "signal_count": 2,
            "sample_counts": {"train": 2, "val": 1, "test": 0},
            "auto_train_trigger": {
                "enabled": True,
                "state": "blocked",
                "ready": False,
                "reason": "insufficient_new_signal_samples",
                "blocked_reasons": ["insufficient_new_signal_samples", "cooldown_active"],
                "blocked_primary_reason": "insufficient_new_signal_samples",
                "blocked_primary_action": "collect_more_signal_samples",
                "blocked_primary_category": "data",
                "blocked_summary": "reason=insufficient_new_signal_samples | action=collect_more_signal_samples | category=data | summary=not enough new signal samples are available yet",
                "min_new_samples": 50,
                "max_interval_days": 7,
                "min_trigger_interval_minutes": 30,
                "failure_backoff_minutes": 15,
                "queue_dedup_scope": "train_config",
                "queue_priority_policy": "promotion_bias",
                "queue_process_batch_size": 5,
                "queue_process_until_idle_max": 12,
                "preference_reinforced_sample_weight": 2.0,
                "effective_eligible_train_samples": 3.0,
                "preference_reinforced_train_samples": 1,
                "queue_gate_reason": "queue_pending_review",
                "queue_gate_action": "review_queue_confirmation",
                "queue_review_mode": "manual_review",
                "threshold_summary": {
                    "min_new_samples": 50,
                    "effective_eligible_train_samples": 3.0,
                    "preference_reinforced_train_samples": 1,
                    "eligible_signal_train_samples": 2,
                    "effective_signal_train_samples": 3.0,
                    "preference_reinforced_signal_train_samples": 1,
                    "remaining_signal_samples": 48,
                    "remaining_effective_train_samples": 47.0,
                    "holdout_required": True,
                    "holdout_ready": True,
                    "max_interval_days": 7,
                    "days_since_last_training": 1.25,
                    "interval_elapsed": False,
                    "min_trigger_interval_minutes": 30,
                    "cooldown_elapsed": False,
                    "cooldown_remaining_minutes": 28.5,
                    "failure_backoff_minutes": 15,
                    "failure_backoff_elapsed": False,
                    "failure_backoff_remaining_minutes": 12.0,
                    "preference_reinforced_sample_weight": 2.0,
                    "summary_line": "samples=2/50 | effective=3/50 | reinforced=1 | holdout=ready | interval=1.25/7d | cooldown=28.5m | backoff=12.0m",
                },
                "eligible_signal_train_samples": 2,
                "eligible_signal_sample_ids": ["smp-1", "smp-2"],
                "effective_signal_train_samples": 3.0,
                "preference_reinforced_signal_train_samples": 1,
                "holdout_ready": True,
                "interval_elapsed": False,
                "cooldown_elapsed": False,
                "cooldown_remaining_minutes": 28.5,
                "failure_backoff_elapsed": False,
                "failure_backoff_remaining_minutes": 12.0,
                "consecutive_failures": 1,
                "days_since_last_training": 1.25,
                "recent_training_version": "20260325-001",
                "policy": {
                    "execution_mode": "deferred",
                    "queue_entry_mode": "awaiting_confirmation",
                    "review_mode": "manual_review",
                    "evaluation_mode": "auto_evaluate",
                    "promotion_mode": "auto_promote",
                    "stop_stage": "confirmation",
                    "evaluation_gate_reason": "review_required_before_execution",
                    "evaluation_gate_action": "review_queue_confirmation",
                    "promote_gate_reason": "review_required_before_execution",
                    "promote_gate_action": "review_queue_confirmation",
                },
                "last_result_summary": "triggered=no | state=blocked | reason=insufficient_new_signal_samples",
                "last_result": {
                    "triggered": False,
                    "state": "blocked",
                    "reason": "insufficient_new_signal_samples",
                    "triggered_version": None,
                    "triggered_state": None,
                    "triggered_num_fresh_samples": 0,
                    "triggered_num_replay_samples": 0,
                },
            },
        }

        text = _format_status(payload, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ PFE STATUS // WORKSPACE: user_default ]", clean)
        self.assertIn("[ AUTO TRAIN TRIGGER ]", clean)
        self.assertIn("enabled:", clean)
        self.assertIn("state:", clean)
        self.assertIn("ready:", clean)
        self.assertIn("insufficient_new_signal_samples", clean)
        self.assertIn("blocked reasons:", clean)
        self.assertIn("cooldown_active", clean)
        self.assertIn("min trigger interval minutes:", clean)
        self.assertIn("failure backoff minutes:", clean)
        self.assertIn("queue dedup scope:", clean)
        self.assertIn("train_config", clean)
        self.assertIn("queue priority policy:", clean)
        self.assertIn("promotion_bias", clean)
        self.assertIn("queue process batch size:", clean)
        self.assertIn("queue process until idle max:", clean)
        self.assertIn("queue gate reason:", clean)
        self.assertIn("queue_pending_review", clean)
        self.assertIn("queue gate action:", clean)
        self.assertIn("review_queue_confirmation", clean)
        self.assertIn("preference reinforced sample weight:", clean)
        self.assertIn("effective eligible train samples:", clean)
        self.assertIn("preference reinforced train samples:", clean)
        self.assertIn("queue review mode:", clean)
        self.assertIn("manual_review", clean)
        self.assertIn("blocked primary reason:", clean)
        self.assertIn("blocked primary action:", clean)
        self.assertIn("collect_more_signal_samples", clean)
        self.assertIn("blocked primary category:", clean)
        self.assertIn("data", clean)
        self.assertIn("eligible signals:", clean)
        self.assertIn("holdout ready:", clean)
        self.assertIn("interval elapsed:", clean)
        self.assertIn("cooldown elapsed:", clean)
        self.assertIn("failure backoff elapsed:", clean)
        self.assertIn("consecutive failures:", clean)
        self.assertIn("recent training version:", clean)
        self.assertIn("20260325-001", clean)
        self.assertIn("blocked summary:", clean)
        self.assertIn("triggered", clean)
        self.assertIn("last result:", clean)

    def test_cli_status_surfaces_candidate_compare_style_hit_rate_context(self) -> None:
        payload = {
            "candidate_summary": {
                "candidate_version": "20260416-002",
                "candidate_state": "pending_eval",
                "candidate_can_promote": False,
                "candidate_needs_promotion": True,
                "promotion_gate_status": "blocked",
                "promotion_gate_reason": "compare_review_required",
                "promotion_gate_action": "inspect_compare_evaluation",
                "promotion_compare_comparison": "right_better",
                "promotion_compare_recommendation": "review",
                "promotion_compare_winner": "20260416-002",
                "promotion_compare_overall_delta": 0.14,
                "promotion_compare_style_preference_hit_rate_delta": 0.4,
                "promotion_compare_personalization_summary": "personalization delta +0.35",
                "promotion_compare_quality_summary": "quality delta +0.02",
                "promotion_compare_summary_line": "left=20260415-001 | right=20260416-002 | recommendation=review",
            }
        }

        text = _format_status(payload, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ CANDIDATE SUMMARY ]", clean)
        self.assertIn("candidate needs promotion:", clean)
        self.assertIn("promotion gate action:", clean)
        self.assertIn("inspect_compare_evaluation", clean)
        self.assertIn("promotion compare style preference hit rate delta:", clean)
        self.assertIn("0.4", clean)
        self.assertIn("promotion compare personalization summary:", clean)
        self.assertIn("promotion compare quality summary:", clean)

    def test_queue_policy_config_round_trip_surfaces_explicit_defaults(self) -> None:
        config = PFEConfig()
        self.assertEqual(config.trainer.trigger.queue_dedup_scope, "train_config")
        self.assertEqual(config.trainer.trigger.queue_priority_policy, "source_default")
        self.assertEqual(config.trainer.trigger.preference_reinforced_sample_weight, 1.5)

        config.trainer.trigger.queue_dedup_scope = "base_model"
        config.trainer.trigger.queue_priority_policy = "promotion_bias"
        config.trainer.trigger.preference_reinforced_sample_weight = 2.0
        config.save(home=self.pfe_home)

        loaded = PFEConfig.load(home=self.pfe_home)
        self.assertEqual(loaded.trainer.trigger.queue_dedup_scope, "base_model")
        self.assertEqual(loaded.trainer.trigger.queue_priority_policy, "promotion_bias")
        self.assertEqual(loaded.trainer.trigger.preference_reinforced_sample_weight, 2.0)

    def test_server_status_surfaces_auto_trigger_blocked_state(self) -> None:
        plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
        app = plan.app

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        self.assertIn("auto_train_trigger", body)
        self.assertIn("enabled", body["auto_train_trigger"])
        self.assertFalse(body["auto_train_trigger"]["enabled"])
        self.assertIn("auto_train_trigger", body["metadata"])
        self.assertEqual(body["metadata"]["auto_train_trigger"]["state"], body["auto_train_trigger"]["state"])
        self.assertIn("queue_process_batch_size", body["auto_train_trigger"])
        self.assertIn("queue_process_until_idle_max", body["auto_train_trigger"])
        self.assertIn("queue_dedup_scope", body["auto_train_trigger"])
        self.assertIn("queue_priority_policy", body["auto_train_trigger"])
        self.assertIn("policy", body["auto_train_trigger"])
        self.assertEqual(body["auto_train_trigger"]["policy"]["execution_mode"], "inline")
        self.assertEqual(body["auto_train_trigger"]["policy"]["queue_entry_mode"], "disabled")
        self.assertEqual(body["auto_train_trigger"]["policy"]["stop_stage"], "trigger")


if __name__ == "__main__":
    unittest.main()
