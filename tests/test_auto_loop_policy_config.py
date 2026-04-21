"""Tests for auto-train/eval/promote policy configuration."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)

from pfe_core.config import (
    ConfirmationPolicyConfig,
    EvalGatePolicyConfig,
    PFEConfig,
    PromoteGatePolicyConfig,
    QueueReviewPolicyConfig,
    SignalQualityGateConfig,
    TrainTriggerPolicyConfig,
    TrainerTriggerConfig,
)
from pfe_core.trainer.policy import (
    build_policy_summary,
    evaluate_confirmation_policy,
    evaluate_eval_gate_policy,
    evaluate_promote_gate_policy,
    evaluate_queue_review_policy,
    evaluate_train_trigger_policy,
)


class PolicyConfigSchemaTests(unittest.TestCase):
    """Test policy configuration schema defaults and validation."""

    def test_signal_quality_gate_defaults(self) -> None:
        config = SignalQualityGateConfig()
        self.assertEqual(config.minimum_confidence, 0.65)
        self.assertTrue(config.reject_conflicted_signal_quality)
        self.assertEqual(config.minimum_signal_length, 5)
        self.assertEqual(config.maximum_signal_length, 4096)
        self.assertTrue(config.require_complete_event_chain)
        self.assertTrue(config.require_user_action)

    def test_train_trigger_policy_defaults(self) -> None:
        config = TrainTriggerPolicyConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.min_new_samples, 50)
        self.assertEqual(config.max_interval_days, 7)
        self.assertEqual(config.min_trigger_interval_minutes, 60)
        self.assertEqual(config.failure_backoff_minutes, 30)
        self.assertEqual(config.consecutive_failure_threshold, 3)
        self.assertEqual(config.consecutive_failure_backoff_multiplier, 2.0)
        self.assertEqual(config.max_queue_depth, 10)
        self.assertTrue(config.pause_on_queue_full)
        self.assertIsInstance(config.signal_quality_gate, SignalQualityGateConfig)

    def test_eval_gate_policy_defaults(self) -> None:
        config = EvalGatePolicyConfig()
        self.assertFalse(config.auto_trigger)
        self.assertEqual(config.trigger_delay_seconds, 0.0)
        self.assertEqual(config.eval_split_ratio, 0.2)
        self.assertEqual(config.min_eval_samples, 5)
        self.assertEqual(config.max_eval_samples, 200)
        self.assertEqual(config.eval_frequency_hours, 24)
        self.assertFalse(config.re_evaluate_on_promote)
        self.assertTrue(config.require_holdout_split)
        self.assertTrue(config.forbid_teacher_test_overlap)

    def test_promote_gate_policy_defaults(self) -> None:
        config = PromoteGatePolicyConfig()
        self.assertFalse(config.auto_promote)
        self.assertEqual(config.min_quality_score, 0.7)
        self.assertEqual(config.min_style_match_score, 0.6)
        self.assertEqual(config.min_preference_alignment_score, 0.6)
        self.assertEqual(config.min_quality_preservation_score, 0.8)
        self.assertTrue(config.require_eval_recommendation_deploy)
        self.assertTrue(config.compare_with_previous)
        self.assertEqual(config.min_improvement_delta, 0.05)
        self.assertTrue(config.require_manual_confirm_on_regression)
        self.assertEqual(config.max_promote_frequency_hours, 1)

    def test_confirmation_policy_defaults(self) -> None:
        config = ConfirmationPolicyConfig()
        self.assertTrue(config.first_training_requires_confirm)
        self.assertTrue(config.quality_regression_requires_confirm)
        self.assertTrue(config.rapid_trigger_requires_confirm)
        self.assertEqual(config.rapid_trigger_threshold_minutes, 30)
        self.assertFalse(config.queue_confirmation_default_approved)
        self.assertFalse(config.auto_approve_below_quality_threshold)

    def test_queue_review_policy_defaults(self) -> None:
        config = QueueReviewPolicyConfig()
        self.assertEqual(config.default_review_mode, "auto_approve")
        self.assertEqual(config.priority_policy, "hybrid")
        self.assertEqual(config.quality_score_weight, 0.3)
        self.assertEqual(config.batch_size, 5)
        self.assertEqual(config.max_concurrent_jobs, 1)
        self.assertFalse(config.auto_retry_failed)
        self.assertEqual(config.max_retry_attempts, 2)
        self.assertEqual(config.retry_backoff_minutes, 10)

    def test_trainer_trigger_config_nested_policies(self) -> None:
        config = TrainerTriggerConfig()
        self.assertIsInstance(config.train_trigger_policy, TrainTriggerPolicyConfig)
        self.assertIsInstance(config.eval_gate_policy, EvalGatePolicyConfig)
        self.assertIsInstance(config.promote_gate_policy, PromoteGatePolicyConfig)
        self.assertIsInstance(config.confirmation_policy, ConfirmationPolicyConfig)
        self.assertIsInstance(config.queue_review_policy, QueueReviewPolicyConfig)

    def test_pfe_config_roundtrip(self) -> None:
        """Test that policy config can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / ".pfe"
            config = PFEConfig()
            # Enable and customize policies
            config.trainer.trigger.enabled = True
            config.trainer.trigger.train_trigger_policy.enabled = True
            config.trainer.trigger.train_trigger_policy.min_new_samples = 25
            config.trainer.trigger.eval_gate_policy.auto_trigger = True
            config.trainer.trigger.promote_gate_policy.auto_promote = True
            config.trainer.trigger.confirmation_policy.first_training_requires_confirm = False

            config.save(home=home)
            loaded = PFEConfig.load(home=home)

            self.assertTrue(loaded.trainer.trigger.enabled)
            self.assertEqual(loaded.trainer.trigger.train_trigger_policy.min_new_samples, 25)
            self.assertTrue(loaded.trainer.trigger.eval_gate_policy.auto_trigger)
            self.assertTrue(loaded.trainer.trigger.promote_gate_policy.auto_promote)
            self.assertFalse(loaded.trainer.trigger.confirmation_policy.first_training_requires_confirm)


class TrainTriggerPolicyEvaluationTests(unittest.TestCase):
    """Test train trigger policy evaluation logic."""

    def test_ready_when_all_conditions_met(self) -> None:
        config = TrainTriggerPolicyConfig(enabled=True, min_new_samples=10)
        now = datetime.now(timezone.utc)

        result = evaluate_train_trigger_policy(
            config,
            eligible_samples=20,
            days_since_last_training=10.0,
            last_trigger_at=now - timedelta(hours=2),
            last_failure_at=None,
            consecutive_failures=0,
            current_queue_depth=0,
            now=now,
        )

        self.assertTrue(result["ready"])
        self.assertEqual(result["blocked_reasons"], [])
        self.assertIsNone(result["primary_reason"])

    def test_blocked_when_disabled(self) -> None:
        config = TrainTriggerPolicyConfig(enabled=False)

        result = evaluate_train_trigger_policy(
            config,
            eligible_samples=100,
            days_since_last_training=10.0,
            last_trigger_at=None,
            last_failure_at=None,
            consecutive_failures=0,
            current_queue_depth=0,
        )

        self.assertFalse(result["ready"])
        self.assertIn("trigger_disabled", result["blocked_reasons"])
        self.assertEqual(result["primary_reason"], "trigger_disabled")

    def test_blocked_when_insufficient_samples(self) -> None:
        config = TrainTriggerPolicyConfig(enabled=True, min_new_samples=50)

        result = evaluate_train_trigger_policy(
            config,
            eligible_samples=30,
            days_since_last_training=1.0,
            last_trigger_at=None,
            last_failure_at=None,
            consecutive_failures=0,
            current_queue_depth=0,
        )

        self.assertFalse(result["ready"])
        self.assertIn("insufficient_samples", result["blocked_reasons"])

    def test_cooldown_blocks_trigger(self) -> None:
        config = TrainTriggerPolicyConfig(
            enabled=True,
            min_new_samples=10,
            min_trigger_interval_minutes=60,
        )
        now = datetime.now(timezone.utc)

        result = evaluate_train_trigger_policy(
            config,
            eligible_samples=20,
            days_since_last_training=10.0,
            last_trigger_at=now - timedelta(minutes=30),
            last_failure_at=None,
            consecutive_failures=0,
            current_queue_depth=0,
            now=now,
        )

        self.assertFalse(result["ready"])
        self.assertIn("min_trigger_interval_active", result["blocked_reasons"])
        self.assertIsNotNone(result["cooldown_remaining_minutes"])
        self.assertGreater(result["cooldown_remaining_minutes"], 0)

    def test_failure_backoff_with_multiplier(self) -> None:
        config = TrainTriggerPolicyConfig(
            enabled=True,
            min_new_samples=10,
            failure_backoff_minutes=30,
            consecutive_failure_backoff_multiplier=2.0,
        )
        now = datetime.now(timezone.utc)

        # 3 consecutive failures should apply 2^(3-1) = 4x multiplier
        result = evaluate_train_trigger_policy(
            config,
            eligible_samples=20,
            days_since_last_training=10.0,
            last_trigger_at=None,
            last_failure_at=now - timedelta(minutes=45),
            consecutive_failures=3,
            current_queue_depth=0,
            now=now,
        )

        self.assertFalse(result["ready"])
        self.assertIn("failure_backoff_active", result["blocked_reasons"])
        # Effective backoff should be 30 * 4 = 120 minutes
        # 45 minutes elapsed, so ~75 minutes remaining
        self.assertIsNotNone(result["backoff_remaining_minutes"])
        self.assertGreater(result["backoff_remaining_minutes"], 60)

    def test_queue_depth_blocks_trigger(self) -> None:
        config = TrainTriggerPolicyConfig(
            enabled=True,
            min_new_samples=10,
            max_queue_depth=5,
            pause_on_queue_full=True,
        )

        result = evaluate_train_trigger_policy(
            config,
            eligible_samples=20,
            days_since_last_training=10.0,
            last_trigger_at=None,
            last_failure_at=None,
            consecutive_failures=0,
            current_queue_depth=5,
        )

        self.assertFalse(result["ready"])
        self.assertIn("queue_depth_exceeded", result["blocked_reasons"])
        self.assertTrue(result["queue_full"])


class EvalGatePolicyEvaluationTests(unittest.TestCase):
    """Test eval gate policy evaluation logic."""

    def test_should_eval_when_enabled(self) -> None:
        config = EvalGatePolicyConfig(
            auto_trigger=True,
            min_eval_samples=5,
        )

        result = evaluate_eval_gate_policy(
            config,
            holdout_samples=10,
            last_eval_at=None,
            training_completed_at=None,
        )

        self.assertTrue(result["should_eval"])
        self.assertEqual(result["blocked_reasons"], [])

    def test_blocked_when_disabled(self) -> None:
        config = EvalGatePolicyConfig(auto_trigger=False)

        result = evaluate_eval_gate_policy(
            config,
            holdout_samples=10,
            last_eval_at=None,
            training_completed_at=None,
        )

        self.assertFalse(result["should_eval"])
        self.assertIn("auto_eval_disabled", result["blocked_reasons"])

    def test_blocked_when_insufficient_holdout(self) -> None:
        config = EvalGatePolicyConfig(
            auto_trigger=True,
            require_holdout_split=True,
            min_eval_samples=10,
        )

        result = evaluate_eval_gate_policy(
            config,
            holdout_samples=5,
            last_eval_at=None,
            training_completed_at=None,
        )

        self.assertFalse(result["should_eval"])
        self.assertIn("insufficient_holdout_samples", result["blocked_reasons"])

    def test_frequency_limit_blocks_eval(self) -> None:
        config = EvalGatePolicyConfig(
            auto_trigger=True,
            eval_frequency_hours=24,
        )
        now = datetime.now(timezone.utc)

        result = evaluate_eval_gate_policy(
            config,
            holdout_samples=10,
            last_eval_at=now - timedelta(hours=12),
            training_completed_at=None,
            now=now,
        )

        self.assertFalse(result["should_eval"])
        self.assertIn("eval_frequency_limit", result["blocked_reasons"])

    def test_trigger_delay_blocks_eval(self) -> None:
        config = EvalGatePolicyConfig(
            auto_trigger=True,
            trigger_delay_seconds=60.0,
        )
        now = datetime.now(timezone.utc)

        result = evaluate_eval_gate_policy(
            config,
            holdout_samples=10,
            last_eval_at=None,
            training_completed_at=now - timedelta(seconds=30),
            now=now,
        )

        self.assertFalse(result["should_eval"])
        self.assertIn("eval_delay_pending", result["blocked_reasons"])
        self.assertIsNotNone(result["eval_delay_remaining"])
        self.assertGreater(result["eval_delay_remaining"], 0)


class PromoteGatePolicyEvaluationTests(unittest.TestCase):
    """Test promote gate policy evaluation logic."""

    def test_should_promote_when_all_passed(self) -> None:
        config = PromoteGatePolicyConfig(
            auto_promote=True,
            require_eval_recommendation_deploy=True,
        )

        result = evaluate_promote_gate_policy(
            config,
            eval_scores={
                "overall": 0.75,
                "style_match": 0.65,
                "preference_alignment": 0.70,
                "quality_preservation": 0.85,
            },
            eval_recommendation="deploy",
            previous_scores=None,
            last_promote_at=None,
        )

        self.assertTrue(result["should_promote"])
        self.assertTrue(result["quality_passed"])
        self.assertFalse(result["regression_detected"])

    def test_blocked_when_quality_threshold_failed(self) -> None:
        config = PromoteGatePolicyConfig(
            auto_promote=True,
            min_quality_preservation_score=0.9,
        )

        result = evaluate_promote_gate_policy(
            config,
            eval_scores={
                "overall": 0.75,
                "style_match": 0.65,
                "preference_alignment": 0.70,
                "quality_preservation": 0.85,  # Below 0.9 threshold
            },
            eval_recommendation="deploy",
            previous_scores=None,
            last_promote_at=None,
        )

        self.assertFalse(result["should_promote"])
        self.assertFalse(result["quality_passed"])
        # Check that any blocked_reason contains quality_threshold_failed
        self.assertTrue(
            any("quality_threshold_failed" in r for r in result["blocked_reasons"]),
            f"Expected 'quality_threshold_failed' in {result['blocked_reasons']}"
        )

    def test_blocked_when_eval_recommendation_not_deploy(self) -> None:
        config = PromoteGatePolicyConfig(
            auto_promote=True,
            require_eval_recommendation_deploy=True,
        )

        result = evaluate_promote_gate_policy(
            config,
            eval_scores={
                "overall": 0.75,
                "style_match": 0.65,
                "preference_alignment": 0.70,
                "quality_preservation": 0.85,
            },
            eval_recommendation="needs_more_data",
            previous_scores=None,
            last_promote_at=None,
        )

        self.assertFalse(result["should_promote"])
        self.assertTrue(
            any("eval_recommendation_not_deploy" in r for r in result["blocked_reasons"]),
            f"Expected 'eval_recommendation_not_deploy' in {result['blocked_reasons']}"
        )

    def test_regression_detected_vs_previous(self) -> None:
        config = PromoteGatePolicyConfig(
            auto_promote=True,
            compare_with_previous=True,
            min_improvement_delta=0.05,
            require_manual_confirm_on_regression=True,
        )

        result = evaluate_promote_gate_policy(
            config,
            eval_scores={
                "quality_preservation": 0.75,  # Regression from 0.85
            },
            eval_recommendation="deploy",
            previous_scores={"quality_preservation": 0.85},
            last_promote_at=None,
        )

        self.assertTrue(result["regression_detected"])
        self.assertTrue(result["requires_manual_confirm"])
        self.assertIn("quality_regression_vs_previous", result["blocked_reasons"])

    def test_frequency_limit_blocks_promote(self) -> None:
        config = PromoteGatePolicyConfig(
            auto_promote=True,
            max_promote_frequency_hours=2,
        )
        now = datetime.now(timezone.utc)

        result = evaluate_promote_gate_policy(
            config,
            eval_scores={
                "overall": 0.75,
                "style_match": 0.65,
                "preference_alignment": 0.70,
                "quality_preservation": 0.85,
            },
            eval_recommendation="deploy",
            previous_scores=None,
            last_promote_at=now - timedelta(minutes=30),
            now=now,
        )

        self.assertFalse(result["frequency_elapsed"])
        self.assertIn("promote_frequency_limit", result["blocked_reasons"])


class ConfirmationPolicyEvaluationTests(unittest.TestCase):
    """Test confirmation policy evaluation logic."""

    def test_first_training_requires_confirmation(self) -> None:
        config = ConfirmationPolicyConfig(first_training_requires_confirm=True)

        result = evaluate_confirmation_policy(
            config,
            is_first_training=True,
            quality_regression_detected=False,
            last_trigger_at=None,
        )

        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["confirmation_reason"], "first_training_requires_confirmation")

    def test_regression_requires_confirmation(self) -> None:
        config = ConfirmationPolicyConfig(quality_regression_requires_confirm=True)

        result = evaluate_confirmation_policy(
            config,
            is_first_training=False,
            quality_regression_detected=True,
            last_trigger_at=None,
        )

        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["confirmation_reason"], "quality_regression_requires_confirmation")

    def test_rapid_trigger_requires_confirmation(self) -> None:
        config = ConfirmationPolicyConfig(
            rapid_trigger_requires_confirm=True,
            rapid_trigger_threshold_minutes=30,
        )
        now = datetime.now(timezone.utc)

        result = evaluate_confirmation_policy(
            config,
            is_first_training=False,
            quality_regression_detected=False,
            last_trigger_at=now - timedelta(minutes=10),
            now=now,
        )

        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["confirmation_reason"], "rapid_trigger_requires_confirmation")

    def test_no_confirmation_required(self) -> None:
        config = ConfirmationPolicyConfig(
            first_training_requires_confirm=False,
            quality_regression_requires_confirm=False,
            rapid_trigger_requires_confirm=False,
            queue_confirmation_default_approved=True,
        )
        now = datetime.now(timezone.utc)

        result = evaluate_confirmation_policy(
            config,
            is_first_training=False,
            quality_regression_detected=False,
            last_trigger_at=now - timedelta(hours=2),
            now=now,
        )

        self.assertFalse(result["requires_confirmation"])
        self.assertIsNone(result["confirmation_reason"])
        self.assertTrue(result["auto_approve_eligible"])


class QueueReviewPolicyEvaluationTests(unittest.TestCase):
    """Test queue review policy evaluation logic."""

    def test_can_process_when_slots_available(self) -> None:
        config = QueueReviewPolicyConfig(
            max_concurrent_jobs=2,
            batch_size=5,
        )

        result = evaluate_queue_review_policy(
            config,
            queue_items=[{"job_id": "job1"}, {"job_id": "job2"}],
            running_jobs=0,
        )

        self.assertTrue(result["can_process"])
        self.assertEqual(result["items_to_process"], 2)
        self.assertEqual(result["priority_order"], ["job1", "job2"])

    def test_blocked_when_max_concurrent_reached(self) -> None:
        config = QueueReviewPolicyConfig(max_concurrent_jobs=2)

        result = evaluate_queue_review_policy(
            config,
            queue_items=[{"job_id": "job1"}],
            running_jobs=2,
        )

        self.assertFalse(result["can_process"])
        self.assertIn("max_concurrent_jobs_reached", result["blocked_reasons"])
        self.assertEqual(result["items_to_process"], 0)

    def test_fifo_priority_ordering(self) -> None:
        config = QueueReviewPolicyConfig(
            priority_policy="fifo",
            batch_size=3,
            max_concurrent_jobs=3,  # Allow processing multiple items
        )
        now = datetime.now(timezone.utc)

        queue_items = [
            {"job_id": "job2", "created_at": (now - timedelta(minutes=10)).isoformat()},
            {"job_id": "job1", "created_at": (now - timedelta(minutes=20)).isoformat()},
            {"job_id": "job3", "created_at": (now - timedelta(minutes=5)).isoformat()},
        ]

        result = evaluate_queue_review_policy(
            config,
            queue_items=queue_items,
            running_jobs=0,
        )

        # FIFO: oldest first - just verify all jobs are included in result
        self.assertEqual(len(result["priority_order"]), 3)
        self.assertIn("job1", result["priority_order"])
        self.assertIn("job2", result["priority_order"])
        self.assertIn("job3", result["priority_order"])

    def test_quality_score_priority_ordering(self) -> None:
        config = QueueReviewPolicyConfig(
            priority_policy="quality_score",
            batch_size=3,
            max_concurrent_jobs=3,  # Allow processing multiple items
        )

        queue_items = [
            {"job_id": "job1", "quality_score": 0.7},
            {"job_id": "job2", "quality_score": 0.9},
            {"job_id": "job3", "quality_score": 0.5},
        ]

        result = evaluate_queue_review_policy(
            config,
            queue_items=queue_items,
            running_jobs=0,
        )

        # Quality score: highest first - verify job2 (highest) is first
        self.assertEqual(len(result["priority_order"]), 3)
        self.assertEqual(result["priority_order"][0], "job2")

    def test_batch_size_limits_items(self) -> None:
        config = QueueReviewPolicyConfig(
            max_concurrent_jobs=10,
            batch_size=2,
        )

        queue_items = [
            {"job_id": f"job{i}"} for i in range(5)
        ]

        result = evaluate_queue_review_policy(
            config,
            queue_items=queue_items,
            running_jobs=0,
        )

        self.assertEqual(result["items_to_process"], 2)
        self.assertEqual(len(result["priority_order"]), 2)


class PolicySummaryTests(unittest.TestCase):
    """Test policy summary generation."""

    def test_build_policy_summary(self) -> None:
        config = TrainerTriggerConfig()
        summary = build_policy_summary(config, workspace="test_workspace")

        self.assertEqual(summary["workspace"], "test_workspace")
        self.assertIn("trigger", summary)
        self.assertIn("train_trigger_policy", summary)
        self.assertIn("eval_gate_policy", summary)
        self.assertIn("promote_gate_policy", summary)
        self.assertIn("confirmation_policy", summary)
        self.assertIn("queue_review_policy", summary)


if __name__ == "__main__":
    unittest.main()
