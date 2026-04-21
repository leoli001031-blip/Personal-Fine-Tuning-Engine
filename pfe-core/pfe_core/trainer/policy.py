"""Auto-train/eval/promote policy evaluation for PFE.

This module provides centralized policy evaluation logic for the
auto-train/eval/promote closed-loop system.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from ..config import (
    ConfirmationPolicyConfig,
    EvalGatePolicyConfig,
    PromoteGatePolicyConfig,
    QueueReviewPolicyConfig,
    TrainTriggerPolicyConfig,
    TrainerTriggerConfig,
)


def evaluate_train_trigger_policy(
    config: TrainTriggerPolicyConfig,
    *,
    eligible_samples: int,
    days_since_last_training: Optional[float],
    last_trigger_at: Optional[datetime],
    last_failure_at: Optional[datetime],
    consecutive_failures: int,
    current_queue_depth: int,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Evaluate training trigger policy and return gate status.

    Returns a dict with:
    - ready: bool - whether training can be triggered
    - blocked_reasons: list[str] - reasons why training is blocked
    - primary_reason: str - the main blocking reason (if any)
    - cooldown_remaining_minutes: Optional[float]
    - backoff_remaining_minutes: Optional[float]
    - policy_summary: dict - human-readable policy state
    """
    if now is None:
        now = datetime.now(timezone.utc)

    blocked_reasons: list[str] = []

    # Check enabled
    if not config.enabled:
        blocked_reasons.append("trigger_disabled")

    # Check sample threshold
    if eligible_samples < config.min_new_samples:
        blocked_reasons.append("insufficient_samples")

    # Check max interval (if interval elapsed, can trigger even with fewer samples)
    interval_elapsed = True
    if days_since_last_training is not None:
        interval_elapsed = days_since_last_training >= config.max_interval_days
    if not interval_elapsed and eligible_samples < config.min_new_samples:
        blocked_reasons.append("max_interval_not_elapsed")

    # Check cooldown (min trigger interval)
    cooldown_elapsed = True
    cooldown_remaining_minutes: Optional[float] = None
    if last_trigger_at is not None:
        elapsed_minutes = (now - last_trigger_at).total_seconds() / 60.0
        cooldown_elapsed = elapsed_minutes >= config.min_trigger_interval_minutes
        if not cooldown_elapsed:
            cooldown_remaining_minutes = round(config.min_trigger_interval_minutes - elapsed_minutes, 2)
            blocked_reasons.append("min_trigger_interval_active")

    # Check failure backoff
    failure_backoff_elapsed = True
    backoff_remaining_minutes: Optional[float] = None
    if last_failure_at is not None:
        # Calculate backoff with multiplier for consecutive failures
        multiplier = max(1.0, config.consecutive_failure_backoff_multiplier ** max(0, consecutive_failures - 1))
        effective_backoff = config.failure_backoff_minutes * multiplier
        elapsed_minutes = (now - last_failure_at).total_seconds() / 60.0
        failure_backoff_elapsed = elapsed_minutes >= effective_backoff
        if not failure_backoff_elapsed:
            backoff_remaining_minutes = round(effective_backoff - elapsed_minutes, 2)
            blocked_reasons.append("failure_backoff_active")

    # Check queue depth
    queue_full = False
    if config.pause_on_queue_full and current_queue_depth >= config.max_queue_depth:
        queue_full = True
        blocked_reasons.append("queue_depth_exceeded")

    # Determine readiness
    ready = len(blocked_reasons) == 0

    # Priority ordering for blocked reasons
    blocker_priority = {
        "trigger_disabled": 0,
        "failure_backoff_active": 1,
        "queue_depth_exceeded": 2,
        "min_trigger_interval_active": 3,
        "insufficient_samples": 4,
        "max_interval_not_elapsed": 5,
    }
    prioritized = sorted(
        blocked_reasons,
        key=lambda r: (blocker_priority.get(r, 99), r),
    )
    primary_reason = prioritized[0] if prioritized else None

    return {
        "ready": ready,
        "blocked_reasons": prioritized,
        "primary_reason": primary_reason,
        "cooldown_elapsed": cooldown_elapsed,
        "cooldown_remaining_minutes": cooldown_remaining_minutes,
        "failure_backoff_elapsed": failure_backoff_elapsed,
        "backoff_remaining_minutes": backoff_remaining_minutes,
        "queue_full": queue_full,
        "consecutive_failures": consecutive_failures,
        "policy_summary": {
            "samples": f"{eligible_samples}/{config.min_new_samples}",
            "interval_elapsed": interval_elapsed,
            "cooldown_elapsed": cooldown_elapsed,
            "backoff_elapsed": failure_backoff_elapsed,
            "queue_ok": not queue_full,
        },
    }


def evaluate_eval_gate_policy(
    config: EvalGatePolicyConfig,
    *,
    holdout_samples: int,
    last_eval_at: Optional[datetime],
    training_completed_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Evaluate evaluation gate policy.

    Returns a dict with:
    - should_eval: bool - whether evaluation should be triggered
    - eval_delay_remaining: Optional[float] - seconds until eval can run
    - blocked_reasons: list[str]
    - policy_summary: dict
    """
    if now is None:
        now = datetime.now(timezone.utc)

    blocked_reasons: list[str] = []

    if not config.auto_trigger:
        blocked_reasons.append("auto_eval_disabled")

    # Check holdout split requirement
    if config.require_holdout_split and holdout_samples < config.min_eval_samples:
        blocked_reasons.append("insufficient_holdout_samples")

    # Check eval frequency
    frequency_elapsed = True
    if last_eval_at is not None:
        hours_since_eval = (now - last_eval_at).total_seconds() / 3600.0
        frequency_elapsed = hours_since_eval >= config.eval_frequency_hours
        if not frequency_elapsed:
            blocked_reasons.append("eval_frequency_limit")

    # Check trigger delay after training
    delay_elapsed = True
    eval_delay_remaining: Optional[float] = None
    if config.trigger_delay_seconds > 0 and training_completed_at is not None:
        seconds_since_training = (now - training_completed_at).total_seconds()
        delay_elapsed = seconds_since_training >= config.trigger_delay_seconds
        if not delay_elapsed:
            eval_delay_remaining = round(config.trigger_delay_seconds - seconds_since_training, 2)
            blocked_reasons.append("eval_delay_pending")

    should_eval = len(blocked_reasons) == 0

    return {
        "should_eval": should_eval,
        "eval_delay_remaining": eval_delay_remaining,
        "blocked_reasons": blocked_reasons,
        "holdout_samples": holdout_samples,
        "min_required": config.min_eval_samples,
        "frequency_elapsed": frequency_elapsed,
        "policy_summary": {
            "auto_trigger": config.auto_trigger,
            "holdout_ready": holdout_samples >= config.min_eval_samples,
            "frequency_elapsed": frequency_elapsed,
            "delay_elapsed": delay_elapsed,
        },
    }


def evaluate_promote_gate_policy(
    config: PromoteGatePolicyConfig,
    *,
    eval_scores: dict[str, float],
    eval_recommendation: str,
    previous_scores: Optional[dict[str, float]],
    last_promote_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Evaluate promotion gate policy.

    Returns a dict with:
    - should_promote: bool
    - blocked_reasons: list[str]
    - quality_passed: bool
    - comparison_passed: bool
    - requires_manual_confirm: bool
    - policy_summary: dict
    """
    if now is None:
        now = datetime.now(timezone.utc)

    blocked_reasons: list[str] = []
    quality_passed = True
    comparison_passed = True

    if not config.auto_promote:
        blocked_reasons.append("auto_promote_disabled")

    # Check quality thresholds
    quality_checks = {
        "overall_quality": (eval_scores.get("overall", 0.0), config.min_quality_score),
        "style_match": (eval_scores.get("style_match", 0.0), config.min_style_match_score),
        "preference_alignment": (eval_scores.get("preference_alignment", 0.0), config.min_preference_alignment_score),
        "quality_preservation": (eval_scores.get("quality_preservation", 0.0), config.min_quality_preservation_score),
    }

    failed_quality = [name for name, (actual, threshold) in quality_checks.items() if actual < threshold]
    if failed_quality:
        quality_passed = False
        blocked_reasons.append(f"quality_threshold_failed:{','.join(failed_quality)}")

    # Check eval recommendation
    if config.require_eval_recommendation_deploy and eval_recommendation != "deploy":
        blocked_reasons.append(f"eval_recommendation_not_deploy:{eval_recommendation}")

    # Check comparison with previous
    regression_detected = False
    if config.compare_with_previous and previous_scores is not None:
        deltas = {
            key: eval_scores.get(key, 0.0) - previous_scores.get(key, 0.0)
            for key in eval_scores
        }
        min_delta = config.min_improvement_delta
        # Require at least no regression on quality preservation
        if deltas.get("quality_preservation", 0.0) < -min_delta:
            comparison_passed = False
            regression_detected = True
            blocked_reasons.append("quality_regression_vs_previous")

    # Check promote frequency
    frequency_elapsed = True
    if last_promote_at is not None:
        hours_since_promote = (now - last_promote_at).total_seconds() / 3600.0
        frequency_elapsed = hours_since_promote >= config.max_promote_frequency_hours
        if not frequency_elapsed:
            blocked_reasons.append("promote_frequency_limit")

    # Determine if manual confirmation is required
    requires_manual_confirm = False
    if config.require_manual_confirm_on_regression and regression_detected:
        requires_manual_confirm = True

    # should_promote is True only when there are no blocking reasons
    # If auto_promote is disabled, it should never auto-promote
    should_promote = len(blocked_reasons) == 0

    return {
        "should_promote": should_promote,
        "blocked_reasons": blocked_reasons,
        "quality_passed": quality_passed,
        "comparison_passed": comparison_passed,
        "regression_detected": regression_detected,
        "requires_manual_confirm": requires_manual_confirm,
        "frequency_elapsed": frequency_elapsed,
        "policy_summary": {
            "quality_checks": {name: f"{actual:.3f}>={threshold:.3f}" for name, (actual, threshold) in quality_checks.items()},
            "quality_passed": quality_passed,
            "comparison_passed": comparison_passed,
            "regression_detected": regression_detected,
            "eval_recommendation": eval_recommendation,
        },
    }


def evaluate_confirmation_policy(
    config: ConfirmationPolicyConfig,
    *,
    is_first_training: bool,
    quality_regression_detected: bool,
    last_trigger_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Evaluate confirmation policy requirements.

    Returns a dict with:
    - requires_confirmation: bool
    - confirmation_reason: Optional[str]
    - auto_approve_eligible: bool
    """
    if now is None:
        now = datetime.now(timezone.utc)

    requires_confirmation = False
    confirmation_reason: Optional[str] = None

    # First training confirmation
    if config.first_training_requires_confirm and is_first_training:
        requires_confirmation = True
        confirmation_reason = "first_training_requires_confirmation"

    # Quality regression confirmation
    elif config.quality_regression_requires_confirm and quality_regression_detected:
        requires_confirmation = True
        confirmation_reason = "quality_regression_requires_confirmation"

    # Rapid trigger confirmation
    elif config.rapid_trigger_requires_confirm and last_trigger_at is not None:
        minutes_since_trigger = (now - last_trigger_at).total_seconds() / 60.0
        if minutes_since_trigger < config.rapid_trigger_threshold_minutes:
            requires_confirmation = True
            confirmation_reason = "rapid_trigger_requires_confirmation"

    # Auto-approve logic
    auto_approve_eligible = False
    if config.queue_confirmation_default_approved and not requires_confirmation:
        auto_approve_eligible = True

    return {
        "requires_confirmation": requires_confirmation,
        "confirmation_reason": confirmation_reason,
        "auto_approve_eligible": auto_approve_eligible,
        "policy_summary": {
            "first_training_check": is_first_training and config.first_training_requires_confirm,
            "regression_check": quality_regression_detected and config.quality_regression_requires_confirm,
            "rapid_trigger_check": last_trigger_at is not None and config.rapid_trigger_requires_confirm,
        },
    }


def evaluate_queue_review_policy(
    config: QueueReviewPolicyConfig,
    *,
    queue_items: list[dict[str, Any]],
    running_jobs: int,
) -> dict[str, Any]:
    """Evaluate queue review policy and return processing recommendations.

    Returns a dict with:
    - can_process: bool
    - items_to_process: int
    - priority_order: list[str] - item IDs in priority order
    - blocked_reasons: list[str]
    """
    blocked_reasons: list[str] = []

    # Check concurrent job limit
    if running_jobs >= config.max_concurrent_jobs:
        blocked_reasons.append("max_concurrent_jobs_reached")

    # Check if queue is empty
    if not queue_items:
        blocked_reasons.append("queue_empty")

    can_process = len(blocked_reasons) == 0

    # Determine items to process
    available_slots = max(0, config.max_concurrent_jobs - running_jobs)
    items_to_process = min(len(queue_items), config.batch_size, available_slots)

    # Priority ordering
    priority_order: list[str] = []
    if queue_items:
        if config.priority_policy == "fifo":
            # Sort by creation time (oldest first)
            sorted_items = sorted(
                queue_items,
                key=lambda x: x.get("created_at", ""),
            )
        elif config.priority_policy == "quality_score":
            # Sort by quality score (highest first)
            sorted_items = sorted(
                queue_items,
                key=lambda x: x.get("quality_score", 0.0),
                reverse=True,
            )
        else:  # hybrid
            # Combine FIFO with quality boost
            def _hybrid_score(item: dict[str, Any]) -> float:
                age_hours = 0.0
                created_at = item.get("created_at")
                if created_at:
                    try:
                        created = datetime.fromisoformat(created_at)
                        age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600.0
                    except Exception:
                        pass
                quality = item.get("quality_score", 0.5)
                # Hybrid: age (older = higher priority) + quality weight
                return age_hours + quality * config.quality_score_weight * 10

            sorted_items = sorted(
                queue_items,
                key=_hybrid_score,
                reverse=True,
            )

        priority_order = [str(item.get("job_id", "")) for item in sorted_items[:items_to_process]]

    return {
        "can_process": can_process,
        "items_to_process": items_to_process,
        "priority_order": priority_order,
        "blocked_reasons": blocked_reasons,
        "policy_summary": {
            "queue_length": len(queue_items),
            "running_jobs": running_jobs,
            "max_concurrent": config.max_concurrent_jobs,
            "batch_size": config.batch_size,
            "priority_policy": config.priority_policy,
        },
    }


def build_policy_summary(
    trigger_config: TrainerTriggerConfig,
    *,
    workspace: Optional[str] = None,
) -> dict[str, Any]:
    """Build a comprehensive policy summary for status/debugging."""
    return {
        "workspace": workspace or "user_default",
        "trigger": {
            "enabled": trigger_config.enabled,
            "min_new_samples": trigger_config.min_new_samples,
            "max_interval_days": trigger_config.max_interval_days,
            "queue_mode": trigger_config.queue_mode,
        },
        "train_trigger_policy": asdict(trigger_config.train_trigger_policy),
        "eval_gate_policy": asdict(trigger_config.eval_gate_policy),
        "promote_gate_policy": asdict(trigger_config.promote_gate_policy),
        "confirmation_policy": asdict(trigger_config.confirmation_policy),
        "queue_review_policy": asdict(trigger_config.queue_review_policy),
    }
