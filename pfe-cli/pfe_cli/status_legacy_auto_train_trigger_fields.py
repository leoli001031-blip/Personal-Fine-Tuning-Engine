"""Shared field groups for legacy auto-train trigger formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TRIGGER_KEYS = (
    "min_new_samples",
    "max_interval_days",
    "min_trigger_interval_minutes",
    "failure_backoff_minutes",
    "queue_mode",
    "queue_dedup_scope",
    "queue_priority_policy",
    "queue_process_batch_size",
    "queue_process_until_idle_max",
    "queue_worker_max_cycles",
    "queue_worker_idle_rounds",
    "queue_worker_poll_seconds",
    "require_queue_confirmation",
    "preference_reinforced_sample_weight",
    "effective_eligible_train_samples",
    "preference_reinforced_train_samples",
    "eligible_signal_train_samples",
    "effective_signal_train_samples",
    "preference_reinforced_signal_train_samples",
    "holdout_ready",
    "interval_elapsed",
    "queue_gate_reason",
    "queue_gate_action",
    "queue_review_mode",
    "blocked_primary_reason",
    "blocked_primary_action",
    "blocked_primary_category",
    "cooldown_elapsed",
    "cooldown_remaining_minutes",
    "failure_backoff_elapsed",
    "failure_backoff_remaining_minutes",
    "days_since_last_training",
    "consecutive_failures",
    "recent_training_version",
)

POLICY_KEYS = (
    "execution_mode",
    "queue_entry_mode",
    "review_mode",
    "evaluation_mode",
    "promotion_mode",
    "stop_stage",
    "evaluation_gate_reason",
    "evaluation_gate_action",
    "promote_gate_reason",
    "promote_gate_action",
    "promotion_requirement",
)

THRESHOLD_KEYS = (
    "min_new_samples",
    "effective_eligible_train_samples",
    "preference_reinforced_train_samples",
    "eligible_signal_train_samples",
    "effective_signal_train_samples",
    "preference_reinforced_signal_train_samples",
    "remaining_signal_samples",
    "remaining_effective_train_samples",
    "holdout_required",
    "holdout_ready",
    "max_interval_days",
    "days_since_last_training",
    "interval_elapsed",
    "min_trigger_interval_minutes",
    "cooldown_elapsed",
    "cooldown_remaining_minutes",
    "failure_backoff_minutes",
    "failure_backoff_elapsed",
    "failure_backoff_remaining_minutes",
    "preference_reinforced_sample_weight",
)

LAST_RESULT_KEYS = (
    "triggered",
    "state",
    "reason",
    "error_stage",
    "triggered_version",
    "triggered_state",
    "triggered_num_fresh_samples",
    "triggered_num_replay_samples",
    "eval_recommendation",
    "eval_comparison",
    "promoted_version",
)


def append_scalar_parts(
    parts: list[str],
    mapping: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    deps: Any,
) -> None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")


__all__ = [
    "LAST_RESULT_KEYS",
    "POLICY_KEYS",
    "THRESHOLD_KEYS",
    "TRIGGER_KEYS",
    "append_scalar_parts",
]
