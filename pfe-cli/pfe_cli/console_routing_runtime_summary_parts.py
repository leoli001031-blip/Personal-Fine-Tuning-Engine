"""Part builders for trigger, gate, and runtime console summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_routing_deps import ConsoleRoutingDeps
from .console_routing_summary_helpers import append_mapping_parts


def trigger_summary_parts(payload: Mapping[str, Any], *, deps: ConsoleRoutingDeps) -> list[str]:
    mapping = deps.coerce_mapping(payload) or {}
    trigger = deps.coerce_mapping(mapping.get("auto_train_trigger")) or {}
    parts: list[str] = []
    append_mapping_parts(
        parts,
        trigger,
        (
            "enabled",
            "state",
            "ready",
            "reason",
            "blocked_primary_reason",
            "blocked_primary_action",
            "blocked_primary_category",
            "queue_gate_reason",
            "queue_gate_action",
            "queue_review_mode",
        ),
        deps=deps,
    )
    blocked_summary = trigger.get("blocked_summary")
    if blocked_summary:
        parts.append(f"blocked_summary={deps.format_scalar(blocked_summary)}")
    return parts


def gate_summary_parts(payload: Mapping[str, Any], *, deps: ConsoleRoutingDeps) -> list[str]:
    mapping = deps.coerce_mapping(payload) or {}
    trigger = deps.coerce_mapping(mapping.get("auto_train_trigger")) or {}
    policy = deps.coerce_mapping(trigger.get("policy")) or {}
    threshold = deps.coerce_mapping(trigger.get("threshold_summary")) or {}
    train_queue = deps.coerce_mapping(mapping.get("train_queue")) or {}
    review_policy = deps.coerce_mapping(train_queue.get("review_policy_summary")) or {}

    parts: list[str] = []
    append_mapping_parts(
        parts,
        policy,
        ("queue_entry_mode", "review_mode", "evaluation_mode", "promotion_mode", "stop_stage"),
        deps=deps,
    )
    append_mapping_parts(
        parts,
        threshold,
        (
            "eligible_signal_train_samples",
            "effective_eligible_train_samples",
            "preference_reinforced_train_samples",
            "min_new_samples",
            "holdout_ready",
            "interval_elapsed",
            "cooldown_elapsed",
            "failure_backoff_elapsed",
        ),
        deps=deps,
    )
    append_mapping_parts(
        parts,
        review_policy,
        ("review_mode", "review_required_now", "next_action", "review_reason"),
        deps=deps,
        prefix="queue_",
    )
    return parts


def runtime_summary_parts(payload: Mapping[str, Any], *, deps: ConsoleRoutingDeps) -> list[str]:
    mapping = deps.coerce_mapping(payload) or {}
    console = deps.coerce_mapping(mapping.get("operations_console")) or {}
    operations_overview = deps.coerce_mapping(mapping.get("operations_overview")) or {}
    runtime = deps.coerce_mapping(console.get("runtime_stability_summary")) or deps.coerce_mapping(
        operations_overview.get("runtime_stability_summary")
    ) or {}
    alert_policy = deps.coerce_mapping(mapping.get("operations_alert_policy")) or {}

    parts: list[str] = []
    append_mapping_parts(
        parts,
        runtime,
        (
            "runner_active",
            "runner_lock_state",
            "runner_stop_requested",
            "daemon_health_state",
            "daemon_heartbeat_state",
            "daemon_lease_state",
            "daemon_restart_policy_state",
            "daemon_recovery_action",
        ),
        deps=deps,
    )
    append_mapping_parts(
        parts,
        alert_policy,
        ("required_action", "action_priority", "remediation_mode", "operator_guidance"),
        deps=deps,
    )
    return parts


__all__ = ["gate_summary_parts", "runtime_summary_parts", "trigger_summary_parts"]
