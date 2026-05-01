"""Derive operations console digests from status surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_console_digest_actions import build_derived_next_actions
from .operations_console_digest_sections import (
    build_candidate_section,
    build_daemon_section,
    build_queue_section,
    build_runner_section,
    build_runner_timeline_section,
)
from .operations_console_digest_summary import build_console_digest_summary
from .operations_formatting_deps import OperationsFormattingDeps


def derive_operations_console_digest(
    *,
    overview: Mapping[str, Any],
    dashboard_surface: Mapping[str, Any],
    alert_policy_surface: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    candidate_history: Mapping[str, Any],
    candidate_timeline: Mapping[str, Any],
    daemon_timeline: Mapping[str, Any],
    runner_timeline: Mapping[str, Any],
    train_queue: Mapping[str, Any],
    deps: OperationsFormattingDeps,
) -> dict[str, Any]:
    queue_history = deps.coerce_mapping(train_queue.get("history_summary")) or {}
    queue_review = deps.coerce_mapping(train_queue.get("review_summary")) or {}
    queue_confirm = deps.coerce_mapping(train_queue.get("confirmation_summary")) or {}
    worker = deps.coerce_mapping(train_queue.get("worker_runner")) or {}

    attention_reason = overview.get("attention_reason")
    derived_next_actions = build_derived_next_actions(
        overview=overview,
        candidate_summary=candidate_summary,
        queue_confirm=queue_confirm,
        worker=worker,
    )
    candidate_section = build_candidate_section(
        candidate_summary=candidate_summary,
        candidate_history=candidate_history,
        candidate_timeline=candidate_timeline,
    )
    queue_section = build_queue_section(
        train_queue=train_queue,
        queue_history=queue_history,
        queue_review=queue_review,
        queue_confirm=queue_confirm,
    )
    runner_section = build_runner_section(worker=worker)
    daemon_section = build_daemon_section(daemon_timeline=daemon_timeline)
    runner_timeline_section = build_runner_timeline_section(runner_timeline=runner_timeline)

    summary = build_console_digest_summary(
        overview=overview,
        dashboard_surface=dashboard_surface,
        alert_policy_surface=alert_policy_surface,
        daemon_timeline=daemon_timeline,
        candidate_section=candidate_section,
        queue_section=queue_section,
        runner_section=runner_section,
        daemon_section=daemon_section,
        runner_timeline_section=runner_timeline_section,
        derived_next_actions=derived_next_actions,
        deps=deps,
    )

    return {
        "attention_needed": summary.attention_needed,
        "attention_reason": attention_reason,
        "summary_line": summary.summary_line,
        "inspection_summary_line": summary.inspection_summary_line,
        "next_actions": summary.next_actions,
        "current_focus": summary.current_focus,
        "required_action": summary.required_action,
        "last_recovery_event": summary.last_recovery_event,
        "last_recovery_reason": summary.last_recovery_reason,
        "last_recovery_note": summary.last_recovery_note,
        "candidate": candidate_section,
        "queue": queue_section,
        "runner": runner_section,
        "daemon": daemon_section,
        "runner_timeline": runner_timeline_section,
    }


__all__ = ["derive_operations_console_digest"]
