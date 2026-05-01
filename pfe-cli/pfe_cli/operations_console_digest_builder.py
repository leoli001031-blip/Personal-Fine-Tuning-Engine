"""Operations console digest builder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_console_digest_derivation import derive_operations_console_digest
from .operations_console_digest_existing import augment_console_with_timelines
from .operations_formatting_deps import OperationsFormattingDeps


def build_operations_console_digest(
    *,
    operations_console: Mapping[str, Any] | None,
    operations_overview: Mapping[str, Any] | None,
    operations_dashboard: Mapping[str, Any] | None = None,
    operations_alert_policy: Mapping[str, Any] | None = None,
    candidate_summary: Mapping[str, Any] | None,
    candidate_history: Mapping[str, Any] | None,
    candidate_timeline: Mapping[str, Any] | None,
    daemon_timeline: Mapping[str, Any] | None,
    runner_timeline: Mapping[str, Any] | None,
    train_queue: Mapping[str, Any] | None,
    deps: OperationsFormattingDeps,
) -> dict[str, Any] | None:
    console = deps.coerce_mapping(operations_console) or {}
    daemon_timeline = deps.coerce_mapping(daemon_timeline) or deps.coerce_mapping(console.get("daemon_timeline")) or {}
    runner_timeline = deps.coerce_mapping(runner_timeline) or deps.coerce_mapping(console.get("runner_timeline")) or {}
    if console:
        return augment_console_with_timelines(
            console,
            daemon_timeline=daemon_timeline,
            runner_timeline=runner_timeline,
        )

    overview = deps.coerce_mapping(operations_overview) or {}
    dashboard_surface = deps.coerce_mapping(operations_dashboard) or {}
    alert_policy_surface = deps.coerce_mapping(operations_alert_policy) or {}
    candidate_summary = deps.coerce_mapping(candidate_summary) or {}
    candidate_history = deps.coerce_mapping(candidate_history) or {}
    candidate_timeline = deps.coerce_mapping(candidate_timeline) or {}
    train_queue = deps.coerce_mapping(train_queue) or {}

    if not any(
        (
            overview,
            candidate_summary,
            candidate_history,
            candidate_timeline,
            daemon_timeline,
            runner_timeline,
            train_queue,
        )
    ):
        return None

    return derive_operations_console_digest(
        overview=overview,
        dashboard_surface=dashboard_surface,
        alert_policy_surface=alert_policy_surface,
        candidate_summary=candidate_summary,
        candidate_history=candidate_history,
        candidate_timeline=candidate_timeline,
        daemon_timeline=daemon_timeline,
        runner_timeline=runner_timeline,
        train_queue=train_queue,
        deps=deps,
    )


__all__ = ["build_operations_console_digest"]
