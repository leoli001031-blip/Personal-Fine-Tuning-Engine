"""Payload extraction for legacy status formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .status_legacy_deps import StatusLegacyFormattingDeps


@dataclass(frozen=True)
class LegacyStatusSections:
    latest_adapter_version: Any
    latest_adapter_state: Any
    latest_adapter_map: dict[str, Any] | None
    recent_adapter_version: Any
    recent_adapter_state: Any
    recent_adapter_map: dict[str, Any] | None
    lifecycle: dict[str, Any] | None
    candidate_summary: dict[str, Any] | None
    compare_evaluation: dict[str, Any] | None
    candidate_history: dict[str, Any] | None
    candidate_timeline: dict[str, Any] | None
    operations_console: dict[str, Any] | None
    daemon_timeline: dict[str, Any] | None
    runner_timeline: dict[str, Any] | None
    train_queue: dict[str, Any] | None
    operations_overview: dict[str, Any] | None
    operations_alerts: list[dict[str, Any]]
    operations_health: dict[str, Any] | None
    operations_recovery: dict[str, Any] | None
    operations_next_actions: list[str]
    operations_dashboard: dict[str, Any] | None
    operations_alert_policy: dict[str, Any] | None
    operations_event_stream: dict[str, Any] | None
    operations_timeline: dict[str, Any] | None


def _adapter_fields(
    mapping: dict[str, Any],
    *,
    version_key: str,
    adapter_key: str,
    deps: StatusLegacyFormattingDeps,
) -> tuple[Any, Any, dict[str, Any] | None]:
    adapter_version = mapping.pop(version_key, None)
    adapter = mapping.pop(adapter_key, None)
    adapter_map = deps.coerce_mapping(adapter)
    if adapter_version is None and adapter_map is not None:
        adapter_version = adapter_map.get("version")
    adapter_state = adapter_map.get("state") if adapter_map is not None else None
    return adapter_version, adapter_state, adapter_map


def extract_legacy_status_sections(
    mapping: dict[str, Any],
    *,
    deps: StatusLegacyFormattingDeps,
) -> LegacyStatusSections:
    latest_adapter_version, latest_adapter_state, latest_adapter_map = _adapter_fields(
        mapping,
        version_key="latest_adapter_version",
        adapter_key="latest_adapter",
        deps=deps,
    )
    recent_adapter_version, recent_adapter_state, recent_adapter_map = _adapter_fields(
        mapping,
        version_key="recent_adapter_version",
        adapter_key="recent_adapter",
        deps=deps,
    )

    operations_console = deps.coerce_mapping(mapping.pop("operations_console", None))
    daemon_timeline = deps.coerce_mapping(mapping.pop("daemon_timeline", None))
    if daemon_timeline is None and operations_console is not None:
        daemon_timeline = deps.coerce_mapping(operations_console.get("daemon_timeline"))
    runner_timeline = deps.coerce_mapping(mapping.pop("runner_timeline", None))
    if runner_timeline is None and operations_console is not None:
        runner_timeline = deps.coerce_mapping(operations_console.get("runner_timeline"))
    operations_event_stream = deps.coerce_mapping(mapping.pop("operations_event_stream", None))
    operations_timeline = deps.coerce_mapping(mapping.pop("operations_timeline", None))
    if operations_timeline is None and operations_console is not None:
        operations_timeline = deps.coerce_mapping(operations_console.get("timelines"))
    if operations_event_stream is None and operations_console is not None:
        operations_event_stream = deps.coerce_mapping(operations_console.get("event_stream"))

    return LegacyStatusSections(
        latest_adapter_version=latest_adapter_version,
        latest_adapter_state=latest_adapter_state,
        latest_adapter_map=latest_adapter_map,
        recent_adapter_version=recent_adapter_version,
        recent_adapter_state=recent_adapter_state,
        recent_adapter_map=recent_adapter_map,
        lifecycle=deps.coerce_mapping(mapping.pop("adapter_lifecycle", None)),
        candidate_summary=deps.coerce_mapping(mapping.pop("candidate_summary", None)),
        compare_evaluation=deps.coerce_mapping(mapping.pop("compare_evaluation", None)),
        candidate_history=deps.coerce_mapping(mapping.pop("candidate_history", None)),
        candidate_timeline=deps.coerce_mapping(mapping.pop("candidate_timeline", None)),
        operations_console=operations_console,
        daemon_timeline=daemon_timeline,
        runner_timeline=runner_timeline,
        train_queue=deps.coerce_mapping(mapping.pop("train_queue", None)),
        operations_overview=deps.coerce_mapping(mapping.pop("operations_overview", None)),
        operations_alerts=deps.coerce_sequence_of_mappings(mapping.pop("operations_alerts", None)),
        operations_health=deps.coerce_mapping(mapping.pop("operations_health", None)),
        operations_recovery=deps.coerce_mapping(mapping.pop("operations_recovery", None)),
        operations_next_actions=deps.coerce_sequence_of_scalars(mapping.pop("operations_next_actions", None)),
        operations_dashboard=deps.coerce_mapping(mapping.pop("operations_dashboard", None)),
        operations_alert_policy=deps.coerce_mapping(mapping.pop("operations_alert_policy", None)),
        operations_event_stream=operations_event_stream,
        operations_timeline=operations_timeline,
    )


__all__ = ["LegacyStatusSections", "extract_legacy_status_sections"]
