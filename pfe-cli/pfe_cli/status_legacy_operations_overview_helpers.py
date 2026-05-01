"""Helpers for legacy operations overview status formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


OVERVIEW_KEYS = (
    "attention_needed",
    "attention_reason",
    "trigger_state",
    "trigger_ready",
    "candidate_version",
    "candidate_state",
    "candidate_needs_promotion",
    "queue_count",
    "awaiting_confirmation_count",
    "runner_active",
    "runner_lock_state",
    "runner_last_event",
)


def first_operations_focus(
    *,
    operations_overview: Mapping[str, Any],
    operations_dashboard: Mapping[str, Any] | None,
    deps: Any,
) -> Any:
    for candidate in (
        operations_overview.get("current_focus"),
        operations_overview.get("monitor_focus"),
        deps.coerce_mapping(operations_dashboard).get("monitor_focus") if operations_dashboard is not None else None,
        operations_overview.get("attention_reason"),
    ):
        if candidate is None:
            continue
        if str(candidate).strip().lower() in {"", "none", "idle", "stable"}:
            continue
        return candidate
    return None


def operations_required_action(
    *,
    operations_overview: Mapping[str, Any],
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    deps: Any,
) -> Any:
    return (
        operations_overview.get("required_action")
        or (
            deps.coerce_mapping(operations_alert_policy).get("required_action")
            if operations_alert_policy is not None
            else None
        )
        or (
            deps.coerce_mapping(operations_dashboard).get("required_action")
            if operations_dashboard is not None
            else None
        )
    )


def operations_overview_parts(
    operations_overview: Mapping[str, Any],
    *,
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    deps: Any,
) -> list[str]:
    overview_parts: list[str] = []
    overview_focus = first_operations_focus(
        operations_overview=operations_overview,
        operations_dashboard=operations_dashboard,
        deps=deps,
    )
    overview_required_action = operations_required_action(
        operations_overview=operations_overview,
        operations_dashboard=operations_dashboard,
        operations_alert_policy=operations_alert_policy,
        deps=deps,
    )
    for key in OVERVIEW_KEYS:
        value = operations_overview.get(key)
        if value is not None:
            overview_parts.append(f"{key}={deps.format_scalar(value)}")
    if overview_focus is not None:
        overview_parts.append(f"monitor_focus={deps.format_scalar(overview_focus)}")
    if overview_required_action is not None:
        overview_parts.append(f"required_action={deps.format_scalar(overview_required_action)}")
    summary_line = operations_overview.get("summary_line")
    inspection_summary_line = operations_overview.get("inspection_summary_line")
    summary_line, inspection_summary_line = deps.prefer_inspection_summary_for_generic_monitor(
        focus=overview_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )
    if summary_line:
        overview_parts.append(f"summary={deps.format_scalar(summary_line)}")
    if inspection_summary_line and inspection_summary_line != summary_line:
        overview_parts.append(f"inspection_summary={deps.format_scalar(inspection_summary_line)}")
    return overview_parts


def append_auto_train_blocker(lines: list[str], operations_overview: Mapping[str, Any], *, deps: Any) -> None:
    auto_train_blocker = deps.coerce_mapping(operations_overview.get("auto_train_blocker"))
    if auto_train_blocker is None:
        return

    blocker_parts: list[str] = []
    for key in ("source", "reason", "action", "category", "summary"):
        value = auto_train_blocker.get(key)
        if value is not None:
            blocker_parts.append(f"{key}={deps.format_scalar(value)}")
    secondary_reasons = deps.coerce_sequence_of_scalars(auto_train_blocker.get("secondary_reasons"))
    secondary_actions = deps.coerce_sequence_of_scalars(auto_train_blocker.get("secondary_actions"))
    if secondary_reasons:
        blocker_parts.append(f"secondary_reasons={deps.format_scalar(secondary_reasons)}")
    if secondary_actions:
        blocker_parts.append(f"secondary_actions={deps.format_scalar(secondary_actions)}")
    if blocker_parts:
        lines.append("auto train blocker: " + " | ".join(blocker_parts))


__all__ = ["append_auto_train_blocker", "operations_overview_parts"]
