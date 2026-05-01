"""Mapped focus action decisions derived from policy summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_action_mapping import action_mapping, action_values, apply_secondary_action_values, summary_mapping
from .console_actions_deps import ConsoleActionsDeps
from .console_focus_context import ConsoleFocusContext


def _summary_mapping(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    key: str,
    *,
    deps: ConsoleActionsDeps,
) -> Mapping[str, Any]:
    return deps.coerce_mapping(first.get(key)) or deps.coerce_mapping(second.get(key)) or {}


def mapped_console_focus_actions(
    context: ConsoleFocusContext,
    *,
    deps: ConsoleActionsDeps,
) -> dict[str, str | None] | None:
    mapped_required = action_mapping(context.required_action)
    if mapped_required is not None:
        return apply_secondary_action_values(mapped_required, action_values(context.alert_policy))

    candidate_action_summary = _summary_mapping(
        context.operations_dashboard,
        context.operations_console,
        "candidate_action_summary",
        deps=deps,
    )
    queue_action_summary = _summary_mapping(
        context.operations_dashboard,
        context.operations_console,
        "queue_action_summary",
        deps=deps,
    )
    runtime_action_summary = _summary_mapping(
        context.operations_dashboard,
        context.operations_console,
        "runtime_action_summary",
        deps=deps,
    )

    if context.current_focus.startswith("candidate"):
        mapped_candidate_summary = summary_mapping(candidate_action_summary, deps=deps)
        if mapped_candidate_summary is not None:
            return mapped_candidate_summary
    if context.current_focus.startswith("queue"):
        mapped_queue_summary = summary_mapping(queue_action_summary, deps=deps)
        if mapped_queue_summary is not None:
            return mapped_queue_summary
    if context.current_focus.startswith("runner") or context.current_focus.startswith("daemon"):
        mapped_runtime_summary = summary_mapping(runtime_action_summary, deps=deps)
        if (
            mapped_runtime_summary is not None
            and str(runtime_action_summary.get("primary_action") or "").strip() == "inspect_runtime_stability"
        ):
            return mapped_runtime_summary

    return None


__all__ = ["mapped_console_focus_actions"]
