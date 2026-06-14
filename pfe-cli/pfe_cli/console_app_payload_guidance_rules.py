"""Payload-derived guidance rule helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_app_action_guidance import _action_command_guidance
from .console_app_data import _mapping, _value


def secondary_action_guidance_labels(actions: list[str]) -> list[str]:
    labels: list[str] = []
    for secondary_action in actions[:2]:
        secondary_guidance = _action_command_guidance(secondary_action)
        if secondary_guidance is None:
            continue
        label = str(secondary_guidance[0] or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def summary_guidance(summary: Mapping[str, Any] | None) -> tuple[str, str] | None:
    summary_map = _mapping(summary)
    primary_action = _value(summary_map, "primary_action", default="")
    primary_guidance = _action_command_guidance(primary_action)
    if primary_guidance is None:
        return None
    secondary_actions = [str(action or "") for action in list(summary_map.get("secondary_actions") or [])]
    secondary_guidance_values = secondary_action_guidance_labels(secondary_actions)
    if secondary_guidance_values:
        return (primary_guidance[0], " ".join(secondary_guidance_values))
    return primary_guidance


def alert_policy_actions(alert_policy: Mapping[str, Any]) -> tuple[str, list[str]]:
    required_action = _value(alert_policy, "required_action", "primary_action", default="")
    secondary_action_values: list[str] = []
    for raw_action in [
        alert_policy.get("secondary_action"),
        *list(alert_policy.get("secondary_actions") or []),
    ]:
        text = str(raw_action or "").strip()
        if text and text not in secondary_action_values:
            secondary_action_values.append(text)
    return str(required_action or ""), secondary_action_values


def payload_summary_guidance(
    *,
    normalized_focus: str,
    operations_dashboard: Mapping[str, Any],
    operations_console: Mapping[str, Any],
) -> tuple[str, str] | None:
    if normalized_focus.startswith("candidate"):
        return summary_guidance(
            operations_dashboard.get("candidate_action_summary") or operations_console.get("candidate_action_summary")
        )
    if normalized_focus.startswith("queue"):
        return summary_guidance(
            operations_dashboard.get("queue_action_summary") or operations_console.get("queue_action_summary")
        )
    if normalized_focus.startswith("runner") or normalized_focus.startswith("daemon"):
        runtime_summary_guidance = summary_guidance(
            operations_dashboard.get("runtime_action_summary") or operations_console.get("runtime_action_summary")
        )
        runtime_summary = _mapping(
            operations_dashboard.get("runtime_action_summary") or operations_console.get("runtime_action_summary")
        )
        if (
            runtime_summary_guidance is not None
            and _value(runtime_summary, "primary_action", default="") == "inspect_runtime_stability"
        ):
            return runtime_summary_guidance
    return None


def candidate_lifecycle_guidance(payload: Mapping[str, Any], normalized_focus: str) -> tuple[str, str] | None:
    candidate_summary = _mapping(payload.get("candidate_summary"))
    if not normalized_focus.startswith("candidate") or normalized_focus == "candidate_ready_for_promotion":
        return None
    can_promote = bool(candidate_summary.get("candidate_can_promote"))
    can_archive = bool(candidate_summary.get("candidate_can_archive"))
    if can_promote:
        return ("/promote", "/candidate /cand sum")
    if can_archive:
        return ("/archive", "/candidate /cand sum")
    return None


__all__ = [
    "alert_policy_actions",
    "candidate_lifecycle_guidance",
    "payload_summary_guidance",
    "secondary_action_guidance_labels",
]
