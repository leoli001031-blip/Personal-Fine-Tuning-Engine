"""Legacy plain-text candidate status formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def append_legacy_candidate_lines(
    lines: list[str],
    mapping: dict[str, Any],
    *,
    candidate_summary: Mapping[str, Any] | None,
    compare_evaluation: Mapping[str, Any] | None,
    deps: Any,
) -> None:
    """Append legacy candidate summary, compare, action, and history lines."""
    _coerce_mapping = deps.coerce_mapping
    _format_compare_evaluation = deps.format_compare_evaluation
    _format_scalar = deps.format_scalar

    if candidate_summary is not None:
        candidate_parts: list[str] = []
        for key in (
            "candidate_version",
            "candidate_state",
            "candidate_can_promote",
            "candidate_can_archive",
            "pending_eval_count",
            "training_count",
            "failed_eval_count",
            "candidate_needs_promotion",
            "promotion_gate_status",
            "promotion_gate_reason",
            "promotion_gate_action",
            "promotion_compare_comparison",
            "promotion_compare_recommendation",
            "promotion_compare_winner",
            "promotion_compare_left_adapter",
            "promotion_compare_right_adapter",
            "promotion_compare_overall_delta",
            "promotion_compare_details_count",
            "promotion_compare_personalization_delta",
            "promotion_compare_quality_delta",
            "promotion_compare_style_preference_hit_rate_delta",
            "promotion_compare_personalization_summary",
            "promotion_compare_quality_summary",
            "promotion_compare_summary_line",
        ):
            value = candidate_summary.get(key)
            if value is not None:
                candidate_parts.append(f"{key}={_format_scalar(value)}")
        if candidate_parts:
            lines.append("candidate summary: " + " | ".join(candidate_parts))
    compare_line = _format_compare_evaluation(compare_evaluation)
    if compare_line is not None:
        lines.append(compare_line)

    candidate_action = _coerce_mapping(mapping.pop("candidate_action", None))
    if candidate_action is not None:
        action_parts: list[str] = []
        for key in (
            "action",
            "status",
            "reason",
            "required_action",
            "operator_note",
            "candidate_version",
            "promoted_version",
            "archived_version",
        ):
            value = candidate_action.get(key)
            if value is not None:
                action_parts.append(f"{key}={_format_scalar(value)}")
        if action_parts:
            lines.append("candidate action: " + " | ".join(action_parts))

    candidate_history = _coerce_mapping(mapping.pop("candidate_history", None))
    if candidate_history is not None:
        history_parts: list[str] = []
        for key in ("count", "last_action", "last_status", "last_reason", "last_note", "last_candidate_version"):
            value = candidate_history.get(key)
            if value is not None:
                history_parts.append(f"{key}={_format_scalar(value)}")
        if history_parts:
            lines.append("candidate history: " + " | ".join(history_parts))


__all__ = ["append_legacy_candidate_lines"]
