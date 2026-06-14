"""Candidate status sections for Matrix terminal status output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping
from .terminal_theme import draw_box, format_key_value


def append_candidate_status_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append candidate summary, action, and history boxes."""
    candidate_summary = _coerce_mapping(mapping.get("candidate_summary"))
    if candidate_summary:
        cands = []
        for key in (
            "candidate_version",
            "candidate_state",
            "candidate_can_promote",
            "candidate_can_archive",
            "pending_eval_count",
            "candidate_needs_promotion",
            "promotion_gate_status",
            "promotion_gate_reason",
            "promotion_gate_action",
            "promotion_compare_comparison",
            "promotion_compare_recommendation",
            "promotion_compare_winner",
            "promotion_compare_overall_delta",
            "promotion_compare_style_preference_hit_rate_delta",
            "promotion_compare_personalization_summary",
            "promotion_compare_quality_summary",
            "promotion_compare_summary_line",
        ):
            value = candidate_summary.get(key)
            if value is not None:
                cands.append(format_key_value(key.replace("_", " "), value))
        if cands:
            lines.append(draw_box("CANDIDATE SUMMARY", cands))
            lines.append("")

    candidate_action = _coerce_mapping(mapping.get("candidate_action"))
    if candidate_action:
        action_lines = []
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
                action_lines.append(format_key_value(key.replace("_", " "), value))
        if action_lines:
            lines.append(draw_box("CANDIDATE ACTION", action_lines))
            lines.append("")

    candidate_history = _coerce_mapping(mapping.get("candidate_history"))
    if candidate_history:
        hist_lines = []
        for key in ("count", "last_action", "last_status", "last_reason", "last_note", "last_candidate_version"):
            value = candidate_history.get(key)
            if value is not None:
                hist_lines.append(format_key_value(key.replace("_", " "), value))
        if hist_lines:
            lines.append(draw_box("CANDIDATE HISTORY", hist_lines))
            lines.append("")


__all__ = ["append_candidate_status_sections"]
