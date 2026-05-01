"""Candidate console summary text."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_routing_deps import ConsoleRoutingDeps
from .console_routing_summary_helpers import append_mapping_parts, render_summary


def console_candidate_summary_text(
    payload: Mapping[str, Any],
    *,
    deps: ConsoleRoutingDeps,
    timeline: Mapping[str, Any] | None = None,
) -> str:
    mapping = deps.coerce_mapping(payload) or {}
    candidate_summary = deps.coerce_mapping(mapping.get("candidate_summary")) or {}
    candidate_timeline = deps.coerce_mapping(timeline) or deps.coerce_mapping(mapping.get("candidate_timeline")) or {}

    parts: list[str] = []
    append_mapping_parts(
        parts,
        candidate_summary,
        (
            "candidate_version",
            "candidate_state",
            "candidate_can_promote",
            "candidate_can_archive",
            "pending_eval_count",
            "training_count",
            "failed_eval_count",
            "candidate_needs_promotion",
            "promotion_compare_comparison",
            "promotion_compare_recommendation",
            "promotion_compare_winner",
            "promotion_compare_left_adapter",
            "promotion_compare_right_adapter",
            "promotion_compare_overall_delta",
            "promotion_compare_style_preference_hit_rate_delta",
            "promotion_compare_personalization_delta",
            "promotion_compare_quality_delta",
            "promotion_compare_personalization_summary",
            "promotion_compare_quality_summary",
            "promotion_compare_summary_line",
        ),
        deps=deps,
    )
    append_mapping_parts(
        parts,
        candidate_timeline,
        ("current_stage", "transition_count", "last_reason", "last_candidate_version"),
        deps=deps,
    )
    return render_summary("PFE candidate summary", parts, fallback="state=idle")


__all__ = ["console_candidate_summary_text"]
