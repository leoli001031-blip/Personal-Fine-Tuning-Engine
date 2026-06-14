"""Review lines for legacy train queue status."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_queue_helpers import append_scalar_parts


def append_review(lines: list[str], train_queue: Mapping[str, Any], *, deps: Any) -> None:
    review_summary = deps.coerce_mapping(train_queue.get("review_summary"))
    if not review_summary:
        return
    review_parts: list[str] = []
    append_scalar_parts(
        review_parts,
        review_summary,
        (
            "reviewed_transition_count",
            "approved_transition_count",
            "rejected_transition_count",
            "last_review_event",
            "last_review_reason",
            "last_review_note",
            "next_job_id",
            "next_confirmation_reason",
        ),
        deps=deps,
    )
    if review_parts:
        lines.append("queue review: " + " | ".join(review_parts))


def append_review_policy(lines: list[str], train_queue: Mapping[str, Any], *, deps: Any) -> None:
    review_policy_summary = deps.coerce_mapping(train_queue.get("review_policy_summary"))
    if not review_policy_summary:
        return
    review_policy_parts: list[str] = []
    append_scalar_parts(
        review_policy_parts,
        review_policy_summary,
        (
            "review_mode",
            "queue_entry_mode",
            "review_required_by_policy",
            "review_required_now",
            "next_action",
            "review_reason",
        ),
        deps=deps,
    )
    if review_policy_parts:
        lines.append("queue review policy: " + " | ".join(review_policy_parts))


__all__ = ["append_review", "append_review_policy"]
