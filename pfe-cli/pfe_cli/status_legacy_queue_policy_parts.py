"""Policy and confirmation lines for legacy train queue status."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_queue_helpers import append_scalar_parts


def append_policy(lines: list[str], train_queue: Mapping[str, Any], *, deps: Any) -> None:
    policy_summary = deps.coerce_mapping(train_queue.get("policy_summary"))
    if not policy_summary:
        return
    policy_parts: list[str] = []
    append_scalar_parts(policy_parts, policy_summary, ("current_priority_source", "current_dedup_scope"), deps=deps)
    for key in ("dedup_scopes", "priority_sources"):
        value = policy_summary.get(key)
        if value:
            policy_parts.append(f"{key}={deps.format_scalar(value)}")
    if policy_parts:
        lines.append("queue policy: " + " | ".join(policy_parts))


def append_confirmation(lines: list[str], train_queue: Mapping[str, Any], *, deps: Any) -> None:
    confirmation_summary = deps.coerce_mapping(train_queue.get("confirmation_summary"))
    if not confirmation_summary:
        return
    confirmation_parts: list[str] = []
    append_scalar_parts(
        confirmation_parts,
        confirmation_summary,
        (
            "confirmation_required_count",
            "awaiting_confirmation_count",
            "next_job_id",
            "next_confirmation_reason",
        ),
        deps=deps,
    )
    if confirmation_parts:
        lines.append("queue confirmation: " + " | ".join(confirmation_parts))


__all__ = ["append_confirmation", "append_policy"]
