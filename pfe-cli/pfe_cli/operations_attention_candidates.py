"""Candidate attention fragments for operations attention formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_attention_context import OperationsAttentionContext


def append_candidate_attention(
    alerts: list[str],
    *,
    candidate_summary: Mapping[str, Any] | None,
    context: OperationsAttentionContext,
    deps: Any,
) -> None:
    if candidate_summary is None:
        return
    candidate_version = candidate_summary.get("candidate_version")
    candidate_state = candidate_summary.get("candidate_state")
    needs_promotion = candidate_summary.get("candidate_needs_promotion")
    if needs_promotion:
        parts = []
        if candidate_version is not None:
            parts.append(f"version={deps.format_scalar(candidate_version)}")
        if candidate_state is not None:
            parts.append(f"state={deps.format_scalar(candidate_state)}")
        if parts:
            alerts.append("candidate_needs_promotion " + " | ".join(parts))
        else:
            alerts.append("candidate_needs_promotion")
    elif candidate_state in {"training", "pending_eval", "failed_eval"} and not (
        context.monitor_alert_emitted and str(context.resolved_focus).strip().lower().startswith("candidate")
    ):
        parts = []
        if candidate_version is not None:
            parts.append(f"version={deps.format_scalar(candidate_version)}")
        parts.append(f"state={deps.format_scalar(candidate_state)}")
        alerts.append("candidate " + " | ".join(parts))


__all__ = ["append_candidate_attention"]
