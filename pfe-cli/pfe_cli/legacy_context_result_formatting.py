"""Legacy contextual result summary formatting helpers."""

from __future__ import annotations

from typing import Any

from .legacy_recent_training_snapshot_parts import (
    recent_training_export_line,
    recent_training_job_line,
    recent_training_summary_line,
)
from .legacy_result_deps import LegacyResultFormattingDeps


def format_incremental_context(
    context: Any,
    *,
    deps: LegacyResultFormattingDeps,
) -> str | None:
    mapping = deps.coerce_mapping(context)
    if mapping is None:
        return None

    parts: list[str] = []
    for key in (
        "requested_base_adapter",
        "parent_adapter_version",
        "parent_base_model",
        "parent_adapter_path",
        "source_adapter_version",
        "source_adapter_path",
        "source_model",
    ):
        value = mapping.get(key)
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")
    if not parts:
        return None
    return "incremental: " + " | ".join(parts)


def format_compare_evaluation(
    compare_evaluation: Any,
    *,
    deps: LegacyResultFormattingDeps,
) -> str | None:
    mapping = deps.coerce_mapping(compare_evaluation)
    if mapping is None:
        return None

    parts: list[str] = []
    for key in (
        "left_adapter",
        "right_adapter",
        "comparison",
        "winner",
        "recommendation",
        "overall_delta",
        "style_preference_hit_rate_delta",
        "personalization_summary",
        "quality_summary",
        "summary_line",
    ):
        value = mapping.get(key)
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")
    if not parts:
        return None
    return "promotion compare: " + " | ".join(parts)


def format_recent_training_snapshot(
    snapshot: Any,
    *,
    deps: LegacyResultFormattingDeps,
) -> list[str] | None:
    mapping = deps.coerce_mapping(snapshot)
    if mapping is None:
        return None

    lines: list[str] = []
    summary_line = recent_training_summary_line(mapping, deps=deps)
    if summary_line is not None:
        lines.append(summary_line)

    incremental_line = format_incremental_context(mapping.get("incremental_context") or mapping, deps=deps)
    if incremental_line is not None:
        lines.append(incremental_line)

    job_line = recent_training_job_line(mapping, deps=deps)
    if job_line is not None:
        lines.append(job_line)

    export_exec_line = recent_training_export_line(mapping, deps=deps)
    if export_exec_line is not None:
        lines.append(export_exec_line)
    return lines or None


__all__ = [
    "format_compare_evaluation",
    "format_incremental_context",
    "format_recent_training_snapshot",
]
