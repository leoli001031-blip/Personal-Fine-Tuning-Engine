"""Compare and result sections for legacy eval formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .legacy_result_deps import LegacyResultFormattingDeps


def append_compare_line(lines: list[str], mapping: Mapping[str, Any], *, deps: LegacyResultFormattingDeps) -> None:
    left_adapter = deps.pick_first(mapping, "left_adapter", "left_adapter_version")
    right_adapter = deps.pick_first(mapping, "right_adapter", "right_adapter_version")
    if left_adapter is None and right_adapter is None:
        return

    parts: list[str] = []
    if left_adapter is not None:
        parts.append(f"left_adapter={deps.format_scalar(left_adapter)}")
    if right_adapter is not None:
        parts.append(f"right_adapter={deps.format_scalar(right_adapter)}")
    for key in ("comparison", "winner", "recommendation", "overall_delta"):
        value = deps.pick_first(mapping, key)
        if value is not None:
            parts.append(f"{key}={deps.format_scalar(value)}")
    for key in ("personalization_summary", "quality_summary", "summary_line"):
        value = deps.pick_first(mapping, key)
        if value:
            parts.append(f"{key}={deps.format_scalar(value)}")
    if parts:
        lines.append("compare: " + " | ".join(parts))


def append_result_line(lines: list[str], mapping: Mapping[str, Any], *, deps: LegacyResultFormattingDeps) -> None:
    recommendation = deps.pick_first(mapping, "recommendation")
    comparison = deps.pick_first(mapping, "comparison")
    if recommendation is None and comparison is None:
        return

    parts: list[str] = []
    if recommendation is not None:
        parts.append(f"recommendation={deps.format_scalar(recommendation)}")
    if comparison is not None:
        parts.append(f"comparison={deps.format_scalar(comparison)}")
    lines.append("result: " + " | ".join(parts))


def append_compare_detail_line(lines: list[str], mapping: Mapping[str, Any], *, deps: LegacyResultFormattingDeps) -> None:
    personalization_summary = deps.pick_first(mapping, "personalization_summary")
    quality_summary = deps.pick_first(mapping, "quality_summary")
    summary_line = deps.pick_first(mapping, "summary_line")
    if not (personalization_summary or quality_summary or summary_line):
        return

    parts: list[str] = []
    if personalization_summary:
        parts.append(f"personalization_summary={deps.format_scalar(personalization_summary)}")
    if quality_summary:
        parts.append(f"quality_summary={deps.format_scalar(quality_summary)}")
    if summary_line:
        parts.append(f"summary_line={deps.format_scalar(summary_line)}")
    lines.append("compare detail: " + " | ".join(parts))


__all__ = ["append_compare_detail_line", "append_compare_line", "append_result_line"]
