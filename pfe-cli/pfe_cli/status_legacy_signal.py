"""Legacy plain-text signal status formatting."""

from __future__ import annotations

from typing import Any

from .status_legacy_signal_sections import (
    append_sample_counts_line,
    append_signal_quality_line,
    append_signal_readiness_line,
    append_signal_sample_counts_line,
    append_signal_sample_details_lines,
)


def append_legacy_sample_and_signal_lines(
    lines: list[str],
    mapping: dict[str, Any],
    *,
    deps: Any,
) -> None:
    """Append legacy sample, signal readiness, and signal quality lines."""
    sample_counts = mapping.pop("sample_counts", None)
    append_sample_counts_line(lines, sample_counts, deps=deps)

    signal_summary = deps.coerce_mapping(mapping.pop("signal_summary", None))
    signal_sample_counts = deps.coerce_mapping(mapping.pop("signal_sample_counts", None))
    signal_sample_details = mapping.pop("signal_sample_details", None)
    signal_quality_summary = deps.coerce_mapping(mapping.pop("signal_quality_summary", None))
    signal_count_value = mapping.pop("signal_count", None)
    signal_sample_count_value = mapping.pop("signal_sample_count", None)
    append_signal_readiness_line(
        lines,
        signal_summary=signal_summary,
        signal_sample_counts=signal_sample_counts,
        signal_count_value=signal_count_value,
        signal_sample_count_value=signal_sample_count_value,
        deps=deps,
    )
    append_signal_sample_counts_line(lines, signal_sample_counts, deps=deps)
    append_signal_sample_details_lines(lines, signal_sample_details, deps=deps)
    append_signal_quality_line(lines, signal_quality_summary, deps=deps)


__all__ = ["append_legacy_sample_and_signal_lines"]
