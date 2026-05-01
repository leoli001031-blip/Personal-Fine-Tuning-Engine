"""Section helpers for legacy signal status formatting."""

from __future__ import annotations

from typing import Any

from .status_legacy_signal_details import append_signal_sample_details_lines
from .status_legacy_signal_quality import append_signal_quality_line


def append_sample_counts_line(lines: list[str], sample_counts: Any, *, deps: Any) -> None:
    if sample_counts is None:
        return
    sample_map = deps.coerce_mapping(sample_counts) or {}
    sample_summary = " | ".join(
        f"{split}={deps.format_scalar(sample_map.get(split))}"
        for split in ("train", "val", "test")
        if split in sample_map
    )
    if sample_summary:
        lines.append(f"sample counts: {sample_summary}")


def append_signal_readiness_line(
    lines: list[str],
    *,
    signal_summary: dict[str, Any] | None,
    signal_sample_counts: dict[str, Any] | None,
    signal_count_value: Any,
    signal_sample_count_value: Any,
    deps: Any,
) -> None:
    if signal_summary is None and signal_sample_counts is None and signal_count_value is None:
        return

    signal_parts: list[str] = []
    if signal_summary is not None:
        signal_parts.extend(_signal_summary_parts(signal_summary, deps=deps))
    elif signal_count_value is not None:
        signal_parts.append(f"count={deps.format_scalar(signal_count_value)}")
    if signal_sample_count_value is not None:
        signal_parts.append(f"samples={deps.format_scalar(signal_sample_count_value)}")
    if signal_parts:
        lines.append("signal readiness: " + " | ".join(signal_parts))


def append_signal_sample_counts_line(
    lines: list[str],
    signal_sample_counts: dict[str, Any] | None,
    *,
    deps: Any,
) -> None:
    if signal_sample_counts is None:
        return
    sample_summary = " | ".join(
        f"{split}={deps.format_scalar(signal_sample_counts.get(split))}"
        for split in ("train", "val", "test")
        if split in signal_sample_counts
    )
    if sample_summary:
        lines.append(f"signal samples: {sample_summary}")


def _signal_summary_parts(signal_summary: dict[str, Any], *, deps: Any) -> list[str]:
    parts: list[str] = []
    for key in (
        "state",
        "collection_enabled",
        "event_chain_ready",
        "event_chain_complete_count",
        "processed_count",
        "latest_signal_id",
        "quality_filter_state",
        "quality_filtered_count",
    ):
        value = signal_summary.get(key)
        if key == "latest_signal_id" and not value:
            continue
        if value is None:
            continue
        parts.append(f"{key}={deps.format_scalar(value)}")
    return parts


__all__ = [
    "append_sample_counts_line",
    "append_signal_quality_line",
    "append_signal_readiness_line",
    "append_signal_sample_counts_line",
    "append_signal_sample_details_lines",
]
