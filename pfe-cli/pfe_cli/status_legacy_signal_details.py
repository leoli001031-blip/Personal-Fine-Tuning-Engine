"""Signal sample detail formatting for legacy status output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def append_signal_sample_details_lines(lines: list[str], signal_sample_details: Any, *, deps: Any) -> None:
    if not signal_sample_details:
        return

    detail_lines: list[str] = []
    detail_values = (
        signal_sample_details
        if isinstance(signal_sample_details, Sequence)
        and not isinstance(signal_sample_details, (str, bytes, bytearray))
        else [signal_sample_details]
    )
    for detail in detail_values[:3]:
        detail_line = _signal_sample_detail_line(detail, deps=deps)
        if detail_line:
            detail_lines.append(detail_line)
    if detail_lines:
        lines.append("signal sample details:")
        for detail_line in detail_lines:
            lines.append(f"  - {detail_line}")


def _signal_sample_detail_line(detail: Any, *, deps: Any) -> str | None:
    detail_map = deps.coerce_mapping(detail) or {}
    parts: list[str] = []
    for key in ("sample_id", "sample_type", "dataset_split", "source_adapter_version"):
        value = detail_map.get(key)
        if value is not None:
            parts.append(f"{key.replace('_', ' ')}={deps.format_scalar(value)}")
    source_event_ids = detail_map.get("source_event_ids")
    if source_event_ids:
        parts.append(f"source_event_ids={deps.format_scalar(source_event_ids)}")
    if not parts:
        return None
    return " | ".join(parts)


__all__ = ["append_signal_sample_details_lines"]
