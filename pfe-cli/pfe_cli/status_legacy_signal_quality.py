"""Signal quality formatting for legacy status output."""

from __future__ import annotations

from typing import Any


def append_signal_quality_line(lines: list[str], signal_quality_summary: dict[str, Any] | None, *, deps: Any) -> None:
    if signal_quality_summary is None:
        return

    quality_parts: list[str] = []
    for key in ("evaluated_count", "passed_count", "filtered_count", "minimum_confidence"):
        value = signal_quality_summary.get(key)
        if value is not None:
            quality_parts.append(f"{key}={deps.format_scalar(value)}")
    filtered_reasons = signal_quality_summary.get("filtered_reasons")
    if filtered_reasons:
        if isinstance(filtered_reasons, dict):
            reason_text = ", ".join(f"{key}:{value}" for key, value in filtered_reasons.items())
            quality_parts.append(f"filtered_reasons={reason_text}")
        else:
            quality_parts.append(f"filtered_reasons={deps.format_scalar(filtered_reasons)}")
    if quality_parts:
        lines.append("signal quality filter: " + " | ".join(quality_parts))


__all__ = ["append_signal_quality_line"]
