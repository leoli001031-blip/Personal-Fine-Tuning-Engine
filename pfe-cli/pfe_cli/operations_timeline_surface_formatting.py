"""Operations timeline surface formatter."""

from __future__ import annotations

from typing import Any

from .operations_formatting_deps import OperationsFormattingDeps


def format_operations_timeline(
    result: Any,
    *,
    deps: OperationsFormattingDeps,
) -> list[str] | None:
    mapping = deps.coerce_mapping(result)
    if not mapping:
        return None

    lines = ["operations timeline:"]
    summary_parts: list[str] = []
    summary_line = mapping.get("summary_line")
    if summary_line:
        summary_parts.append(f"summary={deps.format_scalar(summary_line)}")
    if summary_parts:
        lines.append("  " + " | ".join(summary_parts))

    for label in ("candidate", "queue", "runner", "daemon"):
        section = deps.coerce_mapping(mapping.get(label))
        if not section:
            continue
        section_parts: list[str] = []
        if label == "candidate":
            for key in (
                "current_stage",
                "last_candidate_version",
                "last_reason",
                "latest_timestamp",
                "transition_count",
            ):
                value = section.get(key)
                if value is not None:
                    section_parts.append(f"{key}={deps.format_scalar(value)}")
        elif label == "queue":
            for key in ("count", "last_transition", "last_reason", "latest_timestamp", "transition_count"):
                value = section.get(key)
                if value is not None:
                    section_parts.append(f"{key}={deps.format_scalar(value)}")
        else:
            for key in (
                "count",
                "last_event",
                "last_reason",
                "takeover_event_count",
                "last_takeover_event",
                "last_takeover_reason",
                "recovery_event_count",
                "last_recovery_event",
                "last_recovery_reason",
                "last_recovery_note",
                "recent_anomaly_reason",
                "latest_timestamp",
            ):
                value = section.get(key)
                if value is not None:
                    section_parts.append(f"{key}={deps.format_scalar(value)}")
        if section_parts:
            lines.append(f"  {label}: " + " | ".join(section_parts))
    return lines


__all__ = ["format_operations_timeline"]
