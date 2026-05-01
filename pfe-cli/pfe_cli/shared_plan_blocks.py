"""Generic plan formatting helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .shared_coercion_formatting import coerce_mapping, format_scalar


def plan_summary(plan: Any, fields: Sequence[str]) -> str:
    mapping = coerce_mapping(plan)
    if mapping is None:
        return format_scalar(plan)

    parts: list[str] = []
    for field in fields:
        if field in mapping:
            parts.append(f"{field.replace('_', ' ')}={format_scalar(mapping[field])}")
    if not parts:
        return format_scalar(mapping)
    return " | ".join(parts)


def format_plan_block(title: str, plan: Any, fields: Sequence[str]) -> list[str]:
    lines = [f"{title} plan:"]
    mapping = coerce_mapping(plan)
    if mapping is None:
        lines.append(f"  {format_scalar(plan)}")
        return lines

    lines.append(f"  {plan_summary(mapping, fields)}")
    notes = mapping.get("notes")
    if notes:
        lines.append("  notes:")
        for note in notes if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes, bytearray)) else [notes]:
            lines.append(f"    - {format_scalar(note)}")
    return lines


def format_compact_plan_line(label: str, plan: Any, fields: Sequence[str]) -> str:
    mapping = coerce_mapping(plan)
    if mapping is None:
        return f"{label}: {format_scalar(plan)}"
    parts = [f"{field}={format_scalar(mapping.get(field))}" for field in fields if mapping.get(field) is not None]
    if not parts:
        return f"{label}: {format_scalar(mapping)}"
    return f"{label}: " + " | ".join(parts)


__all__ = ["format_compact_plan_line", "format_plan_block", "plan_summary"]
