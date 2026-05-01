"""Common helpers for Matrix terminal formatters."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from .terminal_theme import MatrixColors


def _coerce_mapping(result: Any) -> dict[str, Any] | None:
    """Convert pydantic-style results into a plain mapping when possible."""
    if isinstance(result, dict):
        return dict(result)
    if is_dataclass(result) and not isinstance(result, type):
        try:
            dumped = asdict(result)
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    to_dict = getattr(result, "dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict()
        except Exception:
            return None
        if isinstance(dumped, dict):
            return dict(dumped)
    return None


def _coerce_sequence_of_mappings(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        return []
    items: list[dict[str, Any]] = []
    for item in result:
        mapping = _coerce_mapping(item)
        if mapping is not None:
            items.append(mapping)
    return items


def _coerce_sequence_of_scalars(result: Any) -> list[str]:
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        return []
    return [str(item) for item in result if item is not None]


def _format_scalar(value: Any) -> str:
    """Format a scalar value."""
    if value is None:
        return f"{MatrixColors.DIM}n/a{MatrixColors.RESET}"
    if isinstance(value, bool):
        return f"{MatrixColors.GREEN}yes{MatrixColors.RESET}" if value else f"{MatrixColors.GRAY}no{MatrixColors.RESET}"
    if isinstance(value, (str, int, float)):
        return f"{MatrixColors.WHITE}{value}{MatrixColors.RESET}"
    if isinstance(value, Mapping):
        return f"{MatrixColors.GRAY}{json.dumps(value, ensure_ascii=False, sort_keys=True)}{MatrixColors.RESET}"
    return f"{MatrixColors.WHITE}{str(value)}{MatrixColors.RESET}"


def _ordered_eval_scores(scores: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return evaluation scores with personalization-oriented keys first."""
    preferred_order = (
        "style_preference_hit_rate",
        "style_match",
        "preference_alignment",
        "quality_preservation",
        "personality_consistency",
    )
    ordered: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for key in preferred_order:
        if key in scores:
            ordered.append((key, scores[key]))
            seen.add(key)
    for key, value in scores.items():
        if key in seen:
            continue
        ordered.append((key, value))
    return ordered
