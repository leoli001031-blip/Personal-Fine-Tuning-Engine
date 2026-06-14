"""Shared eval score ordering helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def ordered_eval_scores(scores: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Return eval scores with personalization metrics highlighted first."""
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


__all__ = ["ordered_eval_scores"]
