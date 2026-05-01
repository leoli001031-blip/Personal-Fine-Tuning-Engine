"""Score and detail sections for legacy eval formatting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .legacy_result_deps import LegacyResultFormattingDeps


def append_scores_line(lines: list[str], mapping: Mapping[str, Any], *, deps: LegacyResultFormattingDeps) -> None:
    scores = deps.coerce_mapping(mapping.get("scores"))
    if not scores:
        scores = deps.coerce_mapping(mapping.get("score_deltas"))
    if not scores:
        return

    score_parts: list[str] = []
    ordered_scores = deps.ordered_eval_scores(scores)
    if ordered_scores:
        for key, value in ordered_scores:
            score_parts.append(f"{key}={deps.format_scalar(value)}")
    if score_parts:
        label = "score_deltas" if mapping.get("score_deltas") is not None and mapping.get("scores") is None else "scores"
        lines.append(f"{label}: " + " | ".join(score_parts))


def append_details_line(lines: list[str], mapping: Mapping[str, Any], *, deps: LegacyResultFormattingDeps) -> None:
    details = mapping.get("details")
    if isinstance(details, Sequence) and not isinstance(details, (str, bytes, bytearray)):
        lines.append(f"details: {deps.format_scalar(len(details))} item(s)")


__all__ = ["append_details_line", "append_scores_line"]
