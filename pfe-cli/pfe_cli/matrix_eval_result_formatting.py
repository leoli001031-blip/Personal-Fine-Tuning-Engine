"""Matrix terminal formatter for eval results."""

from __future__ import annotations

import json
from typing import Any

from .matrix_formatting_common import _coerce_mapping, _ordered_eval_scores
from .terminal_theme import MatrixColors, draw_box, draw_header, format_key_value


def format_eval_result_matrix(result: Any, *, workspace: str | None = None) -> str:
    """Format eval result in Matrix Green terminal style."""
    lines = []

    lines.append(draw_header("EVALUATION RESULT"))

    mapping = _coerce_mapping(result)
    if mapping is None:
        try:
            mapping = json.loads(result)
        except Exception:
            lines.append(f"{MatrixColors.RED}ERROR: Unable to parse evaluation result{MatrixColors.RESET}")
            return "\n".join(lines)

    content = []

    adapter_version = mapping.get("adapter_version", "n/a")
    content.append(format_key_value("adapter", adapter_version))

    base_model = mapping.get("base_model", "n/a")
    content.append(format_key_value("base model", base_model))

    num_samples = mapping.get("num_test_samples", 0)
    content.append(format_key_value("test samples", num_samples))

    recommendation = mapping.get("recommendation", "unknown")
    comparison = mapping.get("comparison", "unknown")

    rec_color = MatrixColors.GREEN if recommendation == "deploy" else MatrixColors.AMBER if recommendation == "review" else MatrixColors.RED
    content.append(format_key_value("recommendation", f"{rec_color}{recommendation.upper()}{MatrixColors.RESET}"))
    content.append(format_key_value("comparison", comparison))

    scores = _coerce_mapping(mapping.get("scores"))
    if scores:
        content.append("")
        content.append(f"{MatrixColors.GREEN_BRIGHT}SCORES:{MatrixColors.RESET}")
        for key, value in _ordered_eval_scores(scores):
            bar_width = int(value * 20)
            bar = "█" * bar_width + "░" * (20 - bar_width)
            content.append(f"  {key:25} [{MatrixColors.GREEN}{bar}{MatrixColors.RESET}] {value:.2f}")

    lines.append(draw_box("EVALUATION METRICS", content))
    lines.append("")

    return "\n".join(lines)


__all__ = ["format_eval_result_matrix"]
