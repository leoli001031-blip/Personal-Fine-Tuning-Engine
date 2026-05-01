"""Line appenders for legacy eval result formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .legacy_eval_compare_sections import append_compare_detail_line, append_compare_line, append_result_line
from .legacy_eval_score_sections import append_details_line, append_scores_line
from .legacy_result_deps import LegacyResultFormattingDeps


def append_eval_metadata_line(
    lines: list[str],
    mapping: Mapping[str, Any],
    *,
    version: Any,
    deps: LegacyResultFormattingDeps,
) -> None:
    base_model = deps.pick_first(mapping, "base_model")
    num_test_samples = deps.pick_first(mapping, "num_test_samples", "num_samples")
    if version is None and base_model is None and num_test_samples is None:
        return

    parts = []
    if version is not None:
        parts.append(f"adapter_version={deps.format_scalar(version)}")
    if base_model is not None:
        parts.append(f"base_model={deps.format_scalar(base_model)}")
    if num_test_samples is not None:
        parts.append(f"num_test_samples={deps.format_scalar(num_test_samples)}")
    lines.append(" | ".join(parts))


__all__ = [
    "append_compare_detail_line",
    "append_compare_line",
    "append_details_line",
    "append_eval_metadata_line",
    "append_result_line",
    "append_scores_line",
]
