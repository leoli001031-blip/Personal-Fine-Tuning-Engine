"""Legacy eval result formatter."""

from __future__ import annotations

from typing import Any

from .legacy_adapter_result_formatting import format_adapter_snapshot_line
from .legacy_eval_mapping import coerce_eval_result_mapping
from .legacy_eval_sections import (
    append_compare_detail_line,
    append_compare_line,
    append_details_line,
    append_eval_metadata_line,
    append_result_line,
    append_scores_line,
)
from .legacy_result_deps import LegacyResultFormattingDeps


def format_eval_result_legacy(
    result: Any,
    *,
    workspace: str | None = None,
    deps: LegacyResultFormattingDeps,
) -> str:
    """Legacy plain text formatter kept for compatibility checks."""

    mapping = coerce_eval_result_mapping(result, deps=deps)
    if mapping is None and isinstance(result, str):
        return result.strip() if result.strip() else deps.format_scalar(result)
    if mapping is None:
        return deps.format_scalar(result)

    lines = ["PFE eval"]
    append_compare_line(lines, mapping, deps=deps)
    version = deps.pick_first(mapping, "adapter_version", "version")
    append_eval_metadata_line(lines, mapping, version=version, deps=deps)

    adapter_snapshot = deps.lookup_adapter_snapshot(str(version) if version is not None else None, workspace=workspace)
    adapter_line = format_adapter_snapshot_line("evaluated adapter", adapter_snapshot, include_latest=True, deps=deps)
    if adapter_line is not None:
        lines.append(adapter_line)

    append_result_line(lines, mapping, deps=deps)
    append_compare_detail_line(lines, mapping, deps=deps)
    append_scores_line(lines, mapping, deps=deps)
    append_details_line(lines, mapping, deps=deps)
    return "\n".join(lines)


__all__ = ["format_eval_result_legacy"]
