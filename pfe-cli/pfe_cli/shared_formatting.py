"""Shared CLI coercion and compact formatting helpers."""

from __future__ import annotations

from .shared_backend_formatting import format_backend_dispatch, format_export_write
from .shared_coercion_formatting import (
    coerce_mapping,
    coerce_sequence_of_mappings,
    coerce_sequence_of_scalars,
    format_scalar,
    ordered_eval_scores,
    pick_first,
    yes_no,
)
from .shared_monitor_formatting import (
    GENERIC_MONITOR_FOCUSES,
    prefer_inspection_summary_for_generic_monitor,
)
from .shared_plan_formatting import (
    format_compact_plan_line,
    format_plan_block,
    format_trainer_block,
    format_trainer_summary,
    plan_summary,
)

__all__ = [
    "GENERIC_MONITOR_FOCUSES",
    "coerce_mapping",
    "coerce_sequence_of_mappings",
    "coerce_sequence_of_scalars",
    "format_backend_dispatch",
    "format_compact_plan_line",
    "format_export_write",
    "format_plan_block",
    "format_scalar",
    "format_trainer_block",
    "format_trainer_summary",
    "ordered_eval_scores",
    "pick_first",
    "plan_summary",
    "prefer_inspection_summary_for_generic_monitor",
    "yes_no",
]
