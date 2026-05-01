"""Legacy train/status/eval summary formatting helpers."""

from __future__ import annotations

from .legacy_adapter_result_formatting import (
    format_adapter_export_artifact_line,
    format_adapter_snapshot_line,
)
from .legacy_context_result_formatting import (
    format_compare_evaluation,
    format_incremental_context,
    format_recent_training_snapshot,
)
from .legacy_eval_result_formatting import format_eval_result_legacy
from .legacy_execution_result_formatting import (
    format_export_execution_summary,
    format_export_toolchain_summary,
    format_job_execution_summary,
    format_real_execution_summary,
)
from .legacy_result_common import format_bytes_compact
from .legacy_result_deps import LegacyResultFormattingDeps
from .legacy_train_result_formatting import format_train_result_legacy

__all__ = [
    "LegacyResultFormattingDeps",
    "format_adapter_export_artifact_line",
    "format_adapter_snapshot_line",
    "format_bytes_compact",
    "format_compare_evaluation",
    "format_eval_result_legacy",
    "format_export_execution_summary",
    "format_export_toolchain_summary",
    "format_incremental_context",
    "format_job_execution_summary",
    "format_real_execution_summary",
    "format_recent_training_snapshot",
    "format_train_result_legacy",
]
