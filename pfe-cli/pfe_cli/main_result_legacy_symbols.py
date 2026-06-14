"""Legacy result formatting symbols for the main compatibility namespace."""

from __future__ import annotations

from typing import Any

from .legacy_result_formatting import (
    format_adapter_export_artifact_line,
    format_adapter_snapshot_line,
    format_compare_evaluation,
    format_eval_result_legacy,
    format_export_execution_summary,
    format_export_toolchain_summary,
    format_incremental_context,
    format_job_execution_summary,
    format_real_execution_summary,
    format_recent_training_snapshot,
    format_train_result_legacy,
)
from .main_deps import make_legacy_result_deps
from .main_result_common import call


def make_result_legacy_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _format_train_result_legacy(result: Any, *, workspace: str | None = None) -> str:
        return format_train_result_legacy(result, workspace=workspace, deps=call(symbols, "_legacy_result_deps"))

    def _legacy_result_deps() -> Any:
        return make_legacy_result_deps(symbols)

    def _format_adapter_snapshot_line(
        label: str,
        snapshot: Any,
        *,
        include_latest: bool = False,
    ) -> str | None:
        return format_adapter_snapshot_line(
            label,
            snapshot,
            include_latest=include_latest,
            deps=call(symbols, "_legacy_result_deps"),
        )

    def _format_adapter_export_artifact_line(label: str, snapshot: Any) -> str | None:
        return format_adapter_export_artifact_line(label, snapshot, deps=call(symbols, "_legacy_result_deps"))

    def _format_job_execution_summary(job_execution: Any) -> str | None:
        return format_job_execution_summary(job_execution, deps=call(symbols, "_legacy_result_deps"))

    def _format_real_execution_summary(job_execution: Any, *, executor_mode: str | None = None) -> str | None:
        return format_real_execution_summary(
            job_execution,
            executor_mode=executor_mode,
            deps=call(symbols, "_legacy_result_deps"),
        )

    def _format_export_execution_summary(export_execution: Any) -> str | None:
        return format_export_execution_summary(export_execution, deps=call(symbols, "_legacy_result_deps"))

    def _format_export_toolchain_summary(export_execution: Any) -> str | None:
        return format_export_toolchain_summary(export_execution, deps=call(symbols, "_legacy_result_deps"))

    def _format_incremental_context(context: Any) -> str | None:
        return format_incremental_context(context, deps=call(symbols, "_legacy_result_deps"))

    def _format_compare_evaluation(compare_evaluation: Any) -> str | None:
        return format_compare_evaluation(compare_evaluation, deps=call(symbols, "_legacy_result_deps"))

    def _format_recent_training_snapshot(snapshot: Any) -> list[str] | None:
        return format_recent_training_snapshot(snapshot, deps=call(symbols, "_legacy_result_deps"))

    def _format_eval_result_legacy(result: Any, *, workspace: str | None = None) -> str:
        return format_eval_result_legacy(result, workspace=workspace, deps=call(symbols, "_legacy_result_deps"))

    return {
        "_format_train_result_legacy": _format_train_result_legacy,
        "_legacy_result_deps": _legacy_result_deps,
        "_format_adapter_snapshot_line": _format_adapter_snapshot_line,
        "_format_adapter_export_artifact_line": _format_adapter_export_artifact_line,
        "_format_job_execution_summary": _format_job_execution_summary,
        "_format_real_execution_summary": _format_real_execution_summary,
        "_format_export_execution_summary": _format_export_execution_summary,
        "_format_export_toolchain_summary": _format_export_toolchain_summary,
        "_format_incremental_context": _format_incremental_context,
        "_format_compare_evaluation": _format_compare_evaluation,
        "_format_recent_training_snapshot": _format_recent_training_snapshot,
        "_format_eval_result_legacy": _format_eval_result_legacy,
    }


__all__ = ["make_result_legacy_symbols"]
