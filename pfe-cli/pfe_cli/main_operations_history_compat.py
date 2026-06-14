"""Install legacy operations history helpers onto the main module namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .main_deps import make_operations_history_formatting_deps
from .operations_history_formatting import (
    candidate_timeline_stage,
    format_candidate_history,
    format_candidate_timeline,
    format_candidate_timeline_item,
    format_daemon_timeline_summary,
    format_runner_timeline_summary,
    format_train_queue_daemon_history,
    format_train_queue_daemon_status,
    format_train_queue_history,
    format_worker_runner_history,
    history_latest_timestamp,
)


def _call(symbols: Mapping[str, Any], name: str, *args: Any, **kwargs: Any) -> Any:
    return symbols[name](*args, **kwargs)


def install_operations_history_compat(symbols: dict[str, Any]) -> None:
    def _operations_history_formatting_deps() -> Any:
        return make_operations_history_formatting_deps(symbols)

    def _format_candidate_history(result: Any) -> str:
        return format_candidate_history(result, deps=_call(symbols, "_operations_history_formatting_deps"))

    def _candidate_timeline_stage(item: Mapping[str, Any] | None) -> str | None:
        return candidate_timeline_stage(item, deps=_call(symbols, "_operations_history_formatting_deps"))

    def _format_candidate_timeline_item(item: Any, *, index: int) -> str:
        return format_candidate_timeline_item(
            item,
            index=index,
            deps=_call(symbols, "_operations_history_formatting_deps"),
        )

    def _format_candidate_timeline(result: Any) -> str:
        return format_candidate_timeline(result, deps=_call(symbols, "_operations_history_formatting_deps"))

    def _format_train_queue_history(result: Any) -> str:
        return format_train_queue_history(result, deps=_call(symbols, "_operations_history_formatting_deps"))

    def _format_worker_runner_history(result: Any) -> str:
        return format_worker_runner_history(result, deps=_call(symbols, "_operations_history_formatting_deps"))

    def _format_train_queue_daemon_status(result: Any) -> str:
        return format_train_queue_daemon_status(result, deps=_call(symbols, "_operations_history_formatting_deps"))

    def _format_daemon_timeline_summary(result: Any) -> str:
        return format_daemon_timeline_summary(result, deps=_call(symbols, "_operations_history_formatting_deps"))

    def _format_runner_timeline_summary(result: Any) -> str:
        return format_runner_timeline_summary(result, deps=_call(symbols, "_operations_history_formatting_deps"))

    def _format_train_queue_daemon_history(result: Any) -> str:
        return format_train_queue_daemon_history(result, deps=_call(symbols, "_operations_history_formatting_deps"))

    def _history_latest_timestamp(items: Any) -> str | None:
        return history_latest_timestamp(items, deps=_call(symbols, "_operations_history_formatting_deps"))

    symbols.update(
        {
            "_operations_history_formatting_deps": _operations_history_formatting_deps,
            "_format_candidate_history": _format_candidate_history,
            "_candidate_timeline_stage": _candidate_timeline_stage,
            "_format_candidate_timeline_item": _format_candidate_timeline_item,
            "_format_candidate_timeline": _format_candidate_timeline,
            "_format_train_queue_history": _format_train_queue_history,
            "_format_worker_runner_history": _format_worker_runner_history,
            "_format_worker_runner_status": _format_worker_runner_history,
            "_format_train_queue_daemon_status": _format_train_queue_daemon_status,
            "_format_daemon_timeline_summary": _format_daemon_timeline_summary,
            "_format_runner_timeline_summary": _format_runner_timeline_summary,
            "_format_train_queue_daemon_history": _format_train_queue_daemon_history,
            "_history_latest_timestamp": _history_latest_timestamp,
        }
    )


__all__ = ["install_operations_history_compat"]
