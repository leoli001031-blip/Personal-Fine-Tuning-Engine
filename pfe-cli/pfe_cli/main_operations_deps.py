"""Dependency builders for operations and daemon compatibility formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .daemon_formatting import DaemonFormattingDeps
from .main_deps_common import symbol
from .operations_formatting import OperationsFormattingDeps
from .operations_history_formatting import OperationsHistoryFormattingDeps


def make_operations_history_formatting_deps(symbols: Mapping[str, Any]) -> OperationsHistoryFormattingDeps:
    return OperationsHistoryFormattingDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        format_scalar=symbol(symbols, "_format_scalar"),
    )


def make_operations_formatting_deps(symbols: Mapping[str, Any]) -> OperationsFormattingDeps:
    return OperationsFormattingDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        coerce_sequence_of_mappings=symbol(symbols, "_coerce_sequence_of_mappings"),
        coerce_sequence_of_scalars=symbol(symbols, "_coerce_sequence_of_scalars"),
        format_scalar=symbol(symbols, "_format_scalar"),
        prefer_inspection_summary_for_generic_monitor=symbol(
            symbols,
            "_prefer_inspection_summary_for_generic_monitor",
        ),
        generic_monitor_focuses=frozenset(symbol(symbols, "_GENERIC_MONITOR_FOCUSES")),
    )


def make_daemon_formatting_deps(symbols: Mapping[str, Any]) -> DaemonFormattingDeps:
    return DaemonFormattingDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        format_scalar=symbol(symbols, "_format_scalar"),
    )
