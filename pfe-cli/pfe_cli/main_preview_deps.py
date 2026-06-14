"""Dependency builders for serve and training preview compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .main_deps_common import symbol
from .serve_formatting import ServeFormattingDeps
from .training_preview_formatting import TrainingPreviewDeps


def make_serve_formatting_deps(symbols: Mapping[str, Any]) -> ServeFormattingDeps:
    return ServeFormattingDeps(
        build_plan_snapshots=symbol(symbols, "_build_plan_snapshots"),
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        format_adapter_snapshot_line=symbol(symbols, "_format_adapter_snapshot_line"),
        format_backend_dispatch=symbol(symbols, "_format_backend_dispatch"),
        format_export_write=symbol(symbols, "_format_export_write"),
        format_recent_training_snapshot=symbol(symbols, "_format_recent_training_snapshot"),
        format_scalar=symbol(symbols, "_format_scalar"),
        format_status_legacy=symbol(symbols, "_format_status_legacy"),
        format_trainer_summary=symbol(symbols, "_format_trainer_summary"),
        lookup_recent_adapter_snapshot=symbol(symbols, "_lookup_recent_adapter_snapshot"),
        optional_module_call=symbol(symbols, "_optional_module_call"),
        read_cli_state=symbol(symbols, "_read_cli_state"),
        lookup_adapter_snapshot=symbol(symbols, "_lookup_adapter_snapshot"),
    )


def make_training_preview_deps(symbols: Mapping[str, Any]) -> TrainingPreviewDeps:
    return TrainingPreviewDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        format_adapter_snapshot_line=symbol(symbols, "_format_adapter_snapshot_line"),
        format_backend_dispatch=symbol(symbols, "_format_backend_dispatch"),
        format_export_write=symbol(symbols, "_format_export_write"),
        format_scalar=symbol(symbols, "_format_scalar"),
        format_trainer_summary=symbol(symbols, "_format_trainer_summary"),
        lookup_adapter_snapshot=symbol(symbols, "_lookup_adapter_snapshot"),
        optional_module_call=symbol(symbols, "_optional_module_call"),
        pick_first=symbol(symbols, "_pick_first"),
    )
