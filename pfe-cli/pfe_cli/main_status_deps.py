"""Dependency builders for status compatibility formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .main_deps_common import symbol
from .plan_snapshot_helpers import PlanSnapshotDeps
from .status_formatting import StatusFormattingDeps
from .status_legacy_formatting import StatusLegacyFormattingDeps


def make_plan_snapshot_deps(symbols: Mapping[str, Any]) -> PlanSnapshotDeps:
    return PlanSnapshotDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        format_plan_block=symbol(symbols, "_format_plan_block"),
        load_latest_adapter_manifest=symbol(symbols, "_load_latest_adapter_manifest"),
        optional_module_call=symbol(symbols, "_optional_module_call"),
    )


def make_status_formatting_deps(symbols: Mapping[str, Any]) -> StatusFormattingDeps:
    return StatusFormattingDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        read_cli_state=symbol(symbols, "_read_cli_state"),
    )


def make_status_legacy_formatting_deps(symbols: Mapping[str, Any]) -> StatusLegacyFormattingDeps:
    return StatusLegacyFormattingDeps(
        build_plan_snapshots=symbol(symbols, "_build_plan_snapshots"),
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        coerce_sequence_of_mappings=symbol(symbols, "_coerce_sequence_of_mappings"),
        coerce_sequence_of_scalars=symbol(symbols, "_coerce_sequence_of_scalars"),
        format_adapter_export_artifact_line=symbol(symbols, "_format_adapter_export_artifact_line"),
        format_backend_dispatch=symbol(symbols, "_format_backend_dispatch"),
        format_compare_evaluation=symbol(symbols, "_format_compare_evaluation"),
        format_daemon_timeline_summary=symbol(symbols, "_format_daemon_timeline_summary"),
        format_export_write=symbol(symbols, "_format_export_write"),
        format_operations_alert_policy=symbol(symbols, "_format_operations_alert_policy"),
        format_operations_alert_surface=symbol(symbols, "_format_operations_alert_surface"),
        format_operations_console_digest=symbol(symbols, "_format_operations_console_digest"),
        format_operations_dashboard=symbol(symbols, "_format_operations_dashboard"),
        format_operations_event_stream=symbol(symbols, "_format_operations_event_stream"),
        format_operations_timeline=symbol(symbols, "_format_operations_timeline"),
        format_ops_attention=symbol(symbols, "_format_ops_attention"),
        format_recent_training_snapshot=symbol(symbols, "_format_recent_training_snapshot"),
        format_runner_timeline_summary=symbol(symbols, "_format_runner_timeline_summary"),
        format_scalar=symbol(symbols, "_format_scalar"),
        format_trainer_summary=symbol(symbols, "_format_trainer_summary"),
        pick_first=symbol(symbols, "_pick_first"),
        prefer_inspection_summary_for_generic_monitor=symbol(
            symbols,
            "_prefer_inspection_summary_for_generic_monitor",
        ),
        read_cli_state=symbol(symbols, "_read_cli_state"),
        read_train_queue_daemon_state=symbol(symbols, "_read_train_queue_daemon_state"),
    )
