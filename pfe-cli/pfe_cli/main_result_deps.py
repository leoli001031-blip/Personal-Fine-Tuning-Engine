"""Dependency builders for result, adapter snapshot, and doctor compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .adapter_snapshot_helpers import AdapterSnapshotDeps
from .doctor_formatting import DoctorFormattingDeps
from .legacy_result_formatting import LegacyResultFormattingDeps
from .main_deps_common import call, symbol


def make_adapter_snapshot_deps(symbols: Mapping[str, Any]) -> AdapterSnapshotDeps:
    return AdapterSnapshotDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        optional_module_call=symbol(symbols, "_optional_module_call"),
        pick_first=symbol(symbols, "_pick_first"),
    )


def make_legacy_result_deps(symbols: Mapping[str, Any]) -> LegacyResultFormattingDeps:
    return LegacyResultFormattingDeps(
        coerce_mapping=symbol(symbols, "_coerce_mapping"),
        format_backend_dispatch=symbol(symbols, "_format_backend_dispatch"),
        format_export_write=symbol(symbols, "_format_export_write"),
        format_scalar=symbol(symbols, "_format_scalar"),
        lookup_adapter_snapshot=lambda version, *, workspace=None: call(
            symbols,
            "_lookup_adapter_snapshot",
            version,
            workspace=workspace,
        ),
        ordered_eval_scores=symbol(symbols, "_ordered_eval_scores"),
        pick_first=symbol(symbols, "_pick_first"),
    )


def make_doctor_formatting_deps(symbols: Mapping[str, Any]) -> DoctorFormattingDeps:
    return DoctorFormattingDeps(
        coerce_mapping=lambda result: call(symbols, "_coerce_mapping", result),
        format_scalar=lambda value: call(symbols, "_format_scalar", value),
        pick_first=lambda mapping, *keys: call(symbols, "_pick_first", mapping, *keys),
        load_latest_adapter_manifest=lambda workspace: call(symbols, "_load_latest_adapter_manifest", workspace),
        optional_module_call=lambda module_name, attr_name, *args, **kwargs: call(
            symbols,
            "_optional_module_call",
            module_name,
            attr_name,
            *args,
            **kwargs,
        ),
        pfe_home=lambda workspace: call(symbols, "_pfe_home", workspace),
        lookup_adapter_snapshot=lambda version, *, workspace=None: call(
            symbols,
            "_lookup_adapter_snapshot",
            version,
            workspace=workspace,
        ),
        lookup_recent_adapter_snapshot=lambda *, workspace=None: call(
            symbols,
            "_lookup_recent_adapter_snapshot",
            workspace=workspace,
        ),
        format_adapter_export_artifact_line=lambda label, snapshot: call(
            symbols,
            "_format_adapter_export_artifact_line",
            label,
            snapshot,
        ),
    )
