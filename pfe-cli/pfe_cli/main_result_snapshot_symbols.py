"""Adapter snapshot symbols for the main compatibility namespace."""

from __future__ import annotations

from typing import Any

from .adapter_snapshot_helpers import lookup_adapter_snapshot, lookup_recent_adapter_snapshot
from .main_deps import make_adapter_snapshot_deps
from .main_result_common import call


def make_result_snapshot_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _adapter_snapshot_deps() -> Any:
        return make_adapter_snapshot_deps(symbols)

    def _lookup_adapter_snapshot(version: str | None, *, workspace: str | None = None) -> dict[str, Any] | None:
        return lookup_adapter_snapshot(version, workspace=workspace, deps=call(symbols, "_adapter_snapshot_deps"))

    def _lookup_recent_adapter_snapshot(*, workspace: str | None = None) -> dict[str, Any] | None:
        return lookup_recent_adapter_snapshot(workspace=workspace, deps=call(symbols, "_adapter_snapshot_deps"))

    return {
        "_adapter_snapshot_deps": _adapter_snapshot_deps,
        "_lookup_adapter_snapshot": _lookup_adapter_snapshot,
        "_lookup_recent_adapter_snapshot": _lookup_recent_adapter_snapshot,
    }


__all__ = ["make_result_snapshot_symbols"]
