"""Plan snapshot symbols for the main compatibility namespace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .main_deps import make_plan_snapshot_deps
from .main_status_common import call
from .plan_snapshot_helpers import build_plan_snapshots, format_plan_snapshot_lines


def make_status_plan_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _plan_snapshot_deps() -> Any:
        return make_plan_snapshot_deps(symbols)

    def _build_plan_snapshots(
        workspace: str | None,
        status_mapping: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_plan_snapshots(workspace, status_mapping, deps=call(symbols, "_plan_snapshot_deps"))

    def _format_plan_snapshot_lines(plan_snapshots: Mapping[str, Any]) -> list[str]:
        return format_plan_snapshot_lines(plan_snapshots, deps=call(symbols, "_plan_snapshot_deps"))

    return {
        "_plan_snapshot_deps": _plan_snapshot_deps,
        "_build_plan_snapshots": _build_plan_snapshots,
        "_format_plan_snapshot_lines": _format_plan_snapshot_lines,
    }


__all__ = ["make_status_plan_symbols"]
