"""Console focus helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_surface_deps import ConsoleSurfaceDeps


def console_dashboard_focus(payload: Mapping[str, Any] | None = None, *, deps: ConsoleSurfaceDeps) -> str:
    operations_dashboard = deps.coerce_mapping((payload or {}).get("operations_dashboard")) or {}
    operations_console = deps.coerce_mapping((payload or {}).get("operations_console")) or {}
    operations_overview = deps.coerce_mapping((payload or {}).get("operations_overview")) or {}
    current_focus = str(operations_dashboard.get("current_focus") or "").strip().lower()
    if current_focus not in {"", "none", "idle", "stable"}:
        return current_focus
    for raw_focus in (
        operations_dashboard.get("monitor_focus"),
        operations_console.get("monitor_focus"),
        operations_overview.get("monitor_focus"),
    ):
        monitor_focus = str(raw_focus or "").strip().lower()
        if monitor_focus:
            return monitor_focus
    return current_focus or "none"


__all__ = ["console_dashboard_focus"]
