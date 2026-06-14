"""Focus and action resolution for operations alert surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_formatting_deps import resolved_first_focus


def surface_focus(
    *,
    alert_policy: Mapping[str, Any],
    console: Mapping[str, Any],
    dashboard: Mapping[str, Any],
    overview: Mapping[str, Any],
) -> Any:
    return resolved_first_focus(
        overview.get("current_focus"),
        overview.get("monitor_focus"),
        dashboard.get("current_focus"),
        dashboard.get("monitor_focus"),
        alert_policy.get("current_focus"),
        console.get("current_focus"),
        overview.get("attention_reason"),
        overview.get("monitor_focus"),
    )


def surface_action(
    *,
    alert_policy: Mapping[str, Any],
    console: Mapping[str, Any],
    dashboard: Mapping[str, Any],
    overview: Mapping[str, Any],
) -> Any:
    return (
        overview.get("required_action")
        or alert_policy.get("required_action")
        or dashboard.get("required_action")
        or console.get("required_action")
    )


__all__ = ["surface_action", "surface_focus"]
