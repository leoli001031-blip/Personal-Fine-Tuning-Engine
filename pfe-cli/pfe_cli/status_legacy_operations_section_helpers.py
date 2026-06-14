"""Helpers for legacy operations detail surface sections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def operations_alert_policy_lines(
    *,
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    deps: Any,
) -> list[str] | None:
    operations_alert_policy_for_display = deps.coerce_mapping(operations_alert_policy)
    dashboard_monitor_focus = (
        deps.coerce_mapping(operations_dashboard).get("monitor_focus") if operations_dashboard is not None else None
    )
    if operations_alert_policy_for_display is not None:
        policy_focus = str(operations_alert_policy_for_display.get("current_focus") or "").strip().lower()
        if policy_focus in {"", "none", "idle", "stable"} and dashboard_monitor_focus is not None:
            operations_alert_policy_for_display["current_focus"] = dashboard_monitor_focus
    return (
        deps.format_operations_alert_policy(operations_alert_policy_for_display)
        if operations_alert_policy_for_display is not None
        else None
    )


def append_lines(lines: list[str], extra_lines: list[str] | None) -> None:
    if extra_lines is not None:
        lines.extend(extra_lines)


def append_split_lines(lines: list[str], text: str | None) -> None:
    if text is not None:
        lines.extend(text.splitlines())


__all__ = ["append_lines", "append_split_lines", "operations_alert_policy_lines"]
