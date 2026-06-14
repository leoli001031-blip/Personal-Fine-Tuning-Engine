"""Compact console status text rendering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_surface_deps import ConsoleSurfaceDeps
from .console_surface_focus import console_dashboard_focus


def console_status_compact_text(
    payload: Mapping[str, Any],
    *,
    workspace: str | None = None,
    deps: ConsoleSurfaceDeps,
) -> str:
    mapping = deps.coerce_mapping(payload) or {}
    latest_adapter = deps.coerce_mapping(mapping.get("latest_adapter")) or {}
    operations_overview = deps.coerce_mapping(mapping.get("operations_overview")) or {}
    operations_console = deps.coerce_mapping(mapping.get("operations_console")) or {}
    operations_dashboard = deps.coerce_mapping(mapping.get("operations_dashboard")) or {}
    operations_alert_policy = deps.coerce_mapping(mapping.get("operations_alert_policy")) or {}
    train_queue = deps.coerce_mapping(mapping.get("train_queue")) or {}
    resolved_focus = console_dashboard_focus(mapping, deps=deps)
    resolved_action = (
        operations_alert_policy.get("required_action")
        or operations_dashboard.get("required_action")
        or operations_console.get("required_action")
        or operations_overview.get("required_action")
        or "observe_and_monitor"
    )
    summary_line = (
        operations_overview.get("summary_line")
        or operations_console.get("summary_line")
        or operations_dashboard.get("summary_line")
        or operations_alert_policy.get("summary_line")
    )
    inspection_summary_line = (
        operations_overview.get("inspection_summary_line")
        or operations_console.get("inspection_summary_line")
        or operations_dashboard.get("inspection_summary_line")
        or operations_alert_policy.get("inspection_summary_line")
    )
    summary_line, _inspection_summary_line = deps.prefer_inspection_summary_for_generic_monitor(
        focus=resolved_focus,
        summary_line=summary_line,
        inspection_summary_line=inspection_summary_line,
    )

    parts: list[str] = [
        f"workspace={deps.format_scalar(workspace or mapping.get('workspace') or 'user_default')}",
        f"latest={deps.format_scalar(latest_adapter.get('version') or 'none')}",
        f"severity={deps.format_scalar(operations_dashboard.get('severity') or 'stable')}",
        f"focus={deps.format_scalar(resolved_focus)}",
        f"action={deps.format_scalar(resolved_action)}",
        f"queue={deps.format_scalar(train_queue.get('count', 0))}",
    ]
    if summary_line:
        parts.append(f"summary={deps.format_scalar(summary_line)}")
    return "\n".join(["PFE status compact", "summary: " + " | ".join(parts)])


__all__ = ["console_status_compact_text"]
