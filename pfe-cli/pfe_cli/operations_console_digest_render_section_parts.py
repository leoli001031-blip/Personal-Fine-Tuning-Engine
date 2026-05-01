"""Render individual operations console digest section parts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_formatting_deps import OperationsFormattingDeps
from .operations_console_digest_section_primary import render_primary_section_parts


def render_console_section_parts(
    label: str,
    section: Mapping[str, Any],
    *,
    console: Mapping[str, Any],
    deps: OperationsFormattingDeps,
) -> list[str]:
    """Render key/value parts for one operations console digest section."""
    section_parts = render_primary_section_parts(label, section, deps=deps)

    daemon_timeline = deps.coerce_mapping(console.get("daemon_timeline"))
    runner_timeline = deps.coerce_mapping(console.get("runner_timeline"))
    if label == "daemon" and daemon_timeline:
        for key in ("count", "last_event", "last_reason", "latest_timestamp"):
            value = daemon_timeline.get(key)
            if value is not None:
                section_parts.append(f"{key}={deps.format_scalar(value)}")
    if label == "runner_timeline" and runner_timeline:
        for key in (
            "count",
            "last_event",
            "last_reason",
            "takeover_event_count",
            "last_takeover_event",
            "last_takeover_reason",
            "recent_anomaly_reason",
            "latest_timestamp",
        ):
            value = runner_timeline.get(key)
            rendered = f"{key}={deps.format_scalar(value)}" if value is not None else None
            if rendered is not None and rendered not in section_parts:
                section_parts.append(rendered)
    return section_parts


__all__ = ["render_console_section_parts"]
