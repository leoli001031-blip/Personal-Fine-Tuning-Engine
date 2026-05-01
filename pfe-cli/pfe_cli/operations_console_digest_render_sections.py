"""Render operations console digest sections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .operations_console_digest_render_section_parts import render_console_section_parts
from .operations_console_digest_render_summary import append_console_digest_summary_line
from .operations_formatting_deps import OperationsFormattingDeps


def append_console_digest_lines(
    lines: list[str],
    console: Mapping[str, Any],
    *,
    deps: OperationsFormattingDeps,
) -> None:
    """Append rendered operations console digest lines."""
    append_console_digest_summary_line(lines, console, deps=deps)

    for label in ("candidate", "queue", "runner", "runner_timeline", "daemon"):
        section = deps.coerce_mapping(console.get(label))
        if not section:
            continue
        section_parts = render_console_section_parts(label, section, console=console, deps=deps)
        if section_parts:
            lines.append(f"  operations console {label}: " + " | ".join(section_parts))

    timelines = deps.coerce_mapping(console.get("timelines"))
    if timelines:
        timeline_summary = timelines.get("summary_line")
        if timeline_summary:
            lines.append(f"  operations console timelines: {deps.format_scalar(timeline_summary)}")

__all__ = ["append_console_digest_lines"]
