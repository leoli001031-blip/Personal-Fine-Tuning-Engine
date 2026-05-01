"""Operations console digest renderer."""

from __future__ import annotations

from typing import Any

from .operations_console_digest_builder import build_operations_console_digest
from .operations_console_digest_render_sections import append_console_digest_lines
from .operations_formatting_deps import OperationsFormattingDeps


def format_operations_console_digest(
    result: Any,
    *,
    deps: OperationsFormattingDeps,
) -> list[str] | None:
    mapping = deps.coerce_mapping(result)
    if mapping is None:
        return None

    operations_console_mapping = deps.coerce_mapping(mapping.pop("operations_console", None))
    daemon_timeline = deps.coerce_mapping(mapping.get("daemon_timeline"))
    runner_timeline = deps.coerce_mapping(mapping.get("runner_timeline"))
    if daemon_timeline is None and operations_console_mapping is not None:
        daemon_timeline = deps.coerce_mapping(operations_console_mapping.get("daemon_timeline"))
    if runner_timeline is None and operations_console_mapping is not None:
        runner_timeline = deps.coerce_mapping(operations_console_mapping.get("runner_timeline"))
    console = build_operations_console_digest(
        operations_console=(
            {**operations_console_mapping, "daemon_timeline": daemon_timeline, "runner_timeline": runner_timeline}
            if operations_console_mapping is not None and (daemon_timeline is not None or runner_timeline is not None)
            else operations_console_mapping
        ),
        operations_overview=deps.coerce_mapping(mapping.get("operations_overview")),
        operations_dashboard=deps.coerce_mapping(mapping.get("operations_dashboard")),
        operations_alert_policy=deps.coerce_mapping(mapping.get("operations_alert_policy")),
        candidate_summary=deps.coerce_mapping(mapping.get("candidate_summary")),
        candidate_history=deps.coerce_mapping(mapping.get("candidate_history")),
        candidate_timeline=deps.coerce_mapping(mapping.get("candidate_timeline")),
        daemon_timeline=daemon_timeline,
        runner_timeline=runner_timeline,
        train_queue=deps.coerce_mapping(mapping.get("train_queue")),
        deps=deps,
    )
    if console is None:
        return None

    lines = ["operations console digest:"]
    append_console_digest_lines(lines, console, deps=deps)

    return lines


__all__ = ["format_operations_console_digest"]
