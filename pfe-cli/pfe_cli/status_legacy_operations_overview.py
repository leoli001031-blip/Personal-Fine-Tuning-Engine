"""Legacy operations overview status formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_operations_overview_helpers import append_auto_train_blocker, operations_overview_parts


def append_legacy_operations_overview_lines(
    lines: list[str],
    *,
    operations_overview: Mapping[str, Any] | None,
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    deps: Any,
) -> None:
    if operations_overview is None:
        return

    overview_parts = operations_overview_parts(
        operations_overview,
        operations_dashboard=operations_dashboard,
        operations_alert_policy=operations_alert_policy,
        deps=deps,
    )
    if overview_parts:
        lines.append("operations overview: " + " | ".join(overview_parts))
    append_auto_train_blocker(lines, operations_overview, deps=deps)


__all__ = ["append_legacy_operations_overview_lines"]
