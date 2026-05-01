"""Top-level helpers for legacy status output formatting."""

from __future__ import annotations

from typing import Any

from .status_legacy_deps import StatusLegacyFormattingDeps
from .status_legacy_sections import LegacyStatusSections


HEADLINE_KEYS = (
    "home",
    "strict_local",
    "provider",
    "signal_count",
    "adapter_versions",
    "workspace",
)


def append_status_headlines(
    lines: list[str],
    mapping: dict[str, Any],
    *,
    deps: StatusLegacyFormattingDeps,
) -> None:
    for key in HEADLINE_KEYS:
        if key in mapping:
            lines.append(f"{key.replace('_', ' ')}: {deps.format_scalar(mapping.pop(key))}")


def append_ops_attention_line(
    lines: list[str],
    *,
    sections: LegacyStatusSections,
    deps: StatusLegacyFormattingDeps,
) -> None:
    ops_attention = deps.format_ops_attention(
        operations_alerts=sections.operations_alerts,
        operations_overview=sections.operations_overview,
        operations_dashboard=sections.operations_dashboard,
        operations_alert_policy=sections.operations_alert_policy,
        candidate_summary=sections.candidate_summary,
        train_queue=sections.train_queue,
        latest_adapter_map=sections.latest_adapter_map,
        recent_adapter_map=sections.recent_adapter_map,
    )
    if ops_attention is not None:
        lines.append(ops_attention)


__all__ = ["append_ops_attention_line", "append_status_headlines"]
