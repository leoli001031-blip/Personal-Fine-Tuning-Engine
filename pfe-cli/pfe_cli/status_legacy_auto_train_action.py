"""Legacy auto-train action status formatting."""

from __future__ import annotations

from typing import Any


_ACTION_KEYS = (
    "action",
    "status",
    "reason",
    "enabled",
    "min_new_samples",
    "max_interval_days",
    "queue_mode",
    "require_queue_confirmation",
    "epochs",
    "backend",
    "triggered",
    "queue_job_id",
    "confirmation_reason",
    "approval_reason",
    "rejection_reason",
    "operator_note",
    "processed_count",
    "completed_count",
    "failed_count",
    "limit",
    "max_iterations",
    "max_cycles",
    "loop_cycles",
    "idle_rounds",
    "poll_interval_seconds",
    "remaining_queued",
    "drained",
    "stopped_reason",
    "triggered_version",
    "promoted_version",
)


def append_legacy_auto_train_action_lines(
    lines: list[str],
    mapping: dict[str, Any],
    *,
    deps: Any,
) -> None:
    auto_trigger_action = deps.coerce_mapping(mapping.pop("auto_train_trigger_action", None))
    if auto_trigger_action is None:
        return

    action_parts: list[str] = []
    for key in _ACTION_KEYS:
        value = auto_trigger_action.get(key)
        if value is not None:
            action_parts.append(f"{key}={deps.format_scalar(value)}")
    if action_parts:
        lines.append("auto train action: " + " | ".join(action_parts))


__all__ = ["append_legacy_auto_train_action_lines"]
