"""Payload-derived command guidance rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_app_action_guidance import _action_command_guidance
from .console_app_data import _mapping, _payload_focus
from .console_app_focus_guidance import _focus_command_guidance
from .console_app_payload_guidance_rules import (
    alert_policy_actions,
    candidate_lifecycle_guidance,
    payload_summary_guidance,
    secondary_action_guidance_labels,
)


def _payload_command_guidance(payload: Mapping[str, Any], focus: str | None = None) -> tuple[str, str]:
    normalized = (focus or _payload_focus(payload) or "none").strip().lower()
    operations_dashboard = _mapping(payload.get("operations_dashboard"))
    operations_console = _mapping(payload.get("operations_console"))
    alert_policy = _mapping(payload.get("operations_alert_policy"))
    required_action, secondary_action_values = alert_policy_actions(alert_policy)
    required_guidance = _action_command_guidance(required_action)
    if required_guidance is not None:
        secondary_guidance_values = secondary_action_guidance_labels(secondary_action_values)
        if secondary_guidance_values:
            return (required_guidance[0], " ".join(secondary_guidance_values))
        return required_guidance

    summary_guidance = payload_summary_guidance(
        normalized_focus=normalized,
        operations_dashboard=operations_dashboard,
        operations_console=operations_console,
    )
    if summary_guidance is not None:
        return summary_guidance

    lifecycle_guidance = candidate_lifecycle_guidance(payload, normalized)
    if lifecycle_guidance is not None:
        return lifecycle_guidance
    return _focus_command_guidance(focus)


__all__ = ["_payload_command_guidance"]
