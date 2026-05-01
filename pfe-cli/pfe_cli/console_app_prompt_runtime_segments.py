"""Runtime segment rendering for the Rich console prompt panel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.text import Text

from .console_app_badges import _ops_badge, _runtime_focus_badges
from .console_app_data import _mapping, _value


def append_prompt_runtime_segments(
    bar: Text,
    *,
    ops_refresh_state: str | None,
    focus: str | None,
    payload: Mapping[str, Any] | None,
) -> None:
    if ops_refresh_state:
        bar.append(" ", style="dim")
        bar.append("o=", style="dim")
        bar.append_text(_ops_badge(ops_refresh_state))
    alert_policy = _mapping((payload or {}).get("operations_alert_policy"))
    runtime_badges = _runtime_focus_badges(
        focus=focus,
        action=_value(alert_policy, "required_action", "primary_action", default="observe_and_monitor"),
        priority=_value(alert_policy, "action_priority", default="p2"),
    )
    for badge in runtime_badges[:2]:
        bar.append(" ", style="dim")
        bar.append_text(badge)


__all__ = ["append_prompt_runtime_segments"]
