"""Footer digest state extraction for the Rich operations console."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rich.text import Text

from .console_app_badges import _ops_badge, _runtime_focus_badges
from .console_app_data import _compact_text, _dashboard_focus, _mapping, _value
from .console_app_prompt_rules import _prompt_action_guidance, _prompt_trigger_category


@dataclass(frozen=True)
class FooterDigestState:
    full_focus: str
    focus: str
    full_action: str
    action: str
    action_priority: str
    handling: str
    severity: str
    status_badge: Text
    primary_action_hint: str
    secondary_action_hint: str
    runtime_badges: list[Text]
    runtime_mode: bool
    trigger_category: str | None


def build_footer_digest_state(
    payload: Mapping[str, Any],
    *,
    interactive: bool = False,
    mode: str = "chat",
    ops_refresh_state: str | None = None,
) -> FooterDigestState:
    dashboard = _mapping(payload.get("operations_dashboard"))
    alert_policy = _mapping(payload.get("operations_alert_policy"))
    resolved_focus = _dashboard_focus(dashboard)
    full_action = _value(alert_policy, "required_action", "primary_action", default="observe_and_monitor")
    action_priority = _value(alert_policy, "action_priority", default="p2")
    help_hint = _footer_help_hint(interactive=interactive, mode=mode)
    primary_action_hint, secondary_action_hint = _prompt_action_guidance(
        mode,
        focus=resolved_focus,
        shortcut_hint=help_hint,
        payload=payload,
    )
    runtime_badges = _runtime_focus_badges(focus=resolved_focus, action=full_action, priority=action_priority)
    severity = _value(dashboard, "severity", default="stable")
    return FooterDigestState(
        full_focus=resolved_focus,
        focus=_compact_text(resolved_focus, max_len=16),
        full_action=full_action,
        action=_compact_text(full_action, max_len=22),
        action_priority=action_priority,
        handling=_value(alert_policy, "remediation_mode", "escalation_mode", default="monitor"),
        severity=severity,
        status_badge=_ops_badge(ops_refresh_state, severity=severity),
        primary_action_hint=primary_action_hint,
        secondary_action_hint=secondary_action_hint,
        runtime_badges=runtime_badges,
        runtime_mode=bool(runtime_badges),
        trigger_category=_prompt_trigger_category(resolved_focus, payload=payload),
    )


def _footer_help_hint(*, interactive: bool, mode: str) -> str:
    if not interactive:
        help_hint = "console,status,doctor"
    elif mode == "chat":
        help_hint = "Enter,/help,/cmd,/quit"
    else:
        help_hint = "/status,/candidate,/daemon,/chat"
    if "," in help_hint:
        return ",".join(part.strip() for part in help_hint.split(",")[:2])
    return help_hint


__all__ = ["FooterDigestState", "build_footer_digest_state"]
