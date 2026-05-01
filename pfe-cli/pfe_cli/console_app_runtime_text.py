"""Runtime status text helpers for console rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.text import Text

from .console_app_data import _compact_text, _value, _yes_no
from .console_app_runtime_state_badges import _runtime_state_badge


def _runtime_stability_text(runtime_stability: Mapping[str, Any], *, severity: str | None = None) -> Text:
    runner_state = _value(
        runtime_stability,
        "runner_lock_state",
        default=("active" if runtime_stability.get("runner_active") else "idle"),
    )
    health_state = _value(runtime_stability, "daemon_health_state", default="stopped")
    heartbeat_state = _value(runtime_stability, "daemon_heartbeat_state", default="idle")
    lease_state = _value(runtime_stability, "daemon_lease_state", default="idle")
    restart_state = _value(runtime_stability, "daemon_restart_policy_state", default="ready")
    recovery_state = _value(runtime_stability, "daemon_recovery_action", default="none")
    text = Text()
    text.append("R ", style="bold")
    text.append_text(_runtime_state_badge("runner", runner_state))
    text.append(" ", style="dim")
    text.append("D ", style="bold")
    text.append_text(_runtime_state_badge("health", health_state))
    if heartbeat_state not in {"idle", "fresh", "n/a"}:
        text.append(" hb ", style="dim")
        text.append_text(_runtime_state_badge("heartbeat", heartbeat_state))
    if lease_state not in {"idle", "valid", "n/a"}:
        text.append(" lease ", style="dim")
        text.append_text(_runtime_state_badge("lease", lease_state))
    if restart_state not in {"ready", "n/a"}:
        text.append(" rs ", style="dim")
        text.append_text(_runtime_state_badge("restart", restart_state))
    if recovery_state not in {"none", "n/a"}:
        text.append(" rec ", style="dim")
        text.append_text(_runtime_state_badge("recover", recovery_state))
    return text


def _handle_text(bits: Sequence[str], alert_policy: Mapping[str, Any]) -> Text:
    text = Text()
    if bits:
        text.append(_compact_text(bits[0], max_len=6), style="bold")
    if len(bits) > 1:
        text.append(" ", style="dim")
        text.append(_compact_text(bits[1], max_len=10), style="white")
    text.append(" ", style="dim")
    text.append(f"h={_yes_no(alert_policy.get('requires_human_review', False))}", style="dim")
    text.append(" ", style="dim")
    text.append(f"a={_yes_no(alert_policy.get('auto_remediation_allowed', False))}", style="dim")
    text.append(" ", style="dim")
    text.append(f"now={_yes_no(alert_policy.get('requires_immediate_action', False))}", style="dim")
    return text


__all__ = ["_handle_text", "_runtime_stability_text"]
