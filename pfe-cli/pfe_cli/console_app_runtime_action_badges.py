"""Runtime action badge helpers for console rendering."""

from __future__ import annotations

from rich.text import Text

from .console_app_badge_basic import _action_badge
from .console_app_runtime_state_badges import _runtime_state_badge


def _runtime_focus_badges(*, focus: str | None, action: str | None, priority: str | None) -> list[Text]:
    normalized_focus = (focus or "").strip().lower()
    badges: list[Text] = []
    if not normalized_focus:
        return badges
    if normalized_focus in {"daemon_stale", "daemon_blocked"}:
        badges.append(_runtime_state_badge("health", normalized_focus.removeprefix("daemon_")))
    elif normalized_focus.startswith("daemon_restart_"):
        badges.append(_runtime_state_badge("restart", normalized_focus.removeprefix("daemon_restart_")))
    elif normalized_focus.startswith("daemon_heartbeat_"):
        badges.append(_runtime_state_badge("heartbeat", normalized_focus.removeprefix("daemon_heartbeat_")))
    elif normalized_focus.startswith("daemon_lease_"):
        badges.append(_runtime_state_badge("lease", normalized_focus.removeprefix("daemon_lease_")))
    elif normalized_focus == "stale_runner_lock":
        badges.append(_runtime_state_badge("runner", "stale"))
    elif normalized_focus == "runner_stop_requested":
        badges.append(_runtime_state_badge("runner", "active"))
    if action and action not in {"observe_and_monitor", "none"}:
        badges.append(_action_badge(action, priority=priority))
    return badges


def _event_runtime_badges(*, source: str | None, reason: str | None) -> list[Text]:
    normalized_source = (source or "").strip().lower()
    normalized_reason = (reason or "").strip().lower()
    badges: list[Text] = []
    if normalized_reason in {"daemon_stale", "daemon_blocked"}:
        state = "stale" if normalized_reason.endswith("stale") else "blocked"
        badges.append(_runtime_state_badge("health", state))
        badges.append(_action_badge("recover_worker_daemon", priority="p0"))
        return badges
    if normalized_reason.startswith("daemon_heartbeat_"):
        badges.append(_runtime_state_badge("heartbeat", normalized_reason.removeprefix("daemon_heartbeat_")))
        badges.append(_action_badge("inspect_daemon_heartbeat", priority="p1"))
        return badges
    if normalized_reason.startswith("daemon_lease_"):
        lease_state = normalized_reason.removeprefix("daemon_lease_")
        badges.append(_runtime_state_badge("lease", lease_state))
        badges.append(
            _action_badge(
                "recover_worker_daemon" if lease_state == "expired" else "inspect_daemon_heartbeat",
                priority="p1",
            )
        )
        return badges
    if normalized_reason.startswith("daemon_restart_"):
        badges.append(_runtime_state_badge("restart", normalized_reason.removeprefix("daemon_restart_")))
        badges.append(_action_badge("inspect_daemon_restart_policy", priority="p1"))
        return badges
    if normalized_reason == "stale_runner_lock":
        badges.append(_runtime_state_badge("runner", "stale"))
        badges.append(_action_badge("inspect_worker_stale_lock", priority="p1"))
        return badges
    if normalized_reason == "runner_stop_requested":
        badges.append(_runtime_state_badge("runner", "active"))
        badges.append(_action_badge("wait_for_runner_shutdown", priority="p2"))
        return badges
    if normalized_source == "daemon":
        if "restart" in normalized_reason:
            badges.append(_action_badge("inspect_daemon_restart_policy", priority="p1"))
        elif "heartbeat" in normalized_reason:
            badges.append(_action_badge("inspect_daemon_heartbeat", priority="p1"))
        elif "lease" in normalized_reason or "recover" in normalized_reason:
            badges.append(_action_badge("recover_worker_daemon", priority="p1"))
    return badges


__all__ = ["_event_runtime_badges", "_runtime_focus_badges"]
