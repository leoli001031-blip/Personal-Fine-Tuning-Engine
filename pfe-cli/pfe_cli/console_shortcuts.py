"""Console shortcut hint helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_actions_deps import ConsoleActionsDeps
from .console_focus_actions import console_focus_actions


def console_shortcut_hint(
    mode_name: str,
    payload: Mapping[str, Any] | None = None,
    *,
    deps: ConsoleActionsDeps,
) -> str:
    operations_dashboard = deps.coerce_mapping((payload or {}).get("operations_dashboard")) or {}
    attention_needed = bool(operations_dashboard.get("attention_needed"))
    current_focus = deps.console_dashboard_focus(payload)
    if mode_name == "chat":
        return "Enter,/do,/see" if attention_needed else "Enter,/help,^C"

    def _primary_shortcut_token() -> str | None:
        focus_actions = console_focus_actions(payload, deps=deps)
        primary_label = str(focus_actions.get("primary_label") or "").strip()
        if not primary_label.startswith("/"):
            return None
        return primary_label.split(" or ")[0].strip()

    if current_focus.startswith("policy_") or current_focus.startswith("auto_train_policy"):
        return "/do,/see,/policy,/chat"
    if current_focus in {"insufficient_new_signal_samples", "holdout_not_ready"}:
        return "/do,/see,/gate,/chat"
    if current_focus == "cooldown_active":
        return "/do,/see,/trigger,/chat"
    if current_focus == "failure_backoff_active":
        return "/do,/see,/retry,/chat"
    if "pending_review" in current_focus or "awaiting_confirmation" in current_focus:
        return "/do,/see,/approve,/chat"
    if current_focus.startswith("daemon") and "restart" in current_focus:
        return "/do,/see,/alerts,/chat"
    if current_focus.startswith("daemon") and ("heartbeat" in current_focus or "lease" in current_focus):
        return "/do,/see,/daemon,/chat"
    if "runner" in current_focus and "stale" in current_focus:
        return "/do,/see,/runner,/chat"
    if "candidate_ready_for_promotion" in current_focus:
        return "/do,/see,/archive,/chat"
    if current_focus in {"queue_waiting_execution", "queue_backlog"}:
        return "/do,/see,/process,/chat"
    if current_focus in {"candidate_idle", "runner_active", "daemon_active"}:
        primary_shortcut = _primary_shortcut_token()
        if primary_shortcut:
            return f"/do,/see,{primary_shortcut},/chat"
    if current_focus.startswith("daemon"):
        primary_shortcut = _primary_shortcut_token()
        if primary_shortcut == "/runtime":
            return "/do,/see,/runtime,/chat"
        return "/do,/see,/daemon,/chat"
    if current_focus.startswith("candidate"):
        primary_shortcut = _primary_shortcut_token()
        if primary_shortcut:
            return f"/do,/see,{primary_shortcut},/chat"
        candidate_summary = deps.coerce_mapping((payload or {}).get("candidate_summary")) or {}
        if bool(candidate_summary.get("candidate_can_promote")) or bool(candidate_summary.get("candidate_can_archive")):
            return "/do,/see,/archive,/chat"
        return "/do,/see,/candidate,/chat"
    if current_focus.startswith("queue"):
        primary_shortcut = _primary_shortcut_token()
        if primary_shortcut:
            return f"/do,/see,{primary_shortcut},/chat"
        return "/do,/see,/queue,/chat"
    if current_focus.startswith("runner"):
        primary_shortcut = _primary_shortcut_token()
        if primary_shortcut == "/runtime":
            return "/do,/see,/runtime,/chat"
        return "/do,/see,/runner,/chat"
    return "/status,/candidate,/daemon,/chat"


__all__ = ["console_shortcut_hint"]
