"""Legacy train queue daemon and history formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_queue_helpers import append_scalar_parts, compact_item


def _resolve_daemon_state(
    mapping: dict[str, Any],
    *,
    train_queue: Mapping[str, Any] | None,
    workspace: str | None,
    deps: Any,
) -> dict[str, Any] | None:
    daemon_state = deps.coerce_mapping(mapping.pop("daemon", None))
    if daemon_state is None and isinstance(train_queue, Mapping):
        daemon_state = deps.coerce_mapping(train_queue.get("daemon"))
    if daemon_state is None:
        daemon_state = deps.read_train_queue_daemon_state(workspace)
    return daemon_state


def _append_daemon(lines: list[str], daemon_state: Mapping[str, Any], *, deps: Any) -> None:
    daemon_parts: list[str] = []
    append_scalar_parts(
        daemon_parts,
        daemon_state,
        ("desired_state", "requested_action", "command_status", "active", "observed_state", "lock_state"),
        deps=deps,
    )
    for key in ("state_path", "history_count", "last_requested_at"):
        value = daemon_state.get(key)
        if value is not None:
            daemon_parts.append(f"{key}={deps.format_scalar(value)}")
    if daemon_parts:
        lines.append("queue daemon: " + " | ".join(daemon_parts))


def _append_history(lines: list[str], train_queue: Mapping[str, Any] | None, *, deps: Any) -> None:
    if not isinstance(train_queue, Mapping):
        return
    history_summary = deps.coerce_mapping(train_queue.get("history_summary"))
    if not history_summary:
        return
    history_parts: list[str] = []
    transition_count = history_summary.get("transition_count")
    if transition_count is not None:
        history_parts.append(f"transition_count={deps.format_scalar(transition_count)}")
    last_transition = deps.coerce_mapping(history_summary.get("last_transition"))
    if last_transition:
        transition_text = compact_item(last_transition, ("job_id", "event", "state"), deps=deps)
        if transition_text:
            history_parts.append(f"last_transition={transition_text}")
    last_reason = history_summary.get("last_reason")
    if last_reason is not None:
        history_parts.append(f"last_reason={deps.format_scalar(last_reason)}")
    if history_parts:
        lines.append("queue history: " + " | ".join(history_parts))


def append_legacy_queue_daemon_lines(
    lines: list[str],
    mapping: dict[str, Any],
    *,
    train_queue: Mapping[str, Any] | None,
    workspace: str | None,
    deps: Any,
) -> None:
    daemon_state = _resolve_daemon_state(mapping, train_queue=train_queue, workspace=workspace, deps=deps)
    if daemon_state is None:
        return
    _append_daemon(lines, daemon_state, deps=deps)
    _append_history(lines, train_queue, deps=deps)


__all__ = ["append_legacy_queue_daemon_lines"]
