"""Legacy plain-text train queue status formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_queue_daemon import append_legacy_queue_daemon_lines
from .status_legacy_queue_policy import append_legacy_queue_policy_lines
from .status_legacy_queue_summary import append_legacy_queue_summary_lines


def append_legacy_train_queue_lines(
    lines: list[str],
    mapping: dict[str, Any],
    *,
    train_queue: Mapping[str, Any] | None,
    workspace: str | None,
    deps: Any,
) -> None:
    """Append legacy train queue, daemon, and queue history lines."""

    if train_queue is not None:
        append_legacy_queue_summary_lines(lines, train_queue, deps=deps)
        append_legacy_queue_policy_lines(lines, train_queue, deps=deps)
    append_legacy_queue_daemon_lines(lines, mapping, train_queue=train_queue, workspace=workspace, deps=deps)


__all__ = ["append_legacy_train_queue_lines"]
