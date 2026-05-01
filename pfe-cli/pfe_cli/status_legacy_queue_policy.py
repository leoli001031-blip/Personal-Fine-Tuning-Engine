"""Legacy train queue policy and review formatting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_queue_policy_parts import append_confirmation, append_policy
from .status_legacy_queue_review_parts import append_review, append_review_policy
from .status_legacy_queue_worker_parts import append_worker


def append_legacy_queue_policy_lines(lines: list[str], train_queue: Mapping[str, Any], *, deps: Any) -> None:
    append_policy(lines, train_queue, deps=deps)
    append_confirmation(lines, train_queue, deps=deps)
    append_review(lines, train_queue, deps=deps)
    append_review_policy(lines, train_queue, deps=deps)
    append_worker(lines, train_queue, deps=deps)


__all__ = ["append_legacy_queue_policy_lines"]
