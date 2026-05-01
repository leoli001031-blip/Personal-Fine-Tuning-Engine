"""Worker runner lines for legacy train queue status."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .status_legacy_queue_helpers import append_scalar_parts


def append_worker(lines: list[str], train_queue: Mapping[str, Any], *, deps: Any) -> None:
    worker_runner = deps.coerce_mapping(train_queue.get("worker_runner"))
    if not worker_runner:
        return
    worker_parts: list[str] = []
    append_scalar_parts(
        worker_parts,
        worker_runner,
        (
            "active",
            "lock_state",
            "stop_requested",
            "processed_count",
            "failed_count",
            "loop_cycles",
            "stopped_reason",
            "max_seconds",
            "idle_sleep_seconds",
            "stale_after_seconds",
            "lease_expires_at",
        ),
        deps=deps,
    )
    if worker_parts:
        lines.append("queue worker runner: " + " | ".join(worker_parts))


__all__ = ["append_worker"]
