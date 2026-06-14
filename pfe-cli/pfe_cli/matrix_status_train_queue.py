"""Train queue status section for Matrix terminal status output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping, _format_scalar
from .terminal_theme import draw_box, format_key_value


def append_train_queue_status_section(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append train queue status box."""
    train_queue = _coerce_mapping(mapping.get("train_queue"))
    if train_queue:
        q_content = []
        count = train_queue.get("count", 0)
        max_priority = train_queue.get("max_priority")
        q_content.append(format_key_value("count", count))
        if max_priority is not None:
            q_content.append(format_key_value("max priority", max_priority))
        counts = _coerce_mapping(train_queue.get("counts"))
        if counts:
            q_content.append(format_key_value("states", ",".join(f"{n}:{counts.get(n)}" for n in sorted(counts))))
        current = _coerce_mapping(train_queue.get("current"))
        if current:
            q_content.append(format_key_value("current", f"{current.get('job_id','')} | {current.get('state','')}"))
        last_item = _coerce_mapping(train_queue.get("last_item"))
        if last_item:
            q_content.append(
                format_key_value(
                    "last",
                    f"{last_item.get('job_id','')} | {last_item.get('state','')} | {last_item.get('adapter_version','')}",
                )
            )
        policy_summary = _coerce_mapping(train_queue.get("policy_summary"))
        if policy_summary:
            ps = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in policy_summary.items() if v is not None)
            if ps:
                q_content.append(format_key_value("policy", ps))
        confirmation_summary = _coerce_mapping(train_queue.get("confirmation_summary"))
        if confirmation_summary:
            cs = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in confirmation_summary.items() if v is not None)
            if cs:
                q_content.append(format_key_value("confirmation", cs))
        worker_runner = _coerce_mapping(train_queue.get("worker_runner"))
        if worker_runner:
            wr_keys = ["active", "lock_state", "stop_requested", "processed_count", "failed_count", "loop_cycles", "last_action", "last_event"]
            wr_parts = " | ".join(
                f"{k.replace('_', ' ')}={_format_scalar(worker_runner.get(k))}"
                for k in wr_keys
                if worker_runner.get(k) is not None
            )
            if wr_parts:
                q_content.append(format_key_value("worker runner", wr_parts))
        if q_content:
            lines.append(draw_box("TRAIN QUEUE", q_content))
            lines.append("")


__all__ = ["append_train_queue_status_section"]
