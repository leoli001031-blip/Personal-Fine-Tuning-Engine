"""Compatibility exports for console slash-command summary text helpers."""

from __future__ import annotations

from .console_routing_candidate_summaries import console_candidate_summary_text
from .console_routing_runtime_summaries import (
    console_gate_summary_text,
    console_runtime_summary_text,
    console_trigger_summary_text,
)
from .console_routing_worker_summaries import (
    console_daemon_summary_text,
    console_queue_summary_text,
    console_runner_summary_text,
)

__all__ = [
    "console_candidate_summary_text",
    "console_daemon_summary_text",
    "console_gate_summary_text",
    "console_queue_summary_text",
    "console_runner_summary_text",
    "console_runtime_summary_text",
    "console_trigger_summary_text",
]
