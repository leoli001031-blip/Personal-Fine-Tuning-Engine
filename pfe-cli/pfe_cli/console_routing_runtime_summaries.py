"""Trigger, gate, and runtime console summary text."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .console_routing_deps import ConsoleRoutingDeps
from .console_routing_runtime_summary_parts import gate_summary_parts, runtime_summary_parts, trigger_summary_parts
from .console_routing_summary_helpers import render_summary


def console_trigger_summary_text(payload: Mapping[str, Any], *, deps: ConsoleRoutingDeps) -> str:
    return render_summary("PFE auto-train trigger summary", trigger_summary_parts(payload, deps=deps), fallback="state=idle")


def console_gate_summary_text(payload: Mapping[str, Any], *, deps: ConsoleRoutingDeps) -> str:
    return render_summary("PFE gate summary", gate_summary_parts(payload, deps=deps), fallback="state=idle")


def console_runtime_summary_text(payload: Mapping[str, Any], *, deps: ConsoleRoutingDeps) -> str:
    return render_summary("PFE runtime stability summary", runtime_summary_parts(payload, deps=deps), fallback="state=idle")


__all__ = ["console_gate_summary_text", "console_runtime_summary_text", "console_trigger_summary_text"]
