"""Signal readiness status section for Matrix terminal output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping
from .terminal_theme import draw_box, format_key_value, status_badge


def append_signal_readiness_section(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append signal readiness status box."""
    signal_readiness = _coerce_mapping(mapping.get("signal_readiness"))
    if signal_readiness is None:
        signal_summary = _coerce_mapping(mapping.get("signal_summary"))
        if signal_summary:
            signal_readiness = {
                "state": signal_summary.get("state", "unknown"),
                "total_ready_signals": signal_summary.get("event_chain_complete_count", 0),
                "event_chain_ready": signal_summary.get("event_chain_ready", False),
                "readiness_reason": signal_summary.get("quality_filter_state", ""),
                "signal_quality_filter": {
                    "total_signals": signal_summary.get("processed_count", 0),
                    "passed_signals": signal_summary.get("quality_filtered_count", 0),
                    "rejected_signals": signal_summary.get("quality_filtered_count", 0),
                    "rejection_reasons": signal_summary.get("quality_filter_reasons", {}),
                },
            }
    if signal_readiness:
        sig_content = []
        state = signal_readiness.get("state", "unknown")
        sig_content.append(format_key_value("state", status_badge(state)))
        total_ready = signal_readiness.get("total_ready_signals", 0)
        sig_content.append(format_key_value("ready signals", total_ready))
        readiness_reason = signal_readiness.get("readiness_reason", "")
        if readiness_reason:
            sig_content.append(format_key_value("reason", readiness_reason))
        event_chain_ready = signal_readiness.get("event_chain_ready")
        if event_chain_ready is not None:
            sig_content.append(format_key_value("event chain ready", "yes" if event_chain_ready else "no"))
        signal_quality = _coerce_mapping(signal_readiness.get("signal_quality_filter"))
        if signal_quality:
            q_total = signal_quality.get("total_signals", 0)
            q_passed = signal_quality.get("passed_signals", 0)
            q_rejected = signal_quality.get("rejected_signals", 0)
            sig_content.append(
                format_key_value("quality filter", f"total={q_total} | passed={q_passed} | rejected={q_rejected}")
            )
            q_reasons = _coerce_mapping(signal_quality.get("rejection_reasons"))
            if q_reasons:
                for reason, count in q_reasons.items():
                    sig_content.append(format_key_value(f"  {reason.replace('_', ' ')}", count))
        lines.append(draw_box("SIGNAL READINESS", sig_content))
        lines.append("")


__all__ = ["append_signal_readiness_section"]
