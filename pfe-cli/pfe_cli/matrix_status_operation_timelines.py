"""Operations timeline status sections for Matrix terminal output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_formatting_common import _coerce_mapping, _coerce_sequence_of_mappings, _format_scalar
from .terminal_theme import MatrixColors, draw_box, format_key_value


def append_operation_timeline_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append runner and daemon timeline status boxes."""
    runner_timeline = _coerce_mapping(mapping.get("runner_timeline"))
    if runner_timeline is None:
        operations_console = _coerce_mapping(mapping.get("operations_console"))
        if operations_console:
            runner_timeline = _coerce_mapping(operations_console.get("runner_timeline"))
    if runner_timeline:
        rt_content = []
        count = runner_timeline.get("count", 0)
        rt_content.append(format_key_value("count", count))
        last_event = runner_timeline.get("last_event")
        if last_event:
            rt_content.append(format_key_value("last event", last_event))
        last_reason = runner_timeline.get("last_reason")
        if last_reason:
            rt_content.append(format_key_value("last reason", last_reason))
        for key in ("takeover_event_count", "last_takeover_event", "last_takeover_reason", "recent_anomaly_reason"):
            value = runner_timeline.get(key)
            if value is not None:
                rt_content.append(format_key_value(key.replace("_", " "), value))
        for key in ("current_active", "current_stop_requested"):
            value = runner_timeline.get(key)
            if value is not None:
                rt_content.append(format_key_value(key.replace("_", " "), "yes" if value else "no"))
        current_lock_state = runner_timeline.get("current_lock_state")
        if current_lock_state is not None:
            rt_content.append(format_key_value("current lock state", current_lock_state))
        events = _coerce_sequence_of_mappings(runner_timeline.get("events") or runner_timeline.get("recent_events"))
        if events:
            for ev in events[:5]:
                ev_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in ev.items() if v is not None)
                if ev_parts:
                    rt_content.append(f"  {MatrixColors.GREEN_DIM}>{MatrixColors.RESET} {ev_parts}")
        takeover_events = _coerce_sequence_of_mappings(runner_timeline.get("recent_takeover_events"))
        if takeover_events:
            for ev in takeover_events[:3]:
                ev_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in ev.items() if v is not None)
                if ev_parts:
                    rt_content.append(f"  {MatrixColors.AMBER}>{MatrixColors.RESET} {ev_parts}")
        if rt_content:
            lines.append(draw_box("RUNNER TIMELINE", rt_content))
            lines.append("")

    daemon_timeline = _coerce_mapping(mapping.get("daemon_timeline"))
    if daemon_timeline is None:
        operations_console = _coerce_mapping(mapping.get("operations_console"))
        if operations_console:
            daemon_timeline = _coerce_mapping(operations_console.get("daemon_timeline"))
    if daemon_timeline:
        dt_content = []
        count = daemon_timeline.get("count", 0)
        dt_content.append(format_key_value("count", count))
        recovery_event_count = daemon_timeline.get("recovery_event_count")
        if recovery_event_count is not None:
            dt_content.append(format_key_value("recovery event count", recovery_event_count))
        last_event = daemon_timeline.get("last_event")
        if last_event:
            dt_content.append(format_key_value("last event", last_event))
        last_reason = daemon_timeline.get("last_reason")
        if last_reason:
            dt_content.append(format_key_value("last reason", last_reason))
        for key in ("last_recovery_event", "last_recovery_reason", "last_recovery_note", "recent_anomaly_reason"):
            value = daemon_timeline.get(key)
            if value is not None:
                dt_content.append(format_key_value(key.replace("_", " "), value))
        latest_timestamp = daemon_timeline.get("latest_timestamp")
        if latest_timestamp is not None:
            dt_content.append(format_key_value("latest timestamp", latest_timestamp))
        events = _coerce_sequence_of_mappings(daemon_timeline.get("events"))
        if events:
            for ev in events[:5]:
                ev_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in ev.items() if v is not None)
                if ev_parts:
                    dt_content.append(f"  {MatrixColors.GREEN_DIM}>{MatrixColors.RESET} {ev_parts}")
        recovery_events = _coerce_sequence_of_mappings(daemon_timeline.get("recent_recovery_events"))
        if recovery_events:
            dt_content.append(f"{MatrixColors.GREEN_BRIGHT}recent recovery events:{MatrixColors.RESET}")
            for ev in recovery_events[:5]:
                ev_parts = " | ".join(f"{k.replace('_', ' ')}={_format_scalar(v)}" for k, v in ev.items() if v is not None)
                if ev_parts:
                    dt_content.append(f"  {MatrixColors.AMBER}>{MatrixColors.RESET} {ev_parts}")
        if dt_content:
            lines.append(draw_box("DAEMON TIMELINE", dt_content))
            lines.append("")


__all__ = ["append_operation_timeline_sections"]
