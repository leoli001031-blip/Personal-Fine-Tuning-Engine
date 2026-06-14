"""Compatibility exports for simple operations surface formatters."""

from __future__ import annotations

from .operations_dashboard_surface_formatting import (
    format_operations_alert_policy,
    format_operations_dashboard,
)
from .operations_event_stream_formatting import format_operations_event_stream
from .operations_timeline_surface_formatting import format_operations_timeline

__all__ = [
    "format_operations_alert_policy",
    "format_operations_dashboard",
    "format_operations_event_stream",
    "format_operations_timeline",
]
