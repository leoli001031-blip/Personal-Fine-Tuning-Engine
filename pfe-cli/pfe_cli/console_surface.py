"""Compatibility exports for console surface text helpers."""

from __future__ import annotations

from .console_surface_deps import ConsoleSurfaceDeps
from .console_surface_focus import console_dashboard_focus
from .console_surface_help import console_help_text
from .console_surface_settings import console_settings_text
from .console_surface_status import console_status_compact_text

__all__ = [
    "ConsoleSurfaceDeps",
    "console_dashboard_focus",
    "console_help_text",
    "console_settings_text",
    "console_status_compact_text",
]
