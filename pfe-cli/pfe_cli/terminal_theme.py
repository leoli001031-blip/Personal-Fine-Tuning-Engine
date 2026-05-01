"""Matrix terminal theme compatibility exports."""

from __future__ import annotations

from .terminal_theme_boot import RICH_THEME, draw_boot_sequence, get_rich_console
from .terminal_theme_boxes import draw_box, draw_header, draw_separator
from .terminal_theme_palette import (
    STYLE_DIM,
    STYLE_ERROR,
    STYLE_HEADER,
    STYLE_INFO,
    STYLE_SUCCESS,
    STYLE_WARNING,
    Borders,
    MatrixColors,
    TerminalStyle,
)
from .terminal_theme_tables import draw_table, format_key_value, progress_bar, status_badge

__all__ = [
    "MatrixColors",
    "Borders",
    "TerminalStyle",
    "STYLE_SUCCESS",
    "STYLE_WARNING",
    "STYLE_ERROR",
    "STYLE_INFO",
    "STYLE_DIM",
    "STYLE_HEADER",
    "draw_box",
    "draw_header",
    "draw_separator",
    "draw_table",
    "status_badge",
    "progress_bar",
    "format_key_value",
    "draw_boot_sequence",
    "RICH_THEME",
    "get_rich_console",
]
