"""Palette, border, and text style primitives for Matrix terminal output."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class MatrixColors:
    """Matrix Green terminal color palette."""

    # Background (OLED Black)
    BG = "\033[48;2;5;5;5m"
    BG_CODE = "#050505"

    # Primary (Matrix Green)
    GREEN = "\033[38;2;51;255;0m"
    GREEN_BRIGHT = "\033[38;2;100;255;80m"
    GREEN_DIM = "\033[38;2;26;128;0m"
    GREEN_CODE = "#33FF00"

    # Warning (Amber)
    AMBER = "\033[38;2;255;176;0m"
    AMBER_CODE = "#FFB000"

    # Error (Red)
    RED = "\033[38;2;255;51;51m"
    RED_CODE = "#FF3333"

    # Text
    WHITE = "\033[38;2;248;250;252m"
    GRAY = "\033[38;2;148;163;184m"
    DIM = "\033[38;2;100;100;100m"

    # Reset
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM_STYLE = "\033[2m"


class Borders:
    """ASCII-style borders for terminal UI."""

    HORIZONTAL = "─"
    VERTICAL = "│"
    TOP_LEFT = "┌"
    TOP_RIGHT = "┐"
    BOTTOM_LEFT = "└"
    BOTTOM_RIGHT = "┘"
    T_LEFT = "├"
    T_RIGHT = "┤"
    T_TOP = "┬"
    T_BOTTOM = "┴"
    CROSS = "┼"

    # Double line for headers
    HORIZONTAL_DOUBLE = "═"
    TOP_LEFT_DOUBLE = "╔"
    TOP_RIGHT_DOUBLE = "╗"
    BOTTOM_LEFT_DOUBLE = "╚"
    BOTTOM_RIGHT_DOUBLE = "╝"


@dataclass
class TerminalStyle:
    """Style configuration for terminal output."""

    color: str = MatrixColors.GREEN
    bold: bool = False
    dim: bool = False

    def apply(self, text: str) -> str:
        """Apply style to text."""
        result = ""
        if self.bold:
            result += MatrixColors.BOLD
        if self.dim:
            result += MatrixColors.DIM_STYLE
        result += self.color + text + MatrixColors.RESET
        return result


STYLE_SUCCESS = TerminalStyle(color=MatrixColors.GREEN, bold=True)
STYLE_WARNING = TerminalStyle(color=MatrixColors.AMBER)
STYLE_ERROR = TerminalStyle(color=MatrixColors.RED, bold=True)
STYLE_INFO = TerminalStyle(color=MatrixColors.GRAY)
STYLE_DIM = TerminalStyle(color=MatrixColors.DIM, dim=True)
STYLE_HEADER = TerminalStyle(color=MatrixColors.GREEN_BRIGHT, bold=True)


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
]
