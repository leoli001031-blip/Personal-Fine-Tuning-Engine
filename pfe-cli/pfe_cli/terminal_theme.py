"""Terminal Hacker Theme for PFE CLI.

Matrix Green OLED terminal aesthetic with:
- Background: #050505 (OLED Black)
- Primary: #33FF00 (Matrix Green)
- Warning: #FFB000 (Amber)
- Error: #FF3333 (Red)
- Font: JetBrains Mono
"""

from __future__ import annotations

import re
from typing import Any
from dataclasses import dataclass

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ANSI Color Codes for Terminal
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


# ASCII Border Characters
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


# Predefined styles
STYLE_SUCCESS = TerminalStyle(color=MatrixColors.GREEN, bold=True)
STYLE_WARNING = TerminalStyle(color=MatrixColors.AMBER)
STYLE_ERROR = TerminalStyle(color=MatrixColors.RED, bold=True)
STYLE_INFO = TerminalStyle(color=MatrixColors.GRAY)
STYLE_DIM = TerminalStyle(color=MatrixColors.DIM, dim=True)
STYLE_HEADER = TerminalStyle(color=MatrixColors.GREEN_BRIGHT, bold=True)


def draw_box(title: str, content: list[str], width: int = 200) -> str:
    """Draw an ASCII box with title.

    ┌──────────────────────────────────────────────────────────────┐
    │  [ TITLE ]                                                   │
    ├──────────────────────────────────────────────────────────────┤
    │  Content line 1                                              │
    │  Content line 2                                              │
    └──────────────────────────────────────────────────────────────┘
    """
    lines = []

    # Top border with title
    title_text = f"[ {title} ]"
    remaining = width - len(title_text) - 4
    lines.append(
        MatrixColors.GREEN + Borders.TOP_LEFT +
        Borders.HORIZONTAL * 2 + title_text + Borders.HORIZONTAL * remaining +
        Borders.TOP_RIGHT + MatrixColors.RESET
    )

    # Content
    for line in content:
        plain = _strip_ansi(line)
        visible_len = len(plain)
        if visible_len > width - 4:
            line = plain[:width - 7] + "..."
            visible_len = len(line)
        padding = " " * (width - 4 - visible_len)
        lines.append(
            MatrixColors.GREEN + Borders.VERTICAL + "  " + MatrixColors.RESET +
            line + padding +
            MatrixColors.GREEN + "  " + Borders.VERTICAL + MatrixColors.RESET
        )

    # Bottom border
    lines.append(
        MatrixColors.GREEN + Borders.BOTTOM_LEFT +
        Borders.HORIZONTAL * (width - 2) +
        Borders.BOTTOM_RIGHT + MatrixColors.RESET
    )

    return "\n".join(lines)


def draw_header(text: str, width: int = 80) -> str:
    """Draw a header line.

    ═══════════════════════════════════════════════════════════════
    [ TEXT ]
    ═══════════════════════════════════════════════════════════════
    """
    lines = []
    lines.append(MatrixColors.GREEN_BRIGHT + Borders.HORIZONTAL_DOUBLE * width + MatrixColors.RESET)

    centered = f"[ {text} ]"
    padding = (width - len(centered)) // 2
    lines.append(
        " " * padding +
        MatrixColors.GREEN_BRIGHT + MatrixColors.BOLD + centered + MatrixColors.RESET
    )

    lines.append(MatrixColors.GREEN_BRIGHT + Borders.HORIZONTAL_DOUBLE * width + MatrixColors.RESET)

    return "\n".join(lines)


def draw_separator(width: int = 80) -> str:
    """Draw a horizontal separator line."""
    return MatrixColors.GREEN_DIM + Borders.HORIZONTAL * width + MatrixColors.RESET


def status_badge(status: str) -> str:
    """Create a status badge with appropriate color.

    [ ACTIVE ]    [ WARNING ]    [ ERROR ]
    """
    status_upper = status.upper()

    if status in ("active", "promoted", "executed", "success", "ready", "healthy", "running"):
        return f"{MatrixColors.GREEN}[ {status_upper} ]{MatrixColors.RESET}"
    elif status in ("warning", "pending", "pending_eval", "blocked", "degraded"):
        return f"{MatrixColors.AMBER}[ {status_upper} ]{MatrixColors.RESET}"
    elif status in ("error", "failed", "critical", "inactive"):
        return f"{MatrixColors.RED}[ {status_upper} ]{MatrixColors.RESET}"
    else:
        return f"{MatrixColors.GRAY}[ {status_upper} ]{MatrixColors.RESET}"


def draw_table(headers: list[str], rows: list[list[str]], width: int = 80) -> str:
    """Draw an ASCII table.

    ┌──────────┬──────────┬──────────┐
    │  HEADER1 │  HEADER2 │  HEADER3 │
    ├──────────┼──────────┼──────────┤
    │  data1   │  data2   │  data3   │
    └──────────┴──────────┴──────────┘
    """
    if not rows:
        return ""

    # Calculate column widths
    col_count = len(headers)
    col_width = (width - col_count - 1) // col_count

    lines = []

    # Top border
    top = Borders.TOP_LEFT
    for i in range(col_count):
        top += Borders.HORIZONTAL * col_width
        if i < col_count - 1:
            top += Borders.T_TOP
    top += Borders.TOP_RIGHT
    lines.append(MatrixColors.GREEN + top + MatrixColors.RESET)

    # Headers
    header_line = Borders.VERTICAL
    for i, h in enumerate(headers):
        padding = " " * ((col_width - len(h)) // 2)
        header_line += padding + MatrixColors.GREEN_BRIGHT + MatrixColors.BOLD + h + MatrixColors.RESET + padding
        if len(h) % 2 != col_width % 2:
            header_line += " "
        header_line += Borders.VERTICAL
    lines.append(header_line)

    # Separator
    sep = Borders.T_LEFT
    for i in range(col_count):
        sep += Borders.HORIZONTAL * col_width
        if i < col_count - 1:
            sep += Borders.CROSS
    sep += Borders.T_RIGHT
    lines.append(MatrixColors.GREEN + sep + MatrixColors.RESET)

    # Rows
    for row in rows:
        row_line = Borders.VERTICAL
        for i, cell in enumerate(row):
            cell_str = str(cell)[:col_width-2]
            padding = " " * (col_width - len(cell_str) - 2)
            row_line += "  " + cell_str + padding + Borders.VERTICAL
        lines.append(row_line)

    # Bottom border
    bottom = Borders.BOTTOM_LEFT
    for i in range(col_count):
        bottom += Borders.HORIZONTAL * col_width
        if i < col_count - 1:
            bottom += Borders.T_BOTTOM
    bottom += Borders.BOTTOM_RIGHT
    lines.append(MatrixColors.GREEN + bottom + MatrixColors.RESET)

    return "\n".join(lines)


def progress_bar(current: int, total: int, width: int = 40) -> str:
    """Draw a progress bar.

    [████████████████████░░░░░░░░░░░░░░░░]  50%
    """
    if total == 0:
        percent = 0
        filled = 0
    else:
        percent = min(100, int((current / total) * 100))
        filled = int((current / total) * width)

    bar = MatrixColors.GREEN + "█" * filled + MatrixColors.GREEN_DIM + "░" * (width - filled) + MatrixColors.RESET

    if percent >= 80:
        percent_str = f"{MatrixColors.GREEN_BRIGHT}{percent:3d}%{MatrixColors.RESET}"
    elif percent >= 50:
        percent_str = f"{MatrixColors.GREEN}{percent:3d}%{MatrixColors.RESET}"
    else:
        percent_str = f"{MatrixColors.GRAY}{percent:3d}%{MatrixColors.RESET}"

    return f"[{bar}] {percent_str}"


def format_key_value(key: str, value: Any, key_width: int = 25) -> str:
    """Format a key-value pair."""
    key_str = f"{key}:".ljust(key_width)
    value_str = str(value)
    return f"{MatrixColors.GREEN_DIM}{key_str}{MatrixColors.RESET}{MatrixColors.WHITE}{value_str}{MatrixColors.RESET}"


def draw_boot_sequence() -> str:
    """Draw a boot sequence splash."""
    lines = [
        "",
        MatrixColors.GREEN_DIM + "    Initializing PFE Core Systems..." + MatrixColors.RESET,
        "",
        MatrixColors.GREEN + "    [■] Loading adapter store..." + MatrixColors.RESET,
        MatrixColors.GREEN + "    [■] Initializing trainer service..." + MatrixColors.RESET,
        MatrixColors.GREEN + "    [■] Mounting signal collector..." + MatrixColors.RESET,
        MatrixColors.GREEN + "    [■] Establishing daemon connection..." + MatrixColors.RESET,
        "",
        MatrixColors.GREEN_BRIGHT + MatrixColors.BOLD + "    >> SYSTEM READY <<" + MatrixColors.RESET,
        "",
    ]
    return "\n".join(lines)


# Rich-compatible styles for when Rich is available
RICH_THEME = {
    "green": "#33FF00",
    "green_bright": "#64FF50",
    "green_dim": "#1A8000",
    "amber": "#FFB000",
    "red": "#FF3333",
    "bg": "#050505",
    "white": "#F8FAFC",
    "gray": "#94A3B8",
}


def get_rich_console():
    """Get a Rich console configured for Matrix theme."""
    try:
        from rich.console import Console
        from rich.theme import Theme

        custom_theme = Theme({
            "matrix.green": RICH_THEME["green"],
            "matrix.green_bright": RICH_THEME["green_bright"],
            "matrix.amber": RICH_THEME["amber"],
            "matrix.red": RICH_THEME["red"],
            "matrix.gray": RICH_THEME["gray"],
            "info": RICH_THEME["gray"],
            "warning": RICH_THEME["amber"],
            "error": RICH_THEME["red"],
            "success": RICH_THEME["green"],
        })

        return Console(
            theme=custom_theme,
            style="on #050505",
            highlight=False,
        )
    except ImportError:
        return None


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