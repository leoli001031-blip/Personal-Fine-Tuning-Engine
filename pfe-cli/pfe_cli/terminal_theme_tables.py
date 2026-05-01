"""Badge, table, progress, and key-value helpers for Matrix terminal output."""

from __future__ import annotations

from typing import Any

from .terminal_theme_palette import Borders, MatrixColors


def status_badge(status: str) -> str:
    """Create a status badge with appropriate color."""
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
    """Draw an ASCII table."""
    if not rows:
        return ""

    col_count = len(headers)
    col_width = (width - col_count - 1) // col_count

    lines = []

    top = Borders.TOP_LEFT
    for i in range(col_count):
        top += Borders.HORIZONTAL * col_width
        if i < col_count - 1:
            top += Borders.T_TOP
    top += Borders.TOP_RIGHT
    lines.append(MatrixColors.GREEN + top + MatrixColors.RESET)

    header_line = Borders.VERTICAL
    for h in headers:
        padding = " " * ((col_width - len(h)) // 2)
        header_line += (
            padding
            + MatrixColors.GREEN_BRIGHT
            + MatrixColors.BOLD
            + h
            + MatrixColors.RESET
            + padding
        )
        if len(h) % 2 != col_width % 2:
            header_line += " "
        header_line += Borders.VERTICAL
    lines.append(header_line)

    sep = Borders.T_LEFT
    for i in range(col_count):
        sep += Borders.HORIZONTAL * col_width
        if i < col_count - 1:
            sep += Borders.CROSS
    sep += Borders.T_RIGHT
    lines.append(MatrixColors.GREEN + sep + MatrixColors.RESET)

    for row in rows:
        row_line = Borders.VERTICAL
        for cell in row:
            cell_str = str(cell)[: col_width - 2]
            padding = " " * (col_width - len(cell_str) - 2)
            row_line += "  " + cell_str + padding + Borders.VERTICAL
        lines.append(row_line)

    bottom = Borders.BOTTOM_LEFT
    for i in range(col_count):
        bottom += Borders.HORIZONTAL * col_width
        if i < col_count - 1:
            bottom += Borders.T_BOTTOM
    bottom += Borders.BOTTOM_RIGHT
    lines.append(MatrixColors.GREEN + bottom + MatrixColors.RESET)

    return "\n".join(lines)


def progress_bar(current: int, total: int, width: int = 40) -> str:
    """Draw a progress bar."""
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


__all__ = ["draw_table", "status_badge", "progress_bar", "format_key_value"]
