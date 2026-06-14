"""Box, header, and separator drawing helpers for Matrix terminal output."""

from __future__ import annotations

from .terminal_theme_palette import Borders, MatrixColors, _strip_ansi


def draw_box(title: str, content: list[str], width: int = 200) -> str:
    """Draw an ASCII box with title."""
    lines = []

    title_text = f"[ {title} ]"
    remaining = width - len(title_text) - 4
    lines.append(
        MatrixColors.GREEN
        + Borders.TOP_LEFT
        + Borders.HORIZONTAL * 2
        + title_text
        + Borders.HORIZONTAL * remaining
        + Borders.TOP_RIGHT
        + MatrixColors.RESET
    )

    for line in content:
        plain = _strip_ansi(line)
        visible_len = len(plain)
        if visible_len > width - 4:
            line = plain[: width - 7] + "..."
            visible_len = len(line)
        padding = " " * (width - 4 - visible_len)
        lines.append(
            MatrixColors.GREEN
            + Borders.VERTICAL
            + "  "
            + MatrixColors.RESET
            + line
            + padding
            + MatrixColors.GREEN
            + "  "
            + Borders.VERTICAL
            + MatrixColors.RESET
        )

    lines.append(
        MatrixColors.GREEN
        + Borders.BOTTOM_LEFT
        + Borders.HORIZONTAL * (width - 2)
        + Borders.BOTTOM_RIGHT
        + MatrixColors.RESET
    )

    return "\n".join(lines)


def draw_header(text: str, width: int = 80) -> str:
    """Draw a header line."""
    lines = []
    lines.append(MatrixColors.GREEN_BRIGHT + Borders.HORIZONTAL_DOUBLE * width + MatrixColors.RESET)

    centered = f"[ {text} ]"
    padding = (width - len(centered)) // 2
    lines.append(
        " " * padding
        + MatrixColors.GREEN_BRIGHT
        + MatrixColors.BOLD
        + centered
        + MatrixColors.RESET
    )

    lines.append(MatrixColors.GREEN_BRIGHT + Borders.HORIZONTAL_DOUBLE * width + MatrixColors.RESET)

    return "\n".join(lines)


def draw_separator(width: int = 80) -> str:
    """Draw a horizontal separator line."""
    return MatrixColors.GREEN_DIM + Borders.HORIZONTAL * width + MatrixColors.RESET


__all__ = ["draw_box", "draw_header", "draw_separator"]
