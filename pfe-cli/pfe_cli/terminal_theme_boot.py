"""Boot splash and optional Rich console support for Matrix terminal output."""

from __future__ import annotations

from .terminal_theme_palette import MatrixColors


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
        MatrixColors.GREEN_BRIGHT
        + MatrixColors.BOLD
        + "    >> SYSTEM READY <<"
        + MatrixColors.RESET,
        "",
    ]
    return "\n".join(lines)


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

        custom_theme = Theme(
            {
                "matrix.green": RICH_THEME["green"],
                "matrix.green_bright": RICH_THEME["green_bright"],
                "matrix.amber": RICH_THEME["amber"],
                "matrix.red": RICH_THEME["red"],
                "matrix.gray": RICH_THEME["gray"],
                "info": RICH_THEME["gray"],
                "warning": RICH_THEME["amber"],
                "error": RICH_THEME["red"],
                "success": RICH_THEME["green"],
            }
        )

        return Console(
            theme=custom_theme,
            style="on #050505",
            highlight=False,
        )
    except ImportError:
        return None


__all__ = ["RICH_THEME", "draw_boot_sequence", "get_rich_console"]
