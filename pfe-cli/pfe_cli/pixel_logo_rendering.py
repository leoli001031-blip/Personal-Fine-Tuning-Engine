"""Render helpers for ZC logo and CLI boot banner."""

from __future__ import annotations

from .pixel_logo_art import (
    ZC_BLOCK_LETTERS,
    ZC_BLOCKY,
    ZC_BOX,
    ZC_CLEAN,
    ZC_COMPACT,
    ZC_LARGE,
    ZC_PIXEL_ART,
)
from .terminal_theme import MatrixColors


def render_zc_logo(style: str = "pixel", glow: bool = True) -> str:
    """Render ZC logo in specified style."""
    if style == "blocky":
        lines = ZC_BLOCKY
    elif style == "large":
        lines = ZC_LARGE
    elif style == "clean":
        lines = ZC_CLEAN
    elif style == "box":
        lines = ZC_BOX
    elif style == "compact":
        lines = ZC_COMPACT
    elif style == "block":
        lines = ZC_BLOCK_LETTERS
    else:
        lines = ZC_PIXEL_ART

    result = []
    for line in lines:
        result.append(f"{MatrixColors.GREEN}{line}{MatrixColors.RESET}")

    return "\n".join(result)


def render_boot_banner(version: str = "2.0.0") -> str:
    """Render full boot banner with logo and system info."""
    lines = []

    lines.append("")
    lines.append(render_zc_logo(style="pixel", glow=True))
    lines.append("")
    lines.append(f"{MatrixColors.GREEN_DIM}{'═' * 50}{MatrixColors.RESET}")
    lines.append(f"{MatrixColors.GREEN_BRIGHT}{MatrixColors.BOLD}  PERSONAL FINETUNE ENGINE{MatrixColors.RESET}")
    lines.append(f"{MatrixColors.GREEN}  Version {version} | Matrix Terminal{MatrixColors.RESET}")
    lines.append(f"{MatrixColors.GREEN_DIM}{'═' * 50}{MatrixColors.RESET}")
    lines.append("")
    lines.append(f"{MatrixColors.GREEN_DIM}  [INITIALIZING SYSTEM COMPONENTS...]{MatrixColors.RESET}")
    lines.append("")

    return "\n".join(lines)


def render_loading_sequence(step: int, total: int = 5) -> str:
    """Render a loading step indicator."""
    filled = int((step / total) * 10)
    bar = f"{MatrixColors.GREEN_BRIGHT}█{MatrixColors.GREEN}" * filled
    bar += f"{MatrixColors.GREEN_DIM}░{MatrixColors.RESET}" * (10 - filled)
    return f"  [{bar}] {step}/{total}"


def render_typing_effect(text: str, progress: float = 1.0) -> str:
    """Render text with typewriter effect."""
    visible_chars = int(len(text) * progress)
    visible = text[:visible_chars]
    cursor = "▌" if progress < 1.0 else ""
    return f"{MatrixColors.GREEN}{visible}{MatrixColors.GREEN_BRIGHT}{cursor}{MatrixColors.RESET}"


__all__ = [
    "render_zc_logo",
    "render_boot_banner",
    "render_loading_sequence",
    "render_typing_effect",
]
