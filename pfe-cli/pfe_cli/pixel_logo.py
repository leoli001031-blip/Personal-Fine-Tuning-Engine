"""Pixel art ZC logo for PFE CLI boot sequence.

Matrix Green OLED terminal aesthetic with 8-bit style ZC monogram.
Colors:
- Background: #050505 (OLED Black)
- Primary: #33FF00 (Matrix Green)
- Bright: #64FF50 (Bright Green)
- Dim: #1A8000 (Dim Green)
"""

from __future__ import annotations

from .terminal_theme import MatrixColors

# Improved ZC Pixel Art - clearer Z and C shapes
# Using full block characters for pixel-perfect look

# Z: Diagonal from top-right to bottom-left
# C: Like a reversed C, open on the right side

ZC_PIXEL_ART = [
    "███████╗  ██████╗",
    "    ██║  ██╔════╝",
    "   ██║   ██║     ",
    "  ██║    ██║     ",
    " ███████╗╚██████╗",
    " ╚══════╝ ╚═════╝",
]

# Alternative: More blocky/8-bit style
ZC_BLOCKY = [
    "▓▓▓▓▓▓▓   ▓▓▓▓▓▓",
    "     ▓▓  ▓▓     ",
    "    ▓▓   ▓▓     ",
    "   ▓▓    ▓▓     ",
    "  ▓▓▓▓▓▓▓  ▓▓▓▓▓",
    "         ▓▓▓▓▓▓",
]

# Large detailed version with spacing
ZC_LARGE = [
    "██████████        ██████████",
    "        ██       ██        ",
    "       ██        ██        ",
    "      ██         ██        ",
    "     ██          ██        ",
    "    ██            ████████ ",
    "   ███████████     ████████",
]

# Minimal clean version
ZC_CLEAN = [
    "██████╗  ██████╗",
    "    ██║ ██╔════╝",
    "   ██║  ██║     ",
    "  ██║   ██║     ",
    "███████╗╚██████╗",
    "╚══════╝ ╚═════╝",
]

# Box style with borders
ZC_BOX = [
    "╔══════════╗  ╔══════════╗",
    "║ ████████ ║  ║ ████████ ║",
    "║      ███ ║  ║ ██       ║",
    "║    ███   ║  ║ ██       ║",
    "║  ███     ║  ║ ██       ║",
    "║ ████████ ║  ║ ████████ ║",
    "╚══════════╝  ╚══════════╝",
]

# Compact version for small displays
ZC_COMPACT = [
    "███ ███",
    " ██ █  ",
    "██  ███",
]

# Very clear block letters
ZC_BLOCK_LETTERS = [
    "███████   ██████ ",
    "     ██  ██      ",
    "    ██   ██      ",
    "   ██    ██      ",
    "  ██████  ██████ ",
    "                 ",
]


def render_zc_logo(style: str = "pixel", glow: bool = True) -> str:
    """Render ZC logo in specified style.

    Args:
        style: "pixel", "blocky", "large", "clean", "box", "compact"
        glow: Add glow effect to edges
    """
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
    else:  # pixel (default)
        lines = ZC_PIXEL_ART

    result = []
    for i, line in enumerate(lines):
        # Simple clean green color for all lines
        result.append(f"{MatrixColors.GREEN}{line}{MatrixColors.RESET}")

    return '\n'.join(result)


def render_boot_banner(version: str = "2.0.0") -> str:
    """Render full boot banner with logo and system info."""
    lines = []

    # Top spacer
    lines.append("")

    # Logo
    lines.append(render_zc_logo(style="pixel", glow=True))

    # Separator
    lines.append("")
    lines.append(f"{MatrixColors.GREEN_DIM}{'═' * 50}{MatrixColors.RESET}")

    # Title
    lines.append(f"{MatrixColors.GREEN_BRIGHT}{MatrixColors.BOLD}  PERSONAL FINETUNE ENGINE{MatrixColors.RESET}")
    lines.append(f"{MatrixColors.GREEN}  Version {version} | Matrix Terminal{MatrixColors.RESET}")

    # Separator
    lines.append(f"{MatrixColors.GREEN_DIM}{'═' * 50}{MatrixColors.RESET}")
    lines.append("")

    # Boot messages
    lines.append(f"{MatrixColors.GREEN_DIM}  [INITIALIZING SYSTEM COMPONENTS...]{MatrixColors.RESET}")
    lines.append("")

    return '\n'.join(lines)


def render_loading_sequence(step: int, total: int = 5) -> str:
    """Render a loading step indicator."""
    blocks = ['░', '▒', '▓', '█']
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


# Static logo for import
LOGO_ASCII = """
     ███████╗  ██████╗
     ╚══███╔╝ ██╔════╝
       ███╔╝  ██║
      ███╔╝   ██║
     ███████╗ ╚██████╗
     ╚══════╝  ╚═════╝
"""

# Glitch effect variants for animation
ZC_GLITCH_VARIANTS = [
    [
        "███████╗  ██████╗",
        "    ██║  ██╔════╝",
        "   ██║   ██║     ",
        "  ██║    ██║     ",
        " ███████╗╚██████╗",
        " ╚══════╝ ╚═════╝",
    ],
    [
        "▓▓▓▓▓▓▓╗  ▓▓▓▓▓▓╗",
        "    ▓▓▓║  ▓▓╔════╝",
        "   ▓▓▓║   ▓▓║     ",
        "  ▓▓▓║    ▓▓║     ",
        " ▓▓▓▓▓▓▓╗ ╚▓▓▓▓▓▓╗",
        " ╚══════╝  ╚═════╝",
    ],
]


# PFE Commands for Matrix-style display
PFE_COMMANDS = {
    "Core": [
        ("generate", "Generate cold-start training samples"),
        ("train", "Train an adapter model"),
        ("dpo", "Train with Direct Preference Optimization"),
        ("eval", "Evaluate adapter performance"),
        ("serve", "Start inference server"),
    ],
    "Status": [
        ("status", "Show engine and adapter status"),
        ("console", "Interactive operations console"),
        ("doctor", "System readiness check"),
        ("dashboard", "Launch web dashboard"),
        ("boot", "Display boot sequence"),
    ],
    "Adapter": [
        ("adapter list", "List adapter versions"),
        ("adapter promote", "Promote adapter to latest"),
        ("adapter rollback", "Rollback to prior version"),
        ("candidate", "Manage candidate lifecycle"),
    ],
    "Pipeline": [
        ("trigger", "Manage auto-train trigger"),
        ("daemon", "Control background daemon"),
        ("collect", "Manage signal collection"),
        ("distill", "Run teacher distillation"),
    ],
    "Config": [
        ("profile", "Manage user profiles"),
        ("scenario", "Configure scenarios"),
        ("route", "Test scenario routing"),
    ],
}


def render_commands_matrix() -> str:
    """Render Matrix-style command reference."""
    lines = []

    # Header
    lines.append("")
    header_top = MatrixColors.GREEN_DIM + "┌" + "─" * 70 + "┐" + MatrixColors.RESET
    header_mid = MatrixColors.GREEN_DIM + "│" + MatrixColors.RESET + "  " + MatrixColors.GREEN_BRIGHT + MatrixColors.BOLD + "AVAILABLE COMMANDS" + MatrixColors.RESET + "                                      " + MatrixColors.GREEN_DIM + "│" + MatrixColors.RESET
    header_sep = MatrixColors.GREEN_DIM + "├" + "─" * 70 + "┤" + MatrixColors.RESET
    lines.append(header_top)
    lines.append(header_mid)
    lines.append(header_sep)

    for category, commands in PFE_COMMANDS.items():
        # Category header - fixed width
        cat_text = "  " + category
        cat_padded = cat_text.ljust(68)
        cat_line = MatrixColors.GREEN_DIM + "│" + MatrixColors.RESET + MatrixColors.AMBER + MatrixColors.BOLD + cat_padded + MatrixColors.RESET + MatrixColors.GREEN_DIM + "│" + MatrixColors.RESET
        lines.append(cat_line)

        for cmd, desc in commands:
            # Command with description
            cmd_padded = cmd.ljust(20)
            desc_padded = desc.ljust(45)
            line = (MatrixColors.GREEN_DIM + "│" + MatrixColors.RESET +
                    "  " + MatrixColors.GREEN + cmd_padded + MatrixColors.RESET +
                    " " + MatrixColors.GRAY + desc_padded + MatrixColors.RESET +
                    MatrixColors.GREEN_DIM + "│" + MatrixColors.RESET)
            lines.append(line)

        lines.append(MatrixColors.GREEN_DIM + "│" + " " * 70 + "│" + MatrixColors.RESET)

    # Footer
    lines.append(MatrixColors.GREEN_DIM + "└" + "─" * 70 + "┘" + MatrixColors.RESET)
    lines.append("")
    lines.append(MatrixColors.GREEN + "  Type 'pfe <command> --help' for detailed usage" + MatrixColors.RESET)
    lines.append("")

    return '\n'.join(lines)


__all__ = [
    "ZC_PIXEL_ART",
    "ZC_BLOCKY",
    "ZC_LARGE",
    "ZC_CLEAN",
    "ZC_BOX",
    "ZC_COMPACT",
    "ZC_BLOCK_LETTERS",
    "ZC_GLITCH_VARIANTS",
    "PFE_COMMANDS",
    "render_zc_logo",
    "render_boot_banner",
    "render_loading_sequence",
    "render_typing_effect",
    "render_commands_matrix",
    "LOGO_ASCII",
]