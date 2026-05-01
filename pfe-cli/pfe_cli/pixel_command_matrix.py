"""Matrix-style command reference rendering."""

from __future__ import annotations

from .terminal_theme import MatrixColors

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

    lines.append("")
    header_top = MatrixColors.GREEN_DIM + "┌" + "─" * 70 + "┐" + MatrixColors.RESET
    header_mid = (
        MatrixColors.GREEN_DIM
        + "│"
        + MatrixColors.RESET
        + "  "
        + MatrixColors.GREEN_BRIGHT
        + MatrixColors.BOLD
        + "AVAILABLE COMMANDS"
        + MatrixColors.RESET
        + "                                      "
        + MatrixColors.GREEN_DIM
        + "│"
        + MatrixColors.RESET
    )
    header_sep = MatrixColors.GREEN_DIM + "├" + "─" * 70 + "┤" + MatrixColors.RESET
    lines.append(header_top)
    lines.append(header_mid)
    lines.append(header_sep)

    for category, commands in PFE_COMMANDS.items():
        cat_text = "  " + category
        cat_padded = cat_text.ljust(68)
        cat_line = (
            MatrixColors.GREEN_DIM
            + "│"
            + MatrixColors.RESET
            + MatrixColors.AMBER
            + MatrixColors.BOLD
            + cat_padded
            + MatrixColors.RESET
            + MatrixColors.GREEN_DIM
            + "│"
            + MatrixColors.RESET
        )
        lines.append(cat_line)

        for cmd, desc in commands:
            cmd_padded = cmd.ljust(20)
            desc_padded = desc.ljust(45)
            line = (
                MatrixColors.GREEN_DIM
                + "│"
                + MatrixColors.RESET
                + "  "
                + MatrixColors.GREEN
                + cmd_padded
                + MatrixColors.RESET
                + " "
                + MatrixColors.GRAY
                + desc_padded
                + MatrixColors.RESET
                + MatrixColors.GREEN_DIM
                + "│"
                + MatrixColors.RESET
            )
            lines.append(line)

        lines.append(MatrixColors.GREEN_DIM + "│" + " " * 70 + "│" + MatrixColors.RESET)

    lines.append(MatrixColors.GREEN_DIM + "└" + "─" * 70 + "┘" + MatrixColors.RESET)
    lines.append("")
    lines.append(MatrixColors.GREEN + "  Type 'pfe <command> --help' for detailed usage" + MatrixColors.RESET)
    lines.append("")

    return "\n".join(lines)


__all__ = ["PFE_COMMANDS", "render_commands_matrix"]
