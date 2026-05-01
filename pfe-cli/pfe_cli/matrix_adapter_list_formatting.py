"""Matrix terminal formatter for adapter lists."""

from __future__ import annotations

from typing import Any

from .terminal_theme import MatrixColors, draw_header, draw_table


def format_adapter_list_matrix(versions: list[dict[str, Any]], *, limit: int = 10) -> str:
    """Format adapter list in Matrix Green terminal style."""
    lines = []

    lines.append(draw_header("ADAPTER VERSIONS"))

    if not versions:
        lines.append(f"{MatrixColors.GRAY}    No adapters found{MatrixColors.RESET}")
        return "\n".join(lines)

    latest_version = None
    for v in versions:
        if v.get("latest") or v.get("state") == "promoted":
            latest_version = v.get("version")
            break

    if latest_version:
        lines.append(f"{MatrixColors.GREEN}    CURRENT LATEST: {latest_version}{MatrixColors.RESET}")
        lines.append("")

    headers = ["VERSION", "STATE", "SAMPLES", "FORMAT"]
    rows = []

    for v in versions[:limit]:
        version = v.get("version", "n/a")
        state = v.get("state", "unknown")
        samples = str(v.get("num_training_samples", 0))
        fmt = v.get("artifact_format", "unknown")

        if version == latest_version:
            version = f"{MatrixColors.GREEN}*{MatrixColors.RESET} {version}"

        rows.append([version, state, samples, fmt])

    lines.append(draw_table(headers, rows))
    lines.append("")

    lines.append(f"{MatrixColors.GREEN_DIM}    Showing {min(limit, len(versions))} of {len(versions)} versions{MatrixColors.RESET}")

    return "\n".join(lines)


__all__ = ["format_adapter_list_matrix"]
