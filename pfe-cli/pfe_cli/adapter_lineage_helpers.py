"""Adapter lineage rendering helpers."""

from __future__ import annotations

from typing import Any


def _load_lineage_tracker() -> Any | None:
    """Load the adapter lineage tracker if available."""
    try:
        from pfe_core.trainer.adapter_lineage import get_lineage_tracker

        return get_lineage_tracker()
    except Exception:
        return None


def _format_tree_line(version: str, node: dict[str, Any] | None, prefix: str = "", is_last: bool = True) -> list[str]:
    """Format a single line of the lineage tree."""
    lines: list[str] = []
    connector = "└── " if is_last else "├── "
    if node:
        state = node.get("state", "unknown")
        forget = " forget" if node.get("forget_detected") else ""
        samples = node.get("num_samples", 0)
        eval_score = node.get("eval_score")
        eval_str = f" eval={eval_score:.3f}" if eval_score is not None else ""
        line = f"{prefix}{connector}{version} [{state}]{forget} samples={samples}{eval_str}"
    else:
        line = f"{prefix}{connector}{version} [not tracked]"
    lines.append(line)
    return lines


def _render_tree(tree: dict[str, Any], prefix: str = "") -> list[str]:
    """Recursively render a lineage tree to text lines."""
    lines: list[str] = []
    version = tree.get("version", "unknown")
    node = tree.get("node")
    children = tree.get("children", [])

    if prefix == "":
        if node:
            state = node.get("state", "unknown")
            forget = " forget" if node.get("forget_detected") else ""
            samples = node.get("num_samples", 0)
            eval_score = node.get("eval_score")
            eval_str = f" eval={eval_score:.3f}" if eval_score is not None else ""
            lines.append(f"{version} [{state}]{forget} samples={samples}{eval_str}")
        else:
            lines.append(f"{version} [not tracked]")
    else:
        lines.extend(_format_tree_line(version, node, prefix=prefix[:-4], is_last=prefix.endswith("    ")))

    for index, child in enumerate(children):
        is_last_child = index == len(children) - 1
        child_prefix = prefix + ("    " if is_last_child else "│   ")
        lines.extend(_render_tree(child, prefix=child_prefix))
    return lines


__all__ = ["_format_tree_line", "_load_lineage_tracker", "_render_tree"]
