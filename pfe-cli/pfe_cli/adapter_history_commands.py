"""Adapter training history command."""

from __future__ import annotations

from typing import Optional

import typer

from .adapter_lineage_helpers import _load_lineage_tracker
from .adapter_store_helpers import _load_adapter_store


def _lineage_eval_suffix(lineage_tracker: object | None, version: str) -> str:
    if not lineage_tracker:
        return ""
    node = lineage_tracker.get_node(version)  # type: ignore[attr-defined]
    if not node:
        return ""
    forget_info = " forget" if node.forget_detected else ""
    if node.eval_score is not None:
        forget_info += f" eval={node.eval_score:.3f}"
    return forget_info


def register_adapter_history_command(adapter_app: typer.Typer) -> None:
    @adapter_app.command("history")
    def history(
        limit: int = typer.Option(20, "--limit", min=1, help="Maximum versions to show."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show training history (alias for lineage --format list)."""

        store_factory = _load_adapter_store()
        if store_factory is None:
            typer.echo("[pfe] adapter history: adapter store backend is not connected yet.")
            return

        store = store_factory() if callable(store_factory) else store_factory
        lineage_tracker = _load_lineage_tracker()

        try:
            rows = store.list_version_records(limit=limit)
        except Exception as exc:
            typer.echo(f"Failed to list versions: {exc}")
            return

        if not rows:
            typer.echo("No adapter versions found.")
            return

        typer.echo(f"Training history (last {len(rows)} versions)")
        for row in rows:
            version = row.get("version", "n/a")
            state = row.get("state", "unknown")
            samples = row.get("num_samples", 0)
            created = row.get("created_at", "n/a")
            forget_info = _lineage_eval_suffix(lineage_tracker, version)
            typer.echo(f"- {version} [{state}] samples={samples} created={created}{forget_info}")


__all__ = ["register_adapter_history_command"]
