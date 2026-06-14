"""Adapter lineage view command."""

from __future__ import annotations

import json
from typing import Any, Optional

import typer

from .adapter_lineage_helpers import _load_lineage_tracker, _render_tree
from .adapter_store_helpers import _load_adapter_store


def _resolve_target_version(store: Any, version: str | None) -> str | None:
    if version and version != "latest":
        return version
    try:
        return store.current_latest_version()
    except Exception:
        return version


def _lineage_json(store: Any, lineage_tracker: Any | None, target_version: str) -> None:
    if lineage_tracker:
        tree = lineage_tracker.get_lineage_tree(target_version)
        typer.echo(json.dumps(tree, ensure_ascii=False, indent=2, default=str))
        return
    try:
        manifest = store._read_manifest(target_version)
    except Exception as exc:
        typer.echo(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        return
    typer.echo(
        json.dumps(
            {
                "version": target_version,
                "parent": manifest.get("metadata", {}).get("training", {}).get("parent_version"),
                "node": manifest,
                "children": [],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def _lineage_list(lineage_tracker: Any | None, target_version: str) -> None:
    if not lineage_tracker:
        typer.echo("Lineage tracker not available.")
        return
    chain = lineage_tracker.get_lineage(target_version)
    if not chain:
        typer.echo(f"No lineage found for {target_version}.")
        return
    typer.echo(f"Lineage for {target_version}")
    for node in chain:
        forget = " forget" if node.forget_detected else ""
        eval_str = f" eval={node.eval_score:.3f}" if node.eval_score is not None else ""
        typer.echo(
            f"- {node.version} [{node.state}]{forget} "
            f"type={node.training_type} samples={node.num_samples}{eval_str}"
        )


def _lineage_tree(lineage_tracker: Any | None, target_version: str) -> None:
    if not lineage_tracker:
        typer.echo("Lineage tracker not available. Use --format json for manifest fallback.")
        return
    tree = lineage_tracker.get_lineage_tree(target_version)
    for line in _render_tree(tree):
        typer.echo(line)


def register_adapter_lineage_view_command(adapter_app: typer.Typer) -> None:
    @adapter_app.command("lineage")
    def lineage(
        version: Optional[str] = typer.Argument(None, help="Adapter version to show lineage for (default: latest)."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        fmt: str = typer.Option("tree", "--format", help="Output format: tree, list, json."),
    ) -> None:
        """Show adapter version lineage (parent-child training history)."""

        store_factory = _load_adapter_store()
        if store_factory is None:
            typer.echo("[pfe] adapter lineage: adapter store backend is not connected yet.")
            return

        store = store_factory() if callable(store_factory) else store_factory
        target_version = _resolve_target_version(store, version)
        if not target_version:
            typer.echo("No adapter version found.")
            return

        lineage_tracker = _load_lineage_tracker()
        if fmt == "json":
            _lineage_json(store, lineage_tracker, target_version)
        elif fmt == "list":
            _lineage_list(lineage_tracker, target_version)
        else:
            _lineage_tree(lineage_tracker, target_version)


__all__ = ["register_adapter_lineage_view_command"]
