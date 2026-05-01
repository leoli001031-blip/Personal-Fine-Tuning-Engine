"""Adapter comparison command."""

from __future__ import annotations

import json
from typing import Any, Optional

import typer

from .adapter_lineage_helpers import _load_lineage_tracker
from .adapter_store_helpers import _load_adapter_store


def _get_version_info(store: Any, lineage_tracker: Any | None, version: str) -> dict[str, Any]:
    info: dict[str, Any] = {"version": version}
    try:
        manifest = store._read_manifest(version)
        info["state"] = manifest.get("state", "unknown")
        info["num_samples"] = manifest.get("num_samples", 0)
        info["created_at"] = manifest.get("created_at", "n/a")
        info["training_type"] = manifest.get("training_backend", "unknown")
        metrics = manifest.get("training_metrics", {})
        info["train_loss"] = metrics.get("train_loss")
        info["eval_loss"] = metrics.get("eval_loss")
        info["forget_detected"] = metrics.get("forget_detected", False)
    except Exception as exc:
        info["error"] = str(exc)

    if lineage_tracker:
        node = lineage_tracker.get_node(version)
        if node:
            info["lineage_training_type"] = node.training_type
            info["lineage_eval_score"] = node.eval_score
            info["lineage_forget_detected"] = node.forget_detected
            info["parent_version"] = node.parent_version
    return info


def _format_compare_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


def _compare_table(
    *,
    version_a: str,
    version_b: str,
    info_a: dict[str, Any],
    info_b: dict[str, Any],
    include_lineage: bool,
) -> None:
    keys = [
        "version",
        "state",
        "num_samples",
        "created_at",
        "training_type",
        "train_loss",
        "eval_loss",
        "forget_detected",
    ]
    if include_lineage:
        keys.extend(["parent_version", "lineage_eval_score"])

    max_key_len = max(len(k) for k in keys)
    typer.echo(f"{'Attribute':<{max_key_len}}  {version_a:<20}  {version_b:<20}")
    typer.echo("-" * (max_key_len + 2 + 20 + 2 + 20))

    for key in keys:
        val_a = _format_compare_value(info_a.get(key, "n/a"))
        val_b = _format_compare_value(info_b.get(key, "n/a"))
        typer.echo(f"{key:<{max_key_len}}  {val_a:<20}  {val_b:<20}")


def register_adapter_compare_command(adapter_app: typer.Typer) -> None:
    @adapter_app.command("compare")
    def compare(
        version_a: str = typer.Argument(..., help="First adapter version to compare."),
        version_b: str = typer.Argument(..., help="Second adapter version to compare."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        fmt: str = typer.Option("table", "--format", help="Output format: table, json."),
    ) -> None:
        """Compare two adapter versions."""

        store_factory = _load_adapter_store()
        if store_factory is None:
            typer.echo("[pfe] adapter compare: adapter store backend is not connected yet.")
            return

        store = store_factory() if callable(store_factory) else store_factory
        lineage_tracker = _load_lineage_tracker()
        info_a = _get_version_info(store, lineage_tracker, version_a)
        info_b = _get_version_info(store, lineage_tracker, version_b)

        if fmt == "json":
            typer.echo(
                json.dumps(
                    {
                        "version_a": info_a,
                        "version_b": info_b,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return

        _compare_table(
            version_a=version_a,
            version_b=version_b,
            info_a=info_a,
            info_b=info_b,
            include_lineage=bool(lineage_tracker),
        )


__all__ = ["register_adapter_compare_command"]
