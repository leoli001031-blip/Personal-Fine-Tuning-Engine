"""Adapter management commands for the PFE CLI."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import typer

from . import formatters_matrix


@dataclass(frozen=True)
class AdapterCommandContext:
    """Shared CLI context for adapter operations."""

    workspace: str | None = None
    config_path: str | None = None


adapter_app = typer.Typer(
    help=(
        "Adapter lifecycle management in strict_local mode. "
        "Only 'promote' updates latest; OpenAI compatibility applies to inference only."
    ),
    no_args_is_help=True,
)


def _load_adapter_store() -> Any | None:
    """Resolve the future high-level adapter store service if it exists."""

    for module_name in (
        "pfe_core.adapter_store.store",
        "pfe_core.services.adapter_store",
        "pfe_core.pipeline.adapter_store",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for attr_name in ("AdapterStore", "get_adapter_store", "create_adapter_store"):
            candidate = getattr(module, attr_name, None)
            if candidate is not None:
                return candidate
    return None


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _format_lifecycle_summary(result: Any) -> list[str] | None:
    mapping = result if isinstance(result, dict) else None
    if mapping is None and isinstance(result, str):
        stripped = result.strip()
        match = re.match(r"^Promoted\s+(.+?)\s+to\s+latest\.$", stripped)
        if match is not None:
            return [f"latest: {match.group(1)}", "lifecycle: promoted"]
        if stripped == "No adapter versions found.":
            return [stripped]

        version_lines: list[tuple[str, str, bool, str, str]] = []
        latest_version = None
        for raw_line in stripped.splitlines():
            line = raw_line.rstrip()
            match = re.match(
                r"^([* ])\s+([^\s]+)\s+state=([^\s]+)\s+samples=([^\s]+)\s+format=([^\s]+)$",
                line,
            )
            if match is None:
                continue
            marker, version, lifecycle, samples, artifact_format = match.groups()
            latest_flag = marker == "*"
            if latest_flag:
                latest_version = version
            version_lines.append((version, lifecycle, latest_flag, samples, artifact_format))
        if version_lines:
            lines = ["Adapter versions"]
            if latest_version is not None:
                lines.append(f"latest: {latest_version}")
            for version, lifecycle, latest_flag, samples, artifact_format in version_lines:
                suffix = " | latest=yes" if latest_flag else ""
                lines.append(
                    f"- {version} | lifecycle={lifecycle}{suffix} | samples={samples} | format={artifact_format}"
                )
            return lines
        return None

    if mapping is None:
        return None

    if "versions" in mapping and isinstance(mapping["versions"], Sequence):
        lines = ["Adapter versions"]
        latest = None
        for item in mapping["versions"]:
            item_map = item if isinstance(item, dict) else None
            if item_map is not None and item_map.get("latest"):
                latest = item_map.get("version")
                break
        if latest is not None:
            lines.append(f"latest: {_format_value(latest)}")
        for item in mapping["versions"]:
            item_map = item if isinstance(item, dict) else None
            if item_map is None:
                lines.append(f"- {_format_value(item)}")
                continue
            version = item_map.get("version", "n/a")
            lifecycle = item_map.get("state", item_map.get("status", "n/a"))
            latest_flag = item_map.get("latest", False)
            suffix = " | latest=yes" if latest_flag else ""
            lines.append(
                f"- {version} | lifecycle={_format_value(lifecycle)}{suffix} | "
                f"samples={_format_value(item_map.get('num_samples', 'n/a'))} | "
                f"format={_format_value(item_map.get('artifact_format', 'n/a'))}"
            )
        return lines

    if "version" in mapping or "state" in mapping or "latest" in mapping:
        version = mapping.get("version")
        state = mapping.get("state", mapping.get("status"))
        latest = mapping.get("latest")
        lines = []
        if version is not None:
            lines.append(f"latest: {_format_value(version)}")
        if state is not None:
            lines.append(f"lifecycle: {_format_value(state)}")
        if latest is not None:
            lines.append(f"latest_pointer: {_format_value(latest)}")
        return lines or None

    return None


def _echo_result(result: Any) -> None:
    # Matrix theme - default style for versions list
    mapping = result if isinstance(result, dict) else None
    if mapping is not None and "versions" in mapping and isinstance(mapping["versions"], Sequence):
        versions = [item if isinstance(item, dict) else {} for item in mapping["versions"]]
        typer.echo(formatters_matrix.format_adapter_list_matrix(versions))
        return

    # Fallback for other result types
    lifecycle_lines = _format_lifecycle_summary(result)
    if lifecycle_lines is not None:
        for line in lifecycle_lines:
            typer.echo(line)
        return

    if mapping is not None:
        for key in sorted(mapping):
            typer.echo(f"{key.replace('_', ' ')}: {_format_value(mapping[key])}")
        return

    typer.echo(_format_value(result))


def _call_store(method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Dispatch to the future adapter store interface when it is available."""

    store_factory = _load_adapter_store()
    if store_factory is None:
        typer.echo(
            f"[pfe] adapter {method_name}: CLI skeleton is wired, but the adapter store backend is not connected yet."
        )
        return None

    store = store_factory() if callable(store_factory) else store_factory
    method = getattr(store, method_name, None)
    if method is None:
        raise typer.BadParameter(
            f"Connected adapter store does not provide '{method_name}'. "
            "Expected a future high-level store interface from pfe_core."
        )
    try:
        return method(*args, **kwargs)
    except typer.Exit:
        raise
    except Exception as exc:
        name = exc.__class__.__name__.lower()
        if "adaptererror" in name:
            typer.secho(f"Adapter error: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if "trainingerror" in name:
            typer.secho(f"Training error: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        raise


@adapter_app.command("list")
def list_versions(
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum versions to display."),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace",
        help="Optional workspace or tenant label for future multi-workspace support.",
    ),
) -> None:
    """List adapter versions and their lifecycle state."""

    result = _call_store("list_versions", limit=limit, workspace=workspace)
    if result is not None:
        _echo_result(result)


@adapter_app.command("promote")
def promote(
    version: str = typer.Argument(..., help="Adapter version to promote to latest."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Promote an adapter version. This is the only supported latest update path."""

    result = _call_store("promote", version, workspace=workspace)
    if result is not None:
        _echo_result(result)


@adapter_app.command("rollback")
def rollback(
    version: str = typer.Argument(..., help="Adapter version or relative index to roll back to."),
    workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
) -> None:
    """Rollback latest to a prior adapter version."""

    result = _call_store("rollback", version, workspace=workspace)
    if result is not None:
        _echo_result(result)


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

    # For root call, don't add prefix/connector
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

    for i, child in enumerate(children):
        is_last_child = i == len(children) - 1
        child_prefix = prefix + ("    " if is_last_child else "│   ")
        lines.extend(_render_tree(child, prefix=child_prefix))
    return lines


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

    # Resolve version
    target_version = version
    if not target_version or target_version == "latest":
        try:
            target_version = store.current_latest_version()
        except Exception:
            pass
    if not target_version:
        typer.echo("No adapter version found.")
        return

    lineage_tracker = _load_lineage_tracker()

    if fmt == "json":
        if lineage_tracker:
            tree = lineage_tracker.get_lineage_tree(target_version)
            typer.echo(json.dumps(tree, ensure_ascii=False, indent=2, default=str))
        else:
            # Fallback: read manifest for parent info
            try:
                manifest = store._read_manifest(target_version)
                typer.echo(json.dumps({
                    "version": target_version,
                    "parent": manifest.get("metadata", {}).get("training", {}).get("parent_version"),
                    "node": manifest,
                    "children": [],
                }, ensure_ascii=False, indent=2, default=str))
            except Exception as e:
                typer.echo(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        return

    if fmt == "list":
        if lineage_tracker:
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
        else:
            typer.echo("Lineage tracker not available.")
        return

    # tree format (default)
    if lineage_tracker:
        tree = lineage_tracker.get_lineage_tree(target_version)
        lines = _render_tree(tree)
        for line in lines:
            typer.echo(line)
    else:
        typer.echo("Lineage tracker not available. Use --format json for manifest fallback.")


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
    except Exception as e:
        typer.echo(f"Failed to list versions: {e}")
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

        forget_info = ""
        if lineage_tracker:
            node = lineage_tracker.get_node(version)
            if node:
                forget_info = " forget" if node.forget_detected else ""
                if node.eval_score is not None:
                    forget_info += f" eval={node.eval_score:.3f}"

        typer.echo(f"- {version} [{state}] samples={samples} created={created}{forget_info}")


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

    def _get_version_info(version: str) -> dict[str, Any]:
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
        except Exception as e:
            info["error"] = str(e)

        # Augment with lineage data
        if lineage_tracker:
            node = lineage_tracker.get_node(version)
            if node:
                info["lineage_training_type"] = node.training_type
                info["lineage_eval_score"] = node.eval_score
                info["lineage_forget_detected"] = node.forget_detected
                info["parent_version"] = node.parent_version

        return info

    info_a = _get_version_info(version_a)
    info_b = _get_version_info(version_b)

    if fmt == "json":
        typer.echo(json.dumps({
            "version_a": info_a,
            "version_b": info_b,
        }, ensure_ascii=False, indent=2, default=str))
        return

    # table format
    keys = ["version", "state", "num_samples", "created_at", "training_type", "train_loss", "eval_loss", "forget_detected"]
    if lineage_tracker:
        keys.extend(["parent_version", "lineage_eval_score"])

    max_key_len = max(len(k) for k in keys)
    typer.echo(f"{'Attribute':<{max_key_len}}  {version_a:<20}  {version_b:<20}")
    typer.echo("-" * (max_key_len + 2 + 20 + 2 + 20))

    for key in keys:
        val_a = info_a.get(key, "n/a")
        val_b = info_b.get(key, "n/a")
        if isinstance(val_a, float):
            val_a = f"{val_a:.4f}"
        if isinstance(val_b, float):
            val_b = f"{val_b:.4f}"
        if val_a is None:
            val_a = "n/a"
        if val_b is None:
            val_b = "n/a"
        typer.echo(f"{key:<{max_key_len}}  {str(val_a):<20}  {str(val_b):<20}")
