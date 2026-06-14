"""Adapter snapshot lookup helpers for CLI summaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .adapter_snapshot_deps import AdapterSnapshotDeps
from .adapter_snapshot_row import snapshot_from_row


def lookup_adapter_snapshot(
    version: str | None,
    *,
    workspace: str | None = None,
    deps: AdapterSnapshotDeps,
) -> dict[str, Any] | None:
    if version is None:
        return None

    store = deps.optional_module_call("pfe_core.adapter_store.store", "create_adapter_store", workspace=workspace)
    if store is None:
        return None

    list_records = getattr(store, "list_version_records", None)
    if not callable(list_records):
        return None

    try:
        rows = list_records(limit=100)
    except Exception:
        return None
    if not isinstance(rows, Sequence):
        return None

    latest_version = None
    if str(version) == "latest":
        current_latest = getattr(store, "current_latest_version", None)
        if callable(current_latest):
            try:
                latest_version = current_latest()
            except Exception:
                latest_version = None
    target_version = latest_version or str(version)

    for row in rows:
        row_map = deps.coerce_mapping(row)
        if row_map is None:
            continue
        if str(row_map.get("version")) == target_version:
            return snapshot_from_row(row_map, latest_version=latest_version, deps=deps)
    return None


def lookup_recent_adapter_snapshot(
    *,
    workspace: str | None = None,
    deps: AdapterSnapshotDeps,
) -> dict[str, Any] | None:
    store = deps.optional_module_call("pfe_core.adapter_store.store", "create_adapter_store", workspace=workspace)
    if store is None:
        return None

    list_records = getattr(store, "list_version_records", None)
    if not callable(list_records):
        return None

    try:
        rows = list_records(limit=1)
    except Exception:
        return None
    if not isinstance(rows, Sequence) or not rows:
        return None

    row_map = deps.coerce_mapping(rows[0])
    if row_map is None:
        return None

    latest_version = None
    current_latest = getattr(store, "current_latest_version", None)
    if callable(current_latest):
        try:
            latest_version = current_latest()
        except Exception:
            latest_version = None

    return snapshot_from_row(row_map, latest_version=latest_version, deps=deps)
