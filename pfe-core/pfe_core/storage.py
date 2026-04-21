"""Compatibility storage facade built on top of db.sqlite helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .db.sqlite import (
    adapter_rows,
    adapter_workspace_path,
    connect,
    deserialize_json,
    ensure_adapter_versions_schema,
    ensure_runtime_dirs,
    ensure_samples_schema,
    ensure_schema,
    ensure_signals_schema,
    list_samples,
    list_signals,
    mark_samples_used,
    record_signal,
    resolve_config_path,
    resolve_home,
    samples_db_path,
    save_samples,
    signals_db_path,
    status_snapshot,
    upsert_adapter_row,
    write_json,
    write_jsonl,
)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def samples_db_path_with_home(home: str | Path | None = None) -> Path:
    return samples_db_path(home)


def signals_db_path_with_home(home: str | Path | None = None) -> Path:
    return signals_db_path(home)


__all__ = [
    "adapter_rows",
    "adapter_workspace_path",
    "connect",
    "deserialize_json",
    "ensure_adapter_versions_schema",
    "ensure_runtime_dirs",
    "ensure_samples_schema",
    "ensure_schema",
    "ensure_signals_schema",
    "list_signals",
    "list_samples",
    "mark_samples_used",
    "record_signal",
    "resolve_config_path",
    "resolve_home",
    "samples_db_path",
    "samples_db_path_with_home",
    "save_samples",
    "signals_db_path",
    "signals_db_path_with_home",
    "status_snapshot",
    "upsert_adapter_row",
    "utcnow_iso",
    "write_json",
    "write_jsonl",
]
