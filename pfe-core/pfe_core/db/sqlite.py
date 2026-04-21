"""SQLite connection and storage helpers for PFE."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..errors import DataError
from .schema import (
    ADAPTER_VERSIONS_TABLE,
    DEFAULT_WORKSPACE,
    SIGNALS_TABLE,
    SAMPLES_TABLE,
    adapter_versions_column_definitions,
    build_schema_statements,
    signals_column_definitions,
)

CONFIG_FILENAME = "config.toml"


def _discover_workspace_home(start: str | Path | None = None) -> Path | None:
    root = Path(start).expanduser() if start is not None else Path.cwd()
    candidates = [root, *root.parents]
    for candidate_root in candidates:
        candidate = candidate_root / ".pfe"
        if candidate.is_dir():
            return candidate
    return None


def resolve_home(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    env_home = os.getenv("PFE_HOME")
    if env_home:
        return Path(env_home).expanduser()
    workspace_home = _discover_workspace_home()
    if workspace_home is not None:
        return workspace_home
    return Path.home() / ".pfe"


def resolve_config_path(path: str | Path | None = None, home: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    return resolve_home(home) / CONFIG_FILENAME


def ensure_runtime_dirs(home: str | Path | None = None) -> Path:
    root = resolve_home(home)
    for relative in ("data", "adapters", "adapters/user_default", "cache", "logs"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root


def adapter_workspace_path(home: str | Path | None = None, workspace: str = DEFAULT_WORKSPACE) -> Path:
    return ensure_runtime_dirs(home) / "adapters" / workspace


def samples_db_path(home: str | Path | None = None) -> Path:
    return ensure_runtime_dirs(home) / "data" / "samples.db"


def signals_db_path(home: str | Path | None = None) -> Path:
    return ensure_runtime_dirs(home) / "data" / "signals.db"


def _sqlite_connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA wal_autocheckpoint=1000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def connect(db_path: str | Path) -> sqlite3.Connection:
    return _sqlite_connect(db_path)


def serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def deserialize_json(value: str | bytes | None) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def _ensure_columns(connection: sqlite3.Connection, table: str, column_definitions: list[str]) -> None:
    existing = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for definition in column_definitions:
        name = definition.split()[0]
        if name in existing:
            continue
        try:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")
            existing.add(name)
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
            existing.add(name)


def ensure_schema(connection: sqlite3.Connection) -> None:
    for statement in build_schema_statements():
        connection.execute(statement)
    _ensure_columns(connection, SIGNALS_TABLE, signals_column_definitions())
    _ensure_columns(connection, ADAPTER_VERSIONS_TABLE, adapter_versions_column_definitions())


def ensure_signals_schema(connection: sqlite3.Connection) -> None:
    ensure_schema(connection)


def ensure_samples_schema(connection: sqlite3.Connection) -> None:
    ensure_schema(connection)


def ensure_adapter_versions_schema(connection: sqlite3.Connection) -> None:
    ensure_schema(connection)


def initialize_database(path_or_connection: str | Path | sqlite3.Connection) -> sqlite3.Connection:
    connection = path_or_connection if isinstance(path_or_connection, sqlite3.Connection) else connect(path_or_connection)
    ensure_schema(connection)
    connection.commit()
    return connection


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _from_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


class SQLiteDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.connection = initialize_database(self.path)

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        cursor = self.connection.execute(sql, parameters)
        self.connection.commit()
        return cursor

    def fetchone(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self.connection.execute(sql, parameters).fetchone()

    def fetchall(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return self.connection.execute(sql, parameters).fetchall()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise


def save_samples(samples: Iterator[dict[str, Any]] | list[dict[str, Any]], home: str | Path | None = None) -> int:
    conn = initialize_database(samples_db_path(home))
    try:
        count = 0
        for sample in samples:
            payload = dict(sample)
            sample_id = payload.get("sample_id") or payload.get("id")
            if not sample_id:
                raise DataError("Training sample is missing sample_id")
            metadata = dict(payload.get("metadata") or {})
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {SAMPLES_TABLE} (
                    id, sample_type, instruction, chosen, rejected, score, source,
                    source_event_ids, source_adapter_version, created_at, used_in_version, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    payload["sample_type"],
                    payload["instruction"],
                    payload["chosen"],
                    payload.get("rejected"),
                    float(payload.get("score", 0.0)),
                    payload["source"],
                    _to_json(payload.get("source_event_ids") or []),
                    payload.get("source_adapter_version"),
                    payload.get("created_at") or _now_iso(),
                    payload.get("used_in_version"),
                    _to_json(metadata),
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def list_samples(
    home: str | Path | None = None,
    *,
    sample_type: str | None = None,
    dataset_split: str | None = None,
    include_used: bool = True,
    exclude_test: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    conn = initialize_database(samples_db_path(home))
    try:
        rows = conn.execute(f"SELECT * FROM {SAMPLES_TABLE} ORDER BY created_at ASC").fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            metadata = _from_json(row["metadata"], {})
            split = metadata.get("dataset_split")
            if sample_type and row["sample_type"] != sample_type:
                continue
            if dataset_split and split != dataset_split:
                continue
            if exclude_test and split == "test":
                continue
            if not include_used and row["used_in_version"]:
                continue
            results.append(
                {
                    "sample_id": row["id"],
                    "sample_type": row["sample_type"],
                    "instruction": row["instruction"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                    "score": row["score"],
                    "source": row["source"],
                    "source_event_ids": _from_json(row["source_event_ids"], []),
                    "source_adapter_version": row["source_adapter_version"],
                    "created_at": row["created_at"],
                    "used_in_version": row["used_in_version"],
                    "metadata": metadata,
                }
            )
        if limit is not None:
            results = results[:limit]
        return results
    finally:
        conn.close()


def mark_samples_used(sample_ids: Iterator[str] | list[str], version: str, home: str | Path | None = None) -> None:
    ids = [sample_id for sample_id in sample_ids if sample_id]
    if not ids:
        return
    conn = initialize_database(samples_db_path(home))
    try:
        conn.executemany(
            f"UPDATE {SAMPLES_TABLE} SET used_in_version = COALESCE(used_in_version, ?) WHERE id = ?",
            [(version, sample_id) for sample_id in ids],
        )
        conn.commit()
    finally:
        conn.close()


def adapter_rows(home: str | Path | None = None, workspace: str = DEFAULT_WORKSPACE) -> list[dict[str, Any]]:
    conn = initialize_database(samples_db_path(home))
    try:
        rows = conn.execute(
            f"SELECT * FROM {ADAPTER_VERSIONS_TABLE} WHERE workspace = ? ORDER BY created_at DESC",
            (workspace,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def upsert_adapter_row(payload: dict[str, Any], home: str | Path | None = None) -> None:
    conn = initialize_database(samples_db_path(home))
    try:
        conn.execute(
            f"""
            INSERT INTO {ADAPTER_VERSIONS_TABLE} (
                version, workspace, base_model, created_at, updated_at, num_samples,
                state, artifact_format, adapter_dir, manifest_path, artifact_path,
                training_config, eval_report, metrics, promoted_at, archived_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(version) DO UPDATE SET
                workspace = excluded.workspace,
                base_model = excluded.base_model,
                updated_at = excluded.updated_at,
                num_samples = excluded.num_samples,
                state = excluded.state,
                artifact_format = excluded.artifact_format,
                adapter_dir = excluded.adapter_dir,
                manifest_path = excluded.manifest_path,
                artifact_path = excluded.artifact_path,
                training_config = excluded.training_config,
                eval_report = excluded.eval_report,
                metrics = excluded.metrics,
                promoted_at = excluded.promoted_at,
                archived_at = excluded.archived_at,
                metadata = excluded.metadata
            """,
            (
                payload["version"],
                payload.get("workspace") or DEFAULT_WORKSPACE,
                payload["base_model"],
                payload["created_at"],
                payload["updated_at"],
                int(payload.get("num_samples", 0)),
                payload["state"],
                payload["artifact_format"],
                payload.get("adapter_dir", ""),
                payload.get("manifest_path"),
                payload.get("artifact_path"),
                _to_json(payload.get("training_config") or {}),
                _to_json(payload.get("eval_report")),
                _to_json(payload.get("metrics")),
                payload.get("promoted_at"),
                payload.get("archived_at"),
                _to_json(payload.get("metadata") or {}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def record_signal(payload: dict[str, Any], home: str | Path | None = None) -> None:
    for field in ("event_id", "request_id", "session_id"):
        if not payload.get(field):
            raise DataError(f"Signal payload missing required field: {field}")
    source_event_ids = payload.get("source_event_ids")
    if not isinstance(source_event_ids, list) or not source_event_ids:
        source_event_ids = [payload.get("source_event_id") or payload["event_id"]]
    event_chain_ids = payload.get("event_chain_ids")
    if not isinstance(event_chain_ids, list) or not event_chain_ids:
        event_chain_ids = list(source_event_ids)
    lineage = {
        "event_id": payload["event_id"],
        "source_event_id": payload.get("source_event_id") or payload["event_id"],
        "source_event_ids": list(source_event_ids),
        "event_chain_ids": list(event_chain_ids),
        "request_id": payload["request_id"],
        "session_id": payload["session_id"],
        "parent_event_id": payload.get("parent_event_id"),
        "adapter_version": payload.get("adapter_version"),
        "event_type": payload.get("event_type", "chat"),
        "scenario": payload.get("scenario"),
    }
    conn = initialize_database(signals_db_path(home))
    try:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {SIGNALS_TABLE} (
                id, source_event_id, request_id, session_id, parent_event_id, source_event_ids,
                event_chain_ids, adapter_version, event_type, timestamp, context, model_output,
                user_input, action_detail, user_action, lineage, metadata, processed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["event_id"],
                payload.get("source_event_id") or payload["event_id"],
                payload["request_id"],
                payload["session_id"],
                payload.get("parent_event_id"),
                _to_json(source_event_ids),
                _to_json(event_chain_ids),
                payload.get("adapter_version"),
                payload.get("event_type", "chat"),
                payload.get("timestamp") or _now_iso(),
                payload.get("context"),
                payload.get("model_output"),
                payload.get("user_input"),
                _to_json(payload.get("action_detail") or {}),
                _to_json(
                    {
                        "user_action": payload.get("user_action"),
                    }
                ),
                _to_json(lineage),
                _to_json(
                    {
                        "scenario": payload.get("scenario"),
                        **dict(payload.get("metadata") or {}),
                    }
                ),
                False,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_signals(
    home: str | Path | None = None,
    *,
    request_id: str | None = None,
    session_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    conn = initialize_database(signals_db_path(home))
    try:
        rows = conn.execute(
            f"SELECT * FROM {SIGNALS_TABLE} ORDER BY timestamp DESC"
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            if request_id and row["request_id"] != request_id:
                continue
            if session_id and row["session_id"] != session_id:
                continue
            action_payload = _from_json(row["user_action"], {})
            metadata = _from_json(row["metadata"], {})
            lineage = _from_json(row["lineage"], {})
            results.append(
                {
                    "signal_id": row["id"],
                    "source_event_id": row["source_event_id"],
                    "request_id": row["request_id"],
                    "session_id": row["session_id"],
                    "parent_event_id": row["parent_event_id"],
                    "source_event_ids": _from_json(row["source_event_ids"], []),
                    "event_chain_ids": _from_json(row["event_chain_ids"], []),
                    "adapter_version": row["adapter_version"],
                    "event_type": row["event_type"],
                    "timestamp": row["timestamp"],
                    "context": row["context"],
                    "model_output": row["model_output"],
                    "user_input": row["user_input"],
                    "action_detail": _from_json(row["action_detail"], {}),
                    "user_action": action_payload.get("user_action"),
                    "metadata": metadata,
                    "lineage": lineage,
                    "processed": bool(row["processed"]),
                }
            )
            if limit is not None and len(results) >= limit:
                break
        return results
    finally:
        conn.close()


def status_snapshot(home: str | Path | None = None, workspace: str = DEFAULT_WORKSPACE) -> dict[str, Any]:
    train = list_samples(home, dataset_split="train")
    val = list_samples(home, dataset_split="val")
    test = list_samples(home, dataset_split="test")
    conn = initialize_database(signals_db_path(home))
    try:
        signal_rows = conn.execute(
            f"SELECT * FROM {SIGNALS_TABLE} ORDER BY timestamp ASC"
        ).fetchall()
        signal_count = len(signal_rows)
        complete_chain_count = sum(
            1
            for row in signal_rows
            if row["source_event_id"] and row["request_id"] and row["session_id"]
        )
        processed_count = sum(1 for row in signal_rows if bool(row["processed"]))
        latest_signal = list_signals(home, limit=1)
        signal_samples = [sample for sample in list_samples(home) if sample.get("source") == "signal"]
        signal_sample_counts = {
            "train": sum(1 for sample in signal_samples if (sample.get("metadata") or {}).get("dataset_split") == "train"),
            "val": sum(1 for sample in signal_samples if (sample.get("metadata") or {}).get("dataset_split") == "val"),
            "test": sum(1 for sample in signal_samples if (sample.get("metadata") or {}).get("dataset_split") == "test"),
        }
        signal_sample_details = [
            {
                "sample_id": sample.get("sample_id"),
                "sample_type": sample.get("sample_type"),
                "dataset_split": (sample.get("metadata") or {}).get("dataset_split"),
                "source_event_ids": list(sample.get("source_event_ids") or []),
                "source_adapter_version": sample.get("source_adapter_version"),
                "score": sample.get("score"),
                "used_in_version": sample.get("used_in_version"),
            }
            for sample in signal_samples[-5:]
        ]
        return {
            "home": str(resolve_home(home)),
            "sample_counts": {"train": len(train), "val": len(val), "test": len(test)},
            "adapter_versions": len(adapter_rows(home, workspace)),
            "signal_count": signal_count,
            "signal_summary": {
                "state": "ready" if signal_count > 0 else "idle",
                "event_chain_ready": complete_chain_count > 0,
                "event_chain_complete_count": complete_chain_count,
                "event_chain_complete_ratio": round(complete_chain_count / signal_count, 3) if signal_count else 0.0,
                "processed_count": processed_count,
                "latest_signal_id": latest_signal[0]["signal_id"] if latest_signal else None,
                "source_event_id_count": sum(1 for row in signal_rows if bool(row["source_event_id"])),
                "request_id_count": sum(1 for row in signal_rows if bool(row["request_id"])),
                "session_id_count": sum(1 for row in signal_rows if bool(row["session_id"])),
            },
            "signal_sample_count": len(signal_samples),
            "signal_sample_counts": signal_sample_counts,
            "signal_sample_details": signal_sample_details,
            "latest_signal": latest_signal[0] if latest_signal else {},
        }
    finally:
        conn.close()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
