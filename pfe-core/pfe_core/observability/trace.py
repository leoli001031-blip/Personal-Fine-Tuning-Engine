"""Signal and version tracing for full-chain observability.

Records the lifecycle of a signal from collection through training
evaluation and promotion, as well as version-level upstream signal lineage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..storage import resolve_home


@dataclass
class TraceNode:
    """A single node in a signal trace."""

    node: str
    timestamp: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "timestamp": self.timestamp,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


@dataclass
class SignalTrace:
    """Complete trace for a single signal."""

    signal_id: str
    nodes: list[TraceNode] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_node(self, node: str, status: str, metadata: dict[str, Any] | None = None) -> None:
        self.nodes.append(
            TraceNode(
                node=node,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=status,
                metadata=dict(metadata) if metadata else {},
            )
        )
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "nodes": [n.to_dict() for n in self.nodes],
        }


@dataclass
class VersionTrace:
    """Trace for an adapter version, aggregating all upstream signal traces."""

    version: str
    signal_traces: list[SignalTrace] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "signal_count": len(self.signal_traces),
            "signal_traces": [st.to_dict() for st in self.signal_traces],
        }


class TraceStore:
    """Persistent store for signal and version traces."""

    def __init__(self, store_dir: Path | str | None = None) -> None:
        if store_dir is None:
            store_dir = resolve_home() / "data" / "traces"
        self.store_dir = Path(store_dir).expanduser()
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _signal_path(self, signal_id: str) -> Path:
        return self.store_dir / f"signal_{signal_id}.json"

    def _version_path(self, version: str) -> Path:
        return self.store_dir / f"version_{version}.json"

    def save_signal_trace(self, trace: SignalTrace) -> None:
        path = self._signal_path(trace.signal_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_signal_trace(self, signal_id: str) -> SignalTrace | None:
        path = self._signal_path(signal_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        trace = SignalTrace(signal_id=data["signal_id"])
        trace.created_at = data.get("created_at", "")
        trace.updated_at = data.get("updated_at", "")
        trace.nodes = [
            TraceNode(
                node=n["node"],
                timestamp=n["timestamp"],
                status=n["status"],
                metadata=dict(n.get("metadata", {})),
            )
            for n in data.get("nodes", [])
        ]
        return trace

    def save_version_trace(self, trace: VersionTrace) -> None:
        path = self._version_path(trace.version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load_version_trace(self, version: str) -> VersionTrace | None:
        path = self._version_path(version)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        trace = VersionTrace(version=data["version"])
        trace.created_at = data.get("created_at", "")
        trace.signal_traces = [
            SignalTrace(
                signal_id=st["signal_id"],
                nodes=[
                    TraceNode(
                        node=n["node"],
                        timestamp=n["timestamp"],
                        status=n["status"],
                        metadata=dict(n.get("metadata", {})),
                    )
                    for n in st.get("nodes", [])
                ],
                created_at=st.get("created_at", ""),
                updated_at=st.get("updated_at", ""),
            )
            for st in data.get("signal_traces", [])
        ]
        return trace

    def list_recent_signal_ids(self, limit: int = 20) -> list[str]:
        paths = sorted(self.store_dir.glob("signal_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem.replace("signal_", "") for p in paths[:limit]]

    def list_recent_version_ids(self, limit: int = 20) -> list[str]:
        paths = sorted(self.store_dir.glob("version_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem.replace("version_", "") for p in paths[:limit]]


# Global default store instance
_default_store: TraceStore | None = None


def _get_default_store() -> TraceStore:
    global _default_store
    if _default_store is None:
        _default_store = TraceStore()
    return _default_store


def trace_signal(signal_id: str) -> SignalTrace:
    """Get or create a ``SignalTrace`` for *signal_id*."""
    store = _get_default_store()
    trace = store.load_signal_trace(signal_id)
    if trace is None:
        trace = SignalTrace(signal_id=signal_id)
        store.save_signal_trace(trace)
    return trace


def trace_version(version: str) -> VersionTrace:
    """Get or create a ``VersionTrace`` for *version*."""
    store = _get_default_store()
    vt = store.load_version_trace(version)
    if vt is None:
        vt = VersionTrace(version=version)
        store.save_version_trace(vt)
    return vt


def record_signal_node(
    signal_id: str,
    node: str,
    status: str,
    metadata: dict[str, Any] | None = None,
) -> SignalTrace:
    """Record a trace node for a signal and persist it."""
    trace = trace_signal(signal_id)
    trace.add_node(node, status, metadata)
    _get_default_store().save_signal_trace(trace)
    return trace


def append_signal_to_version(version: str, signal_id: str) -> VersionTrace:
    """Append a signal trace to a version trace."""
    vt = trace_version(version)
    st = trace_signal(signal_id)
    # Avoid duplicates
    existing_ids = {t.signal_id for t in vt.signal_traces}
    if st.signal_id not in existing_ids:
        vt.signal_traces.append(st)
        _get_default_store().save_version_trace(vt)
    return vt
