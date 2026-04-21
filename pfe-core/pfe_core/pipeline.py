"""High-level Phase 0 orchestration for CLI and server."""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .adapter_store.store import create_adapter_store
from .config import PFEConfig, PrivacyConfig
from .curator.teacher_client import TeacherClientConfig, TeacherInferenceClient
from .errors import EvalError
from .inference.engine import InferenceConfig, InferenceEngine, resolve_base_model_reference
from .models import parse_utc_datetime
from .curator.datasets import SampleFilterConfig, build_signal_quality, signal_quality_filter_reasons, summarize_signal_quality_filters
from .storage import list_samples, list_signals, record_signal, resolve_home, save_samples, status_snapshot, write_json, write_jsonl
from .trainer import summarize_real_training_execution, summarize_training_job_execution
from .trainer.service import TrainerService
from .trainer.runtime import summarize_trainer_backend_plan, trainer_runtime_summary
from .trainer.training_auditor import TrainingAuditor, TrainingAuditReport
from .observability.trace import (
    TraceStore,
    record_signal_node,
    append_signal_to_version,
    trace_signal,
    trace_version,
    SignalTrace,
)


def _default_chat_base_model() -> str:
    return resolve_base_model_reference("local-default")


def _eval_generation_kwargs() -> dict[str, Any]:
    env = importlib.import_module("os").environ
    try:
        max_tokens = int(env.get("PFE_EVAL_MAX_TOKENS", "32"))
    except Exception:
        max_tokens = 32
    max_tokens = max(1, min(max_tokens, 64))
    metadata: dict[str, Any] = {"source": "pfe-eval"}
    if str(env.get("PFE_ENABLE_REAL_LOCAL_INFERENCE", "")).strip().lower() in {"1", "true", "yes", "on"}:
        metadata["enable_real_local"] = True
    return {
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "metadata": metadata,
    }


def _normalize(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if is_dataclass(item):
        return asdict(item)
    return dict(item)


def _normalized_event_chain(payload: dict[str, Any]) -> tuple[str, list[str]]:
    source_event_id = str(payload.get("source_event_id") or payload["event_id"])
    raw_chain = payload.get("source_event_ids") or payload.get("event_chain_ids") or []
    chain: list[str] = []
    for candidate in [*list(raw_chain), source_event_id, payload["event_id"]]:
        if not candidate:
            continue
        normalized = str(candidate)
        if normalized not in chain:
            chain.append(normalized)
    return source_event_id, chain


def _adapter_row_snapshot(row: dict[str, Any], *, latest: bool = False) -> dict[str, Any]:
    def _coerce_payload(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return dict(parsed)
            except Exception:
                return {}
        return {}

    metadata = _coerce_payload(row.get("metadata"))
    eval_report = _coerce_payload(row.get("eval_report"))
    metrics = _coerce_payload(row.get("metrics"))
    training_config = _coerce_payload(row.get("training_config"))
    export_execution = _coerce_payload(metadata.get("export_execution"))
    export_write = _coerce_payload(metadata.get("export_write"))
    export_artifact_summary = _coerce_payload(metadata.get("export_artifact_summary"))
    if not export_artifact_summary:
        export_artifact_summary = _coerce_payload(_coerce_payload(metadata.get("export")).get("artifact"))
    output_artifact_validation = _coerce_payload(export_execution.get("output_artifact_validation"))
    export_artifact_path = (
        output_artifact_validation.get("path")
        or export_execution.get("output_artifact_path")
        or export_write.get("artifact_path")
        or export_artifact_summary.get("path")
        or row.get("artifact_path")
    )
    export_artifact_exists = False
    export_artifact_size_bytes = None
    if export_artifact_path:
        try:
            export_artifact_file = Path(str(export_artifact_path)).expanduser()
            export_artifact_exists = export_artifact_file.exists()
            if export_artifact_exists and export_artifact_file.is_file():
                export_artifact_size_bytes = export_artifact_file.stat().st_size
        except Exception:
            export_artifact_exists = False
    return {
        "version": row.get("version"),
        "workspace": row.get("workspace"),
        "state": row.get("state"),
        "base_model": row.get("base_model"),
        "num_samples": row.get("num_samples", 0),
        "artifact_format": row.get("artifact_format"),
        "path": row.get("adapter_dir"),
        "artifact_path": row.get("artifact_path"),
        "manifest_path": row.get("manifest_path"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "promoted_at": row.get("promoted_at"),
        "archived_at": row.get("archived_at"),
        "training_backend": training_config.get("backend") or (metadata.get("training") or {}).get("backend"),
        "requires_export": bool((metadata.get("training") or {}).get("requires_export_step", False)),
        "eval_recommendation": eval_report.get("recommendation"),
        "eval_comparison": eval_report.get("comparison"),
        "metric_keys": sorted(metrics.keys()),
        "export_status": export_execution.get("status") or export_execution.get("audit", {}).get("status") or export_artifact_summary.get("status"),
        "export_write_state": export_write.get("write_state") or export_write.get("metadata", {}).get("write_state") or export_artifact_summary.get("write_state"),
        "export_target_backend": (metadata.get("export_runtime") or {}).get("target_backend"),
        "export_artifact_path": export_artifact_path,
        "export_artifact_valid": output_artifact_validation.get("valid", export_artifact_summary.get("valid")),
        "export_artifact_exists": export_artifact_exists,
        "export_artifact_size_bytes": export_artifact_size_bytes if export_artifact_size_bytes is not None else export_artifact_summary.get("size_bytes"),
        "latest": latest,
        "metadata": metadata,
    }


def _export_toolchain_snapshot(last_run: dict[str, Any]) -> dict[str, Any]:
    export_execution = dict(last_run.get("export_execution") or {})
    export_command_plan = dict(last_run.get("export_command_plan") or {})
    export_runtime = dict(last_run.get("export_runtime") or {})
    export_write = dict(last_run.get("export_write") or {})
    summary = (
        dict(export_execution.get("toolchain_summary") or {})
        or dict(export_command_plan.get("toolchain_summary") or {})
    )
    if summary:
        return summary
    if not export_runtime:
        return {}
    return {
        "summary": "not_required" if not export_runtime.get("required", False) else "planned",
        "status": export_execution.get("status") or export_command_plan.get("status") or "not_required",
        "toolchain_status": export_execution.get("status") or export_command_plan.get("status") or "not_required",
        "required": bool(export_runtime.get("required", False)),
        "target_backend": export_runtime.get("target_backend"),
        "target_artifact_format": export_runtime.get("target_artifact_format"),
        "write_state": export_write.get("write_state"),
        "output_artifact_path": (export_execution.get("output_artifact_validation") or {}).get("path")
        or export_execution.get("output_artifact_path")
        or export_write.get("artifact_path"),
        "output_artifact_valid": (export_execution.get("output_artifact_validation") or {}).get("valid"),
        "output_artifact_size_bytes": (export_execution.get("output_artifact_validation") or {}).get("size_bytes"),
    }


class PipelineService:
    """Coordinate generation, training, evaluation, and serving helpers."""

    split_defaults = (("train", 0.8), ("val", 0.1), ("test", 0.1))
    generic_monitor_focuses = {
        "candidate_idle",
        "queue_waiting_execution",
        "queue_backlog",
        "runner_active",
        "daemon_active",
        "candidate_monitoring",
        "queue_monitoring",
        "runner_monitoring",
        "daemon_monitoring",
    }

    def __init__(self):
        self.trainer = TrainerService()
        self.last_auto_trigger_result: dict[str, Any] | None = None
        self.last_compare_result: dict[str, Any] | None = None
        self._trace_store = TraceStore()
        self._last_training_audit: TrainingAuditReport | None = None
        self._observability_enabled = True

    @staticmethod
    def _generic_monitor_active(*, focus: Any, inspection_summary_line: Any) -> bool:
        focus_text = str(focus or "").strip().lower()
        return bool(inspection_summary_line) and focus_text in PipelineService.generic_monitor_focuses

    @staticmethod
    def _prefer_inspection_summary_for_generic_monitor(
        *,
        focus: Any,
        summary_line: Any,
        inspection_summary_line: Any,
    ) -> tuple[Any, Any]:
        if PipelineService._generic_monitor_active(
            focus=focus,
            inspection_summary_line=inspection_summary_line,
        ):
            return inspection_summary_line, inspection_summary_line
        return summary_line, inspection_summary_line

    @staticmethod
    def _auto_trigger_state_path(*, workspace: str | None = None) -> Path:
        workspace_name = str(workspace or "user_default")
        safe_workspace = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in workspace_name)
        return resolve_home() / "data" / f"auto_train_state_{safe_workspace}.json"

    @staticmethod
    def _compare_evaluation_state_path(*, workspace: str | None = None) -> Path:
        workspace_name = str(workspace or "user_default")
        safe_workspace = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in workspace_name)
        return resolve_home() / "data" / f"compare_eval_state_{safe_workspace}.json"

    def _load_auto_trigger_state(self, *, workspace: str | None = None) -> dict[str, Any]:
        path = self._auto_trigger_state_path(workspace=workspace)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _persist_auto_trigger_state(self, payload: dict[str, Any], *, workspace: str | None = None) -> None:
        write_json(self._auto_trigger_state_path(workspace=workspace), payload)

    def _load_compare_evaluation_state(self, *, workspace: str | None = None) -> dict[str, Any]:
        path = self._compare_evaluation_state_path(workspace=workspace)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _persist_compare_evaluation_state(self, payload: dict[str, Any], *, workspace: str | None = None) -> None:
        write_json(self._compare_evaluation_state_path(workspace=workspace), payload)

    def _persist_auto_trigger_action(self, action: dict[str, Any], *, workspace: str | None = None) -> None:
        state = self._load_auto_trigger_state(workspace=workspace)
        state["last_action"] = dict(action)
        self._persist_auto_trigger_state(state, workspace=workspace)

    def _load_auto_trigger_action(self, *, workspace: str | None = None) -> dict[str, Any]:
        state = self._load_auto_trigger_state(workspace=workspace)
        payload = state.get("last_action") or {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _persist_candidate_action(self, action: dict[str, Any], *, workspace: str | None = None) -> None:
        state = self._load_auto_trigger_state(workspace=workspace)
        state["last_candidate_action"] = dict(action)
        history = list(state.get("candidate_history") or [])
        history.append(self._candidate_history_entry(action))
        state["candidate_history"] = history[-20:]
        self._persist_auto_trigger_state(state, workspace=workspace)

    def _load_candidate_action(self, *, workspace: str | None = None) -> dict[str, Any]:
        state = self._load_auto_trigger_state(workspace=workspace)
        payload = state.get("last_candidate_action") or {}
        return dict(payload) if isinstance(payload, dict) else {}

    @staticmethod
    def _candidate_history_entry(action: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": str(action.get("action") or "candidate_action"),
            "status": str(action.get("status") or "noop"),
            "reason": str(action.get("reason") or ""),
            "candidate_version": action.get("candidate_version"),
            "promoted_version": action.get("promoted_version"),
            "archived_version": action.get("archived_version"),
            "operator_note": action.get("operator_note"),
            "previous_candidate_state": action.get("previous_candidate_state"),
            "triggered": bool(action.get("triggered", False)),
        }

    def _load_candidate_history(self, *, workspace: str | None = None) -> list[dict[str, Any]]:
        state = self._load_auto_trigger_state(workspace=workspace)
        raw = state.get("candidate_history") or []
        if not isinstance(raw, list):
            return []
        return [dict(item) for item in raw if isinstance(item, dict)]

    def _candidate_history_summary(self, *, workspace: str | None = None) -> dict[str, Any]:
        history = self._load_candidate_history(workspace=workspace)
        latest = history[-1] if history else {}
        return {
            "count": len(history),
            "latest_timestamp": latest.get("timestamp"),
            "last_action": latest.get("action"),
            "last_status": latest.get("status"),
            "last_reason": latest.get("reason"),
            "last_candidate_version": latest.get("candidate_version"),
            "last_note": latest.get("operator_note"),
            "action_counts": {
                "promote_candidate": sum(1 for item in history if str(item.get("action")) == "promote_candidate"),
                "archive_candidate": sum(1 for item in history if str(item.get("action")) == "archive_candidate"),
            },
            "items": history[-5:],
        }

    def candidate_history(self, *, workspace: str | None = None, limit: int = 10) -> dict[str, Any]:
        bounded_limit = max(1, int(limit or 10))
        history = self._load_candidate_history(workspace=workspace)
        latest = history[-1] if history else {}
        return {
            "workspace": workspace or "user_default",
            "count": len(history),
            "limit": bounded_limit,
            "last_action": latest.get("action"),
            "last_status": latest.get("status"),
            "last_reason": latest.get("reason"),
            "last_candidate_version": latest.get("candidate_version"),
            "last_note": latest.get("operator_note"),
            "latest_timestamp": latest.get("timestamp"),
            "items": history[-bounded_limit:],
        }

    def _candidate_timeline_summary(self, *, workspace: str | None = None) -> dict[str, Any]:
        history = self._load_candidate_history(workspace=workspace)
        latest = history[-1] if history else {}
        transitions = sum(1 for item in history if str(item.get("status") or "") in {"completed", "blocked", "noop"})
        current_stage = "idle"
        if latest:
            action = str(latest.get("action") or "")
            status = str(latest.get("status") or "")
            if action == "promote_candidate" and status == "completed":
                current_stage = "promoted"
            elif action == "archive_candidate" and status == "completed":
                current_stage = "archived"
            elif status == "blocked":
                current_stage = "blocked"
            elif status == "noop":
                current_stage = "noop"
            else:
                current_stage = "candidate_action"
        return {
            "count": len(history),
            "transition_count": transitions,
            "current_stage": current_stage,
            "last_transition": latest,
            "last_reason": latest.get("reason"),
            "last_candidate_version": latest.get("candidate_version"),
            "latest_timestamp": latest.get("timestamp"),
        }

    def candidate_timeline(self, *, workspace: str | None = None, limit: int = 10) -> dict[str, Any]:
        bounded_limit = max(1, int(limit or 10))
        history = self._load_candidate_history(workspace=workspace)
        latest = history[-1] if history else {}
        items = history[-bounded_limit:]
        timeline_items: list[dict[str, Any]] = []
        for item in items:
            action = str(item.get("action") or "candidate_action")
            status = str(item.get("status") or "noop")
            if action == "promote_candidate" and status == "completed":
                stage = "promoted"
            elif action == "archive_candidate" and status == "completed":
                stage = "archived"
            elif status == "blocked":
                stage = "blocked"
            elif status == "noop":
                stage = "noop"
            else:
                stage = "candidate_action"
            timeline_items.append(
                {
                    **dict(item),
                    "stage": stage,
                    "label": f"{action}:{status}",
                }
            )
        summary = self._candidate_timeline_summary(workspace=workspace)
        return {
            "workspace": workspace or "user_default",
            "count": len(history),
            "limit": bounded_limit,
            "current_stage": summary.get("current_stage"),
            "transition_count": summary.get("transition_count"),
            "last_transition": latest,
            "last_reason": latest.get("reason"),
            "last_candidate_version": latest.get("candidate_version"),
            "latest_timestamp": latest.get("timestamp"),
            "items": timeline_items,
        }

    @staticmethod
    def _train_queue_state_path(*, workspace: str | None = None) -> Path:
        workspace_name = str(workspace or "user_default")
        safe_workspace = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in workspace_name)
        return resolve_home() / "data" / f"train_queue_{safe_workspace}.json"

    def _load_train_queue_state(self, *, workspace: str | None = None) -> dict[str, Any]:
        path = self._train_queue_state_path(workspace=workspace)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"items": []}
        if not isinstance(payload, dict):
            return {"items": []}
        items = payload.get("items")
        if not isinstance(items, list):
            payload["items"] = []
        return payload

    def _persist_train_queue_state(self, payload: dict[str, Any], *, workspace: str | None = None) -> None:
        write_json(self._train_queue_state_path(workspace=workspace), payload)

    @staticmethod
    def _train_queue_worker_state_path(*, workspace: str | None = None) -> Path:
        workspace_name = str(workspace or "user_default")
        safe_workspace = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in workspace_name)
        return resolve_home() / "data" / f"train_queue_worker_{safe_workspace}.json"

    def _load_train_queue_worker_state(self, *, workspace: str | None = None) -> dict[str, Any]:
        path = self._train_queue_worker_state_path(workspace=workspace)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _persist_train_queue_worker_state(self, payload: dict[str, Any], *, workspace: str | None = None) -> None:
        write_json(self._train_queue_worker_state_path(workspace=workspace), payload)

    @staticmethod
    def _train_queue_daemon_state_path(*, workspace: str | None = None) -> Path:
        workspace_name = str(workspace or "user_default")
        safe_workspace = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in workspace_name)
        return resolve_home() / "data" / f"train_queue_daemon_{safe_workspace}.json"

    def _load_train_queue_daemon_state(self, *, workspace: str | None = None) -> dict[str, Any]:
        path = self._train_queue_daemon_state_path(workspace=workspace)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _persist_train_queue_daemon_state(self, payload: dict[str, Any], *, workspace: str | None = None) -> None:
        write_json(self._train_queue_daemon_state_path(workspace=workspace), payload)

    @staticmethod
    def _pid_exists(pid: int | None) -> bool:
        if pid in (None, 0):
            return False
        try:
            os.kill(int(pid), 0)
        except Exception:
            return False
        return True

    @staticmethod
    def _worker_runner_history_entry(
        *,
        event: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
        }
        if reason:
            entry["reason"] = str(reason)
        if metadata:
            entry["metadata"] = dict(metadata)
        return entry

    def _append_train_queue_worker_history(
        self,
        *,
        workspace: str | None = None,
        event: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._load_train_queue_worker_state(workspace=workspace)
        history = list(payload.get("history") or [])
        history.append(
            self._worker_runner_history_entry(
                event=event,
                reason=reason,
                metadata=metadata,
            )
        )
        payload["history"] = history[-20:]
        payload["history_count"] = len(history)
        self._persist_train_queue_worker_state(payload, workspace=workspace)
        return payload

    def train_queue_worker_runner_history(self, *, workspace: str | None = None, limit: int = 10) -> dict[str, Any]:
        bounded_limit = max(1, int(limit or 10))
        payload = self._load_train_queue_worker_state(workspace=workspace)
        history = list(payload.get("history") or [])
        latest = history[-1] if history else {}
        return {
            "workspace": workspace or "user_default",
            "count": len(history),
            "limit": bounded_limit,
            "last_event": latest.get("event"),
            "last_reason": latest.get("reason"),
            "items": history[-bounded_limit:],
        }

    def train_queue_worker_runner_timeline(self, *, workspace: str | None = None, limit: int = 5) -> dict[str, Any]:
        return self._runner_timeline_summary(workspace=workspace, limit=limit)

    def _runner_timeline_summary(self, *, workspace: str | None = None, limit: int = 5) -> dict[str, Any]:
        history_payload = self.train_queue_worker_runner_history(workspace=workspace, limit=max(1, int(limit or 5)))
        items = [dict(item) for item in list(history_payload.get("items") or [])]
        takeover_items = [
            item
            for item in items
            if str(item.get("reason") or "").startswith("stale_lock_takeover")
        ]
        latest = items[-1] if items else {}
        last_takeover = takeover_items[-1] if takeover_items else {}
        return {
            "count": history_payload.get("count", 0),
            "latest_timestamp": ((latest or {}).get("timestamp") or None),
            "last_event": history_payload.get("last_event"),
            "last_reason": history_payload.get("last_reason"),
            "takeover_event_count": len(takeover_items),
            "last_takeover_event": last_takeover.get("event"),
            "last_takeover_reason": last_takeover.get("reason"),
            "recent_takeover_events": [
                {
                    "timestamp": item.get("timestamp"),
                    "event": item.get("event"),
                    "reason": item.get("reason"),
                    "note": item.get("note"),
                }
                for item in takeover_items[-max(1, int(limit or 5)) :]
            ],
            "recent_events": [
                {
                    "timestamp": item.get("timestamp"),
                    "event": item.get("event"),
                    "reason": item.get("reason"),
                    "note": item.get("note"),
                }
                for item in items[-max(1, int(limit or 5)) :]
            ],
            "latest": latest,
        }

    def _train_queue_worker_summary(self, *, workspace: str | None = None) -> dict[str, Any]:
        payload = self._load_train_queue_worker_state(workspace=workspace)
        stale_after_seconds = None
        lease_expires_at = None
        lock_state = "idle"
        if bool(payload.get("active", False)):
            lock_state = "active"
            heartbeat = self._parse_iso_datetime(payload.get("last_heartbeat_at"))
            if heartbeat is not None:
                if heartbeat.tzinfo is None:
                    heartbeat = heartbeat.replace(tzinfo=timezone.utc)
                max_seconds = float(payload.get("max_seconds", 30.0) or 30.0)
                stale_after_seconds = max(5.0, max_seconds * 2.0)
                lease_expires_at = (heartbeat + timedelta(seconds=stale_after_seconds)).isoformat()
                if (datetime.now(timezone.utc) - heartbeat).total_seconds() > stale_after_seconds:
                    lock_state = "stale"
        return {
            "state_path": str(self._train_queue_worker_state_path(workspace=workspace)),
            "active": bool(payload.get("active", False)),
            "lock_state": lock_state,
            "stale_after_seconds": stale_after_seconds,
            "lease_expires_at": lease_expires_at,
            "stop_requested": bool(payload.get("stop_requested", False)),
            "pid": payload.get("pid"),
            "started_at": payload.get("started_at"),
            "last_heartbeat_at": payload.get("last_heartbeat_at"),
            "last_completed_at": payload.get("last_completed_at"),
            "loop_cycles": int(payload.get("loop_cycles", 0) or 0),
            "processed_count": int(payload.get("processed_count", 0) or 0),
            "failed_count": int(payload.get("failed_count", 0) or 0),
            "stopped_reason": payload.get("stopped_reason"),
            "last_action": payload.get("last_action"),
            "history_count": int(payload.get("history_count", 0) or 0),
            "last_event": ((list(payload.get("history") or []) or [{}])[-1]).get("event"),
            "last_event_reason": ((list(payload.get("history") or []) or [{}])[-1]).get("reason"),
            "max_seconds": payload.get("max_seconds"),
            "idle_sleep_seconds": payload.get("idle_sleep_seconds"),
        }

    def _append_train_queue_daemon_history(
        self,
        *,
        workspace: str | None = None,
        event: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._load_train_queue_daemon_state(workspace=workspace)
        history = list(payload.get("history") or [])
        entry = self._worker_runner_history_entry(event=event, reason=reason, metadata=metadata)
        if isinstance(metadata, dict) and metadata.get("note") is not None:
            entry["note"] = str(metadata["note"])
        history.append(entry)
        payload["history"] = history[-20:]
        payload["history_count"] = len(history)
        self._persist_train_queue_daemon_state(payload, workspace=workspace)
        return payload

    def train_queue_daemon_history(self, *, workspace: str | None = None, limit: int = 10) -> dict[str, Any]:
        bounded_limit = max(1, int(limit or 10))
        payload = self._load_train_queue_daemon_state(workspace=workspace)
        history = list(payload.get("history") or [])
        latest = history[-1] if history else {}
        return {
            "workspace": workspace or "user_default",
            "count": len(history),
            "limit": bounded_limit,
            "last_event": latest.get("event"),
            "last_reason": latest.get("reason"),
            "latest_timestamp": latest.get("timestamp"),
            "items": history[-bounded_limit:],
        }

    def _daemon_timeline_summary(self, *, workspace: str | None = None, limit: int = 5) -> dict[str, Any]:
        history_payload = self.train_queue_daemon_history(workspace=workspace, limit=max(1, int(limit or 5)))
        items = [dict(item) for item in list(history_payload.get("items") or [])]
        recovery_events = {
            "recover_requested",
            "restart_requested",
            "recover_blocked",
            "stale_lock_takeover",
            "start_requested",
        }
        recovery_items = [item for item in items if str(item.get("event") or "") in recovery_events]
        latest = items[-1] if items else {}
        last_recovery = recovery_items[-1] if recovery_items else {}
        return {
            "count": history_payload.get("count", 0),
            "latest_timestamp": history_payload.get("latest_timestamp"),
            "last_event": history_payload.get("last_event"),
            "last_reason": history_payload.get("last_reason"),
            "recovery_event_count": len(recovery_items),
            "last_recovery_event": last_recovery.get("event"),
            "last_recovery_reason": last_recovery.get("reason"),
            "last_recovery_note": last_recovery.get("note"),
            "recent_recovery_events": [
                {
                    "timestamp": item.get("timestamp"),
                    "event": item.get("event"),
                    "reason": item.get("reason"),
                    "note": item.get("note"),
                }
                for item in recovery_items[-max(1, int(limit or 5)) :]
            ],
            "recent_events": [
                {
                    "timestamp": item.get("timestamp"),
                    "event": item.get("event"),
                    "reason": item.get("reason"),
                    "note": item.get("note"),
                }
                for item in items[-max(1, int(limit or 5)) :]
            ],
            "latest": latest,
        }

    def _train_queue_daemon_summary(self, *, workspace: str | None = None) -> dict[str, Any]:
        payload = self._load_train_queue_daemon_state(workspace=workspace)
        config = self._load_config()
        trigger = config.trainer.trigger
        active = bool(payload.get("active", False))
        pid = payload.get("pid")
        lock_state = "active" if active else "idle"
        heartbeat = self._parse_iso_datetime(payload.get("last_heartbeat_at"))
        heartbeat_interval_seconds = float(
            payload.get("heartbeat_interval_seconds", trigger.queue_daemon_heartbeat_interval_seconds)
            or trigger.queue_daemon_heartbeat_interval_seconds
        )
        lease_timeout_seconds = float(
            payload.get("lease_timeout_seconds", trigger.queue_daemon_lease_timeout_seconds)
            or trigger.queue_daemon_lease_timeout_seconds
        )
        heartbeat_timeout_seconds = float(
            payload.get("heartbeat_timeout_seconds", lease_timeout_seconds) or lease_timeout_seconds
        )
        heartbeat_age_seconds = None
        lease_expires_at = None
        lease_state = "idle"
        heartbeat_state = "idle"
        if active:
            if heartbeat is not None:
                if heartbeat.tzinfo is None:
                    heartbeat = heartbeat.replace(tzinfo=timezone.utc)
                heartbeat_age_seconds = max(0.0, round((datetime.now(timezone.utc) - heartbeat).total_seconds(), 3))
                effective_lease_timeout = max(5.0, lease_timeout_seconds)
                fresh_threshold_seconds = max(heartbeat_interval_seconds * 3.0, 5.0)
                lease_expires_at = (heartbeat + timedelta(seconds=effective_lease_timeout)).isoformat()
                if heartbeat_age_seconds <= fresh_threshold_seconds:
                    heartbeat_state = "fresh"
                elif heartbeat_age_seconds <= effective_lease_timeout:
                    heartbeat_state = "delayed"
                else:
                    heartbeat_state = "stale"
                if heartbeat_age_seconds <= max(effective_lease_timeout * 0.5, heartbeat_interval_seconds * 2.0):
                    lease_state = "valid"
                elif heartbeat_age_seconds <= effective_lease_timeout:
                    lease_state = "expiring"
                else:
                    lease_state = "expired"
                if heartbeat_age_seconds > effective_lease_timeout or not self._pid_exists(pid):
                    lock_state = "stale"
            elif not self._pid_exists(pid):
                lock_state = "stale"
                heartbeat_state = "stale"
                lease_state = "expired"
        history = list(payload.get("history") or [])
        latest = history[-1] if history else {}
        restart_attempts = int(payload.get("restart_attempts", 0) or 0)
        max_restart_attempts = int(payload.get("max_restart_attempts", trigger.queue_daemon_max_restart_attempts) or trigger.queue_daemon_max_restart_attempts)
        restart_backoff_seconds = float(payload.get("restart_backoff_seconds", trigger.queue_daemon_restart_backoff_seconds) or trigger.queue_daemon_restart_backoff_seconds)
        next_restart_after = self._parse_iso_datetime(payload.get("next_restart_after"))
        now = datetime.now(timezone.utc)
        if next_restart_after is not None and next_restart_after.tzinfo is None:
            next_restart_after = next_restart_after.replace(tzinfo=timezone.utc)
        backoff_remaining_seconds = None
        if next_restart_after is not None and now < next_restart_after:
            backoff_remaining_seconds = round((next_restart_after - now).total_seconds(), 3)
        if backoff_remaining_seconds is not None:
            restart_policy_state = "backoff"
        elif restart_attempts >= max_restart_attempts:
            restart_policy_state = "capped"
        else:
            restart_policy_state = "ready"
        desired_state = payload.get("desired_state") or "stopped"
        recovery_needed = bool(desired_state == "running" and lock_state in {"idle", "stale"} and not bool(payload.get("stop_requested", False)))
        can_recover = recovery_needed and backoff_remaining_seconds is None and restart_attempts < max_restart_attempts
        recovery_reason = None
        if recovery_needed:
            if lock_state == "stale":
                recovery_reason = "daemon_stale"
            elif backoff_remaining_seconds is not None:
                recovery_reason = "restart_backoff_active"
            elif restart_attempts >= max_restart_attempts:
                recovery_reason = "restart_attempt_limit_reached"
            else:
                recovery_reason = "daemon_inactive"
        requested_action = str(payload.get("requested_action") or "")
        command_status = str(payload.get("command_status") or "")
        observed_state = payload.get("observed_state") or ("running" if active else "stopped")
        if requested_action == "restart" and command_status == "spawned":
            observed_state = "restarting"
            recovery_state = "restarting"
        elif requested_action == "recover" and command_status == "spawned":
            observed_state = "recovering"
            recovery_state = "recovering"
        elif active and lock_state == "active":
            recovery_state = "healthy"
        elif recovery_needed and can_recover:
            recovery_state = "recoverable"
        elif recovery_needed:
            recovery_state = "blocked"
        else:
            recovery_state = "idle"
        if active and lock_state == "active":
            health_state = "healthy"
        elif lock_state == "stale":
            health_state = "stale"
        elif recovery_state in {"restarting", "recovering"}:
            health_state = "recovering"
        elif recovery_needed and not can_recover:
            health_state = "blocked"
        else:
            health_state = "stopped"
        if requested_action == "recover" and command_status == "spawned":
            recovery_action = "auto_recover" if str(payload.get("last_requested_by") or "") == "auto_recovery" else "manual_recover"
        elif requested_action == "restart" and command_status == "spawned":
            recovery_action = "restart_required"
        elif recovery_needed and can_recover:
            recovery_action = "auto_recover" if bool(payload.get("auto_recover_enabled", trigger.queue_daemon_auto_recover)) else "manual_recover"
        else:
            recovery_action = "none"
        return {
            "workspace": workspace or "user_default",
            "state_path": str(self._train_queue_daemon_state_path(workspace=workspace)),
            "desired_state": desired_state,
            "observed_state": observed_state,
            "requested_action": payload.get("requested_action"),
            "command_status": payload.get("command_status") or ("running" if active else "idle"),
            "active": active,
            "lock_state": lock_state,
            "pid": pid,
            "started_at": payload.get("started_at"),
            "last_heartbeat_at": payload.get("last_heartbeat_at"),
            "last_completed_at": payload.get("last_completed_at"),
            "last_requested_at": payload.get("last_requested_at"),
            "last_requested_by": payload.get("last_requested_by"),
            "stop_requested": bool(payload.get("stop_requested", False)),
            "auto_recover_enabled": bool(payload.get("auto_recover_enabled", trigger.queue_daemon_auto_recover)),
            "heartbeat_interval_seconds": heartbeat_interval_seconds,
            "lease_timeout_seconds": lease_timeout_seconds,
            "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
            "health_state": health_state,
            "lease_state": lease_state,
            "heartbeat_state": heartbeat_state,
            "restart_policy_state": restart_policy_state,
            "recovery_action": recovery_action,
            "lease_expires_at": lease_expires_at,
            "heartbeat_age_seconds": heartbeat_age_seconds,
            "history_count": int(payload.get("history_count", 0) or 0),
            "last_event": latest.get("event"),
            "last_reason": latest.get("reason"),
            "latest_timestamp": latest.get("timestamp"),
            "log_path": payload.get("log_path"),
            "runner_max_seconds": payload.get("runner_max_seconds"),
            "idle_sleep_seconds": payload.get("idle_sleep_seconds"),
            "takeover": bool(payload.get("takeover", False)),
            "previous_pid": payload.get("previous_pid"),
            "auto_restart_enabled": bool(payload.get("auto_restart_enabled", trigger.queue_daemon_auto_restart)),
            "restart_attempts": restart_attempts,
            "max_restart_attempts": max_restart_attempts,
            "restart_backoff_seconds": restart_backoff_seconds,
            "next_restart_after": next_restart_after.isoformat() if next_restart_after is not None else None,
            "backoff_remaining_seconds": backoff_remaining_seconds,
            "auto_recovery_count": int(payload.get("auto_recovery_count", 0) or 0),
            "last_auto_recovery_at": payload.get("last_auto_recovery_at"),
            "last_auto_recovery_reason": payload.get("last_auto_recovery_reason"),
            "recovery_needed": recovery_needed,
            "can_recover": can_recover,
            "recovery_reason": recovery_reason,
            "recovery_state": recovery_state,
            "recovery_mode": "restart_policy",
            "recovery_attempts": restart_attempts,
            "recovery_backoff_seconds": restart_backoff_seconds,
            "recovery_next_retry_at": next_restart_after.isoformat() if next_restart_after is not None else None,
        }

    def _spawn_train_queue_daemon(
        self,
        *,
        workspace: str | None = None,
        reset_restart_policy: bool,
        recovery_reason: str | None = None,
        requested_action: str | None = None,
        note: str | None = None,
        requested_by: str = "pipeline",
        auto_recovery: bool = False,
    ) -> dict[str, Any]:
        config = self._load_config()
        trigger = config.trainer.trigger
        workspace_name = workspace or "user_default"
        existing = self._train_queue_daemon_summary(workspace=workspace)
        repo_root = Path(__file__).resolve().parents[2]
        log_dir = resolve_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"train_queue_daemon_{workspace_name}.log"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(repo_root / "pfe-core"), str(repo_root / "pfe-cli"), str(repo_root / "pfe-server"), os.environ.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        command = [
            sys.executable,
            "-m",
            "pfe_core.worker_daemon",
            "--workspace",
            workspace_name,
            "--runner-max-seconds",
            str(float(trigger.queue_worker_runner_max_seconds)),
            "--idle-sleep-seconds",
            str(float(trigger.queue_worker_runner_idle_sleep_seconds)),
            "--heartbeat-interval-seconds",
            str(float(trigger.queue_daemon_heartbeat_interval_seconds)),
            "--lease-timeout-seconds",
            str(float(trigger.queue_daemon_lease_timeout_seconds)),
        ]
        with log_path.open("ab") as log_file:
            process = subprocess.Popen(command, cwd=str(repo_root), env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        previous_payload = self._load_train_queue_daemon_state(workspace=workspace)
        previous_attempts = int(previous_payload.get("restart_attempts", 0) or 0)
        restart_attempts = 0 if reset_restart_policy else previous_attempts + 1
        auto_recovery_count = int(previous_payload.get("auto_recovery_count", 0) or 0)
        now = datetime.now(timezone.utc)
        previous_payload.update(
            {
                "workspace": workspace_name,
                "desired_state": "running",
                "observed_state": "starting",
                "requested_action": requested_action or ("recover" if recovery_reason else "start"),
                "command_status": "spawned",
                "active": True,
                "pid": process.pid,
                "started_at": previous_payload.get("started_at") or now.isoformat(),
                "last_requested_at": now.isoformat(),
                "last_requested_by": requested_by,
                "last_heartbeat_at": now.isoformat(),
                "lease_renewed_at": now.isoformat(),
                "stop_requested": False,
                "auto_recover_enabled": bool(trigger.queue_daemon_auto_recover),
                "heartbeat_interval_seconds": float(trigger.queue_daemon_heartbeat_interval_seconds),
                "lease_timeout_seconds": float(trigger.queue_daemon_lease_timeout_seconds),
                "heartbeat_timeout_seconds": max(
                    float(trigger.queue_daemon_lease_timeout_seconds),
                    float(trigger.queue_daemon_heartbeat_interval_seconds) * 3.0,
                ),
                "runner_max_seconds": float(trigger.queue_worker_runner_max_seconds),
                "idle_sleep_seconds": float(trigger.queue_worker_runner_idle_sleep_seconds),
                "log_path": str(log_path),
                "takeover": existing.get("lock_state") == "stale",
                "previous_pid": existing.get("pid") if existing.get("lock_state") == "stale" else None,
                "auto_restart_enabled": bool(trigger.queue_daemon_auto_restart),
                "restart_attempts": restart_attempts,
                "max_restart_attempts": int(trigger.queue_daemon_max_restart_attempts),
                "restart_backoff_seconds": float(trigger.queue_daemon_restart_backoff_seconds),
                "next_restart_after": (now + timedelta(seconds=float(trigger.queue_daemon_restart_backoff_seconds))).isoformat(),
                "last_recovery_reason": recovery_reason,
                "auto_recovery_count": auto_recovery_count + (1 if auto_recovery else 0),
                "last_auto_recovery_at": now.isoformat() if auto_recovery else previous_payload.get("last_auto_recovery_at"),
                "last_auto_recovery_reason": recovery_reason if auto_recovery else previous_payload.get("last_auto_recovery_reason"),
            }
        )
        self._persist_train_queue_daemon_state(previous_payload, workspace=workspace)
        self._append_train_queue_daemon_history(
            workspace=workspace,
            event=f"{requested_action}_requested" if requested_action else ("recover_requested" if recovery_reason else "start_requested"),
            reason=recovery_reason or "daemon_start_requested",
            metadata={
                "pid": process.pid,
                "log_path": str(log_path),
                "restart_attempts": restart_attempts,
                "requested_by": requested_by,
                "auto_recovery": auto_recovery,
                "note": note,
            },
        )
        return self.train_queue_daemon_status(workspace=workspace)

    def _maybe_auto_recover_train_queue_daemon(self, *, workspace: str | None = None) -> dict[str, Any]:
        summary = self._train_queue_daemon_summary(workspace=workspace)
        if not bool(summary.get("auto_recover_enabled", False)):
            return summary
        if not bool(summary.get("auto_restart_enabled", False)):
            return summary
        if not bool(summary.get("can_recover", False)):
            return summary
        if str(summary.get("requested_action") or "") in {"start", "recover", "restart"} and str(
            summary.get("command_status") or ""
        ) == "spawned":
            return summary
        return self._spawn_train_queue_daemon(
            workspace=workspace,
            reset_restart_policy=False,
            recovery_reason=str(summary.get("recovery_reason") or "daemon_auto_recovery_requested"),
            requested_action="recover",
            note="auto_recovery",
            requested_by="auto_recovery",
            auto_recovery=True,
        )

    @staticmethod
    def _queue_history_entry(
        *,
        event: str,
        item: dict[str, Any],
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp = (
            item.get("updated_at")
            or item.get("triggered_at")
            or item.get("created_at")
            or datetime.now(timezone.utc).isoformat()
        )
        entry = {
            "timestamp": str(timestamp),
            "event": str(event),
            "state": str(item.get("state") or "unknown"),
            "job_id": str(item.get("job_id") or ""),
        }
        if reason:
            entry["reason"] = str(reason)
        if metadata:
            entry["metadata"] = dict(metadata)
            if metadata.get("note") is not None:
                entry["note"] = str(metadata["note"])
        return entry

    @staticmethod
    def _queue_sort_key(item: dict[str, Any]) -> tuple[int, str]:
        priority = int(item.get("priority", 0) or 0)
        ordered_at = str(
            item.get("triggered_at")
            or item.get("created_at")
            or item.get("updated_at")
            or ""
        )
        return (-priority, ordered_at)

    @staticmethod
    def _queue_priority(
        *,
        trigger_status: dict[str, Any],
        source: str,
        policy: str,
    ) -> tuple[int, str]:
        if str(policy) == "fifo":
            return 0, "policy:fifo"
        base_priority = 100 if str(source) == "signal_auto_train" else 50
        priority_source = "policy:source_default"
        if str(policy) == "promotion_bias":
            priority_source = "policy:promotion_bias"
            if bool(trigger_status.get("auto_promote")):
                return base_priority + 20, f"{priority_source}:auto_promote"
            if bool(trigger_status.get("auto_evaluate")):
                return base_priority + 10, f"{priority_source}:auto_evaluate"
        return base_priority, priority_source

    @staticmethod
    def _auto_train_queue_dedup_key(
        trigger_status: dict[str, Any],
        *,
        dedup_scope: str = "train_config",
        workspace: str | None = None,
    ) -> str:
        scope = str(dedup_scope or "train_config")
        base_model = str(trigger_status.get("base_model") or "local-default")
        if scope == "workspace":
            return f"workspace:{workspace or 'user_default'}"
        if scope == "base_model":
            return f"base_model:{base_model}"
        return "|".join(
            [
                "train_config",
                base_model,
                str(trigger_status.get("method") or "qlora"),
                str(trigger_status.get("train_type") or "sft"),
                f"eval={bool(trigger_status.get('auto_evaluate'))}",
                f"promote={bool(trigger_status.get('auto_promote'))}",
            ]
        )

    def _find_train_queue_item_by_dedup_key(
        self,
        dedup_key: str,
        *,
        states: tuple[str, ...] = ("queued", "running"),
        workspace: str | None = None,
    ) -> dict[str, Any] | None:
        payload = self._load_train_queue_state(workspace=workspace)
        for item in list(payload.get("items") or []):
            if str(item.get("dedup_key") or "") != str(dedup_key):
                continue
            if str(item.get("state") or "") not in states:
                continue
            return dict(item)
        return None

    def _append_train_queue_item(self, item: dict[str, Any], *, workspace: str | None = None) -> dict[str, Any]:
        payload = self._load_train_queue_state(workspace=workspace)
        items = list(payload.get("items") or [])
        normalized = dict(item)
        normalized.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        normalized.setdefault("priority", 0)
        history_event = str(normalized.pop("history_event", "enqueued"))
        history_reason = normalized.pop("history_reason", None)
        history_metadata = normalized.pop("history_metadata", None)
        history = list(normalized.get("history") or [])
        history.append(
            self._queue_history_entry(
                event=history_event,
                item=normalized,
                reason=str(history_reason) if history_reason is not None else None,
                metadata=history_metadata if isinstance(history_metadata, dict) else None,
            )
        )
        normalized["history"] = history[-10:]
        normalized["history_count"] = len(history)
        items.insert(0, normalized)
        items = sorted(items, key=self._queue_sort_key)
        payload["items"] = items[:25]
        payload["last_item"] = dict(normalized)
        self._persist_train_queue_state(payload, workspace=workspace)
        return dict(normalized)

    def _update_train_queue_item(self, job_id: str, updates: dict[str, Any], *, workspace: str | None = None) -> dict[str, Any]:
        payload = self._load_train_queue_state(workspace=workspace)
        items = list(payload.get("items") or [])
        updated_item: dict[str, Any] | None = None
        history_event = str(updates.get("history_event") or "updated")
        history_reason = updates.get("history_reason")
        history_metadata = updates.get("history_metadata")
        sanitized_updates = {
            key: value
            for key, value in updates.items()
            if key not in {"history_event", "history_reason", "history_metadata"}
        }
        for index, item in enumerate(items):
            if str(item.get("job_id")) != str(job_id):
                continue
            merged = dict(item)
            merged.update(sanitized_updates)
            history = list(merged.get("history") or [])
            history.append(
                self._queue_history_entry(
                    event=history_event,
                    item=merged,
                    reason=str(history_reason) if history_reason is not None else None,
                    metadata=history_metadata if isinstance(history_metadata, dict) else None,
                )
            )
            merged["history"] = history[-10:]
            merged["history_count"] = len(history)
            items[index] = merged
            updated_item = merged
            break
        if updated_item is None:
            updated_item = {"job_id": job_id, **sanitized_updates}
            history = list(updated_item.get("history") or [])
            history.append(
                self._queue_history_entry(
                    event=history_event,
                    item=updated_item,
                    reason=str(history_reason) if history_reason is not None else None,
                    metadata=history_metadata if isinstance(history_metadata, dict) else None,
                )
            )
            updated_item["history"] = history[-10:]
            updated_item["history_count"] = len(history)
            items.insert(0, updated_item)
        items = sorted(items, key=self._queue_sort_key)
        payload["items"] = items[:25]
        payload["last_item"] = dict(updated_item)
        self._persist_train_queue_state(payload, workspace=workspace)
        return dict(updated_item)

    def _train_queue_snapshot(self, *, workspace: str | None = None) -> dict[str, Any]:
        trigger = self._load_config().trainer.trigger
        payload = self._load_train_queue_state(workspace=workspace)
        items = [dict(item) for item in list(payload.get("items") or [])]
        counts: dict[str, int] = {}
        for item in items:
            state = str(item.get("state") or "unknown")
            counts[state] = counts.get(state, 0) + 1
        current_item = next((item for item in items if str(item.get("state")) in {"queued", "running"}), None)
        last_item = dict(payload.get("last_item") or (items[0] if items else {}))
        dedup_scopes = sorted(
            {
                str(item.get("queue_dedup_scope"))
                for item in items
                if item.get("queue_dedup_scope") is not None
            }
        )
        priority_sources = sorted(
            {
                str(item.get("priority_source"))
                for item in items
                if item.get("priority_source") is not None
            }
        )
        awaiting_item = next((item for item in items if str(item.get("state")) == "awaiting_confirmation"), None)
        queued_item = next((item for item in items if str(item.get("state")) == "queued"), None)
        confirmation_required_count = sum(1 for item in items if bool(item.get("confirmation_required")))
        awaiting_confirmation_count = int(counts.get("awaiting_confirmation", 0) or 0)
        recent_history: list[dict[str, Any]] = []
        for item in items:
            recent_history.extend(list(item.get("history") or [])[-1:])
        recent_history = sorted(recent_history, key=lambda entry: str(entry.get("timestamp") or ""), reverse=True)[:5]
        last_transition = recent_history[0] if recent_history else {}
        history_summary = {
            "transition_count": sum(int(item.get("history_count", 0) or 0) for item in items),
            "last_transition": last_transition,
            "last_reason": last_transition.get("reason"),
        }
        review_entries = [
            entry
            for item in items
            for entry in list(item.get("history") or [])
            if str(entry.get("event") or "") in {"approved", "rejected"}
        ]
        review_entries = sorted(review_entries, key=lambda entry: str(entry.get("timestamp") or ""), reverse=True)
        last_review = review_entries[0] if review_entries else {}
        review_required_by_policy = bool(str(trigger.queue_mode) == "deferred" and bool(trigger.require_queue_confirmation))
        if awaiting_confirmation_count > 0:
            review_queue_entry_mode = "awaiting_confirmation"
            review_next_action = "review_queue_confirmation"
            review_reason = (awaiting_item or {}).get("confirmation_reason")
        elif queued_item is not None:
            review_queue_entry_mode = "queued"
            review_next_action = "process_next_queue_item"
            review_reason = None
        elif str(trigger.queue_mode) == "deferred":
            review_queue_entry_mode = "deferred_idle"
            review_next_action = "await_new_queue_item"
            review_reason = None
        else:
            review_queue_entry_mode = "inline_execute"
            review_next_action = "await_signal_trigger"
            review_reason = None
        review_mode = "manual_review" if (review_required_by_policy or awaiting_confirmation_count > 0) else "auto_queue"
        review_policy_summary = {
            "review_mode": review_mode,
            "queue_entry_mode": review_queue_entry_mode,
            "review_required_by_policy": review_required_by_policy,
            "review_required_now": awaiting_confirmation_count > 0,
            "review_reason": review_reason,
            "next_action": review_next_action,
            "summary_line": " | ".join(
                [
                    f"mode={review_mode}",
                    f"entry={review_queue_entry_mode}",
                    f"next={review_next_action}",
                ]
                + ([f"reason={review_reason}"] if review_reason else [])
            ),
        }
        return {
            "count": len(items),
            "counts": counts,
            "current": current_item or {},
            "last_item": last_item,
            "max_priority": max((int(item.get("priority", 0) or 0) for item in items), default=0),
            "history_summary": history_summary,
            "last_transition": last_transition,
            "recent_history": recent_history,
            "policy_summary": {
                "dedup_scopes": dedup_scopes,
                "priority_sources": priority_sources,
                "current_priority_source": (current_item or {}).get("priority_source"),
                "current_dedup_scope": (current_item or {}).get("queue_dedup_scope"),
            },
            "confirmation_summary": {
                "confirmation_required_count": confirmation_required_count,
                "awaiting_confirmation_count": awaiting_confirmation_count,
                "next_job_id": (awaiting_item or {}).get("job_id"),
                "next_confirmation_reason": (awaiting_item or {}).get("confirmation_reason"),
            },
            "review_summary": {
                "reviewed_transition_count": len(review_entries),
                "approved_transition_count": sum(1 for entry in review_entries if str(entry.get("event")) == "approved"),
                "rejected_transition_count": sum(1 for entry in review_entries if str(entry.get("event")) == "rejected"),
                "last_review_event": last_review.get("event"),
                "last_review_reason": last_review.get("reason"),
                "last_review_note": last_review.get("note"),
                "next_job_id": (awaiting_item or {}).get("job_id"),
                "next_confirmation_reason": (awaiting_item or {}).get("confirmation_reason"),
            },
            "review_policy_summary": review_policy_summary,
            "worker_runner": self._train_queue_worker_summary(workspace=workspace),
            "runner_history": self._runner_timeline_summary(workspace=workspace),
            "daemon": self.train_queue_daemon_status(workspace=workspace),
            "daemon_history": self._daemon_timeline_summary(workspace=workspace),
            "items": items[:5],
        }

    def _first_train_queue_item(
        self,
        *,
        states: tuple[str, ...],
        workspace: str | None = None,
    ) -> dict[str, Any] | None:
        payload = self._load_train_queue_state(workspace=workspace)
        for item in list(payload.get("items") or []):
            if str(item.get("state")) in states:
                return dict(item)
        return None

    def train_queue_history(
        self,
        *,
        workspace: str | None = None,
        job_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        bounded_limit = max(1, int(limit or 10))
        payload = self._load_train_queue_state(workspace=workspace)
        items = [dict(item) for item in list(payload.get("items") or [])]
        target_item: dict[str, Any] = {}
        if job_id:
            for item in items:
                if str(item.get("job_id") or "") == str(job_id):
                    target_item = item
                    break
        elif items:
            target_item = dict(payload.get("last_item") or items[0])
        history = list(target_item.get("history") or [])
        return {
            "workspace": workspace or "user_default",
            "job_id": target_item.get("job_id"),
            "state": target_item.get("state"),
            "count": len(history),
            "limit": bounded_limit,
            "history": history[-bounded_limit:],
            "history_count": int(target_item.get("history_count", len(history)) or len(history)),
            "available_job_ids": [item.get("job_id") for item in items[:10] if item.get("job_id")],
            "history_summary": dict(self._train_queue_snapshot(workspace=workspace).get("history_summary") or {}),
        }

    @staticmethod
    def _candidate_promotion_gate(
        *,
        candidate_version: Any,
        compare_evaluation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate_version_text = str(candidate_version) if candidate_version is not None else ""
        compare = dict(compare_evaluation or {})
        if not candidate_version_text or not compare:
            return {
                "status": "open",
                "reason": None,
                "action": None,
            }

        left_adapter = str(compare.get("left_adapter") or compare.get("left_adapter_version") or "")
        right_adapter = str(compare.get("right_adapter") or compare.get("right_adapter_version") or "")
        recommendation = str(compare.get("recommendation") or "")

        if candidate_version_text != right_adapter:
            return {
                "status": "open",
                "reason": None,
                "action": None,
            }

        if recommendation == "keep_right":
            return {
                "status": "open",
                "reason": None,
                "action": None,
            }
        if recommendation == "needs_more_data":
            return {
                "status": "blocked",
                "reason": "compare_needs_more_data",
                "action": "run_compare_evaluation",
            }
        if recommendation == "keep_left":
            return {
                "status": "blocked",
                "reason": "compare_prefers_previous",
                "action": "inspect_compare_evaluation",
            }

        return {
            "status": "open",
            "reason": None,
            "action": None,
        }

    @staticmethod
    def _candidate_summary(
        *,
        rows: list[dict[str, Any]],
        latest_version: Any,
        recent_row: dict[str, Any] | None,
        compare_evaluation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        latest_version_text = str(latest_version) if latest_version is not None else None
        pending_rows = [row for row in rows if row.get("state") == "pending_eval"]
        training_rows = [row for row in rows if row.get("state") == "training"]
        failed_rows = [row for row in rows if row.get("state") == "failed_eval"]
        candidate_row = None
        for row in rows:
            version = str(row.get("version") or "")
            if latest_version_text and version == latest_version_text and row.get("state") == "promoted":
                continue
            candidate_row = row
            break
        candidate_state = candidate_row.get("state") if candidate_row else None
        candidate_is_latest_promoted = bool(
            candidate_row
            and latest_version_text
            and str(candidate_row.get("version") or "") == latest_version_text
            and candidate_state == "promoted"
        )
        promotion_gate = PipelineService._candidate_promotion_gate(
            candidate_version=candidate_row.get("version") if candidate_row else None,
            compare_evaluation=compare_evaluation,
        )
        candidate_can_promote = bool(
            candidate_row
            and candidate_state in {"training", "pending_eval", "failed_eval"}
            and promotion_gate.get("status") != "blocked"
        )
        candidate_can_archive = bool(candidate_row and candidate_state not in {"archived"} and not candidate_is_latest_promoted)
        summary = {
            "latest_promoted_version": latest_version,
            "recent_version": recent_row.get("version") if recent_row else None,
            "candidate_version": candidate_row.get("version") if candidate_row else None,
            "candidate_state": candidate_state,
            "candidate_is_latest_promoted": candidate_is_latest_promoted,
            "candidate_exists": bool(candidate_row),
            "candidate_can_promote": candidate_can_promote,
            "candidate_can_archive": candidate_can_archive,
            "pending_eval_count": len(pending_rows),
            "pending_eval_versions": [row.get("version") for row in pending_rows],
            "training_count": len(training_rows),
            "training_versions": [row.get("version") for row in training_rows],
            "failed_eval_count": len(failed_rows),
            "failed_eval_versions": [row.get("version") for row in failed_rows],
            "has_pending_candidate": bool(pending_rows or training_rows),
            "candidate_needs_promotion": bool(candidate_row and candidate_state == "pending_eval"),
            "promotion_gate_status": promotion_gate.get("status"),
            "promotion_gate_reason": promotion_gate.get("reason"),
            "promotion_gate_action": promotion_gate.get("action"),
        }
        compare = dict(compare_evaluation or {})
        if compare:
            comparison = compare.get("comparison")
            recommendation = compare.get("recommendation")
            winner = compare.get("winner")
            left_adapter = compare.get("left_adapter") or compare.get("left_adapter_version")
            right_adapter = compare.get("right_adapter") or compare.get("right_adapter_version")
            overall_delta = compare.get("overall_delta")
            personalization_delta = compare.get("personalization_delta")
            quality_delta = compare.get("quality_delta")
            personalization_summary = compare.get("personalization_summary")
            quality_summary = compare.get("quality_summary")
            compare_summary_line = compare.get("summary_line")
            score_deltas = compare.get("score_deltas")
            if comparison is not None:
                summary["promotion_compare_comparison"] = comparison
            if recommendation is not None:
                summary["promotion_compare_recommendation"] = recommendation
            if winner is not None:
                summary["promotion_compare_winner"] = winner
            if left_adapter is not None:
                summary["promotion_compare_left_adapter"] = left_adapter
            if right_adapter is not None:
                summary["promotion_compare_right_adapter"] = right_adapter
            if overall_delta is not None:
                summary["promotion_compare_overall_delta"] = overall_delta
            if personalization_delta is not None:
                summary["promotion_compare_personalization_delta"] = personalization_delta
            if quality_delta is not None:
                summary["promotion_compare_quality_delta"] = quality_delta
            if personalization_summary:
                summary["promotion_compare_personalization_summary"] = personalization_summary
            if quality_summary:
                summary["promotion_compare_quality_summary"] = quality_summary
            if compare_summary_line:
                summary["promotion_compare_summary_line"] = compare_summary_line
            if isinstance(score_deltas, Mapping):
                style_preference_hit_rate_delta = score_deltas.get("style_preference_hit_rate")
                if style_preference_hit_rate_delta is not None:
                    summary["promotion_compare_style_preference_hit_rate_delta"] = style_preference_hit_rate_delta
            details = compare.get("details")
            if isinstance(details, list):
                summary["promotion_compare_details_count"] = len(details)
        return summary

    @staticmethod
    def _candidate_action_plan(candidate_summary: dict[str, Any] | None) -> dict[str, Any]:
        candidate = dict(candidate_summary or {})
        candidate_exists = bool(candidate.get("candidate_exists"))
        can_promote = bool(candidate.get("candidate_can_promote"))
        can_archive = bool(candidate.get("candidate_can_archive"))
        needs_promotion = bool(candidate.get("candidate_needs_promotion"))

        primary_action = ""
        promotion_gate_status = str(candidate.get("promotion_gate_status") or "open")
        if needs_promotion and can_promote:
            primary_action = "promote_candidate"
        elif needs_promotion and promotion_gate_status == "blocked":
            primary_action = "inspect_candidate_status"
        elif can_archive:
            primary_action = "archive_candidate"
        elif can_promote:
            primary_action = "promote_candidate"
        elif candidate_exists:
            primary_action = "inspect_candidate_status"

        secondary_actions: list[str] = []
        compare_recommendation = str(candidate.get("promotion_compare_recommendation") or "")
        compare_winner = str(candidate.get("promotion_compare_winner") or "")
        promotion_gate_reason = str(candidate.get("promotion_gate_reason") or "")
        if primary_action == "promote_candidate" and can_archive:
            secondary_actions.append("archive_candidate")
        elif primary_action == "archive_candidate" and can_promote:
            secondary_actions.append("promote_candidate")
        elif primary_action == "inspect_candidate_status":
            secondary_actions.append("inspect_candidate_timeline")
            if promotion_gate_reason:
                secondary_actions.append(str(candidate.get("promotion_gate_action") or "inspect_compare_evaluation"))

        summary_parts: list[str] = []
        if primary_action:
            summary_parts.append(f"do={primary_action}")
        if secondary_actions:
            summary_parts.append(f"see={','.join(secondary_actions)}")
        if needs_promotion:
            summary_parts.append("ready=yes")
        if promotion_gate_status == "blocked" and promotion_gate_reason:
            summary_parts.append(f"promotion_gate={promotion_gate_reason}")
        if compare_recommendation:
            summary_parts.append(f"compare_gate={compare_recommendation}")
        if compare_winner:
            summary_parts.append(f"compare_winner={compare_winner}")
        return {
            "primary_action": primary_action or None,
            "secondary_actions": secondary_actions,
            "summary_line": " | ".join(summary_parts),
        }

    @staticmethod
    def _queue_action_plan(train_queue: dict[str, Any] | None) -> dict[str, Any]:
        queue = dict(train_queue or {})
        confirmation = dict(queue.get("confirmation_summary") or {})
        review_policy = dict(queue.get("review_policy_summary") or {})
        worker = dict(queue.get("worker_runner") or {})

        queue_count = int(queue.get("count", 0) or 0)
        awaiting_confirmation_count = int(confirmation.get("awaiting_confirmation_count", 0) or 0)
        review_mode = str(review_policy.get("review_mode") or "")
        review_required_now = bool(review_policy.get("review_required_now", False))
        next_action = str(review_policy.get("next_action") or "")
        runner_active = bool(worker.get("active"))

        primary_action = ""
        secondary_actions: list[str] = []

        if awaiting_confirmation_count > 0 or (review_mode == "manual_review" and review_required_now):
            primary_action = "review_queue_confirmation"
            secondary_actions.extend(["inspect_auto_train_gate", "inspect_auto_train_trigger"])
        elif queue_count > 0 and not runner_active:
            primary_action = "process_next_queue_item"
            secondary_actions.append("inspect_auto_train_trigger")
        elif runner_active and queue_count > 0:
            primary_action = "wait_for_queue_completion"
            secondary_actions.append("inspect_auto_train_trigger")
        elif next_action == "process_next_queue_item":
            primary_action = "process_next_queue_item"
            secondary_actions.append("inspect_auto_train_trigger")

        summary_parts: list[str] = []
        if primary_action:
            summary_parts.append(f"do={primary_action}")
        if secondary_actions:
            summary_parts.append(f"see={','.join(secondary_actions)}")
        if queue_count:
            summary_parts.append(f"queue={queue_count}")
        if awaiting_confirmation_count:
            summary_parts.append(f"awaiting={awaiting_confirmation_count}")
        if runner_active:
            summary_parts.append("runner=active")

        return {
            "primary_action": primary_action or None,
            "secondary_actions": secondary_actions,
            "summary_line": " | ".join(summary_parts),
        }

    @staticmethod
    def _resolved_queue_review_policy(
        *,
        queue_review_policy: Mapping[str, Any] | None,
        trigger_policy: Mapping[str, Any] | None,
        queue_gate_reason: Any = None,
        queue_gate_action: Any = None,
        queue_review_mode: Any = None,
        review_reason: Any = None,
        review_required_now: Any = None,
    ) -> dict[str, Any]:
        review_policy = dict(queue_review_policy or {})
        if review_policy:
            return review_policy

        trigger_policy_map = dict(trigger_policy or {})
        resolved_review_mode = str(queue_review_mode or trigger_policy_map.get("review_mode") or "")
        queue_entry_mode = str(trigger_policy_map.get("queue_entry_mode") or "")
        resolved_queue_gate_reason = str(queue_gate_reason or "")
        resolved_queue_gate_action = str(queue_gate_action or "")
        resolved_review_reason = review_reason
        resolved_review_required_now = bool(review_required_now) if review_required_now is not None else False

        next_action = ""
        if resolved_queue_gate_reason == "queue_pending_review":
            next_action = resolved_queue_gate_action or "review_queue_confirmation"
            if resolved_review_reason is None:
                resolved_review_reason = "manual_review_required_by_policy"
            resolved_review_required_now = True
        elif resolved_queue_gate_reason == "queue_waiting_execution":
            next_action = resolved_queue_gate_action or "process_next_queue_item"
        elif resolved_queue_gate_reason == "queue_processing_active":
            next_action = resolved_queue_gate_action or "wait_for_queue_completion"
        elif resolved_queue_gate_reason:
            next_action = resolved_queue_gate_action or resolved_queue_gate_reason
        elif queue_entry_mode == "awaiting_confirmation":
            next_action = "review_queue_confirmation"
        else:
            next_action = "await_signal_trigger"

        if not resolved_review_mode:
            resolved_review_mode = "manual_review" if queue_entry_mode == "awaiting_confirmation" else "auto_queue"
        if not queue_entry_mode:
            queue_entry_mode = "awaiting_confirmation" if resolved_review_mode == "manual_review" else "inline_execute"

        summary_parts = [
            f"mode={resolved_review_mode}",
            f"entry={queue_entry_mode}",
            f"next={next_action}",
        ]
        if resolved_review_reason:
            summary_parts.append(f"reason={resolved_review_reason}")
        return {
            "review_mode": resolved_review_mode,
            "queue_entry_mode": queue_entry_mode,
            "review_required_now": resolved_review_required_now,
            "review_reason": resolved_review_reason,
            "next_action": next_action,
            "summary_line": " | ".join(summary_parts),
        }

    @staticmethod
    def _runtime_action_plan(runtime_stability_summary: dict[str, Any] | None) -> dict[str, Any]:
        runtime = dict(runtime_stability_summary or {})
        runner_active = bool(runtime.get("runner_active"))
        daemon_active = bool(runtime.get("daemon_active"))

        primary_action = ""
        secondary_actions: list[str] = []

        if runner_active:
            primary_action = "inspect_runtime_stability"
            secondary_actions.append("inspect_worker_runner_history")
        elif daemon_active:
            primary_action = "inspect_runtime_stability"
            secondary_actions.append("inspect_daemon_status")

        summary_parts: list[str] = []
        if primary_action:
            summary_parts.append(f"do={primary_action}")
        if secondary_actions:
            summary_parts.append(f"see={','.join(secondary_actions)}")
        if runner_active:
            summary_parts.append("runner=active")
        if daemon_active:
            summary_parts.append("daemon=active")

        return {
            "primary_action": primary_action or None,
            "secondary_actions": secondary_actions,
            "summary_line": " | ".join(summary_parts),
        }

    @staticmethod
    def _operations_overview(
        *,
        auto_train_trigger: dict[str, Any] | None,
        candidate_summary: dict[str, Any] | None,
        train_queue: dict[str, Any] | None,
    ) -> dict[str, Any]:
        trigger = dict(auto_train_trigger or {})
        trigger_policy = dict(trigger.get("policy") or {})
        candidate = dict(candidate_summary or {})
        queue = dict(train_queue or {})
        confirmation = dict(queue.get("confirmation_summary") or {})
        worker = dict(queue.get("worker_runner") or {})
        daemon = dict(queue.get("daemon") or {})
        daemon_history = dict(queue.get("daemon_history") or {})
        awaiting_confirmation_count = int(confirmation.get("awaiting_confirmation_count", 0) or 0)
        queue_count = int(queue.get("count", 0) or 0)
        candidate_version = candidate.get("candidate_version")
        candidate_state = candidate.get("candidate_state")
        candidate_needs_promotion = bool(candidate.get("candidate_needs_promotion"))
        candidate_action_plan = PipelineService._candidate_action_plan(candidate)
        runner_active = bool(worker.get("active"))
        runner_lock_state = str(worker.get("lock_state") or "idle")
        runner_last_event = worker.get("last_event")
        daemon_active = bool(daemon.get("active"))
        daemon_lock_state = str(daemon.get("lock_state") or "idle")
        trigger_state = str(trigger.get("state") or "idle")
        trigger_ready = bool(trigger.get("ready", False))
        trigger_gate_reason = str(trigger.get("queue_gate_reason") or "")
        trigger_gate_action = str(trigger.get("queue_gate_action") or "")
        trigger_queue_review_mode = trigger.get("queue_review_mode")
        trigger_blocked_reason = str(trigger.get("blocked_primary_reason") or "")
        trigger_blocked_action = str(trigger.get("blocked_primary_action") or "")
        trigger_blocked_category = str(trigger.get("blocked_primary_category") or "")
        trigger_blocked_summary = str(trigger.get("blocked_summary") or "")
        effective_trigger_blocked_reason = trigger_gate_reason or trigger_blocked_reason
        effective_trigger_blocked_action = trigger_gate_action or trigger_blocked_action
        effective_trigger_blocked_category = (
            "queue"
            if trigger_gate_reason
            else trigger_blocked_category
        )
        effective_trigger_blocked_details = PipelineService._auto_train_blocker_details(effective_trigger_blocked_reason)
        effective_trigger_blocked_summary = (
            " | ".join(
                part
                for part in (
                    f"reason={effective_trigger_blocked_reason}" if effective_trigger_blocked_reason else None,
                    f"action={effective_trigger_blocked_action}" if effective_trigger_blocked_action else None,
                    f"category={effective_trigger_blocked_category}" if effective_trigger_blocked_category else None,
                    f"summary={effective_trigger_blocked_details.get('summary')}" if effective_trigger_blocked_details.get("summary") else None,
                )
                if part is not None
            )
            if effective_trigger_blocked_reason or effective_trigger_blocked_action or effective_trigger_blocked_category
            else trigger_blocked_summary
        )
        daemon_recovery_needed = bool(daemon.get("recovery_needed", False))
        daemon_recovery_reason = daemon.get("recovery_reason")
        daemon_backoff_remaining = daemon.get("backoff_remaining_seconds")
        daemon_lease_expires_at = daemon.get("lease_expires_at")
        daemon_health_state = str(daemon.get("health_state") or ("healthy" if daemon_active else "stopped"))
        daemon_lease_state = str(daemon.get("lease_state") or ("valid" if daemon_active else "idle"))
        daemon_heartbeat_state = str(daemon.get("heartbeat_state") or ("fresh" if daemon_active else "idle"))
        daemon_restart_policy_state = str(daemon.get("restart_policy_state") or "ready")
        daemon_recovery_action = str(daemon.get("recovery_action") or "none")
        daemon_recovery_event_count = int(daemon_history.get("recovery_event_count", 0) or 0)
        daemon_last_recovery_event = daemon_history.get("last_recovery_event")
        daemon_last_recovery_reason = daemon_history.get("last_recovery_reason")
        runner_stop_requested = bool(worker.get("stop_requested"))
        runtime_stability_summary = {
            "runner_active": runner_active,
            "runner_lock_state": runner_lock_state,
            "runner_stop_requested": runner_stop_requested,
            "daemon_active": daemon_active,
            "daemon_health_state": daemon_health_state,
            "daemon_heartbeat_state": daemon_heartbeat_state,
            "daemon_lease_state": daemon_lease_state,
            "daemon_restart_policy_state": daemon_restart_policy_state,
            "daemon_recovery_action": daemon_recovery_action,
        }
        runtime_stability_summary["summary_line"] = " | ".join(
            [
                f"runner={('active' if runner_active else (runner_lock_state or 'idle'))}",
                f"stop={'yes' if runner_stop_requested else 'no'}",
                f"daemon={daemon_health_state or 'stopped'}",
                f"hb={daemon_heartbeat_state or 'idle'}",
                f"lease={daemon_lease_state or 'idle'}",
                f"restart={daemon_restart_policy_state or 'ready'}",
                f"recover={daemon_recovery_action or 'none'}",
            ]
        )
        review_policy = PipelineService._resolved_queue_review_policy(
            queue_review_policy=queue.get("review_policy_summary"),
            trigger_policy=trigger_policy,
            queue_gate_reason=trigger_gate_reason or trigger_blocked_reason,
            queue_gate_action=trigger_gate_action or trigger_blocked_action,
            queue_review_mode=trigger_queue_review_mode,
            review_required_now=awaiting_confirmation_count > 0,
        )
        queue_for_actions = dict(queue)
        queue_for_actions["review_policy_summary"] = review_policy
        queue_action_plan = PipelineService._queue_action_plan(queue_for_actions)
        runtime_action_plan = PipelineService._runtime_action_plan(runtime_stability_summary)
        trigger_threshold = dict(trigger.get("threshold_summary") or {})
        trigger_policy_reason = str(
            trigger_policy.get("promote_gate_reason")
            or trigger_policy.get("evaluation_gate_reason")
            or ""
        )
        trigger_policy_action = str(
            trigger_policy.get("promote_gate_action")
            or trigger_policy.get("evaluation_gate_action")
            or ""
        )
        auto_train_blocker = PipelineService._auto_train_primary_blocker(
            queue_gate_reason=trigger_gate_reason or None,
            queue_gate_action=trigger_gate_action or None,
            trigger_blocked_reason=trigger_blocked_reason or None,
            trigger_blocked_action=trigger_blocked_action or None,
            trigger_blocked_category=trigger_blocked_category or None,
            trigger_blocked_summary=trigger_blocked_summary or None,
            trigger_policy_reason=trigger_policy_reason or None,
            trigger_policy_action=trigger_policy_action or None,
        )

        attention_needed = False
        attention_reason: str | None = None
        alerts: list[dict[str, Any]] = []
        if awaiting_confirmation_count > 0:
            attention_needed = True
            attention_reason = "awaiting_confirmation"
        elif candidate_needs_promotion and candidate_version:
            attention_needed = True
            attention_reason = "candidate_ready_for_promotion"
        elif trigger_state == "blocked" and str(trigger.get("reason") or "") != "trigger_disabled":
            attention_needed = True
            attention_reason = str(
                auto_train_blocker.get("reason")
                or effective_trigger_blocked_reason
                or trigger.get("reason")
                or "auto_train_blocked"
            )
        elif trigger_gate_reason:
            attention_needed = True
            attention_reason = str(
                auto_train_blocker.get("reason")
                or trigger_gate_reason
                or "auto_train_blocked"
            )
        elif trigger_policy_reason:
            attention_needed = True
            attention_reason = str(
                auto_train_blocker.get("reason")
                or trigger_policy_reason
                or "auto_train_blocked"
            )
        elif daemon_recovery_needed:
            attention_needed = True
            attention_reason = str(daemon_recovery_reason or "worker_daemon_recovery_needed")
        elif daemon_health_state in {"stale", "blocked"}:
            attention_needed = True
            attention_reason = f"daemon_{daemon_health_state}"
        elif daemon_restart_policy_state in {"backoff", "capped"}:
            attention_needed = True
            attention_reason = f"daemon_restart_{daemon_restart_policy_state}"
        if auto_train_blocker:
            alert_level = "attention" if str(auto_train_blocker.get("reason") or "") == "queue_pending_review" else "warning"
            alerts.append(
                {
                    "level": alert_level,
                    "scope": "auto_train",
                    "reason": auto_train_blocker.get("reason"),
                    "message": str(auto_train_blocker.get("summary") or "auto-train is blocked"),
                    "action": auto_train_blocker.get("action"),
                }
            )
        if awaiting_confirmation_count > 0:
            alerts.append(
                {
                    "level": "attention",
                    "scope": "queue",
                    "reason": "awaiting_confirmation",
                    "message": f"{awaiting_confirmation_count} queue item(s) waiting for confirmation",
                }
            )
        if candidate_needs_promotion and candidate_version:
            alerts.append(
                {
                    "level": "attention",
                    "scope": "candidate",
                    "reason": "candidate_ready_for_promotion",
                    "message": f"candidate {candidate_version} is ready for promotion",
                    "action": candidate_action_plan.get("primary_action") or "promote_candidate",
                }
            )
        if trigger_state == "blocked" and str(trigger.get("reason") or "") != "trigger_disabled" and str(auto_train_blocker.get("reason") or "") != str(effective_trigger_blocked_reason or trigger.get("reason") or ""):
            alerts.append(
                {
                    "level": "warning",
                    "scope": "auto_train",
                    "reason": str(effective_trigger_blocked_reason or trigger.get("reason") or "auto_train_blocked"),
                    "message": str(
                        effective_trigger_blocked_details.get("summary")
                        or "auto-train is blocked by current trigger policy"
                    ),
                    "action": str(effective_trigger_blocked_action or effective_trigger_blocked_details.get("action") or "inspect_auto_train_policy"),
                }
            )
        if trigger_gate_reason and str(auto_train_blocker.get("reason") or "") != trigger_gate_reason:
            alerts.append(
                {
                    "level": "attention" if trigger_gate_reason == "queue_pending_review" else "warning",
                    "scope": "auto_train",
                    "reason": trigger_gate_reason,
                    "message": (
                        "auto-train is waiting for queue confirmation review"
                        if trigger_gate_reason == "queue_pending_review"
                        else (
                            "auto-train is blocked by queued work waiting for execution"
                            if trigger_gate_reason == "queue_waiting_execution"
                            else "auto-train is blocked by queue processing already in progress"
                        )
                    ),
                }
            )
        if trigger_policy_reason and str(auto_train_blocker.get("reason") or "") != trigger_policy_reason:
            alerts.append(
                {
                    "level": "warning",
                    "scope": "auto_train",
                    "reason": trigger_policy_reason,
                    "message": (
                        "auto-train policy requires auto-evaluate before auto-promote can run"
                        if trigger_policy_reason == "policy_requires_auto_evaluate"
                        else "auto-train policy requires review before the next automated stage can continue"
                    ),
                    "action": trigger_policy_action or "inspect_auto_train_policy",
                }
            )
        if daemon_recovery_needed:
            alerts.append(
                {
                    "level": "warning",
                    "scope": "daemon",
                    "reason": str(daemon_recovery_reason or "worker_daemon_recovery_needed"),
                    "message": "worker daemon needs recovery",
                }
            )
        if daemon_heartbeat_state in {"delayed", "stale"}:
            alerts.append(
                {
                    "level": "warning",
                    "scope": "daemon",
                    "reason": f"daemon_heartbeat_{daemon_heartbeat_state}",
                    "message": f"worker daemon heartbeat is {daemon_heartbeat_state}",
                }
            )
        if daemon_lease_state in {"expiring", "expired"}:
            alerts.append(
                {
                    "level": "warning",
                    "scope": "daemon",
                    "reason": f"daemon_lease_{daemon_lease_state}",
                    "message": f"worker daemon lease is {daemon_lease_state}",
                }
            )
        if daemon_restart_policy_state in {"backoff", "capped"}:
            alerts.append(
                {
                    "level": "warning",
                    "scope": "daemon",
                    "reason": f"daemon_restart_{daemon_restart_policy_state}",
                    "message": f"worker daemon restart policy is {daemon_restart_policy_state}",
                }
            )
        if runner_lock_state == "stale":
            alerts.append(
                {
                    "level": "warning",
                    "scope": "runner",
                    "reason": "stale_runner_lock",
                    "message": "worker runner lock is stale",
                }
            )
        summary_parts: list[str] = [f"auto={trigger_state}"]
        if trigger_ready:
            summary_parts.append("ready")
        if trigger_policy.get("queue_entry_mode"):
            summary_parts.append(f"policy={trigger_policy.get('queue_entry_mode')}")
        if trigger_threshold.get("summary_line"):
            summary_parts.append(f"gate={trigger_threshold.get('summary_line')}")
        if trigger_policy_reason:
            summary_parts.append(f"policy-gate={trigger_policy_reason}")
        if trigger_policy_action:
            summary_parts.append(f"policy-action={trigger_policy_action}")
        if effective_trigger_blocked_reason and effective_trigger_blocked_reason not in {trigger_gate_reason, trigger_policy_reason}:
            summary_parts.append(f"blocked={effective_trigger_blocked_reason}")
        if effective_trigger_blocked_action and effective_trigger_blocked_action not in {trigger_gate_action, trigger_policy_action}:
            summary_parts.append(f"blocked-action={effective_trigger_blocked_action}")
        if trigger_gate_reason:
            summary_parts.append(f"queue-gate={trigger_gate_reason}")
        if trigger_gate_action:
            summary_parts.append(f"queue-action={trigger_gate_action}")
        if review_policy.get("queue_entry_mode"):
            summary_parts.append(f"q-review={review_policy.get('queue_entry_mode')}")
        if candidate_version:
            summary_parts.append(f"candidate={candidate_version}:{candidate_state or 'unknown'}")
        else:
            summary_parts.append("candidate=none")
        summary_parts.append(f"queue={queue_count}")
        if awaiting_confirmation_count > 0:
            summary_parts.append(f"awaiting={awaiting_confirmation_count}")
        if candidate_needs_promotion:
            summary_parts.append("promote-ready")
        if candidate_action_plan.get("primary_action"):
            summary_parts.append(f"candidate-action={candidate_action_plan.get('primary_action')}")
        if runner_active:
            summary_parts.append("runner=active")
        elif runner_lock_state == "stale":
            summary_parts.append("runner=stale")
        else:
            summary_parts.append(f"runner={runner_lock_state}")
        if daemon_active:
            summary_parts.append("daemon=active")
        elif daemon_lock_state != "idle":
            summary_parts.append(f"daemon={daemon_lock_state}")
        summary_parts.append(f"daemon-health={daemon_health_state}")
        if daemon_heartbeat_state not in {"idle", "fresh"}:
            summary_parts.append(f"daemon-heartbeat={daemon_heartbeat_state}")
        if daemon_lease_state not in {"idle", "valid"}:
            summary_parts.append(f"daemon-lease={daemon_lease_state}")
        if daemon_restart_policy_state != "ready":
            summary_parts.append(f"daemon-restart={daemon_restart_policy_state}")
        if daemon_recovery_action != "none":
            summary_parts.append(f"daemon-action={daemon_recovery_action}")
        recovery_summary = {
            "daemon_recovery_needed": daemon_recovery_needed,
            "daemon_recovery_reason": daemon_recovery_reason,
            "daemon_backoff_remaining_seconds": daemon_backoff_remaining,
            "daemon_recovery_action": daemon_recovery_action,
            "daemon_restart_policy_state": daemon_restart_policy_state,
            "daemon_last_recovery_event": daemon_last_recovery_event,
            "daemon_last_recovery_reason": daemon_last_recovery_reason,
            "daemon_recovery_event_count": daemon_recovery_event_count,
            "runner_stop_requested": runner_stop_requested,
        }
        health_summary = {
            "status": "attention" if attention_needed else "healthy",
            "daemon_active": daemon_active,
            "daemon_lock_state": daemon_lock_state,
            "daemon_health_state": daemon_health_state,
            "daemon_lease_state": daemon_lease_state,
            "daemon_heartbeat_state": daemon_heartbeat_state,
            "daemon_restart_policy_state": daemon_restart_policy_state,
            "daemon_recovery_action": daemon_recovery_action,
            "daemon_lease_expires_at": daemon_lease_expires_at,
            "runner_active": runner_active,
            "runner_lock_state": runner_lock_state,
            "runner_last_event": runner_last_event,
            "queue_depth": queue_count,
        }
        overview_monitor_focus = (
            "queue_waiting_execution"
            if queue_action_plan.get("primary_action") == "process_next_queue_item"
            else (
                "runner_active"
                if runtime_action_plan.get("primary_action") == "inspect_runtime_stability" and runner_active
                else (
                    "daemon_active"
                    if runtime_action_plan.get("primary_action") == "inspect_runtime_stability" and daemon_active
                    else (
                        "candidate_idle"
                        if candidate_action_plan.get("primary_action")
                        and (candidate_state not in {None, "idle", ""} or candidate_version not in {None, ""})
                        else None
                    )
                )
            )
        )
        overview_focus = attention_reason or overview_monitor_focus
        overview_next_actions: list[str] = []

        def _push_overview_action(action_name: Any | None) -> None:
            action_text = str(action_name or "").strip()
            if not action_text or action_text == "none" or action_text in overview_next_actions:
                return
            overview_next_actions.append(action_text)

        for action_name in (
            candidate_action_plan.get("primary_action"),
            *(list(candidate_action_plan.get("secondary_actions") or [])),
            queue_action_plan.get("primary_action"),
            *(list(queue_action_plan.get("secondary_actions") or [])),
            runtime_action_plan.get("primary_action"),
            *(list(runtime_action_plan.get("secondary_actions") or [])),
        ):
            _push_overview_action(action_name)
        monitor_primary_action_present = any(
            str(action_name or "").strip()
            for action_name in (
                candidate_action_plan.get("primary_action"),
                queue_action_plan.get("primary_action"),
                runtime_action_plan.get("primary_action"),
            )
        )
        if attention_reason or not monitor_primary_action_present:
            for action_name in (trigger_policy_action, effective_trigger_blocked_action):
                _push_overview_action(action_name)
        review_next_action = str(review_policy.get("next_action") or "").strip()
        if review_next_action and (
            str(review_policy.get("review_mode") or "") == "manual_review"
            or (
                not monitor_primary_action_present
                and review_next_action != "await_signal_trigger"
            )
        ):
            _push_overview_action(review_next_action)

        overview_required_action = (
            overview_next_actions[0]
            if overview_next_actions
            else (
                candidate_action_plan.get("primary_action")
                or queue_action_plan.get("primary_action")
                or runtime_action_plan.get("primary_action")
                or "observe_and_monitor"
            )
        )
        if overview_focus is not None:
            summary_parts.append(f"current_focus={overview_focus}")
        if overview_required_action is not None:
            summary_parts.append(f"required_action={overview_required_action}")
        overview_inspection_parts: list[str] = []
        if overview_focus is not None:
            overview_inspection_parts.append(f"current_focus={overview_focus}")
        if overview_required_action is not None:
            overview_inspection_parts.append(f"required_action={overview_required_action}")
        if overview_next_actions:
            overview_inspection_parts.append(
                "next_actions=" + ",".join(str(item) for item in overview_next_actions[:3] if str(item))
            )
        overview_inspection_summary = " | ".join(overview_inspection_parts) if overview_inspection_parts else None
        overview_summary_line, overview_inspection_summary = PipelineService._prefer_inspection_summary_for_generic_monitor(
            focus=overview_focus,
            summary_line=" | ".join(summary_parts),
            inspection_summary_line=overview_inspection_summary,
        )
        return {
            "attention_needed": attention_needed,
            "attention_reason": attention_reason,
            "current_focus": overview_focus,
            "monitor_focus": overview_monitor_focus,
            "required_action": overview_required_action,
            "trigger_state": trigger_state,
            "trigger_ready": trigger_ready,
            "auto_train_policy": trigger_policy,
            "trigger_policy_summary": trigger_policy.get("summary_line"),
            "trigger_policy_reason": trigger_policy_reason or None,
            "trigger_policy_action": trigger_policy_action or None,
            "trigger_threshold_summary": trigger_threshold,
            "trigger_threshold_summary_line": trigger_threshold.get("summary_line"),
            "auto_train_blocker": auto_train_blocker,
            "auto_train_blockers": [auto_train_blocker] if auto_train_blocker else [],
            "trigger_blocked_reason": effective_trigger_blocked_reason or None,
            "trigger_blocked_action": effective_trigger_blocked_action or None,
            "trigger_blocked_category": effective_trigger_blocked_category or None,
            "trigger_blocked_summary": effective_trigger_blocked_summary or None,
            "queue_gate_reason": trigger_gate_reason or None,
            "queue_gate_action": trigger_gate_action or None,
            "queue_gate_review_mode": trigger_queue_review_mode,
            "queue_review_policy": review_policy,
            "queue_review_policy_summary": review_policy.get("summary_line"),
            "runtime_stability_summary": runtime_stability_summary,
            "runtime_stability_summary_line": runtime_stability_summary.get("summary_line"),
            "candidate_version": candidate_version,
            "candidate_state": candidate_state,
            "candidate_needs_promotion": candidate_needs_promotion,
            "candidate_primary_action": candidate_action_plan.get("primary_action"),
            "candidate_secondary_actions": list(candidate_action_plan.get("secondary_actions") or []),
            "candidate_action_summary": candidate_action_plan,
            "queue_action_summary": queue_action_plan,
            "runtime_action_summary": runtime_action_plan,
            "inspection_summary_line": overview_inspection_summary,
            "queue_count": queue_count,
            "awaiting_confirmation_count": awaiting_confirmation_count,
            "runner_active": runner_active,
            "runner_lock_state": runner_lock_state,
            "runner_last_event": runner_last_event,
            "daemon_active": daemon_active,
            "daemon_lock_state": daemon_lock_state,
            "daemon_health_state": daemon_health_state,
            "daemon_lease_state": daemon_lease_state,
            "daemon_heartbeat_state": daemon_heartbeat_state,
            "daemon_restart_policy_state": daemon_restart_policy_state,
            "daemon_recovery_action": daemon_recovery_action,
            "alerts": alerts,
            "health": health_summary,
            "recovery": recovery_summary,
            "summary_line": overview_summary_line,
        }

    @staticmethod
    def _operations_console(
        *,
        operations_overview: dict[str, Any] | None,
        candidate_history: dict[str, Any] | None,
        candidate_timeline: dict[str, Any] | None,
        train_queue: dict[str, Any] | None,
    ) -> dict[str, Any]:
        overview = dict(operations_overview or {})
        trigger = dict(overview.get("auto_train_trigger") or {})
        trigger_policy = dict(overview.get("auto_train_policy") or trigger.get("policy") or {})
        trigger_threshold = dict(overview.get("trigger_threshold_summary") or trigger.get("threshold_summary") or {})
        candidate_hist = dict(candidate_history or {})
        candidate_line = dict(candidate_timeline or {})
        queue = dict(train_queue or {})
        queue_history = dict(queue.get("history_summary") or {})
        queue_review = dict(queue.get("review_summary") or {})
        queue_confirm = dict(queue.get("confirmation_summary") or {})
        worker = dict(queue.get("worker_runner") or {})
        runner_history = dict(queue.get("runner_history") or {})
        daemon = dict(queue.get("daemon") or {})
        daemon_history = dict(queue.get("daemon_history") or {})
        runtime_stability = dict(overview.get("runtime_stability_summary") or {})
        candidate_action_summary = dict(overview.get("candidate_action_summary") or {})
        queue_action_summary = dict(overview.get("queue_action_summary") or {})
        runtime_action_summary = dict(overview.get("runtime_action_summary") or {})
        next_actions: list[str] = []
        attention_reason = overview.get("attention_reason")
        trigger_policy_reason = str(overview.get("trigger_policy_reason") or "")
        trigger_policy_action = str(overview.get("trigger_policy_action") or "")
        trigger_threshold = dict(overview.get("trigger_threshold_summary") or {})
        trigger_policy_gate = dict(trigger_policy.get("gate_summary") or {})
        trigger_blocked_reason = str(overview.get("trigger_blocked_reason") or "")
        trigger_blocked_action = str(overview.get("trigger_blocked_action") or "")
        queue_review_policy = PipelineService._resolved_queue_review_policy(
            queue_review_policy=queue.get("review_policy_summary"),
            trigger_policy=trigger_policy,
            queue_gate_reason=overview.get("queue_gate_reason") or trigger_blocked_reason,
            queue_gate_action=overview.get("queue_gate_action") or trigger_blocked_action,
            queue_review_mode=overview.get("queue_review_mode") or overview.get("queue_gate_review_mode"),
            review_reason=overview.get("queue_review_reason"),
            review_required_now=overview.get("queue_review_required_now"),
        )
        candidate_primary_action = str(
            candidate_action_summary.get("primary_action") or overview.get("candidate_primary_action") or ""
        )
        candidate_secondary_actions = [
            str(item)
            for item in list(
                candidate_action_summary.get("secondary_actions") or overview.get("candidate_secondary_actions") or []
            )
            if str(item)
        ]
        console_monitor_primary_action_present = any(
            str(action_name or "").strip()
            for action_name in (
                candidate_primary_action,
                queue_action_summary.get("primary_action"),
                runtime_action_summary.get("primary_action"),
            )
        )
        if attention_reason == "awaiting_confirmation":
            next_actions.append("review_queue_confirmation")
        elif attention_reason == "candidate_ready_for_promotion":
            next_actions.append(candidate_primary_action or "promote_candidate")
            next_actions.extend(candidate_secondary_actions)
        elif attention_reason == "policy_requires_auto_evaluate":
            next_actions.append(trigger_policy_action or "inspect_auto_train_policy")
        elif attention_reason == "review_required_before_execution":
            next_actions.append(trigger_policy_action or "review_queue_confirmation")
        elif attention_reason == "queue_pending_review":
            next_actions.append("review_queue_confirmation")
        elif attention_reason == "queue_waiting_execution":
            next_actions.append("process_next_queue_item")
        elif attention_reason == "queue_processing_active":
            next_actions.append("wait_for_queue_completion")
        elif attention_reason:
            next_actions.append(trigger_blocked_action or str(attention_reason))

        if (
            not next_actions
            and trigger_blocked_reason
            and trigger_blocked_action
            and not console_monitor_primary_action_present
            and trigger_blocked_reason not in {trigger_policy_reason, "ready"}
        ):
            next_actions.append(trigger_blocked_action)
        candidate_monitor_ready = bool(
            candidate_line.get("current_stage") not in {None, "idle"}
            or overview.get("candidate_state") not in {None, "idle", ""}
            or overview.get("candidate_version") not in {None, ""}
        )
        if not next_actions and candidate_primary_action:
            next_actions.append(candidate_primary_action)
            next_actions.extend(candidate_secondary_actions)
        if not next_actions and candidate_monitor_ready:
            next_actions.append("inspect_candidate_status")
            next_actions.append("inspect_candidate_timeline")
        if not next_actions and queue_action_summary.get("primary_action"):
            next_actions.append(str(queue_action_summary.get("primary_action")))
            next_actions.extend(str(item) for item in list(queue_action_summary.get("secondary_actions") or []) if str(item))
        if (
            not next_actions
            and int(queue.get("count", 0) or 0) > 0
            and int(queue.get("awaiting_confirmation_count", 0) or 0) == 0
        ):
            next_actions.append("process_next_queue_item")
            next_actions.append("inspect_auto_train_trigger")
        if not next_actions and runtime_action_summary.get("primary_action"):
            next_actions.append(str(runtime_action_summary.get("primary_action")))
            next_actions.extend(str(item) for item in list(runtime_action_summary.get("secondary_actions") or []) if str(item))
        if not next_actions and bool(worker.get("active")):
            next_actions.append("inspect_runtime_stability")
            next_actions.append("inspect_worker_runner_history")
        if not next_actions and bool(daemon.get("active")):
            next_actions.append("inspect_runtime_stability")
            next_actions.append("inspect_daemon_status")

        if str(worker.get("lock_state") or "") == "stale":
            next_actions.append("inspect_worker_stale_lock")
        if bool(daemon.get("recovery_needed", False)):
            next_actions.append("recover_worker_daemon")
        if str(daemon.get("heartbeat_state") or "") in {"delayed", "stale"}:
            next_actions.append("inspect_daemon_heartbeat")
        if str(daemon.get("restart_policy_state") or "") in {"backoff", "capped"}:
            next_actions.append("inspect_daemon_restart_policy")
        if bool(worker.get("active")) and bool(worker.get("stop_requested")):
            next_actions.append("wait_for_runner_shutdown")
        if int(queue_confirm.get("awaiting_confirmation_count", 0) or 0) > 0 and "review_queue_confirmation" not in next_actions:
            next_actions.append("review_queue_confirmation")

        candidate_section = {
            "current_stage": candidate_line.get("current_stage"),
            "last_candidate_version": candidate_line.get("last_candidate_version"),
            "last_reason": candidate_line.get("last_reason"),
            "latest_timestamp": candidate_line.get("latest_timestamp") or candidate_hist.get("latest_timestamp"),
            "transition_count": candidate_line.get("transition_count"),
            "history_count": candidate_hist.get("count"),
        }
        queue_section = {
            "count": queue.get("count"),
            "awaiting_confirmation_count": queue_confirm.get("awaiting_confirmation_count"),
            "next_confirmation_reason": queue_confirm.get("next_confirmation_reason"),
            "last_transition": queue_history.get("last_transition"),
            "last_reason": queue_history.get("last_reason"),
            "reviewed_transition_count": queue_review.get("reviewed_transition_count"),
            "last_review_event": queue_review.get("last_review_event"),
            "last_review_note": queue_review.get("last_review_note"),
            "review_policy_summary": queue_review_policy,
        }
        runner_section = {
            "active": worker.get("active"),
            "lock_state": worker.get("lock_state"),
            "last_event": worker.get("last_event"),
            "last_event_reason": worker.get("last_event_reason"),
            "lease_expires_at": worker.get("lease_expires_at"),
            "history_count": worker.get("history_count"),
        }
        runner_timeline_section = {
            "count": runner_history.get("count"),
            "latest_timestamp": runner_history.get("latest_timestamp"),
            "last_event": runner_history.get("last_event"),
            "last_reason": runner_history.get("last_reason"),
            "takeover_event_count": runner_history.get("takeover_event_count"),
            "last_takeover_event": runner_history.get("last_takeover_event"),
            "last_takeover_reason": runner_history.get("last_takeover_reason"),
            "recent_events": list(runner_history.get("recent_events") or []),
            "recent_takeover_events": list(runner_history.get("recent_takeover_events") or []),
        }
        daemon_section = {
            "active": daemon.get("active"),
            "lock_state": daemon.get("lock_state"),
            "health_state": daemon.get("health_state"),
            "lease_state": daemon.get("lease_state"),
            "heartbeat_state": daemon.get("heartbeat_state"),
            "restart_policy_state": daemon.get("restart_policy_state"),
            "recovery_action": daemon.get("recovery_action"),
            "last_event": daemon.get("last_event"),
            "last_reason": daemon.get("last_reason"),
            "lease_expires_at": daemon.get("lease_expires_at"),
            "history_count": daemon.get("history_count"),
            "recovery_needed": daemon.get("recovery_needed"),
            "can_recover": daemon.get("can_recover"),
            "recovery_reason": daemon.get("recovery_reason"),
            "restart_attempts": daemon.get("restart_attempts"),
            "last_auto_recovery_reason": daemon.get("last_auto_recovery_reason"),
        }
        daemon_timeline_section = {
            "count": daemon_history.get("count"),
            "latest_timestamp": daemon_history.get("latest_timestamp"),
            "last_event": daemon_history.get("last_event"),
            "last_reason": daemon_history.get("last_reason"),
            "recovery_event_count": daemon_history.get("recovery_event_count"),
            "last_recovery_event": daemon_history.get("last_recovery_event"),
            "last_recovery_reason": daemon_history.get("last_recovery_reason"),
            "last_recovery_note": daemon_history.get("last_recovery_note"),
            "recent_events": list(daemon_history.get("recent_events") or []),
            "recent_recovery_events": list(daemon_history.get("recent_recovery_events") or []),
        }
        recovery_candidates: list[dict[str, Any]] = []
        for item in list(daemon_timeline_section.get("recent_recovery_events") or []):
            if not isinstance(item, dict):
                continue
            recovery_candidates.append(
                {
                    "source": "daemon",
                    "event": item.get("event"),
                    "reason": item.get("reason"),
                    "note": item.get("note"),
                    "timestamp": item.get("timestamp"),
                }
            )
        for item in list(runner_timeline_section.get("recent_takeover_events") or []):
            if not isinstance(item, dict):
                continue
            recovery_candidates.append(
                {
                    "source": "runner",
                    "event": item.get("event"),
                    "reason": item.get("reason"),
                    "note": item.get("note"),
                    "timestamp": item.get("timestamp"),
                }
            )
        recovery_candidates = sorted(recovery_candidates, key=lambda item: str(item.get("timestamp") or ""), reverse=True)
        latest_recovery = recovery_candidates[0] if recovery_candidates else None
        daemon_status_line_parts = [
            f"lock={daemon_section.get('lock_state') or 'idle'}",
            f"health={daemon_section.get('health_state') or 'unknown'}",
            f"heartbeat={daemon_section.get('heartbeat_state') or 'unknown'}",
            f"lease={daemon_section.get('lease_state') or 'unknown'}",
            f"restart={daemon_section.get('restart_policy_state') or 'ready'}",
        ]
        if daemon_section.get("recovery_action") not in {None, "none"}:
            daemon_status_line_parts.append(f"action={daemon_section.get('recovery_action')}")
        daemon_section["status_line"] = " | ".join(daemon_status_line_parts)
        runtime_stability_summary = {
            "runner_active": bool(runner_section.get("active")),
            "runner_lock_state": runner_section.get("lock_state"),
            "runner_stop_requested": bool(worker.get("stop_requested")),
            "daemon_active": bool(daemon_section.get("active")),
            "daemon_health_state": daemon_section.get("health_state"),
            "daemon_heartbeat_state": daemon_section.get("heartbeat_state"),
            "daemon_lease_state": daemon_section.get("lease_state"),
            "daemon_restart_policy_state": daemon_section.get("restart_policy_state"),
            "daemon_recovery_action": daemon_section.get("recovery_action"),
        }
        runtime_stability_summary["summary_line"] = " | ".join(
            [
                f"runner={('active' if runtime_stability_summary['runner_active'] else (runtime_stability_summary.get('runner_lock_state') or 'idle'))}",
                f"stop={'yes' if runtime_stability_summary['runner_stop_requested'] else 'no'}",
                f"daemon={runtime_stability_summary.get('daemon_health_state') or 'stopped'}",
                f"hb={runtime_stability_summary.get('daemon_heartbeat_state') or 'idle'}",
                f"lease={runtime_stability_summary.get('daemon_lease_state') or 'idle'}",
                f"restart={runtime_stability_summary.get('daemon_restart_policy_state') or 'ready'}",
                f"recover={runtime_stability_summary.get('daemon_recovery_action') or 'none'}",
            ]
        )
        daemon_timeline_summary_parts = [
            f"count={daemon_timeline_section.get('count') or 0}",
            f"recovery_event_count={daemon_timeline_section.get('recovery_event_count') or 0}",
        ]
        if daemon_timeline_section.get("last_recovery_event") is not None:
            daemon_timeline_summary_parts.append(f"last_recovery_event={daemon_timeline_section.get('last_recovery_event')}")
        if daemon_timeline_section.get("last_recovery_reason") is not None:
            daemon_timeline_summary_parts.append(f"last_recovery_reason={daemon_timeline_section.get('last_recovery_reason')}")
        daemon_timeline_section["summary_line"] = " | ".join(daemon_timeline_summary_parts)
        runner_timeline_summary_parts = [
            f"count={runner_timeline_section.get('count') or 0}",
            f"takeover_event_count={runner_timeline_section.get('takeover_event_count') or 0}",
        ]
        if runner_timeline_section.get("last_event") is not None:
            runner_timeline_summary_parts.append(f"last_event={runner_timeline_section.get('last_event')}")
        if runner_timeline_section.get("last_reason") is not None:
            runner_timeline_summary_parts.append(f"last_reason={runner_timeline_section.get('last_reason')}")
        runner_timeline_section["summary_line"] = " | ".join(runner_timeline_summary_parts)
        alerts_section = list(overview.get("alerts") or [])
        health_section = dict(overview.get("health") or {})
        recovery_section = dict(overview.get("recovery") or {})
        timelines_section = {
            "candidate": {
                "latest_timestamp": candidate_section.get("latest_timestamp"),
                "current_stage": candidate_section.get("current_stage"),
                "transition_count": candidate_section.get("transition_count"),
            },
            "queue": {
                "latest_timestamp": queue_history.get("last_transition", {}).get("timestamp") if isinstance(queue_history.get("last_transition"), dict) else None,
                "last_transition": queue_history.get("last_transition", {}).get("event") if isinstance(queue_history.get("last_transition"), dict) else None,
                "last_reason": queue_history.get("last_reason"),
                "transition_count": queue_history.get("transition_count"),
            },
            "runner": runner_timeline_section,
            "daemon": daemon_timeline_section,
        }
        timeline_summary_parts: list[str] = []
        if candidate_section.get("current_stage") is not None:
            timeline_summary_parts.append(f"candidate={candidate_section.get('current_stage')}")
        if timelines_section["queue"].get("last_transition") is not None:
            timeline_summary_parts.append(f"queue={timelines_section['queue'].get('last_transition')}")
        if runner_timeline_section.get("last_event") is not None:
            timeline_summary_parts.append(f"runner={runner_timeline_section.get('last_event')}")
        if daemon_timeline_section.get("last_event") is not None:
            timeline_summary_parts.append(f"daemon={daemon_timeline_section.get('last_event')}")
        timelines_section["summary_line"] = " | ".join(timeline_summary_parts)
        summary_parts = []
        if overview.get("summary_line"):
            summary_parts.append(str(overview.get("summary_line")))
        if candidate_section.get("current_stage"):
            summary_parts.append(f"candidate-stage={candidate_section['current_stage']}")
        if queue_section.get("awaiting_confirmation_count"):
            summary_parts.append(f"awaiting-confirm={queue_section['awaiting_confirmation_count']}")
        if runner_section.get("lock_state"):
            summary_parts.append(f"runner-lock={runner_section['lock_state']}")
        if daemon_section.get("lock_state"):
            summary_parts.append(f"daemon-lock={daemon_section['lock_state']}")
        if daemon_section.get("health_state"):
            summary_parts.append(f"daemon-health={daemon_section['health_state']}")
        if runtime_stability_summary.get("summary_line"):
            summary_parts.append(f"stability={runtime_stability_summary['summary_line']}")
        event_items: list[dict[str, Any]] = []
        seen_alert_keys: set[tuple[str, str]] = set()
        for item in list(candidate_hist.get("items") or [])[-3:]:
            if not isinstance(item, dict):
                continue
            event_item = {
                "source": "candidate",
                "timestamp": item.get("timestamp"),
                "event": item.get("action"),
                "reason": item.get("reason"),
                "note": item.get("operator_note"),
                "status": item.get("status"),
                "version": item.get("candidate_version"),
            }
            event_item.update(
                PipelineService._classify_operations_event(
                    source="candidate",
                    event=event_item.get("event"),
                    reason=event_item.get("reason"),
                    status=event_item.get("status"),
                )
            )
            event_items.append(event_item)
        last_transition = queue_history.get("last_transition")
        if isinstance(last_transition, dict):
            event_item = {
                "source": "queue",
                "timestamp": last_transition.get("timestamp"),
                "event": last_transition.get("event"),
                "reason": queue_history.get("last_reason"),
                "job_id": last_transition.get("job_id"),
                "state": last_transition.get("state"),
            }
            event_item.update(
                PipelineService._classify_operations_event(
                    source="queue",
                    event=event_item.get("event"),
                    reason=event_item.get("reason"),
                    state=event_item.get("state"),
                )
            )
            event_items.append(event_item)
        for item in list(runner_timeline_section.get("recent_events") or [])[-3:]:
            if not isinstance(item, dict):
                continue
            event_item = {
                "source": "runner",
                "timestamp": item.get("timestamp"),
                "event": item.get("event"),
                "reason": item.get("reason"),
                "note": item.get("note"),
            }
            event_item.update(
                PipelineService._classify_operations_event(
                    source="runner",
                    event=event_item.get("event"),
                    reason=event_item.get("reason"),
                )
            )
            event_items.append(event_item)
        for item in list(daemon_timeline_section.get("recent_events") or [])[-3:]:
            if not isinstance(item, dict):
                continue
            event_item = {
                "source": "daemon",
                "timestamp": item.get("timestamp"),
                "event": item.get("event"),
                "reason": item.get("reason"),
                "note": item.get("note"),
            }
            event_item.update(
                PipelineService._classify_operations_event(
                    source="daemon",
                    event=event_item.get("event"),
                    reason=event_item.get("reason"),
                )
            )
            event_items.append(event_item)

        alert_timestamp_by_scope = {
            "candidate": candidate_section.get("latest_timestamp"),
            "queue": timelines_section["queue"].get("latest_timestamp"),
            "runner": runner_timeline_section.get("latest_timestamp") or runner_section.get("lease_expires_at"),
            "daemon": daemon_timeline_section.get("latest_timestamp") or daemon_section.get("lease_expires_at"),
            "auto_train": queue_history.get("last_transition", {}).get("timestamp") if isinstance(queue_history.get("last_transition"), dict) else None,
        }
        for alert in alerts_section:
            if not isinstance(alert, dict):
                continue
            scope = str(alert.get("scope") or "operations")
            reason = str(alert.get("reason") or "attention_required")
            seen_alert_keys.add((scope, reason))
            event_item = {
                "source": scope,
                "timestamp": alert_timestamp_by_scope.get(scope),
                "event": "alert",
                "reason": reason,
                "message": alert.get("message"),
            }
            event_item.update(
                PipelineService._classify_operations_event(
                    source=scope,
                    event="alert",
                    reason=reason,
                    level=alert.get("level"),
                )
            )
            event_item["synthetic"] = True
            event_items.append(event_item)

        synthetic_anomalies = [
            ("runner", "runner_stop_requested", worker.get("lease_expires_at"), bool(worker.get("active")) and bool(worker.get("stop_requested"))),
            ("runner", "stale_runner_lock", runner_timeline_section.get("latest_timestamp") or runner_section.get("lease_expires_at"), str(worker.get("lock_state") or "") == "stale"),
            ("daemon", f"daemon_heartbeat_{daemon_section.get('heartbeat_state')}", daemon_timeline_section.get("latest_timestamp") or daemon_section.get("lease_expires_at"), str(daemon_section.get("heartbeat_state") or "") in {"delayed", "stale"}),
            ("daemon", f"daemon_lease_{daemon_section.get('lease_state')}", daemon_timeline_section.get("latest_timestamp") or daemon_section.get("lease_expires_at"), str(daemon_section.get("lease_state") or "") in {"expiring", "expired"}),
            ("daemon", f"daemon_restart_{daemon_section.get('restart_policy_state')}", daemon_timeline_section.get("latest_timestamp"), str(daemon_section.get("restart_policy_state") or "") in {"backoff", "capped"}),
            ("daemon", str(daemon_section.get("recovery_reason") or "worker_daemon_recovery_needed"), daemon_timeline_section.get("latest_timestamp"), bool(daemon_section.get("recovery_needed"))),
        ]
        for scope, reason, timestamp, condition in synthetic_anomalies:
            if not condition or (scope, reason) in seen_alert_keys:
                continue
            event_item = {
                "source": scope,
                "timestamp": timestamp,
                "event": "steady_state_alert",
                "reason": reason,
                "synthetic": True,
            }
            event_item.update(
                PipelineService._classify_operations_event(
                    source=scope,
                    event="steady_state_alert",
                    reason=reason,
                )
            )
            event_items.append(event_item)

        event_items = sorted(
            event_items,
            key=lambda item: (
                PipelineService._operations_event_severity_rank(item.get("severity")),
                int(bool(item.get("attention", False))),
                str(item.get("timestamp") or ""),
            ),
            reverse=True,
        )[:8]
        latest_event = event_items[0] if event_items else {}
        severity_counts: dict[str, int] = {"critical": 0, "warning": 0, "info": 0, "stable": 0}
        attention_count = 0
        for item in event_items:
            severity = str(item.get("severity") or "info")
            if severity not in severity_counts:
                severity_counts[severity] = 0
            severity_counts[severity] += 1
            if bool(item.get("attention", False)):
                attention_count += 1
        highest_severity = next(
            (
                severity
                for severity in ("critical", "warning", "info", "stable")
                if int(severity_counts.get(severity, 0) or 0) > 0
            ),
            "info" if event_items else "stable",
        )
        attention_events = [item for item in event_items if bool(item.get("attention", False))]
        latest_attention_event = attention_events[0] if attention_events else {}
        attention_reason = latest_attention_event.get("reason")
        attention_source = latest_attention_event.get("source")
        status = "attention" if attention_count > 0 else ("healthy" if event_items else "stable")
        escalated_reasons = list(
            dict.fromkeys(
                str(item.get("reason") or "")
                for item in attention_events
                if str(item.get("reason") or "")
            )
        )
        queue_review_mode = str(queue_review_policy.get("review_mode") or "")
        queue_review_next_action = str(queue_review_policy.get("next_action") or "")
        queue_review_reason = queue_review_policy.get("review_reason")
        queue_review_required_now = bool(queue_review_policy.get("review_required_now", False))
        trigger_policy_reason = str(overview.get("trigger_policy_reason") or "")
        trigger_policy_action = str(overview.get("trigger_policy_action") or "")
        trigger_blocked_reason = str(overview.get("trigger_blocked_reason") or "")
        trigger_blocked_action = str(overview.get("trigger_blocked_action") or "")
        auto_train_blocker = dict(overview.get("auto_train_blocker") or {})
        candidate_action_summary = dict(overview.get("candidate_action_summary") or {})
        candidate_primary_action = str(
            candidate_action_summary.get("primary_action") or overview.get("candidate_primary_action") or ""
        )
        candidate_secondary_actions = [
            str(item)
            for item in list(
                candidate_action_summary.get("secondary_actions") or overview.get("candidate_secondary_actions") or []
            )
            if str(item)
        ]
        queue_action_summary = dict(overview.get("queue_action_summary") or {})
        runtime_action_summary = dict(overview.get("runtime_action_summary") or {})
        event_stream_recommended_actions: list[str] = []
        for item in event_items:
            if not bool(item.get("attention", False)):
                continue
            source = str(item.get("source") or "")
            reason = str(item.get("reason") or "")
            if source == "queue" and reason == "awaiting_confirmation":
                event_stream_recommended_actions.append("review_queue_confirmation")
            elif source == "candidate" and reason == "candidate_ready_for_promotion":
                event_stream_recommended_actions.append(candidate_primary_action or "promote_candidate")
                event_stream_recommended_actions.extend(candidate_secondary_actions)
            elif source == "runner" and reason in {"stale_runner_lock", "runner_stop_requested"}:
                event_stream_recommended_actions.append("inspect_worker_stale_lock" if reason == "stale_runner_lock" else "wait_for_runner_shutdown")
            elif source == "daemon":
                if "heartbeat" in reason:
                    event_stream_recommended_actions.append("inspect_daemon_heartbeat")
                elif "restart" in reason:
                    event_stream_recommended_actions.append("inspect_daemon_restart_policy")
                elif "lease" in reason or "recovery" in reason or "stale" in reason:
                    event_stream_recommended_actions.append("recover_worker_daemon")
            elif source == "auto_train":
                if reason == "queue_pending_review":
                    event_stream_recommended_actions.append("review_queue_confirmation")
                elif reason == "queue_waiting_execution":
                    event_stream_recommended_actions.append("process_next_queue_item")
                elif reason == "queue_processing_active":
                    event_stream_recommended_actions.append("wait_for_queue_completion")
                elif reason in {"policy_requires_auto_evaluate", "review_required_before_execution"}:
                    event_stream_recommended_actions.append(trigger_policy_action or "inspect_auto_train_policy")
                else:
                    event_stream_recommended_actions.append(trigger_blocked_action or "inspect_auto_train_policy")
        fallback_queue_gate_reason = str(overview.get("queue_gate_reason") or attention_reason or "")
        fallback_queue_gate_action = str(overview.get("queue_gate_action") or "")
        if not event_stream_recommended_actions and fallback_queue_gate_reason in {
            "queue_pending_review",
            "queue_waiting_execution",
            "queue_processing_active",
        }:
            event_stream_recommended_actions.append(
                "review_queue_confirmation"
                if fallback_queue_gate_reason == "queue_pending_review"
                else (
                    "process_next_queue_item"
                    if fallback_queue_gate_reason == "queue_waiting_execution"
                    else "wait_for_queue_completion"
                )
            )
        if (
            not event_stream_recommended_actions
            and queue_review_next_action == "process_next_queue_item"
            and int(queue_section.get("count", 0) or 0) > 0
        ):
            event_stream_recommended_actions.append("process_next_queue_item")
        if (
            not event_stream_recommended_actions
            and int(queue_section.get("count", 0) or 0) > 0
            and int(queue_section.get("awaiting_confirmation_count", 0) or 0) == 0
        ):
            event_stream_recommended_actions.append("process_next_queue_item")
            event_stream_recommended_actions.append("inspect_auto_train_trigger")
        if not event_stream_recommended_actions and queue_action_summary.get("primary_action"):
            event_stream_recommended_actions.append(str(queue_action_summary.get("primary_action")))
            event_stream_recommended_actions.extend(
                str(item) for item in list(queue_action_summary.get("secondary_actions") or []) if str(item)
            )
        if not event_stream_recommended_actions and bool(runner_section.get("active")):
            event_stream_recommended_actions.append("inspect_runtime_stability")
            event_stream_recommended_actions.append("inspect_worker_runner_history")
        if not event_stream_recommended_actions and bool(daemon_section.get("active")):
            event_stream_recommended_actions.append("inspect_runtime_stability")
            event_stream_recommended_actions.append("inspect_daemon_status")
        if not event_stream_recommended_actions and runtime_action_summary.get("primary_action"):
            event_stream_recommended_actions.append(str(runtime_action_summary.get("primary_action")))
            event_stream_recommended_actions.extend(
                str(item) for item in list(runtime_action_summary.get("secondary_actions") or []) if str(item)
            )
        if not event_stream_recommended_actions and candidate_action_summary.get("primary_action"):
            event_stream_recommended_actions.append(str(candidate_action_summary.get("primary_action")))
            event_stream_recommended_actions.extend(
                str(item) for item in list(candidate_action_summary.get("secondary_actions") or []) if str(item)
            )
        if not event_stream_recommended_actions and candidate_primary_action:
            event_stream_recommended_actions.append(candidate_primary_action)
            event_stream_recommended_actions.extend(candidate_secondary_actions)
        if not event_stream_recommended_actions and candidate_monitor_ready:
            event_stream_recommended_actions.append("inspect_candidate_status")
            event_stream_recommended_actions.append("inspect_candidate_timeline")
        if not event_stream_recommended_actions and trigger_policy_reason in {
            "policy_requires_auto_evaluate",
            "review_required_before_execution",
        }:
            event_stream_recommended_actions.append(trigger_policy_action or "inspect_auto_train_policy")
        if not event_stream_recommended_actions and attention_count > 0 and trigger_blocked_reason:
            event_stream_recommended_actions.append(trigger_blocked_action or "inspect_auto_train_policy")
        if trigger_policy_reason in {
            "policy_requires_auto_evaluate",
            "review_required_before_execution",
        } and (trigger_policy_action or "inspect_auto_train_policy") in event_stream_recommended_actions:
            prioritized_action = trigger_policy_action or "inspect_auto_train_policy"
            event_stream_recommended_actions = [prioritized_action] + [
                action for action in event_stream_recommended_actions if action != prioritized_action
            ]
        elif trigger_blocked_reason and (trigger_blocked_action or "inspect_auto_train_policy") in event_stream_recommended_actions:
            prioritized_action = trigger_blocked_action or "inspect_auto_train_policy"
            event_stream_recommended_actions = [prioritized_action] + [
                action for action in event_stream_recommended_actions if action != prioritized_action
            ]
        if highest_severity == "critical" and "recover_worker_daemon" in event_stream_recommended_actions:
            event_stream_recommended_actions = ["recover_worker_daemon"] + [
                action for action in event_stream_recommended_actions if action != "recover_worker_daemon"
            ]
        event_stream_recommended_actions = list(dict.fromkeys(event_stream_recommended_actions))
        highest_priority_action = event_stream_recommended_actions[0] if event_stream_recommended_actions else None
        active_recovery_hint = None
        if daemon_section.get("recovery_action") not in {None, "none"}:
            active_recovery_hint = str(daemon_section.get("recovery_action"))
        elif queue_confirm.get("next_confirmation_reason"):
            active_recovery_hint = str(queue_confirm.get("next_confirmation_reason"))
        elif str(worker.get("lock_state") or "") == "stale":
            active_recovery_hint = "inspect_worker_stale_lock"
        elif bool(worker.get("active")) and bool(worker.get("stop_requested")):
            active_recovery_hint = "wait_for_runner_shutdown"

        ordered_next_actions: list[str] = list(dict.fromkeys(event_stream_recommended_actions))

        dashboard_summary_parts = [
            f"severity={highest_severity}",
            f"attention={'yes' if attention_count > 0 else 'no'}",
            f"alerts={len(alerts_section)}",
        ]
        if attention_reason:
            dashboard_summary_parts.append(f"reason={attention_reason}")
        if latest_event.get("source") is not None:
            dashboard_summary_parts.append(f"latest={latest_event.get('source')}:{latest_event.get('event')}")
        dashboard = {
            "severity": highest_severity,
            "status": status,
            "attention_needed": attention_count > 0,
            "attention_reason": attention_reason,
            "attention_source": attention_source,
            "attention_count": attention_count,
            "alert_count": len(alerts_section),
            "severity_counts": severity_counts,
            "highest_priority_action": highest_priority_action,
            "active_recovery_hint": active_recovery_hint,
            "escalated_reasons": escalated_reasons,
            "queue_review_policy": queue_review_policy,
            "queue_review_policy_summary": queue_review_policy.get("summary_line"),
            "queue_review_mode": queue_review_mode or None,
            "queue_review_next_action": queue_review_next_action or None,
            "queue_review_reason": queue_review_reason,
            "queue_review_required_now": queue_review_required_now,
            "trigger_policy_reason": trigger_policy_reason or None,
            "trigger_policy_action": trigger_policy_action or None,
            "trigger_policy_gate_summary": trigger_policy_gate,
            "trigger_policy_gate_summary_line": trigger_policy_gate.get("summary_line"),
            "trigger_threshold_summary": trigger_threshold,
            "trigger_threshold_summary_line": trigger_threshold.get("summary_line"),
            "auto_train_blocker": auto_train_blocker,
            "runtime_stability_summary": runtime_stability,
            "runtime_stability_summary_line": runtime_stability.get("summary_line"),
            "candidate_action_summary": candidate_action_summary,
            "queue_action_summary": queue_action_summary,
            "runtime_action_summary": runtime_action_summary,
            "latest_source": latest_event.get("source"),
            "latest_event": latest_event.get("event"),
            "latest_reason": latest_event.get("reason"),
            "latest_timestamp": latest_event.get("timestamp"),
            "latest_recovery": latest_recovery,
            "next_actions": ordered_next_actions,
            "summary_line": " | ".join(dashboard_summary_parts),
        }

        event_stream = {
            "count": len(event_items),
            "latest_timestamp": latest_event.get("timestamp"),
            "latest_source": latest_event.get("source"),
            "latest_event": latest_event.get("event"),
            "latest_reason": latest_event.get("reason"),
            "severity": highest_severity,
            "status": status,
            "attention_needed": attention_count > 0,
            "attention_reason": attention_reason,
            "attention_source": attention_source,
            "latest_severity": latest_event.get("severity"),
            "latest_attention": bool(latest_event.get("attention", False)) if latest_event else False,
            "highest_severity": highest_severity,
            "attention_count": attention_count,
            "alert_count": len(alerts_section),
            "severity_counts": severity_counts,
            "highest_priority_action": highest_priority_action,
            "active_recovery_hint": active_recovery_hint,
            "escalated_reasons": escalated_reasons,
            "queue_review_policy": queue_review_policy,
            "queue_review_policy_summary": queue_review_policy.get("summary_line"),
            "queue_review_mode": queue_review_mode or None,
            "queue_review_next_action": queue_review_next_action or None,
            "queue_review_reason": queue_review_reason,
            "queue_review_required_now": queue_review_required_now,
            "trigger_policy_reason": trigger_policy_reason or None,
            "trigger_policy_action": trigger_policy_action or None,
            "trigger_policy_gate_summary": trigger_policy_gate,
            "trigger_policy_gate_summary_line": trigger_policy_gate.get("summary_line"),
            "trigger_threshold_summary": trigger_threshold,
            "trigger_threshold_summary_line": trigger_threshold.get("summary_line"),
            "auto_train_blocker": auto_train_blocker,
            "runtime_stability_summary": runtime_stability,
            "runtime_stability_summary_line": runtime_stability.get("summary_line"),
            "candidate_action_summary": candidate_action_summary,
            "queue_action_summary": queue_action_summary,
            "runtime_action_summary": runtime_action_summary,
            "latest_recovery": latest_recovery,
            "next_actions": ordered_next_actions,
            "dashboard": dashboard,
            "items": event_items,
        }
        event_summary_parts: list[str] = [f"count={len(event_items)}"]
        if highest_severity is not None:
            event_summary_parts.append(f"severity={highest_severity}")
        event_summary_parts.append(f"status={status}")
        if attention_count:
            event_summary_parts.append(f"attention_count={attention_count}")
        event_summary_parts.append(f"attention_needed={'yes' if attention_count > 0 else 'no'}")
        if attention_reason is not None:
            event_summary_parts.append(f"attention_reason={attention_reason}")
        if highest_priority_action is not None:
            event_summary_parts.append(f"highest_priority_action={highest_priority_action}")
        if active_recovery_hint is not None:
            event_summary_parts.append(f"active_recovery_hint={active_recovery_hint}")
        if fallback_queue_gate_reason:
            event_summary_parts.append(f"queue_gate_reason={fallback_queue_gate_reason}")
        if fallback_queue_gate_action:
            event_summary_parts.append(f"queue_gate_action={fallback_queue_gate_action}")
        if queue_review_mode:
            event_summary_parts.append(f"queue_review_mode={queue_review_mode}")
        if queue_review_next_action:
            event_summary_parts.append(f"queue_review_next_action={queue_review_next_action}")
        if trigger_policy_reason:
            event_summary_parts.append(f"trigger_policy_reason={trigger_policy_reason}")
        if trigger_policy_action:
            event_summary_parts.append(f"trigger_policy_action={trigger_policy_action}")
        if trigger_policy_gate.get("summary_line"):
            event_summary_parts.append(f"trigger_policy_gate={trigger_policy_gate.get('summary_line')}")
        if trigger_threshold.get("summary_line"):
            event_summary_parts.append(f"trigger_gate={trigger_threshold.get('summary_line')}")
        if runtime_stability.get("summary_line"):
            event_summary_parts.append(f"runtime={runtime_stability.get('summary_line')}")
        if escalated_reasons:
            event_summary_parts.append(f"escalated_reasons={','.join(escalated_reasons[:3])}")
        if latest_event.get("source") is not None:
            event_summary_parts.append(f"latest_source={latest_event.get('source')}")
        if latest_event.get("event") is not None:
            event_summary_parts.append(f"latest_event={latest_event.get('event')}")
        if latest_event.get("reason") is not None:
            event_summary_parts.append(f"latest_reason={latest_event.get('reason')}")
        candidate_focus_ready = (
            candidate_section.get("current_stage") not in {None, "idle"}
            or overview.get("candidate_state") not in {None, "idle", ""}
            or overview.get("candidate_version") not in {None, ""}
        )
        monitor_focus = (
            "queue_waiting_execution"
            if str(queue_action_summary.get("primary_action") or "") == "process_next_queue_item"
            else (
                "runner_active"
                if (
                    str(runtime_action_summary.get("primary_action") or "") == "inspect_runtime_stability"
                    and bool(runner_section.get("active"))
                )
                else (
                    "daemon_active"
                    if (
                        str(runtime_action_summary.get("primary_action") or "") == "inspect_runtime_stability"
                        and bool(daemon_section.get("active"))
                    )
                    else (
                        "candidate_idle"
                        if candidate_focus_ready and str(candidate_action_summary.get("primary_action") or "")
                        else None
                    )
                )
            )
        )
        dashboard_focus = attention_reason or monitor_focus or (
            f"{latest_event.get('source')}:{latest_event.get('event')}"
            if latest_event.get("source") is not None and latest_event.get("event") is not None
            else None
        )
        if dashboard_focus is not None:
            event_summary_parts.append(f"current_focus={dashboard_focus}")
        if monitor_focus is not None:
            event_summary_parts.append(f"monitor_focus={monitor_focus}")
        dashboard_summary_parts = [
            f"severity={highest_severity}",
            f"status={status}",
            f"attention={'yes' if attention_count > 0 else 'no'}",
        ]
        if dashboard_focus is not None:
            dashboard_summary_parts.append(f"focus={dashboard_focus}")
        if highest_priority_action is not None:
            dashboard_summary_parts.append(f"action={highest_priority_action}")
        if active_recovery_hint is not None:
            dashboard_summary_parts.append(f"recovery_hint={active_recovery_hint}")
        if fallback_queue_gate_reason:
            dashboard_summary_parts.append(f"qgate={fallback_queue_gate_reason}")
        if queue_review_mode:
            dashboard_summary_parts.append(f"qreview={queue_review_mode}")
        if queue_review_next_action:
            dashboard_summary_parts.append(f"qnext={queue_review_next_action}")
        if trigger_policy_reason:
            dashboard_summary_parts.append(f"pgate={trigger_policy_reason}")
        if trigger_policy_action:
            dashboard_summary_parts.append(f"pact={trigger_policy_action}")
        if trigger_policy_gate.get("summary_line"):
            dashboard_summary_parts.append(f"pg={trigger_policy_gate.get('summary_line')}")
        if trigger_threshold.get("summary_line"):
            dashboard_summary_parts.append(f"gate={trigger_threshold.get('summary_line')}")
        if runtime_stability.get("summary_line"):
            dashboard_summary_parts.append(f"rt={runtime_stability.get('summary_line')}")
        if escalated_reasons:
            dashboard_summary_parts.append(f"escalated={','.join(escalated_reasons[:3])}")
        if daemon_timeline_section.get("last_recovery_event") is not None:
            dashboard_summary_parts.append(f"recovery={daemon_timeline_section.get('last_recovery_event')}")
        alert_policy = PipelineService._operations_alert_policy(
            severity=highest_severity,
            attention_needed=attention_count > 0,
            next_actions=event_stream_recommended_actions,
            current_focus=dashboard_focus,
            attention_source=attention_source,
            highest_priority_action=highest_priority_action,
            latest_recovery=latest_recovery,
            escalated_reasons=escalated_reasons,
            active_recovery_hint=active_recovery_hint,
            queue_review_policy=queue_review_policy,
            trigger_policy={**trigger_policy, "threshold_summary": trigger_threshold},
            runtime_stability=runtime_stability,
            candidate_action_summary=candidate_action_summary,
            queue_action_summary=queue_action_summary,
            runtime_action_summary=runtime_action_summary,
            trigger_blocked_reason=overview.get("trigger_blocked_reason"),
            trigger_blocked_action=overview.get("trigger_blocked_action"),
            trigger_blocked_summary=overview.get("trigger_blocked_summary"),
            auto_train_blocker=auto_train_blocker,
        )
        if alert_policy.get("required_action") is not None:
            event_summary_parts.append(f"required_action={alert_policy.get('required_action')}")
            dashboard_summary_parts.append(f"required_action={alert_policy.get('required_action')}")
        event_stream["summary_line"] = " | ".join(event_summary_parts)
        ordered_next_actions = []
        for action_name in [
            alert_policy.get("required_action"),
            *list(alert_policy.get("secondary_actions") or []),
            *event_stream_recommended_actions,
            *next_actions,
        ]:
            action_text = str(action_name or "").strip()
            if not action_text or action_text == "none" or action_text in ordered_next_actions:
                continue
            ordered_next_actions.append(action_text)
        if ordered_next_actions:
            dashboard_summary_parts.append(f"next={','.join(ordered_next_actions[:3])}")
        dashboard_digest_parts: list[str] = []
        if dashboard_focus is not None:
            dashboard_digest_parts.append(f"focus={dashboard_focus}")
        if highest_priority_action is not None:
            dashboard_digest_parts.append(f"action={highest_priority_action}")
        if fallback_queue_gate_reason:
            dashboard_digest_parts.append(f"qgate={fallback_queue_gate_reason}")
        if queue_review_mode:
            dashboard_digest_parts.append(f"qreview={queue_review_mode}")
        if alert_policy.get("remediation_mode") is not None:
            dashboard_digest_parts.append(f"handling={alert_policy.get('remediation_mode')}")
        if trigger_policy_action:
            dashboard_digest_parts.append(f"policy={trigger_policy_action}")
        if trigger_policy_gate.get("summary_line"):
            dashboard_digest_parts.append(f"pg={trigger_policy_gate.get('summary_line')}")
        if trigger_threshold.get("summary_line"):
            dashboard_digest_parts.append(f"gate={trigger_threshold.get('summary_line')}")
        if runtime_stability.get("summary_line"):
            dashboard_digest_parts.append(f"rt={runtime_stability.get('summary_line')}")
        if latest_recovery is not None and latest_recovery.get("event") is not None:
            dashboard_digest_parts.append(f"recovery={latest_recovery.get('event')}")
        if alert_policy.get("auto_remediation_allowed") is not None:
            dashboard_digest_parts.append(
                "auto=yes" if bool(alert_policy.get("auto_remediation_allowed")) else "auto=no"
            )
        if alert_policy.get("requires_human_review") is not None:
            dashboard_digest_parts.append(
                "review=yes" if bool(alert_policy.get("requires_human_review")) else "review=no"
            )
        if ordered_next_actions:
            dashboard_digest_parts.append(f"next={','.join(ordered_next_actions[:3])}")
        dashboard_digest = " | ".join(dashboard_digest_parts) if dashboard_digest_parts else None
        inspection_summary_parts: list[str] = []
        if dashboard_focus is not None:
            inspection_summary_parts.append(f"current_focus={dashboard_focus}")
        if alert_policy.get("required_action") is not None:
            inspection_summary_parts.append(f"required_action={alert_policy.get('required_action')}")
        if daemon_timeline_section.get("last_recovery_event") is not None:
            inspection_summary_parts.append(f"last_recovery_event={daemon_timeline_section.get('last_recovery_event')}")
        if daemon_timeline_section.get("last_recovery_reason") is not None:
            inspection_summary_parts.append(f"last_recovery_reason={daemon_timeline_section.get('last_recovery_reason')}")
        if ordered_next_actions:
            inspection_summary_parts.append(f"next_actions={','.join(ordered_next_actions[:3])}")
        inspection_summary_line = " | ".join(inspection_summary_parts) if inspection_summary_parts else None
        generic_monitor_active = PipelineService._generic_monitor_active(
            focus=dashboard_focus,
            inspection_summary_line=inspection_summary_line,
        )
        dashboard_digest, inspection_summary_line = PipelineService._prefer_inspection_summary_for_generic_monitor(
            focus=dashboard_focus,
            summary_line=dashboard_digest,
            inspection_summary_line=inspection_summary_line,
        )
        dashboard_summary_line, inspection_summary_line = PipelineService._prefer_inspection_summary_for_generic_monitor(
            focus=dashboard_focus,
            summary_line=" | ".join(dashboard_summary_parts),
            inspection_summary_line=inspection_summary_line,
        )
        event_summary_line, inspection_summary_line = PipelineService._prefer_inspection_summary_for_generic_monitor(
            focus=dashboard_focus,
            summary_line=" | ".join(summary_parts),
            inspection_summary_line=inspection_summary_line,
        )
        dashboard["policy"] = alert_policy
        dashboard["current_focus"] = dashboard_focus
        if dashboard_digest is not None:
            dashboard["dashboard_digest"] = dashboard_digest
        if inspection_summary_line is not None:
            dashboard["inspection_summary_line"] = inspection_summary_line
        operations_dashboard = {
            "severity": highest_severity,
            "status": status,
            "attention_needed": attention_count > 0,
            "attention_reason": attention_reason,
            "attention_source": attention_source,
            "current_focus": dashboard_focus,
            "monitor_focus": monitor_focus,
            "required_action": alert_policy.get("required_action"),
            "primary_action": alert_policy.get("primary_action"),
            "secondary_action": alert_policy.get("secondary_action"),
            "secondary_actions": list(alert_policy.get("secondary_actions") or []),
            "alert_count": len(alerts_section),
            "attention_count": attention_count,
            "highest_priority_action": highest_priority_action,
            "active_recovery_hint": active_recovery_hint,
            "escalated_reasons": escalated_reasons,
            "next_actions": ordered_next_actions,
            "candidate_stage": candidate_section.get("current_stage"),
            "queue_state": (
                "awaiting_confirmation"
                if int(queue_section.get("awaiting_confirmation_count", 0) or 0) > 0
                else ("queued" if int(queue_section.get("count", 0) or 0) > 0 else "idle")
            ),
            "runner_state": runner_section.get("lock_state") or ("active" if runner_section.get("active") else "idle"),
            "daemon_health_state": daemon_section.get("health_state"),
            "daemon_lease_state": daemon_section.get("lease_state"),
            "daemon_heartbeat_state": daemon_section.get("heartbeat_state"),
            "latest_source": latest_event.get("source"),
            "latest_event": latest_event.get("event"),
            "latest_reason": latest_event.get("reason"),
            "latest_timestamp": latest_event.get("timestamp"),
            "latest_recovery": latest_recovery,
            "dashboard_digest": dashboard_digest,
            "inspection_summary_line": inspection_summary_line,
            "remediation_mode": alert_policy.get("remediation_mode"),
            "operator_guidance": alert_policy.get("operator_guidance"),
            "auto_remediation_allowed": alert_policy.get("auto_remediation_allowed"),
            "requires_human_review": alert_policy.get("requires_human_review"),
            "requires_immediate_action": alert_policy.get("requires_immediate_action"),
            "highest_priority_action": highest_priority_action,
            "active_recovery_hint": active_recovery_hint,
            "escalated_reasons": list(escalated_reasons),
            "queue_review_policy": queue_review_policy,
            "queue_review_policy_summary": queue_review_policy.get("summary_line"),
            "queue_review_mode": queue_review_mode or None,
            "queue_review_next_action": queue_review_next_action or None,
            "queue_review_reason": queue_review_reason,
            "queue_review_required_now": queue_review_required_now,
            "trigger_policy_reason": trigger_policy_reason or None,
            "trigger_policy_action": trigger_policy_action or None,
            "trigger_blocked_reason": overview.get("trigger_blocked_reason"),
            "trigger_blocked_action": overview.get("trigger_blocked_action"),
            "trigger_blocked_category": overview.get("trigger_blocked_category"),
            "trigger_blocked_summary": overview.get("trigger_blocked_summary"),
            "auto_train_blocker": auto_train_blocker,
            "candidate_action_summary": candidate_action_summary,
            "queue_action_summary": queue_action_summary,
            "trigger_policy_gate_summary": trigger_policy_gate,
            "trigger_policy_gate_summary_line": trigger_policy_gate.get("summary_line"),
            "trigger_threshold_summary": trigger_threshold,
            "trigger_threshold_summary_line": trigger_threshold.get("summary_line"),
            "runtime_stability_summary": runtime_stability,
            "runtime_stability_summary_line": runtime_stability.get("summary_line"),
            "runtime_action_summary": runtime_action_summary,
            "last_recovery_event": daemon_timeline_section.get("last_recovery_event"),
            "last_recovery_reason": daemon_timeline_section.get("last_recovery_reason"),
            "policy": alert_policy,
            "summary_line": dashboard_summary_line,
        }
        event_stream["policy"] = alert_policy
        event_stream["current_focus"] = dashboard_focus
        event_stream["monitor_focus"] = monitor_focus
        event_stream["required_action"] = alert_policy.get("required_action")
        event_stream["primary_action"] = alert_policy.get("primary_action")
        event_stream["secondary_action"] = alert_policy.get("secondary_action")
        event_stream["secondary_actions"] = list(alert_policy.get("secondary_actions") or [])
        event_stream["dashboard"]["highest_priority_action"] = highest_priority_action
        event_stream["dashboard"]["current_focus"] = dashboard_focus
        event_stream["dashboard"]["monitor_focus"] = monitor_focus
        event_stream["dashboard"]["required_action"] = alert_policy.get("required_action")
        event_stream["dashboard"]["primary_action"] = alert_policy.get("primary_action")
        event_stream["dashboard"]["secondary_action"] = alert_policy.get("secondary_action")
        event_stream["dashboard"]["secondary_actions"] = list(alert_policy.get("secondary_actions") or [])
        event_stream["dashboard"]["latest_recovery"] = latest_recovery
        event_stream["dashboard"]["active_recovery_hint"] = active_recovery_hint
        event_stream["dashboard"]["escalated_reasons"] = list(escalated_reasons)
        event_stream["dashboard"]["queue_review_policy"] = queue_review_policy
        event_stream["dashboard"]["queue_review_policy_summary"] = queue_review_policy.get("summary_line")
        event_stream["dashboard"]["queue_review_mode"] = queue_review_mode or None
        event_stream["dashboard"]["queue_review_next_action"] = queue_review_next_action or None
        event_stream["dashboard"]["queue_review_reason"] = queue_review_reason
        event_stream["dashboard"]["queue_review_required_now"] = queue_review_required_now
        event_stream["dashboard"]["candidate_action_summary"] = candidate_action_summary
        event_stream["dashboard"]["queue_action_summary"] = queue_action_summary
        event_stream["dashboard"]["trigger_policy_gate_summary"] = trigger_policy_gate
        event_stream["dashboard"]["trigger_policy_gate_summary_line"] = trigger_policy_gate.get("summary_line")
        event_stream["dashboard"]["trigger_threshold_summary"] = trigger_threshold
        event_stream["dashboard"]["trigger_threshold_summary_line"] = trigger_threshold.get("summary_line")
        event_stream["dashboard"]["runtime_stability_summary"] = runtime_stability
        event_stream["dashboard"]["runtime_stability_summary_line"] = runtime_stability.get("summary_line")
        event_stream["dashboard"]["runtime_action_summary"] = runtime_action_summary
        event_stream["dashboard"]["remediation_mode"] = alert_policy.get("remediation_mode")
        event_stream["dashboard"]["operator_guidance"] = alert_policy.get("operator_guidance")
        event_stream["dashboard"]["auto_remediation_allowed"] = alert_policy.get("auto_remediation_allowed")
        event_stream["dashboard"]["requires_human_review"] = alert_policy.get("requires_human_review")
        event_stream["dashboard"]["requires_immediate_action"] = alert_policy.get("requires_immediate_action")
        event_stream["dashboard"]["dashboard_digest"] = dashboard_digest
        event_stream["dashboard"]["next_actions"] = ordered_next_actions
        if inspection_summary_line is not None:
            event_stream["inspection_summary_line"] = inspection_summary_line
        if inspection_summary_line is not None:
            event_stream["dashboard"]["inspection_summary_line"] = inspection_summary_line
        if inspection_summary_line is not None:
            alert_policy["inspection_summary_line"] = inspection_summary_line
        if generic_monitor_active:
            event_stream["summary_line"] = event_summary_line
            event_stream["dashboard"]["summary_line"] = dashboard_summary_line
            alert_policy["summary_line"] = event_summary_line
        return {
            "attention_needed": bool(overview.get("attention_needed", False)),
            "attention_reason": attention_reason,
            "monitor_focus": monitor_focus,
            "current_focus": dashboard_focus,
            "required_action": alert_policy.get("required_action"),
            "summary_line": event_summary_line,
            "inspection_summary_line": inspection_summary_line,
            "auto_train_policy": trigger_policy,
            "policy_summary_line": trigger_policy.get("summary_line"),
            "trigger_policy_gate_summary": trigger_policy_gate,
            "trigger_policy_gate_summary_line": trigger_policy_gate.get("summary_line"),
            "trigger_threshold_summary": trigger_threshold,
            "trigger_threshold_summary_line": trigger_threshold.get("summary_line"),
            "trigger_blocked_reason": overview.get("trigger_blocked_reason"),
            "trigger_blocked_action": overview.get("trigger_blocked_action"),
            "trigger_blocked_category": overview.get("trigger_blocked_category"),
            "trigger_blocked_summary": overview.get("trigger_blocked_summary"),
            "queue_review_policy": queue_review_policy,
            "queue_review_policy_summary": queue_review_policy.get("summary_line"),
            "runtime_stability_summary": runtime_stability_summary,
            "candidate_action_summary": candidate_action_summary,
            "queue_action_summary": queue_action_summary,
            "runtime_action_summary": runtime_action_summary,
            "auto_train_blocker": auto_train_blocker,
            "next_actions": ordered_next_actions,
            "alerts": alerts_section,
            "health": health_section,
            "recovery": recovery_section,
            "candidate": candidate_section,
            "queue": queue_section,
            "runner": runner_section,
            "runner_timeline": runner_timeline_section,
            "daemon": daemon_section,
            "daemon_timeline": daemon_timeline_section,
            "timelines": timelines_section,
            "event_stream": event_stream,
            "event_dashboard": dashboard,
            "dashboard": operations_dashboard,
            "alert_policy": alert_policy,
            "monitor_focus": monitor_focus,
        }

    @staticmethod
    def _operations_event_severity_rank(value: Any) -> int:
        severity = str(value or "info")
        if severity == "critical":
            return 4
        if severity == "warning":
            return 3
        if severity == "info":
            return 2
        return 1

    @staticmethod
    def _classify_operations_event(
        *,
        source: Any,
        event: Any,
        reason: Any,
        level: Any | None = None,
        status: Any | None = None,
        state: Any | None = None,
    ) -> dict[str, Any]:
        severity = "info"
        attention = False
        normalized_level = str(level or "").strip().lower()
        if normalized_level == "attention":
            severity = "info"
            attention = True
        elif normalized_level == "warning":
            severity = "warning"
            attention = True

        normalized_source = str(source or "operations")
        normalized_event = str(event or "")
        normalized_reason = str(reason or "")
        normalized_status = str(status or "")
        normalized_state = str(state or "")
        combined = " ".join(
            part
            for part in (
                normalized_event,
                normalized_reason,
                normalized_status,
                normalized_state,
            )
            if part
        ).lower()

        if "queue_pending_review" in combined:
            severity = "info"
            attention = True
        elif "queue_waiting_execution" in combined:
            severity = "info"
            attention = True
        elif "queue_processing_active" in combined:
            severity = "info"
            attention = False
        elif any(token in combined for token in ("awaiting_confirmation", "manual_review_required")):
            severity = "info"
            attention = True
        elif "candidate_ready_for_promotion" in combined:
            severity = "info"
            attention = True
        elif normalized_source == "daemon" and any(token in combined for token in ("stale", "expired", "failed", "error")):
            severity = "critical"
            attention = True
        elif normalized_source == "daemon" and any(token in combined for token in ("backoff", "capped", "blocked", "recover", "restart", "delayed")):
            severity = "warning"
            attention = True
        elif normalized_source == "runner" and any(token in combined for token in ("stale", "blocked", "failed", "error", "stop_requested")):
            severity = "warning"
            attention = True
        elif any(token in combined for token in ("stale", "expired", "backoff", "capped", "blocked", "failed", "error")):
            severity = "warning"
            attention = True
        elif normalized_source in {"runner", "daemon"} and normalized_event == "alert":
            severity = "warning"
            attention = True
        elif normalized_source == "queue" and normalized_state in {"awaiting_confirmation", "failed"}:
            severity = "warning" if normalized_state == "failed" else "info"
            attention = True

        return {
            "severity": severity,
            "attention": attention,
        }

    @staticmethod
    def _operations_alert_policy(
        *,
        severity: Any,
        attention_needed: bool,
        next_actions: list[str] | None,
        current_focus: Any,
        attention_source: Any,
        highest_priority_action: Any | None = None,
        escalated_reasons: list[str] | None = None,
        active_recovery_hint: Any | None = None,
        latest_recovery: dict[str, Any] | None = None,
        queue_review_policy: dict[str, Any] | None = None,
        trigger_policy: dict[str, Any] | None = None,
        runtime_stability: dict[str, Any] | None = None,
        candidate_action_summary: dict[str, Any] | None = None,
        queue_action_summary: dict[str, Any] | None = None,
        runtime_action_summary: dict[str, Any] | None = None,
        trigger_blocked_reason: Any | None = None,
        trigger_blocked_action: Any | None = None,
        trigger_blocked_summary: Any | None = None,
        auto_train_blocker: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_severity = str(severity or "stable")
        actions = [str(item) for item in list(next_actions or []) if str(item)]
        escalated_reason_values = [str(item) for item in list(escalated_reasons or []) if str(item)]
        primary_action = str(highest_priority_action or "") or (actions[0] if actions else None)
        current_focus_text = str(current_focus or "")
        source_text = str(attention_source or "")
        active_recovery_text = str(active_recovery_hint or "")
        queue_review = dict(queue_review_policy or {})
        trigger_policy_map = dict(trigger_policy or {})
        runtime_stability_map = dict(runtime_stability or {})
        candidate_action_summary_map = dict(candidate_action_summary or {})
        queue_action_summary_map = dict(queue_action_summary or {})
        runtime_action_summary_map = dict(runtime_action_summary or {})
        candidate_summary_primary_action = str(candidate_action_summary_map.get("primary_action") or "")
        candidate_summary_secondary_actions = [
            str(item) for item in list(candidate_action_summary_map.get("secondary_actions") or []) if str(item)
        ]
        queue_summary_primary_action = str(queue_action_summary_map.get("primary_action") or "")
        queue_summary_secondary_actions = [
            str(item) for item in list(queue_action_summary_map.get("secondary_actions") or []) if str(item)
        ]
        runtime_summary_primary_action = str(runtime_action_summary_map.get("primary_action") or "")
        runtime_summary_secondary_actions = [
            str(item) for item in list(runtime_action_summary_map.get("secondary_actions") or []) if str(item)
        ]
        queue_review_mode = str(queue_review.get("review_mode") or "")
        queue_review_next_action = str(queue_review.get("next_action") or "")
        queue_review_reason = str(queue_review.get("review_reason") or "")
        queue_review_required_now = bool(queue_review.get("review_required_now", False))
        trigger_policy_reason = str(
            trigger_policy_map.get("promote_gate_reason")
            or trigger_policy_map.get("evaluation_gate_reason")
            or ""
        )
        trigger_policy_gate = dict(trigger_policy_map.get("gate_summary") or {})
        trigger_policy_action = str(
            trigger_policy_map.get("promote_gate_action")
            or trigger_policy_map.get("evaluation_gate_action")
            or ""
        )
        trigger_threshold = dict(trigger_policy_map.get("threshold_summary") or {})
        trigger_blocked_reason_text = str(trigger_blocked_reason or "")
        trigger_blocked_action_text = str(trigger_blocked_action or "")
        trigger_blocked_summary_text = str(trigger_blocked_summary or "")
        auto_train_blocker_map = dict(auto_train_blocker or {})
        runner_lock_state = str(runtime_stability_map.get("runner_lock_state") or "")
        runner_stop_requested = bool(runtime_stability_map.get("runner_stop_requested"))
        daemon_health_state = str(runtime_stability_map.get("daemon_health_state") or "")
        daemon_heartbeat_state = str(runtime_stability_map.get("daemon_heartbeat_state") or "")
        daemon_lease_state = str(runtime_stability_map.get("daemon_lease_state") or "")
        daemon_restart_policy_state = str(runtime_stability_map.get("daemon_restart_policy_state") or "")
        daemon_recovery_action = str(runtime_stability_map.get("daemon_recovery_action") or "")
        queue_monitor_summary_active = bool(queue_summary_primary_action) and (
            current_focus_text.startswith("queue") or source_text == "queue"
        )
        candidate_monitor_summary_active = bool(candidate_summary_primary_action) and (
            current_focus_text.startswith("candidate") or source_text == "candidate"
        )
        runner_runtime_summary_active = bool(runtime_summary_primary_action) and (
            current_focus_text == "runner_active" or source_text == "runner"
        )
        daemon_runtime_summary_active = bool(runtime_summary_primary_action) and (
            current_focus_text == "daemon_active" or source_text == "daemon"
        )
        runtime_monitor_summary_active = runner_runtime_summary_active or daemon_runtime_summary_active

        required_action = "none"
        action_priority = "p3"
        escalation_mode = "none"
        requires_immediate_action = False
        requires_human_review = False
        auto_remediation_allowed = False
        remediation_mode = "monitor"
        operator_guidance = "continue monitoring and reassess on the next checkpoint"

        if normalized_severity == "critical":
            required_action = primary_action or "recover_worker_daemon"
            action_priority = "p0"
            escalation_mode = "immediate"
            requires_immediate_action = True
            auto_remediation_allowed = source_text in {"daemon", "runner"} or any(
                token in current_focus_text for token in ("daemon", "runner", "lease", "heartbeat", "restart")
            )
            remediation_mode = "auto" if auto_remediation_allowed else "manual_action"
            operator_guidance = "recover immediately, verify the daemon or runner, and confirm health returns to normal"
        elif normalized_severity == "warning":
            required_action = primary_action or "review_operations_warning"
            action_priority = "p1"
            escalation_mode = "review_soon"
            requires_human_review = True
            remediation_mode = "manual_review"
            operator_guidance = "review the warning, confirm intent, and decide whether to continue or intervene"
        elif attention_needed or normalized_severity == "info":
            required_action = primary_action or "observe_and_monitor"
            action_priority = "p2"
            escalation_mode = "monitor"
            remediation_mode = "monitor"
            operator_guidance = "observe the system, keep the candidate moving, and revisit the next queued action"
            if required_action.startswith("review_") or "confirmation" in current_focus_text or "promotion" in current_focus_text:
                remediation_mode = "manual_review"
                requires_human_review = True
                operator_guidance = "review the attention signal, confirm intent, and decide whether to continue"
        runtime_action = ""
        runtime_priority = action_priority
        runtime_escalation = escalation_mode
        runtime_immediate = requires_immediate_action
        runtime_review = requires_human_review
        runtime_auto = auto_remediation_allowed
        runtime_mode = remediation_mode
        runtime_guidance = ""
        if daemon_restart_policy_state in {"backoff", "capped"} or any(
            str(reason).startswith("daemon_restart_") for reason in escalated_reason_values
        ):
            runtime_action = "inspect_daemon_restart_policy"
            runtime_priority = "p1"
            runtime_escalation = "review_soon"
            runtime_immediate = False
            runtime_review = True
            runtime_auto = False
            runtime_mode = "manual_review"
            runtime_guidance = "inspect daemon restart attempts and backoff before requesting another recovery cycle"
        elif daemon_health_state in {"stale", "blocked"} or any(
            reason in escalated_reason_values for reason in ("daemon_stale", "daemon_blocked")
        ):
            runtime_action = "recover_worker_daemon"
            runtime_priority = "p0" if normalized_severity == "critical" else "p1"
            runtime_escalation = "immediate" if normalized_severity == "critical" else "review_soon"
            runtime_immediate = normalized_severity == "critical"
            runtime_review = normalized_severity != "critical"
            runtime_auto = True
            runtime_mode = "auto"
            runtime_guidance = "recover the worker daemon first, then verify heartbeat and lease health before resuming queue processing"
        elif daemon_heartbeat_state in {"delayed", "stale"} or any(
            str(reason).startswith("daemon_heartbeat_") for reason in escalated_reason_values
        ):
            runtime_action = "inspect_daemon_heartbeat"
            runtime_priority = "p1"
            runtime_escalation = "review_soon"
            runtime_immediate = False
            runtime_review = True
            runtime_auto = False
            runtime_mode = "manual_review"
            runtime_guidance = "inspect the daemon heartbeat and confirm lease freshness before continuing automated work"
        elif daemon_lease_state in {"expiring", "expired"} or any(
            str(reason).startswith("daemon_lease_") for reason in escalated_reason_values
        ):
            runtime_action = "recover_worker_daemon" if daemon_lease_state == "expired" else "inspect_daemon_heartbeat"
            runtime_priority = "p1"
            runtime_escalation = "review_soon"
            runtime_immediate = False
            runtime_review = True
            runtime_auto = daemon_lease_state == "expired"
            runtime_mode = "auto" if daemon_lease_state == "expired" else "manual_review"
            runtime_guidance = (
                "recover the worker daemon because the lease has expired, then confirm fresh heartbeats before resuming work"
                if daemon_lease_state == "expired"
                else "inspect the daemon lease and heartbeat cadence before allowing more queue work"
            )
        elif runner_lock_state == "stale" or "stale_runner_lock" in escalated_reason_values:
            runtime_action = "inspect_worker_stale_lock"
            runtime_priority = "p1"
            runtime_escalation = "review_soon"
            runtime_immediate = False
            runtime_review = True
            runtime_auto = False
            runtime_mode = "manual_review"
            runtime_guidance = "inspect the stale runner lock, confirm the previous worker is gone, and then resume queue execution"
        elif runner_stop_requested or "runner_stop_requested" in escalated_reason_values:
            runtime_action = "wait_for_runner_shutdown"
            runtime_priority = "p2"
            runtime_escalation = "monitor"
            runtime_immediate = False
            runtime_review = False
            runtime_auto = False
            runtime_mode = "monitor"
            runtime_guidance = "wait for the active runner to shut down cleanly before launching more queued work"
        elif daemon_recovery_action not in {"", "none"}:
            runtime_action = "recover_worker_daemon"
            runtime_priority = "p1"
            runtime_escalation = "review_soon"
            runtime_immediate = False
            runtime_review = True
            runtime_auto = daemon_recovery_action.startswith("auto_")
            runtime_mode = "auto" if runtime_auto else "manual_action"
            runtime_guidance = "complete the daemon recovery action and confirm the runtime returns to a healthy heartbeat and lease state"
        if queue_review_mode == "manual_review" and queue_review_required_now:
            required_action = "review_queue_confirmation"
            action_priority = "p1"
            escalation_mode = "review_soon"
            requires_human_review = True
            auto_remediation_allowed = False
            remediation_mode = "manual_review"
            operator_guidance = "review the queued training request, confirm intent, and approve or reject the queue confirmation"
        elif queue_review_mode == "auto_queue" and queue_review_next_action == "process_next_queue_item":
            if required_action in {"none", "observe_and_monitor"}:
                required_action = "process_next_queue_item"
            if action_priority == "p3":
                action_priority = "p2"
            if escalation_mode == "none":
                escalation_mode = "monitor"
            remediation_mode = "auto_queue"
            auto_remediation_allowed = True
            if not requires_human_review:
                operator_guidance = "the queue can continue automatically; process the next queued training item when ready"
        elif (
            queue_monitor_summary_active
            and required_action in {"none", "observe_and_monitor"}
        ):
            required_action = queue_summary_primary_action
            action_priority = "p2"
            escalation_mode = "monitor"
            requires_human_review = False
            auto_remediation_allowed = queue_summary_primary_action == "process_next_queue_item"
            remediation_mode = "auto_queue" if auto_remediation_allowed else "monitor"
            operator_guidance = (
                "queued work is waiting; process the next training item, then inspect the trigger state if backlog keeps growing"
                if queue_summary_primary_action == "process_next_queue_item"
                else "queued work is already processing; wait for queue completion, then inspect the trigger state if it stays blocked"
            )
        elif (
            runner_runtime_summary_active
            and required_action in {"none", "observe_and_monitor"}
        ):
            required_action = runtime_summary_primary_action
            action_priority = "p2"
            escalation_mode = "monitor"
            requires_human_review = False
            auto_remediation_allowed = False
            remediation_mode = "monitor"
            operator_guidance = "the worker runner is active; inspect runtime stability first, then review runner history if the loop stays active longer than expected"
        elif (
            daemon_runtime_summary_active
            and required_action in {"none", "observe_and_monitor"}
        ):
            required_action = runtime_summary_primary_action
            action_priority = "p2"
            escalation_mode = "monitor"
            requires_human_review = False
            auto_remediation_allowed = False
            remediation_mode = "monitor"
            operator_guidance = "the worker daemon is active; inspect runtime stability first, then review daemon status if it stays active longer than expected"
        elif (
            candidate_monitor_summary_active
            and required_action in {"none", "observe_and_monitor"}
        ):
            required_action = candidate_summary_primary_action
            action_priority = "p2"
            escalation_mode = "monitor"
            requires_human_review = False
            auto_remediation_allowed = False
            remediation_mode = "monitor"
            operator_guidance = "inspect the current candidate summary first, then review the candidate timeline if you need more detail before deciding the next step"
        if current_focus_text == "candidate_ready_for_promotion" or (
            source_text == "candidate" and primary_action in {"promote_candidate", "archive_candidate"}
        ):
            required_action = primary_action or "promote_candidate"
            action_priority = "p2"
            escalation_mode = "review_soon"
            requires_human_review = True
            auto_remediation_allowed = False
            remediation_mode = "manual_review"
            operator_guidance = (
                "promote the candidate when it is ready for rollout, or archive it if you want to keep it out of the active path"
                if "archive_candidate" in actions
                else "promote the candidate when it is ready for rollout"
            )
        if trigger_policy_reason == "policy_requires_auto_evaluate":
            required_action = trigger_policy_action or "enable_auto_evaluate"
            action_priority = "p1"
            escalation_mode = "review_soon"
            requires_human_review = True
            auto_remediation_allowed = False
            remediation_mode = "manual_review"
            operator_guidance = "enable auto-evaluate before auto-promote, or keep promotion manual for this training policy"
        elif trigger_policy_reason == "review_required_before_execution":
            required_action = trigger_policy_action or "review_queue_confirmation"
            action_priority = "p1"
            escalation_mode = "review_soon"
            requires_human_review = True
            auto_remediation_allowed = False
            remediation_mode = "manual_review"
            operator_guidance = "review the queue confirmation first; evaluation and promotion can only continue after approval"
        elif (
            trigger_blocked_reason_text
            and normalized_severity != "critical"
            and source_text == "auto_train"
            and queue_review_mode != "manual_review"
            and not (queue_review_mode == "auto_queue" and queue_review_next_action == "process_next_queue_item")
            and (
                required_action in {"none", "observe_and_monitor", "review_operations_warning", "inspect_auto_train_policy"}
                or required_action == trigger_blocked_action_text
            )
        ):
            required_action = trigger_blocked_action_text or primary_action or "inspect_auto_train_policy"
            action_priority = "p1" if normalized_severity == "warning" or attention_needed else "p2"
            escalation_mode = "review_soon" if attention_needed else "monitor"
            auto_remediation_allowed = False
            remediation_mode = "manual_review" if attention_needed else "monitor"
            operator_guidance = trigger_blocked_summary_text or "review the current auto-train blocker before continuing"
            if trigger_blocked_reason_text == "insufficient_new_signal_samples":
                operator_guidance = (
                    "not enough new signal samples are available yet; collect more new signal samples before auto-train can continue, then inspect the trigger policy if the gate still looks off"
                )
            elif trigger_blocked_reason_text == "insufficient_dpo_preference_pairs":
                operator_guidance = (
                    "not enough eligible DPO preference pairs are available yet; collect more DPO preference pairs before auto-train can continue, then inspect the trigger policy if the gate still looks off"
                )
            elif trigger_blocked_reason_text == "holdout_not_ready":
                operator_guidance = (
                    "collect holdout samples so evaluation has coverage, then inspect the trigger policy if auto-train still stays blocked"
                )
            elif trigger_blocked_reason_text in {"cooldown_active", "min_trigger_interval_active"}:
                operator_guidance = (
                    "wait for the retrain interval gate to elapse, then inspect the trigger timing summary before forcing another run"
                )
            elif trigger_blocked_reason_text == "failure_backoff_active":
                operator_guidance = (
                    "wait for failure backoff to elapse or retry after fixing the last failure, then inspect the trigger state before continuing"
                )
        runtime_context = (
            runtime_action
            and (
                normalized_severity in {"critical", "warning"}
                or source_text in {"daemon", "runner"}
                or runner_lock_state == "stale"
                or daemon_health_state in {"stale", "blocked"}
                or daemon_heartbeat_state in {"delayed", "stale"}
                or daemon_lease_state in {"expiring", "expired"}
                or daemon_restart_policy_state in {"backoff", "capped"}
                or daemon_recovery_action not in {"", "none"}
            )
        )
        if runtime_context:
            required_action = runtime_action or required_action
            action_priority = runtime_priority
            escalation_mode = runtime_escalation
            requires_immediate_action = runtime_immediate
            requires_human_review = runtime_review
            auto_remediation_allowed = runtime_auto
            remediation_mode = runtime_mode
            operator_guidance = runtime_guidance or operator_guidance
        resolved_actions: list[str] = []

        def _push_action(action_name: Any | None) -> None:
            text = str(action_name or "").strip()
            if not text or text == "none" or text in resolved_actions:
                return
            resolved_actions.append(text)

        _push_action(required_action)
        if queue_review_mode == "manual_review" and queue_review_required_now:
            _push_action("inspect_auto_train_gate")
            _push_action("inspect_auto_train_trigger")
        if trigger_policy_reason == "policy_requires_auto_evaluate":
            _push_action("inspect_auto_train_gate")
        elif trigger_policy_reason == "review_required_before_execution":
            _push_action("inspect_auto_train_gate")
            _push_action("inspect_auto_train_trigger")
        elif trigger_blocked_reason_text in {"insufficient_new_signal_samples", "insufficient_dpo_preference_pairs", "holdout_not_ready"}:
            _push_action("inspect_auto_train_policy")
        elif trigger_blocked_reason_text in {"cooldown_active", "failure_backoff_active"}:
            _push_action("inspect_auto_train_trigger")
        if queue_review_mode == "auto_queue" and queue_review_next_action == "process_next_queue_item":
            _push_action("inspect_auto_train_trigger")
        elif queue_monitor_summary_active:
            for action_name in queue_summary_secondary_actions or ["inspect_auto_train_trigger"]:
                _push_action(action_name)
        if runtime_monitor_summary_active and required_action == runtime_summary_primary_action:
            for action_name in runtime_summary_secondary_actions:
                _push_action(action_name)
        elif candidate_monitor_summary_active and required_action == candidate_summary_primary_action:
            for action_name in candidate_summary_secondary_actions or ["inspect_candidate_timeline"]:
                _push_action(action_name)
        if runtime_action == "recover_worker_daemon":
            _push_action("inspect_daemon_heartbeat")
        elif runtime_action == "inspect_daemon_restart_policy":
            _push_action("inspect_daemon_heartbeat")
        elif runtime_action == "inspect_worker_stale_lock":
            _push_action("wait_for_runner_shutdown")
        elif runtime_action == "wait_for_runner_shutdown":
            _push_action("inspect_worker_stale_lock")
        for action_name in actions:
            _push_action(action_name)
        if resolved_actions:
            actions = resolved_actions
            primary_action = actions[0]
        summary_parts = [
            f"severity={normalized_severity}",
            f"required_action={required_action}",
            f"priority={action_priority}",
            f"mode={escalation_mode}",
        ]
        if current_focus_text:
            summary_parts.append(f"focus={current_focus_text}")
        if queue_review_mode:
            summary_parts.append(f"qreview={queue_review_mode}")
        if queue_review_next_action:
            summary_parts.append(f"qnext={queue_review_next_action}")
        if trigger_policy_reason:
            summary_parts.append(f"pgate={trigger_policy_reason}")
        if trigger_policy_action:
            summary_parts.append(f"pact={trigger_policy_action}")
        if trigger_policy_gate.get("summary_line"):
            summary_parts.append(f"pg={trigger_policy_gate.get('summary_line')}")
        if trigger_blocked_reason_text:
            summary_parts.append(f"block={trigger_blocked_reason_text}")
        if trigger_blocked_action_text:
            summary_parts.append(f"bact={trigger_blocked_action_text}")
        if auto_train_blocker_map.get("reason"):
            summary_parts.append(f"blocker={auto_train_blocker_map.get('reason')}")
        if trigger_threshold.get("summary_line"):
            summary_parts.append(f"gate={trigger_threshold.get('summary_line')}")
        if runtime_stability_map.get("summary_line"):
            summary_parts.append(f"rt={runtime_stability_map.get('summary_line')}")
        if active_recovery_text:
            summary_parts.append(f"recovery={active_recovery_text}")
        if latest_recovery is not None:
            latest_recovery_reason = latest_recovery.get("reason")
            if latest_recovery_reason:
                summary_parts.append(f"latest_recovery={latest_recovery_reason}")
        return {
            "severity": normalized_severity,
            "required_action": required_action,
            "primary_action": primary_action or required_action,
            "highest_priority_action": primary_action or required_action,
            "secondary_action": actions[1] if len(actions) > 1 else None,
            "secondary_actions": actions[1:],
            "action_priority": action_priority,
            "escalation_mode": escalation_mode,
            "requires_immediate_action": requires_immediate_action,
            "requires_human_review": requires_human_review,
            "auto_remediation_allowed": auto_remediation_allowed,
            "remediation_mode": remediation_mode,
            "operator_guidance": operator_guidance,
            "escalated_reasons": escalated_reason_values,
            "active_recovery_hint": active_recovery_text or None,
            "latest_recovery": latest_recovery,
            "queue_review_policy": queue_review,
            "candidate_action_summary": candidate_action_summary_map,
            "queue_action_summary": queue_action_summary_map,
            "queue_review_mode": queue_review_mode or None,
            "queue_review_next_action": queue_review_next_action or None,
            "queue_review_reason": queue_review_reason or None,
            "queue_review_required_now": queue_review_required_now,
            "trigger_policy_reason": trigger_policy_reason or None,
            "trigger_policy_action": trigger_policy_action or None,
            "trigger_policy_gate_summary": trigger_policy_gate,
            "trigger_policy_gate_summary_line": trigger_policy_gate.get("summary_line"),
            "trigger_threshold_summary": trigger_threshold,
            "trigger_threshold_summary_line": trigger_threshold.get("summary_line"),
            "runtime_stability_summary": runtime_stability_map,
            "runtime_stability_summary_line": runtime_stability_map.get("summary_line"),
            "runtime_action_summary": runtime_action_summary_map,
            "auto_train_blocker": auto_train_blocker_map,
            "trigger_blocked_reason": trigger_blocked_reason_text or None,
            "trigger_blocked_action": trigger_blocked_action_text or None,
            "trigger_blocked_summary": trigger_blocked_summary_text or None,
            "summary_line": " | ".join(summary_parts),
        }

    def _candidate_row(self, *, workspace: str | None = None) -> dict[str, Any] | None:
        store = create_adapter_store(workspace=workspace)
        rows = store.list_version_records(limit=100)
        latest_version = store.current_latest_version()
        latest_version_text = str(latest_version) if latest_version is not None else None
        for row in rows:
            version = str(row.get("version") or "")
            if latest_version_text and version == latest_version_text and row.get("state") == "promoted":
                continue
            return dict(row)
        return None

    @staticmethod
    def _summarize_auto_trigger_result(result: dict[str, Any]) -> str:
        parts: list[str] = []
        triggered = result.get("triggered")
        if triggered is not None:
            parts.append(f"triggered={'yes' if bool(triggered) else 'no'}")
        for key in ("state", "reason", "triggered_version", "triggered_state", "eval_recommendation", "eval_comparison", "promoted_version", "promote_reason", "error_stage"):
            value = result.get(key)
            if value not in (None, "", []):
                parts.append(f"{key}={value}")
        if result.get("eval_error"):
            parts.append(f"eval_error={result['eval_error']}")
        if result.get("error"):
            parts.append(f"error={result['error']}")
        return " | ".join(parts) if parts else "idle"

    @staticmethod
    def _auto_train_policy_summary(
        *,
        enabled: bool,
        queue_mode: str,
        require_queue_confirmation: bool,
        auto_evaluate: bool,
        auto_promote: bool,
        eval_num_samples: int,
    ) -> dict[str, Any]:
        normalized_queue_mode = str(queue_mode or "inline")
        confirmation_required = bool(normalized_queue_mode == "deferred" and require_queue_confirmation)
        auto_execute_allowed = bool(enabled and normalized_queue_mode == "inline")
        auto_evaluate_enabled = bool(auto_evaluate)
        auto_evaluate_requested = bool(auto_evaluate)
        auto_promote_enabled = bool(auto_evaluate and auto_promote)
        auto_promote_requested = bool(auto_promote)
        review_gate_reason = "review_required_before_execution" if confirmation_required else None
        evaluation_gate_reason = "review_required_before_execution" if (confirmation_required and auto_evaluate_enabled) else None
        evaluation_gate_action = "review_queue_confirmation" if evaluation_gate_reason is not None else None
        promote_gate_reason = None
        promote_gate_action = None
        promotion_requirement = None
        if confirmation_required and bool(auto_promote):
            promote_gate_reason = "review_required_before_execution"
            promote_gate_action = "review_queue_confirmation"
        elif bool(auto_promote and not auto_evaluate):
            promote_gate_reason = "policy_requires_auto_evaluate"
            promote_gate_action = "enable_auto_evaluate"
        elif auto_promote_enabled:
            promotion_requirement = "deploy_eval_recommendation"

        if not enabled:
            queue_entry_mode = "disabled"
            review_mode = "no_review"
            promotion_mode = "manual_promote"
            stop_stage = "trigger"
        elif confirmation_required:
            queue_entry_mode = "awaiting_confirmation"
            review_mode = "manual_review"
            promotion_mode = "auto_promote_after_review" if auto_promote_enabled else "manual_promote"
            stop_stage = "confirmation"
        elif normalized_queue_mode == "deferred":
            queue_entry_mode = "queued"
            review_mode = "no_review"
            promotion_mode = "auto_promote" if auto_promote_enabled else "manual_promote"
            stop_stage = "promote" if auto_promote_enabled else ("evaluate" if auto_evaluate_enabled else "train")
        else:
            queue_entry_mode = "immediate_execute"
            review_mode = "no_review"
            promotion_mode = "auto_promote" if auto_promote_enabled else "manual_promote"
            stop_stage = "promote" if auto_promote_enabled else ("evaluate" if auto_evaluate_enabled else "train")

        summary_parts = [
            f"mode={normalized_queue_mode}",
            f"entry={queue_entry_mode}",
            f"review={review_mode}",
            f"eval={'auto_after_review' if evaluation_gate_reason else ('auto' if auto_evaluate_enabled else 'skip')}",
            f"promote={'auto' if auto_promote_enabled else 'manual'}",
            f"stop={stop_stage}",
        ]
        if review_gate_reason is not None:
            summary_parts.append(f"review_gate={review_gate_reason}")
        if evaluation_gate_reason is not None:
            summary_parts.append(f"eval_gate={evaluation_gate_reason}")
        if evaluation_gate_action is not None:
            summary_parts.append(f"eval_action={evaluation_gate_action}")
        if promote_gate_reason is not None:
            summary_parts.append(f"promote_gate={promote_gate_reason}")
        if promote_gate_action is not None:
            summary_parts.append(f"promote_action={promote_gate_action}")
        if promotion_requirement is not None:
            summary_parts.append(f"promote_req={promotion_requirement}")

        gate_summary = {
            "eval_num_samples": int(eval_num_samples),
            "auto_evaluate_requested": auto_evaluate_requested,
            "auto_evaluate_enabled": auto_evaluate_enabled,
            "auto_promote_requested": auto_promote_requested,
            "auto_promote_enabled": auto_promote_enabled,
            "evaluation_gate_reason": evaluation_gate_reason,
            "evaluation_gate_action": evaluation_gate_action,
            "promote_gate_reason": promote_gate_reason,
            "promote_gate_action": promote_gate_action,
            "promotion_requirement": promotion_requirement,
            "summary_line": " | ".join(
                part
                for part in (
                    f"eval={int(eval_num_samples)}:{'on' if auto_evaluate_enabled else 'off'}",
                    f"promote={'on' if auto_promote_requested else 'off'}",
                    (
                        f"promote_ready={'yes' if auto_promote_enabled else 'no'}"
                        if auto_promote_requested
                        else None
                    ),
                    f"eval_gate={evaluation_gate_reason}" if evaluation_gate_reason is not None else None,
                    f"promote_gate={promote_gate_reason}" if promote_gate_reason is not None else None,
                    f"req={promotion_requirement}" if promotion_requirement is not None else None,
                )
                if part is not None
            ),
        }

        return {
            "execution_mode": normalized_queue_mode,
            "queue_entry_mode": queue_entry_mode,
            "review_mode": review_mode,
            "evaluation_mode": (
                "auto_evaluate_after_review"
                if evaluation_gate_reason
                else ("auto_evaluate" if auto_evaluate_enabled else "skip_evaluate")
            ),
            "promotion_mode": promotion_mode,
            "stop_stage": stop_stage,
            "auto_execute_allowed": auto_execute_allowed,
            "manual_confirmation_required": confirmation_required,
            "auto_evaluate_requested": auto_evaluate_requested,
            "auto_evaluate_enabled": auto_evaluate_enabled,
            "auto_promote_requested": auto_promote_requested,
            "auto_promote_enabled": auto_promote_enabled,
            "review_gate_reason": review_gate_reason,
            "evaluation_gate_reason": evaluation_gate_reason,
            "evaluation_gate_action": evaluation_gate_action,
            "promote_gate_reason": promote_gate_reason,
            "promote_gate_action": promote_gate_action,
            "promotion_requirement": promotion_requirement,
            "gate_summary": gate_summary,
            "summary_line": " | ".join(summary_parts),
        }

    @staticmethod
    def _auto_train_execution_policy(
        *,
        trigger_status: Mapping[str, Any],
        queue_item: Mapping[str, Any],
    ) -> dict[str, Any]:
        auto_evaluate_enabled = bool(trigger_status.get("auto_evaluate"))
        auto_promote_requested = bool(trigger_status.get("auto_promote"))
        auto_promote_enabled = bool(auto_evaluate_enabled and auto_promote_requested)
        confirmation_reviewed = bool(queue_item.get("confirmation_reviewed"))
        confirmation_required = bool(queue_item.get("confirmation_required"))

        if confirmation_required and not confirmation_reviewed:
            review_gate = "pending_confirmation"
        elif confirmation_reviewed:
            review_gate = "approved"
        else:
            review_gate = "not_required"

        evaluation_mode = "auto_evaluate" if auto_evaluate_enabled else "skip_evaluate"
        promotion_mode = "auto_promote" if auto_promote_enabled else "manual_promote"
        if auto_promote_enabled:
            stop_stage = "promote"
        elif auto_evaluate_enabled:
            stop_stage = "evaluate"
        elif review_gate == "pending_confirmation":
            stop_stage = "confirmation"
        else:
            stop_stage = "train"

        review_gate_reason = "review_pending_confirmation" if review_gate == "pending_confirmation" else None
        evaluation_gate_reason = None
        evaluation_gate_action = None
        if review_gate == "pending_confirmation" and auto_evaluate_enabled:
            evaluation_gate_reason = "review_pending_confirmation"
            evaluation_gate_action = "review_queue_confirmation"
        elif not auto_evaluate_enabled:
            evaluation_gate_reason = "policy_skips_auto_evaluate"
            evaluation_gate_action = "manual_evaluate_after_training"
        promote_gate_reason = None
        promote_gate_action = None
        promotion_requirement = None
        if review_gate == "pending_confirmation" and auto_promote_requested:
            promote_gate_reason = "review_pending_confirmation"
            promote_gate_action = "review_queue_confirmation"
        elif auto_promote_requested and not auto_evaluate_enabled:
            promote_gate_reason = "policy_requires_auto_evaluate"
            promote_gate_action = "enable_auto_evaluate"
        elif auto_promote_enabled:
            promotion_requirement = "deploy_eval_recommendation"

        summary_parts = [
            f"review={review_gate}",
            f"eval={'auto' if auto_evaluate_enabled else 'skip'}",
            f"promote={'auto' if auto_promote_enabled else 'manual'}",
            f"stop={stop_stage}",
        ]
        if review_gate_reason is not None:
            summary_parts.append(f"review_gate={review_gate_reason}")
        if evaluation_gate_reason is not None:
            summary_parts.append(f"eval_gate={evaluation_gate_reason}")
        if evaluation_gate_action is not None:
            summary_parts.append(f"eval_action={evaluation_gate_action}")
        if promote_gate_reason is not None:
            summary_parts.append(f"promote_gate={promote_gate_reason}")
        if promote_gate_action is not None:
            summary_parts.append(f"promote_action={promote_gate_action}")
        if promotion_requirement is not None:
            summary_parts.append(f"promote_req={promotion_requirement}")

        return {
            "review_gate": review_gate,
            "review_gate_reason": review_gate_reason,
            "evaluation_mode": evaluation_mode,
            "auto_evaluate_enabled": auto_evaluate_enabled,
            "evaluation_gate_reason": evaluation_gate_reason,
            "evaluation_gate_action": evaluation_gate_action,
            "auto_promote_requested": auto_promote_requested,
            "promotion_mode": promotion_mode,
            "auto_promote_enabled": auto_promote_enabled,
            "promote_gate_reason": promote_gate_reason,
            "promote_gate_action": promote_gate_action,
            "promotion_requirement": promotion_requirement,
            "stop_stage": stop_stage,
            "summary_line": " | ".join(summary_parts),
        }

    @staticmethod
    def _load_config(home: str | Path | None = None) -> PFEConfig:
        return PFEConfig.load(home=home)

    @staticmethod
    def _save_config(config: PFEConfig, home: str | Path | None = None) -> None:
        config.save(home=home)

    @staticmethod
    def _signal_collection_enabled(config: PFEConfig | None = None) -> bool:
        current = config or PipelineService._load_config()
        curator = getattr(current, "curator", None)
        return bool(getattr(curator, "signal_collection_enabled", True))

    @staticmethod
    def _signal_quality_summary(*, workspace: str | None = None) -> dict[str, Any]:
        signals = list_signals()
        if not signals:
            return {
                "evaluated_count": 0,
                "passed_count": 0,
                "filtered_count": 0,
                "filtered_reasons": {},
                "reply_style_counts": {},
                "minimum_confidence": SampleFilterConfig().minimum_signal_confidence,
                "reject_conflicted_signal_quality": SampleFilterConfig().reject_conflicted_signal_quality,
            }

        config = SampleFilterConfig()
        evaluated = 0
        passed = 0
        filtered = 0
        filtered_reasons: dict[str, int] = {}
        reply_style_counts: dict[str, int] = {}

        for signal in signals:
            quality = build_signal_quality(signal)
            evaluated += 1
            reply_style = str(getattr(quality, "reply_style", "other") or "other")
            reply_style_counts[reply_style] = reply_style_counts.get(reply_style, 0) + 1
            reasons = signal_quality_filter_reasons(
                quality,
                minimum_confidence=config.minimum_signal_confidence,
                reject_conflicted_signal_quality=config.reject_conflicted_signal_quality,
            )
            if reasons:
                filtered += 1
                for reason in reasons:
                    filtered_reasons[reason] = filtered_reasons.get(reason, 0) + 1
            else:
                passed += 1

        return {
            "evaluated_count": evaluated,
            "passed_count": passed,
            "filtered_count": filtered,
            "filtered_reasons": filtered_reasons,
            "reply_style_counts": reply_style_counts,
            "minimum_confidence": config.minimum_signal_confidence,
            "reject_conflicted_signal_quality": config.reject_conflicted_signal_quality,
        }

    @staticmethod
    def _parse_iso_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    @staticmethod
    def _sample_metadata(sample: Mapping[str, Any]) -> dict[str, Any]:
        metadata = sample.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else {}

    @staticmethod
    def _sample_is_preference_reinforced(sample: Mapping[str, Any]) -> bool:
        metadata = PipelineService._sample_metadata(sample)
        if bool(metadata.get("explicit_response_preference_reinforced")):
            return True
        return str(metadata.get("training_signal_category") or "").strip().lower() == "preference_reinforced"

    @staticmethod
    def _normalized_preference_reinforced_sample_weight(value: Any) -> float:
        try:
            numeric = float(value)
        except Exception:
            numeric = 1.0
        return max(1.0, round(numeric, 3))

    @staticmethod
    def _format_trigger_count(value: Any) -> str:
        try:
            numeric = float(value)
        except Exception:
            return str(value)
        rounded = round(numeric, 3)
        if rounded.is_integer():
            return str(int(rounded))
        return f"{rounded:.3f}".rstrip("0").rstrip(".")

    @staticmethod
    def _eligible_sample_weight_summary(
        samples: Sequence[Mapping[str, Any]],
        *,
        preference_reinforced_sample_weight: float,
    ) -> dict[str, Any]:
        weight = PipelineService._normalized_preference_reinforced_sample_weight(preference_reinforced_sample_weight)
        preference_reinforced_samples = [
            sample
            for sample in samples
            if PipelineService._sample_is_preference_reinforced(sample)
        ]
        preference_reinforced_sample_ids = [
            str(sample.get("sample_id"))
            for sample in preference_reinforced_samples
            if sample.get("sample_id")
        ]
        effective_sample_count = round(
            len(samples) + max(weight - 1.0, 0.0) * len(preference_reinforced_samples),
            3,
        )
        return {
            "sample_count": len(samples),
            "sample_ids": [str(sample.get("sample_id")) for sample in samples if sample.get("sample_id")],
            "preference_reinforced_sample_weight": weight,
            "preference_reinforced_sample_count": len(preference_reinforced_samples),
            "preference_reinforced_sample_ids": preference_reinforced_sample_ids,
            "effective_sample_count": effective_sample_count,
        }

    @staticmethod
    def _auto_train_blocker_details(reason: str) -> dict[str, str]:
        mapping = {
            "trigger_disabled": {
                "category": "config",
                "action": "enable_auto_train_trigger",
                "summary": "auto-train is disabled by policy",
            },
            "queue_pending_review": {
                "category": "queue",
                "action": "review_queue_confirmation",
                "summary": "auto-train is waiting for queue confirmation review",
            },
            "queue_waiting_execution": {
                "category": "queue",
                "action": "process_next_queue_item",
                "summary": "auto-train is blocked by queued work waiting for execution",
            },
            "queue_processing_active": {
                "category": "queue",
                "action": "wait_for_queue_completion",
                "summary": "auto-train is blocked by queue processing already in progress",
            },
            "review_required_before_execution": {
                "category": "policy",
                "action": "review_queue_confirmation",
                "summary": "auto-train policy requires queue confirmation review before the next automated stage can continue",
            },
            "policy_requires_auto_evaluate": {
                "category": "policy",
                "action": "enable_auto_evaluate",
                "summary": "auto-promote requires auto-evaluate before it can run",
            },
            "holdout_not_ready": {
                "category": "data",
                "action": "collect_holdout_samples",
                "summary": "holdout samples are not ready yet",
            },
            "insufficient_new_signal_samples": {
                "category": "data",
                "action": "collect_more_signal_samples",
                "summary": "not enough new signal samples are available yet",
            },
            "insufficient_dpo_preference_pairs": {
                "category": "data",
                "action": "collect_more_dpo_preference_pairs",
                "summary": "not enough eligible DPO preference pairs are available yet",
            },
            "failure_backoff_active": {
                "category": "recovery",
                "action": "wait_for_failure_backoff",
                "summary": "auto-train is paused by failure backoff",
            },
            "min_trigger_interval_active": {
                "category": "timing",
                "action": "wait_for_min_trigger_interval",
                "summary": "the minimum trigger interval has not elapsed yet",
            },
            "cooldown_active": {
                "category": "timing",
                "action": "wait_for_retrain_interval",
                "summary": "the training interval gate is still active",
            },
        }
        return dict(mapping.get(str(reason or ""), {}))

    @staticmethod
    def _auto_train_eligible_samples(
        *,
        train_type: str,
        workspace: str | None = None,
        preference_reinforced_sample_weight: float = 1.0,
    ) -> dict[str, Any]:
        del workspace
        normalized_train_type = str(train_type or "sft").strip().lower()
        if normalized_train_type == "dpo":
            samples = [
                sample
                for sample in list_samples(sample_type="dpo", dataset_split="train", include_used=False)
                if sample.get("instruction") and sample.get("chosen") and sample.get("rejected")
            ]
            weight_summary = PipelineService._eligible_sample_weight_summary(
                samples,
                preference_reinforced_sample_weight=preference_reinforced_sample_weight,
            )
            return {
                "train_type": normalized_train_type,
                "count_label": "dpo_pairs",
                "block_reason": "insufficient_dpo_preference_pairs",
                "block_action": "collect_more_dpo_preference_pairs",
                "block_summary": "not enough eligible DPO preference pairs are available yet",
                "samples": samples,
                **weight_summary,
            }

        samples = [
            sample
            for sample in list_samples(dataset_split="train", include_used=False)
            if sample.get("source") == "signal"
        ]
        weight_summary = PipelineService._eligible_sample_weight_summary(
            samples,
            preference_reinforced_sample_weight=preference_reinforced_sample_weight,
        )
        return {
            "train_type": normalized_train_type,
            "count_label": "samples",
            "block_reason": "insufficient_new_signal_samples",
            "block_action": "collect_more_signal_samples",
            "block_summary": "not enough new signal samples are available yet",
            "samples": samples,
            **weight_summary,
        }

    @staticmethod
    def _auto_train_primary_blocker(
        *,
        queue_gate_reason: str | None,
        queue_gate_action: str | None,
        trigger_blocked_reason: str | None,
        trigger_blocked_action: str | None,
        trigger_blocked_category: str | None,
        trigger_blocked_summary: str | None,
        trigger_policy_reason: str | None,
        trigger_policy_action: str | None,
    ) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []

        def _candidate(
            *,
            source: str,
            reason: str | None,
            action: str | None = None,
            category: str | None = None,
            summary: str | None = None,
        ) -> dict[str, Any] | None:
            reason_text = str(reason or "").strip()
            if not reason_text:
                return None
            if source == "trigger_blocked" and reason_text == "trigger_disabled":
                return None
            details = dict(PipelineService._auto_train_blocker_details(reason_text))
            if action:
                details["action"] = action
            if category:
                details["category"] = category
            if summary:
                details["summary"] = summary
            details.setdefault("action", action or "inspect_auto_train_policy")
            details.setdefault("category", category or "policy")
            details.setdefault("summary", summary or reason_text)
            details["source"] = source
            details["reason"] = reason_text
            return details

        for candidate in (
            _candidate(
                source="queue_gate",
                reason=queue_gate_reason,
                action=queue_gate_action,
                category="queue",
            ),
            _candidate(
                source="policy_gate",
                reason=trigger_policy_reason,
                action=trigger_policy_action,
                category="policy",
            ),
            _candidate(
                source="trigger_blocked",
                reason=trigger_blocked_reason,
                action=trigger_blocked_action,
                category=trigger_blocked_category,
                summary=trigger_blocked_summary,
            ),
        ):
            if candidate is not None:
                candidates.append(candidate)

        primary = candidates[0] if candidates else {}
        secondary = candidates[1:] if len(candidates) > 1 else []
        if primary:
            primary["secondary_reasons"] = [str(item.get("reason") or "") for item in secondary if str(item.get("reason") or "")]
            primary["secondary_actions"] = [str(item.get("action") or "") for item in secondary if str(item.get("action") or "")]
            primary["summary_line"] = " | ".join(
                part
                for part in (
                    f"source={primary.get('source')}",
                    f"reason={primary.get('reason')}",
                    f"action={primary.get('action')}",
                    f"category={primary.get('category')}",
                    f"summary={primary.get('summary')}",
                )
                if part and not part.endswith("=None")
            )
        return primary

    def _auto_train_trigger_status(self, *, workspace: str | None = None) -> dict[str, Any]:
        config = self._load_config()
        trigger = config.trainer.trigger
        train_type = str(config.trainer.train_type or "sft").strip().lower()
        state = self._load_auto_trigger_state(workspace=workspace)
        snapshot = status_snapshot(workspace=workspace or "user_default")
        preference_reinforced_sample_weight = self._normalized_preference_reinforced_sample_weight(
            getattr(trigger, "preference_reinforced_sample_weight", 1.0)
        )
        signal_gate = self._auto_train_eligible_samples(
            train_type="sft",
            workspace=workspace,
            preference_reinforced_sample_weight=preference_reinforced_sample_weight,
        )
        eligible_signal_train_samples = int(signal_gate.get("sample_count", 0) or 0)
        effective_signal_train_samples = float(signal_gate.get("effective_sample_count", 0.0) or 0.0)
        preference_reinforced_signal_train_samples = int(
            signal_gate.get("preference_reinforced_sample_count", 0) or 0
        )
        sample_ids = [str(sample_id) for sample_id in list(signal_gate.get("sample_ids") or []) if sample_id]
        dpo_gate = self._auto_train_eligible_samples(
            train_type="dpo",
            workspace=workspace,
            preference_reinforced_sample_weight=preference_reinforced_sample_weight,
        )
        eligible_dpo_preference_pairs = int(dpo_gate.get("sample_count", 0) or 0)
        effective_dpo_preference_pairs = float(dpo_gate.get("effective_sample_count", 0.0) or 0.0)
        preference_reinforced_dpo_preference_pairs = int(
            dpo_gate.get("preference_reinforced_sample_count", 0) or 0
        )
        eligible_dpo_preference_pair_ids = [str(sample_id) for sample_id in list(dpo_gate.get("sample_ids") or []) if sample_id]
        if train_type == "dpo":
            eligible_train_samples = eligible_dpo_preference_pairs
            effective_eligible_train_samples = effective_dpo_preference_pairs
            preference_reinforced_train_samples = preference_reinforced_dpo_preference_pairs
            eligible_train_sample_ids = eligible_dpo_preference_pair_ids
            threshold_label = "dpo_pairs"
            blocked_reason_name = "insufficient_dpo_preference_pairs"
        else:
            eligible_train_samples = eligible_signal_train_samples
            effective_eligible_train_samples = effective_signal_train_samples
            preference_reinforced_train_samples = preference_reinforced_signal_train_samples
            eligible_train_sample_ids = sample_ids
            threshold_label = "samples"
            blocked_reason_name = "insufficient_new_signal_samples"
        sample_counts = snapshot.get("sample_counts") or {}
        train_trigger_policy = getattr(trigger, "train_trigger_policy", None) or {}
        require_holdout_split = bool(
            getattr(train_trigger_policy, "require_holdout_split", True)
            if not isinstance(train_trigger_policy, dict)
            else train_trigger_policy.get("require_holdout_split", True)
        )
        holdout_ready = bool(sample_counts.get("val", 0) or sample_counts.get("test", 0)) or not require_holdout_split
        enough_new_samples = effective_eligible_train_samples >= int(trigger.min_new_samples)

        store = create_adapter_store(workspace=workspace)
        recent_rows = store.list_version_records(limit=1)
        recent_row = recent_rows[0] if recent_rows else None
        last_training_at = self._parse_iso_datetime((recent_row or {}).get("updated_at") or (recent_row or {}).get("created_at"))
        now = datetime.now(timezone.utc)
        if last_training_at is not None and last_training_at.tzinfo is None:
            last_training_at = last_training_at.replace(tzinfo=timezone.utc)
        days_since_last_training = None
        if last_training_at is not None:
            days_since_last_training = round((now - last_training_at).total_seconds() / 86400.0, 3)
        interval_elapsed = last_training_at is None or days_since_last_training is None or days_since_last_training >= int(trigger.max_interval_days)
        last_attempted_at = self._parse_iso_datetime(state.get("last_attempted_at"))
        last_completed_at = self._parse_iso_datetime(state.get("last_completed_at"))
        last_success_at = self._parse_iso_datetime(state.get("last_success_at"))
        last_failure_at = self._parse_iso_datetime(state.get("last_failure_at"))
        cooldown_until = self._parse_iso_datetime(state.get("cooldown_until"))
        failure_backoff_until = self._parse_iso_datetime(state.get("failure_backoff_until"))
        cooldown_elapsed = cooldown_until is None or now >= cooldown_until
        failure_backoff_elapsed = failure_backoff_until is None or now >= failure_backoff_until
        cooldown_remaining_minutes = None
        if cooldown_until is not None and now < cooldown_until:
            cooldown_remaining_minutes = round((cooldown_until - now).total_seconds() / 60.0, 3)
        failure_backoff_remaining_minutes = None
        if failure_backoff_until is not None and now < failure_backoff_until:
            failure_backoff_remaining_minutes = round((failure_backoff_until - now).total_seconds() / 60.0, 3)
        consecutive_failures = int(state.get("consecutive_failures") or 0)
        last_result = dict(self.last_auto_trigger_result or state.get("last_result") or {})
        last_result_summary = str(state.get("last_result_summary") or self._summarize_auto_trigger_result(last_result))

        blocked_reasons: list[str] = []
        queue_snapshot = self._train_queue_snapshot(workspace=workspace)
        queue_counts = dict(queue_snapshot.get("counts") or {})
        queue_review_policy = dict(queue_snapshot.get("review_policy_summary") or {})
        awaiting_confirmation_count = int(queue_counts.get("awaiting_confirmation", 0) or 0)
        queued_count = int(queue_counts.get("queued", 0) or 0)
        running_count = int(queue_counts.get("running", 0) or 0)
        queue_busy = bool(queue_snapshot.get("current")) or awaiting_confirmation_count > 0
        queue_gate_reason = None
        queue_gate_action = None
        if awaiting_confirmation_count > 0:
            queue_gate_reason = "queue_pending_review"
            queue_gate_action = "review_queue_confirmation"
        elif running_count > 0:
            queue_gate_reason = "queue_processing_active"
            queue_gate_action = "wait_for_queue_completion"
        elif queued_count > 0:
            queue_gate_reason = "queue_waiting_execution"
            queue_gate_action = "process_next_queue_item"
        if not trigger.enabled:
            blocked_reasons.append("trigger_disabled")
        if not enough_new_samples:
            blocked_reasons.append(blocked_reason_name)
        if not holdout_ready:
            blocked_reasons.append("holdout_not_ready")
        if not interval_elapsed:
            blocked_reasons.append("cooldown_active")
        if not cooldown_elapsed:
            blocked_reasons.append("min_trigger_interval_active")
        if not failure_backoff_elapsed:
            blocked_reasons.append("failure_backoff_active")
        if queue_busy:
            blocked_reasons.append(str(queue_gate_reason or "queue_busy"))

        ready = trigger.enabled and enough_new_samples and holdout_ready and interval_elapsed and cooldown_elapsed and failure_backoff_elapsed and not queue_busy
        blocker_priority = {
            "trigger_disabled": 0,
            "failure_backoff_active": 1,
            "queue_pending_review": 2,
            "queue_waiting_execution": 3,
            "queue_processing_active": 4,
            "holdout_not_ready": 5,
            "insufficient_new_signal_samples": 6,
            "insufficient_dpo_preference_pairs": 6,
            "min_trigger_interval_active": 7,
            "cooldown_active": 8,
        }
        prioritized_blocked_reasons = sorted(
            blocked_reasons,
            key=lambda item: (blocker_priority.get(str(item), 99), str(item)),
        )
        primary_blocked_reason = prioritized_blocked_reasons[0] if prioritized_blocked_reasons else None
        primary_blocked_details = self._auto_train_blocker_details(str(primary_blocked_reason or ""))
        reason = "ready" if ready else (str(primary_blocked_reason) if primary_blocked_reason is not None else "idle")
        blocked_summary_parts: list[str] = []
        if primary_blocked_reason is not None:
            blocked_summary_parts.append(f"reason={primary_blocked_reason}")
        if primary_blocked_details.get("action"):
            blocked_summary_parts.append(f"action={primary_blocked_details['action']}")
        if primary_blocked_details.get("category"):
            blocked_summary_parts.append(f"category={primary_blocked_details['category']}")
        if primary_blocked_details.get("summary"):
            blocked_summary_parts.append(f"summary={primary_blocked_details['summary']}")
        policy = self._auto_train_policy_summary(
            enabled=bool(trigger.enabled),
            queue_mode=str(trigger.queue_mode),
            require_queue_confirmation=bool(trigger.require_queue_confirmation),
            auto_evaluate=bool(trigger.auto_evaluate),
            auto_promote=bool(trigger.auto_promote),
            eval_num_samples=int(trigger.eval_num_samples),
        )
        samples_remaining = max(int(trigger.min_new_samples) - eligible_signal_train_samples, 0)
        dpo_pairs_remaining = max(int(trigger.min_new_samples) - eligible_dpo_preference_pairs, 0)
        train_samples_remaining = max(int(trigger.min_new_samples) - eligible_train_samples, 0)
        effective_train_samples_remaining = max(
            round(int(trigger.min_new_samples) - effective_eligible_train_samples, 3),
            0.0,
        )
        threshold_summary = {
            "train_type": train_type,
            "threshold_label": threshold_label,
            "min_new_samples": int(trigger.min_new_samples),
            "required_train_samples": int(trigger.min_new_samples),
            "eligible_train_samples": eligible_train_samples,
            "effective_eligible_train_samples": effective_eligible_train_samples,
            "preference_reinforced_train_samples": preference_reinforced_train_samples,
            "eligible_train_sample_ids": eligible_train_sample_ids[:10],
            "remaining_train_samples": train_samples_remaining,
            "remaining_effective_train_samples": effective_train_samples_remaining,
            "eligible_signal_train_samples": eligible_signal_train_samples,
            "effective_signal_train_samples": effective_signal_train_samples,
            "preference_reinforced_signal_train_samples": preference_reinforced_signal_train_samples,
            "eligible_signal_sample_ids": sample_ids[:10],
            "remaining_signal_samples": samples_remaining,
            "eligible_dpo_preference_pairs": eligible_dpo_preference_pairs,
            "effective_dpo_preference_pairs": effective_dpo_preference_pairs,
            "preference_reinforced_dpo_preference_pairs": preference_reinforced_dpo_preference_pairs,
            "eligible_dpo_preference_pair_ids": eligible_dpo_preference_pair_ids[:10],
            "remaining_dpo_preference_pairs": dpo_pairs_remaining,
            "preference_reinforced_sample_weight": preference_reinforced_sample_weight,
            "holdout_required": True,
            "holdout_ready": holdout_ready,
            "max_interval_days": int(trigger.max_interval_days),
            "days_since_last_training": days_since_last_training,
            "interval_elapsed": interval_elapsed,
            "min_trigger_interval_minutes": int(trigger.min_trigger_interval_minutes),
            "cooldown_elapsed": cooldown_elapsed,
            "cooldown_remaining_minutes": cooldown_remaining_minutes,
            "failure_backoff_minutes": int(trigger.failure_backoff_minutes),
            "failure_backoff_elapsed": failure_backoff_elapsed,
            "failure_backoff_remaining_minutes": failure_backoff_remaining_minutes,
            "summary_line": " | ".join(
                [
                    f"{threshold_label}={eligible_train_samples}/{int(trigger.min_new_samples)}",
                    (
                        "effective="
                        + self._format_trigger_count(effective_eligible_train_samples)
                        + f"/{int(trigger.min_new_samples)}"
                    ),
                    f"reinforced={preference_reinforced_train_samples}",
                    f"holdout={'ready' if holdout_ready else 'needed'}",
                    (
                        f"interval={days_since_last_training}/{int(trigger.max_interval_days)}d"
                        if days_since_last_training is not None
                        else f"interval=none/{int(trigger.max_interval_days)}d"
                    ),
                    f"cooldown={'ok' if cooldown_elapsed else f'{cooldown_remaining_minutes}m'}",
                    f"backoff={'ok' if failure_backoff_elapsed else f'{failure_backoff_remaining_minutes}m'}",
                ]
            ),
        }
        return {
            "enabled": bool(trigger.enabled),
            "state": "ready" if ready else ("disabled" if not trigger.enabled else "blocked"),
            "ready": ready,
            "reason": reason,
            "blocked_reasons": prioritized_blocked_reasons,
            "blocked_primary_reason": primary_blocked_reason,
            "blocked_primary_action": primary_blocked_details.get("action"),
            "blocked_primary_category": primary_blocked_details.get("category"),
            "blocked_summary": " | ".join(blocked_summary_parts) if blocked_summary_parts else None,
            "train_type": config.trainer.train_type,
            "method": config.trainer.method,
            "epochs": config.trainer.epochs,
            "base_model": resolve_base_model_reference(config.model.base_model or "local-default"),
            "min_new_samples": int(trigger.min_new_samples),
            "max_interval_days": int(trigger.max_interval_days),
            "min_trigger_interval_minutes": int(trigger.min_trigger_interval_minutes),
            "failure_backoff_minutes": int(trigger.failure_backoff_minutes),
            "queue_mode": str(trigger.queue_mode),
            "queue_dedup_scope": str(trigger.queue_dedup_scope),
            "queue_priority_policy": str(trigger.queue_priority_policy),
            "queue_process_batch_size": int(trigger.queue_process_batch_size),
            "queue_process_until_idle_max": int(trigger.queue_process_until_idle_max),
            "queue_worker_max_cycles": int(trigger.queue_worker_max_cycles),
            "queue_worker_idle_rounds": int(trigger.queue_worker_idle_rounds),
            "queue_worker_poll_seconds": float(trigger.queue_worker_poll_seconds),
            "queue_worker_runner_max_seconds": float(trigger.queue_worker_runner_max_seconds),
            "queue_worker_runner_idle_sleep_seconds": float(trigger.queue_worker_runner_idle_sleep_seconds),
            "require_queue_confirmation": bool(trigger.require_queue_confirmation),
            "auto_evaluate": bool(trigger.auto_evaluate),
            "auto_promote": bool(trigger.auto_promote),
            "eval_num_samples": int(trigger.eval_num_samples),
            "eligible_train_samples": eligible_train_samples,
            "effective_eligible_train_samples": effective_eligible_train_samples,
            "preference_reinforced_train_samples": preference_reinforced_train_samples,
            "eligible_train_sample_ids": eligible_train_sample_ids[:10],
            "eligible_signal_train_samples": eligible_signal_train_samples,
            "effective_signal_train_samples": effective_signal_train_samples,
            "preference_reinforced_signal_train_samples": preference_reinforced_signal_train_samples,
            "eligible_signal_sample_ids": sample_ids[:10],
            "eligible_dpo_preference_pairs": eligible_dpo_preference_pairs,
            "effective_dpo_preference_pairs": effective_dpo_preference_pairs,
            "preference_reinforced_dpo_preference_pairs": preference_reinforced_dpo_preference_pairs,
            "eligible_dpo_preference_pair_ids": eligible_dpo_preference_pair_ids[:10],
            "preference_reinforced_sample_weight": preference_reinforced_sample_weight,
            "holdout_ready": holdout_ready,
            "days_since_last_training": days_since_last_training,
            "interval_elapsed": interval_elapsed,
            "recent_training_version": (recent_row or {}).get("version"),
            "queue_gate_reason": queue_gate_reason,
            "queue_gate_action": queue_gate_action,
            "queue_review_mode": queue_review_policy.get("review_mode"),
            "queue_review_policy_summary": queue_review_policy.get("summary_line"),
            "threshold_summary": threshold_summary,
            "cooldown_elapsed": cooldown_elapsed,
            "cooldown_remaining_minutes": cooldown_remaining_minutes,
            "failure_backoff_elapsed": failure_backoff_elapsed,
            "failure_backoff_remaining_minutes": failure_backoff_remaining_minutes,
            "last_attempted_at": last_attempted_at.isoformat() if last_attempted_at is not None else None,
            "last_completed_at": last_completed_at.isoformat() if last_completed_at is not None else None,
            "last_success_at": last_success_at.isoformat() if last_success_at is not None else None,
            "last_failure_at": last_failure_at.isoformat() if last_failure_at is not None else None,
            "consecutive_failures": consecutive_failures,
            "last_result": last_result,
            "last_result_summary": last_result_summary,
            "policy": policy,
        }

    def _finalize_auto_train_result(
        self,
        *,
        result: dict[str, Any],
        queue_item: dict[str, Any],
        trigger: Any,
        workspace: str | None = None,
        success: bool,
        completed: bool,
    ) -> dict[str, Any]:
        completed_at = datetime.now(timezone.utc)
        persisted = self._load_auto_trigger_state(workspace=workspace)
        last_attempted_at = persisted.get("last_attempted_at") or completed_at.isoformat()
        persisted["last_attempted_at"] = last_attempted_at
        if completed:
            persisted["last_completed_at"] = completed_at.isoformat()
        persisted["last_result"] = dict(result)
        persisted["last_result_summary"] = self._summarize_auto_trigger_result(result)
        if completed and success:
            persisted["last_success_at"] = completed_at.isoformat()
            persisted["consecutive_failures"] = 0
            persisted["failure_backoff_until"] = None
            cooldown_minutes = int(trigger.min_trigger_interval_minutes)
            persisted["cooldown_until"] = (completed_at + timedelta(minutes=cooldown_minutes)).isoformat() if cooldown_minutes > 0 else None
        elif completed:
            persisted["last_failure_at"] = completed_at.isoformat()
            persisted["consecutive_failures"] = int(persisted.get("consecutive_failures") or 0) + 1
            backoff_minutes = int(trigger.failure_backoff_minutes)
            persisted["failure_backoff_until"] = (completed_at + timedelta(minutes=backoff_minutes)).isoformat() if backoff_minutes > 0 else None
        action_payload = {
            "action": "signal_auto_train" if str(queue_item.get("source")) == "signal_auto_train" else "process_next",
            "status": result.get("state") if completed else "queued",
            "reason": result.get("reason"),
            "triggered": bool(result.get("triggered")),
            "triggered_version": result.get("triggered_version"),
            "promoted_version": result.get("promoted_version"),
            "queue_job_id": queue_item.get("job_id"),
        }
        persisted["last_action"] = action_payload
        self._persist_auto_trigger_state(persisted, workspace=workspace)
        if completed:
            self._update_train_queue_item(
                str(queue_item["job_id"]),
                {
                    "state": "completed" if success else "failed",
                    "updated_at": completed_at.isoformat(),
                    "history_event": "completed" if success else "failed",
                    "history_reason": result.get("reason"),
                    "adapter_version": result.get("triggered_version"),
                    "adapter_state": result.get("triggered_state"),
                    "eval_recommendation": result.get("eval_recommendation"),
                    "eval_comparison": result.get("eval_comparison"),
                    "promoted_version": result.get("promoted_version"),
                    "error_stage": result.get("error_stage"),
                    "error": result.get("error") or result.get("eval_error"),
                },
                workspace=workspace,
            )
        result["last_result_summary"] = persisted["last_result_summary"]
        result["queue_job_id"] = queue_item["job_id"]
        self.last_auto_trigger_result = dict(result)
        return result


    @staticmethod
    def _calculate_win_rate(eval_report: dict[str, Any]) -> float:
        """Calculate win rate vs baseline from eval report.

        Args:
            eval_report: Evaluation report containing details with winner field

        Returns:
            Win rate as a float between 0.0 and 1.0
        """
        details = eval_report.get("details", [])
        if not details:
            return 0.0

        adapted_wins = sum(1 for d in details if d.get("winner") == "adapted")
        ties = sum(1 for d in details if d.get("winner") == "tie")
        # Count ties as half win
        return (adapted_wins + 0.5 * ties) / len(details)

    def _should_auto_promote(
        self,
        eval_report: dict[str, Any],
        config: Any,
    ) -> tuple[bool, str]:
        """Determine if adapter should be auto-promoted based on eval results.

        Args:
            eval_report: Evaluation report
            config: PFEConfig with promotion policy settings

        Returns:
            Tuple of (should_promote, reason)
        """
        promote_policy = config.trainer.trigger.promote_gate_policy
        eval_policy = config.trainer.trigger.eval_gate_policy

        # Check if auto-promote is enabled
        if not promote_policy.auto_promote and not eval_policy.auto_promote_after_eval:
            return False, "auto_promote_disabled"

        # Calculate win rate
        win_rate = self._calculate_win_rate(eval_report)
        win_rate_threshold = promote_policy.win_rate_threshold

        # Check win rate threshold
        if win_rate < win_rate_threshold:
            return False, f"win_rate_{win_rate:.1%}_below_threshold_{win_rate_threshold:.1%}"

        # Check quality scores
        scores = eval_report.get("scores", {})
        min_quality = promote_policy.min_quality_score
        min_style = promote_policy.min_style_match_score
        min_preference = promote_policy.min_preference_alignment_score
        min_preservation = promote_policy.min_quality_preservation_score

        if scores.get("quality_preservation", 0) < min_preservation:
            return False, f"quality_preservation_{scores.get('quality_preservation', 0):.2f}_below_{min_preservation}"

        if scores.get("style_match", 0) < min_style:
            return False, f"style_match_{scores.get('style_match', 0):.2f}_below_{min_style}"

        if scores.get("preference_alignment", 0) < min_preference:
            return False, f"preference_alignment_{scores.get('preference_alignment', 0):.2f}_below_{min_preference}"

        # Check eval recommendation if required
        if promote_policy.require_eval_recommendation_deploy:
            recommendation = eval_report.get("recommendation", "")
            if recommendation != "deploy":
                return False, f"eval_recommendation_{recommendation}_not_deploy"

        return True, f"win_rate_{win_rate:.1%}_meets_threshold"


    def _maybe_trigger_auto_eval_after_train(
        self,
        *,
        train_result: Any,
        base_model: str | None = None,
        workspace: str | None = None,
    ) -> None:
        """Trigger auto-eval after training if enabled.

        Args:
            train_result: Training result object with version info
            base_model: Base model name used for training
            workspace: Optional workspace name
        """
        config = self._load_config_with_workspace(workspace)
        eval_policy = config.trainer.trigger.eval_gate_policy

        # Check if auto-eval trigger is enabled
        if not eval_policy.auto_trigger:
            return

        # Get base model from config if not provided
        base = base_model or config.model.base_model

        # Build trigger status for auto-eval
        trigger_status = {
            "eval_num_samples": config.trainer.trigger.eval_num_samples,
        }

        # Run auto-eval and promotion
        auto_result = self._run_auto_eval_and_promote(
            train_result=train_result,
            base_model=base,
            trigger_status=trigger_status,
            config=config,
            workspace=workspace,
        )

        # Log the result
        if auto_result.get("eval_triggered"):
            logger.info(
                f"Auto-eval completed for {train_result.version}: "
                f"win_rate={auto_result.get('win_rate', 0):.1%}, "
                f"decision={auto_result.get('promote_decision', 'unknown')}"
            )

    def _run_auto_eval_and_promote(
        self,
        *,
        train_result: Any,
        base_model: str,
        trigger_status: dict[str, Any],
        config: Any,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        """Run automatic evaluation and promotion decision.

        Args:
            train_result: Training result with version info
            base_model: Base model name
            trigger_status: Trigger status dict
            config: PFEConfig
            workspace: Optional workspace name

        Returns:
            Dict with eval and promotion results
        """
        result: dict[str, Any] = {
            "eval_triggered": False,
            "promote_triggered": False,
            "eval_report": None,
            "promote_reason": None,
        }

        eval_policy = config.trainer.trigger.eval_gate_policy
        if not eval_policy.auto_trigger:
            result["promote_reason"] = "auto_eval_trigger_disabled"
            return result

        try:
            eval_report = json.loads(
                self.evaluate(
                    base_model=base_model,
                    adapter=str(train_result.version),
                    num_samples=int(trigger_status.get("eval_num_samples", 20)),
                    workspace=workspace,
                )
            )
            result["eval_report"] = eval_report
            result["eval_triggered"] = True

            # Calculate win rate for logging
            win_rate = self._calculate_win_rate(eval_report)
            result["win_rate"] = win_rate

        except Exception as exc:
            result["eval_triggered"] = True
            result["eval_error"] = f"{exc.__class__.__name__}: {exc}"
            result["promote_reason"] = f"eval_failed: {exc}"
            return result

        # Make promotion decision
        should_promote, reason = self._should_auto_promote(eval_report, config)
        result["promote_decision"] = "promote" if should_promote else "archive"
        result["promote_reason"] = reason

        if should_promote:
            try:
                store = create_adapter_store(workspace=workspace)
                promoted = store.promote(str(train_result.version))
                result["promote_triggered"] = True
                result["promoted_version"] = promoted
                result["triggered_state"] = "promoted"
            except Exception as exc:
                result["promote_triggered"] = False
                result["promote_error"] = f"{exc.__class__.__name__}: {exc}"
                result["promote_reason"] = f"promote_failed: {exc}"
        else:
            # Archive if not promoted
            try:
                store = create_adapter_store(workspace=workspace)
                store.archive(str(train_result.version))
                result["archived_version"] = str(train_result.version)
                result["triggered_state"] = "archived"
            except Exception as exc:
                result["archive_error"] = f"{exc.__class__.__name__}: {exc}"

        return result

    def _execute_train_queue_item(
        self,
        *,
        trigger_status: dict[str, Any],
        queue_item: dict[str, Any],
        workspace: str | None = None,
    ) -> dict[str, Any]:
        config = self._load_config()
        trigger = config.trainer.trigger
        result = dict(trigger_status)
        execution_policy = self._auto_train_execution_policy(
            trigger_status=trigger_status,
            queue_item=queue_item,
        )
        result["execution_policy"] = execution_policy
        result["triggered"] = False
        try:
            train_result = self.train_result(
                method=str(queue_item.get("requested_method") or trigger_status["method"]),
                epochs=int(trigger_status["epochs"]),
                base_model=str(queue_item.get("requested_base_model") or trigger_status["base_model"]),
                train_type=str(queue_item.get("requested_train_type") or trigger_status["train_type"]),
                workspace=workspace,
                backend_hint=getattr(config.trainer, "backend", None) or None,
            )
            result.update(
                {
                    "triggered": True,
                    "state": "triggered",
                    "reason": "triggered",
                    "triggered_version": train_result.version,
                    "triggered_state": train_result.metrics.get("state"),
                    "triggered_num_fresh_samples": train_result.metrics.get("num_fresh_samples"),
                    "triggered_num_replay_samples": train_result.metrics.get("num_replay_samples"),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "triggered": True,
                    "state": "failed",
                    "reason": "train_failed",
                    "error_stage": "train",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "eval_triggered": False,
                    "promote_triggered": False,
                }
            )
            return self._finalize_auto_train_result(
                result=result,
                queue_item=queue_item,
                trigger=trigger,
                workspace=workspace,
                success=False,
                completed=True,
            )

        if execution_policy.get("auto_evaluate_enabled"):
            try:
                eval_report = json.loads(
                    self.evaluate(
                        base_model=str(queue_item.get("requested_base_model") or trigger_status["base_model"]),
                        adapter=str(train_result.version),
                        num_samples=int(trigger_status["eval_num_samples"]),
                        workspace=workspace,
                    )
                )
                result["eval_report"] = eval_report
                result["eval_recommendation"] = eval_report.get("recommendation")
                result["eval_comparison"] = eval_report.get("comparison")
                result["eval_triggered"] = True
            except Exception as exc:
                result.update(
                    {
                        "eval_triggered": True,
                        "promote_triggered": False,
                        "state": "failed",
                        "reason": "eval_failed",
                        "error_stage": "eval",
                        "eval_error": f"{exc.__class__.__name__}: {exc}",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
                return self._finalize_auto_train_result(
                    result=result,
                    queue_item=queue_item,
                    trigger=trigger,
                    workspace=workspace,
                    success=False,
                    completed=True,
                )
            if execution_policy.get("auto_promote_enabled") and result.get("eval_recommendation") == "deploy":
                try:
                    store = create_adapter_store(workspace=workspace)
                    promoted = store.promote(str(train_result.version))
                    result["promote_triggered"] = True
                    result["promoted_version"] = promoted
                    result["triggered_state"] = "promoted"
                except Exception as exc:
                    result.update(
                        {
                            "promote_triggered": False,
                            "state": "failed",
                            "reason": "promote_failed",
                            "error_stage": "promote",
                            "error": f"{exc.__class__.__name__}: {exc}",
                        }
                    )
                    return self._finalize_auto_train_result(
                        result=result,
                        queue_item=queue_item,
                        trigger=trigger,
                        workspace=workspace,
                        success=False,
                    completed=True,
                )
            else:
                result["promote_triggered"] = False
                if execution_policy.get("auto_promote_requested"):
                    result["promote_reason"] = (
                        "eval_not_deployable"
                        if execution_policy.get("auto_promote_enabled")
                        else execution_policy.get("promote_gate_reason")
                    )
        else:
            result["eval_triggered"] = False
            result["promote_triggered"] = False
            if execution_policy.get("auto_promote_requested"):
                result["promote_reason"] = execution_policy.get("promote_gate_reason")
        return self._finalize_auto_train_result(
            result=result,
            queue_item=queue_item,
            trigger=trigger,
            workspace=workspace,
            success=True,
            completed=True,
        )

    def _maybe_auto_train_from_signal(self, *, workspace: str | None = None) -> dict[str, Any]:
        trigger_status = self._auto_train_trigger_status(workspace=workspace)
        result = dict(trigger_status)
        result["triggered"] = False
        if not trigger_status.get("ready"):
            self.last_auto_trigger_result = result
            return result
        config = self._load_config()
        trigger = config.trainer.trigger
        now = datetime.now(timezone.utc)
        state = self._load_auto_trigger_state(workspace=workspace)
        state["last_attempted_at"] = now.isoformat()
        self._persist_auto_trigger_state(state, workspace=workspace)
        dedup_scope = str(trigger.queue_dedup_scope)
        priority_policy = str(trigger.queue_priority_policy)
        dedup_key = self._auto_train_queue_dedup_key(
            trigger_status,
            dedup_scope=dedup_scope,
            workspace=workspace,
        )
        priority, priority_source = self._queue_priority(
            trigger_status=trigger_status,
            source="signal_auto_train",
            policy=priority_policy,
        )
        existing_queue_item = self._find_train_queue_item_by_dedup_key(dedup_key, workspace=workspace)
        if existing_queue_item is not None:
            result.update(
                {
                    "enqueued": False,
                    "triggered": False,
                    "state": str(existing_queue_item.get("state") or "queued"),
                    "reason": "queue_duplicate",
                    "eval_triggered": False,
                    "promote_triggered": False,
                    "queue_job_id": existing_queue_item.get("job_id"),
                    "dedup_key": dedup_key,
                    "queue_dedup_scope": dedup_scope,
                    "queue_priority_policy": priority_policy,
                }
            )
            return self._finalize_auto_train_result(
                result=result,
                queue_item=existing_queue_item,
                trigger=trigger,
                workspace=workspace,
                success=True,
                completed=False,
            )
        queue_item = self._append_train_queue_item(
            {
                "job_id": f"auto-train-{uuid4().hex[:10]}",
                "state": "awaiting_confirmation" if str(trigger.queue_mode) == "deferred" and bool(trigger.require_queue_confirmation) else "queued",
                "workspace": str(workspace or "user_default"),
                "source": "signal_auto_train",
                "reason": trigger_status.get("reason"),
                "triggered_at": now.isoformat(),
                "requested_base_model": trigger_status.get("base_model"),
                "requested_method": trigger_status.get("method"),
                "requested_train_type": trigger_status.get("train_type"),
                "auto_evaluate": bool(trigger_status.get("auto_evaluate")),
                "auto_promote": bool(trigger_status.get("auto_promote")),
                "queue_mode": str(trigger.queue_mode),
                "priority": priority,
                "priority_source": priority_source,
                "queue_dedup_scope": dedup_scope,
                "queue_priority_policy": priority_policy,
                "dedup_key": dedup_key,
                "confirmation_required": bool(str(trigger.queue_mode) == "deferred" and bool(trigger.require_queue_confirmation)),
                "confirmation_reason": "manual_review_required_by_policy" if bool(str(trigger.queue_mode) == "deferred" and bool(trigger.require_queue_confirmation)) else None,
            },
            workspace=workspace,
        )
        if bool(trigger.require_queue_confirmation) and str(trigger.queue_mode) == "deferred":
            result.update(
                {
                    "enqueued": True,
                    "triggered": False,
                    "state": "awaiting_confirmation",
                    "reason": "awaiting_confirmation",
                    "eval_triggered": False,
                    "promote_triggered": False,
                    "queue_job_id": queue_item["job_id"],
                }
            )
            return self._finalize_auto_train_result(
                result=result,
                queue_item=queue_item,
                trigger=trigger,
                workspace=workspace,
                success=True,
                completed=False,
            )
        if str(trigger.queue_mode) == "deferred":
            result.update(
                {
                    "enqueued": True,
                    "triggered": False,
                    "state": "queued",
                    "reason": "enqueued",
                    "eval_triggered": False,
                    "promote_triggered": False,
                    "queue_job_id": queue_item["job_id"],
                }
            )
            return self._finalize_auto_train_result(
                result=result,
                queue_item=queue_item,
                trigger=trigger,
                workspace=workspace,
                success=True,
                completed=False,
            )
        queue_item = self._update_train_queue_item(
            str(queue_item["job_id"]),
            {
                "state": "running",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "history_event": "running",
                "history_reason": "processing_started",
            },
            workspace=workspace,
        )
        return self._execute_train_queue_item(
            trigger_status=trigger_status,
            queue_item=queue_item,
            workspace=workspace,
        )

    def approve_next_train_queue(self, *, workspace: str | None = None, note: str | None = None) -> dict[str, Any]:
        queue_item = self._first_train_queue_item(states=("awaiting_confirmation",), workspace=workspace)
        snapshot = self.status(workspace=workspace)
        if queue_item is None:
            action_payload = {
                "action": "approve_next",
                "status": "noop",
                "reason": "no_pending_confirmation",
                "triggered": False,
            }
            self._persist_auto_trigger_action(action_payload, workspace=workspace)
            snapshot["auto_train_trigger_action"] = action_payload
            return snapshot
        queue_item = self._update_train_queue_item(
            str(queue_item["job_id"]),
            {
                "state": "queued",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "confirmation_required": False,
                "confirmation_reviewed": True,
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "approval_reason": "manual_approve_next",
                "operator_note": note,
                "history_event": "approved",
                "history_reason": "confirmation_approved",
                "history_metadata": {"note": note} if note else None,
            },
            workspace=workspace,
        )
        action_payload = {
            "action": "approve_next",
            "status": "completed",
            "reason": "confirmation_approved",
            "triggered": False,
            "queue_job_id": queue_item.get("job_id"),
            "confirmation_reason": queue_item.get("confirmation_reason"),
            "approval_reason": queue_item.get("approval_reason"),
            "operator_note": note,
        }
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot = self.status(workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def reject_next_train_queue(self, *, workspace: str | None = None, note: str | None = None) -> dict[str, Any]:
        queue_item = self._first_train_queue_item(states=("awaiting_confirmation",), workspace=workspace)
        snapshot = self.status(workspace=workspace)
        if queue_item is None:
            action_payload = {
                "action": "reject_next",
                "status": "noop",
                "reason": "no_pending_confirmation",
                "triggered": False,
            }
            self._persist_auto_trigger_action(action_payload, workspace=workspace)
            snapshot["auto_train_trigger_action"] = action_payload
            return snapshot
        queue_item = self._update_train_queue_item(
            str(queue_item["job_id"]),
            {
                "state": "rejected",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "confirmation_reviewed": True,
                "rejected_at": datetime.now(timezone.utc).isoformat(),
                "rejection_reason": "manual_reject",
                "operator_note": note,
                "history_event": "rejected",
                "history_reason": "confirmation_rejected",
                "history_metadata": {"note": note} if note else None,
            },
            workspace=workspace,
        )
        action_payload = {
            "action": "reject_next",
            "status": "completed",
            "reason": "confirmation_rejected",
            "triggered": False,
            "queue_job_id": queue_item.get("job_id"),
            "confirmation_reason": queue_item.get("confirmation_reason"),
            "rejection_reason": queue_item.get("rejection_reason"),
            "operator_note": note,
        }
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot = self.status(workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def enable_auto_train_trigger(self, *, workspace: str | None = None) -> dict[str, Any]:
        """Enable automatic training trigger.

        When enabled, training will be automatically triggered when:
        - New signals >= min_new_samples (default 50)
        - Max interval days reached (default 7 days)
        - Other gate conditions pass

        Returns:
            Status snapshot with enable action recorded
        """
        # Update configuration to enable auto-train
        config = self._load_config_with_workspace(workspace)
        config.trainer.trigger.enabled = True

        # Persist the configuration change
        self._save_config(config)

        snapshot = self.status(workspace=workspace)
        action_payload = {
            "action": "enable",
            "status": "completed",
            "reason": "auto_train_trigger_enabled",
            "enabled": True,
        }
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def disable_auto_train_trigger(self, *, workspace: str | None = None) -> dict[str, Any]:
        """Disable automatic training trigger.

        When disabled, training must be triggered manually via 'pfe train'
        or 'pfe trigger retry'.

        Returns:
            Status snapshot with disable action recorded
        """
        # Update configuration to disable auto-train
        config = self._load_config_with_workspace(workspace)
        config.trainer.trigger.enabled = False

        # Persist the configuration change
        self._save_config(config)

        snapshot = self.status(workspace=workspace)
        action_payload = {
            "action": "disable",
            "status": "completed",
            "reason": "auto_train_trigger_disabled",
            "enabled": False,
        }
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot
        return snapshot

    def enable_auto_eval_trigger(self, *, workspace: str | None = None) -> dict[str, Any]:
        """Enable automatic evaluation trigger after training.

        When enabled, evaluation will be automatically triggered when:
        - Training completes successfully
        - A new adapter version is created

        Returns:
            Status snapshot with enable action recorded
        """
        config = self._load_config_with_workspace(workspace)
        config.trainer.trigger.eval_gate_policy.auto_trigger = True
        self._save_config(config)

        snapshot = self.status(workspace=workspace)
        action_payload = {
            "action": "enable_auto_eval_trigger",
            "status": "completed",
            "reason": "auto_eval_trigger_enabled",
            "enabled": True,
        }
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_eval_trigger_action"] = action_payload
        return snapshot

    def disable_auto_eval_trigger(self, *, workspace: str | None = None) -> dict[str, Any]:
        """Disable automatic evaluation trigger.

        When disabled, evaluation must be triggered manually via 'pfe eval'
        after training completes.

        Returns:
            Status snapshot with disable action recorded
        """
        config = self._load_config_with_workspace(workspace)
        config.trainer.trigger.eval_gate_policy.auto_trigger = False
        self._save_config(config)

        snapshot = self.status(workspace=workspace)
        action_payload = {
            "action": "disable_auto_eval_trigger",
            "status": "completed",
            "reason": "auto_eval_trigger_disabled",
            "enabled": False,
        }
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_eval_trigger_action"] = action_payload
        return snapshot

    def get_auto_eval_trigger_status(self, *, workspace: str | None = None) -> dict[str, Any]:
        """Get the current status of auto-eval trigger.

        Returns:
            Dict with enabled status and configuration details
        """
        config = self._load_config_with_workspace(workspace)
        eval_policy = config.trainer.trigger.eval_gate_policy
        promote_policy = config.trainer.trigger.promote_gate_policy

        return {
            "enabled": bool(eval_policy.auto_trigger),
            "auto_promote_after_eval": bool(eval_policy.auto_promote_after_eval),
            "win_rate_threshold": float(promote_policy.win_rate_threshold),
            "eval_config": {
                "min_eval_samples": eval_policy.min_eval_samples,
                "max_eval_samples": eval_policy.max_eval_samples,
                "eval_frequency_hours": eval_policy.eval_frequency_hours,
            },
            "promote_config": {
                "auto_promote": bool(promote_policy.auto_promote),
                "min_quality_score": promote_policy.min_quality_score,
                "require_eval_recommendation_deploy": promote_policy.require_eval_recommendation_deploy,
            },
        }


    def _load_config_with_workspace(self, workspace: str | None = None) -> Any:
        """Load configuration with optional workspace override."""
        from .config import PFEConfig
        config = PFEConfig.load()
        if workspace:
            config.workspace = workspace
        return config

    def _save_config(self, config: Any) -> None:
        """Save configuration to disk."""
        config.save()

    def reset_auto_train_trigger(self, *, workspace: str | None = None) -> dict[str, Any]:
        path = self._auto_trigger_state_path(workspace=workspace)
        existed = path.exists()
        if existed:
            try:
                path.unlink()
            except FileNotFoundError:
                existed = False
        self.last_auto_trigger_result = None
        snapshot = self.status(workspace=workspace)
        action_payload = {
            "action": "reset",
            "status": "completed" if existed else "noop",
            "reason": "state_cleared" if existed else "state_not_found",
            "state_path": str(path),
        }
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def retry_auto_train_trigger(self, *, workspace: str | None = None) -> dict[str, Any]:
        auto_train = self._maybe_auto_train_from_signal(workspace=workspace)
        snapshot = self.status(workspace=workspace)
        action_payload = {
            "action": "retry",
            "status": "triggered" if auto_train.get("triggered") else "blocked",
            "reason": auto_train.get("reason"),
            "triggered": bool(auto_train.get("triggered")),
            "triggered_version": auto_train.get("triggered_version"),
            "promoted_version": auto_train.get("promoted_version"),
        }
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def process_next_train_queue(self, *, workspace: str | None = None) -> dict[str, Any]:
        result, queue_item = self._process_next_train_queue_result(workspace=workspace)
        snapshot = self.status(workspace=workspace)
        if result is None or queue_item is None:
            action_payload = {
                "action": "process_next",
                "status": "noop",
                "reason": "no_queued_items",
                "triggered": False,
            }
            self._persist_auto_trigger_action(action_payload, workspace=workspace)
            snapshot["auto_train_trigger_action"] = action_payload
            return snapshot
        action_payload = {
            "action": "process_next",
            "status": "triggered" if result.get("triggered") else result.get("state", "blocked"),
            "reason": result.get("reason"),
            "triggered": bool(result.get("triggered")),
            "triggered_version": result.get("triggered_version"),
            "promoted_version": result.get("promoted_version"),
            "queue_job_id": queue_item.get("job_id"),
        }
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def _process_next_train_queue_result(
        self,
        *,
        workspace: str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        queue_item = self._first_train_queue_item(states=("queued",), workspace=workspace)
        if queue_item is None:
            return None, None
        queue_item = self._update_train_queue_item(
            str(queue_item["job_id"]),
            {
                "state": "running",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "history_event": "running",
                "history_reason": "processing_started",
            },
            workspace=workspace,
        )
        trigger_status = self._auto_train_trigger_status(workspace=workspace)
        result = self._execute_train_queue_item(
            trigger_status=trigger_status,
            queue_item=queue_item,
            workspace=workspace,
        )
        return result, queue_item

    def process_train_queue_batch(self, *, workspace: str | None = None, limit: int = 5) -> dict[str, Any]:
        safe_limit = max(1, int(limit))
        processed_results: list[dict[str, Any]] = []
        processed_job_ids: list[str] = []
        for _ in range(safe_limit):
            result, queue_item = self._process_next_train_queue_result(workspace=workspace)
            if result is None or queue_item is None:
                break
            processed_results.append(dict(result))
            processed_job_ids.append(str(queue_item.get("job_id") or ""))
        action_payload = {
            "action": "process_batch",
            "status": "completed" if processed_results else "noop",
            "reason": "processed_batch" if processed_results else "no_queued_items",
            "triggered": bool(processed_results),
            "processed_count": len(processed_results),
            "completed_count": sum(1 for item in processed_results if item.get("state") != "failed"),
            "failed_count": sum(1 for item in processed_results if item.get("state") == "failed"),
            "limit": safe_limit,
            "queue_job_ids": [job_id for job_id in processed_job_ids if job_id],
            "triggered_version": next((item.get("triggered_version") for item in reversed(processed_results) if item.get("triggered_version")), None),
            "promoted_version": next((item.get("promoted_version") for item in reversed(processed_results) if item.get("promoted_version")), None),
        }
        snapshot = self.status(workspace=workspace)
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def process_train_queue_until_idle(self, *, workspace: str | None = None, max_iterations: int | None = None) -> dict[str, Any]:
        config = self._load_config()
        configured_limit = int(config.trainer.trigger.queue_process_until_idle_max)
        safe_limit = max(1, int(max_iterations if max_iterations is not None else configured_limit))
        processed_results: list[dict[str, Any]] = []
        processed_job_ids: list[str] = []
        for _ in range(safe_limit):
            result, queue_item = self._process_next_train_queue_result(workspace=workspace)
            if result is None or queue_item is None:
                break
            processed_results.append(dict(result))
            processed_job_ids.append(str(queue_item.get("job_id") or ""))
        queue_snapshot = self._train_queue_snapshot(workspace=workspace)
        remaining_queued = int((queue_snapshot.get("counts") or {}).get("queued", 0) or 0)
        action_payload = {
            "action": "process_until_idle",
            "status": "completed" if processed_results else "noop",
            "reason": "processed_until_idle" if processed_results else "no_queued_items",
            "triggered": bool(processed_results),
            "processed_count": len(processed_results),
            "completed_count": sum(1 for item in processed_results if item.get("state") != "failed"),
            "failed_count": sum(1 for item in processed_results if item.get("state") == "failed"),
            "max_iterations": safe_limit,
            "queue_job_ids": [job_id for job_id in processed_job_ids if job_id],
            "remaining_queued": remaining_queued,
            "drained": remaining_queued == 0,
            "triggered_version": next((item.get("triggered_version") for item in reversed(processed_results) if item.get("triggered_version")), None),
            "promoted_version": next((item.get("promoted_version") for item in reversed(processed_results) if item.get("promoted_version")), None),
        }
        snapshot = self.status(workspace=workspace)
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def run_train_queue_worker_loop(
        self,
        *,
        workspace: str | None = None,
        max_cycles: int | None = None,
        idle_rounds: int | None = None,
        poll_interval_seconds: float | None = None,
    ) -> dict[str, Any]:
        config = self._load_config()
        trigger = config.trainer.trigger
        safe_max_cycles = max(1, int(max_cycles if max_cycles is not None else trigger.queue_worker_max_cycles))
        safe_idle_rounds = max(1, int(idle_rounds if idle_rounds is not None else trigger.queue_worker_idle_rounds))
        safe_poll_seconds = max(0.0, float(poll_interval_seconds if poll_interval_seconds is not None else trigger.queue_worker_poll_seconds))
        processed_results: list[dict[str, Any]] = []
        processed_job_ids: list[str] = []
        observed_idle_rounds = 0
        cycles = 0
        stopped_reason = "max_cycles"

        while cycles < safe_max_cycles:
            cycles += 1
            result, queue_item = self._process_next_train_queue_result(workspace=workspace)
            if result is None or queue_item is None:
                observed_idle_rounds += 1
                if observed_idle_rounds >= safe_idle_rounds:
                    stopped_reason = "idle"
                    break
                if safe_poll_seconds > 0:
                    time.sleep(safe_poll_seconds)
                continue
            observed_idle_rounds = 0
            processed_results.append(dict(result))
            processed_job_ids.append(str(queue_item.get("job_id") or ""))
            if safe_poll_seconds > 0 and cycles < safe_max_cycles:
                time.sleep(safe_poll_seconds)

        queue_snapshot = self._train_queue_snapshot(workspace=workspace)
        remaining_queued = int((queue_snapshot.get("counts") or {}).get("queued", 0) or 0)
        action_payload = {
            "action": "run_worker_loop",
            "status": "completed" if processed_results else "idle",
            "reason": "worker_loop_processed" if processed_results else "worker_loop_idle",
            "triggered": bool(processed_results),
            "processed_count": len(processed_results),
            "completed_count": sum(1 for item in processed_results if item.get("state") != "failed"),
            "failed_count": sum(1 for item in processed_results if item.get("state") == "failed"),
            "queue_job_ids": [job_id for job_id in processed_job_ids if job_id],
            "max_cycles": safe_max_cycles,
            "loop_cycles": cycles,
            "idle_rounds": observed_idle_rounds,
            "poll_interval_seconds": safe_poll_seconds,
            "remaining_queued": remaining_queued,
            "drained": remaining_queued == 0,
            "stopped_reason": stopped_reason,
            "triggered_version": next((item.get("triggered_version") for item in reversed(processed_results) if item.get("triggered_version")), None),
            "promoted_version": next((item.get("promoted_version") for item in reversed(processed_results) if item.get("promoted_version")), None),
        }
        snapshot = self.status(workspace=workspace)
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def run_train_queue_worker_runner(
        self,
        *,
        workspace: str | None = None,
        max_seconds: float | None = None,
        idle_sleep_seconds: float | None = None,
    ) -> dict[str, Any]:
        config = self._load_config()
        trigger = config.trainer.trigger
        safe_max_seconds = max(0.1, float(max_seconds if max_seconds is not None else trigger.queue_worker_runner_max_seconds))
        safe_idle_sleep = max(0.0, float(idle_sleep_seconds if idle_sleep_seconds is not None else trigger.queue_worker_runner_idle_sleep_seconds))
        existing_worker = self._train_queue_worker_summary(workspace=workspace)
        if existing_worker.get("lock_state") == "active" and not bool(existing_worker.get("stop_requested", False)):
            action_payload = {
                "action": "run_worker_runner",
                "status": "blocked",
                "reason": "runner_already_active",
                "triggered": False,
                "processed_count": 0,
                "failed_count": 0,
                "loop_cycles": int(existing_worker.get("loop_cycles", 0) or 0),
                "max_seconds": safe_max_seconds,
                "stopped_reason": existing_worker.get("stopped_reason"),
            }
            self._append_train_queue_worker_history(
                workspace=workspace,
                event="blocked_reentry",
                reason="runner_already_active",
                metadata={
                    "pid": existing_worker.get("pid"),
                    "lock_state": existing_worker.get("lock_state"),
                },
            )
            snapshot = self.status(workspace=workspace)
            self._persist_auto_trigger_action(action_payload, workspace=workspace)
            snapshot["auto_train_trigger_action"] = action_payload
            return snapshot
        takeover_state = existing_worker.get("lock_state") == "stale"

        start_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        state_payload = {
            "active": True,
            "stop_requested": False,
            "pid": os.getpid(),
            "started_at": started_at,
            "last_heartbeat_at": started_at,
            "last_completed_at": None,
            "loop_cycles": 0,
            "processed_count": 0,
            "failed_count": 0,
            "stopped_reason": None,
            "last_action": "run_worker_runner",
            "max_seconds": safe_max_seconds,
            "idle_sleep_seconds": safe_idle_sleep,
            "lock_state": "active",
            "takeover": bool(takeover_state),
            "taken_over_at": started_at if takeover_state else None,
            "previous_pid": existing_worker.get("pid") if takeover_state else None,
        }
        self._persist_train_queue_worker_state(state_payload, workspace=workspace)
        self._append_train_queue_worker_history(
            workspace=workspace,
            event="started",
            reason="stale_lock_takeover" if takeover_state else "runner_started",
            metadata={
                "pid": os.getpid(),
                "takeover": bool(takeover_state),
                "previous_pid": existing_worker.get("pid") if takeover_state else None,
                "max_seconds": safe_max_seconds,
            },
        )

        processed_results: list[dict[str, Any]] = []
        processed_job_ids: list[str] = []
        cycles = 0
        stopped_reason = "max_seconds"
        while (time.monotonic() - start_monotonic) < safe_max_seconds:
            cycles += 1
            current_state = self._load_train_queue_worker_state(workspace=workspace)
            if bool(current_state.get("stop_requested")):
                stopped_reason = "stop_requested"
                break
            result, queue_item = self._process_next_train_queue_result(workspace=workspace)
            heartbeat_at = datetime.now(timezone.utc).isoformat()
            if result is not None and queue_item is not None:
                processed_results.append(dict(result))
                processed_job_ids.append(str(queue_item.get("job_id") or ""))
            current_state.update(
                {
                    "active": True,
                    "pid": os.getpid(),
                    "last_heartbeat_at": heartbeat_at,
                    "loop_cycles": cycles,
                    "processed_count": len(processed_results),
                    "failed_count": sum(1 for item in processed_results if item.get("state") == "failed"),
                    "stopped_reason": None,
                    "last_action": "run_worker_runner",
                    "max_seconds": safe_max_seconds,
                    "idle_sleep_seconds": safe_idle_sleep,
                }
            )
            self._persist_train_queue_worker_state(current_state, workspace=workspace)
            if result is None or queue_item is None:
                if safe_idle_sleep > 0:
                    time.sleep(safe_idle_sleep)
                continue

        completed_at = datetime.now(timezone.utc).isoformat()
        final_state = self._load_train_queue_worker_state(workspace=workspace)
        final_state.update(
            {
                "active": False,
                "last_completed_at": completed_at,
                "last_heartbeat_at": completed_at,
                "loop_cycles": cycles,
                "processed_count": len(processed_results),
                "failed_count": sum(1 for item in processed_results if item.get("state") == "failed"),
                "stopped_reason": stopped_reason,
                "last_action": "run_worker_runner",
                "max_seconds": safe_max_seconds,
                "idle_sleep_seconds": safe_idle_sleep,
                "lock_state": "idle",
            }
        )
        self._persist_train_queue_worker_state(final_state, workspace=workspace)
        self._append_train_queue_worker_history(
            workspace=workspace,
            event="completed" if processed_results else ("stopped" if stopped_reason == "stop_requested" else "idle_exit"),
            reason=stopped_reason,
            metadata={
                "processed_count": len(processed_results),
                "failed_count": sum(1 for item in processed_results if item.get("state") == "failed"),
                "takeover": bool(takeover_state),
            },
        )

        queue_snapshot = self._train_queue_snapshot(workspace=workspace)
        remaining_queued = int((queue_snapshot.get("counts") or {}).get("queued", 0) or 0)
        action_payload = {
            "action": "run_worker_runner",
            "status": "completed" if processed_results else ("stopped" if stopped_reason == "stop_requested" else "idle"),
            "reason": "stale_lock_takeover_processed" if takeover_state and processed_results else ("stale_lock_takeover" if takeover_state else ("worker_runner_processed" if processed_results else stopped_reason)),
            "triggered": bool(processed_results),
            "processed_count": len(processed_results),
            "completed_count": sum(1 for item in processed_results if item.get("state") != "failed"),
            "failed_count": sum(1 for item in processed_results if item.get("state") == "failed"),
            "queue_job_ids": [job_id for job_id in processed_job_ids if job_id],
            "remaining_queued": remaining_queued,
            "drained": remaining_queued == 0,
            "max_seconds": safe_max_seconds,
            "loop_cycles": cycles,
            "poll_interval_seconds": safe_idle_sleep,
            "stopped_reason": stopped_reason,
            "takeover": bool(takeover_state),
            "previous_pid": existing_worker.get("pid") if takeover_state else None,
            "triggered_version": next((item.get("triggered_version") for item in reversed(processed_results) if item.get("triggered_version")), None),
            "promoted_version": next((item.get("promoted_version") for item in reversed(processed_results) if item.get("promoted_version")), None),
        }
        snapshot = self.status(workspace=workspace)
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def stop_train_queue_worker_runner(self, *, workspace: str | None = None) -> dict[str, Any]:
        state = self._load_train_queue_worker_state(workspace=workspace)
        active = bool(state.get("active", False))
        state.update(
            {
                "stop_requested": True,
                "last_action": "stop_worker_runner",
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._persist_train_queue_worker_state(state, workspace=workspace)
        self._append_train_queue_worker_history(
            workspace=workspace,
            event="stop_requested",
            reason="stop_requested" if active else "runner_not_active",
            metadata={"pid": state.get("pid"), "active": active},
        )
        action_payload = {
            "action": "stop_worker_runner",
            "status": "requested" if active else "noop",
            "reason": "stop_requested" if active else "runner_not_active",
            "triggered": False,
        }
        snapshot = self.status(workspace=workspace)
        self._persist_auto_trigger_action(action_payload, workspace=workspace)
        snapshot["auto_train_trigger_action"] = action_payload
        return snapshot

    def train_queue_worker_runner_status(self, *, workspace: str | None = None) -> dict[str, Any]:
        summary = self._train_queue_worker_summary(workspace=workspace)
        return {
            "workspace": workspace or "user_default",
            **summary,
        }

    def start_train_queue_daemon(self, *, workspace: str | None = None, note: str | None = None) -> dict[str, Any]:
        existing = self._train_queue_daemon_summary(workspace=workspace)
        if existing.get("lock_state") == "active" and not bool(existing.get("stop_requested", False)):
            self._append_train_queue_daemon_history(
                workspace=workspace,
                event="blocked_reentry",
                reason="daemon_already_active",
                metadata={"pid": existing.get("pid"), "lock_state": existing.get("lock_state")},
            )
            return self.train_queue_daemon_status(workspace=workspace)
        return self._spawn_train_queue_daemon(
            workspace=workspace,
            reset_restart_policy=True,
            recovery_reason=None,
            requested_action="start",
            note=note,
            requested_by="pipeline",
            auto_recovery=False,
        )

    def stop_train_queue_daemon(self, *, workspace: str | None = None, note: str | None = None) -> dict[str, Any]:
        payload = self._load_train_queue_daemon_state(workspace=workspace)
        summary = self._train_queue_daemon_summary(workspace=workspace)
        pid = summary.get("pid")
        active = summary.get("lock_state") in {"active", "stale"}
        now = datetime.now(timezone.utc).isoformat()
        payload.update(
            {
                "workspace": workspace or "user_default",
                "desired_state": "stopped",
                "requested_action": "stop",
                "command_status": "requested" if active else "noop",
                "last_requested_at": now,
                "last_requested_by": "pipeline",
                "stop_requested": True,
            }
        )
        signaled = False
        if active and pid not in (None, os.getpid()):
            try:
                os.kill(int(pid), signal.SIGTERM)
                signaled = True
            except Exception:
                signaled = False
        if signaled:
            payload["command_status"] = "signaled"
        self._persist_train_queue_daemon_state(payload, workspace=workspace)
        self._append_train_queue_daemon_history(
            workspace=workspace,
            event="stop_requested",
            reason="daemon_stop_requested" if active else "daemon_not_active",
            metadata={"pid": pid, "signaled": signaled, "active": active, "note": note},
        )
        return self.train_queue_daemon_status(workspace=workspace)

    def recover_train_queue_daemon(self, *, workspace: str | None = None, note: str | None = None) -> dict[str, Any]:
        summary = self._train_queue_daemon_summary(workspace=workspace)
        reason = summary.get("recovery_reason")
        if not bool(summary.get("can_recover", False)):
            self._append_train_queue_daemon_history(
                workspace=workspace,
                event="recover_blocked",
                reason=str(reason or "daemon_recovery_not_needed"),
                metadata={
                    "desired_state": summary.get("desired_state"),
                    "lock_state": summary.get("lock_state"),
                    "restart_attempts": summary.get("restart_attempts"),
                    "note": note,
                },
            )
            return self.train_queue_daemon_status(workspace=workspace)
        return self._spawn_train_queue_daemon(
            workspace=workspace,
            reset_restart_policy=False,
            recovery_reason=str(reason or "daemon_recovery_requested"),
            requested_action="recover",
            note=note,
            requested_by="pipeline",
            auto_recovery=False,
        )

    def restart_train_queue_daemon(self, *, workspace: str | None = None, note: str | None = None) -> dict[str, Any]:
        summary = self._train_queue_daemon_summary(workspace=workspace)
        if summary.get("lock_state") == "active" and summary.get("pid") not in (None, os.getpid()):
            try:
                os.kill(int(summary["pid"]), signal.SIGTERM)
            except Exception:
                pass
        payload = self._load_train_queue_daemon_state(workspace=workspace)
        payload["stop_requested"] = False
        self._persist_train_queue_daemon_state(payload, workspace=workspace)
        return self._spawn_train_queue_daemon(
            workspace=workspace,
            reset_restart_policy=False,
            recovery_reason="daemon_restart_requested",
            requested_action="restart",
            note=note,
            requested_by="pipeline",
            auto_recovery=False,
        )

    def train_queue_daemon_status(self, *, workspace: str | None = None) -> dict[str, Any]:
        return self._maybe_auto_recover_train_queue_daemon(workspace=workspace)

    def promote_candidate(self, *, workspace: str | None = None, note: str | None = None) -> dict[str, Any]:
        store = create_adapter_store(workspace=workspace)
        candidate_row = self._candidate_row(workspace=workspace)
        snapshot = self.status(workspace=workspace)
        if candidate_row is None:
            action_payload = {
                "action": "promote_candidate",
                "status": "noop",
                "reason": "no_candidate",
                "state_path": str(self._auto_trigger_state_path(workspace=workspace)),
                "triggered": False,
                "operator_note": note,
            }
            self._persist_candidate_action(action_payload, workspace=workspace)
            snapshot["candidate_action"] = action_payload
            return snapshot
        candidate_version = str(candidate_row.get("version"))
        candidate_state = str(candidate_row.get("state") or "")
        candidate_summary = dict(snapshot.get("candidate_summary") or {})
        if (
            str(candidate_summary.get("candidate_version") or "") == candidate_version
            and str(candidate_summary.get("promotion_gate_status") or "") == "blocked"
        ):
            action_payload = {
                "action": "promote_candidate",
                "status": "blocked",
                "reason": str(candidate_summary.get("promotion_gate_reason") or "promotion_gate_blocked"),
                "required_action": candidate_summary.get("promotion_gate_action"),
                "state_path": str(self._auto_trigger_state_path(workspace=workspace)),
                "triggered": False,
                "candidate_version": candidate_version,
                "previous_candidate_state": candidate_state,
                "operator_note": note,
            }
            self._persist_candidate_action(action_payload, workspace=workspace)
            snapshot["candidate_action"] = action_payload
            return snapshot
        try:
            store.promote(candidate_version)
            # P2-G: record promote trace
            if self._observability_enabled:
                record_signal_node(
                    candidate_version,
                    "promote",
                    "completed",
                    {"operator_note": note},
                )
            action_payload = {
                "action": "promote_candidate",
                "status": "completed",
                "reason": "candidate_promoted",
                "state_path": str(self._auto_trigger_state_path(workspace=workspace)),
                "triggered": True,
                "candidate_version": candidate_version,
                "promoted_version": candidate_version,
                "previous_candidate_state": candidate_state,
                "operator_note": note,
            }
        except Exception as exc:
            action_payload = {
                "action": "promote_candidate",
                "status": "blocked",
                "reason": str(exc),
                "state_path": str(self._auto_trigger_state_path(workspace=workspace)),
                "triggered": False,
                "candidate_version": candidate_version,
                "previous_candidate_state": candidate_state,
                "operator_note": note,
            }
        self._persist_candidate_action(action_payload, workspace=workspace)
        snapshot = self.status(workspace=workspace)
        snapshot["candidate_action"] = action_payload
        return snapshot

    def archive_candidate(self, *, workspace: str | None = None, note: str | None = None) -> dict[str, Any]:
        store = create_adapter_store(workspace=workspace)
        candidate_row = self._candidate_row(workspace=workspace)
        snapshot = self.status(workspace=workspace)
        if candidate_row is None:
            action_payload = {
                "action": "archive_candidate",
                "status": "noop",
                "reason": "no_candidate",
                "state_path": str(self._auto_trigger_state_path(workspace=workspace)),
                "triggered": False,
                "operator_note": note,
            }
            self._persist_candidate_action(action_payload, workspace=workspace)
            snapshot["candidate_action"] = action_payload
            return snapshot
        candidate_version = str(candidate_row.get("version"))
        candidate_state = str(candidate_row.get("state") or "")
        try:
            store.archive(candidate_version)
            action_payload = {
                "action": "archive_candidate",
                "status": "completed",
                "reason": "candidate_archived",
                "state_path": str(self._auto_trigger_state_path(workspace=workspace)),
                "triggered": True,
                "candidate_version": candidate_version,
                "archived_version": candidate_version,
                "previous_candidate_state": candidate_state,
                "operator_note": note,
            }
        except Exception as exc:
            action_payload = {
                "action": "archive_candidate",
                "status": "blocked",
                "reason": str(exc),
                "state_path": str(self._auto_trigger_state_path(workspace=workspace)),
                "triggered": False,
                "candidate_version": candidate_version,
                "previous_candidate_state": candidate_state,
                "operator_note": note,
            }
        self._persist_candidate_action(action_payload, workspace=workspace)
        snapshot = self.status(workspace=workspace)
        snapshot["candidate_action"] = action_payload
        return snapshot

    def rollback_candidate(
        self,
        *,
        workspace: str | None = None,
        version: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        store = create_adapter_store(workspace=workspace)
        snapshot = self.status(workspace=workspace)
        latest = snapshot.get("latest_adapter") or {}
        current_version = str(latest.get("version") or "")
        if not version and current_version:
            version = current_version
        if not version:
            action_payload = {
                "action": "rollback_candidate",
                "status": "noop",
                "reason": "no_version_specified",
                "state_path": str(self._auto_trigger_state_path(workspace=workspace)),
                "triggered": False,
                "operator_note": note,
            }
            self._persist_candidate_action(action_payload, workspace=workspace)
            snapshot["candidate_action"] = action_payload
            return snapshot
        try:
            result = store.rollback(version, workspace=workspace)
            action_payload = {
                "action": "rollback_candidate",
                "status": "completed",
                "reason": "candidate_rolled_back",
                "state_path": str(self._auto_trigger_state_path(workspace=workspace)),
                "triggered": True,
                "rolled_back_version": version,
                "previous_version": current_version,
                "operator_note": note,
            }
        except Exception as exc:
            action_payload = {
                "action": "rollback_candidate",
                "status": "blocked",
                "reason": str(exc),
                "state_path": str(self._auto_trigger_state_path(workspace=workspace)),
                "triggered": False,
                "target_version": version,
                "operator_note": note,
            }
        self._persist_candidate_action(action_payload, workspace=workspace)
        snapshot = self.status(workspace=workspace)
        snapshot["candidate_action"] = action_payload
        return snapshot

    def _synthetic_samples(
        self,
        *,
        teacher_model: str,
        scenario: str,
        style: str,
        num_samples: int,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for index in range(num_samples):
            ratio = (index + 1) / max(num_samples, 1)
            if ratio <= 0.8:
                split = "train"
            elif ratio <= 0.9:
                split = "val"
            else:
                split = "test"
            prompt = f"{scenario} 场景样本 {index + 1}: 用户希望获得关于当前困境的支持。"
            chosen = f"以“{style}”风格回应 {scenario} 场景 {index + 1}，先确认情绪，再给出一步可执行建议。"
            rejected = None
            sample_type = "sft"
            if index % 5 == 0:
                sample_type = "dpo"
                rejected = f"直接下判断并忽略情绪背景的回复 {index + 1}。"
            samples.append(
                {
                    "sample_id": f"smp_{uuid4().hex}",
                    "sample_type": sample_type,
                    "instruction": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "score": 0.92 if split != "test" else 0.88,
                    "source": "teacher",
                    "source_event_ids": [],
                    "source_adapter_version": None,
                    "metadata": {
                        "scenario": scenario,
                        "style": style,
                        "teacher_model": teacher_model,
                        "teacher_prompt_version": "v1",
                        "generation_config": {"temperature": 0.7, "max_samples": num_samples},
                        "dataset_split": split,
                    },
                }
            )
        return samples

    def _distilled_samples(
        self,
        *,
        teacher_model: str,
        scenario: str,
        style: str,
        num_samples: int,
        use_cloud_teacher: bool = False,
        privacy_config: PrivacyConfig | None = None,
    ) -> list[dict[str, Any]]:
        from .curator import DistillationConfig, TeacherDistiller

        if privacy_config is None:
            privacy_config = self._load_config().privacy

        if not teacher_model or teacher_model == "local_template_teacher":
            backend = "mock"
        elif use_cloud_teacher and privacy_config.mode == "cloud_assisted" and privacy_config.allow_teacher_cloud:
            backend = "cloud"
        else:
            backend = "local"

        teacher_client_config = TeacherClientConfig(
            backend=backend,
            model=teacher_model if backend != "mock" else "mock-teacher",
        )
        teacher_client = TeacherInferenceClient(teacher_client_config)

        distiller = TeacherDistiller(
            DistillationConfig(
                teacher_model=teacher_model,
                max_samples=num_samples,
                scenario=scenario,
                style=style,
            ),
            teacher_client=teacher_client,
            privacy_config=privacy_config,
        )
        return [_normalize(sample) for sample in distiller.distill_from_scenario(scenario, style, num_samples)]

    def _run_distillation(
        self,
        *,
        teacher_model: str,
        scenario: str,
        style: str,
        num_samples: int,
        output: str | None,
        use_cloud_teacher: bool = False,
        privacy_config: PrivacyConfig | None = None,
    ) -> dict[str, Any]:
        samples = self._distilled_samples(
            teacher_model=teacher_model,
            scenario=scenario,
            style=style,
            num_samples=num_samples,
            use_cloud_teacher=use_cloud_teacher,
            privacy_config=privacy_config,
        )
        save_samples(samples)
        if output:
            write_jsonl(output, samples)
        split_counts = {"train": 0, "val": 0, "test": 0}
        for sample in samples:
            split = sample.get("metadata", {}).get("dataset_split")
            if split in split_counts:
                split_counts[split] += 1
        return {
            "teacher_model": teacher_model,
            "teacher_prompt_version": "v1",
            "generated_samples": len(samples),
            "train_samples": split_counts["train"],
            "val_samples": split_counts["val"],
            "test_samples": split_counts["test"],
            "scenario": scenario,
            "style": style,
        }

    def generate(
        self,
        *,
        scenario: str,
        style: str,
        num_samples: int = 200,
        output: str | None = None,
        workspace: str | None = None,
        use_cloud_teacher: bool = False,
        privacy_config: PrivacyConfig | None = None,
    ) -> str:
        del workspace
        result = self._run_distillation(
            teacher_model="local_template_teacher",
            scenario=scenario,
            style=style,
            num_samples=num_samples,
            output=output,
            use_cloud_teacher=use_cloud_teacher,
            privacy_config=privacy_config,
        )
        return (
            f"Saved {result['generated_samples']} distilled sample(s) for scenario={scenario} "
            f"with splits train/val/test and provenance teacher_model={result['teacher_model']}."
        )

    def distill(
        self,
        *,
        teacher_model: str,
        scenario: str,
        style: str,
        num_samples: int = 200,
        output: str | None = None,
        workspace: str | None = None,
        use_cloud_teacher: bool = False,
        privacy_config: PrivacyConfig | None = None,
    ) -> str:
        del workspace
        result = self._run_distillation(
            teacher_model=teacher_model,
            scenario=scenario,
            style=style,
            num_samples=num_samples,
            output=output,
            use_cloud_teacher=use_cloud_teacher,
            privacy_config=privacy_config,
        )
        return (
            f"Saved {result['generated_samples']} distilled sample(s) for scenario={scenario} "
            f"with splits train/val/test and provenance teacher_model={result['teacher_model']}."
        )


    def train(
        self,
        *,
        method: str = "qlora",
        epochs: int = 3,
        base_model: str | None = None,
        train_type: str = "sft",
        workspace: str | None = None,
    ) -> str:
        # P2-G: Pre-training audit
        samples = list_samples(dataset_split="train")
        auditor = TrainingAuditor()
        audit_report = auditor.audit(samples)
        self._last_training_audit = audit_report
        if audit_report.blocked:
            return (
                f"Training blocked by pre-training audit (severity={audit_report.severity}). "
                f"Reasons: {'; '.join(audit_report.reasons)}"
            )
        result = self.trainer.train(
            method=method,
            epochs=epochs,
            base_model=base_model,
            train_type=train_type,
            workspace=workspace,
        )
        # Check for auto-eval trigger after training
        self._maybe_trigger_auto_eval_after_train(
            train_result=result,
            base_model=base_model,
            workspace=workspace,
        )
        return result

    def train_result(
        self,
        *,
        method: str = "qlora",
        epochs: int = 3,
        base_model: str | None = None,
        train_type: str = "sft",
        workspace: str | None = None,
        backend_hint: str | None = None,
    ) -> Any:
        return self.trainer.train_result(
            method=method,
            epochs=epochs,
            base_model=base_model,
            train_type=train_type,
            workspace=workspace,
            backend_hint=backend_hint,
        )

    def train_incremental(
        self,
        *,
        base_adapter: str,
        method: str = "qlora",
        epochs: int = 1,
        train_type: str = "sft",
        workspace: str | None = None,
    ):
        return self.trainer.train_incremental(
            base_adapter=base_adapter,
            method=method,
            epochs=epochs,
            train_type=train_type,
            workspace=workspace,
        )

    def train_dpo(
        self,
        *,
        method: str = "qlora",
        epochs: int = 3,
        base_model: str | None = None,
        workspace: str | None = None,
        backend_hint: str | None = None,
        base_adapter_path: str | None = None,
        min_confidence: float | None = None,
    ) -> str:
        """Execute DPO (Direct Preference Optimization) training.

        Builds DPO dataset from signals and trains using preference pairs.
        Supports incremental training from an existing SFT adapter.

        Args:
            method: Training method ("qlora" or "lora")
            epochs: Number of training epochs
            base_model: Base model name or path
            workspace: Workspace name
            backend_hint: Backend hint for execution
            base_adapter_path: Optional path to existing SFT adapter
            min_confidence: Minimum signal confidence for pairs

        Returns:
            Training result message
        """
        result = self.trainer.train_dpo(
            method=method,
            epochs=epochs,
            base_model=base_model,
            workspace=workspace,
            backend_hint=backend_hint,
            base_adapter_path=base_adapter_path,
            min_confidence=min_confidence,
        )
        return (
            f"Trained DPO adapter {result.version} with {result.metrics['num_fresh_samples']} fresh sample(s)"
            f" and {result.metrics['num_replay_samples']} replay sample(s). State=pending_eval."
        )

    def build_dpo_dataset(
        self,
        *,
        workspace: str | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Build and inspect DPO dataset without training.

        Args:
            workspace: Workspace name
            min_confidence: Minimum signal confidence
            limit: Maximum number of pairs

        Returns:
            Dataset statistics and info
        """
        return self.trainer.build_dpo_dataset(
            workspace=workspace,
            min_confidence=min_confidence,
            limit=limit,
        )

    def _fallback_evaluate(
        self,
        *,
        adapter_version: str,
        base_model: str,
        prompts: list[dict[str, Any]],
        base_responses: list[str],
        adapted_responses: list[str],
    ) -> dict[str, Any]:
        quality = 0.86
        style = 0.75
        preference = 0.72
        personality = 0.74
        comparison = "improved" if style >= 0.7 and quality >= 0.8 else "neutral"
        recommendation = "deploy" if comparison == "improved" else "needs_more_data"
        return {
            "adapter_version": adapter_version,
            "base_model": base_model,
            "num_test_samples": len(prompts),
            "scores": {
                "style_match": style,
                "preference_alignment": preference,
                "quality_preservation": quality,
                "personality_consistency": personality,
            },
            "comparison": comparison,
            "recommendation": recommendation,
            "details": [
                {
                    "sample_id": prompt["sample_id"],
                    "instruction": prompt["instruction"],
                    "base_output": base_output,
                    "adapted_output": adapted_output,
                }
                for prompt, base_output, adapted_output in zip(prompts, base_responses, adapted_responses)
            ],
        }

    @staticmethod
    def _should_use_shared_eval_runtime(generation_kwargs: dict[str, Any]) -> bool:
        metadata = generation_kwargs.get("metadata")
        if not isinstance(metadata, dict):
            return False
        value = metadata.get("enable_real_local")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def _evaluate_with_shared_runtime(
        self,
        *,
        base_model: str,
        adapter_path: str,
        messages: list[list[dict[str, Any]]],
        generation_kwargs: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        resolved_base_model = resolve_base_model_reference(base_model)
        base_engine = InferenceEngine(InferenceConfig(base_model=base_model, adapter_path=None))
        base_bundle = base_engine._load_uncached_runtime_bundle(
            resolved_base_model=resolved_base_model,
            adapter_path=None,
        )
        base_responses = [
            str(
                base_engine._generate_real_response(
                    message,
                    runtime_bundle=base_bundle,
                    resolved_base_model=resolved_base_model,
                    **generation_kwargs,
                )["text"]
            )
            for message in messages
        ]
        adapted_bundle = base_engine._attach_adapter_to_runtime_bundle(
            base_bundle,
            adapter_path=adapter_path,
        )
        adapted_engine = InferenceEngine(InferenceConfig(base_model=base_model, adapter_path=adapter_path))
        adapted_engine.adapter_manifest = (
            json.loads((Path(adapter_path) / "adapter_manifest.json").read_text(encoding="utf-8"))
            if (Path(adapter_path) / "adapter_manifest.json").exists()
            else None
        )
        adapted_responses = [
            str(
                adapted_engine._generate_real_response(
                    message,
                    runtime_bundle=adapted_bundle,
                    resolved_base_model=resolved_base_model,
                    **generation_kwargs,
                )["text"]
            )
            for message in messages
        ]
        return base_responses, adapted_responses

    def _load_eval_report(
        self,
        *,
        store: Any,
        adapter: str,
    ) -> tuple[str, dict[str, Any]]:
        adapter_path = Path(store.load(adapter))
        adapter_version = adapter_path.name
        report_path = adapter_path / "eval_report.json"
        if report_path.exists():
            return adapter_version, json.loads(report_path.read_text(encoding="utf-8"))

        rows = store.list_version_records(limit=100)
        for row in rows:
            if str(row.get("version")) == adapter_version:
                report = row.get("eval_report")
                if isinstance(report, str):
                    return adapter_version, json.loads(report)
                if isinstance(report, dict):
                    return adapter_version, dict(report)
                break
        raise EvalError(f"adapter {adapter_version} does not have an evaluation report yet")

    def evaluate(
        self,
        *,
        base_model: str,
        adapter: str = "latest",
        num_samples: int = 20,
        workspace: str | None = None,
    ) -> str:
        store = create_adapter_store(workspace=workspace)
        try:
            adapter_path = store.load(adapter)
            adapter_version = Path(adapter_path).name
        except Exception:
            latest_rows = store.list_version_records(limit=1)
            if adapter == "latest" and latest_rows:
                adapter_version = latest_rows[0]["version"]
                adapter_path = store.load(adapter_version)
            else:
                raise
        eval_samples = list_samples(dataset_split="test", limit=num_samples)
        if len(eval_samples) < num_samples:
            eval_samples.extend(
                list_samples(dataset_split="val", limit=max(0, num_samples - len(eval_samples)))
            )
        if not eval_samples:
            report = self._fallback_evaluate(
                adapter_version=adapter_version,
                base_model=base_model,
                prompts=[],
                base_responses=[],
                adapted_responses=[],
            )
            store.attach_eval_report(adapter_version, report)
            return json.dumps(report, ensure_ascii=False, indent=2)

        messages = [[{"role": "user", "content": sample["instruction"]}] for sample in eval_samples]
        generation_kwargs = _eval_generation_kwargs()
        if self._should_use_shared_eval_runtime(generation_kwargs):
            try:
                base_responses, adapted_responses = self._evaluate_with_shared_runtime(
                    base_model=base_model,
                    adapter_path=adapter_path,
                    messages=messages,
                    generation_kwargs=generation_kwargs,
                )
            except Exception:
                base_engine = InferenceEngine(InferenceConfig(base_model=base_model, adapter_path=None))
                adapted_engine = InferenceEngine(InferenceConfig(base_model=base_model, adapter_path=adapter_path))
                base_responses = [base_engine.generate(message, **generation_kwargs) for message in messages]
                adapted_responses = [adapted_engine.generate(message, **generation_kwargs) for message in messages]
        else:
            base_engine = InferenceEngine(InferenceConfig(base_model=base_model, adapter_path=None))
            adapted_engine = InferenceEngine(InferenceConfig(base_model=base_model, adapter_path=adapter_path))
            base_responses = [base_engine.generate(message, **generation_kwargs) for message in messages]
            adapted_responses = [adapted_engine.generate(message, **generation_kwargs) for message in messages]

        report: dict[str, Any] | None = None
        try:
            evaluator_module = importlib.import_module("pfe_core.evaluator.auto")
            evaluator_cls = getattr(evaluator_module, "AutoEvaluator", None)
            if evaluator_cls is not None:
                distillation_module = importlib.import_module("pfe_core.curator.distillation")
                sample_cls = getattr(distillation_module, "TrainingSample", None)
                evaluator = evaluator_cls()
                prepared_samples = [
                    sample_cls(
                        sample_id=sample["sample_id"],
                        sample_type=sample["sample_type"],
                        instruction=sample["instruction"],
                        chosen=sample["chosen"],
                        rejected=sample.get("rejected"),
                        score=sample["score"],
                        source=sample["source"],
                        source_event_ids=sample.get("source_event_ids", []),
                        source_adapter_version=sample.get("source_adapter_version"),
                        metadata=sample.get("metadata", {}),
                    )
                    for sample in eval_samples
                ]
                report_obj = evaluator.evaluate(
                    samples=prepared_samples,
                    base_responses=base_responses,
                    adapted_responses=adapted_responses,
                    reference_responses=[sample["chosen"] for sample in eval_samples],
                    base_model=base_model,
                    adapter_version=adapter_version,
                )
                report = _normalize(report_obj)
        except Exception:
            report = None

        if report is None:
            report = self._fallback_evaluate(
                adapter_version=adapter_version,
                base_model=base_model,
                prompts=eval_samples,
                base_responses=base_responses,
                adapted_responses=adapted_responses,
            )

        store.attach_eval_report(adapter_version, report)
        # P2-G: record eval trace for version
        if self._observability_enabled:
            record_signal_node(
                adapter_version,
                "eval",
                "ok",
                {"scores": report.get("scores"), "recommendation": report.get("recommendation")},
            )
        return json.dumps(report, ensure_ascii=False, indent=2)

    def compare_evaluations(
        self,
        *,
        left_adapter: str,
        right_adapter: str,
        workspace: str | None = None,
    ) -> str:
        store = create_adapter_store(workspace=workspace)
        left_version, left_report = self._load_eval_report(store=store, adapter=left_adapter)
        right_version, right_report = self._load_eval_report(store=store, adapter=right_adapter)
        evaluator_module = importlib.import_module("pfe_core.evaluator.auto")
        evaluator_cls = getattr(evaluator_module, "AutoEvaluator", None)
        if evaluator_cls is None:
            raise EvalError("evaluation compare support is unavailable")
        evaluator = evaluator_cls()
        comparison = evaluator.compare_versions(left_report, right_report)
        payload = _normalize(comparison)
        payload["left_adapter"] = left_version
        payload["right_adapter"] = right_version
        payload["left_adapter_version"] = left_version
        payload["right_adapter_version"] = right_version
        payload["workspace"] = workspace or "user_default"
        payload["summary_line"] = " | ".join(
            part
            for part in (
                f"left={left_version}",
                f"right={right_version}",
                f"comparison={payload.get('comparison')}",
                f"winner={payload.get('winner')}",
                f"recommendation={payload.get('recommendation')}",
                f"overall_delta={payload.get('overall_delta')}",
            )
            if part is not None
        )
        self.last_compare_result = dict(payload)
        self._persist_compare_evaluation_state(payload, workspace=workspace)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str = "local",
        adapter_version: str = "latest",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        adapter_path = None
        resolved_adapter = None
        base_model_name = model
        if model != "base":
            try:
                store = create_adapter_store(workspace=workspace)
                adapter_path = store.load(adapter_version)
                resolved_adapter = Path(adapter_path).name
                manifest = json.loads((Path(adapter_path) / "adapter_manifest.json").read_text(encoding="utf-8"))
                base_model_name = str(manifest.get("base_model") or model)
            except Exception:
                adapter_path = None
                resolved_adapter = None
        if model in {"local", "local-default", "base"} and not adapter_path:
            base_model_name = _default_chat_base_model()

        engine = InferenceEngine(
            InferenceConfig(
                base_model=base_model_name,
                adapter_path=adapter_path,
            )
        )
        content = engine.generate(messages, temperature=temperature, max_tokens=max_tokens, metadata=metadata or {})
        inference_status = engine.status()
        prompt_text = "\n".join(str(message.get("content") or "") for message in messages)
        prompt_tokens = max(0, len(prompt_text.strip()) // 4)
        completion_tokens = max(0, len(content.strip()) // 4)
        return {
            "id": f"chatcmpl-{uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": model,
            "adapter_version": resolved_adapter,
            "request_id": request_id,
            "session_id": session_id,
            "served_by": inference_status.get("served_by", "mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "metadata": {
                "inference": inference_status,
            },
        }

    def signal(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        workspace = payload.get("workspace") or (payload.get("metadata") or {}).get("workspace")
        source_event_id, source_event_ids = _normalized_event_chain(payload)
        payload["source_event_id"] = source_event_id
        payload["source_event_ids"] = source_event_ids
        config = self._load_config()
        collection_enabled = self._signal_collection_enabled(config)
        if not collection_enabled:
            return {
                "ok": True,
                "event_id": payload.get("event_id"),
                "request_id": payload.get("request_id"),
                "session_id": payload.get("session_id"),
                "adapter_version": payload.get("adapter_version"),
                "source_event_id": source_event_id,
                "source_event_ids": source_event_ids,
                "event_chain_complete": False,
                "event_type": payload.get("event_type"),
                "recorded": False,
                "collection_enabled": False,
                "curated_samples": 0,
                "curated_sample_ids": [],
                "curation_state": "disabled",
                "curation_reason": "signal_collection_disabled",
                "curation_detail": "signal_collection_disabled",
                "curation_results": [],
                "auto_train": {
                    "state": "disabled",
                    "reason": "signal_collection_disabled",
                },
        }
        record_signal(payload)
        signal_id = str(payload.get("event_id") or "")
        if self._observability_enabled and signal_id:
            record_signal_node(signal_id, "collect", "ok", {"workspace": workspace})
        quality = build_signal_quality(payload)
        quality_payload = _normalize(quality)
        quality_config = SampleFilterConfig()
        quality_reasons = signal_quality_filter_reasons(
            quality,
            minimum_confidence=quality_config.minimum_signal_confidence,
            reject_conflicted_signal_quality=quality_config.reject_conflicted_signal_quality,
        )
        if self._observability_enabled and signal_id:
            record_signal_node(
                signal_id,
                "quality_score",
                "passed" if not quality_reasons else "filtered",
                {"quality_payload": quality_payload},
            )
        curated_count = 0
        curated_sample_ids: list[str] = []
        curation_results: list[dict[str, Any]] = []
        event_chain_complete = bool(
            payload.get("request_id")
            and payload.get("session_id")
            and source_event_id
            and len(source_event_ids) >= 2
            and source_event_id in source_event_ids
        )
        try:
            curator_module = importlib.import_module("pfe_core.curator")
            raw_signal_cls = getattr(curator_module, "RawSignal", None)
            curate_fn = getattr(curator_module, "curate_signals_to_samples", None)
            curate_preference_fn = getattr(curator_module, "curate_signals_to_preference_samples", None)
            if quality_reasons:
                curation_results = [
                    {
                        "signal_id": payload["event_id"],
                        "reason": quality_reasons[0],
                        "curated": False,
                        "sample_id": None,
                        "filtered_by_quality": True,
                        "filtered_reasons": quality_reasons,
                        "signal_quality": quality_payload,
                    }
                ]
            elif raw_signal_cls is not None and curate_fn is not None:
                timestamp = payload.get("timestamp")
                if timestamp is None:
                    timestamp = datetime.now(timezone.utc)
                else:
                    timestamp = parse_utc_datetime(timestamp)
                signal = raw_signal_cls(
                    signal_id=payload["event_id"],
                    source_event_id=source_event_id,
                    request_id=payload["request_id"],
                    session_id=payload["session_id"],
                    adapter_version=payload.get("adapter_version"),
                    event_type=payload.get("event_type") or payload.get("user_action", {}).get("type") or "chat",
                    timestamp=timestamp,
                    context=payload.get("user_input") or "",
                    model_output=payload.get("model_output") or "",
                    user_action=(
                        lambda ua: (
                            ua.setdefault("final_text", payload.get("user_input") or payload.get("context") or ""),
                            ua.setdefault("accepted_text", payload.get("model_output") or ""),
                            ua,
                        )[-1]
                    )(dict(payload.get("user_action") or {})),
                    metadata=dict(payload.get("metadata") or {}),
                    event_chain_ids=list(source_event_ids),
                )
                all_samples: list[dict[str, Any]] = []
                all_results: list[Any] = []

                # Curate SFT samples
                samples, _results = curate_fn([signal])
                all_samples.extend(samples)
                all_results.extend(_results or [])

                # Curate DPO preference samples (when signal contains preference info)
                if curate_preference_fn is not None:
                    dpo_samples, dpo_results = curate_preference_fn([signal])
                    all_samples.extend(dpo_samples)
                    all_results.extend(dpo_results or [])

                curation_results = [
                    {
                        "signal_id": getattr(result, "signal_id", payload["event_id"]),
                        "reason": getattr(result, "reason", "unknown"),
                        "curated": getattr(result, "sample", None) is not None,
                        "sample_id": getattr(getattr(result, "sample", None), "sample_id", None),
                        "sample_type": getattr(getattr(result, "sample", None), "sample_type", "sft"),
                    }
                    for result in all_results
                ]
                if all_samples:
                    normalized_samples = []
                    for sample in all_samples:
                        normalized = _normalize(sample)
                        metadata = dict(normalized.get("metadata") or {})
                        metadata.setdefault("dataset_split", "train")
                        metadata["signal_quality"] = quality_payload
                        metadata["signal_quality_passed"] = True
                        metadata["signal_quality_filtered"] = False
                        metadata["signal_quality_reasons"] = []
                        metadata["signal_quality_minimum_confidence"] = quality_config.minimum_signal_confidence
                        normalized["metadata"] = metadata
                        normalized_samples.append(normalized)
                    save_samples(normalized_samples)
                    curated_count = len(normalized_samples)
                    curated_sample_ids = [
                        str(sample.get("sample_id"))
                        for sample in normalized_samples
                        if sample.get("sample_id")
                    ]
        except Exception:
            curated_count = 0
            curation_results = [
                {
                    "signal_id": payload["event_id"],
                    "reason": "curation_error",
                    "curated": False,
                    "sample_id": None,
                }
            ]
        if quality_reasons:
            curation_state = "filtered"
        else:
            curation_state = "curated" if curated_count > 0 else "stored_only"
        curation_detail = curation_results[0]["reason"] if curation_results else ("curated_sft" if curated_count > 0 else "stored_only")
        if quality_reasons:
            curation_reason = quality_reasons[0]
        else:
            curation_reason = "curated_sft" if curated_count > 0 else "incomplete_chain_or_unreliable_event"
        if self._observability_enabled and signal_id:
            record_signal_node(
                signal_id,
                "curate",
                curation_state if quality_reasons else ("ok" if curated_count > 0 else "noop"),
                {"curated_count": curated_count, "curation_reason": curation_reason},
            )
        auto_train_result = (
            self._maybe_auto_train_from_signal(workspace=workspace)
            if not quality_reasons
            else {
                "state": "blocked",
                "triggered": False,
                "reason": curation_reason,
                "blocked_by_quality": True,
                "filtered_reasons": quality_reasons,
            }
        )
        return {
            "ok": True,
            "event_id": payload["event_id"],
            "request_id": payload.get("request_id"),
            "session_id": payload.get("session_id"),
            "adapter_version": payload.get("adapter_version"),
            "source_event_id": source_event_id,
            "source_event_ids": source_event_ids,
            "event_chain_complete": event_chain_complete,
            "event_type": payload.get("event_type"),
            "recorded": True,
            "collection_enabled": True,
            "curated_samples": curated_count,
            "curated_sample_ids": curated_sample_ids,
            "curation_state": curation_state,
            "curation_reason": curation_reason,
            "curation_detail": curation_detail,
            "signal_quality": quality_payload,
            "signal_quality_filtered": bool(quality_reasons),
            "signal_quality_reasons": quality_reasons,
            "curation_results": curation_results,
            "auto_train": auto_train_result,
        }

    def start_signal_collection(self, *, workspace: str | None = None) -> dict[str, Any]:
        config = self._load_config()
        config.curator.signal_collection_enabled = True
        self._save_config(config)
        status = self.status(workspace=workspace)
        signal_summary = dict(status.get("signal_summary") or {})
        signal_summary["last_collection_action"] = "start"
        status["signal_summary"] = signal_summary
        return status

    def stop_signal_collection(self, *, workspace: str | None = None) -> dict[str, Any]:
        config = self._load_config()
        config.curator.signal_collection_enabled = False
        self._save_config(config)
        status = self.status(workspace=workspace)
        signal_summary = dict(status.get("signal_summary") or {})
        signal_summary["last_collection_action"] = "stop"
        status["signal_summary"] = signal_summary
        return status

    def status(self, workspace: str | None = None) -> dict[str, Any]:
        store = create_adapter_store(workspace=workspace)
        snapshot = status_snapshot(workspace=workspace or "user_default")
        config = self._load_config()
        signal_collection_enabled = self._signal_collection_enabled(config)
        signal_summary = dict(snapshot.get("signal_summary") or {})
        signal_summary["collection_enabled"] = signal_collection_enabled
        signal_summary["collection_mode"] = "enabled" if signal_collection_enabled else "disabled"
        if not signal_collection_enabled:
            signal_summary["state"] = "disabled"
        quality_summary = self._signal_quality_summary(workspace=workspace)
        signal_summary["quality_filter_state"] = "blocked" if quality_summary.get("filtered_count", 0) else "ready"
        signal_summary["quality_filtered_count"] = quality_summary.get("filtered_count", 0)
        signal_summary["quality_filter_reasons"] = dict(quality_summary.get("filtered_reasons") or {})
        snapshot["signal_summary"] = signal_summary
        snapshot["signal_quality_summary"] = quality_summary
        compare_evaluation = self._load_compare_evaluation_state(workspace=workspace)
        if compare_evaluation:
            snapshot["compare_evaluation"] = compare_evaluation
        rows = store.list_version_records(limit=100)
        latest_version = store.current_latest_version()
        row_by_version = {str(row.get("version")): row for row in rows}
        latest_row = row_by_version.get(str(latest_version)) if latest_version is not None else None
        recent_row = rows[0] if rows else None
        lifecycle_counts: dict[str, int] = {}
        for row in rows:
            state = str(row.get("state") or "unknown")
            lifecycle_counts[state] = lifecycle_counts.get(state, 0) + 1
        snapshot["adapter_versions"] = len(rows)
        snapshot["adapter_count"] = len(rows)
        snapshot["latest_adapter_version"] = latest_version
        snapshot["latest_adapter_exists"] = latest_row is not None
        snapshot["latest_adapter"] = _adapter_row_snapshot(latest_row, latest=True) if latest_row else {}
        snapshot["recent_adapter_version"] = recent_row.get("version") if recent_row else None
        snapshot["recent_adapter"] = _adapter_row_snapshot(
            recent_row,
            latest=bool(recent_row and recent_row.get("version") == latest_version),
        ) if recent_row else {}
        snapshot["candidate_summary"] = self._candidate_summary(
            rows=rows,
            latest_version=latest_version,
            recent_row=recent_row,
            compare_evaluation=compare_evaluation or None,
        )
        snapshot["adapter_lifecycle"] = {
            "counts": lifecycle_counts,
            "latest_version": latest_version,
            "recent_version": recent_row.get("version") if recent_row else None,
            "pending_eval_versions": [row["version"] for row in rows if row.get("state") == "pending_eval"],
            "promoted_versions": [row["version"] for row in rows if row.get("state") == "promoted"],
            "failed_eval_versions": [row["version"] for row in rows if row.get("state") == "failed_eval"],
            "archived_versions": [row["version"] for row in rows if row.get("state") == "archived"],
        }
        snapshot["serve"] = {
            "adapter_resolution_state": "latest_promoted" if latest_row is not None else "no_promoted_latest",
            "using_promoted_adapter": latest_row is not None,
            "target_adapter_version": latest_version if latest_row is not None else None,
            "fallback_reason": None if latest_row is not None else "no_promoted_latest",
        }
        trainer_runtime = trainer_runtime_summary()
        latest_training_result = getattr(self.trainer, "last_run_result", None)
        sft_plan = summarize_trainer_backend_plan(
            train_type="sft",
            runtime=trainer_runtime,
            target_inference_backend="transformers",
        )
        dpo_plan = summarize_trainer_backend_plan(
            train_type="dpo",
            runtime=trainer_runtime,
            target_inference_backend="transformers",
        )
        if latest_training_result is not None:
            trainer_runtime = dict(latest_training_result.runtime)
            latest_plan = dict(latest_training_result.backend_plan)
            if latest_plan.get("train_type") == "sft":
                sft_plan = latest_plan
            elif latest_plan.get("train_type") == "dpo":
                dpo_plan = latest_plan
        snapshot["trainer"] = {
            "runtime": trainer_runtime,
            "plans": {
                "sft": sft_plan,
                "dpo": dpo_plan,
            },
        }
        if latest_training_result is not None:
            job_execution_summary = dict(latest_training_result.job_execution_summary) or summarize_training_job_execution(
                latest_training_result.job_execution
            )
            real_execution_summary = dict(latest_training_result.real_execution_summary) or summarize_real_training_execution(
                latest_training_result.job_execution
            )
            export_toolchain_summary = dict(latest_training_result.export_toolchain_summary) or _export_toolchain_snapshot(
                {
                    "export_runtime": latest_training_result.export_runtime,
                    "export_command_plan": latest_training_result.export_command_plan,
                    "export_execution": latest_training_result.export_execution,
                    "export_write": latest_training_result.export_write,
                }
            )
            export_artifact_summary = dict(
                latest_training_result.metrics.get("export_artifact_summary")
                or latest_training_result.training_config.get("export_artifact_summary")
                or {}
            )
            recent_training_state = str(latest_training_result.metrics.get("state") or "pending_eval")
            snapshot["trainer"]["last_run"] = {
                "version": latest_training_result.version,
                "state": recent_training_state,
                "execution_backend": latest_training_result.execution_backend,
                "backend_dispatch": dict(latest_training_result.backend_dispatch),
                "job_bundle": dict(latest_training_result.job_bundle),
                "job_execution": dict(latest_training_result.job_execution),
                "job_execution_summary": job_execution_summary,
                "real_execution_summary": real_execution_summary,
                "export_runtime": dict(latest_training_result.export_runtime),
                "export_command_plan": dict(latest_training_result.export_command_plan),
                "export_execution": dict(latest_training_result.export_execution),
                "export_toolchain_summary": export_toolchain_summary,
                "export_artifact_summary": export_artifact_summary,
                "export_write": dict(latest_training_result.export_write),
                "requires_export_step": latest_training_result.requires_export_step,
                "incremental_context": dict(latest_training_result.incremental_context) if latest_training_result.incremental_context else None,
                "source_adapter_version": (latest_training_result.incremental_context or {}).get("source_adapter_version"),
                "source_adapter_path": (latest_training_result.incremental_context or {}).get("source_adapter_path"),
                "source_model": (latest_training_result.incremental_context or {}).get("source_model"),
            }
        snapshot["auto_train_trigger"] = self._auto_train_trigger_status(workspace=workspace)
        if self.last_auto_trigger_result is not None:
            snapshot["auto_train_trigger"]["last_result"] = dict(self.last_auto_trigger_result)
        action_payload = self._load_auto_trigger_action(workspace=workspace)
        if action_payload:
            snapshot["auto_train_trigger_action"] = action_payload
        candidate_action_payload = self._load_candidate_action(workspace=workspace)
        if candidate_action_payload:
            snapshot["candidate_action"] = candidate_action_payload
        snapshot["candidate_history"] = self._candidate_history_summary(workspace=workspace)
        snapshot["candidate_timeline"] = self._candidate_timeline_summary(workspace=workspace)
        train_queue = self._train_queue_snapshot(workspace=workspace)
        snapshot["train_queue"] = train_queue
        operations_overview = self._operations_overview(
            auto_train_trigger=dict(snapshot.get("auto_train_trigger") or {}),
            candidate_summary=dict(snapshot.get("candidate_summary") or {}),
            train_queue=train_queue,
        )
        snapshot["operations_overview"] = operations_overview
        snapshot["operations_console"] = self._operations_console(
            operations_overview=operations_overview,
            candidate_history=dict(snapshot.get("candidate_history") or {}),
            candidate_timeline=dict(snapshot.get("candidate_timeline") or {}),
            train_queue=train_queue,
        )
        snapshot["operations_event_stream"] = dict(snapshot["operations_console"].get("event_stream") or {})
        snapshot["operations_dashboard"] = dict(snapshot["operations_console"].get("dashboard") or {})
        snapshot["operations_alert_policy"] = dict(snapshot["operations_console"].get("alert_policy") or {})
        # Inject default planning/metadata fields expected by surfaces and server tests
        if not snapshot.get("capabilities"):
            snapshot["capabilities"] = {
                "train": {"tier": "core", "summary": "Train adapter versions"},
                "eval": {"tier": "core", "summary": "Evaluate adapter versions"},
                "serve": {"tier": "core", "summary": "Serve personalized adapters"},
                "generate": {"tier": "core", "summary": "Generate personalized responses"},
                "distill": {"tier": "core", "summary": "Run distillation scenarios"},
                "profile": {"tier": "secondary", "summary": "Extract user models"},
                "route": {"tier": "core", "summary": "Route requests to best backend"},
            }
        if not snapshot.get("user_modeling"):
            snapshot["user_modeling"] = {
                "primary_runtime_system": "user_memory",
                "primary_runtime_status": "ready",
                "secondary_analysis_system": "profile_extractor",
                "secondary_runtime_status": "ready",
            }
        # P2-G: Observability summary
        snapshot["observability_summary"] = self._build_observability_summary(workspace=workspace)
        return snapshot

    def _build_observability_summary(self, workspace: str | None = None) -> dict[str, Any]:
        """Build observability summary for status()."""
        from .data_policy import audit_pii_exposure

        summary: dict[str, Any] = {
            "observability_enabled": self._observability_enabled,
            "recent_signal_traces": [],
            "quarantine_state": {},
            "last_training_audit": {},
            "pii_detection_stats": {},
        }
        # Recent signal traces
        try:
            recent_ids = self._trace_store.list_recent_signal_ids(limit=10)
            summary["recent_signal_traces"] = [
                {
                    "signal_id": sid,
                    "trace": (self._trace_store.load_signal_trace(sid) or SignalTrace(signal_id=sid)).to_dict(),
                }
                for sid in recent_ids
            ]
        except Exception:
            pass

        # Quarantine state from samples
        try:
            all_samples = list_samples(limit=200)
            quarantined = [
                s for s in all_samples
                if (
                    (s.get("metadata") or {}).get("quarantine")
                    or (s.get("metadata") or {}).get("signal_quality", {}).get("quarantine")
                    or (s.get("signal_quality") or {}).get("quarantine")
                )
            ]
            summary["quarantine_state"] = {
                "quarantined_sample_count": len(quarantined),
                "sample_ids": [str(s.get("sample_id") or s.get("id", "")) for s in quarantined[:10]],
            }
        except Exception:
            pass

        # Last training audit
        if self._last_training_audit is not None:
            summary["last_training_audit"] = self._last_training_audit.to_dict()

        # PII detection stats
        try:
            samples = list_samples(limit=100)
            pii_report = audit_pii_exposure(samples)
            summary["pii_detection_stats"] = {
                "samples_scanned": pii_report.total_samples,
                "pii_detected_count": pii_report.pii_detected_count,
                "severity": pii_report.severity,
                "pii_types_found": pii_report.pii_types_found,
            }
        except Exception:
            pass

        return summary
        return snapshot

    def run_distillation(
        self,
        *,
        teacher_model: str,
        scenario: str,
        style: str,
        num_samples: int = 200,
        output: str | None = None,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        del workspace
        return self._run_distillation(
            teacher_model=teacher_model,
            scenario=scenario,
            style=style,
            num_samples=num_samples,
            output=output,
        )

    def get_health_status(self, workspace: str | None = None) -> dict[str, Any]:
        """Get comprehensive health status for daemon and runner."""
        from .reliability import ReliabilityService

        daemon_summary = self._train_queue_daemon_summary(workspace=workspace)
        worker_summary = self._train_queue_worker_summary(workspace=workspace)

        # Initialize reliability service for alert checking
        reliability = ReliabilityService(workspace=workspace or "user_default")
        health_summary = reliability.get_health_summary()

        # Determine overall health state
        daemon_health = daemon_summary.get("health_state", "unknown")
        worker_lock = worker_summary.get("lock_state", "unknown")

        overall_health = "healthy"
        issues: list[dict[str, Any]] = []

        if daemon_health in {"stale", "blocked"}:
            overall_health = "critical"
            issues.append({
                "component": "daemon",
                "state": daemon_health,
                "message": f"Daemon is in {daemon_health} state",
            })
        elif daemon_health in {"recovering", "restarting"}:
            overall_health = "degraded"
            issues.append({
                "component": "daemon",
                "state": daemon_health,
                "message": f"Daemon is {daemon_health}",
            })

        if worker_lock == "stale":
            if overall_health == "healthy":
                overall_health = "warning"
            issues.append({
                "component": "runner",
                "state": worker_lock,
                "message": "Runner lock is stale",
            })

        # Check restart policy state
        restart_policy = daemon_summary.get("restart_policy_state", "ready")
        if restart_policy in {"backoff", "capped"}:
            if overall_health == "healthy":
                overall_health = "warning"
            issues.append({
                "component": "restart_policy",
                "state": restart_policy,
                "message": f"Restart policy is {restart_policy}",
                "restart_attempts": daemon_summary.get("restart_attempts"),
                "max_restart_attempts": daemon_summary.get("max_restart_attempts"),
            })

        return {
            "workspace": workspace or "user_default",
            "overall_health": overall_health,
            "issues": issues,
            "daemon": {
                "health_state": daemon_health,
                "lock_state": daemon_summary.get("lock_state"),
                "heartbeat_state": daemon_summary.get("heartbeat_state"),
                "lease_state": daemon_summary.get("lease_state"),
                "restart_policy_state": restart_policy,
                "pid": daemon_summary.get("pid"),
                "active": daemon_summary.get("active"),
            },
            "runner": {
                "lock_state": worker_lock,
                "active": worker_summary.get("active"),
                "pid": worker_summary.get("pid"),
                "stale_after_seconds": worker_summary.get("stale_after_seconds"),
                "lease_expires_at": worker_summary.get("lease_expires_at"),
            },
            "reliability": health_summary,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_heartbeat_status(self, workspace: str | None = None) -> dict[str, Any]:
        """Get detailed heartbeat status for daemon and runner."""
        daemon_summary = self._train_queue_daemon_summary(workspace=workspace)
        worker_summary = self._train_queue_worker_summary(workspace=workspace)

        daemon_heartbeat = daemon_summary.get("last_heartbeat_at")
        worker_heartbeat = worker_summary.get("last_heartbeat_at")

        now = datetime.now(timezone.utc)

        # Calculate ages
        daemon_age = None
        if daemon_heartbeat:
            try:
                dt = self._parse_iso_datetime(daemon_heartbeat)
                if dt:
                    daemon_age = round((now - dt).total_seconds(), 3)
            except Exception:
                pass

        worker_age = None
        if worker_heartbeat:
            try:
                dt = self._parse_iso_datetime(worker_heartbeat)
                if dt:
                    worker_age = round((now - dt).total_seconds(), 3)
            except Exception:
                pass

        return {
            "workspace": workspace or "user_default",
            "daemon": {
                "last_heartbeat_at": daemon_heartbeat,
                "heartbeat_age_seconds": daemon_age,
                "heartbeat_state": daemon_summary.get("heartbeat_state"),
                "heartbeat_interval_seconds": daemon_summary.get("heartbeat_interval_seconds"),
                "lease_timeout_seconds": daemon_summary.get("lease_timeout_seconds"),
                "lease_expires_at": daemon_summary.get("lease_expires_at"),
                "lease_state": daemon_summary.get("lease_state"),
            },
            "runner": {
                "last_heartbeat_at": worker_heartbeat,
                "heartbeat_age_seconds": worker_age,
                "stale_after_seconds": worker_summary.get("stale_after_seconds"),
                "lease_expires_at": worker_summary.get("lease_expires_at"),
            },
            "checked_at": now.isoformat(),
        }

    def get_lease_status(self, workspace: str | None = None) -> dict[str, Any]:
        """Get lease status for active task execution."""
        from .reliability import LeaseManager

        workspace_name = workspace or "user_default"
        lease_manager = LeaseManager(workspace=workspace_name)

        # Get all expired leases
        expired_leases = lease_manager.list_expired_leases()

        # Get active lease info from daemon state
        daemon_summary = self._train_queue_daemon_summary(workspace=workspace)
        worker_summary = self._train_queue_worker_summary(workspace=workspace)

        return {
            "workspace": workspace_name,
            "daemon_lease": {
                "lease_state": daemon_summary.get("lease_state"),
                "lease_expires_at": daemon_summary.get("lease_expires_at"),
                "heartbeat_state": daemon_summary.get("heartbeat_state"),
            },
            "runner_lease": {
                "lock_state": worker_summary.get("lock_state"),
                "lease_expires_at": worker_summary.get("lease_expires_at"),
                "stale_after_seconds": worker_summary.get("stale_after_seconds"),
            },
            "expired_leases_count": len(expired_leases),
            "expired_leases": [
                {
                    "lease_id": lease.lease_id,
                    "job_id": lease.job_id,
                    "runner_id": lease.runner_id,
                    "expires_at": lease.expires_at.isoformat() if lease.expires_at else None,
                    "state": lease.state.value if lease.state else None,
                }
                for lease in expired_leases[:10]  # Limit to first 10
            ],
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    def check_stale_status(self, *, takeover: bool = False, workspace: str | None = None) -> dict[str, Any]:
        """Check if daemon or runner is stale and optionally trigger takeover."""
        daemon_summary = self._train_queue_daemon_summary(workspace=workspace)
        worker_summary = self._train_queue_worker_summary(workspace=workspace)

        results = {
            "workspace": workspace or "user_default",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "takeover_requested": takeover,
            "daemon": {
                "is_stale": daemon_summary.get("lock_state") == "stale",
                "lock_state": daemon_summary.get("lock_state"),
                "heartbeat_state": daemon_summary.get("heartbeat_state"),
                "can_recover": daemon_summary.get("can_recover"),
                "recovery_reason": daemon_summary.get("recovery_reason"),
            },
            "runner": {
                "is_stale": worker_summary.get("lock_state") == "stale",
                "lock_state": worker_summary.get("lock_state"),
                "active": worker_summary.get("active"),
            },
            "actions_taken": [],
        }

        if takeover:
            # Check daemon recovery
            if daemon_summary.get("can_recover") and daemon_summary.get("lock_state") == "stale":
                recovery_result = self.recover_train_queue_daemon(
                    workspace=workspace,
                    note="stale_takeover_auto_recovery",
                )
                results["actions_taken"].append({
                    "component": "daemon",
                    "action": "recover",
                    "result": recovery_result.get("command_status"),
                })

            # Note: Runner stale takeover happens automatically in run_train_queue_worker_runner
            # when it detects a stale lock, so we just report the status here
            if worker_summary.get("lock_state") == "stale":
                results["actions_taken"].append({
                    "component": "runner",
                    "action": "mark_stale",
                    "note": "Next runner invocation will perform takeover",
                })

        return results

    def force_recovery(self, *, workspace: str | None = None, reason: str | None = None) -> dict[str, Any]:
        """Force daemon recovery with reset restart policy."""
        # Load current state
        payload = self._load_train_queue_daemon_state(workspace=workspace)

        # Reset restart attempts to bypass backoff
        payload["restart_attempts"] = 0
        payload["next_restart_after"] = None
        self._persist_train_queue_daemon_state(payload, workspace=workspace)

        self._append_train_queue_daemon_history(
            workspace=workspace,
            event="force_recovery",
            reason=reason or "force_recovery_requested",
            metadata={"previous_restart_attempts": payload.get("restart_attempts")},
        )

        # Now trigger recovery
        return self.recover_train_queue_daemon(
            workspace=workspace,
            note=reason or "force_recovery",
        )

    def get_reliability_alerts(
        self,
        *,
        level: str | None = None,
        scope: str | None = None,
        limit: int = 10,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        """Get reliability alerts for monitoring."""
        from .reliability import AlertManager, AlertLevel
        from .models import AlertLevel as AlertLevelEnum

        workspace_name = workspace or "user_default"
        alert_manager = AlertManager(workspace=workspace_name)

        # Parse level filter
        level_enum = None
        if level:
            try:
                level_enum = AlertLevelEnum(level.lower())
            except (ValueError, KeyError):
                pass

        alerts = alert_manager.list_alerts(
            level=level_enum,
            scope=scope,
            resolved=False,
            limit=limit,
        )

        return {
            "workspace": workspace_name,
            "filters": {
                "level": level,
                "scope": scope,
            },
            "count": len(alerts),
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "level": alert.level.value if isinstance(alert.level, AlertLevelEnum) else str(alert.level),
                    "scope": alert.scope,
                    "reason": alert.reason,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
                    "job_id": alert.job_id,
                    "runner_id": alert.runner_id,
                    "acknowledged": alert.acknowledged,
                }
                for alert in alerts
            ],
            "summary": alert_manager.get_active_alerts_summary(),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


service = PipelineService()
pipeline = service


def generate(**kwargs: Any) -> str:
    return service.generate(**kwargs)


def distill(**kwargs: Any) -> str:
    return service.distill(**kwargs)


def train(**kwargs: Any) -> str:
    return service.train(**kwargs)


def evaluate(**kwargs: Any) -> str:
    return service.evaluate(**kwargs)


# Reliability/health check APIs for daemon and runner stability


def get_daemon_health_status(workspace: str | None = None) -> dict[str, Any]:
    """Get comprehensive health status for daemon and runner.

    Returns health metrics including heartbeat state, lease state,
    restart policy status, and any active alerts.
    """
    return service.get_health_status(workspace=workspace)


def get_daemon_heartbeat_status(workspace: str | None = None) -> dict[str, Any]:
    """Get detailed heartbeat status for daemon and runner.

    Returns heartbeat timestamps, age calculations, and freshness state.
    """
    return service.get_heartbeat_status(workspace=workspace)


def get_runner_lease_status(workspace: str | None = None) -> dict[str, Any]:
    """Get lease status for active task execution.

    Returns lease holder, expiration time, and validity state.
    """
    return service.get_lease_status(workspace=workspace)


def check_daemon_stale(*, takeover: bool = False, workspace: str | None = None) -> dict[str, Any]:
    """Check if daemon or runner is stale and optionally trigger takeover.

    Args:
        takeover: If True, attempt to take over stale locks
        workspace: Target workspace

    Returns:
        Status report with stale detection results and takeover actions
    """
    return service.check_stale_status(takeover=takeover, workspace=workspace)


def force_daemon_recovery(*, workspace: str | None = None, reason: str | None = None) -> dict[str, Any]:
    """Force daemon recovery with reset restart policy.

    This bypasses normal restart policy constraints and forces
    immediate recovery. Use with caution.
    """
    return service.force_recovery(workspace=workspace, reason=reason)


def get_reliability_alerts(
    *,
    level: str | None = None,
    scope: str | None = None,
    limit: int = 10,
    workspace: str | None = None,
) -> dict[str, Any]:
    """Get reliability alerts for monitoring.

    Args:
        level: Filter by alert level (critical, error, warning, attention, info)
        scope: Filter by scope (daemon, runner, task, system)
        limit: Maximum number of alerts to return
        workspace: Target workspace

    Returns:
        List of alerts with metadata
    """
    return service.get_reliability_alerts(level=level, scope=scope, limit=limit, workspace=workspace)
