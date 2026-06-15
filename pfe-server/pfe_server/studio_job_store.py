from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .studio_jobs import append_training_job_event, build_training_jobs_payload, training_job_payload


ACTIVE_TRAINING_STATUSES = {"queued", "running"}
TERMINAL_TRAINING_STATUSES = {"completed", "failed", "cancelled"}

_STATE_LOCK = threading.RLock()


def load_json_state(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def save_json_state(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(dict(payload), ensure_ascii=False, default=str), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


class StudioTrainingJobStore:
    def __init__(
        self,
        *,
        workspace: str,
        workspace_dir: Path,
        memory_jobs: dict[str, dict[str, Any]],
        overall_state: dict[str, dict[str, Any]],
    ) -> None:
        self.workspace = workspace
        self.workspace_dir = workspace_dir
        self.memory_jobs = memory_jobs
        self.overall_state = overall_state

    @property
    def jobs_path(self) -> Path:
        return self.workspace_dir / "training_jobs.json"

    @property
    def state_path(self) -> Path:
        return self.workspace_dir / "training_status.json"

    def stored_jobs(self) -> dict[str, Any]:
        with _STATE_LOCK:
            return load_json_state(self.jobs_path)

    def current_overall_state(self) -> dict[str, Any]:
        with _STATE_LOCK:
            state = self.overall_state.get(self.workspace)
            if state is None:
                state = load_json_state(self.state_path)
            return dict(state) if state else {}

    def build_jobs_payload(self, *, limit: int = 20) -> dict[str, Any]:
        with _STATE_LOCK:
            return build_training_jobs_payload(
                workspace=self.workspace,
                stored_jobs=load_json_state(self.jobs_path),
                memory_jobs=self.memory_jobs,
                overall_state=self.overall_state.get(self.workspace) or load_json_state(self.state_path),
                limit=limit,
            )

    def active_job(self) -> dict[str, Any] | None:
        active = self.build_jobs_payload(limit=0).get("active")
        return dict(active) if isinstance(active, Mapping) else None

    def get_job(self, job_id: str, *, mutable: bool = False) -> dict[str, Any] | None:
        with _STATE_LOCK:
            memory_entry = self.memory_jobs.get(job_id)
            if isinstance(memory_entry, dict) and self._belongs_to_workspace(memory_entry):
                return memory_entry if mutable else dict(memory_entry)
            stored_entry = load_json_state(self.jobs_path).get(job_id)
            if isinstance(stored_entry, Mapping):
                return dict(stored_entry)
        return None

    def persist_job(self, job_id: str, job_entry: dict[str, Any]) -> None:
        with _STATE_LOCK:
            self.memory_jobs[job_id] = job_entry
            stored = load_json_state(self.jobs_path)
            stored[job_id] = job_entry
            save_json_state(self.jobs_path, stored)

    def persist_overall(self, workspace: str, state: dict[str, Any]) -> None:
        with _STATE_LOCK:
            self.overall_state[workspace] = state
            save_json_state(self.workspace_dir_for(workspace) / "training_status.json", state)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with _STATE_LOCK:
            job_entry = self.get_job(job_id, mutable=True)
            if job_entry is None:
                return {"outcome": "not_found"}
            status = str(job_entry.get("status") or "")
            if status in TERMINAL_TRAINING_STATUSES:
                return {"outcome": "not_cancellable", "job": training_job_payload(job_entry)}

            if status == "queued":
                job_entry["status"] = "cancelled"
                job_entry["cancellation_requested"] = False
                append_training_job_event(
                    job_entry,
                    event_type="cancelled",
                    status="cancelled",
                    message="training job cancelled before start",
                )
                action = "cancelled"
                message = "训练任务已取消。"
            else:
                job_entry["cancellation_requested"] = True
                append_training_job_event(
                    job_entry,
                    event_type="cancel_requested",
                    status=status or "running",
                    message="cancellation requested; running training cannot be interrupted",
                )
                action = "cancel_requested"
                message = "已记录停止请求；当前训练无法被强行中断。"

            self.persist_job(job_id, job_entry)
            self.persist_overall(
                self.workspace,
                {
                    "state": job_entry.get("status"),
                    "adapter_version": job_entry.get("adapter_version"),
                    "job_id": job_id,
                    "cancellation_requested": bool(job_entry.get("cancellation_requested")),
                    "updated_at": job_entry.get("updated_at"),
                },
            )
            return {
                "outcome": "ok",
                "action": action,
                "message": message,
                "job": training_job_payload(job_entry),
            }

    def mark_retry_requested(self, job_id: str) -> dict[str, Any]:
        with _STATE_LOCK:
            job_entry = self.get_job(job_id, mutable=True)
            if job_entry is None:
                return {"outcome": "not_found"}
            status = str(job_entry.get("status") or "")
            if status not in {"failed", "cancelled"}:
                return {"outcome": "not_retryable", "job": training_job_payload(job_entry)}
            append_training_job_event(
                job_entry,
                event_type="retry_requested",
                status=status,
                message="training job retry requested",
                metadata={"retry_api": f"/pfe/training/jobs/{job_id}/retry"},
            )
            self.persist_job(job_id, job_entry)
            return {"outcome": "ok", "job": job_entry}

    def workspace_dir_for(self, workspace: str) -> Path:
        if workspace == self.workspace:
            return self.workspace_dir
        return self.workspace_dir.parent / workspace

    def _belongs_to_workspace(self, job_entry: Mapping[str, Any]) -> bool:
        entry_workspace = job_entry.get("workspace")
        return not entry_workspace or str(entry_workspace) == self.workspace


__all__ = [
    "ACTIVE_TRAINING_STATUSES",
    "StudioTrainingJobStore",
    "TERMINAL_TRAINING_STATUSES",
    "load_json_state",
    "save_json_state",
]
