from __future__ import annotations

import time
from typing import Any, Mapping


def training_job_payload(job_entry: Mapping[str, Any]) -> dict[str, Any]:
    job_id = str(job_entry.get("job_id") or "")
    payload = dict(job_entry)
    if job_id:
        payload["status_url"] = f"/pfe/training/jobs/{job_id}"
        payload["events_url"] = f"/pfe/training/jobs/{job_id}/events"
        payload["cancel_url"] = f"/pfe/training/jobs/{job_id}/cancel"
        payload["retry_url"] = f"/pfe/training/jobs/{job_id}/retry"
    checkpoints = job_entry.get("checkpoints")
    payload["checkpoint_count"] = len(checkpoints) if isinstance(checkpoints, list) else 0
    events = job_entry.get("events")
    payload["event_count"] = len(events) if isinstance(events, list) else 0
    if isinstance(events, list) and events:
        payload["latest_event"] = events[-1]
    result = payload.get("result")
    if result is not None:
        payload["result_summary"] = str(result)[:240]
    return payload


def training_job_event(
    *,
    job_id: str,
    event_type: str,
    status: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
    now_seconds: float | None = None,
) -> dict[str, Any]:
    now_value = time.time() if now_seconds is None else now_seconds
    return {
        "event_id": f"{job_id}-{event_type}-{int(now_value * 1000)}",
        "job_id": job_id,
        "type": event_type,
        "status": status,
        "message": message,
        "metadata": dict(metadata or {}),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_value)),
    }


def append_training_job_event(
    job_entry: dict[str, Any],
    *,
    event_type: str,
    status: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
    now_seconds: float | None = None,
) -> dict[str, Any]:
    events = job_entry.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        job_entry["events"] = events
    event = training_job_event(
        job_id=str(job_entry.get("job_id") or ""),
        event_type=event_type,
        status=status,
        message=message,
        metadata=metadata,
        now_seconds=now_seconds,
    )
    events.append(event)
    job_entry["updated_at"] = event["created_at"]
    return event


def latest_training_event_type(job_entry: Mapping[str, Any]) -> str | None:
    events = job_entry.get("events")
    if isinstance(events, list) and events:
        latest = events[-1]
        if isinstance(latest, Mapping):
            return str(latest.get("type") or "")
    return None


def build_training_jobs_payload(
    *,
    workspace: str,
    stored_jobs: Mapping[str, Any],
    memory_jobs: Mapping[str, Any],
    overall_state: Mapping[str, Any] | None,
    limit: int = 20,
) -> dict[str, Any]:
    jobs_by_id: dict[str, dict[str, Any]] = {
        str(job_id): dict(job_entry)
        for job_id, job_entry in stored_jobs.items()
        if isinstance(job_entry, Mapping)
    }
    for job_id, job_entry in memory_jobs.items():
        if not isinstance(job_entry, Mapping):
            continue
        entry_workspace = job_entry.get("workspace")
        if (entry_workspace and str(entry_workspace) == workspace) or str(job_id) in jobs_by_id:
            jobs_by_id[str(job_id)] = dict(job_entry)
    jobs = [training_job_payload(job_entry) for job_entry in jobs_by_id.values()]
    jobs.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    if limit > 0:
        jobs = jobs[:limit]
    active = next((item for item in jobs if item.get("status") in {"queued", "running"}), None)
    latest = jobs[0] if jobs else None
    state = dict(overall_state or {})
    if not state:
        state = {"state": "idle", "adapter_version": None}
    return {
        "workspace": workspace,
        "count": len(jobs_by_id),
        "items": jobs,
        "latest": latest,
        "active": active,
        "state": state,
        "create_api": "POST /pfe/training/jobs",
    }


__all__ = [
    "append_training_job_event",
    "build_training_jobs_payload",
    "latest_training_event_type",
    "training_job_event",
    "training_job_payload",
]
