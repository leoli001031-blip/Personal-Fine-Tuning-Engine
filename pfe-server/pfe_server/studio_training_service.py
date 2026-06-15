from __future__ import annotations

import threading
from typing import Any, Callable, Mapping
from uuid import uuid4

from .studio_jobs import (
    append_training_job_event,
    latest_training_event_type,
    training_job_payload,
)
from .studio_eval_jobs import utc_timestamp


PersistJob = Callable[[str, dict[str, Any]], None]
PersistOverall = Callable[[str, dict[str, Any]], None]
BuildJobsPayload = Callable[[int], dict[str, Any]]
StartBackground = Callable[[Callable[[], None]], None]


def extract_training_adapter_version(result_msg: Any) -> str | None:
    for token in str(result_msg).split():
        if token.startswith("2") and len(token) >= 8:
            return token
    return None


def build_training_job_entry(
    *,
    job_id: str,
    workspace: str,
    method: str,
    training_config: Mapping[str, Any],
    retry_of: str | None = None,
    now_seconds: float | None = None,
) -> dict[str, Any]:
    now = utc_timestamp(now_seconds)
    job_entry = {
        "job_id": job_id,
        "workspace": workspace,
        "status": "queued",
        "method": method,
        "adapter_version": None,
        "checkpoints": [],
        "events": [],
        "training_config": dict(training_config),
        "created_at": now,
        "updated_at": now,
    }
    if retry_of:
        job_entry["retry_of"] = retry_of
    return job_entry


def training_overall_state(job_id: str, job_entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state": job_entry.get("status"),
        "adapter_version": job_entry.get("adapter_version"),
        "job_id": job_id,
        "updated_at": job_entry.get("updated_at"),
    }


def run_training_job(
    job_entry: dict[str, Any],
    *,
    pipeline: Any,
    method: str,
    retry_of: str | None,
    persist_job: PersistJob,
    persist_overall: PersistOverall,
) -> None:
    job_id = str(job_entry.get("job_id") or "")
    workspace = str(job_entry.get("workspace") or "")
    try:
        if job_entry.get("status") == "cancelled" or job_entry.get("cancellation_requested"):
            job_entry["status"] = "cancelled"
            if latest_training_event_type(job_entry) != "cancelled":
                append_training_job_event(
                    job_entry,
                    event_type="cancelled",
                    status="cancelled",
                    message="training job cancelled before start",
                )
            return
        job_entry["status"] = "running"
        append_training_job_event(
            job_entry,
            event_type="started",
            status="running",
            message="training job started",
            metadata={"method": method, "retry_of": retry_of},
        )
        persist_job(job_id, job_entry)
        if method == "dpo":
            result_msg = pipeline.train_dpo()
        else:
            result_msg = pipeline.train()
        version = extract_training_adapter_version(result_msg)
        job_entry["status"] = "completed"
        job_entry["adapter_version"] = version
        job_entry["result"] = result_msg
        append_training_job_event(
            job_entry,
            event_type="completed",
            status="completed",
            message="training job completed",
            metadata={
                "adapter_version": version,
                "result_summary": str(result_msg)[:240],
                "cancellation_requested": bool(job_entry.get("cancellation_requested")),
                "retry_of": retry_of,
            },
        )
    except Exception as exc:
        job_entry["status"] = "failed"
        job_entry["error"] = str(exc)
        append_training_job_event(
            job_entry,
            event_type="failed",
            status="failed",
            message="training job failed",
            metadata={"error": str(exc), "retry_of": retry_of},
        )
    finally:
        persist_job(job_id, job_entry)
        persist_overall(workspace, training_overall_state(job_id, job_entry))


def default_thread_starter(target: Callable[[], None]) -> None:
    threading.Thread(target=target, daemon=True).start()


def start_training_job(
    *,
    workspace: str,
    pipeline: Any,
    method: str,
    training_config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    retry_of: str | None = None,
    persist_job: PersistJob,
    persist_overall: PersistOverall,
    build_jobs_payload: BuildJobsPayload,
    job_id_factory: Callable[[], str] | None = None,
    start_background: StartBackground | None = None,
) -> dict[str, Any]:
    job_id = str(job_id_factory() if job_id_factory else uuid4())
    job_entry = build_training_job_entry(
        job_id=job_id,
        workspace=workspace,
        method=method,
        training_config=training_config,
        retry_of=retry_of,
    )
    append_training_job_event(
        job_entry,
        event_type="queued",
        status="queued",
        message="training job queued",
        metadata={"method": method, "workspace": workspace, "retry_of": retry_of},
    )
    persist_job(job_id, job_entry)

    starter = start_background or default_thread_starter
    starter(
        lambda: run_training_job(
            job_entry,
            pipeline=pipeline,
            method=method,
            retry_of=retry_of,
            persist_job=persist_job,
            persist_overall=persist_overall,
        )
    )
    payload = {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/pfe/training/jobs/{job_id}",
        "job": training_job_payload(job_entry),
        "jobs": build_jobs_payload(10),
        "preflight": dict(preflight),
    }
    if retry_of:
        payload["retry_of"] = retry_of
        payload["action"] = "retry_started"
    return payload


__all__ = [
    "build_training_job_entry",
    "default_thread_starter",
    "extract_training_adapter_version",
    "run_training_job",
    "start_training_job",
    "training_overall_state",
]
