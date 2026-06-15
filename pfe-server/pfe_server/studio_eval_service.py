from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from .studio_eval_jobs import (
    build_eval_completed_state,
    build_eval_failed_state,
    build_eval_running_state,
    build_eval_status_payload,
)


PersistEvalState = Callable[[str, dict[str, Any]], None]
BuildAdaptersPayload = Callable[[], dict[str, Any]]
LoadAdapterPath = Callable[[str], Any]
StartBackground = Callable[[Callable[[], None]], None]


def load_eval_report(adapter_path: Any) -> dict[str, Any]:
    report_path = Path(adapter_path) / "eval_report.json"
    if not report_path.exists():
        return {}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def run_eval_job(
    *,
    pipeline: Any,
    workspace: str,
    version: str,
    requested_version: str,
    job_id: str,
    request_body: Mapping[str, Any],
    default_base_model: Callable[[], str],
    load_adapter_path: LoadAdapterPath,
    persist_state: PersistEvalState,
) -> None:
    try:
        result = pipeline.evaluate(
            base_model=request_body.get("base_model") or default_base_model(),
            adapter=version,
            num_samples=int(request_body.get("num_samples", 20)),
            workspace=workspace,
        )
        eval_report = load_eval_report(load_adapter_path(version))
        state = build_eval_completed_state(
            version=version,
            requested_version=requested_version,
            raw_result=result,
            job_id=job_id,
            eval_report=eval_report,
        )
    except Exception as exc:
        state = build_eval_failed_state(
            version=version,
            requested_version=requested_version,
            error=exc,
            job_id=job_id,
        )
    persist_state(workspace, state)


def default_thread_starter(target: Callable[[], None]) -> None:
    threading.Thread(target=target, daemon=True).start()


def start_eval_job(
    *,
    pipeline: Any,
    workspace: str,
    version: str,
    requested_version: str,
    request_body: Mapping[str, Any],
    default_base_model: Callable[[], str],
    load_adapter_path: LoadAdapterPath,
    persist_state: PersistEvalState,
    build_adapters_payload: BuildAdaptersPayload,
    job_id_factory: Callable[[], str] | None = None,
    start_background: StartBackground | None = None,
) -> dict[str, Any]:
    job_id = str(job_id_factory() if job_id_factory else uuid4())
    running_state = build_eval_running_state(
        version=version,
        requested_version=requested_version,
        job_id=job_id,
    )
    persist_state(workspace, running_state)

    starter = start_background or default_thread_starter
    starter(
        lambda: run_eval_job(
            pipeline=pipeline,
            workspace=workspace,
            version=version,
            requested_version=requested_version,
            job_id=job_id,
            request_body=request_body,
            default_base_model=default_base_model,
            load_adapter_path=load_adapter_path,
            persist_state=persist_state,
        )
    )
    payload = build_eval_status_payload(running_state)
    payload["adapters"] = build_adapters_payload()
    return payload


__all__ = [
    "default_thread_starter",
    "load_eval_report",
    "run_eval_job",
    "start_eval_job",
]
