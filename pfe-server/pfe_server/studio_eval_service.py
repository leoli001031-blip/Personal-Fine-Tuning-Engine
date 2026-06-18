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
    running_eval_summary,
)
from .studio_eval_suite import (
    merge_studio_eval_suite_report,
    normalize_suite_names,
    run_studio_eval_suite,
)


PersistEvalState = Callable[[str, dict[str, Any]], None]
BuildAdaptersPayload = Callable[[], dict[str, Any]]
LoadAdapterPath = Callable[[str], Any]
StartBackground = Callable[[Callable[[], None]], None]


def evaluate_pipeline(
    pipeline: Any,
    *,
    base_model: str,
    adapter: str,
    num_samples: int,
    workspace: str,
) -> Any:
    target = getattr(pipeline, "pipeline", pipeline)
    return target.evaluate(
        base_model=base_model,
        adapter=adapter,
        num_samples=num_samples,
        workspace=workspace,
    )


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
        base_model = request_body.get("base_model") or default_base_model()
        result = evaluate_pipeline(
            pipeline,
            base_model=base_model,
            adapter=version,
            num_samples=int(request_body.get("num_samples", 20)),
            workspace=workspace,
        )
        adapter_path = load_adapter_path(version)
        eval_report = load_eval_report(adapter_path)
        suite = normalize_suite_names(request_body.get("suite"))
        if suite:
            from pfe_core.adapter_store import create_adapter_store
            from pfe_core.storage import list_samples

            suite_report = run_studio_eval_suite(
                base_model=str(base_model),
                adapter_path=str(adapter_path),
                adapter_version=version,
                suite=suite,
                samples=list_samples(limit=100),
            )
            eval_report = merge_studio_eval_suite_report(eval_report, suite_report)
            create_adapter_store(workspace=workspace).attach_eval_report(version, eval_report)
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


def _mark_started_version_running(adapters: Mapping[str, Any], version: str) -> dict[str, Any]:
    payload = dict(adapters)
    versions = []
    if isinstance(payload.get("versions"), list):
        for item in list(payload.get("versions") or []):
            if not isinstance(item, Mapping):
                continue
            version_item = dict(item)
            if str(version_item.get("version") or "") == version:
                version_item["eval_running"] = True
                version_item["can_eval"] = False
                version_item["eval_summary"] = running_eval_summary()
            versions.append(version_item)
        payload["versions"] = versions
    pending = payload.get("pending_eval_adapter")
    if isinstance(pending, Mapping) and str(pending.get("version") or "") == version:
        pending_item = dict(pending)
        pending_item["eval_running"] = True
        pending_item["can_eval"] = False
        pending_item["eval_summary"] = running_eval_summary()
        payload["pending_eval_adapter"] = pending_item
    return payload


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
    payload["adapters"] = _mark_started_version_running(build_adapters_payload(), version)
    return payload


__all__ = [
    "default_thread_starter",
    "evaluate_pipeline",
    "load_eval_report",
    "run_eval_job",
    "start_eval_job",
]
