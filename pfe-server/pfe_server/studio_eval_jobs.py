from __future__ import annotations

import time
from typing import Any, Mapping


EVAL_STATUS_URL = "/pfe/eval/status"


def utc_timestamp(now_seconds: float | None = None) -> str:
    now_value = time.time() if now_seconds is None else now_seconds
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_value))


def running_eval_version(state: Mapping[str, Any]) -> str:
    if str(state.get("state") or "") != "running":
        return ""
    return str(state.get("version") or "")


def running_eval_summary() -> dict[str, Any]:
    return {
        "state": "running",
        "label": "评估中",
        "recommendation": None,
        "comparison": None,
        "scores": {},
        "summary_line": "评估结论：评估中",
    }


def build_eval_running_state(
    *,
    version: str,
    requested_version: str,
    job_id: str,
    now_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "state": "running",
        "version": version,
        "requested_version": requested_version,
        "job_id": job_id,
        "status_url": EVAL_STATUS_URL,
        "updated_at": utc_timestamp(now_seconds),
    }


def build_eval_completed_state(
    *,
    version: str,
    requested_version: str,
    raw_result: Any,
    job_id: str,
    eval_report: Mapping[str, Any] | None = None,
    now_seconds: float | None = None,
) -> dict[str, Any]:
    payload = {
        "state": "completed",
        "version": version,
        "requested_version": requested_version,
        "raw_result": str(raw_result),
        "job_id": job_id,
        "status_url": EVAL_STATUS_URL,
        "updated_at": utc_timestamp(now_seconds),
    }
    if isinstance(eval_report, Mapping):
        payload.update(dict(eval_report))
    return payload


def build_eval_failed_state(
    *,
    version: str,
    requested_version: str,
    error: Any,
    job_id: str,
    now_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "state": "failed",
        "version": version,
        "requested_version": requested_version,
        "error": str(error),
        "job_id": job_id,
        "status_url": EVAL_STATUS_URL,
        "updated_at": utc_timestamp(now_seconds),
    }


def build_eval_status_payload(
    state: Mapping[str, Any] | None,
    *,
    adapters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(state or {"state": "idle"})
    payload.setdefault("status_url", EVAL_STATUS_URL)
    if adapters is not None:
        payload["adapters"] = dict(adapters)
    return payload


def auto_eval_state_from_last_result(last_result: Mapping[str, Any]) -> dict[str, Any] | None:
    if not last_result.get("eval_triggered"):
        return None
    payload = {
        "state": "completed",
        "version": last_result.get("triggered_version") or last_result.get("promoted_version"),
        "recommendation": last_result.get("eval_recommendation"),
        "comparison": last_result.get("eval_comparison"),
        "auto_evaluate": True,
    }
    if last_result.get("eval_error") or (last_result.get("error_stage") == "eval"):
        payload["state"] = "failed"
        payload["error"] = last_result.get("eval_error") or last_result.get("error")
    return payload


__all__ = [
    "EVAL_STATUS_URL",
    "auto_eval_state_from_last_result",
    "build_eval_completed_state",
    "build_eval_failed_state",
    "build_eval_running_state",
    "build_eval_status_payload",
    "running_eval_summary",
    "running_eval_version",
    "utc_timestamp",
]
