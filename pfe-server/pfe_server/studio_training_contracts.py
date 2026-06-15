from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SUPPORTED_STUDIO_TRAINING_METHODS = ("sft", "dpo")


@dataclass(frozen=True)
class TrainingJobRequest:
    method: str
    training_config: dict[str, Any]
    confirmed: bool
    raw_method: str | None = None

    @property
    def unsupported_method(self) -> str | None:
        return self.method if self.method not in SUPPORTED_STUDIO_TRAINING_METHODS else None

    def payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "training_config": dict(self.training_config),
            "confirmed": self.confirmed,
        }


def _mapping_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def training_request_from_body(
    body: Mapping[str, Any],
    query_params: Mapping[str, str] | None = None,
    confirmed: bool | None = None,
) -> TrainingJobRequest:
    raw_method_value = body.get("method")
    raw_method = str(raw_method_value).strip().lower() if raw_method_value is not None else None
    method = raw_method or "sft"
    return TrainingJobRequest(
        method=method,
        raw_method=raw_method,
        confirmed=(
            bool(confirmed)
            if confirmed is not None
            else _mapping_bool(query_params or {}, "confirm") or _mapping_bool(body, "confirm")
        ),
        training_config={
            k: v
            for k, v in body.items()
            if k not in ("method", "auto_trigger", "confirm")
        },
    )


def build_training_preflight_payload(
    *,
    request: TrainingJobRequest,
    readiness: Mapping[str, Any],
    workspace: str,
    base_model: str,
) -> dict[str, Any]:
    training_config = request.training_config
    model_source = dict(readiness.get("model", {}).get("source") or {})
    runtime_deps = dict(readiness.get("runtime", {}).get("dependencies") or {})
    inference = dict(readiness.get("inference") or {})
    resolved_base_model = str(
        training_config.get("base_model")
        or readiness.get("configuration", {}).get("base_model")
        or base_model
    )

    blocked_by: list[str] = []
    warnings: list[str] = []
    next_actions: list[dict[str, str]] = []
    if not model_source.get("ok"):
        blocked_by.append(str(model_source.get("state") or "model_source_not_ready"))
        next_actions.append(
            {
                "id": "choose_local_model",
                "label": "选择本地模型",
                "detail": "启动训练前需要一个已存在的本地基础模型目录。",
            }
        )
    if not runtime_deps.get("ok"):
        warnings.append("runtime_dependencies_missing")
    if not inference.get("real_local_enabled"):
        warnings.append("real_local_inference_disabled")

    return {
        "kind": "pfe_training_preflight",
        "ready": not blocked_by,
        "requires_confirmation": True,
        "confirm_api": "POST /pfe/training/jobs",
        "request": request.payload(),
        "method": request.method,
        "workspace": workspace,
        "base_model": resolved_base_model,
        "blocked_by": blocked_by,
        "warnings": warnings,
        "next_actions": next_actions,
        "preview": {
            "method": request.method,
            "training_config": training_config,
            "will_create_job": False,
            "will_start_background_training": False,
        },
        "readiness": {
            "summary": readiness.get("summary"),
            "model": readiness.get("model"),
            "runtime_dependencies": runtime_deps,
            "inference": inference,
            "version": readiness.get("version"),
        },
    }


def build_legacy_training_trigger_preflight_payload(
    *,
    request: TrainingJobRequest,
    workspace: str,
    base_model: str,
) -> dict[str, Any]:
    resolved_base_model = str(request.training_config.get("base_model") or base_model)
    return {
        "kind": "pfe_training_preflight",
        "ready": True,
        "requires_confirmation": False,
        "confirm_api": "POST /pfe/training/trigger",
        "request": request.payload(),
        "method": request.method,
        "workspace": workspace,
        "base_model": resolved_base_model,
        "blocked_by": [],
        "warnings": ["legacy_trigger_bypasses_studio_preflight"],
        "next_actions": [],
        "preview": {
            "method": request.method,
            "training_config": request.training_config,
            "will_create_job": True,
            "will_start_background_training": True,
            "legacy_endpoint": True,
        },
    }


__all__ = [
    "SUPPORTED_STUDIO_TRAINING_METHODS",
    "TrainingJobRequest",
    "build_legacy_training_trigger_preflight_payload",
    "build_training_preflight_payload",
    "training_request_from_body",
]
