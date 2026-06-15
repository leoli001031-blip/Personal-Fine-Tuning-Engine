from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol, Union
from uuid import uuid4

from .auth import AccessContext, ServerSecurityConfig, authorize_request, normalize_headers
from .models import (
    AutoTrainTriggerActionResponse,
    CandidateActionResponse,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionUsage,
    DistillRunRequest,
    DistillRunResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    PFEStatusResponse,
    ServeRuntimeInfo,
    SignalIngestRequest,
    SignalIngestResponse,
)
from .studio_contracts import (
    build_handoff_closed_loop,
    build_handoff_copy_text,
    build_handoff_javascript_snippet,
    build_handoff_python_snippet,
    build_runtime_payload as build_studio_runtime_payload,
)
from .studio_eval_jobs import (
    EVAL_STATUS_URL,
    auto_eval_state_from_last_result,
    build_eval_running_state,
    build_eval_status_payload,
    running_eval_summary,
    running_eval_version,
)
from .studio_eval_service import start_eval_job as start_studio_eval_job
from .studio_jobs import (
    append_training_job_event as append_studio_training_job_event,
    latest_training_event_type as latest_studio_training_event_type,
    training_job_event as studio_training_job_event,
    training_job_payload as studio_training_job_payload,
)
from .studio_job_store import (
    StudioTrainingJobStore,
    load_json_state as load_studio_json_state,
    save_json_state as save_studio_json_state,
)
from .studio_resources import (
    build_models_payload as build_studio_models_payload,
    build_workspaces_payload as build_studio_workspaces_payload,
    model_candidate as studio_model_candidate,
    validate_model_reference as validate_studio_model_reference,
    workspace_paths as studio_workspace_paths,
    workspace_record as studio_workspace_record,
    workspace_slug_issues as studio_workspace_slug_issues,
)
from .studio_training_contracts import (
    SUPPORTED_STUDIO_TRAINING_METHODS,
    TrainingJobRequest,
    build_legacy_training_trigger_preflight_payload as build_studio_legacy_training_trigger_preflight_payload,
    build_training_preflight_payload as build_studio_training_preflight_payload,
    training_request_from_body as studio_training_request_from_body,
)
from .studio_training_service import start_training_job as start_studio_training_job

# ChatCollector integration - import from pfe_core if available
def _try_import_chat_collector() -> tuple[bool, Any, Any, Any]:
    """Try to import ChatCollector from pfe_core."""
    try:
        # Try to add pfe-core to path
        candidate = Path(__file__).resolve().parents[2] / "pfe-core"
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

        from pfe_core.collector.chat_collector import ChatCollector
        from pfe_core.collector.config import CollectorConfig
        from pfe_core.models import ChatInteraction
        return True, ChatCollector, CollectorConfig, ChatInteraction
    except Exception:
        return False, None, None, None


CHAT_COLLECTOR_AVAILABLE, ChatCollector, CollectorConfig, ChatInteraction = _try_import_chat_collector()


# Session-level pending interactions storage for signal collection
_pending_interactions: dict[str, dict[str, Any]] = {}


def _default_chat_base_model() -> str:
    try:
        from pfe_core.inference.engine import resolve_base_model_reference

        return resolve_base_model_reference("local-default")
    except Exception:
        return "local-default"


def _resolve_pfe_home() -> Path:
    """Resolve the active PFE home with env/workspace awareness."""
    try:
        from pfe_core.storage import resolve_home

        return resolve_home()
    except Exception:
        return Path(os.getenv("PFE_HOME") or (Path.home() / ".pfe"))


def _get_or_create_collector(workspace: str) -> Optional[Any]:
    """Get or create a ChatCollector instance for the workspace."""
    if not CHAT_COLLECTOR_AVAILABLE or ChatCollector is None:
        return None
    try:
        config = CollectorConfig()
        return ChatCollector(workspace=workspace, config=config, home=str(_resolve_pfe_home()))
    except Exception:
        return None


def _normalize_feedback_action(action: str) -> tuple[str, str]:
    """Return collector action and normalized signal type for a feedback action."""
    action_map = {
        "accept": ("continue", "accept"),
        "reject": ("delete", "reject"),
        "delete": ("delete", "reject"),
        "edit": ("edit", "edit"),
        "regenerate": ("regenerate", "regenerate"),
    }
    return action_map.get(action, ("continue", action))


def _fallback_feedback_signal(action: str, response_time_seconds: float | None = None) -> tuple[str, float, str]:
    """Provide deterministic feedback semantics when collector extraction is unavailable."""
    _, normalized_signal_type = _normalize_feedback_action(action)
    if normalized_signal_type == "accept":
        if response_time_seconds is not None and response_time_seconds < 5:
            return normalized_signal_type, 0.9, "accept_quick_response"
        if response_time_seconds is not None and response_time_seconds > 60:
            return normalized_signal_type, 0.4, "accept_slow_response"
        return normalized_signal_type, 0.7, "accept_normal"
    if normalized_signal_type == "reject":
        return normalized_signal_type, 0.95, "reject_explicit_feedback"
    if normalized_signal_type == "regenerate":
        return normalized_signal_type, 0.85, "regenerate_explicit_feedback"
    if normalized_signal_type == "edit":
        return normalized_signal_type, 0.8, "edit_explicit_feedback"
    return normalized_signal_type, 0.7, "feedback_fallback"


def _build_feedback_interaction(request: FeedbackRequest, pending: dict[str, Any] | None) -> Any | None:
    """Build a ChatInteraction when enough context exists to extract feedback."""
    if ChatInteraction is None:
        return None
    user_message = request.user_message or (pending.get("user_message") if pending else None)
    assistant_message = request.assistant_message or (pending.get("assistant_message") if pending else None)
    if not user_message or not assistant_message:
        return None
    return ChatInteraction(
        session_id=request.session_id,
        request_id=request.request_id,
        user_message=user_message,
        assistant_message=assistant_message,
        adapter_version=request.adapter_version or (pending.get("adapter_version") if pending else None),
        response_time_seconds=request.response_time_seconds,
    )


def _store_pending_interaction(
    session_id: str,
    request_id: str,
    user_message: str,
    assistant_message: str,
    adapter_version: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    """Store interaction data for later signal extraction."""
    global _pending_interactions
    if session_id not in _pending_interactions:
        _pending_interactions[session_id] = {}

    _pending_interactions[session_id][request_id] = {
        "user_message": user_message,
        "assistant_message": assistant_message,
        "adapter_version": adapter_version,
        "timestamp": timestamp or time.time(),
    }

    # Cleanup old sessions (keep only last 100 sessions)
    if len(_pending_interactions) > 100:
        oldest_key = next(iter(_pending_interactions))
        del _pending_interactions[oldest_key]


def _get_pending_interaction(session_id: str, request_id: str) -> Optional[dict[str, Any]]:
    """Retrieve pending interaction data."""
    global _pending_interactions
    session_data = _pending_interactions.get(session_id, {})
    return session_data.get(request_id)


def _remove_pending_interaction(session_id: str, request_id: str) -> None:
    """Remove pending interaction after signal extraction."""
    global _pending_interactions
    if session_id in _pending_interactions:
        _pending_interactions[session_id].pop(request_id, None)
        if not _pending_interactions[session_id]:
            del _pending_interactions[session_id]

try:  # FastAPI is optional in the current workspace snapshot.
    from fastapi import FastAPI, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in the current environment.
    FastAPI = None  # type: ignore[assignment]
    Request = Any  # type: ignore[assignment]
    Response = Any  # type: ignore[assignment]
    HTMLResponse = None  # type: ignore[assignment]
    JSONResponse = None  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[assignment]
    FASTAPI_AVAILABLE = False


@dataclass
class RequestEnvelope:
    method: str
    path: str
    headers: dict[str, str]
    client_host: Optional[str]
    body: bytes
    query_params: dict[str, str] = field(default_factory=dict)


@dataclass
class ServiceBundle:
    inference: Any
    pipeline: Any
    security: ServerSecurityConfig
    provider: str = "mock"
    workspace: str = "user_default"
    started_at: float = field(default_factory=time.time)
    runtime_probe: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServePlan:
    runtime: ServeRuntimeInfo
    app: Any
    uvicorn_module: Optional[str] = None
    uvicorn_kwargs: dict[str, Any] = field(default_factory=dict)
    runtime_probe: dict[str, Any] = field(default_factory=dict)

    @property
    def command(self) -> list[str]:
        return self.runtime.command

    @property
    def runner(self) -> dict[str, Any]:
        return self.runtime.runner


class InferenceService(Protocol):
    async def generate_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        ...

    async def status(self) -> dict[str, Any]:
        ...


class PipelineService(Protocol):
    async def ingest_signal(self, request: SignalIngestRequest) -> SignalIngestResponse:
        ...

    async def run_distillation(self, request: DistillRunRequest) -> DistillRunResponse:
        ...

    async def status(self) -> dict[str, Any]:
        ...

    async def reset_auto_train_trigger(self) -> dict[str, Any]:
        ...

    async def retry_auto_train_trigger(self) -> dict[str, Any]:
        ...

    async def process_next_train_queue(self) -> dict[str, Any]:
        ...

    async def process_train_queue_batch(self, limit: int = 5) -> dict[str, Any]:
        ...

    async def process_train_queue_until_idle(self, max_iterations: int | None = None) -> dict[str, Any]:
        ...

    async def run_train_queue_worker_runner(
        self,
        max_seconds: float | None = None,
        idle_sleep_seconds: float | None = None,
    ) -> dict[str, Any]:
        ...

    async def stop_train_queue_worker_runner(self) -> dict[str, Any]:
        ...

    async def train_queue_worker_runner_status(self) -> dict[str, Any]:
        ...

    async def train_queue_worker_runner_history(self, limit: int = 10) -> dict[str, Any]:
        ...

    async def start_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        ...

    async def stop_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        ...

    async def train_queue_daemon_status(self) -> dict[str, Any]:
        ...

    async def train_queue_daemon_history(self, limit: int = 10) -> dict[str, Any]:
        ...

    async def recover_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        ...

    async def restart_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        ...

    async def approve_next_train_queue(self, note: str | None = None) -> dict[str, Any]:
        ...

    async def reject_next_train_queue(self, note: str | None = None) -> dict[str, Any]:
        ...

    async def promote_candidate(self, note: str | None = None) -> dict[str, Any]:
        ...

    async def archive_candidate(self, note: str | None = None) -> dict[str, Any]:
        ...

    async def candidate_history(self, limit: int = 10) -> dict[str, Any]:
        ...

    async def candidate_timeline(self, limit: int = 10) -> dict[str, Any]:
        ...

    async def train_queue_history(self, job_id: str | None = None, limit: int = 10) -> dict[str, Any]:
        ...


class MockInferenceService:
    async def generate_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        last_user = ""
        for message in reversed(request.messages):
            if message.role == "user" and message.content:
                last_user = message.content
                break
        reply = self._build_reply(last_user=last_user, metadata=request.metadata)
        completion_tokens = max(1, math.ceil(len(reply) / 4))
        prompt_tokens = max(1, math.ceil(sum(len(m.content or "") for m in request.messages) / 4))
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionResponseMessage(content=reply),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            request_id=request.request_id,
            session_id=request.session_id,
            adapter_version=request.adapter_version,
            served_by="mock",
            metadata={
                "note": "OpenAI-compatible inference only; personalization requires /pfe/signal.",
                "request_metadata": request.metadata,
            },
        )

    async def status(self) -> dict[str, Any]:
        return {"backend": "mock", "healthy": True}

    def on_user_action(
        self,
        session_id: str,
        action: str,
        next_message: str | None = None,
        edited_text: str | None = None,
        user_message: str | None = None,
        assistant_message: str | None = None,
    ) -> list[Any]:
        """Mock signal extraction from user actions.

        Returns mock signals that match the expected signal types and confidences.
        """
        from dataclasses import dataclass, field
        from datetime import datetime, timezone
        import uuid

        @dataclass
        class MockSignal:
            signal_id: str
            signal_type: str
            confidence: float
            extraction_rule: str
            timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

        # Map actions to signal types and confidences
        action_map = {
            "accept": ("accept", 0.7),
            "reject": ("reject", 0.95),
            "delete": ("reject", 0.95),
            "edit": ("edit", 0.8),
            "regenerate": ("regenerate", 0.85),
        }

        signal_type, confidence = action_map.get(action, (action, 0.7))

        # Adjust confidence based on response time for accept signals
        if signal_type == "accept":
            # This is handled in the feedback handler, but we could add logic here
            pass

        return [MockSignal(
            signal_id=f"mock-sig-{uuid.uuid4().hex[:8]}",
            signal_type=signal_type,
            confidence=confidence,
            extraction_rule=f"mock_{action}",
        )]

    def _build_reply(self, last_user: str, metadata: Mapping[str, Any]) -> str:
        style = str(metadata.get("style_hint", "helpful")) if metadata else "helpful"
        if last_user:
            return f"[mock-{style}] I heard: {last_user}"
        return f"[mock-{style}] Ready when you are."


class MockPipelineService:
    def __init__(self) -> None:
        self.signals: list[SignalIngestRequest] = []
        self.distill_runs: list[DistillRunRequest] = []

    async def ingest_signal(self, request: SignalIngestRequest) -> SignalIngestResponse:
        self.signals.append(request)
        return SignalIngestResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            adapter_version=request.adapter_version,
        )

    async def run_distillation(self, request: DistillRunRequest) -> DistillRunResponse:
        self.distill_runs.append(request)
        train_count, val_count, test_count = self._split_counts(request.num_samples, request.train_split, request.val_split, request.test_split)
        return DistillRunResponse(
            teacher_model=request.teacher_model,
            teacher_prompt_version=request.teacher_prompt_version,
            requested_samples=request.num_samples,
            generated_samples=request.num_samples,
            train_samples=train_count,
            val_samples=val_count,
            test_samples=test_count,
            metadata={
                "scenario": request.scenario,
                "style": request.style,
                "generation_config": request.generation_config,
                "source_event_ids": request.source_event_ids,
                "use_cloud_teacher": request.use_cloud_teacher,
                "note": "test split is reserved for evaluation and must not be used for training.",
            },
        )

    async def status(self) -> dict[str, Any]:
        return {"backend": "mock", "signals": len(self.signals), "distill_runs": len(self.distill_runs)}

    async def reset_auto_train_trigger(self) -> dict[str, Any]:
        return _fallback_status_snapshot()

    async def retry_auto_train_trigger(self) -> dict[str, Any]:
        return _fallback_status_snapshot()

    async def process_next_train_queue(self) -> dict[str, Any]:
        return _fallback_status_snapshot()

    async def process_train_queue_batch(self, limit: int = 5) -> dict[str, Any]:
        del limit
        return _fallback_status_snapshot()

    async def process_train_queue_until_idle(self, max_iterations: int | None = None) -> dict[str, Any]:
        del max_iterations
        return _fallback_status_snapshot()

    async def run_train_queue_worker_runner(
        self,
        max_seconds: float | None = None,
        idle_sleep_seconds: float | None = None,
    ) -> dict[str, Any]:
        del max_seconds, idle_sleep_seconds
        return _fallback_status_snapshot()

    async def stop_train_queue_worker_runner(self) -> dict[str, Any]:
        return _fallback_status_snapshot()

    async def train_queue_worker_runner_status(self) -> dict[str, Any]:
        return {"active": False, "stop_requested": False, "processed_count": 0, "failed_count": 0}

    async def train_queue_worker_runner_history(self, limit: int = 10) -> dict[str, Any]:
        del limit
        return {"count": 0, "items": [], "last_event": None, "last_reason": None}

    async def train_queue_worker_runner_timeline(self, limit: int = 10) -> dict[str, Any]:
        del limit
        return {
            "count": 0,
            "items": [],
            "current_stage": "idle",
            "transition_count": 0,
            "last_transition": {},
            "last_reason": None,
            "latest_timestamp": None,
        }

    async def start_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        del note
        return {
            "workspace": "user_default",
            "desired_state": "running",
            "observed_state": "starting",
            "command_status": "requested",
            "active": False,
            "lock_state": "idle",
            "history_count": 0,
        }

    async def stop_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        del note
        return {
            "workspace": "user_default",
            "desired_state": "stopped",
            "observed_state": "stopped",
            "command_status": "requested",
            "active": False,
            "lock_state": "idle",
            "history_count": 0,
        }

    async def train_queue_daemon_status(self) -> dict[str, Any]:
        return {
            "workspace": "user_default",
            "desired_state": "stopped",
            "observed_state": "stopped",
            "command_status": "idle",
            "active": False,
            "lock_state": "idle",
            "history_count": 0,
        }

    async def train_queue_daemon_history(self, limit: int = 10) -> dict[str, Any]:
        del limit
        return {"workspace": "user_default", "count": 0, "items": [], "last_event": None, "last_reason": None}

    async def recover_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        del note
        return {
            "workspace": "user_default",
            "desired_state": "running",
            "observed_state": "stopped",
            "command_status": "blocked",
            "active": False,
            "lock_state": "idle",
            "recovery_needed": False,
            "can_recover": False,
            "recovery_reason": "daemon_recovery_not_needed",
        }

    async def restart_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        del note
        return {
            "workspace": "user_default",
            "desired_state": "running",
            "observed_state": "starting",
            "command_status": "requested",
            "active": False,
            "lock_state": "idle",
            "recovery_state": "restarting",
        }

    async def approve_next_train_queue(self, note: str | None = None) -> dict[str, Any]:
        del note
        return _fallback_status_snapshot()

    async def reject_next_train_queue(self, note: str | None = None) -> dict[str, Any]:
        del note
        return _fallback_status_snapshot()

    async def promote_candidate(self, note: str | None = None) -> dict[str, Any]:
        del note
        snapshot = _fallback_status_snapshot()
        snapshot["candidate_action"] = {
            "action": "promote_candidate",
            "status": "noop",
            "reason": "no_candidate",
            "triggered": False,
        }
        return snapshot

    async def archive_candidate(self, note: str | None = None) -> dict[str, Any]:
        del note
        snapshot = _fallback_status_snapshot()
        snapshot["candidate_action"] = {
            "action": "archive_candidate",
            "status": "noop",
            "reason": "no_candidate",
            "triggered": False,
        }
        return snapshot

    async def candidate_history(self, limit: int = 10) -> dict[str, Any]:
        del limit
        return {"count": 0, "items": [], "last_action": None, "last_status": None}

    async def candidate_timeline(self, limit: int = 10) -> dict[str, Any]:
        del limit
        return {
            "count": 0,
            "items": [],
            "current_stage": "idle",
            "transition_count": 0,
            "last_transition": {},
            "last_reason": None,
            "last_candidate_version": None,
            "latest_timestamp": None,
        }

    async def train_queue_history(self, job_id: str | None = None, limit: int = 10) -> dict[str, Any]:
        del job_id, limit
        return {"count": 0, "history": [], "history_summary": {}, "available_job_ids": []}

    def _split_counts(self, total: int, train_split: float, val_split: float, test_split: float) -> tuple[int, int, int]:
        splits = [max(0.0, train_split), max(0.0, val_split), max(0.0, test_split)]
        split_total = sum(splits) or 1.0
        normalized = [value / split_total for value in splits]
        train_count = int(round(total * normalized[0]))
        val_count = int(round(total * normalized[1]))
        test_count = max(0, total - train_count - val_count)
        if train_count + val_count + test_count != total:
            test_count = max(0, total - train_count - val_count)
        return train_count, val_count, test_count


def _default_services() -> ServiceBundle:
    return ServiceBundle(
        inference=MockInferenceService(),
        pipeline=MockPipelineService(),
        security=ServerSecurityConfig(),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_core_import_path() -> Optional[Path]:
    candidate = _repo_root() / "pfe-core"
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        return candidate
    return None


def _load_core_services() -> Optional[ServiceBundle]:
    try:
        _ensure_core_import_path()
        from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter

        return ServiceBundle(
            inference=InferenceServiceAdapter(),
            pipeline=PipelineServiceAdapter(),
            security=ServerSecurityConfig(),
            provider="core",
        )
    except Exception:
        return None


def _select_default_services() -> ServiceBundle:
    core_bundle = _load_core_services()
    bundle = core_bundle if core_bundle is not None else _default_services()
    env_workspace = os.getenv("PFE_WORKSPACE")
    if env_workspace:
        bundle.workspace = env_workspace
    return bundle


def _json_response(payload: Any, status_code: int = 200) -> Any:
    if FASTAPI_AVAILABLE and JSONResponse is not None:
        return JSONResponse(content=payload, status_code=status_code)
    return payload, status_code


def _response_status_payload(response: Any) -> tuple[int, dict[str, Any]]:
    if isinstance(response, tuple):
        payload = response[0] if response else {}
        status_code = int(response[1]) if len(response) > 1 else 200
        return status_code, dict(payload) if isinstance(payload, Mapping) else {}
    status_code = int(getattr(response, "status_code", 200) or 200)
    body = getattr(response, "body", b"{}")
    try:
        payload = json.loads(body.decode("utf-8") if isinstance(body, bytes) else str(body))
    except Exception:
        payload = {}
    return status_code, payload if isinstance(payload, dict) else {}


def _frontend_html() -> str:
    html_path = _repo_root() / "pfe-server" / "pfe_server" / "static" / "chat.html"
    return html_path.read_text(encoding="utf-8")


def _studio_html() -> str:
    html_path = _repo_root() / "pfe-server" / "pfe_server" / "static" / "studio.html"
    return html_path.read_text(encoding="utf-8")


def _html_response(html: str, status_code: int = 200) -> Any:
    if FASTAPI_AVAILABLE and HTMLResponse is not None:
        return HTMLResponse(content=html, status_code=status_code)
    return html, status_code, {"content-type": "text/html; charset=utf-8"}


def _error_payload(decision_detail: str, code: str, hint: Optional[str] = None) -> dict[str, Any]:
    return ErrorResponse(detail=decision_detail, code=code, hint=hint).model_dump(mode="json")


def _load_status_snapshot(workspace: str) -> dict[str, Any]:
    try:
        _ensure_core_import_path()
        from pfe_core.adapter_store import create_adapter_store
        from pfe_core.storage import status_snapshot

        store = create_adapter_store(workspace=workspace)
        snapshot = status_snapshot(workspace=workspace)
        latest_version = store.current_latest_version()
        latest_adapter: dict[str, Any] = {}
        if latest_version:
            adapter_dir: Path | None = None
            manifest: dict[str, Any] = {}
            try:
                adapter_dir = Path(store.load(latest_version))
                manifest = json.loads((adapter_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
            except Exception:
                pass
            latest_adapter = {
                "version": latest_version,
                "state": manifest.get("state"),
                "artifact_format": manifest.get("artifact_format"),
                "base_model": manifest.get("base_model"),
                "created_at": manifest.get("created_at"),
                "path": str(adapter_dir) if adapter_dir is not None else None,
                "training_backend": manifest.get("training_backend"),
                "requires_export": manifest.get("requires_export"),
                "metadata": manifest.get("metadata") or {},
            }
        rows = store.list_version_records(limit=1000)
        snapshot["adapter_count"] = len(rows)
        snapshot["latest_adapter"] = latest_adapter
        snapshot["latest_adapter_version"] = latest_version
        snapshot["latest_adapter_exists"] = bool(latest_version)
        recent_row = rows[0] if rows else None
        snapshot["recent_adapter_version"] = recent_row.get("version") if recent_row else None
        snapshot["recent_adapter"] = _snapshot_adapter_row(
            recent_row,
            latest=bool(recent_row and recent_row.get("version") == latest_version),
        ) if recent_row else {}
        pending_rows = [row for row in rows if row.get("state") == "pending_eval"]
        training_rows = [row for row in rows if row.get("state") == "training"]
        failed_rows = [row for row in rows if row.get("state") == "failed_eval"]
        candidate_row = None
        for row in rows:
            version = str(row.get("version") or "")
            if latest_version and version == str(latest_version) and row.get("state") == "promoted":
                continue
            candidate_row = row
            break
        snapshot["candidate_summary"] = {
            "latest_promoted_version": latest_version,
            "recent_version": recent_row.get("version") if recent_row else None,
            "candidate_version": candidate_row.get("version") if candidate_row else None,
            "candidate_state": candidate_row.get("state") if candidate_row else None,
            "candidate_exists": bool(candidate_row),
            "candidate_can_promote": bool(candidate_row and candidate_row.get("state") in {"training", "pending_eval", "failed_eval"}),
            "candidate_can_archive": bool(candidate_row and candidate_row.get("state") not in {"archived"}),
            "pending_eval_count": len(pending_rows),
            "pending_eval_versions": [row.get("version") for row in pending_rows],
            "training_count": len(training_rows),
            "training_versions": [row.get("version") for row in training_rows],
            "failed_eval_count": len(failed_rows),
            "failed_eval_versions": [row.get("version") for row in failed_rows],
            "has_pending_candidate": bool(pending_rows or training_rows),
            "candidate_needs_promotion": bool(candidate_row and candidate_row.get("state") == "pending_eval"),
        }
        snapshot["candidate_action"] = {
            "action": "promote_candidate",
            "status": "noop",
            "reason": "no_candidate_action",
            "triggered": False,
        }
        snapshot["train_queue"] = {"count": 0, "counts": {}, "current": {}, "last_item": {}, "items": []}
        snapshot["serve"] = {
            "adapter_resolution_state": "latest_promoted" if latest_version else "no_promoted_latest",
            "using_promoted_adapter": bool(latest_version),
            "target_adapter_version": latest_version,
            "fallback_reason": None if latest_version else "no_promoted_latest",
        }
        snapshot["adapter_lifecycle"] = {
            "counts": {
                state: sum(1 for row in rows if row.get("state") == state)
                for state in {"training", "pending_eval", "promoted", "failed_eval", "archived"}
            },
            "latest_version": latest_version,
            "recent_version": recent_row.get("version") if recent_row else None,
            "pending_eval_versions": [row.get("version") for row in rows if row.get("state") == "pending_eval"],
            "promoted_versions": [row.get("version") for row in rows if row.get("state") == "promoted"],
            "failed_eval_versions": [row.get("version") for row in rows if row.get("state") == "failed_eval"],
            "archived_versions": [row.get("version") for row in rows if row.get("state") == "archived"],
        }
        snapshot["provider"] = "core"
        return snapshot
    except Exception:
        return {
            "provider": "mock",
            "sample_counts": {"train": 0, "val": 0, "test": 0},
            "adapter_count": 0,
            "signal_count": 0,
            "latest_adapter": {},
            "latest_adapter_version": None,
        }


def _route_access(
    envelope: RequestEnvelope,
    *,
    security: ServerSecurityConfig,
    endpoint_kind: str,
    cloud_requested: bool = False,
) -> tuple[bool, Any]:
    decision = authorize_request(
        AccessContext(
            path=envelope.path,
            method=envelope.method,
            client_host=envelope.client_host,
            headers=envelope.headers,
            endpoint_kind=endpoint_kind,
            cloud_requested=cloud_requested,
        ),
        security,
    )
    if decision.allowed:
        return True, decision
    return False, _json_response(
        _error_payload(
            decision.detail,
            decision.code,
            decision.hint,
        ),
        status_code=decision.status_code,
    )


def _extract_status_counts(status_payload: dict[str, Any]) -> dict[str, int]:
    raw_counts = status_payload.get("sample_counts") or {}
    return {
        "train": int(raw_counts.get("train", 0) or 0),
        "val": int(raw_counts.get("val", 0) or 0),
        "test": int(raw_counts.get("test", 0) or 0),
    }


def _build_runtime_payload(envelope: RequestEnvelope, services: ServiceBundle) -> dict[str, Any]:
    return build_studio_runtime_payload(
        headers=envelope.headers,
        runtime_probe=services.runtime_probe,
        workspace=services.workspace,
        provider=services.provider,
        allow_remote_access=services.security.allow_remote_access,
        privacy_mode=services.security.privacy_mode,
        auth_mode=services.security.auth_mode,
        started_at=services.started_at,
    )


def _workspace_slug_issues(name: str) -> list[dict[str, str]]:
    return studio_workspace_slug_issues(name)


def _workspace_paths(name: str) -> dict[str, Path]:
    return studio_workspace_paths(_resolve_pfe_home(), name)


def _workspace_record(name: str, *, current: str) -> dict[str, Any]:
    return studio_workspace_record(_resolve_pfe_home(), name, current=current)


def _set_active_workspace(services: ServiceBundle, name: str) -> tuple[str, str]:
    previous = services.workspace
    services.workspace = name
    os.environ["PFE_WORKSPACE"] = name
    return previous, services.workspace


def _build_workspaces_payload(services: ServiceBundle) -> dict[str, Any]:
    return build_studio_workspaces_payload(
        _resolve_pfe_home(),
        str(services.workspace or "user_default"),
        env_workspace=os.getenv("PFE_WORKSPACE"),
    )


def _load_config_base_model() -> str:
    try:
        _ensure_core_import_path()
        from pfe_core.config import PFEConfig

        return str(PFEConfig.load(home=_resolve_pfe_home()).model.base_model)
    except Exception:
        return _default_chat_base_model()


def _config_path_label() -> str | None:
    try:
        _ensure_core_import_path()
        from pfe_core.config import PFEConfig

        return str(PFEConfig.default_path(home=_resolve_pfe_home()))
    except Exception:
        return None


def _model_candidate(model_id: str, *, source: str, selected: str | None = None) -> dict[str, Any]:
    return studio_model_candidate(model_id, source=source, selected=selected)


def _build_models_payload(services: ServiceBundle) -> dict[str, Any]:
    del services
    return build_studio_models_payload(
        _load_config_base_model(),
        env_model=os.getenv("PFE_REAL_LOCAL_MODEL"),
        repo_models_dir=_repo_root() / "models",
    )


def _selected_model_handoff(models_payload: Mapping[str, Any]) -> dict[str, Any]:
    selected = str(models_payload.get("selected") or "")
    selected_label = str(models_payload.get("selected_label") or selected or "local")
    selected_candidate: Mapping[str, Any] = {}
    for candidate in list(models_payload.get("candidates") or []):
        if isinstance(candidate, Mapping) and str(candidate.get("id") or "") == selected:
            selected_candidate = candidate
            break
    return {
        "selected": selected,
        "label": str(selected_candidate.get("label") or selected_label),
        "source": selected_candidate.get("source"),
        "local_path": selected_candidate.get("local_path"),
        "exists": selected_candidate.get("exists"),
    }


def _build_handoff_payload(envelope: RequestEnvelope, services: ServiceBundle) -> dict[str, Any]:
    runtime = _build_runtime_payload(envelope, services)
    models = _build_models_payload(services)
    adapters = _build_adapters_payload(services)
    api = runtime.get("api") if isinstance(runtime.get("api"), Mapping) else {}
    current_version = None
    if isinstance(adapters.get("current"), Mapping):
        current_version = adapters["current"].get("version")
    payload: dict[str, Any] = {
        "kind": "pfe_studio_handoff",
        "workspace": runtime.get("workspace"),
        "status": runtime.get("status"),
        "summary": "复制后可直接接入本机网页或 OpenAI-compatible chat API。",
        "urls": {
            "web": runtime.get("web_url"),
            "studio": runtime.get("studio_url"),
            "api": runtime.get("api_url"),
            "feedback": api.get("feedback_url") or f"{runtime.get('base_url')}/pfe/feedback",
            "dashboard": runtime.get("dashboard_url"),
        },
        "model": {
            **_selected_model_handoff(models),
            "api_parameter": api.get("model_parameter") or "local",
            "aliases": list(api.get("model_aliases") or []),
        },
        "version": {
            "current": current_version,
            "latest": adapters.get("latest_version"),
            "count": int(adapters.get("count") or 0),
            "pending_count": len(list(adapters.get("pending") or [])),
        },
        "api": api,
        "runtime": {
            "provider": runtime.get("provider"),
            "access_scope": runtime.get("access_scope"),
            "auth_mode": runtime.get("auth_mode"),
            "api_key_required": runtime.get("api_key_required"),
        },
    }
    payload["closed_loop"] = build_handoff_closed_loop(payload)
    payload["snippets"] = {
        "javascript": build_handoff_javascript_snippet(payload),
        "python": build_handoff_python_snippet(payload),
    }
    payload["copy_text"] = build_handoff_copy_text(payload)
    return payload


def _env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _set_real_local_inference(enabled: bool) -> tuple[bool, bool]:
    previous = _env_flag_enabled("PFE_ENABLE_REAL_LOCAL_INFERENCE")
    if enabled:
        os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"
    else:
        os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
    return previous, _env_flag_enabled("PFE_ENABLE_REAL_LOCAL_INFERENCE")


def _module_available(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _selected_model_candidate(models_payload: Mapping[str, Any]) -> dict[str, Any]:
    selected = str(models_payload.get("selected") or "")
    for candidate in list(models_payload.get("candidates") or []):
        if str(candidate.get("id") or "") == selected:
            return dict(candidate)
    return _model_candidate(selected, source="config", selected=selected)


def _model_source_readiness(candidate: Mapping[str, Any]) -> dict[str, Any]:
    local_path = candidate.get("local_path")
    exists = bool(candidate.get("exists"))
    model_id = str(candidate.get("id") or "")
    if local_path:
        return {
            "ok": exists,
            "state": "ready" if exists else "missing",
            "label": "本地模型已找到" if exists else "本地模型不存在",
            "reason": None if exists else "selected local model path does not exist",
            "path": local_path,
        }
    return {
        "ok": False,
        "state": "needs_local_path",
        "label": "需要本地模型路径",
        "reason": "real local inference requires a local base model path",
        "path": None,
        "model_id": model_id,
    }


def _runtime_dependency_readiness() -> dict[str, Any]:
    modules = {
        "torch": _module_available("torch"),
        "transformers": _module_available("transformers"),
        "llama_cpp": _module_available("llama_cpp"),
        "peft": _module_available("peft"),
    }
    transformers_ready = bool(modules["torch"] and modules["transformers"])
    llama_cpp_ready = bool(modules["llama_cpp"])
    return {
        "ok": transformers_ready or llama_cpp_ready,
        "transformers_ready": transformers_ready,
        "llama_cpp_ready": llama_cpp_ready,
        "modules": modules,
        "preferred_runtime": "transformers" if transformers_ready else ("llama_cpp" if llama_cpp_ready else None),
        "missing_core_modules": [
            name
            for name in ("torch", "transformers")
            if not modules[name]
        ],
    }


def _build_readiness_payload(envelope: RequestEnvelope, services: ServiceBundle) -> dict[str, Any]:
    runtime = _build_runtime_payload(envelope, services)
    models = _build_models_payload(services)
    adapters = _build_adapters_payload(services)
    selected_candidate = _selected_model_candidate(models)
    selected_model = str(models.get("selected") or "")
    model_source = _model_source_readiness(selected_candidate)
    runtime_deps = _runtime_dependency_readiness()
    real_local_enabled = _env_flag_enabled("PFE_ENABLE_REAL_LOCAL_INFERENCE")
    real_local_ready = bool(real_local_enabled and model_source["ok"] and runtime_deps["ok"])
    current_adapter = adapters.get("current")

    blockers: list[str] = []
    next_actions: list[dict[str, str]] = []
    if not real_local_enabled:
        blockers.append("real_local_inference_disabled")
        next_actions.append(
            {
                "id": "enable_real_local_inference",
                "label": "开启真实本地模型",
                "detail": "在 Studio 中开启真实本地模型；当前服务进程内立即生效。",
            }
        )
    if not model_source["ok"]:
        blockers.append(str(model_source["state"]))
        next_actions.append(
            {
                "id": "choose_local_model",
                "label": "选择本地模型",
                "detail": "在模型选择里使用已存在的本地模型目录。",
            }
        )
    if not runtime_deps["ok"]:
        blockers.append("runtime_dependencies_missing")
        next_actions.append(
            {
                "id": "install_runtime_dependencies",
                "label": "补齐本地推理依赖",
                "detail": "真实本地模型需要 torch+transformers 或 llama.cpp 运行时。",
            }
        )
    if current_adapter is None:
        next_actions.append(
            {
                "id": "continue_without_version",
                "label": "继续使用基础模型",
                "detail": "当前没有已使用的个性化版本，仍可用基础模型或模板回复。",
            }
        )

    inference_mode = "real_local" if real_local_ready else "template"
    inference_label = "真实本地模型" if real_local_ready else "模板回复"
    summary_state = "ready" if real_local_ready else "needs_setup"
    summary_label = "可继续" if real_local_ready else "需确认"
    if real_local_ready:
        summary_text = "本机服务已就绪，回复来自真实本地模型。"
    elif not real_local_enabled:
        summary_text = "本机服务已就绪，当前回复来自模板回复；真实本地模型尚未开启。"
    elif not model_source["ok"]:
        summary_text = "本机服务已就绪，但真实本地模型需要一个可用的本地模型目录。"
    else:
        summary_text = "本机服务已就绪，但真实本地推理依赖还不完整。"

    return {
        "summary": {
            "state": summary_state,
            "label": summary_label,
            "text": summary_text,
            "blockers": blockers,
        },
        "inference": {
            "mode": inference_mode,
            "mode_label": inference_label,
            "real_local_enabled": real_local_enabled,
            "real_local_ready": real_local_ready,
            "provider": services.provider,
            "control_api": "PUT /pfe/config/real-local",
            "effective_scope": "current_process_next_request",
        },
        "model": {
            "selected": selected_model,
            "selected_label": models.get("selected_label"),
            "candidate": selected_candidate,
            "source": model_source,
        },
        "configuration": {
            "base_model": selected_model,
            "effective_scope": "next_chat_request",
            "reload_required": False,
            "applies_to_models": ["local", "local-default", "base"],
            "config_path": _config_path_label(),
        },
        "runtime": {
            "service": runtime,
            "dependencies": runtime_deps,
        },
        "version": {
            "current": current_adapter,
            "latest_version": adapters.get("latest_version"),
            "pending_count": len(list(adapters.get("pending") or [])),
            "count": adapters.get("count", 0),
        },
        "next_actions": next_actions,
        "checks": {
            "server": {"ok": runtime.get("status") == "passed", "label": "本机服务"},
            "model_source": {"ok": bool(model_source["ok"]), "label": model_source["label"]},
            "runtime_dependencies": {"ok": bool(runtime_deps["ok"]), "label": "本地推理依赖"},
            "real_local_flag": {"ok": real_local_enabled, "label": "真实本地模型开关"},
            "model_configuration": {"ok": True, "label": "模型配置"},
            "model_version": {"ok": current_adapter is not None, "label": "模型版本"},
        },
    }


def _query_bool(query_params: Mapping[str, str], key: str) -> bool:
    return str(query_params.get(key, "")).strip().lower() in {"1", "true", "yes", "on"}


def _body_bool(body: Mapping[str, Any], key: str) -> bool:
    value = body.get(key)
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_request_json(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body or b"{}")
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _build_training_preflight_payload(
    envelope: RequestEnvelope,
    services: ServiceBundle,
    request: TrainingJobRequest,
) -> dict[str, Any]:
    return build_studio_training_preflight_payload(
        request=request,
        readiness=_build_readiness_payload(envelope, services),
        workspace=services.workspace,
        base_model=_load_config_base_model(),
    )


def _build_legacy_training_trigger_preflight_payload(
    services: ServiceBundle,
    request: TrainingJobRequest,
) -> dict[str, Any]:
    return build_studio_legacy_training_trigger_preflight_payload(
        request=request,
        workspace=services.workspace,
        base_model=_load_config_base_model(),
    )


def _training_job_payload(job_entry: Mapping[str, Any]) -> dict[str, Any]:
    return studio_training_job_payload(job_entry)


def _training_job_event(
    *,
    job_id: str,
    event_type: str,
    status: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return studio_training_job_event(
        job_id=job_id,
        event_type=event_type,
        status=status,
        message=message,
        metadata=metadata,
    )


def _append_training_job_event(
    job_entry: dict[str, Any],
    *,
    event_type: str,
    status: str,
    message: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return append_studio_training_job_event(
        job_entry,
        event_type=event_type,
        status=status,
        message=message,
        metadata=metadata,
    )


def _latest_training_event_type(job_entry: Mapping[str, Any]) -> str | None:
    return latest_studio_training_event_type(job_entry)


def _build_training_jobs_payload(services: ServiceBundle, *, limit: int = 20) -> dict[str, Any]:
    return _training_job_store(services.workspace).build_jobs_payload(limit=limit)


def _validate_model_reference(model_id: str) -> dict[str, Any]:
    return validate_studio_model_reference(model_id)


def _save_config_base_model(model_id: str) -> tuple[str, Path]:
    _ensure_core_import_path()
    from pfe_core.config import PFEConfig

    home = _resolve_pfe_home()
    config = PFEConfig.load(home=home)
    previous = str(config.model.base_model)
    config.model.base_model = model_id
    config_path = config.save(home=home)
    return previous, config_path


def _coerce_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, Mapping) else {}
        except Exception:
            return {}
    return {}


def _compact_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _selected_metrics(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value is not None:
            selected[key] = value
    return selected


def _adapter_training_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _coerce_json_mapping(row.get("metrics"))
    try:
        num_samples = int(row.get("num_samples") or 0)
    except Exception:
        num_samples = 0
    selected = _selected_metrics(
        metrics,
        (
            "loss",
            "train_loss",
            "eval_loss",
            "num_fresh_samples",
            "num_replay_samples",
            "preference_reinforced_fresh_sample_count",
        ),
    )
    parts = [f"训练样本 {num_samples}"]
    for key, value in list(selected.items())[:2]:
        parts.append(f"{key} {_compact_metric_value(value)}")
    return {
        "num_samples": num_samples,
        "metrics": selected,
        "summary_line": " / ".join(parts),
    }


def _adapter_eval_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    raw_state = str(row.get("state") or "").lower()
    report = _coerce_json_mapping(row.get("eval_report"))
    if not report:
        if raw_state in {"pending_eval", "training"}:
            label = "待评估"
            state = "not_run"
        elif raw_state == "promoted":
            label = "已设为当前"
            state = "current"
        elif raw_state == "archived":
            label = "已归档"
            state = "archived"
        else:
            label = "暂无评估"
            state = "missing"
        return {
            "state": state,
            "label": label,
            "recommendation": None,
            "comparison": None,
            "scores": {},
            "summary_line": f"评估结论：{label}",
        }

    recommendation = str(report.get("recommendation") or "").lower()
    comparison = report.get("comparison")
    scores = _coerce_json_mapping(report.get("scores"))
    selected_scores = _selected_metrics(
        scores,
        (
            "style_match",
            "preference_alignment",
            "personality_consistency",
            "style_preference_hit_rate",
            "quality_preservation",
            "generic_quality_score",
        ),
    )
    if recommendation == "deploy":
        label = "评估通过"
        state = "passed"
    elif recommendation == "keep_previous":
        label = "有问题"
        state = "failed"
    elif recommendation == "needs_more_data":
        label = "需更多数据"
        state = "needs_more_data"
    else:
        label = "已评估"
        state = "completed"
    parts = [f"评估结论：{label}"]
    if comparison:
        parts.append(str(comparison))
    for key, value in list(selected_scores.items())[:2]:
        parts.append(f"{key} {_compact_metric_value(value)}")
    return {
        "state": state,
        "label": label,
        "recommendation": recommendation or None,
        "comparison": comparison,
        "scores": selected_scores,
        "summary_line": " / ".join(parts),
    }


def _adapter_decision_summary(
    row: Mapping[str, Any],
    *,
    latest_version: str | None,
    eval_summary: Mapping[str, Any],
) -> dict[str, Any]:
    version = str(row.get("version") or "")
    raw_state = str(row.get("state") or "").lower()
    recommendation = str(eval_summary.get("recommendation") or "").lower()
    if latest_version and version == str(latest_version):
        label = "当前使用中"
        return {"label": label, "tone": "ok", "primary_action": None, "summary_line": "当前：使用中"}
    if recommendation == "deploy":
        label = "建议设为当前"
        return {"label": label, "tone": "ok", "primary_action": "promote", "summary_line": "建议：设为当前"}
    if recommendation == "keep_previous" or raw_state == "failed_eval":
        label = "建议保留旧版"
        return {"label": label, "tone": "bad", "primary_action": "archive", "summary_line": "建议：保留旧版"}
    if recommendation == "needs_more_data":
        label = "先补样本或手动确认"
        return {"label": label, "tone": "warn", "primary_action": None, "summary_line": f"建议：{label}"}
    if raw_state in {"pending_eval", "training"}:
        label = "可手动确认"
        return {"label": label, "tone": "warn", "primary_action": "promote", "summary_line": f"建议：{label}"}
    if raw_state == "promoted":
        label = "可回退到此版本"
        return {"label": label, "tone": "warn", "primary_action": "rollback", "summary_line": f"建议：{label}"}
    if raw_state == "archived":
        label = "已归档"
        return {"label": label, "tone": "", "primary_action": None, "summary_line": "状态：已归档"}
    label = "待确认"
    return {"label": label, "tone": "warn", "primary_action": None, "summary_line": f"建议：{label}"}


def _current_eval_state(workspace: str) -> dict[str, Any]:
    state = _eval_overall_state.get(workspace)
    if state is None:
        state = _load_json_state(_eval_state_path(workspace))
    return dict(state) if isinstance(state, Mapping) else {}


def _adapter_user_state(row: Mapping[str, Any], *, latest_version: str | None) -> str:
    version = str(row.get("version") or "")
    raw_state = str(row.get("state") or "").lower()
    if latest_version and version == str(latest_version):
        return "使用中"
    if raw_state in {"pending_eval", "training"}:
        return "待确认"
    if raw_state == "promoted":
        return "可回退"
    if raw_state == "archived":
        return "已归档"
    if raw_state == "failed_eval":
        return "有问题"
    return "待验证"


def _build_adapters_payload(services: ServiceBundle) -> dict[str, Any]:
    try:
        _ensure_core_import_path()
        from pfe_core.adapter_store import create_adapter_store

        store = create_adapter_store(workspace=services.workspace)
        latest_version = store.current_latest_version()
        rows = store.list_version_records(limit=100)
        eval_state = _current_eval_state(services.workspace)
        running_version = running_eval_version(eval_state)
        versions = []
        for row in rows:
            item = _snapshot_adapter_row(row, latest=bool(latest_version and row.get("version") == latest_version))
            raw_state = str(row.get("state") or "").lower()
            eval_summary = _adapter_eval_summary(row)
            version = str(item.get("version") or "")
            eval_running = bool(running_version and version == running_version)
            if eval_running:
                eval_summary = running_eval_summary()
            item["user_state"] = _adapter_user_state(row, latest_version=latest_version)
            item["training_summary"] = _adapter_training_summary(row)
            item["eval_summary"] = eval_summary
            item["decision"] = _adapter_decision_summary(row, latest_version=latest_version, eval_summary=eval_summary)
            item["can_promote"] = item["user_state"] in {"待确认", "可回退"}
            item["can_rollback"] = bool(not item.get("latest") and raw_state in {"promoted", "archived"})
            item["can_archive"] = bool(not item.get("latest") and raw_state != "archived")
            item["can_eval"] = bool(raw_state in {"pending_eval", "failed_eval", "promoted"} and not eval_running)
            item["eval_running"] = eval_running
            item["action_api"] = {
                "promote": f"/pfe/adapters/{item.get('version')}/promote",
                "rollback": f"/pfe/adapters/{item.get('version')}/rollback",
                "archive": f"/pfe/adapters/{item.get('version')}/archive",
                "eval": "/pfe/eval",
            }
            versions.append(item)
        pending = [item for item in versions if item.get("user_state") == "待确认"]
        fallback = next((item for item in versions if item.get("user_state") == "可回退"), None)
        return {
            "latest_version": latest_version,
            "current": next((item for item in versions if item.get("latest")), None),
            "pending": pending,
            "fallback": fallback,
            "versions": versions,
            "count": len(versions),
        }
    except Exception as exc:
        return {
            "latest_version": None,
            "current": None,
            "pending": [],
            "fallback": None,
            "versions": [],
            "count": 0,
            "error": str(exc),
        }


def _build_status_export_metadata(snapshot: Mapping[str, Any], services: ServiceBundle) -> dict[str, Any]:
    latest_adapter = snapshot.get("latest_adapter") or {}
    latest_metadata = dict(latest_adapter.get("metadata") or {})
    training_metadata = dict(latest_metadata.get("training") or {})
    backend_plan = dict(latest_metadata.get("backend_plan") or {})
    export_artifact_summary = dict(
        latest_metadata.get("export_artifact_summary")
        or dict(latest_metadata.get("export") or {}).get("artifact")
        or {}
    )
    trainer_snapshot = dict(snapshot.get("trainer") or {})
    last_run = dict(trainer_snapshot.get("last_run") or {})
    last_export_runtime = dict(last_run.get("export_runtime") or {})
    last_export_execution = dict(last_run.get("export_execution") or {})
    last_export_write = dict(last_run.get("export_write") or {})
    output_artifact_validation = dict(last_export_execution.get("output_artifact_validation") or {})
    command_metadata = dict(last_export_execution.get("command_metadata") or last_run.get("export_command_plan", {}).get("command_metadata") or {})
    tool_resolution = dict(last_export_execution.get("tool_resolution") or last_run.get("export_command_plan", {}).get("tool_resolution") or {})
    payload = {
        "workspace": services.workspace,
        "provider": services.provider,
        "latest_adapter_version": snapshot.get("latest_adapter_version"),
        "latest_adapter_state": latest_adapter.get("state"),
        "artifact_format": latest_adapter.get("artifact_format"),
        "artifact_path": latest_adapter.get("path"),
        "export_target": "gguf_merged" if latest_adapter.get("artifact_format") == "gguf_merged" else "adapter_manifest",
        "training_backend": latest_adapter.get("training_backend") or training_metadata.get("backend"),
        "requires_export_step": latest_adapter.get("requires_export"),
        "runtime_device": training_metadata.get("runtime_device"),
        "recommended_backend": backend_plan.get("recommended_backend"),
        "output_artifact_path": export_artifact_summary.get("path"),
        "output_artifact_valid": export_artifact_summary.get("valid"),
        "output_artifact_size_bytes": export_artifact_summary.get("size_bytes"),
        "export_artifact_path": export_artifact_summary.get("path"),
        "export_artifact_valid": export_artifact_summary.get("valid"),
        "export_artifact_size_bytes": export_artifact_summary.get("size_bytes"),
        "converter_kind": export_artifact_summary.get("converter_kind"),
        "outtype": export_artifact_summary.get("outtype"),
        "toolchain_resolved_path": export_artifact_summary.get("tool_path"),
    }
    if last_export_runtime or last_export_execution or last_export_write:
        payload.update(
            {
                "recommended_backend": last_export_runtime.get("target_backend", payload.get("recommended_backend")),
                "requires_export_step": bool(
                    last_run.get("requires_export_step", last_export_runtime.get("required", payload.get("requires_export_step")))
                ),
                "artifact_name": last_export_runtime.get("artifact_name"),
                "artifact_directory": last_export_write.get("output_dir") or last_export_runtime.get("output_dir"),
                "output_dir": last_export_write.get("output_dir") or last_export_runtime.get("output_dir"),
                "command": list(last_run.get("export_command_plan", {}).get("command") or []),
                "execution_status": (
                    last_export_execution.get("audit", {}).get("status")
                    if isinstance(last_export_execution.get("audit"), dict)
                    else None
                ),
                "output_artifact_path": (
                    output_artifact_validation.get("path")
                    or last_export_execution.get("output_artifact_path")
                    or last_export_write.get("artifact_path")
                    or payload.get("output_artifact_path")
                ),
                "output_artifact_valid": output_artifact_validation.get("valid", payload.get("output_artifact_valid")),
                "output_artifact_size_bytes": output_artifact_validation.get(
                    "size_bytes",
                    payload.get("output_artifact_size_bytes"),
                ),
                "export_artifact_path": (
                    output_artifact_validation.get("path")
                    or last_export_execution.get("output_artifact_path")
                    or last_export_write.get("artifact_path")
                    or payload.get("export_artifact_path")
                ),
                "export_artifact_valid": output_artifact_validation.get(
                    "valid",
                    payload.get("export_artifact_valid"),
                ),
                "export_artifact_size_bytes": output_artifact_validation.get(
                    "size_bytes",
                    payload.get("export_artifact_size_bytes"),
                ),
                "converter_kind": command_metadata.get("converter_kind", payload.get("converter_kind")),
                "outtype": command_metadata.get("outtype", payload.get("outtype")),
                "toolchain_resolved_path": tool_resolution.get(
                    "resolved_path",
                    payload.get("toolchain_resolved_path"),
                ),
                "materialized": bool(last_export_write.get("metadata", {}).get("materialized", False)),
                "write_state": last_export_write.get("write_state")
                or last_export_write.get("metadata", {}).get("write_state"),
            }
        )
    try:
        from pfe_core.inference import build_export_runtime_spec, materialize_export_plan

        target_backend = "llama_cpp" if latest_adapter.get("artifact_format") == "gguf_merged" else "transformers"
        runtime_spec = build_export_runtime_spec(
            target_backend=target_backend,
            source_artifact_format=latest_adapter.get("artifact_format"),
            adapter_dir=latest_adapter.get("path"),
            workspace=services.workspace,
            source_adapter_version=snapshot.get("latest_adapter_version"),
            source_model=latest_adapter.get("base_model"),
        )
        materialized = materialize_export_plan(
            target_backend=target_backend,
            source_artifact_format=latest_adapter.get("artifact_format"),
            adapter_dir=latest_adapter.get("path"),
            workspace=services.workspace,
            source_adapter_version=snapshot.get("latest_adapter_version"),
            source_model=latest_adapter.get("base_model"),
        )
        payload.update(
            {
                "required": runtime_spec.required,
                "artifact_name": runtime_spec.artifact_name,
                "artifact_directory": payload.get("artifact_directory") or materialized.artifact_directory,
                "output_dir": payload.get("output_dir") or materialized.output_dir,
                "constraint": runtime_spec.constraint,
                "recommended_backend": payload.get("recommended_backend") or target_backend,
                "requires_export_step": bool(payload.get("requires_export_step", runtime_spec.required)),
                "placeholder_files": materialized.placeholder_files,
                "materialized": payload.get("materialized", getattr(materialized, "materialized", True)),
                "write_state": payload.get(
                    "write_state",
                    getattr(
                        materialized,
                        "write_state",
                        "materialized" if getattr(materialized, "placeholder_files", []) else "pending",
                    ),
                ),
                "manifest_patch_description": materialized.manifest_patch_description,
            }
        )
    except Exception:
        payload.update(
            {
                "recommended_backend": "llama_cpp" if latest_adapter.get("artifact_format") == "gguf_merged" else "transformers",
                "requires_export_step": latest_adapter.get("artifact_format") != "gguf_merged",
                "artifact_directory": latest_adapter.get("path") or str(_default_status_workspace_dir(services.workspace)),
                "output_dir": latest_adapter.get("path") or str(_default_status_workspace_dir(services.workspace)),
                "placeholder_files": [],
                "materialized": False,
                "write_state": "pending",
            }
        )
        pass
    return payload


def _default_status_workspace_dir(workspace: str) -> Path:
    base_home = _resolve_pfe_home()
    return base_home / "workspaces" / workspace


# In-memory runtime state for training/eval jobs (per-process; sufficient for tests)
_training_jobs_state: dict[str, dict[str, Any]] = {}
_training_overall_state: dict[str, dict[str, Any]] = {}
_eval_overall_state: dict[str, dict[str, Any]] = {}


def _training_jobs_path(workspace: str) -> Path:
    return _default_status_workspace_dir(workspace) / "training_jobs.json"


def _training_state_path(workspace: str) -> Path:
    return _default_status_workspace_dir(workspace) / "training_status.json"


def _eval_state_path(workspace: str) -> Path:
    return _default_status_workspace_dir(workspace) / "eval_status.json"


def _load_json_state(path: Path) -> dict[str, Any]:
    return load_studio_json_state(path)


def _save_json_state(path: Path, payload: dict[str, Any]) -> None:
    save_studio_json_state(path, payload)


def _training_job_store(workspace: str) -> StudioTrainingJobStore:
    return StudioTrainingJobStore(
        workspace=workspace,
        workspace_dir=_default_status_workspace_dir(workspace),
        memory_jobs=_training_jobs_state,
        overall_state=_training_overall_state,
    )


def _load_latest_training_artifacts(snapshot: Mapping[str, Any], services: ServiceBundle) -> dict[str, Any]:
    latest_adapter = snapshot.get("latest_adapter") or {}
    adapter_path = latest_adapter.get("path")
    if not adapter_path:
        return {}
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        return {}
    payload: dict[str, Any] = {"adapter_dir": str(adapter_dir)}
    for filename, key in (("training_meta.json", "training_meta"), ("adapter_manifest.json", "manifest")):
        candidate = adapter_dir / filename
        if not candidate.exists():
            payload[key] = {}
            continue
        try:
            payload[key] = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            payload[key] = {}
    return payload


async def _collect_runtime_probe_checks(app_obj: Any, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "state": "skipped",
            "reason": "dry_run=True",
            "checks": [],
        }
    checks: list[dict[str, Any]] = []
    for method, path, body in [
        ("GET", "/healthz", None),
        ("GET", "/pfe/status", None),
    ]:
        result = await smoke_test_request(app_obj, path=path, method=method, body=body)
        checks.append(
            {
                "method": method,
                "path": path,
                "status_code": result["status_code"],
                "ok": int(result["status_code"]) == 200,
                "response_body_keys": sorted(result.get("body", {}).keys()) if isinstance(result.get("body"), dict) else [],
            }
        )
    last_result = checks[-1] if checks else {}
    probe_ok = all(item.get("ok") for item in checks)
    return {
        "state": "ok" if probe_ok else "degraded",
        "checks": checks,
        "last_serve_check": last_result,
        "summary": {
            "checked_paths": [item["path"] for item in checks],
            "healthy_checks": sum(1 for item in checks if item.get("ok")),
            "total_checks": len(checks),
        },
    }


def _build_status_trainer_metadata(snapshot: Mapping[str, Any], services: ServiceBundle) -> dict[str, Any]:
    export_metadata = _build_status_export_metadata(snapshot, services)
    latest_adapter = snapshot.get("latest_adapter") or {}
    trainer_snapshot = dict(snapshot.get("trainer") or {})
    last_run = dict(trainer_snapshot.get("last_run") or {})
    artifact_payload = _load_latest_training_artifacts(snapshot, services)
    training_meta = dict(artifact_payload.get("training_meta") or {})
    job_bundle = dict(last_run.get("job_bundle") or {})
    job_execution = dict(last_run.get("job_execution") or {})
    real_execution_summary = dict(training_meta.get("real_execution_summary") or {})
    if not real_execution_summary:
        try:
            from pfe_core.trainer.executors import summarize_real_training_execution

            real_execution_summary = summarize_real_training_execution(job_execution)
        except Exception:
            real_execution_summary = {}
    if not real_execution_summary:
        audit = dict(job_execution.get("audit") or {})
        metadata = dict(job_execution.get("metadata") or {})
        real_execution_summary = {
            "state": job_execution.get("status") or audit.get("status") or "unknown",
            "kind": None,
            "backend": job_execution.get("backend") or last_run.get("execution_backend"),
            "runner_status": job_execution.get("status") or audit.get("status"),
            "execution_mode": metadata.get("execution_state") or job_execution.get("execution_mode"),
            "attempted": bool(job_execution.get("attempted", False)) if job_execution else False,
            "available": None,
            "success": bool(job_execution.get("success", False)) if job_execution else None,
            "missing_modules": [],
            "num_examples": None,
            "train_loss": None,
            "output_dir": job_execution.get("output_dir") or export_metadata.get("output_dir"),
            "error": None,
        }
    export_toolchain_summary = dict(training_meta.get("export_toolchain_summary") or {})
    if not export_toolchain_summary:
        export_execution = dict(last_run.get("export_execution") or {})
        export_command_plan = dict(last_run.get("export_command_plan") or {})
        export_write = dict(last_run.get("export_write") or {})
        export_audit = dict(export_execution.get("audit") or {})
        export_toolchain_summary = {
            "state": export_audit.get("status") or export_command_plan.get("status") or "unknown",
            "status": export_audit.get("status") or export_command_plan.get("status") or "unknown",
            "toolchain_status": export_execution.get("toolchain_status") or export_command_plan.get("status") or "unknown",
            "toolchain_reason": export_audit.get("reason") or export_command_plan.get("reason"),
            "required": bool(export_metadata.get("requires_export_step", True)),
            "target_backend": export_metadata.get("recommended_backend"),
            "target_artifact_format": export_metadata.get("export_target"),
            "artifact_name": export_metadata.get("artifact_name"),
            "output_dir": export_metadata.get("output_dir"),
            "write_state": export_write.get("write_state") or export_metadata.get("write_state"),
            "execution_status": export_audit.get("status"),
            "outcome": export_execution.get("outcome"),
            "audit_summary": dict(export_execution.get("audit_summary") or {}),
            "command_metadata": dict(export_command_plan.get("command_metadata") or export_execution.get("command_metadata") or {}),
        }
    export_artifact_summary = dict(training_meta.get("export_artifact_summary") or last_run.get("export_artifact_summary") or {})
    if not export_artifact_summary:
        export_artifact_summary = {
            "status": export_metadata.get("execution_status"),
            "write_state": export_metadata.get("write_state"),
            "target_backend": export_metadata.get("recommended_backend"),
            "target_artifact_format": export_metadata.get("export_target"),
            "artifact_name": export_metadata.get("artifact_name"),
            "path": export_metadata.get("export_artifact_path") or export_metadata.get("output_artifact_path"),
            "valid": export_metadata.get("export_artifact_valid") if "export_artifact_valid" in export_metadata else export_metadata.get("output_artifact_valid"),
            "size_bytes": export_metadata.get("export_artifact_size_bytes") if "export_artifact_size_bytes" in export_metadata else export_metadata.get("output_artifact_size_bytes"),
            "converter_kind": export_metadata.get("converter_kind"),
            "outtype": export_metadata.get("outtype"),
            "tool_path": export_metadata.get("toolchain_resolved_path"),
        }
    job_bundle_summary = {
        "state": "materialized" if job_bundle else "absent",
        "ready": bool(job_bundle.get("ready", False)) if job_bundle else False,
        "dry_run": bool(job_bundle.get("dry_run", True)) if job_bundle else True,
        "executor_mode": job_bundle.get("executor_mode") or job_bundle.get("metadata", {}).get("executor_mode"),
        "execution_backend": job_bundle.get("execution_backend") or last_run.get("execution_backend"),
        "execution_executor": job_bundle.get("execution_executor") or last_run.get("execution_executor"),
        "script_path": job_bundle.get("script_path"),
        "job_json_path": job_bundle.get("job_json_path"),
        "command": list(job_bundle.get("command") or []),
        "materialized_files": list(job_bundle.get("audit", {}).get("materialized_files", []))
        if isinstance(job_bundle.get("audit"), dict)
        else [],
    }
    job_execution_summary = {
        "state": job_execution.get("status") or job_execution.get("audit", {}).get("status") or "unknown",
        "attempted": bool(job_execution.get("attempted", False)) if job_execution else False,
        "success": bool(job_execution.get("success", False)) if job_execution else False,
        "returncode": job_execution.get("returncode"),
        "exit_code": job_execution.get("exit_code"),
        "runner_status": (
            job_execution.get("audit", {}).get("status")
            if isinstance(job_execution.get("audit"), dict)
            else None
        ),
        "stdout_present": bool(job_execution.get("stdout")),
        "stderr_present": bool(job_execution.get("stderr")),
        "command": list(job_execution.get("command") or []),
        "runner_result_status": job_execution.get("runner_result", {}).get("status")
        if isinstance(job_execution.get("runner_result"), dict)
        else None,
    }
    artifact_directory = export_metadata.get("artifact_directory") or latest_adapter.get("path")
    output_dir = export_metadata.get("output_dir") or artifact_directory
    return {
        "recommended_backend": export_metadata.get("recommended_backend", "transformers"),
        "requires_export_step": bool(export_metadata.get("requires_export_step", True)),
        "artifact_directory": artifact_directory,
        "output_dir": output_dir,
        "latest_adapter_version": snapshot.get("latest_adapter_version"),
        "latest_adapter_state": latest_adapter.get("state"),
        "export_target": export_metadata.get("export_target"),
        "job_bundle": job_bundle_summary,
        "job_execution": job_execution_summary,
        "real_execution_summary": real_execution_summary,
        "export_toolchain_summary": export_toolchain_summary,
        "export_artifact_summary": export_artifact_summary,
        "last_run_status": last_run.get("status"),
        "real_execution": real_execution_summary,
        "export_toolchain": export_toolchain_summary,
    }


def _build_status_lifecycle_metadata(snapshot: Mapping[str, Any], services: ServiceBundle) -> dict[str, Any]:
    sample_counts = _extract_status_counts(snapshot or {})
    signal_summary = dict(snapshot.get("signal_summary") or {})
    signal_sample_counts = dict(snapshot.get("signal_sample_counts") or {})
    signal_sample_details = list(snapshot.get("signal_sample_details") or [])
    latest_adapter = snapshot.get("latest_adapter") or {}
    latest_state = str(latest_adapter.get("state") or "unknown")
    latest_version = snapshot.get("latest_adapter_version")
    trainer_snapshot = dict(snapshot.get("trainer") or {})
    trainer_metadata = _build_status_trainer_metadata(snapshot, services)
    last_run = dict(trainer_snapshot.get("last_run") or {})
    last_job_bundle = dict(last_run.get("job_bundle") or {})
    last_job_execution = dict(last_run.get("job_execution") or {})
    real_execution_summary = dict(trainer_metadata.get("real_execution_summary") or {})
    export_toolchain_summary = dict(trainer_metadata.get("export_toolchain_summary") or {})
    export_metadata = _build_status_export_metadata(snapshot, services)
    server_runtime = _build_status_server_runtime_metadata(services)
    train_ready = sample_counts.get("train", 0) > 0
    eval_ready = sample_counts.get("val", 0) > 0 or sample_counts.get("test", 0) > 0
    promoted = latest_state == "promoted"
    latest_exists = bool(snapshot.get("latest_adapter_exists", False))
    serve_snapshot = dict(snapshot.get("serve") or {})
    return {
        "train": {
            "state": "ready" if train_ready else "idle",
            "sample_counts": sample_counts,
            "dataset_splits": {
                "train": sample_counts.get("train", 0),
                "val": sample_counts.get("val", 0),
                "test": sample_counts.get("test", 0),
            },
        },
        "eval": {
            "state": "ready" if eval_ready else "waiting_for_holdout",
            "scope": "holdout_only" if eval_ready else "unavailable",
            "latest_adapter_version": latest_version,
            "latest_adapter_state": latest_state,
        },
        "promotion": {
            "state": latest_state,
            "latest_adapter_version": latest_version,
            "latest_is_promoted": promoted,
            "latest_is_latest": promoted or latest_state in {"pending_eval", "failed_eval", "archived", "training"},
            "can_promote": latest_state in {"pending_eval", "failed_eval", "archived", "training"},
            "requires_export_step": bool(export_metadata.get("requires_export_step", True)),
            "export_target": export_metadata.get("export_target"),
            "last_run_status": last_run.get("status"),
            "last_job_bundle_state": "materialized" if last_job_bundle else "absent",
            "last_job_execution_state": last_job_execution.get("status")
            or last_job_execution.get("audit", {}).get("status")
            or "unknown",
            "last_job_execution_success": bool(last_job_execution.get("success", False)) if last_job_execution else False,
            "last_job_execution_attempted": bool(last_job_execution.get("attempted", False)) if last_job_execution else False,
            "last_real_execution_state": real_execution_summary.get("state") if isinstance(real_execution_summary, Mapping) else None,
            "last_export_toolchain_status": export_toolchain_summary.get("toolchain_status")
            if isinstance(export_toolchain_summary, Mapping)
            else None,
        },
        "serve": {
            "state": server_runtime.get("probe_status", {}).get("state", "unknown"),
            "launch_mode": server_runtime.get("launch_mode", "dry_run"),
            "probe_state": server_runtime.get("probe_status", {}).get("state", "unknown"),
            "healthy_checks": server_runtime.get("probe_status", {}).get("summary", {}).get("healthy_checks", 0),
            "total_checks": server_runtime.get("probe_status", {}).get("summary", {}).get("total_checks", 0),
            "last_check": server_runtime.get("last_serve_check", {}),
            "probe_paths": server_runtime.get("probe_paths", []),
            "latest_adapter_exists": latest_exists,
            "adapter_resolution_state": serve_snapshot.get("adapter_resolution_state", "latest_promoted" if latest_exists else "no_promoted_latest"),
            "using_promoted_adapter": serve_snapshot.get("using_promoted_adapter", latest_exists),
            "fallback_reason": serve_snapshot.get("fallback_reason"),
        },
        "signal": {
            "state": signal_summary.get("state", "ready" if snapshot.get("signal_count", 0) else "idle"),
            "event_chain_ready": bool(signal_summary.get("event_chain_ready", False)),
            "signal_count": int(snapshot.get("signal_count", 0) or 0),
            "event_chain_complete_count": int(signal_summary.get("event_chain_complete_count", 0) or 0),
            "event_chain_complete_ratio": signal_summary.get("event_chain_complete_ratio", 0.0),
            "processed_count": int(signal_summary.get("processed_count", 0) or 0),
            "sample_count": int(snapshot.get("signal_sample_count", 0) or 0),
            "sample_counts": signal_sample_counts,
            "sample_details": signal_sample_details,
            "latest_signal_id": signal_summary.get("latest_signal_id"),
        },
    }


def _build_status_server_runtime_metadata(services: ServiceBundle) -> dict[str, Any]:
    runtime_probe = dict(services.runtime_probe or {})
    try:
        import uvicorn  # type: ignore

        uvicorn_available = True
        uvicorn_module = getattr(uvicorn, "__name__", "uvicorn")
    except Exception:
        uvicorn_available = False
        uvicorn_module = None
    probe_paths = runtime_probe.get("probe_paths") or [
        {"method": "GET", "path": "/healthz"},
        {"method": "GET", "path": "/pfe/status"},
        {"method": "POST", "path": "/pfe/signal"},
    ]
    return {
        "app_target": runtime_probe.get("app_target", "pfe_server.app:app"),
        "app_type": runtime_probe.get("app_type", "unknown"),
        "dry_run": bool(runtime_probe.get("dry_run", True)),
        "uvicorn_available": bool(runtime_probe.get("uvicorn_available", uvicorn_available)),
        "uvicorn_module": runtime_probe.get("uvicorn_module", uvicorn_module),
        "launch_mode": runtime_probe.get("launch_mode", "dry_run"),
        "serve_summary": dict(runtime_probe.get("serve_summary") or {}),
        "launch_state": dict(runtime_probe.get("launch_state") or {}),
        "probe_status": dict(runtime_probe.get("probe_status") or {"state": "unknown"}),
        "last_serve_check": dict(runtime_probe.get("last_serve_check") or {}),
        "command": list(runtime_probe.get("command") or []),
        "runner": dict(runtime_probe.get("runner") or {}),
        "probe_paths": probe_paths,
        "workspace": services.workspace,
        "provider": services.provider,
        "started_at": services.started_at,
    }


def _build_daemon_timeline_surface(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    train_queue = dict(snapshot.get("train_queue") or {})
    daemon_snapshot = dict(snapshot.get("daemon") or train_queue.get("daemon") or {})
    operations_console = dict(snapshot.get("operations_console") or {})
    timeline = dict(train_queue.get("daemon_history") or {})
    if not timeline:
        timeline = dict(snapshot.get("daemon_timeline") or {})
    if not timeline:
        timeline = dict(operations_console.get("daemon_timeline") or {})
    if not timeline:
        return {}

    recent_events = [
        dict(item)
        for item in list(timeline.get("recent_events") or [])
        if isinstance(item, Mapping)
    ]
    recovery_events = {"recover_requested", "restart_requested", "recover_blocked", "stale_lock_takeover", "start_requested"}
    recent_recovery_events: list[dict[str, Any]] = []
    for item in recent_events:
        event = str(item.get("event") or "")
        if event in recovery_events:
            recent_recovery_events.append({key: item.get(key) for key in ("timestamp", "event", "reason", "note") if item.get(key) is not None})

    latest_recovery = recent_recovery_events[-1] if recent_recovery_events else {}
    recent_anomaly_event = latest_recovery.get("event")
    recent_anomaly_reason = latest_recovery.get("reason")
    if not recent_anomaly_reason and daemon_snapshot:
        recent_anomaly_reason = (
            daemon_snapshot.get("recovery_reason")
            or daemon_snapshot.get("last_reason")
            or daemon_snapshot.get("health_state")
        )
        recent_anomaly_event = recent_anomaly_event or daemon_snapshot.get("recovery_action") or daemon_snapshot.get("last_event")
    if recent_events and not timeline.get("latest_timestamp"):
        timeline["latest_timestamp"] = recent_events[-1].get("timestamp")
    if latest_recovery:
        timeline.setdefault("last_recovery_event", latest_recovery.get("event"))
        timeline.setdefault("last_recovery_reason", latest_recovery.get("reason"))
        timeline.setdefault("last_recovery_note", latest_recovery.get("note"))
    if recent_anomaly_event is not None:
        timeline.setdefault("recent_anomaly_event", recent_anomaly_event)
    if recent_anomaly_reason is not None:
        timeline.setdefault("recent_anomaly_reason", recent_anomaly_reason)
    timeline["count"] = int(timeline.get("count", len(recent_events)) or len(recent_events))
    timeline["recovery_event_count"] = int(timeline.get("recovery_event_count", len(recent_recovery_events)) or len(recent_recovery_events))
    timeline["recent_events"] = recent_events
    timeline["recent_recovery_events"] = recent_recovery_events[-5:]

    summary_parts: list[str] = []
    for key in (
        "count",
        "recovery_event_count",
        "last_event",
        "last_reason",
        "last_recovery_event",
        "last_recovery_reason",
        "last_recovery_note",
        "recent_anomaly_event",
        "recent_anomaly_reason",
    ):
        value = timeline.get(key)
        if value is not None:
            summary_parts.append(f"{key}={value}")
    if timeline.get("latest_timestamp") is not None:
        summary_parts.append(f"latest_timestamp={timeline.get('latest_timestamp')}")
    if summary_parts:
        timeline["summary_line"] = " | ".join(summary_parts)
    return timeline


def _build_runner_timeline_surface(
    snapshot: Mapping[str, Any],
    runner_history: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    train_queue = dict(snapshot.get("train_queue") or {})
    worker_runner = dict(train_queue.get("worker_runner") or {})
    history_payload = dict(runner_history or {})
    event_source = list(history_payload.get("items") or history_payload.get("recent_events") or [])
    items = [dict(item) for item in event_source if isinstance(item, Mapping)]
    if not items and not worker_runner:
        return {}

    latest = items[-1] if items else {}
    recent_events = [
        {key: item.get(key) for key in ("timestamp", "event", "reason") if item.get(key) is not None}
        for item in items[-5:]
    ]
    takeover_event_names = {"blocked_reentry", "stale_lock_takeover"}
    takeover_events = [dict(item) for item in list(history_payload.get("recent_takeover_events") or []) if isinstance(item, Mapping)]
    if not takeover_events:
        takeover_events = [
            {
                key: item.get(key)
                for key in ("timestamp", "event", "reason", "note")
                if item.get(key) is not None
            }
            for item in items
            if str(item.get("event") or "") in takeover_event_names
            or "takeover" in str(item.get("event") or "")
        ]
    latest_takeover = takeover_events[-1] if takeover_events else {}
    recent_anomaly_reason = latest_takeover.get("reason")
    recent_anomaly_event = latest_takeover.get("event")
    if not recent_anomaly_reason and worker_runner:
        recent_anomaly_reason = worker_runner.get("last_event_reason") or worker_runner.get("last_event")
        recent_anomaly_event = worker_runner.get("last_event")
    timeline = {
        "count": int(history_payload.get("count", len(items)) or len(items)),
        "latest_timestamp": history_payload.get("latest_timestamp") or latest.get("timestamp"),
        "last_event": history_payload.get("last_event") or latest.get("event"),
        "last_reason": history_payload.get("last_reason") or latest.get("reason"),
        "current_active": bool(worker_runner.get("active", False)),
        "current_lock_state": worker_runner.get("lock_state") or "idle",
        "current_stop_requested": bool(worker_runner.get("stop_requested", False)),
        "current_lease_expires_at": worker_runner.get("lease_expires_at"),
        "current_processed_count": int(worker_runner.get("processed_count", 0) or 0),
        "current_failed_count": int(worker_runner.get("failed_count", 0) or 0),
        "current_loop_cycles": int(worker_runner.get("loop_cycles", 0) or 0),
        "recent_events": recent_events,
        "takeover_event_count": int(history_payload.get("takeover_event_count", len(takeover_events)) or len(takeover_events)),
        "last_takeover_event": history_payload.get("last_takeover_event") or latest_takeover.get("event"),
        "last_takeover_reason": history_payload.get("last_takeover_reason") or latest_takeover.get("reason"),
        "recent_takeover_events": takeover_events[-5:],
    }
    if recent_anomaly_event is not None:
        timeline["recent_anomaly_event"] = recent_anomaly_event
    if recent_anomaly_reason is not None:
        timeline["recent_anomaly_reason"] = recent_anomaly_reason
    summary_parts: list[str] = []
    for key in (
        "count",
        "takeover_event_count",
        "last_event",
        "last_reason",
        "last_takeover_event",
        "last_takeover_reason",
        "current_active",
        "current_lock_state",
        "current_stop_requested",
        "recent_anomaly_reason",
    ):
        value = timeline.get(key)
        if value is not None:
            summary_parts.append(f"{key}={value}")
    if timeline.get("latest_timestamp") is not None:
        summary_parts.append(f"latest_timestamp={timeline.get('latest_timestamp')}")
    if summary_parts:
        timeline["summary_line"] = " | ".join(summary_parts)
    return timeline


def _build_status_operations_surface(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    overview = dict(snapshot.get("operations_overview") or {})
    console = dict(snapshot.get("operations_console") or {})
    candidate_section = dict(console.get("candidate") or {})
    queue_section = dict(console.get("queue") or {})
    runner_section = dict(console.get("runner") or {})
    daemon_section = dict(console.get("daemon") or {})
    daemon_snapshot = dict(snapshot.get("daemon") or snapshot.get("train_queue", {}).get("daemon") or {})
    if daemon_snapshot:
        for key, value in daemon_snapshot.items():
            daemon_section.setdefault(key, value)
    alerts = overview.get("alerts")
    if not isinstance(alerts, list):
        alerts = []
    health = dict(overview.get("health") or console.get("health") or {})
    recovery = dict(overview.get("recovery") or console.get("recovery") or {})
    next_actions = console.get("next_actions")
    if not isinstance(next_actions, list):
        next_actions = list(overview.get("next_actions") or [])
    summary_line = overview.get("summary_line") or console.get("summary_line")
    attention_reason = overview.get("attention_reason") or console.get("attention_reason")
    attention_needed = overview.get("attention_needed")
    if attention_needed is None:
        attention_needed = bool(attention_reason or next_actions or alerts)
    if "status" not in health:
        health["status"] = "attention" if bool(attention_needed) else "ok"
    health.setdefault("health_state", daemon_section.get("health_state") or overview.get("daemon_health_state"))
    health.setdefault("lease_state", daemon_section.get("lease_state") or overview.get("daemon_lease_state"))
    health.setdefault("heartbeat_state", daemon_section.get("heartbeat_state") or overview.get("daemon_heartbeat_state"))
    health.setdefault("daemon_health_state", health.get("health_state"))
    health.setdefault("daemon_lease_state", health.get("lease_state"))
    health.setdefault("daemon_heartbeat_state", health.get("heartbeat_state"))
    health.setdefault(
        "restart_policy_state",
        daemon_section.get("restart_policy_state") or overview.get("daemon_restart_policy_state"),
    )
    health.setdefault("daemon_restart_policy_state", health.get("restart_policy_state"))
    health.setdefault("recovery_action", daemon_section.get("recovery_action") or overview.get("daemon_recovery_action"))
    health.setdefault("daemon_lock_state", daemon_section.get("lock_state") or overview.get("daemon_lock_state") or "idle")
    health.setdefault("runner_lock_state", runner_section.get("lock_state") or overview.get("runner_lock_state") or "idle")
    health.setdefault(
        "candidate_state",
        candidate_section.get("current_stage") or overview.get("candidate_state") or snapshot.get("candidate_summary", {}).get("candidate_state"),
    )
    if "queue_state" not in health:
        if int(queue_section.get("awaiting_confirmation_count", overview.get("awaiting_confirmation_count", 0)) or 0) > 0:
            health["queue_state"] = "awaiting_confirmation"
        elif int(queue_section.get("count", overview.get("queue_count", 0)) or 0) > 0:
            health["queue_state"] = "queued"
        else:
            health["queue_state"] = "idle"
    if recovery and "daemon_recovery_state" not in recovery:
        recovery["daemon_recovery_state"] = (
            "recovering"
            if recovery.get("daemon_recovery_needed") and not recovery.get("daemon_backoff_remaining_seconds")
            else "idle"
        )
    recovery.setdefault("daemon_recovery_action", daemon_section.get("recovery_action") or overview.get("daemon_recovery_action"))
    if not recovery:
        recovery = {
            "daemon_recovery_needed": bool(daemon_section.get("recovery_needed", False)),
            "daemon_recovery_reason": daemon_section.get("recovery_reason"),
            "daemon_recovery_state": daemon_section.get("recovery_state") or ("recoverable" if daemon_section.get("can_recover") else "idle"),
            "recovery_needed": bool(daemon_section.get("recovery_needed", False)),
            "recovery_reason": daemon_section.get("recovery_reason"),
            "daemon_recovery_action": daemon_section.get("recovery_action") or "none",
        }
    operations_next_actions_payload = list(next_actions)
    if not operations_next_actions_payload:
        operations_next_actions_payload = []
        if bool(health.get("status") == "attention"):
            if health.get("queue_state") == "awaiting_confirmation":
                operations_next_actions_payload.append("review_queue_confirmation")
            if health.get("candidate_state") in {"pending_eval", "training", "failed_eval"}:
                operations_next_actions_payload.append("review_candidate_promotion")
            if recovery.get("daemon_recovery_needed"):
                operations_next_actions_payload.append("recover_worker_daemon")
    if not next_actions:
        next_actions = []
        awaiting_confirmation = int(queue_section.get("awaiting_confirmation_count", overview.get("awaiting_confirmation_count", 0)) or 0)
        candidate_needs_promotion = bool(
            overview.get("candidate_needs_promotion")
            or snapshot.get("candidate_summary", {}).get("candidate_needs_promotion", False)
        )
        if awaiting_confirmation > 0:
            next_actions.append("review_queue_confirmation")
        if candidate_needs_promotion:
            next_actions.append("review_candidate_promotion")
        if recovery.get("daemon_recovery_needed"):
            next_actions.append("recover_worker_daemon")
    return {
        "attention_needed": bool(attention_needed),
        "attention_reason": attention_reason,
        "summary_line": summary_line,
        "alerts": list(alerts),
        "health": health,
        "recovery": recovery,
        "next_actions": operations_next_actions_payload or list(next_actions),
    }


def _build_operations_event_stream_surface(
    snapshot: Mapping[str, Any],
    operations_surface: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    console = dict(snapshot.get("operations_console") or {})
    event_stream = dict(console.get("event_stream") or {})
    existing_operations_surface = dict(operations_surface or {})

    candidate_history = dict(snapshot.get("candidate_history") or {})
    train_queue = dict(snapshot.get("train_queue") or {})
    queue_history = dict(train_queue.get("history_summary") or {})
    runner_timeline = dict(snapshot.get("runner_timeline") or {})
    daemon_timeline = dict(snapshot.get("daemon_timeline") or {})
    overview = dict(snapshot.get("operations_overview") or {})
    if not existing_operations_surface:
        existing_operations_surface = _build_status_operations_surface(snapshot)
    alert_surface = dict(existing_operations_surface or {})
    alerts = list(alert_surface.get("alerts") or [])
    health = dict(alert_surface.get("health") or {})
    recovery = dict(alert_surface.get("recovery") or {})
    next_actions = list(alert_surface.get("next_actions") or [])
    if event_stream:
        normalized = dict(event_stream)
        normalized.setdefault("attention_needed", bool(alert_surface.get("attention_needed")))
        normalized.setdefault("attention_reason", alert_surface.get("attention_reason"))
        normalized.setdefault("attention_source", normalized.get("latest_source") if normalized.get("attention_needed") else None)
        normalized.setdefault("status", "attention" if normalized.get("attention_needed") else "healthy")
        normalized.setdefault("alert_count", len(alerts))
        normalized["next_actions"] = list(normalized.get("next_actions") or next_actions)
        if normalized.get("highest_priority_action") is None:
            normalized["highest_priority_action"] = alert_surface.get("highest_priority_action")
        if normalized.get("active_recovery_hint") is None:
            queue_confirmation = dict(train_queue.get("confirmation_summary") or {})
            normalized["active_recovery_hint"] = (
                queue_confirmation.get("next_confirmation_reason")
                or alert_surface.get("active_recovery_hint")
            )
        normalized["escalated_reasons"] = list(
            normalized.get("escalated_reasons")
            or alert_surface.get("escalated_reasons")
            or []
        )
        latest_recovery = normalized.get("latest_recovery")
        if not latest_recovery:
            latest_recovery = alert_surface.get("latest_recovery")
        normalized["latest_recovery"] = latest_recovery if latest_recovery else None
        dashboard = dict(normalized.get("dashboard") or {})
        dashboard.setdefault("severity", normalized.get("severity"))
        dashboard.setdefault("status", normalized.get("status"))
        dashboard.setdefault("attention_needed", normalized.get("attention_needed"))
        dashboard.setdefault("attention_reason", normalized.get("attention_reason"))
        dashboard.setdefault("attention_source", normalized.get("attention_source"))
        dashboard.setdefault("highest_priority_action", normalized.get("highest_priority_action"))
        dashboard.setdefault("active_recovery_hint", normalized.get("active_recovery_hint"))
        dashboard.setdefault("escalated_reasons", list(normalized.get("escalated_reasons") or []))
        dashboard.setdefault("latest_recovery", normalized.get("latest_recovery"))
        dashboard.setdefault("latest_source", normalized.get("latest_source"))
        dashboard.setdefault("latest_event", normalized.get("latest_event"))
        dashboard.setdefault("latest_reason", normalized.get("latest_reason"))
        dashboard.setdefault("next_actions", list(normalized.get("next_actions") or next_actions))
        dashboard.setdefault("summary_line", normalized.get("summary_line"))
        normalized["dashboard"] = dashboard
        return normalized

    event_items: list[dict[str, Any]] = []
    for item in list(candidate_history.get("items") or [])[-3:]:
        if not isinstance(item, Mapping):
            continue
        event_items.append(
            {
                "source": "candidate",
                "timestamp": item.get("timestamp"),
                "event": item.get("action"),
                "reason": item.get("reason"),
                "note": item.get("operator_note"),
                "status": item.get("status"),
                "version": item.get("candidate_version"),
            }
        )
    last_transition = queue_history.get("last_transition")
    if isinstance(last_transition, Mapping):
        event_items.append(
            {
                "source": "queue",
                "timestamp": last_transition.get("timestamp"),
                "event": last_transition.get("event"),
                "reason": queue_history.get("last_reason"),
                "job_id": last_transition.get("job_id"),
                "state": last_transition.get("state"),
            }
        )
    for item in list(runner_timeline.get("recent_events") or [])[-3:]:
        if not isinstance(item, Mapping):
            continue
        event_items.append(
            {
                "source": "runner",
                "timestamp": item.get("timestamp"),
                "event": item.get("event"),
                "reason": item.get("reason"),
                "note": item.get("note"),
            }
        )
    for item in list(daemon_timeline.get("recent_events") or [])[-3:]:
        if not isinstance(item, Mapping):
            continue
        event_items.append(
            {
                "source": "daemon",
                "timestamp": item.get("timestamp"),
                "event": item.get("event"),
                "reason": item.get("reason"),
                "note": item.get("note"),
            }
        )

    event_items = sorted(
        event_items,
        key=lambda item: str(item.get("timestamp") or ""),
        reverse=True,
    )[:8]
    latest_event = event_items[0] if event_items else {}
    attention_needed = bool(
        alert_surface.get("attention_needed")
        or overview.get("attention_needed")
        or health.get("status") == "attention"
        or alerts
        or next_actions
    )
    attention_reason = (
        alert_surface.get("attention_reason")
        or overview.get("attention_reason")
        or health.get("status")
        or latest_event.get("reason")
    )
    latest_source = latest_event.get("source")
    latest_reason = latest_event.get("reason")
    latest_event_name = latest_event.get("event")
    alert_reasons = {
        str(alert.get("reason"))
        for alert in alerts
        if isinstance(alert, Mapping) and alert.get("reason") is not None
    }
    alert_severities = {
        str(alert.get("severity"))
        for alert in alerts
        if isinstance(alert, Mapping) and alert.get("severity") is not None
    }
    critical_reasons = {
        "daemon_stale",
        "daemon_heartbeat_stale",
        "daemon_lease_expired",
        "daemon_restart_backoff",
        "daemon_restart_capped",
        "daemon_recoverable_stale",
    }
    warning_reasons = {
        "awaiting_confirmation",
        "candidate_ready_for_promotion",
        "runner_blocked_reentry",
        "queue_waiting_confirmation",
        "manual_review_required_by_policy",
    }
    severity = "ok"
    if event_items or attention_needed:
        severity = "info" if event_items and not attention_needed else "warning"
    if (
        bool(recovery.get("daemon_recovery_needed"))
        or str(health.get("daemon_health_state") or "").lower() in {"stale", "expired"}
        or str(health.get("daemon_lease_state") or "").lower() == "expired"
        or str(health.get("daemon_heartbeat_state") or "").lower() == "stale"
        or (
            str(latest_source or "").lower() == "daemon"
            and str(latest_reason or "").lower() in critical_reasons
        )
        or any(reason in critical_reasons for reason in alert_reasons)
        or any(severity_name in {"critical", "error"} for severity_name in alert_severities)
    ):
        severity = "critical"
    elif severity != "critical":
        if (
            str(latest_source or "").lower() == "queue"
            and str(latest_reason or "").lower() in warning_reasons
        ) or (
            str(latest_source or "").lower() == "candidate"
            and str(latest_reason or "").lower() in warning_reasons
        ) or (
            str(latest_source or "").lower() == "runner"
            and str(latest_reason or "").lower() in warning_reasons
        ):
            severity = "warning"
        elif attention_needed:
            severity = "warning" if event_items else "warning"
        elif event_items:
            severity = "info"
    event_stream = {
        "count": len(event_items),
        "latest_timestamp": latest_event.get("timestamp"),
        "latest_source": latest_event.get("source"),
        "latest_event": latest_event.get("event"),
        "latest_reason": latest_event.get("reason"),
        "attention_needed": attention_needed,
        "attention_reason": attention_reason,
        "attention_source": latest_source if attention_needed else None,
        "severity": severity,
        "status": "attention" if severity in {"warning", "critical"} else "healthy",
        "alert_count": len(alerts),
        "items": event_items,
    }
    event_summary_parts: list[str] = [f"count={len(event_items)}"]
    event_summary_parts.append(f"severity={severity}")
    event_summary_parts.append(f"attention_needed={'yes' if attention_needed else 'no'}")
    if attention_reason is not None:
        event_summary_parts.append(f"attention_reason={attention_reason}")
    if latest_event.get("source") is not None:
        event_summary_parts.append(f"latest_source={latest_event.get('source')}")
    if latest_event.get("event") is not None:
        event_summary_parts.append(f"latest_event={latest_event.get('event')}")
    if latest_event.get("reason") is not None:
        event_summary_parts.append(f"latest_reason={latest_event.get('reason')}")
    event_stream["summary_line"] = " | ".join(event_summary_parts)
    dashboard = {
        "severity": severity,
        "status": event_stream["status"],
        "attention_needed": attention_needed,
        "attention_reason": attention_reason,
        "attention_source": latest_source if attention_needed else None,
        "next_actions": next_actions,
        "alert_count": len(alerts),
        "summary_line": event_stream["summary_line"],
    }
    if health:
        dashboard["health"] = {
            key: health.get(key)
            for key in (
                "status",
                "health_state",
                "daemon_health_state",
                "daemon_lock_state",
                "daemon_lease_state",
                "daemon_heartbeat_state",
                "runner_lock_state",
                "candidate_state",
                "queue_state",
            )
            if health.get(key) is not None
        }
    if recovery:
        dashboard["recovery"] = {
            key: recovery.get(key)
            for key in (
                "daemon_recovery_needed",
                "daemon_recovery_reason",
                "daemon_recovery_state",
                "daemon_recovery_action",
                "recovery_needed",
                "recovery_reason",
            )
            if recovery.get(key) is not None
        }
    if event_items:
        dashboard["latest_event"] = {
            "source": latest_source,
            "event": latest_event_name,
            "reason": latest_reason,
            "timestamp": latest_event.get("timestamp"),
            "severity": severity,
            "attention": attention_needed,
        }
    event_stream["dashboard"] = dashboard
    event_stream["next_actions"] = list(next_actions)
    return event_stream


def _build_operations_inspection_surface(
    *,
    operations_surface: Mapping[str, Any] | None,
    operations_dashboard: Mapping[str, Any] | None,
    operations_alert_policy: Mapping[str, Any] | None,
    operations_event_stream: Mapping[str, Any] | None,
    daemon_timeline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    alert_surface = dict(operations_surface or {})
    dashboard = dict(operations_dashboard or {})
    alert_policy = dict(operations_alert_policy or {})
    event_stream = dict(operations_event_stream or {})
    recovery = dict(alert_surface.get("recovery") or {})
    health = dict(alert_surface.get("health") or {})
    next_actions = list(alert_surface.get("next_actions") or [])
    if not next_actions:
        next_actions = list(dashboard.get("next_actions") or alert_policy.get("next_actions") or event_stream.get("next_actions") or [])
    current_focus = (
        dashboard.get("current_focus")
        or alert_policy.get("current_focus")
        or alert_surface.get("attention_reason")
        or dashboard.get("attention_reason")
        or event_stream.get("attention_reason")
        or recovery.get("daemon_recovery_reason")
        or health.get("queue_state")
        or health.get("candidate_state")
        or health.get("daemon_health_state")
    )
    if current_focus is None:
        if recovery.get("daemon_recovery_needed"):
            current_focus = "daemon_recovery"
        elif health.get("queue_state") == "awaiting_confirmation":
            current_focus = "awaiting_confirmation"
        elif health.get("candidate_state") in {"pending_eval", "training", "failed_eval"}:
            current_focus = "candidate_ready_for_promotion"
        elif str(health.get("daemon_health_state") or "").lower() in {"stale", "expired"}:
            current_focus = "daemon_stale"
        elif next_actions:
            next_action = str(next_actions[0])
            if next_action == "review_queue_confirmation":
                current_focus = "awaiting_confirmation"
            elif next_action == "review_candidate_promotion":
                current_focus = "candidate_ready_for_promotion"
            elif next_action == "recover_worker_daemon":
                current_focus = "daemon_stale"
            elif next_action == "inspect_worker_stale_lock":
                current_focus = "runner_stale_lock"
            else:
                current_focus = next_action
    required_action = (
        alert_policy.get("required_action")
        or dashboard.get("required_action")
        or (next_actions[0] if next_actions else None)
    )
    last_recovery_event = (
        (daemon_timeline or {}).get("last_recovery_event")
        or dashboard.get("last_recovery_event")
        or alert_policy.get("last_recovery_event")
    )
    last_recovery_reason = (
        (daemon_timeline or {}).get("last_recovery_reason")
        or dashboard.get("last_recovery_reason")
        or alert_policy.get("last_recovery_reason")
    )
    last_recovery_note = (
        (daemon_timeline or {}).get("last_recovery_note")
        or dashboard.get("last_recovery_note")
        or alert_policy.get("last_recovery_note")
    )
    inspection_summary_parts: list[str] = []
    if current_focus is not None:
        inspection_summary_parts.append(f"current_focus={current_focus}")
    if required_action is not None:
        inspection_summary_parts.append(f"required_action={required_action}")
    if last_recovery_event is not None:
        inspection_summary_parts.append(f"last_recovery_event={last_recovery_event}")
    if last_recovery_reason is not None:
        inspection_summary_parts.append(f"last_recovery_reason={last_recovery_reason}")
    if next_actions:
        inspection_summary_parts.append("next_actions=" + ",".join(str(item) for item in next_actions if item is not None))
    inspection_summary_line = " | ".join(inspection_summary_parts) if inspection_summary_parts else None
    return {
        "current_focus": current_focus,
        "required_action": required_action,
        "last_recovery_event": last_recovery_event,
        "last_recovery_reason": last_recovery_reason,
        "last_recovery_note": last_recovery_note,
        "next_actions": next_actions,
        "inspection_summary_line": inspection_summary_line,
    }


def _build_operations_timeline_surface(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    console = dict(snapshot.get("operations_console") or {})
    timelines = dict(console.get("timelines") or {})

    candidate_timeline = dict(snapshot.get("candidate_timeline") or {})
    candidate_history = dict(snapshot.get("candidate_history") or {})
    train_queue = dict(snapshot.get("train_queue") or {})
    queue_history = dict(train_queue.get("history_summary") or {})
    runner_timeline = dict(snapshot.get("runner_timeline") or {})
    daemon_timeline = dict(snapshot.get("daemon_timeline") or {})

    candidate_section = {
        "current_stage": candidate_timeline.get("current_stage") or candidate_history.get("last_status") or candidate_history.get("last_action"),
        "last_candidate_version": candidate_timeline.get("last_candidate_version")
        or candidate_history.get("last_candidate_version"),
        "last_reason": candidate_timeline.get("last_reason") or candidate_history.get("last_reason"),
        "latest_timestamp": candidate_timeline.get("latest_timestamp") or candidate_history.get("latest_timestamp"),
        "transition_count": candidate_timeline.get("transition_count") or candidate_history.get("count"),
    }
    queue_section = {
        "count": train_queue.get("count"),
        "last_transition": queue_history.get("last_transition", {}).get("event") if isinstance(queue_history.get("last_transition"), Mapping) else queue_history.get("last_transition"),
        "last_reason": queue_history.get("last_reason"),
        "latest_timestamp": queue_history.get("last_transition", {}).get("timestamp") if isinstance(queue_history.get("last_transition"), Mapping) else queue_history.get("latest_timestamp"),
        "transition_count": queue_history.get("transition_count"),
    }
    runner_section = {
        "count": runner_timeline.get("count"),
        "last_event": runner_timeline.get("last_event"),
        "last_reason": runner_timeline.get("last_reason"),
        "takeover_event_count": runner_timeline.get("takeover_event_count"),
        "last_takeover_event": runner_timeline.get("last_takeover_event"),
        "last_takeover_reason": runner_timeline.get("last_takeover_reason"),
        "recent_anomaly_reason": runner_timeline.get("recent_anomaly_reason"),
        "latest_timestamp": runner_timeline.get("latest_timestamp"),
    }
    daemon_section = {
        "count": daemon_timeline.get("count"),
        "last_event": daemon_timeline.get("last_event"),
        "last_reason": daemon_timeline.get("last_reason"),
        "recovery_event_count": daemon_timeline.get("recovery_event_count"),
        "last_recovery_event": daemon_timeline.get("last_recovery_event"),
        "last_recovery_reason": daemon_timeline.get("last_recovery_reason"),
        "last_recovery_note": daemon_timeline.get("last_recovery_note"),
        "recent_anomaly_reason": daemon_timeline.get("recent_anomaly_reason"),
        "latest_timestamp": daemon_timeline.get("latest_timestamp"),
    }
    if timelines:
        merged = dict(timelines)
        merged.setdefault("candidate", candidate_section)
        merged.setdefault("queue", queue_section)
        merged.setdefault("runner", runner_section)
        merged.setdefault("daemon", daemon_section)
        summary_parts: list[str] = []
        if candidate_section.get("current_stage") is not None:
            summary_parts.append(f"candidate={candidate_section.get('current_stage')}")
        if queue_section.get("last_transition") is not None:
            summary_parts.append(f"queue={queue_section.get('last_transition')}")
        if runner_section.get("last_event") is not None:
            summary_parts.append(f"runner={runner_section.get('last_event')}")
        if runner_section.get("recent_anomaly_reason") is not None:
            summary_parts.append(f"runner_anomaly={runner_section.get('recent_anomaly_reason')}")
        if daemon_section.get("last_event") is not None:
            summary_parts.append(f"daemon={daemon_section.get('last_event')}")
        if daemon_section.get("recent_anomaly_reason") is not None:
            summary_parts.append(f"daemon_anomaly={daemon_section.get('recent_anomaly_reason')}")
        if summary_parts and not merged.get("summary_line"):
            merged["summary_line"] = " | ".join(summary_parts)
        return merged

    summary_parts: list[str] = []
    if candidate_section.get("current_stage") is not None:
        summary_parts.append(f"candidate={candidate_section.get('current_stage')}")
    if queue_section.get("last_transition") is not None:
        summary_parts.append(f"queue={queue_section.get('last_transition')}")
    if runner_section.get("last_event") is not None:
        summary_parts.append(f"runner={runner_section.get('last_event')}")
    if runner_section.get("recent_anomaly_reason") is not None:
        summary_parts.append(f"runner_anomaly={runner_section.get('recent_anomaly_reason')}")
    if daemon_section.get("last_event") is not None:
        summary_parts.append(f"daemon={daemon_section.get('last_event')}")
    if daemon_section.get("recent_anomaly_reason") is not None:
        summary_parts.append(f"daemon_anomaly={daemon_section.get('recent_anomaly_reason')}")
    return {
        "candidate": candidate_section,
        "queue": queue_section,
        "runner": runner_section,
        "daemon": daemon_section,
        "summary_line": " | ".join(summary_parts),
    }


def _fallback_status_snapshot() -> dict[str, Any]:
    return {
        "provider": "mock",
        "sample_counts": {"train": 0, "val": 0, "test": 0},
        "adapter_count": 0,
        "signal_count": 0,
        "signal_summary": {
            "state": "idle",
            "event_chain_ready": False,
            "event_chain_complete_count": 0,
            "event_chain_complete_ratio": 0.0,
            "processed_count": 0,
            "latest_signal_id": None,
            "source_event_id_count": 0,
            "request_id_count": 0,
            "session_id_count": 0,
        },
        "signal_sample_count": 0,
        "signal_sample_counts": {"train": 0, "val": 0, "test": 0},
        "signal_sample_details": [],
        "auto_train_trigger": {
            "enabled": False,
            "state": "disabled",
            "ready": False,
            "reason": "trigger_disabled",
            "blocked_reasons": ["trigger_disabled"],
            "min_new_samples": 50,
            "max_interval_days": 7,
            "queue_process_batch_size": 5,
            "queue_process_until_idle_max": 10,
            "require_queue_confirmation": False,
            "preference_reinforced_sample_weight": 1.5,
            "effective_eligible_train_samples": 0.0,
            "preference_reinforced_train_samples": 0,
            "eligible_signal_train_samples": 0,
            "effective_signal_train_samples": 0.0,
            "preference_reinforced_signal_train_samples": 0,
            "eligible_signal_sample_ids": [],
            "effective_dpo_preference_pairs": 0.0,
            "preference_reinforced_dpo_preference_pairs": 0,
            "holdout_ready": False,
            "interval_elapsed": True,
            "days_since_last_training": None,
            "recent_training_version": None,
        },
        "auto_train_trigger_action": {
            "action": "retry",
            "status": "blocked",
            "reason": "trigger_disabled",
            "triggered": False,
        },
        "candidate_summary": {
            "latest_promoted_version": None,
            "recent_version": None,
            "candidate_version": None,
            "candidate_state": None,
            "candidate_exists": False,
            "candidate_can_promote": False,
            "candidate_can_archive": False,
            "pending_eval_count": 0,
            "pending_eval_versions": [],
            "training_count": 0,
            "training_versions": [],
            "failed_eval_count": 0,
            "failed_eval_versions": [],
            "has_pending_candidate": False,
            "candidate_needs_promotion": False,
        },
        "candidate_action": {
            "action": "promote_candidate",
            "status": "noop",
            "reason": "no_candidate_action",
            "triggered": False,
        },
        "train_queue": {"count": 0, "counts": {}, "current": {}, "last_item": {}, "items": []},
        "latest_adapter": {},
        "latest_adapter_version": None,
    }


def _snapshot_adapter_row(row: Mapping[str, Any] | dict[str, Any] | None, *, latest: bool = False) -> dict[str, Any]:
    if not row:
        return {}
    metadata = row.get("metadata") if isinstance(row, Mapping) else None
    if isinstance(metadata, str) and metadata.strip():
        try:
            parsed = json.loads(metadata)
            metadata = parsed if isinstance(parsed, dict) else {}
        except Exception:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "version": row.get("version"),
        "state": row.get("state"),
        "artifact_format": row.get("artifact_format"),
        "base_model": row.get("base_model"),
        "path": row.get("adapter_dir"),
        "artifact_path": row.get("artifact_path"),
        "manifest_path": row.get("manifest_path"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "latest": latest,
        "metadata": metadata,
    }


async def _safe_status_call(callable_obj: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        result = callable_obj()
        if hasattr(result, "__await__"):
            result = await result
        return dict(result) if isinstance(result, Mapping) else dict(result or {})
    except Exception:
        return dict(fallback)


def _last_client_host(request: Any) -> Optional[str]:
    client = getattr(request, "client", None)
    return getattr(client, "host", None) if client is not None else None


def _normalize_query_params(request: Any) -> dict[str, str]:
    query_params = getattr(request, "query_params", None)
    if query_params is None:
        return {}
    if hasattr(query_params, "multi_items"):
        return {str(key): str(value) for key, value in query_params.multi_items()}
    return {str(key): str(value) for key, value in dict(query_params).items()}


async def _envelope_from_fastapi_request(request: Request) -> RequestEnvelope:
    body = await request.body()
    headers = normalize_headers(dict(request.headers.items()))
    return RequestEnvelope(
        method=request.method.upper(),
        path=request.url.path,
        headers=headers,
        client_host=_last_client_host(request),
        body=body,
        query_params=_normalize_query_params(request),
    )


async def _envelope_from_asgi_scope(scope: Mapping[str, Any], receive: Callable[[], Awaitable[dict[str, Any]]]) -> RequestEnvelope:
    body_parts: list[bytes] = []
    while True:
        event = await receive()
        if event.get("type") != "http.request":
            continue
        body_parts.append(event.get("body", b""))
        if not event.get("more_body", False):
            break
    headers = {
        key.decode("latin1").lower(): value.decode("latin1")
        for key, value in scope.get("headers", [])
    }
    query_string = scope.get("query_string", b"")
    query_params: dict[str, str] = {}
    if query_string:
        from urllib.parse import parse_qsl

        query_params = {key: value for key, value in parse_qsl(query_string.decode("latin1"), keep_blank_values=True)}
    client = scope.get("client")
    client_host = client[0] if client else None
    return RequestEnvelope(
        method=str(scope.get("method", "GET")).upper(),
        path=str(scope.get("path", "/")),
        headers=headers,
        client_host=client_host,
        body=b"".join(body_parts),
        query_params=query_params,
    )


async def handle_chat_completions(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="inference")
    if not allowed:
        return denial
    try:
        request = ChatCompletionRequest.model_validate_json(envelope.body or b"{}")
    except Exception as exc:  # pragma: no cover - defensive JSON guard
        return _json_response(
            _error_payload("invalid chat completion request", "invalid_request", str(exc)),
            status_code=422,
        )

    # Extract user message for signal collection
    user_message = ""
    for message in reversed(request.messages):
        if message.role == "user" and message.content:
            user_message = message.content
            break

    # Generate response
    response_start_time = time.time()
    response = await services.inference.generate_chat_completion(request)
    response_time = time.time() - response_start_time

    # Extract assistant message
    assistant_message = ""
    if response.choices and response.choices[0].message:
        assistant_message = response.choices[0].message.content or ""

    # Store interaction for potential signal extraction
    session_id = request.session_id or f"sess-{uuid4().hex[:8]}"
    request_id = request.request_id or f"req-{uuid4().hex[:8]}"

    _store_pending_interaction(
        session_id=session_id,
        request_id=request_id,
        user_message=user_message,
        assistant_message=assistant_message,
        adapter_version=request.adapter_version,
        timestamp=response_start_time,
    )

    # Add signal collection metadata to response
    response_metadata = response.metadata or {}
    response_metadata["signal_collection"] = {
        "session_id": session_id,
        "request_id": request_id,
        "interaction_stored": True,
        "response_time_seconds": round(response_time, 3),
    }
    response.metadata = response_metadata
    response.session_id = session_id
    response.request_id = request_id

    return _json_response(response.model_dump(mode="json"), status_code=200)


async def handle_signal_ingest(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    try:
        request = SignalIngestRequest.model_validate_json(envelope.body or b"{}")
    except Exception as exc:  # pragma: no cover - defensive JSON guard
        return _json_response(
            _error_payload("invalid signal request", "invalid_request", str(exc)),
            status_code=422,
        )
    response = await services.pipeline.ingest_signal(request)
    return _json_response(response.model_dump(mode="json"), status_code=200)


async def handle_feedback(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    """Handle implicit feedback from user interactions.

    This endpoint receives feedback signals that are automatically
    inferred from user behavior (e.g., continuing conversation = accept).

    Supports action types: accept, reject, edit, regenerate, delete
    """
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    try:
        request = FeedbackRequest.model_validate_json(envelope.body or b"{}")
    except Exception as exc:
        return _json_response(
            _error_payload("invalid feedback request", "invalid_request", str(exc)),
            status_code=422,
        )

    # Get pending interaction data if available
    pending = _get_pending_interaction(request.session_id, request.request_id)

    # Use provided messages or fallback to pending interaction
    user_message = request.user_message or (pending.get("user_message") if pending else "")
    assistant_message = request.assistant_message or (pending.get("assistant_message") if pending else "")
    collector_action, _ = _normalize_feedback_action(request.action)

    # Extract signals using ChatCollector if available
    signals = []
    signal_id = None
    signal_type, confidence, extraction_rule = _fallback_feedback_signal(
        request.action,
        request.response_time_seconds,
    )

    pipeline_ingest_result: dict[str, Any] | None = None

    if CHAT_COLLECTOR_AVAILABLE and ChatCollector is not None and CollectorConfig is not None and ChatInteraction is not None:
        try:
            collector = _get_or_create_collector(services.workspace)
            interaction = _build_feedback_interaction(request, pending)
            if collector and interaction is not None:
                # Use the same normalization path for pending and explicit feedback payloads.
                extracted_signals = collector.on_interaction(
                    interaction=interaction,
                    next_user_message=request.next_message if collector_action == "continue" else None,
                    edited_text=request.edited_text if collector_action == "edit" else None,
                    action=collector_action,
                )

                if extracted_signals:
                    signals = extracted_signals
                    signal = extracted_signals[0]
                    signal_id = getattr(signal, 'signal_id', None)
                    signal_type = getattr(signal, 'signal_type', request.action)
                    confidence = getattr(signal, 'confidence', 0.7)
                    extraction_rule = getattr(signal, 'extraction_rule', "extracted")

                    # Ingest signal to pipeline if available
                    if hasattr(services.pipeline, 'ingest_signal') and hasattr(signal, 'to_dict'):
                        try:
                            from .models import SignalIngestRequest
                            signal_data = signal.to_dict()
                            signal_request = SignalIngestRequest(
                                event_id=signal_data.get("event_id", f"sig-{uuid4().hex[:12]}"),
                                request_id=request.request_id,
                                session_id=request.session_id,
                                adapter_version=request.adapter_version,
                                event_type=signal_type,
                                user_input=user_message,
                                model_output=assistant_message,
                                user_action=signal_data.get("user_action", {"type": request.action}),
                                metadata={
                                    "confidence": confidence,
                                    "extraction_rule": extraction_rule,
                                    "source": "chat_collector",
                                },
                            )
                            ingest_response = await services.pipeline.ingest_signal(signal_request)
                            if hasattr(ingest_response, "model_dump"):
                                pipeline_ingest_result = ingest_response.model_dump(mode="json")
                            elif isinstance(ingest_response, Mapping):
                                pipeline_ingest_result = dict(ingest_response)
                        except Exception as e:
                            # Signal ingestion failure should not fail feedback
                            print(f"Signal ingestion failed: {e}")

        except Exception as e:
            # Collector failure should not fail feedback
            print(f"ChatCollector signal extraction failed: {e}")

    if pending:
        _remove_pending_interaction(request.session_id, request.request_id)

    # Generate signal_id if not set
    if signal_id is None:
        signal_id = f"sig-{uuid4().hex[:12]}"

    return _json_response(
        FeedbackResponse(
            success=True,
            signal_id=signal_id,
            signal_type=signal_type,
            confidence=round(confidence, 2),
            message=f"Feedback recorded: {signal_type} (confidence: {confidence:.2f})",
            session_id=request.session_id,
            request_id=request.request_id,
            metadata={
                "action": request.action,
                "response_time_seconds": request.response_time_seconds,
                "signals_extracted": len(signals),
                "extraction_rule": extraction_rule,
                "collector_available": CHAT_COLLECTOR_AVAILABLE,
                "pending_found": pending is not None,
                "pipeline_ingest": pipeline_ingest_result or {},
            },
        ).model_dump(mode="json"),
        status_code=200,
    )


async def handle_distill_run(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    try:
        request = DistillRunRequest.model_validate_json(envelope.body or b"{}")
    except Exception as exc:  # pragma: no cover - defensive JSON guard
        return _json_response(
            _error_payload("invalid distillation request", "invalid_request", str(exc)),
            status_code=422,
        )
    allowed, denial = _route_access(
        envelope,
        security=services.security,
        endpoint_kind="management",
        cloud_requested=request.use_cloud_teacher,
    )
    if not allowed:
        return denial
    response = await services.pipeline.run_distillation(request)
    return _json_response(response.model_dump(mode="json"), status_code=200)


async def handle_status(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    inference_status = await _safe_status_call(services.inference.status, {"backend": services.provider, "healthy": True})
    pipeline_status = await _safe_status_call(
        services.pipeline.status,
        {"backend": services.provider, "signals": 0, "distill_runs": 0},
    )
    snapshot = _load_status_snapshot(services.workspace)
    if not snapshot:
        snapshot = _fallback_status_snapshot()
    if isinstance(pipeline_status, dict):
        trainer_snapshot = pipeline_status.get("trainer")
        if isinstance(trainer_snapshot, dict) and trainer_snapshot:
            snapshot["trainer"] = trainer_snapshot
        for key in (
            "recent_adapter",
            "recent_adapter_version",
            "serve",
            "adapter_lifecycle",
            "auto_train_trigger",
            "auto_train_trigger_action",
            "candidate_summary",
            "candidate_action",
            "candidate_history",
            "candidate_timeline",
            "train_queue",
            "operations_overview",
            "operations_console",
            "capabilities",
            "user_modeling",
        ):
            value = pipeline_status.get(key)
            if value:
                snapshot[key] = value
    daemon_snapshot = dict((snapshot.get("train_queue", {}) or {}).get("daemon") or {})
    if daemon_snapshot:
        merged_daemon = dict(snapshot.get("daemon") or {})
        merged_daemon.update(daemon_snapshot)
        snapshot["daemon"] = merged_daemon
        console_snapshot = dict(snapshot.get("operations_console") or {})
        if console_snapshot:
            console_daemon = dict(console_snapshot.get("daemon") or {})
            console_daemon.update(
                {
                    key: merged_daemon.get(key)
                    for key in (
                        "health_state",
                        "lease_state",
                        "heartbeat_state",
                        "restart_policy_state",
                        "recovery_action",
                        "recovery_needed",
                        "can_recover",
                        "recovery_reason",
                        "recovery_state",
                        "recovery_mode",
                        "recovery_attempts",
                        "recovery_backoff_seconds",
                        "recovery_next_retry_at",
                    )
                    if merged_daemon.get(key) is not None
                }
            )
            console_snapshot["daemon"] = console_daemon
            snapshot["operations_console"] = console_snapshot
    daemon_timeline = _build_daemon_timeline_surface(snapshot)
    runner_history = await services.pipeline.train_queue_worker_runner_timeline(limit=5)
    runner_timeline = _build_runner_timeline_surface(snapshot, runner_history)
    operations_timeline = _build_operations_timeline_surface(snapshot)
    operations_surface = _build_status_operations_surface(snapshot)
    operations_event_stream = _build_operations_event_stream_surface(snapshot, operations_surface)
    if runner_timeline:
        merged_runner_timeline = dict(snapshot.get("runner_timeline") or {})
        merged_runner_timeline.update(runner_timeline)
        snapshot["runner_timeline"] = merged_runner_timeline
        console_snapshot = dict(snapshot.get("operations_console") or {})
        if console_snapshot:
            console_runner_timeline = dict(console_snapshot.get("runner_timeline") or {})
            console_runner_timeline.update(
                {
                    key: merged_runner_timeline.get(key)
                    for key in (
                        "count",
                        "latest_timestamp",
                        "last_event",
                        "last_reason",
                        "current_active",
                        "current_lock_state",
                        "current_stop_requested",
                        "current_lease_expires_at",
                        "current_processed_count",
                        "current_failed_count",
                        "current_loop_cycles",
                        "summary_line",
                        "recent_events",
                    )
                    if merged_runner_timeline.get(key) is not None
                }
            )
            console_snapshot["runner_timeline"] = console_runner_timeline
            snapshot["operations_console"] = console_snapshot
    operations_console_payload = dict(snapshot.get("operations_console") or {})
    operations_console_payload["event_stream"] = dict(operations_event_stream or {})
    operations_dashboard = dict(operations_console_payload.get("dashboard") or {})
    operations_alert_policy = dict(operations_console_payload.get("alert_policy") or {})
    inspection_surface = _build_operations_inspection_surface(
        operations_surface=operations_surface,
        operations_dashboard=operations_dashboard,
        operations_alert_policy=operations_alert_policy,
        operations_event_stream=operations_event_stream,
        daemon_timeline=daemon_timeline,
    )
    if inspection_surface:
        current_focus = inspection_surface.get("current_focus")
        required_action = inspection_surface.get("required_action")
        last_recovery_event = inspection_surface.get("last_recovery_event")
        last_recovery_reason = inspection_surface.get("last_recovery_reason")
        last_recovery_note = inspection_surface.get("last_recovery_note")
        next_actions = list(inspection_surface.get("next_actions") or [])
        if current_focus is not None:
            operations_dashboard["current_focus"] = current_focus
            operations_alert_policy["current_focus"] = current_focus
        if required_action is not None:
            operations_dashboard["required_action"] = required_action
            operations_alert_policy["required_action"] = required_action
        if last_recovery_event is not None:
            operations_dashboard["last_recovery_event"] = last_recovery_event
            operations_alert_policy["last_recovery_event"] = last_recovery_event
        if last_recovery_reason is not None:
            operations_dashboard["last_recovery_reason"] = last_recovery_reason
            operations_alert_policy["last_recovery_reason"] = last_recovery_reason
        if last_recovery_note is not None:
            operations_dashboard["last_recovery_note"] = last_recovery_note
            operations_alert_policy["last_recovery_note"] = last_recovery_note
        if next_actions:
            operations_dashboard["next_actions"] = list(next_actions)
            operations_alert_policy["next_actions"] = list(next_actions)
            operations_console_payload["next_actions"] = list(next_actions)
        inspection_summary_line = inspection_surface.get("inspection_summary_line")
        if inspection_summary_line:
            operations_dashboard["inspection_summary_line"] = inspection_summary_line
            operations_alert_policy["inspection_summary_line"] = inspection_summary_line
            operations_console_payload["inspection_summary_line"] = inspection_summary_line
        event_dashboard = dict(operations_event_stream.get("dashboard") or {})
        if current_focus is not None:
            event_dashboard["current_focus"] = current_focus
        if required_action is not None:
            event_dashboard["required_action"] = required_action
        if last_recovery_event is not None:
            event_dashboard["last_recovery_event"] = last_recovery_event
        if last_recovery_reason is not None:
            event_dashboard["last_recovery_reason"] = last_recovery_reason
        if last_recovery_note is not None:
            event_dashboard["last_recovery_note"] = last_recovery_note
        if next_actions:
            event_dashboard["next_actions"] = list(next_actions)
        if inspection_summary_line:
            event_dashboard["inspection_summary_line"] = inspection_summary_line
        operations_event_stream["dashboard"] = event_dashboard
    operations_console_payload["dashboard"] = operations_dashboard
    operations_console_payload["alert_policy"] = operations_alert_policy
    trainer_metadata = _build_status_trainer_metadata(snapshot, services)
    sample_counts = _extract_status_counts(snapshot or {})
    latest_adapter = snapshot.get("latest_adapter") or {}
    payload = PFEStatusResponse(
        provider=services.provider,
        mode=services.security.privacy_mode if services.security.privacy_mode in {"strict_local", "cloud_assisted"} else "strict_local",
        strict_local=services.security.privacy_mode == "strict_local",
        allow_remote_access=services.security.allow_remote_access,
        uptime_seconds=time.time() - services.started_at,
        inference_backend=str(inference_status.get("backend", "mock")),
        pipeline_backend=str(pipeline_status.get("backend", "mock")),
        adapter_count=int(snapshot.get("adapter_count", 0) or 0),
        signal_count=int((snapshot or {}).get("signal_count", 0) or 0),
        distill_run_count=int(pipeline_status.get("distill_runs", 0)),
        sample_counts=sample_counts,
        signal_summary=dict(snapshot.get("signal_summary") or {}),
        signal_sample_count=int(snapshot.get("signal_sample_count", 0) or 0),
        signal_sample_counts=dict(snapshot.get("signal_sample_counts") or {}),
        signal_sample_details=list(snapshot.get("signal_sample_details") or []),
        auto_train_trigger=dict(snapshot.get("auto_train_trigger") or {}),
        auto_train_trigger_action=_build_auto_train_action_response(snapshot),
        candidate_summary=dict(snapshot.get("candidate_summary") or {}),
        candidate_action=_build_candidate_action_response(snapshot),
        candidate_history=dict(snapshot.get("candidate_history") or {}),
        candidate_timeline=dict(snapshot.get("candidate_timeline") or {}),
        train_queue=dict(snapshot.get("train_queue") or {}),
        operations_overview=dict(snapshot.get("operations_overview") or {}),
        operations_console=operations_console_payload,
        operations_alerts=list(operations_surface.get("alerts") or []),
        operations_health=dict(operations_surface.get("health") or {}),
        operations_recovery=dict(operations_surface.get("recovery") or {}),
        operations_next_actions=list(operations_surface.get("next_actions") or []),
        operations_dashboard=operations_dashboard,
        operations_alert_policy=operations_alert_policy,
        operations_event_stream=dict(operations_event_stream or {}),
        operations_timeline=dict(operations_timeline or {}),
        daemon_timeline=dict(daemon_timeline or {}),
        operations_daemon_timeline=dict(daemon_timeline or {}),
        runner_timeline=dict(runner_timeline or {}),
        operations_runner_timeline=dict(runner_timeline or {}),
        latest_adapter=latest_adapter,
        runtime={
            "workspace": services.workspace,
            "latest_adapter_version": snapshot.get("latest_adapter_version"),
            "health_source": services.provider,
            "management_local_only": services.security.local_management_only,
        },
        metadata={
            "inference": inference_status,
            "capabilities": dict(snapshot.get("capabilities") or {}),
            "user_modeling": dict(snapshot.get("user_modeling") or {}),
            "auto_train_trigger": dict(snapshot.get("auto_train_trigger") or {}),
            "auto_train_trigger_action": dict(snapshot.get("auto_train_trigger_action") or {}),
            "candidate_summary": dict(snapshot.get("candidate_summary") or {}),
            "candidate_action": dict(snapshot.get("candidate_action") or {}),
            "candidate_history": dict(snapshot.get("candidate_history") or {}),
            "candidate_timeline": dict(snapshot.get("candidate_timeline") or {}),
            "train_queue": dict(snapshot.get("train_queue") or {}),
            "operations_overview": dict(snapshot.get("operations_overview") or {}),
            "operations_console": dict(operations_console_payload),
            "operations_alerts": list(operations_surface.get("alerts") or []),
            "operations_health": dict(operations_surface.get("health") or {}),
            "operations_recovery": dict(operations_surface.get("recovery") or {}),
            "operations_next_actions": list(operations_surface.get("next_actions") or []),
            "operations_dashboard": operations_dashboard,
            "operations_alert_policy": operations_alert_policy,
            "operations_event_stream": dict(operations_event_stream or {}),
            "operations_timeline": dict(operations_timeline or {}),
            "daemon_timeline": dict(daemon_timeline or {}),
            "operations_daemon_timeline": dict(daemon_timeline or {}),
            "runner_timeline": dict(runner_timeline or {}),
            "operations_runner_timeline": dict(runner_timeline or {}),
            "export": _build_status_export_metadata(snapshot, services),
            "trainer": trainer_metadata,
            "real_execution": dict(trainer_metadata.get("real_execution_summary") or {}),
            "export_toolchain": dict(trainer_metadata.get("export_toolchain_summary") or {}),
            "lifecycle": _build_status_lifecycle_metadata(snapshot, services),
            "server_runtime": _build_status_server_runtime_metadata(services),
            "signal": {
                "count": int((snapshot or {}).get("signal_count", 0) or 0),
                "latest_signal": dict((snapshot or {}).get("latest_signal") or {}),
                "summary": dict(snapshot.get("signal_summary") or {}),
                "sample_count": int(snapshot.get("signal_sample_count", 0) or 0),
                "sample_counts": dict(snapshot.get("signal_sample_counts") or {}),
                "sample_details": list(snapshot.get("signal_sample_details") or []),
            },
            "pipeline": pipeline_status,
            "snapshot": snapshot,
        },
    )
    return _json_response(payload.model_dump(mode="json"), status_code=200)


def _build_auto_train_action_response(snapshot: Mapping[str, Any]) -> AutoTrainTriggerActionResponse:
    action = dict(snapshot.get("auto_train_trigger_action") or {})
    trigger = dict(snapshot.get("auto_train_trigger") or {})
    train_queue = dict(snapshot.get("train_queue") or {})
    return AutoTrainTriggerActionResponse(
        action=str(action.get("action") or "retry"),
        status=str(action.get("status") or "blocked"),
        reason=action.get("reason"),
        state_path=action.get("state_path"),
        triggered=bool(action.get("triggered", False)),
        queue_job_id=action.get("queue_job_id"),
        queue_job_ids=[str(item) for item in list(action.get("queue_job_ids") or []) if item],
        confirmation_reason=action.get("confirmation_reason"),
        approval_reason=action.get("approval_reason"),
        rejection_reason=action.get("rejection_reason"),
        operator_note=action.get("operator_note"),
        processed_count=int(action.get("processed_count", 0) or 0),
        completed_count=int(action.get("completed_count", 0) or 0),
        failed_count=int(action.get("failed_count", 0) or 0),
        limit=int(action.get("limit")) if action.get("limit") not in (None, "") else None,
        max_iterations=int(action.get("max_iterations")) if action.get("max_iterations") not in (None, "") else None,
        max_cycles=int(action.get("max_cycles")) if action.get("max_cycles") not in (None, "") else None,
        max_seconds=float(action.get("max_seconds")) if action.get("max_seconds") not in (None, "") else None,
        loop_cycles=int(action.get("loop_cycles")) if action.get("loop_cycles") not in (None, "") else None,
        idle_rounds=int(action.get("idle_rounds")) if action.get("idle_rounds") not in (None, "") else None,
        poll_interval_seconds=float(action.get("poll_interval_seconds")) if action.get("poll_interval_seconds") not in (None, "") else None,
        remaining_queued=int(action.get("remaining_queued", 0) or 0),
        drained=bool(action.get("drained", False)),
        stopped_reason=action.get("stopped_reason"),
        triggered_version=action.get("triggered_version"),
        promoted_version=action.get("promoted_version"),
        auto_train_trigger=trigger,
        review_summary=dict(train_queue.get("review_summary") or {}),
        metadata={
            "auto_train_trigger_action": action,
            "snapshot": dict(snapshot),
        },
    )


def _build_candidate_action_response(snapshot: Mapping[str, Any]) -> CandidateActionResponse:
    action = dict(snapshot.get("candidate_action") or {})
    candidate_summary = dict(snapshot.get("candidate_summary") or {})
    candidate_history = dict(snapshot.get("candidate_history") or {})
    return CandidateActionResponse(
        action=str(action.get("action") or "promote_candidate"),
        status=str(action.get("status") or "noop"),
        reason=action.get("reason"),
        state_path=action.get("state_path"),
        triggered=bool(action.get("triggered", False)),
        candidate_version=action.get("candidate_version"),
        promoted_version=action.get("promoted_version"),
        archived_version=action.get("archived_version"),
        operator_note=action.get("operator_note"),
        candidate_summary=candidate_summary,
        candidate_history=candidate_history,
        metadata={
            "candidate_action": action,
            "candidate_history": candidate_history,
            "snapshot": dict(snapshot),
        },
    )


async def handle_auto_train_reset(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    snapshot = await services.pipeline.reset_auto_train_trigger()
    payload = _build_auto_train_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_auto_train_retry(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    snapshot = await services.pipeline.retry_auto_train_trigger()
    payload = _build_auto_train_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_auto_train_process_next(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    snapshot = await services.pipeline.process_next_train_queue()
    payload = _build_auto_train_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


def _parse_positive_query_int(query_params: Mapping[str, str], key: str, default: int) -> int:
    raw = query_params.get(key)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(1, value)


def _parse_optional_note(query_params: Mapping[str, str]) -> str | None:
    raw = query_params.get("note")
    if raw is None:
        return None
    note = str(raw).strip()
    return note or None


async def handle_auto_train_process_batch(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    limit = _parse_positive_query_int(envelope.query_params, "limit", 5)
    snapshot = await services.pipeline.process_train_queue_batch(limit=limit)
    payload = _build_auto_train_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_auto_train_process_until_idle(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    max_iterations = _parse_positive_query_int(envelope.query_params, "max_iterations", 10)
    snapshot = await services.pipeline.process_train_queue_until_idle(max_iterations=max_iterations)
    payload = _build_auto_train_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_auto_train_run_worker_loop(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    max_cycles = _parse_positive_query_int(envelope.query_params, "max_cycles", 10)
    idle_rounds = _parse_positive_query_int(envelope.query_params, "idle_rounds", 1)
    poll_interval_value = envelope.query_params.get("poll_interval_seconds")
    try:
        poll_interval_seconds = float(poll_interval_value) if poll_interval_value not in (None, "") else 0.0
    except Exception:
        poll_interval_seconds = 0.0
    snapshot = await services.pipeline.run_train_queue_worker_loop(
        max_cycles=max_cycles,
        idle_rounds=idle_rounds,
        poll_interval_seconds=max(0.0, poll_interval_seconds),
    )
    payload = _build_auto_train_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_auto_train_run_worker_runner(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    max_seconds_value = envelope.query_params.get("max_seconds")
    idle_sleep_value = envelope.query_params.get("idle_sleep_seconds")
    try:
        max_seconds = float(max_seconds_value) if max_seconds_value not in (None, "") else None
    except Exception:
        max_seconds = None
    try:
        idle_sleep_seconds = float(idle_sleep_value) if idle_sleep_value not in (None, "") else None
    except Exception:
        idle_sleep_seconds = None
    snapshot = await services.pipeline.run_train_queue_worker_runner(
        max_seconds=max_seconds,
        idle_sleep_seconds=idle_sleep_seconds,
    )
    payload = _build_auto_train_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_auto_train_stop_worker_runner(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    snapshot = await services.pipeline.stop_train_queue_worker_runner()
    payload = _build_auto_train_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_auto_train_worker_runner_status(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    payload = await services.pipeline.train_queue_worker_runner_status()
    timeline = await services.pipeline.train_queue_worker_runner_timeline(limit=5)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["runner_timeline"] = dict(timeline or {})
        payload["operations_runner_timeline"] = dict(timeline or {})
    return _json_response(payload, status_code=200)


async def handle_auto_train_worker_runner_history(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    limit = _parse_positive_query_int(envelope.query_params, "limit", 10)
    payload = await services.pipeline.train_queue_worker_runner_history(limit=limit)
    return _json_response(payload, status_code=200)


async def handle_auto_train_start_worker_daemon(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    payload = await services.pipeline.start_train_queue_daemon(note=_parse_optional_note(envelope.query_params))
    return _json_response(payload, status_code=200)


async def handle_auto_train_stop_worker_daemon(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    payload = await services.pipeline.stop_train_queue_daemon(note=_parse_optional_note(envelope.query_params))
    return _json_response(payload, status_code=200)


async def handle_auto_train_worker_daemon_status(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    payload = await services.pipeline.train_queue_daemon_status()
    return _json_response(payload, status_code=200)


async def handle_auto_train_worker_daemon_history(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    limit = _parse_positive_query_int(envelope.query_params, "limit", 10)
    payload = await services.pipeline.train_queue_daemon_history(limit=limit)
    return _json_response(payload, status_code=200)


async def handle_auto_train_recover_worker_daemon(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    payload = await services.pipeline.recover_train_queue_daemon(note=_parse_optional_note(envelope.query_params))
    return _json_response(payload, status_code=200)


async def handle_auto_train_restart_worker_daemon(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    payload = await services.pipeline.restart_train_queue_daemon(note=_parse_optional_note(envelope.query_params))
    return _json_response(payload, status_code=200)


async def handle_auto_train_approve_next(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    snapshot = await services.pipeline.approve_next_train_queue(note=_parse_optional_note(envelope.query_params))
    payload = _build_auto_train_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_auto_train_reject_next(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    snapshot = await services.pipeline.reject_next_train_queue(note=_parse_optional_note(envelope.query_params))
    payload = _build_auto_train_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_candidate_promote(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    snapshot = await services.pipeline.promote_candidate(note=_parse_optional_note(envelope.query_params))
    payload = _build_candidate_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_candidate_archive(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    snapshot = await services.pipeline.archive_candidate(note=_parse_optional_note(envelope.query_params))
    payload = _build_candidate_action_response(snapshot)
    return _json_response(payload.model_dump(mode="json"), status_code=200)


async def handle_candidate_history(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    limit = _parse_positive_query_int(envelope.query_params, "limit", 10)
    payload = await services.pipeline.candidate_history(limit=limit)
    return _json_response(payload, status_code=200)


async def handle_candidate_timeline(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    limit = _parse_positive_query_int(envelope.query_params, "limit", 10)
    payload = await services.pipeline.candidate_timeline(limit=limit)
    return _json_response(payload, status_code=200)


async def handle_train_queue_history(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    limit = _parse_positive_query_int(envelope.query_params, "limit", 10)
    job_id = envelope.query_params.get("job_id")
    payload = await services.pipeline.train_queue_history(job_id=job_id, limit=limit)
    return _json_response(payload, status_code=200)


async def handle_signals(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    try:
        from pfe_core.db.sqlite import list_signals
        signals = list_signals()
        return _json_response({"signals": signals}, status_code=200)
    except Exception as exc:
        return _json_response({"signals": [], "error": str(exc)}, status_code=500)



def _start_training_job(
    services: ServiceBundle,
    *,
    method: str,
    training_config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    retry_of: str | None = None,
    reason: str | None = None,
) -> Any:
    workspace = services.workspace
    store = _training_job_store(workspace)

    payload = start_studio_training_job(
        workspace=workspace,
        pipeline=services.pipeline,
        method=method,
        training_config=training_config,
        preflight=preflight,
        retry_of=retry_of,
        reason=reason,
        persist_job=store.persist_job,
        persist_overall=store.persist_overall,
        build_jobs_payload=lambda limit: store.build_jobs_payload(limit=limit),
    )
    return _json_response(payload, status_code=202)


async def handle_training_jobs(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    body = _load_request_json(envelope.body)
    request = studio_training_request_from_body(body, envelope.query_params)
    if request.unsupported_method:
        payload = _error_payload(
            "unsupported training method",
            "unsupported_training_method",
            "choose sft or dpo for Studio training jobs",
        )
        payload.update(
            {
                "kind": "pfe_training_request_error",
                "request": request.payload(),
                "supported_methods": list(SUPPORTED_STUDIO_TRAINING_METHODS),
            }
        )
        return _json_response(payload, status_code=400)
    preflight = _build_training_preflight_payload(envelope, services, request)
    if not request.confirmed:
        payload = _error_payload(
            "confirmation required",
            "confirmation_required",
            "pass confirm=true to start this training job",
        )
        payload.update(
            {
                "kind": "pfe_training_preflight_required",
                "request": request.payload(),
                "preflight": preflight,
                "confirm_api": "POST /pfe/training/jobs",
            }
        )
        return _json_response(payload, status_code=409)
    if not preflight.get("ready"):
        payload = _error_payload(
            "training preflight failed",
            "training_preflight_failed",
            "resolve preflight blockers before starting this training job",
        )
        payload.update(
            {
                "kind": "pfe_training_preflight_failed",
                "request": request.payload(),
                "preflight": preflight,
            }
        )
        return _json_response(payload, status_code=409)
    store = _training_job_store(services.workspace)
    active_job = store.active_job()
    if active_job is not None:
        payload = _error_payload(
            "training job is already active",
            "training_job_already_active",
            "wait for the active job to finish or cancel it before starting another one",
        )
        payload.update({"active_job": active_job, "jobs": store.build_jobs_payload(limit=10)})
        return _json_response(payload, status_code=409)
    return _start_training_job(
        services,
        method=request.method,
        training_config=request.training_config,
        preflight=preflight,
    )


async def handle_training_jobs_list(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    limit = _parse_positive_query_int(envelope.query_params, "limit", 20)
    return _json_response(_build_training_jobs_payload(services, limit=limit), status_code=200)


async def handle_training_job(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    path_parts = envelope.path.strip("/").split("/")
    job_id = path_parts[-1] if path_parts else ""
    if not job_id or job_id == "jobs":
        return _json_response(_error_payload("job_id required", "bad_request"), status_code=400)
    job_entry = _training_job_store(services.workspace).get_job(job_id)
    if job_entry is None:
        return _json_response(_error_payload("job not found", "not_found"), status_code=404)
    return _json_response(_training_job_payload(job_entry), status_code=200)


async def handle_training_job_events(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    path_parts = envelope.path.strip("/").split("/")
    job_id = path_parts[-2] if len(path_parts) >= 2 and path_parts[-1] == "events" else ""
    if not job_id:
        return _json_response(_error_payload("job_id required", "bad_request"), status_code=400)
    workspace = services.workspace
    job_entry = _training_job_store(workspace).get_job(job_id)
    if job_entry is None:
        return _json_response(_error_payload("job not found", "not_found"), status_code=404)
    events = job_entry.get("events")
    items = list(events) if isinstance(events, list) else []
    return _json_response(
        {
            "job_id": job_id,
            "workspace": workspace,
            "count": len(items),
            "items": items,
            "latest": items[-1] if items else None,
            "job": _training_job_payload(job_entry),
        },
        status_code=200,
    )


async def handle_training_job_cancel(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    body = _load_request_json(envelope.body)
    confirmed = _query_bool(envelope.query_params, "confirm") or _body_bool(body, "confirm")
    path_parts = envelope.path.strip("/").split("/")
    job_id = path_parts[-2] if len(path_parts) >= 2 and path_parts[-1] == "cancel" else ""
    if not job_id:
        return _json_response(_error_payload("job_id required", "bad_request"), status_code=400)
    if not confirmed:
        return _json_response(
            _error_payload("confirmation required", "confirmation_required", "pass confirm=true to cancel this training job"),
            status_code=409,
        )
    store = _training_job_store(services.workspace)
    result = store.cancel_job(job_id)
    if result["outcome"] == "not_found":
        return _json_response(_error_payload("job not found", "not_found"), status_code=404)
    if result["outcome"] == "not_cancellable":
        return _json_response(
            _error_payload("job is no longer cancellable", "job_not_cancellable", "only queued or running jobs can be cancelled"),
            status_code=409,
        )
    return _json_response(
        {
            "success": True,
            "action": result["action"],
            "message": result["message"],
            "job_id": job_id,
            "job": result["job"],
            "jobs": store.build_jobs_payload(limit=10),
        },
        status_code=200,
    )


async def handle_training_job_retry(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    body = _load_request_json(envelope.body)
    confirmed = _query_bool(envelope.query_params, "confirm") or _body_bool(body, "confirm")
    path_parts = envelope.path.strip("/").split("/")
    job_id = path_parts[-2] if len(path_parts) >= 2 and path_parts[-1] == "retry" else ""
    if not job_id:
        return _json_response(_error_payload("job_id required", "bad_request"), status_code=400)

    store = _training_job_store(services.workspace)
    job_entry = store.get_job(job_id)
    if job_entry is None:
        return _json_response(_error_payload("job not found", "not_found"), status_code=404)
    status = str(job_entry.get("status") or "")
    if status not in {"failed", "cancelled"}:
        return _json_response(
            _error_payload("job is not retryable", "job_not_retryable", "only failed or cancelled jobs can be retried"),
            status_code=409,
        )

    original_config = job_entry.get("training_config")
    retry_body = dict(original_config) if isinstance(original_config, Mapping) else {}
    method = str(job_entry.get("method") or "").strip().lower()
    if not method:
        return _json_response(
            _error_payload(
                "training method missing",
                "training_method_missing",
                "this job was created without a retryable Studio training method",
            ),
            status_code=409,
        )
    retry_body["method"] = method
    retry_request = studio_training_request_from_body(retry_body, envelope.query_params, confirmed=confirmed)
    if retry_request.unsupported_method:
        payload = _error_payload(
            "unsupported training method",
            "unsupported_training_method",
            "choose sft or dpo before retrying this training job",
        )
        payload.update(
            {
                "kind": "pfe_training_request_error",
                "retry_of": job_id,
                "request": retry_request.payload(),
                "supported_methods": list(SUPPORTED_STUDIO_TRAINING_METHODS),
            }
        )
        return _json_response(payload, status_code=400)
    preflight = _build_training_preflight_payload(envelope, services, retry_request)
    if not confirmed:
        payload = _error_payload(
            "confirmation required",
            "confirmation_required",
            "pass confirm=true to retry this training job",
        )
        payload.update(
            {
                "kind": "pfe_training_preflight_required",
                "preflight": preflight,
                "confirm_api": f"POST /pfe/training/jobs/{job_id}/retry",
                "retry_of": job_id,
                "request": retry_request.payload(),
            }
        )
        return _json_response(payload, status_code=409)
    if not preflight.get("ready"):
        payload = _error_payload(
            "training preflight failed",
            "training_preflight_failed",
            "resolve preflight blockers before retrying this training job",
        )
        payload.update(
            {
                "kind": "pfe_training_preflight_failed",
                "preflight": preflight,
                "retry_of": job_id,
                "request": retry_request.payload(),
            }
        )
        return _json_response(payload, status_code=409)

    active_job = store.active_job()
    if active_job is not None:
        payload = _error_payload(
            "training job is already active",
            "training_job_already_active",
            "wait for the active job to finish or cancel it before retrying this job",
        )
        payload.update({"active_job": active_job, "retry_of": job_id, "jobs": store.build_jobs_payload(limit=10)})
        return _json_response(payload, status_code=409)

    retry_result = store.mark_retry_requested(job_id)
    if retry_result["outcome"] == "not_found":
        return _json_response(_error_payload("job not found", "not_found"), status_code=404)
    if retry_result["outcome"] == "not_retryable":
        return _json_response(
            _error_payload("job is not retryable", "job_not_retryable", "only failed or cancelled jobs can be retried"),
            status_code=409,
        )
    return _start_training_job(
        services,
        method=retry_request.method,
        training_config=retry_request.training_config,
        preflight=preflight,
        retry_of=job_id,
    )


async def handle_training_status(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    workspace = services.workspace
    state = _training_job_store(workspace).current_overall_state()
    if not state:
        state = {"state": "idle", "adapter_version": None}
    # Derive auto-train state from pipeline when no explicit server trigger is tracked
    if state.get("state") == "idle":
        pipeline = getattr(services.pipeline, "pipeline", None)
        if pipeline is not None and hasattr(pipeline, "_load_auto_trigger_state"):
            try:
                auto_state = pipeline._load_auto_trigger_state(workspace=workspace)
                last_result = dict(auto_state.get("last_result") or {})
                if last_result.get("triggered"):
                    auto_state_out = {
                        "state": "completed",
                        "adapter_version": last_result.get("triggered_version") or last_result.get("promoted_version"),
                        "reason": last_result.get("reason", "auto_train"),
                        "auto_train": True,
                        "promoted_version": last_result.get("promoted_version"),
                    }
                    if last_result.get("error"):
                        auto_state_out["state"] = "failed"
                        auto_state_out["error"] = last_result.get("error")
                    return _json_response(auto_state_out, status_code=200)
            except Exception:
                pass
    return _json_response(dict(state), status_code=200)


async def handle_training_trigger(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    workspace = services.workspace
    body = _load_request_json(envelope.body)
    reason = str(body.pop("reason", "manual_trigger") or "manual_trigger")
    body.setdefault("method", "sft")
    request = studio_training_request_from_body(body, envelope.query_params, confirmed=True)
    if request.unsupported_method:
        payload = _error_payload(
            "unsupported training method",
            "unsupported_training_method",
            "choose sft or dpo for legacy training trigger",
        )
        payload.update(
            {
                "kind": "pfe_training_request_error",
                "legacy_endpoint": "/pfe/training/trigger",
                "request": request.payload(),
                "supported_methods": list(SUPPORTED_STUDIO_TRAINING_METHODS),
            }
        )
        return _json_response(payload, status_code=400)

    store = _training_job_store(workspace)
    active_job = store.active_job()
    if active_job is not None:
        return _json_response(
            {
                "kind": "pfe_training_trigger_legacy_active",
                "legacy_endpoint": "/pfe/training/trigger",
                "state": active_job.get("status") or "running",
                "reason": reason,
                "job_id": active_job.get("job_id"),
                "active_job": active_job,
                "jobs": store.build_jobs_payload(limit=10),
            },
            status_code=202,
        )

    preflight = _build_legacy_training_trigger_preflight_payload(services, request)
    status_code, payload = _response_status_payload(
        _start_training_job(
            services,
            method=request.method,
            training_config=request.training_config,
            preflight=preflight,
            reason=reason,
        )
    )
    payload.update(
        {
            "legacy_endpoint": "/pfe/training/trigger",
            "state": payload.get("status"),
            "reason": reason,
        }
    )
    return _json_response(payload, status_code=status_code)


async def handle_eval_run(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    workspace = services.workspace
    body = _load_request_json(envelope.body)
    confirmed = _query_bool(envelope.query_params, "confirm") or _body_bool(body, "confirm")
    version = str(body.get("version") or envelope.query_params.get("version") or "latest")
    if not confirmed:
        payload = _error_payload(
            "confirmation required",
            "confirmation_required",
            "pass confirm=true to evaluate this version",
        )
        payload.update(
            {
                "version": version,
                "confirm_api": "POST /pfe/eval",
                "status_url": EVAL_STATUS_URL,
            }
        )
        return _json_response(payload, status_code=409)

    try:
        from pfe_core.adapter_store import create_adapter_store
        store = create_adapter_store(workspace=workspace)
        latest = store.current_latest_version()
        eval_version = version if version != "latest" else (latest or "")
        if not eval_version:
            return _json_response(_error_payload("adapter not found", "not_found"), status_code=404)
        store.load(eval_version)
    except Exception as exc:
        return _json_response(_error_payload(str(exc), "not_found"), status_code=404)

    current_eval_state = _current_eval_state(workspace)
    if isinstance(current_eval_state, Mapping) and str(current_eval_state.get("state") or "") == "running":
        return _json_response(
            _error_payload("evaluation is already running", "eval_already_running", f"wait for {EVAL_STATUS_URL} to finish"),
            status_code=409,
        )

    def _persist_eval_state(eval_workspace: str, state: dict[str, Any]) -> None:
        _eval_overall_state[eval_workspace] = state
        _save_json_state(_eval_state_path(eval_workspace), state)

    payload = start_studio_eval_job(
        pipeline=services.pipeline,
        workspace=workspace,
        version=eval_version,
        requested_version=version,
        request_body=body,
        default_base_model=_default_chat_base_model,
        load_adapter_path=store.load,
        persist_state=_persist_eval_state,
        build_adapters_payload=lambda: _build_adapters_payload(services),
    )
    return _json_response(payload, status_code=202)


async def handle_eval_status(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    workspace = services.workspace
    state = _current_eval_state(workspace)
    # Derive auto-eval state from pipeline when no explicit server eval is tracked
    if not state or state.get("state") == "idle":
        pipeline = getattr(services.pipeline, "pipeline", None)
        if pipeline is not None and hasattr(pipeline, "_load_auto_trigger_state"):
            try:
                auto_state = pipeline._load_auto_trigger_state(workspace=workspace)
                last_result = dict(auto_state.get("last_result") or {})
                eval_state = auto_eval_state_from_last_result(last_result)
                if eval_state:
                    return _json_response(eval_state, status_code=200)
            except Exception:
                pass
    payload = build_eval_status_payload(state, adapters=_build_adapters_payload(services))
    return _json_response(payload, status_code=200)


async def handle_adapter_latest(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    try:
        from pfe_core.adapter_store import create_adapter_store
        store = create_adapter_store(workspace=services.workspace)
        version = store.current_latest_version()
        if not version:
            return _json_response(_error_payload("no adapters found", "not_found"), status_code=404)
        return _json_response({"version": version}, status_code=200)
    except Exception as exc:
        return _json_response(_error_payload(str(exc), "internal_error"), status_code=500)


async def handle_adapter_version(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    path_parts = envelope.path.strip("/").split("/")
    version = path_parts[-1] if path_parts else ""
    if not version:
        return _json_response(_error_payload("version required", "bad_request"), status_code=400)
    try:
        from pfe_core.adapter_store import create_adapter_store
        store = create_adapter_store(workspace=services.workspace)
        rows = store.list_version_records(limit=100)
        for row in rows:
            if str(row.get("version")) == version:
                return _json_response(dict(row), status_code=200)
        return _json_response(_error_payload("adapter not found", "not_found"), status_code=404)
    except Exception as exc:
        return _json_response(_error_payload(str(exc), "internal_error"), status_code=500)


def _adapter_action_version(envelope: RequestEnvelope, action: str) -> str:
    path_parts = envelope.path.strip("/").split("/")
    if len(path_parts) >= 4 and path_parts[-1] == action:
        return path_parts[-2]
    return ""


def _adapter_action_confirmed(envelope: RequestEnvelope, body: Mapping[str, Any]) -> bool:
    return _query_bool(envelope.query_params, "confirm") or _body_bool(body, "confirm")


def _adapter_action_payload(
    *,
    services: ServiceBundle,
    action: str,
    version: str,
    previous_version: str | None,
    message: str,
) -> dict[str, Any]:
    adapters = _build_adapters_payload(services)
    return {
        "success": True,
        "action": action,
        "version": version,
        "target_version": version,
        "previous_version": previous_version,
        "current_version": adapters.get("latest_version"),
        "message": message,
        "adapters": adapters,
        "fallback": adapters.get("fallback"),
    }


def _rollback_adapter_version(store: Any, version: str) -> str:
    rows = store.list_version_records(limit=1000)
    row = next((item for item in rows if str(item.get("version") or "") == version), None)
    if row and str(row.get("state") or "").lower() == "archived":
        adapter_dir = Path(store.load(version))
        manifest_path = adapter_dir / "adapter_manifest.json"
        manifest: dict[str, Any] = {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        metrics = manifest.get("training_metrics") if isinstance(manifest.get("training_metrics"), dict) else None
        num_samples = int(manifest.get("num_samples", row.get("num_samples", 0)) or 0)
        store.mark_pending_eval(version, num_samples=num_samples, metrics=metrics)
    return store.rollback(version)


async def handle_adapter_promote(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    body = _load_request_json(envelope.body)
    version = _adapter_action_version(envelope, "promote")
    if not version:
        return _json_response(_error_payload("version required", "bad_request"), status_code=400)
    if not _adapter_action_confirmed(envelope, body):
        return _json_response(
            _error_payload("confirmation required", "confirmation_required", "pass confirm=true to promote this version"),
            status_code=409,
        )
    try:
        from pfe_core.adapter_store import create_adapter_store
        store = create_adapter_store(workspace=services.workspace)
        previous_version = store.current_latest_version()
        message = store.promote(version)
        return _json_response(
            _adapter_action_payload(
                services=services,
                action="promote",
                version=version,
                previous_version=previous_version,
                message=message,
            ),
            status_code=200,
        )
    except Exception as exc:
        return _json_response(_error_payload(str(exc), "adapter_action_failed"), status_code=409)


async def handle_adapter_rollback(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    body = _load_request_json(envelope.body)
    version = _adapter_action_version(envelope, "rollback")
    if not version:
        return _json_response(_error_payload("version required", "bad_request"), status_code=400)
    if not _adapter_action_confirmed(envelope, body):
        return _json_response(
            _error_payload("confirmation required", "confirmation_required", "pass confirm=true to rollback to this version"),
            status_code=409,
        )
    try:
        from pfe_core.adapter_store import create_adapter_store
        store = create_adapter_store(workspace=services.workspace)
        previous_version = store.current_latest_version()
        message = _rollback_adapter_version(store, version)
        return _json_response(
            _adapter_action_payload(
                services=services,
                action="rollback",
                version=version,
                previous_version=previous_version,
                message=message,
            ),
            status_code=200,
        )
    except Exception as exc:
        return _json_response(_error_payload(str(exc), "adapter_action_failed"), status_code=409)


async def handle_adapter_archive(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    body = _load_request_json(envelope.body)
    version = _adapter_action_version(envelope, "archive")
    if not version:
        return _json_response(_error_payload("version required", "bad_request"), status_code=400)
    if not _adapter_action_confirmed(envelope, body):
        return _json_response(
            _error_payload("confirmation required", "confirmation_required", "pass confirm=true to archive this version"),
            status_code=409,
        )
    try:
        from pfe_core.adapter_store import create_adapter_store
        store = create_adapter_store(workspace=services.workspace)
        previous_version = store.current_latest_version()
        message = store.archive(version)
        return _json_response(
            _adapter_action_payload(
                services=services,
                action="archive",
                version=version,
                previous_version=previous_version,
                message=message,
            ),
            status_code=200,
        )
    except Exception as exc:
        return _json_response(_error_payload(str(exc), "adapter_action_failed"), status_code=409)


async def handle_training_checkpoints(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    path_parts = envelope.path.strip("/").split("/")
    job_id = path_parts[-2] if len(path_parts) >= 3 else ""
    checkpoints: list[dict[str, Any]] = []
    return _json_response({"job_id": job_id, "checkpoints": checkpoints}, status_code=200)


async def handle_training_dead_letter(
    envelope: RequestEnvelope,
    services: ServiceBundle,
) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    try:
        from pfe_core.reliability import DeadLetterQueue
        dlq = DeadLetterQueue(workspace=services.workspace)
        entries = dlq.list_entries(limit=1000)
        return _json_response(
            {
                "entries": [
                    {
                        "job_id": entry.job_id,
                        "failure_reason": entry.failure_reason,
                        "retry_count": entry.retry_count,
                        "created_at": entry.failed_at.isoformat() if entry.failed_at else None,
                        "resolved": entry.resolved,
                    }
                    for entry in entries
                ]
            },
            status_code=200,
        )
    except Exception:
        return _json_response({"entries": []}, status_code=200)


async def handle_frontend(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    del envelope, services
    return _html_response(_studio_html(), status_code=200)


async def handle_chat_frontend(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    del envelope, services
    return _html_response(_frontend_html(), status_code=200)


async def handle_studio_frontend(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    del envelope, services
    return _html_response(_studio_html(), status_code=200)


async def handle_dashboard_frontend(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    """Serve the dashboard HTML."""
    del envelope, services
    html_path = _repo_root() / "pfe-server" / "pfe_server" / "static" / "dashboard.html"
    if html_path.exists():
        return _html_response(html_path.read_text(encoding="utf-8"), status_code=200)
    return _json_response({"error": "dashboard.html not found"}, status_code=404)


async def handle_runtime(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    return _json_response(_build_runtime_payload(envelope, services), status_code=200)


async def handle_workspaces(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    if envelope.method == "GET":
        return _json_response(_build_workspaces_payload(services), status_code=200)
    body = _load_request_json(envelope.body)
    name = str(body.get("name") or body.get("workspace") or body.get("id") or "").strip()
    validation_issues = _workspace_slug_issues(name)
    if validation_issues:
        return _json_response(
            {
                "saved": False,
                "selected": name,
                "validation": {"valid": False, "issues": validation_issues},
            },
            status_code=422,
        )
    paths = _workspace_paths(name)
    existed = paths["adapters"].exists() or paths["state"].exists()
    paths["adapters"].mkdir(parents=True, exist_ok=True)
    paths["state"].mkdir(parents=True, exist_ok=True)
    switch_requested = "switch" not in body or _body_bool(body, "switch")
    previous = services.workspace
    current = services.workspace
    if switch_requested:
        previous, current = _set_active_workspace(services, name)
    return _json_response(
        {
            "saved": True,
            "created": not existed,
            "switched": switch_requested,
            "changed": previous != current,
            "previous": previous,
            "current": current,
            "selected": name,
            "workspace": _workspace_record(name, current=current),
            "workspaces": _build_workspaces_payload(services),
            "runtime": _build_runtime_payload(envelope, services),
            "adapters": _build_adapters_payload(services),
            "readiness": _build_readiness_payload(envelope, services),
            "effective_scope": "current_process_next_request",
            "persisted": False,
        },
        status_code=200,
    )


async def handle_models(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    return _json_response(_build_models_payload(services), status_code=200)


async def handle_handoff(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    return _json_response(_build_handoff_payload(envelope, services), status_code=200)


async def handle_handoff_test(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    body = _load_request_json(envelope.body)
    message = str(body.get("message") or "hello from PFE Studio handoff test")
    action = str(body.get("action") or "accept").strip().lower()
    if action not in {"accept", "reject", "edit", "regenerate", "delete"}:
        action = "accept"
    model_parameter = "local"
    handoff = _build_handoff_payload(envelope, services)
    model_payload = handoff.get("model") if isinstance(handoff.get("model"), Mapping) else {}
    if isinstance(model_payload, Mapping):
        model_parameter = str(model_payload.get("api_parameter") or model_parameter)

    chat_envelope = RequestEnvelope(
        method="POST",
        path="/v1/chat/completions",
        headers=dict(envelope.headers),
        client_host=envelope.client_host,
        body=json.dumps(
            {
                "model": model_parameter,
                "messages": [{"role": "user", "content": message}],
                "metadata": {"source": "pfe_studio_handoff_test"},
            }
        ).encode("utf-8"),
    )
    chat_status, chat_payload = _response_status_payload(
        await handle_chat_completions(chat_envelope, services)
    )
    session_id = str(chat_payload.get("session_id") or "")
    request_id = str(chat_payload.get("request_id") or "")
    assistant_message = ""
    choices = chat_payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message_payload = choice.get("message") if isinstance(choice, Mapping) else {}
        if isinstance(message_payload, Mapping):
            assistant_message = str(message_payload.get("content") or "")

    feedback_status = 0
    feedback_payload: dict[str, Any] = {}
    if chat_status == 200 and session_id and request_id:
        feedback_envelope = RequestEnvelope(
            method="POST",
            path="/pfe/feedback",
            headers=dict(envelope.headers),
            client_host=envelope.client_host,
            body=json.dumps(
                {
                    "session_id": session_id,
                    "request_id": request_id,
                    "action": action,
                    "response_time_seconds": 0.1,
                    "user_message": message,
                    "assistant_message": assistant_message,
                    "metadata": {"source": "pfe_studio_handoff_test"},
                }
            ).encode("utf-8"),
        )
        feedback_status, feedback_payload = _response_status_payload(
            await handle_feedback(feedback_envelope, services)
        )

    chat_ok = chat_status == 200 and bool(session_id and request_id)
    feedback_ok = feedback_status == 200 and bool(feedback_payload.get("success"))
    ok = chat_ok and feedback_ok
    return _json_response(
        {
            "kind": "pfe_handoff_test",
            "ok": ok,
            "summary": "聊天和反馈闭环已通过。" if ok else "接入测试未通过。",
            "contract": {
                "chat_url": handoff.get("urls", {}).get("api") if isinstance(handoff.get("urls"), Mapping) else None,
                "feedback_url": handoff.get("urls", {}).get("feedback") if isinstance(handoff.get("urls"), Mapping) else None,
                "response_id_fields": ["session_id", "request_id"],
            },
            "chat": {
                "ok": chat_ok,
                "status_code": chat_status,
                "session_id": session_id,
                "request_id": request_id,
                "model": chat_payload.get("model"),
                "served_by": chat_payload.get("served_by"),
                "assistant_preview": assistant_message[:160],
            },
            "feedback": {
                "ok": feedback_ok,
                "status_code": feedback_status,
                "signal_type": feedback_payload.get("signal_type"),
                "signal_id": feedback_payload.get("signal_id"),
                "pending_found": bool(feedback_payload.get("metadata", {}).get("pending_found"))
                if isinstance(feedback_payload.get("metadata"), Mapping)
                else False,
            },
            "tested_at": round(time.time(), 3),
        },
        status_code=200,
    )


async def handle_readiness(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    return _json_response(_build_readiness_payload(envelope, services), status_code=200)


async def handle_model_config(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    try:
        body = json.loads(envelope.body or b"{}")
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return _json_response(_error_payload("JSON object expected", "invalid_request"), status_code=400)
    model_id = str(body.get("base_model") or body.get("model") or "").strip()
    validation = _validate_model_reference(model_id)
    if not validation["valid"]:
        return _json_response(
            {
                "saved": False,
                "validate_only": _query_bool(envelope.query_params, "validate_only"),
                "selected": model_id,
                "validation": validation,
            },
            status_code=422,
        )
    validate_only = _query_bool(envelope.query_params, "validate_only")
    previous_model = _load_config_base_model()
    changed = model_id != previous_model
    config_path: Path | None = None
    if not validate_only:
        previous_model, config_path = _save_config_base_model(model_id)
        changed = model_id != previous_model
    return _json_response(
        {
            "saved": not validate_only,
            "validate_only": validate_only,
            "changed": changed,
            "previous": previous_model,
            "selected": model_id,
            "config_path": str(config_path) if config_path else None,
            "effective_scope": "not_applied" if validate_only else "next_chat_request",
            "reload_required": False,
            "applies_to_models": ["local", "local-default", "base"],
            "validation": validation,
            "models": _build_models_payload(services),
        },
        status_code=200,
    )


async def handle_real_local_config(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    body = _load_request_json(envelope.body)
    if "enabled" not in body:
        return _json_response(
            _error_payload("enabled is required", "invalid_request"),
            status_code=400,
        )
    requested = _body_bool(body, "enabled")
    previous, enabled = _set_real_local_inference(requested)
    return _json_response(
        {
            "saved": True,
            "previous": previous,
            "enabled": enabled,
            "changed": previous != enabled,
            "env_var": "PFE_ENABLE_REAL_LOCAL_INFERENCE",
            "persisted": False,
            "effective_scope": "current_process_next_request",
            "reload_required": False,
            "readiness": _build_readiness_payload(envelope, services),
        },
        status_code=200,
    )


async def handle_adapters(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    return _json_response(_build_adapters_payload(services), status_code=200)


async def handle_dashboard_metrics(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    """Get complete dashboard metrics."""
    del envelope
    try:
        from .dashboard_api import get_dashboard_api
        api = get_dashboard_api(workspace=services.workspace)
        return _json_response(api.get_dashboard_data(), status_code=200)
    except Exception as e:
        return _json_response({"error": str(e)}, status_code=500)


async def handle_dashboard_training(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    """Get training metrics."""
    del envelope
    try:
        from .dashboard_api import get_dashboard_api
        api = get_dashboard_api(workspace=services.workspace)
        return _json_response(api.get_training_metrics(), status_code=200)
    except Exception as e:
        return _json_response({"error": str(e)}, status_code=500)


async def handle_dashboard_signals(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    """Get signal quality metrics."""
    del envelope
    try:
        from .dashboard_api import get_dashboard_api
        api = get_dashboard_api(workspace=services.workspace)
        return _json_response(api.get_signal_quality(), status_code=200)
    except Exception as e:
        return _json_response({"error": str(e)}, status_code=500)


async def handle_dashboard_adapters(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    """Get adapter comparison data."""
    del envelope
    try:
        from .dashboard_api import get_dashboard_api
        api = get_dashboard_api(workspace=services.workspace)
        return _json_response(api.get_adapter_comparison(), status_code=200)
    except Exception as e:
        return _json_response({"error": str(e)}, status_code=500)


async def handle_dashboard_health(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    """Get system health metrics."""
    del envelope
    try:
        from .dashboard_api import get_dashboard_api
        api = get_dashboard_api(workspace=services.workspace)
        return _json_response(api.get_system_health(), status_code=200)
    except Exception as e:
        return _json_response({"error": str(e)}, status_code=500)


async def handle_observability_trace(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    """GET /pfe/observability/trace/{signal_id}"""
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    signal_id = envelope.path.split("/")[-1]
    try:
        from pfe_core.observability.trace import trace_signal
        trace = trace_signal(signal_id)
        return _json_response(trace.to_dict(), status_code=200)
    except Exception as exc:
        return _json_response({"error": str(exc), "signal_id": signal_id}, status_code=500)


async def handle_observability_version(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    """GET /pfe/observability/version/{version}"""
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    version = envelope.path.split("/")[-1]
    try:
        from pfe_core.observability.trace import trace_version
        vt = trace_version(version)
        return _json_response(vt.to_dict(), status_code=200)
    except Exception as exc:
        return _json_response({"error": str(exc), "version": version}, status_code=500)


async def handle_audit_pii(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    """GET /pfe/audit/pii"""
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    try:
        from pfe_core.data_policy import audit_pii_exposure
        from pfe_core.storage import list_samples
        samples = list_samples(limit=200)
        report = audit_pii_exposure(samples)
        return _json_response(report.to_dict(), status_code=200)
    except Exception as exc:
        return _json_response({"error": str(exc)}, status_code=500)


async def handle_audit_training(envelope: RequestEnvelope, services: ServiceBundle) -> Any:
    """GET /pfe/audit/training"""
    allowed, denial = _route_access(envelope, security=services.security, endpoint_kind="management")
    if not allowed:
        return denial
    try:
        from pfe_core.trainer.training_auditor import TrainingAuditor
        from pfe_core.storage import list_samples
        samples = list_samples(limit=200)
        auditor = TrainingAuditor()
        report = auditor.audit(samples)
        return _json_response(report.to_dict(), status_code=200)
    except Exception as exc:
        return _json_response({"error": str(exc)}, status_code=500)


class _LiteASGIApp:
    def __init__(self, services: ServiceBundle) -> None:
        self.services = services

    async def __call__(self, scope: Mapping[str, Any], receive: Callable[[], Awaitable[dict[str, Any]]], send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        if scope.get("type") != "http":
            await send({"type": "http.response.start", "status": 404, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"detail":"not found"}'})
            return
        envelope = await _envelope_from_asgi_scope(scope, receive)
        response = await self._dispatch(envelope)
        if isinstance(response, tuple) and len(response) == 2:
            payload, status_code = response
            response_headers = {"content-type": "application/json; charset=utf-8"}
        elif isinstance(response, tuple) and len(response) == 3:
            payload, status_code, response_headers = response
        else:
            payload, status_code = response, 200
            response_headers = {"content-type": "application/json; charset=utf-8"}
        if isinstance(payload, bytes):
            body = payload
        elif isinstance(payload, str):
            body = payload.encode("utf-8")
        else:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": int(status_code),
                "headers": [
                    (str(key).encode("latin1"), str(value).encode("latin1"))
                    for key, value in response_headers.items()
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def _dispatch(self, envelope: RequestEnvelope) -> Any:
        if envelope.path == "/" and envelope.method == "GET":
            return await handle_frontend(envelope, self.services)
        if envelope.path in {"/chat", "/pfe/chat"} and envelope.method == "GET":
            return await handle_chat_frontend(envelope, self.services)
        if envelope.path in {"/studio", "/pfe/studio"} and envelope.method == "GET":
            return await handle_studio_frontend(envelope, self.services)
        if envelope.path == "/healthz" and envelope.method == "GET":
            return _json_response({"status": "ok"}, status_code=200)
        if envelope.path == "/v1/chat/completions" and envelope.method == "POST":
            return await handle_chat_completions(envelope, self.services)
        if envelope.path == "/pfe/signal" and envelope.method == "POST":
            return await handle_signal_ingest(envelope, self.services)
        if envelope.path == "/pfe/feedback" and envelope.method == "POST":
            return await handle_feedback(envelope, self.services)
        if envelope.path == "/pfe/distill/run" and envelope.method == "POST":
            return await handle_distill_run(envelope, self.services)
        if envelope.path == "/pfe/auto-train/reset" and envelope.method == "POST":
            return await handle_auto_train_reset(envelope, self.services)
        if envelope.path == "/pfe/auto-train/retry" and envelope.method == "POST":
            return await handle_auto_train_retry(envelope, self.services)
        if envelope.path == "/pfe/auto-train/process-next" and envelope.method == "POST":
            return await handle_auto_train_process_next(envelope, self.services)
        if envelope.path == "/pfe/auto-train/process-batch" and envelope.method == "POST":
            return await handle_auto_train_process_batch(envelope, self.services)
        if envelope.path == "/pfe/auto-train/process-until-idle" and envelope.method == "POST":
            return await handle_auto_train_process_until_idle(envelope, self.services)
        if envelope.path == "/pfe/auto-train/run-worker-loop" and envelope.method == "POST":
            return await handle_auto_train_run_worker_loop(envelope, self.services)
        if envelope.path == "/pfe/auto-train/run-worker-runner" and envelope.method == "POST":
            return await handle_auto_train_run_worker_runner(envelope, self.services)
        if envelope.path == "/pfe/auto-train/stop-worker-runner" and envelope.method == "POST":
            return await handle_auto_train_stop_worker_runner(envelope, self.services)
        if envelope.path == "/pfe/auto-train/worker-runner" and envelope.method == "GET":
            return await handle_auto_train_worker_runner_status(envelope, self.services)
        if envelope.path == "/pfe/auto-train/worker-runner/history" and envelope.method == "GET":
            return await handle_auto_train_worker_runner_history(envelope, self.services)
        if envelope.path == "/pfe/auto-train/start-worker-daemon" and envelope.method == "POST":
            return await handle_auto_train_start_worker_daemon(envelope, self.services)
        if envelope.path == "/pfe/auto-train/stop-worker-daemon" and envelope.method == "POST":
            return await handle_auto_train_stop_worker_daemon(envelope, self.services)
        if envelope.path == "/pfe/auto-train/worker-daemon" and envelope.method == "GET":
            return await handle_auto_train_worker_daemon_status(envelope, self.services)
        if envelope.path == "/pfe/auto-train/worker-daemon/history" and envelope.method == "GET":
            return await handle_auto_train_worker_daemon_history(envelope, self.services)
        if envelope.path == "/pfe/auto-train/recover-worker-daemon" and envelope.method == "POST":
            return await handle_auto_train_recover_worker_daemon(envelope, self.services)
        if envelope.path == "/pfe/auto-train/restart-worker-daemon" and envelope.method == "POST":
            return await handle_auto_train_restart_worker_daemon(envelope, self.services)
        if envelope.path == "/pfe/auto-train/approve-next" and envelope.method == "POST":
            return await handle_auto_train_approve_next(envelope, self.services)
        if envelope.path == "/pfe/auto-train/reject-next" and envelope.method == "POST":
            return await handle_auto_train_reject_next(envelope, self.services)
        if envelope.path == "/pfe/candidate/promote" and envelope.method == "POST":
            return await handle_candidate_promote(envelope, self.services)
        if envelope.path == "/pfe/candidate/archive" and envelope.method == "POST":
            return await handle_candidate_archive(envelope, self.services)
        if envelope.path == "/pfe/candidate/history" and envelope.method == "GET":
            return await handle_candidate_history(envelope, self.services)
        if envelope.path == "/pfe/candidate/timeline" and envelope.method == "GET":
            return await handle_candidate_timeline(envelope, self.services)
        if envelope.path == "/pfe/auto-train/queue-history" and envelope.method == "GET":
            return await handle_train_queue_history(envelope, self.services)
        if envelope.path == "/pfe/status" and envelope.method == "GET":
            return await handle_status(envelope, self.services)
        if envelope.path == "/pfe/runtime" and envelope.method == "GET":
            return await handle_runtime(envelope, self.services)
        if envelope.path == "/pfe/workspaces" and envelope.method in {"GET", "POST"}:
            return await handle_workspaces(envelope, self.services)
        if envelope.path == "/pfe/models" and envelope.method == "GET":
            return await handle_models(envelope, self.services)
        if envelope.path == "/pfe/handoff" and envelope.method == "GET":
            return await handle_handoff(envelope, self.services)
        if envelope.path == "/pfe/handoff/test" and envelope.method == "POST":
            return await handle_handoff_test(envelope, self.services)
        if envelope.path == "/pfe/readiness" and envelope.method == "GET":
            return await handle_readiness(envelope, self.services)
        if envelope.path == "/pfe/config/model" and envelope.method == "PUT":
            return await handle_model_config(envelope, self.services)
        if envelope.path == "/pfe/config/real-local" and envelope.method == "PUT":
            return await handle_real_local_config(envelope, self.services)
        if envelope.path == "/pfe/training/jobs" and envelope.method == "POST":
            return await handle_training_jobs(envelope, self.services)
        if envelope.path == "/pfe/training/jobs" and envelope.method == "GET":
            return await handle_training_jobs_list(envelope, self.services)
        if envelope.path.startswith("/pfe/training/jobs/") and envelope.path.endswith("/cancel") and envelope.method == "POST":
            return await handle_training_job_cancel(envelope, self.services)
        if envelope.path.startswith("/pfe/training/jobs/") and envelope.path.endswith("/retry") and envelope.method == "POST":
            return await handle_training_job_retry(envelope, self.services)
        if envelope.path.startswith("/pfe/training/jobs/") and envelope.path.endswith("/events") and envelope.method == "GET":
            return await handle_training_job_events(envelope, self.services)
        if envelope.path.startswith("/pfe/training/jobs/") and envelope.path.endswith("/checkpoints") and envelope.method == "GET":
            return await handle_training_checkpoints(envelope, self.services)
        if envelope.path.startswith("/pfe/training/jobs/") and envelope.method == "GET":
            return await handle_training_job(envelope, self.services)
        if envelope.path == "/pfe/training/status" and envelope.method == "GET":
            return await handle_training_status(envelope, self.services)
        if envelope.path == "/pfe/training/trigger" and envelope.method == "POST":
            return await handle_training_trigger(envelope, self.services)
        if envelope.path == "/pfe/eval" and envelope.method == "POST":
            return await handle_eval_run(envelope, self.services)
        if envelope.path == "/pfe/eval/status" and envelope.method == "GET":
            return await handle_eval_status(envelope, self.services)
        if envelope.path == "/pfe/adapters" and envelope.method == "GET":
            return await handle_adapters(envelope, self.services)
        if envelope.path == "/pfe/adapters/latest" and envelope.method == "GET":
            return await handle_adapter_latest(envelope, self.services)
        if envelope.path.startswith("/pfe/adapters/") and envelope.path.endswith("/promote") and envelope.method == "POST":
            return await handle_adapter_promote(envelope, self.services)
        if envelope.path.startswith("/pfe/adapters/") and envelope.path.endswith("/rollback") and envelope.method == "POST":
            return await handle_adapter_rollback(envelope, self.services)
        if envelope.path.startswith("/pfe/adapters/") and envelope.path.endswith("/archive") and envelope.method == "POST":
            return await handle_adapter_archive(envelope, self.services)
        if envelope.path.startswith("/pfe/adapters/") and envelope.method == "GET":
            return await handle_adapter_version(envelope, self.services)
        if envelope.path == "/pfe/training/dead-letter" and envelope.method == "GET":
            return await handle_training_dead_letter(envelope, self.services)
        # Dashboard routes
        if envelope.path == "/dashboard" and envelope.method == "GET":
            return await handle_dashboard_frontend(envelope, self.services)
        if envelope.path == "/pfe/dashboard" and envelope.method == "GET":
            return await handle_dashboard_frontend(envelope, self.services)
        if envelope.path == "/pfe/dashboard/metrics" and envelope.method == "GET":
            return await handle_dashboard_metrics(envelope, self.services)
        if envelope.path == "/pfe/dashboard/training" and envelope.method == "GET":
            return await handle_dashboard_training(envelope, self.services)
        if envelope.path == "/pfe/dashboard/signals" and envelope.method == "GET":
            return await handle_dashboard_signals(envelope, self.services)
        if envelope.path == "/pfe/dashboard/adapters" and envelope.method == "GET":
            return await handle_dashboard_adapters(envelope, self.services)
        if envelope.path == "/pfe/dashboard/health" and envelope.method == "GET":
            return await handle_dashboard_health(envelope, self.services)
        return _json_response(_error_payload("not found", "not_found"), status_code=404)


def create_app(
    services: Optional[ServiceBundle] = None,
    *,
    security: Optional[ServerSecurityConfig] = None,
) -> Any:
    bundle = services or _select_default_services()
    if security is not None:
        bundle.security = security

    if FASTAPI_AVAILABLE and FastAPI is not None:
        app = FastAPI(title="PFE Server", version="0.1.0")

        # Add CORS middleware if origins are configured
        cors_origins = bundle.security.cors_origins
        if cors_origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        @app.get("/", response_class=HTMLResponse)
        async def frontend(request: Request) -> Any:
            return await handle_frontend(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/chat", response_class=HTMLResponse)
        async def chat_frontend(request: Request) -> Any:
            return await handle_chat_frontend(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/chat", response_class=HTMLResponse)
        async def pfe_chat_frontend(request: Request) -> Any:
            return await handle_chat_frontend(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/studio", response_class=HTMLResponse)
        async def studio_frontend(request: Request) -> Any:
            return await handle_studio_frontend(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/studio", response_class=HTMLResponse)
        async def pfe_studio_frontend(request: Request) -> Any:
            return await handle_studio_frontend(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Any:
            return await handle_chat_completions(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/signal")
        async def pfe_signal(request: Request) -> Any:
            return await handle_signal_ingest(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/signals")
        async def pfe_signals(request: Request) -> Any:
            return await handle_signals(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/feedback")
        async def pfe_feedback(request: Request) -> Any:
            return await handle_feedback(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/distill/run")
        async def pfe_distill_run(request: Request) -> Any:
            return await handle_distill_run(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/reset")
        async def pfe_auto_train_reset(request: Request) -> Any:
            return await handle_auto_train_reset(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/retry")
        async def pfe_auto_train_retry(request: Request) -> Any:
            return await handle_auto_train_retry(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/process-next")
        async def pfe_auto_train_process_next(request: Request) -> Any:
            return await handle_auto_train_process_next(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/process-batch")
        async def pfe_auto_train_process_batch(request: Request) -> Any:
            return await handle_auto_train_process_batch(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/process-until-idle")
        async def pfe_auto_train_process_until_idle(request: Request) -> Any:
            return await handle_auto_train_process_until_idle(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/run-worker-loop")
        async def pfe_auto_train_run_worker_loop(request: Request) -> Any:
            return await handle_auto_train_run_worker_loop(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/run-worker-runner")
        async def pfe_auto_train_run_worker_runner(request: Request) -> Any:
            return await handle_auto_train_run_worker_runner(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/stop-worker-runner")
        async def pfe_auto_train_stop_worker_runner(request: Request) -> Any:
            return await handle_auto_train_stop_worker_runner(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/auto-train/worker-runner")
        async def pfe_auto_train_worker_runner(request: Request) -> Any:
            return await handle_auto_train_worker_runner_status(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/auto-train/worker-runner/history")
        async def pfe_auto_train_worker_runner_history(request: Request) -> Any:
            return await handle_auto_train_worker_runner_history(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/start-worker-daemon")
        async def pfe_auto_train_start_worker_daemon(request: Request) -> Any:
            return await handle_auto_train_start_worker_daemon(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/stop-worker-daemon")
        async def pfe_auto_train_stop_worker_daemon(request: Request) -> Any:
            return await handle_auto_train_stop_worker_daemon(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/auto-train/worker-daemon")
        async def pfe_auto_train_worker_daemon(request: Request) -> Any:
            return await handle_auto_train_worker_daemon_status(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/auto-train/worker-daemon/history")
        async def pfe_auto_train_worker_daemon_history(request: Request) -> Any:
            return await handle_auto_train_worker_daemon_history(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/recover-worker-daemon")
        async def pfe_auto_train_recover_worker_daemon(request: Request) -> Any:
            return await handle_auto_train_recover_worker_daemon(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/restart-worker-daemon")
        async def pfe_auto_train_restart_worker_daemon(request: Request) -> Any:
            return await handle_auto_train_restart_worker_daemon(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/approve-next")
        async def pfe_auto_train_approve_next(request: Request) -> Any:
            return await handle_auto_train_approve_next(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/auto-train/reject-next")
        async def pfe_auto_train_reject_next(request: Request) -> Any:
            return await handle_auto_train_reject_next(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/candidate/promote")
        async def pfe_candidate_promote(request: Request) -> Any:
            return await handle_candidate_promote(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/candidate/archive")
        async def pfe_candidate_archive(request: Request) -> Any:
            return await handle_candidate_archive(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/candidate/history")
        async def pfe_candidate_history(request: Request) -> Any:
            return await handle_candidate_history(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/candidate/timeline")
        async def pfe_candidate_timeline(request: Request) -> Any:
            return await handle_candidate_timeline(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/auto-train/queue-history")
        async def pfe_auto_train_queue_history(request: Request) -> Any:
            return await handle_train_queue_history(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/status")
        async def pfe_status(request: Request) -> Any:
            return await handle_status(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/runtime")
        async def pfe_runtime(request: Request) -> Any:
            return await handle_runtime(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/workspaces")
        async def pfe_workspaces(request: Request) -> Any:
            return await handle_workspaces(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/workspaces")
        async def pfe_workspaces_create(request: Request) -> Any:
            return await handle_workspaces(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/models")
        async def pfe_models(request: Request) -> Any:
            return await handle_models(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/handoff")
        async def pfe_handoff(request: Request) -> Any:
            return await handle_handoff(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/handoff/test")
        async def pfe_handoff_test(request: Request) -> Any:
            return await handle_handoff_test(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/readiness")
        async def pfe_readiness(request: Request) -> Any:
            return await handle_readiness(await _envelope_from_fastapi_request(request), bundle)

        @app.put("/pfe/config/model")
        async def pfe_config_model(request: Request) -> Any:
            return await handle_model_config(await _envelope_from_fastapi_request(request), bundle)

        @app.put("/pfe/config/real-local")
        async def pfe_config_real_local(request: Request) -> Any:
            return await handle_real_local_config(await _envelope_from_fastapi_request(request), bundle)

        # Dashboard routes
        @app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard_frontend(request: Request) -> Any:
            return await handle_dashboard_frontend(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/dashboard", response_class=HTMLResponse)
        async def pfe_dashboard_frontend(request: Request) -> Any:
            return await handle_dashboard_frontend(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/dashboard/metrics")
        async def pfe_dashboard_metrics(request: Request) -> Any:
            return await handle_dashboard_metrics(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/dashboard/training")
        async def pfe_dashboard_training(request: Request) -> Any:
            return await handle_dashboard_training(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/dashboard/signals")
        async def pfe_dashboard_signals(request: Request) -> Any:
            return await handle_dashboard_signals(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/dashboard/adapters")
        async def pfe_dashboard_adapters(request: Request) -> Any:
            return await handle_dashboard_adapters(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/dashboard/health")
        async def pfe_dashboard_health(request: Request) -> Any:
            return await handle_dashboard_health(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/training/jobs")
        async def pfe_training_jobs(request: Request) -> Any:
            return await handle_training_jobs(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/training/jobs")
        async def pfe_training_jobs_list(request: Request) -> Any:
            return await handle_training_jobs_list(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/training/jobs/{job_id}/events")
        async def pfe_training_job_events(request: Request) -> Any:
            return await handle_training_job_events(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/training/jobs/{job_id}/cancel")
        async def pfe_training_job_cancel(request: Request) -> Any:
            return await handle_training_job_cancel(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/training/jobs/{job_id}/retry")
        async def pfe_training_job_retry(request: Request) -> Any:
            return await handle_training_job_retry(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/training/jobs/{job_id}")
        async def pfe_training_job(request: Request) -> Any:
            return await handle_training_job(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/training/status")
        async def pfe_training_status(request: Request) -> Any:
            return await handle_training_status(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/training/trigger")
        async def pfe_training_trigger(request: Request) -> Any:
            return await handle_training_trigger(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/eval")
        async def pfe_eval_run(request: Request) -> Any:
            return await handle_eval_run(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/eval/status")
        async def pfe_eval_status(request: Request) -> Any:
            return await handle_eval_status(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/adapters/latest")
        async def pfe_adapter_latest(request: Request) -> Any:
            return await handle_adapter_latest(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/adapters")
        async def pfe_adapters(request: Request) -> Any:
            return await handle_adapters(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/adapters/{version}/promote")
        async def pfe_adapter_promote(request: Request) -> Any:
            return await handle_adapter_promote(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/adapters/{version}/rollback")
        async def pfe_adapter_rollback(request: Request) -> Any:
            return await handle_adapter_rollback(await _envelope_from_fastapi_request(request), bundle)

        @app.post("/pfe/adapters/{version}/archive")
        async def pfe_adapter_archive(request: Request) -> Any:
            return await handle_adapter_archive(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/adapters/{version}")
        async def pfe_adapter_version(request: Request) -> Any:
            return await handle_adapter_version(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/training/jobs/{job_id}/checkpoints")
        async def pfe_training_checkpoints(request: Request) -> Any:
            return await handle_training_checkpoints(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/training/dead-letter")
        async def pfe_training_dead_letter(request: Request) -> Any:
            return await handle_training_dead_letter(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/observability/trace/{signal_id}")
        async def pfe_observability_trace(request: Request) -> Any:
            return await handle_observability_trace(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/observability/version/{version}")
        async def pfe_observability_version(request: Request) -> Any:
            return await handle_observability_version(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/audit/pii")
        async def pfe_audit_pii(request: Request) -> Any:
            return await handle_audit_pii(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/pfe/audit/training")
        async def pfe_audit_training(request: Request) -> Any:
            return await handle_audit_training(await _envelope_from_fastapi_request(request), bundle)

        @app.get("/healthz")
        async def healthz() -> dict[str, str]:
            return {"status": "ok"}

        app.state.pfe_services = bundle
        return app

    return _LiteASGIApp(bundle)


def _build_serve_plan(
    *,
    port: int = 8921,
    host: str = "127.0.0.1",
    adapter: str = "latest",
    api_key: Optional[str] = None,
    allow_remote_access: bool = False,
    cors_origins: Optional[list[str]] = None,
    workspace: Optional[str] = None,
    dry_run: bool = True,
) -> ServePlan:
    del adapter
    bundle = _select_default_services()
    if workspace:
        bundle.workspace = workspace
    if api_key:
        os.environ[bundle.security.api_key_env] = api_key
    bundle.security.allow_remote_access = allow_remote_access
    if cors_origins:
        bundle.security.cors_origins = cors_origins
    app_obj = create_app(bundle)
    try:
        import uvicorn  # type: ignore

        uvicorn_available = True
        uvicorn_module = getattr(uvicorn, "__name__", "uvicorn")
    except Exception:
        uvicorn_available = False
        uvicorn_module = None
    app_target = "pfe_server.app:app"
    command = [sys.executable, "-m", "uvicorn", app_target, "--host", host, "--port", str(port)]
    runner = {
        "kind": "uvicorn.run" if uvicorn_available else "dry_run",
        "target": app_target,
        "kwargs": {
            "host": host,
            "port": port,
            "reload": False,
            "factory": False,
        },
    }
    if not uvicorn_available:
        runner["reason"] = "uvicorn is not installed in this environment."
    runtime = ServeRuntimeInfo(
        provider=bundle.provider,
        workspace=bundle.workspace,
        host=host,
        port=port,
        dry_run=dry_run,
        uvicorn_available=uvicorn_available,
        app_target=app_target,
        app_type=type(app_obj).__name__,
        command=command,
        runner=runner,
        notes=[
            "default serve() remains dry-run friendly for tests",
            "use dry_run=False to launch uvicorn when available",
        ],
    )
    runtime_probe = {
        "app_target": app_target,
        "app_type": type(app_obj).__name__,
        "dry_run": dry_run,
        "uvicorn_available": uvicorn_available,
        "uvicorn_module": uvicorn_module,
        "launch_mode": "uvicorn.run" if uvicorn_available and not dry_run else "dry_run",
        "command": command,
        "runner": runner,
        "probe_paths": [
            {"method": "GET", "path": "/healthz"},
            {"method": "GET", "path": "/pfe/status"},
            {"method": "POST", "path": "/pfe/signal"},
            {"method": "POST", "path": "/pfe/distill/run"},
        ],
    }
    bundle.runtime_probe = runtime_probe
    if dry_run:
        probe_status = {
            "state": "skipped",
            "reason": "dry_run=True",
            "checks": [],
            "summary": {"checked_paths": [], "healthy_checks": 0, "total_checks": 0},
        }
        last_serve_check: dict[str, Any] = {}
    else:
        try:
            probe_status = asyncio.run(_collect_runtime_probe_checks(app_obj, dry_run=False))
        except RuntimeError:
            probe_status = {
                "state": "deferred",
                "reason": "event loop already running; probe not executed",
                "checks": [],
                "summary": {"checked_paths": [], "healthy_checks": 0, "total_checks": 0},
            }
        last_serve_check = dict(probe_status.get("last_serve_check") or {})
    launch_state = {
        "before": {
            "dry_run": dry_run,
            "uvicorn_available": uvicorn_available,
            "launch_mode": "uvicorn.run" if uvicorn_available and not dry_run else "dry_run",
        },
        "after": {
            "probe_state": probe_status.get("state"),
            "last_serve_check": last_serve_check,
        },
    }
    serve_summary = {
        "launch_mode": "uvicorn.run" if uvicorn_available and not dry_run else "dry_run",
        "probe_state": probe_status.get("state"),
        "probe_status": dict(probe_status),
        "probe_summary": dict(probe_status.get("summary") or {}),
        "checked_paths": [item.get("path") for item in probe_status.get("checks", [])],
    }
    runtime_probe.update(
        {
            "launch_state": launch_state,
            "serve_summary": serve_summary,
            "probe_status": probe_status,
            "last_serve_check": last_serve_check,
        }
    )
    bundle.runtime_probe = runtime_probe
    return ServePlan(
        runtime=runtime,
        app=app_obj,
        uvicorn_module=uvicorn_module,
        uvicorn_kwargs=runner["kwargs"],
        runtime_probe=runtime_probe,
    )


def build_serve_plan(
    *,
    port: int = 8921,
    host: str = "127.0.0.1",
    adapter: str = "latest",
    api_key: Optional[str] = None,
    allow_remote_access: bool = False,
    cors_origins: Optional[list[str]] = None,
    workspace: Optional[str] = None,
    dry_run: bool = True,
) -> ServePlan:
    return _build_serve_plan(
        port=port,
        host=host,
        adapter=adapter,
        api_key=api_key,
        allow_remote_access=allow_remote_access,
        cors_origins=cors_origins,
        workspace=workspace,
        dry_run=dry_run,
    )


def serve(
    *,
    port: int = 8921,
    host: str = "127.0.0.1",
    adapter: str = "latest",
    api_key: Optional[str] = None,
    allow_remote_access: bool = False,
    cors_origins: Optional[list[str]] = None,
    workspace: Optional[str] = None,
    dry_run: bool = True,
) -> str:
    plan = build_serve_plan(
        port=port,
        host=host,
        adapter=adapter,
        api_key=api_key,
        allow_remote_access=allow_remote_access,
        cors_origins=cors_origins,
        workspace=workspace,
        dry_run=dry_run,
    )
    if not dry_run and plan.runtime.uvicorn_available:
        import uvicorn  # type: ignore

        uvicorn.run(plan.app, host=host, port=port, reload=False)
        return f"started uvicorn on {host}:{port}"
    return (
        f"PFE server plan ready on {plan.runtime.host}:{plan.runtime.port} "
        f"(provider={plan.runtime.provider}, app={plan.runtime.app_type}, "
        f"uvicorn_available={plan.runtime.uvicorn_available}, dry_run={plan.runtime.dry_run}). "
        f"command={' '.join(plan.command)}"
    )


async def smoke_test_request(
    app_obj: Any,
    path: str = "/healthz",
    *,
    method: str = "GET",
    body: Optional[Union[dict[str, Any], list[Any], str, bytes]] = None,
    headers: Optional[Mapping[str, str]] = None,
    query_params: Optional[Mapping[str, Any]] = None,
    client_host: str = "127.0.0.1",
) -> dict[str, Any]:
    if body is None:
        payload = b""
    elif isinstance(body, bytes):
        payload = body
    elif isinstance(body, str):
        payload = body.encode("utf-8")
    else:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    header_items = [(str(key).encode("latin1"), str(value).encode("latin1")) for key, value in (headers or {}).items()]
    if payload and not any(key.lower() == b"content-type" for key, _ in header_items):
        header_items.append((b"content-type", b"application/json"))
    query_string = b""
    if query_params:
        from urllib.parse import urlencode

        query_string = urlencode([(str(key), str(value)) for key, value in query_params.items()]).encode("latin1")
    scope = {
        "type": "http",
        "method": method.upper(),
        "path": path,
        "headers": header_items,
        "client": (client_host, 0),
        "query_string": query_string,
    }
    state: dict[str, Any] = {"payload": payload, "sent": []}

    async def receive() -> dict[str, Any]:
        if state["payload"] is not None:
            chunk = state["payload"]
            state["payload"] = None
            return {"type": "http.request", "body": chunk, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        state["sent"].append(message)

    await app_obj(scope, receive, send)
    status = 500
    response_headers: dict[str, str] = {}
    response_body: Any = {}
    response_text = ""
    for message in state["sent"]:
        if message.get("type") == "http.response.start":
            status = int(message.get("status", 500))
            raw_headers = message.get("headers", [])
            response_headers = {
                key.decode("latin1").lower(): value.decode("latin1")
                for key, value in raw_headers
            }
        elif message.get("type") == "http.response.body":
            raw = message.get("body", b"{}")
            if isinstance(raw, bytes):
                response_text = raw.decode("utf-8", errors="replace")
                try:
                    response_body = json.loads(response_text)
                except Exception:
                    response_body = {"raw": response_text}
    return {
        "status_code": status,
        "headers": response_headers,
        "body": response_body,
        "text": response_text,
        "raw_body": response_text,
        "request": {
            "method": method.upper(),
            "path": path,
            "query_params": dict(query_params or {}),
            "client_host": client_host,
        },
    }


app = create_app()
