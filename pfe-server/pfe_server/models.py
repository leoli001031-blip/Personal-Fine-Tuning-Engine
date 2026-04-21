from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pfe_core.models import normalize_utc_datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_timestamp() -> int:
    return int(_utc_now().timestamp())


class PFEBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, protected_namespaces=())


class ChatMessage(PFEBaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(PFEBaseModel):
    model: str = "local-default"
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    adapter_version: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatCompletionResponseMessage(PFEBaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(PFEBaseModel):
    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "error"] = "stop"


class ChatCompletionUsage(PFEBaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(PFEBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=_utc_timestamp)
    model: str = "local-default"
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    adapter_version: Optional[str] = None
    served_by: Literal["local", "mock", "cloud"] = "mock"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalIngestRequest(PFEBaseModel):
    event_id: str
    request_id: str
    session_id: str
    adapter_version: Optional[str] = None
    source_event_id: Optional[str] = None
    source_event_ids: list[str] = Field(default_factory=list)
    event_type: Literal["chat", "edit", "copy", "interrupt", "accept", "reject", "regenerate"]
    timestamp: datetime = Field(default_factory=_utc_now)
    user_input: Optional[str] = None
    model_output: Optional[str] = None
    user_action: dict[str, Any] = Field(default_factory=dict)
    action_detail: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", mode="after")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value)


class SignalIngestResponse(PFEBaseModel):
    signal_id: str = Field(default_factory=lambda: f"signal-{uuid4().hex}")
    status: Literal["accepted", "duplicate", "ignored"] = "accepted"
    stored: bool = True
    request_id: str
    session_id: str
    adapter_version: Optional[str] = None
    source_event_id: Optional[str] = None
    source_event_ids: list[str] = Field(default_factory=list)
    event_chain_complete: bool = False
    curated_samples: int = 0
    curated_sample_ids: list[str] = Field(default_factory=list)
    dataset_split_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DistillRunRequest(PFEBaseModel):
    scenario: str = "life-coach"
    style: str = "warm"
    num_samples: int = 200
    teacher_model: str = "mock-teacher"
    teacher_prompt_version: str = "v1"
    generation_config: dict[str, Any] = Field(default_factory=dict)
    use_cloud_teacher: bool = False
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    source_event_ids: list[str] = Field(default_factory=list)
    seed_text: Optional[str] = None


class DistillRunResponse(PFEBaseModel):
    run_id: str = Field(default_factory=lambda: f"distill-{uuid4().hex}")
    status: Literal["queued", "completed", "rejected"] = "completed"
    teacher_model: str
    teacher_prompt_version: str
    requested_samples: int
    generated_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    dataset_split: Literal["train", "val", "test", "mixed"] = "mixed"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AutoTrainTriggerStatus(PFEBaseModel):
    enabled: bool = False
    state: Literal["idle", "disabled", "blocked", "ready", "triggered"] = "idle"
    ready: bool = False
    reason: str = "idle"
    blocked_reasons: list[str] = Field(default_factory=list)
    train_type: Optional[str] = None
    method: Optional[str] = None
    epochs: Optional[int] = None
    base_model: Optional[str] = None
    min_new_samples: Optional[int] = None
    max_interval_days: Optional[int] = None
    min_trigger_interval_minutes: Optional[int] = None
    failure_backoff_minutes: Optional[int] = None
    queue_dedup_scope: Optional[str] = None
    queue_priority_policy: Optional[str] = None
    queue_process_batch_size: Optional[int] = None
    queue_process_until_idle_max: Optional[int] = None
    queue_worker_max_cycles: Optional[int] = None
    queue_worker_idle_rounds: Optional[int] = None
    queue_worker_poll_seconds: Optional[float] = None
    require_queue_confirmation: bool = False
    preference_reinforced_sample_weight: Optional[float] = None
    effective_eligible_train_samples: float = 0.0
    preference_reinforced_train_samples: int = 0
    eligible_signal_train_samples: int = 0
    effective_signal_train_samples: float = 0.0
    preference_reinforced_signal_train_samples: int = 0
    eligible_signal_sample_ids: list[str] = Field(default_factory=list)
    effective_dpo_preference_pairs: float = 0.0
    preference_reinforced_dpo_preference_pairs: int = 0
    holdout_ready: bool = False
    days_since_last_training: Optional[float] = None
    interval_elapsed: bool = False
    cooldown_elapsed: bool = True
    cooldown_remaining_minutes: Optional[float] = None
    failure_backoff_elapsed: bool = True
    failure_backoff_remaining_minutes: Optional[float] = None
    last_attempted_at: Optional[str] = None
    last_completed_at: Optional[str] = None
    last_success_at: Optional[str] = None
    last_failure_at: Optional[str] = None
    consecutive_failures: int = 0
    recent_training_version: Optional[str] = None
    triggered: bool = False
    last_result_summary: Optional[str] = None
    last_result: dict[str, Any] = Field(default_factory=dict)


class AutoTrainTriggerActionResponse(PFEBaseModel):
    action: str = "retry"
    status: str = "blocked"
    reason: Optional[str] = None
    state_path: Optional[str] = None
    triggered: bool = False
    queue_job_id: Optional[str] = None
    queue_job_ids: list[str] = Field(default_factory=list)
    confirmation_reason: Optional[str] = None
    approval_reason: Optional[str] = None
    rejection_reason: Optional[str] = None
    operator_note: Optional[str] = None
    processed_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    limit: Optional[int] = None
    max_iterations: Optional[int] = None
    max_cycles: Optional[int] = None
    max_seconds: Optional[float] = None
    loop_cycles: Optional[int] = None
    idle_rounds: Optional[int] = None
    poll_interval_seconds: Optional[float] = None
    remaining_queued: int = 0
    drained: bool = False
    stopped_reason: Optional[str] = None
    triggered_version: Optional[str] = None
    promoted_version: Optional[str] = None
    auto_train_trigger: AutoTrainTriggerStatus = Field(default_factory=AutoTrainTriggerStatus)
    review_summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateActionResponse(PFEBaseModel):
    action: str = "promote_candidate"
    status: str = "noop"
    reason: Optional[str] = None
    state_path: Optional[str] = None
    triggered: bool = False
    candidate_version: Optional[str] = None
    promoted_version: Optional[str] = None
    archived_version: Optional[str] = None
    operator_note: Optional[str] = None
    candidate_summary: dict[str, Any] = Field(default_factory=dict)
    candidate_history: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PFEStatusResponse(PFEBaseModel):
    service: str = "pfe-server"
    provider: str = "mock"
    mode: Literal["strict_local", "cloud_assisted"] = "strict_local"
    strict_local: bool = True
    allow_remote_access: bool = False
    healthy: bool = True
    uptime_seconds: float = 0.0
    inference_backend: str = "mock"
    pipeline_backend: str = "mock"
    adapter_count: int = 0
    signal_count: int = 0
    distill_run_count: int = 0
    sample_counts: dict[str, int] = Field(default_factory=dict)
    signal_summary: dict[str, Any] = Field(default_factory=dict)
    signal_sample_count: int = 0
    signal_sample_counts: dict[str, int] = Field(default_factory=dict)
    signal_sample_details: list[dict[str, Any]] = Field(default_factory=list)
    auto_train_trigger: AutoTrainTriggerStatus = Field(default_factory=AutoTrainTriggerStatus)
    auto_train_trigger_action: AutoTrainTriggerActionResponse = Field(default_factory=AutoTrainTriggerActionResponse)
    candidate_summary: dict[str, Any] = Field(default_factory=dict)
    candidate_action: CandidateActionResponse = Field(default_factory=CandidateActionResponse)
    candidate_history: dict[str, Any] = Field(default_factory=dict)
    candidate_timeline: dict[str, Any] = Field(default_factory=dict)
    train_queue: dict[str, Any] = Field(default_factory=dict)
    operations_overview: dict[str, Any] = Field(default_factory=dict)
    operations_console: dict[str, Any] = Field(default_factory=dict)
    operations_alerts: list[dict[str, Any]] = Field(default_factory=list)
    operations_health: dict[str, Any] = Field(default_factory=dict)
    operations_recovery: dict[str, Any] = Field(default_factory=dict)
    operations_next_actions: list[str] = Field(default_factory=list)
    operations_dashboard: dict[str, Any] = Field(default_factory=dict)
    operations_alert_policy: dict[str, Any] = Field(default_factory=dict)
    operations_event_stream: dict[str, Any] = Field(default_factory=dict)
    operations_timeline: dict[str, Any] = Field(default_factory=dict)
    runner_timeline: dict[str, Any] = Field(default_factory=dict)
    operations_runner_timeline: dict[str, Any] = Field(default_factory=dict)
    daemon_timeline: dict[str, Any] = Field(default_factory=dict)
    operations_daemon_timeline: dict[str, Any] = Field(default_factory=dict)
    latest_adapter: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ServeRuntimeInfo(PFEBaseModel):
    service: str = "pfe-server"
    provider: str = "mock"
    workspace: str = "user_default"
    host: str = "127.0.0.1"
    port: int = 8921
    dry_run: bool = True
    uvicorn_available: bool = False
    app_target: str = "pfe_server.app:app"
    app_type: str = "mock"
    command: list[str] = Field(default_factory=list)
    runner: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class FeedbackRequest(PFEBaseModel):
    """Implicit feedback from user interaction.

    This model captures user behavior signals that indicate
    acceptance/rejection of assistant responses without
    requiring explicit feedback buttons.
    """
    session_id: str
    request_id: str
    action: Literal["accept", "reject", "edit", "regenerate", "delete"]
    response_time_seconds: Optional[float] = None
    edited_text: Optional[str] = None
    next_message: Optional[str] = None
    adapter_version: Optional[str] = None
    user_message: Optional[str] = None
    assistant_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(PFEBaseModel):
    """Response to feedback submission."""
    success: bool = True
    signal_id: Optional[str] = None
    signal_type: Optional[str] = None
    confidence: float = 0.0
    message: str = "Feedback recorded"
    session_id: str
    request_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(PFEBaseModel):
    detail: str
    code: str = "pfe_error"
    hint: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
