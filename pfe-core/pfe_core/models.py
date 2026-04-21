"""Core data models for PFE."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .adapter_store.lifecycle import AdapterArtifactFormat, AdapterLifecycleState


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_utc_datetime(value: datetime | str | None, *, default: datetime | None = None) -> datetime:
    if value is None:
        return normalize_utc_datetime(default or utc_now())
    if isinstance(value, datetime):
        return normalize_utc_datetime(value)
    return normalize_utc_datetime(datetime.fromisoformat(str(value).replace("Z", "+00:00")))


def new_id() -> str:
    return uuid4().hex


class TaskState(str, Enum):
    """Task lifecycle states for queue items."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALLED = "stalled"  # Detected as stale/frozen
    RECOVERING = "recovering"  # In recovery process


class RunnerState(str, Enum):
    """Runner health states."""

    IDLE = "idle"
    HEALTHY = "healthy"
    DELAYED = "delayed"  # Heartbeat delayed but within lease
    STALE = "stale"  # Lease expired
    FAILED = "failed"  # Explicitly marked as failed
    RECOVERING = "recovering"


class LeaseState(str, Enum):
    """Lease lifecycle states."""

    IDLE = "idle"
    VALID = "valid"
    EXPIRING = "expiring"  # Within warning threshold
    EXPIRED = "expired"
    RELEASED = "released"


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    ATTENTION = "attention"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryAction(str, Enum):
    """Recovery actions for failed tasks."""

    NONE = "none"
    RETRY = "retry"
    RESTART = "restart"
    RESUME = "resume"  # Resume from checkpoint
    SKIP = "skip"
    DEAD_LETTER = "dead_letter"


class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid", protected_namespaces=())


@dataclass
class RunnerHeartbeat:
    """Heartbeat information from runner to daemon.

    Used to track runner health and detect stale/failed runners.
    """

    runner_id: str = field(default_factory=new_id)
    job_id: Optional[str] = None
    workspace: str = "user_default"
    pid: Optional[int] = None
    timestamp: datetime = field(default_factory=utc_now)
    sequence_number: int = 0  # Monotonically increasing sequence
    state: RunnerState = RunnerState.HEALTHY
    progress_percent: float = 0.0
    current_step: str = ""  # e.g., "loading_model", "training_epoch_1"
    step_details: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)  # Live training metrics
    memory_usage_mb: Optional[float] = None
    cpu_percent: Optional[float] = None

    def is_fresh(self, max_age_seconds: float = 30.0) -> bool:
        """Check if heartbeat is fresh within given age threshold."""
        age = (utc_now() - self.timestamp).total_seconds()
        return age <= max_age_seconds

    def is_stale(self, lease_timeout_seconds: float = 60.0) -> bool:
        """Check if heartbeat has gone stale (lease expired)."""
        age = (utc_now() - self.timestamp).total_seconds()
        return age > lease_timeout_seconds


@dataclass
class TaskLease:
    """Lease for task execution to prevent multiple workers competing.

    When a worker picks up a task, it acquires a lease with a timeout.
    The lease must be renewed periodically during task execution.
    """

    lease_id: str = field(default_factory=new_id)
    job_id: str = ""
    runner_id: str = ""
    workspace: str = "user_default"
    acquired_at: datetime = field(default_factory=utc_now)
    expires_at: datetime = field(default_factory=lambda: utc_now() + timedelta(seconds=60))
    renewed_at: Optional[datetime] = None
    renewal_count: int = 0
    state: LeaseState = LeaseState.VALID
    # Configurable timeouts (can be overridden per task)
    lease_timeout_seconds: float = 60.0
    warning_threshold_seconds: float = 45.0  # When to start warning about expiration

    def is_valid(self) -> bool:
        """Check if lease is still valid (not expired)."""
        if self.state in {LeaseState.EXPIRED, LeaseState.RELEASED}:
            return False
        return utc_now() < self.expires_at

    def is_expiring_soon(self) -> bool:
        """Check if lease is expiring within warning threshold."""
        if self.state in {LeaseState.EXPIRED, LeaseState.RELEASED}:
            return False
        remaining = (self.expires_at - utc_now()).total_seconds()
        return remaining <= self.warning_threshold_seconds

    def renew(self, extension_seconds: Optional[float] = None) -> None:
        """Renew the lease for another period."""
        extension = extension_seconds or self.lease_timeout_seconds
        self.expires_at = utc_now() + timedelta(seconds=extension)
        self.renewed_at = utc_now()
        self.renewal_count += 1
        self.state = LeaseState.VALID

    def release(self) -> None:
        """Release the lease (task completed or failed)."""
        self.state = LeaseState.RELEASED
        self.expires_at = utc_now()

    def mark_expired(self) -> None:
        """Mark lease as expired."""
        self.state = LeaseState.EXPIRED


@dataclass
class RecoveryCheckpoint:
    """Checkpoint for training recovery after interruption.

    Saves training state periodically to enable resume from interruption.
    """

    checkpoint_id: str = field(default_factory=new_id)
    job_id: str = ""
    workspace: str = "user_default"
    created_at: datetime = field(default_factory=utc_now)
    # Training progress
    epoch: int = 0
    global_step: int = 0
    samples_processed: int = 0
    # Model state
    model_state_path: Optional[str] = None  # Path to model checkpoint
    optimizer_state_path: Optional[str] = None
    scheduler_state_path: Optional[str] = None
    # Training configuration snapshot
    training_config: dict[str, Any] = field(default_factory=dict)
    # Metrics at checkpoint time
    metrics: dict[str, Any] = field(default_factory=dict)
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    # Recovery metadata
    is_resumable: bool = True
    resume_attempt_count: int = 0
    last_resume_at: Optional[datetime] = None

    def can_resume(self, max_resume_attempts: int = 3) -> bool:
        """Check if checkpoint can be used for recovery."""
        if not self.is_resumable:
            return False
        if self.resume_attempt_count >= max_resume_attempts:
            return False
        return self.model_state_path is not None


@dataclass
class DeadLetterEntry:
    """Entry for permanently failed tasks (dead letter queue).

    Tasks that exceed max retries or fail permanently are moved here
    for manual inspection.
    """

    entry_id: str = field(default_factory=new_id)
    job_id: str = ""
    workspace: str = "user_default"
    failed_at: datetime = field(default_factory=utc_now)
    # Failure details
    failure_reason: str = ""
    failure_category: str = ""  # e.g., "oom", "crash", "timeout", "logic_error"
    error_details: dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    # Retry history
    retry_count: int = 0
    retry_history: list[dict[str, Any]] = field(default_factory=list)
    # Original task details
    original_task: dict[str, Any] = field(default_factory=dict)
    last_checkpoint: Optional[RecoveryCheckpoint] = None
    # Manual resolution
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_action: Optional[str] = None  # "retry", "discard", "manual_fix"
    resolution_note: Optional[str] = None


@dataclass
class RestartPolicy:
    """Restart policy configuration for daemon/runner recovery.

    Implements exponential backoff for consecutive failures.
    """

    max_restart_attempts: int = 3
    base_backoff_seconds: float = 15.0
    max_backoff_seconds: float = 300.0  # 5 minutes max
    backoff_multiplier: float = 2.0
    reset_after_seconds: float = 3600.0  # Reset counter after 1 hour of success

    # Current state (persisted)
    current_attempt: int = 0
    last_failure_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    next_restart_after: Optional[datetime] = None

    def calculate_backoff(self) -> float:
        """Calculate current backoff delay using exponential backoff."""
        if self.current_attempt == 0:
            return 0.0
        delay = self.base_backoff_seconds * (self.backoff_multiplier ** (self.current_attempt - 1))
        return min(delay, self.max_backoff_seconds)

    def should_restart(self) -> tuple[bool, Optional[float]]:
        """Check if restart is allowed and return backoff delay if any."""
        # Check if we should reset the counter due to time elapsed
        if self.last_success_at is not None:
            elapsed = (utc_now() - self.last_success_at).total_seconds()
            if elapsed >= self.reset_after_seconds:
                self.current_attempt = 0
                return True, 0.0

        if self.current_attempt >= self.max_restart_attempts:
            return False, None

        # Check if we're still in backoff period
        if self.next_restart_after is not None:
            remaining = (self.next_restart_after - utc_now()).total_seconds()
            if remaining > 0:
                return False, remaining

        return True, 0.0

    def record_failure(self) -> float:
        """Record a failure and return the backoff delay."""
        self.current_attempt += 1
        self.last_failure_at = utc_now()
        delay = self.calculate_backoff()
        if delay > 0:
            self.next_restart_after = utc_now() + timedelta(seconds=delay)
        return delay

    def record_success(self) -> None:
        """Record a successful execution."""
        self.last_success_at = utc_now()
        # Don't reset immediately, let should_restart handle the logic

    def reset(self) -> None:
        """Manually reset the policy state."""
        self.current_attempt = 0
        self.last_failure_at = None
        self.next_restart_after = None


@dataclass
class AlertThreshold:
    """Alert thresholds for operations monitoring.

    Defines when to trigger alerts based on various metrics.
    """

    # Consecutive failure thresholds
    consecutive_failures_warning: int = 2
    consecutive_failures_error: int = 3
    consecutive_failures_critical: int = 5

    # Heartbeat thresholds (seconds)
    heartbeat_delay_warning: float = 10.0  # Heartbeat delayed by 10s
    heartbeat_delay_error: float = 30.0
    lease_expiry_warning: float = 15.0  # Lease expiring in 15s

    # Task stall detection (seconds)
    task_stall_warning: float = 300.0  # 5 minutes
    task_stall_error: float = 600.0  # 10 minutes
    task_stall_critical: float = 1800.0  # 30 minutes

    # Recovery thresholds
    max_recovery_attempts_warning: int = 2
    dead_letter_queue_size_warning: int = 5
    dead_letter_queue_size_critical: int = 10

    def check_consecutive_failures(self, count: int) -> AlertLevel:
        """Determine alert level for consecutive failures."""
        if count >= self.consecutive_failures_critical:
            return AlertLevel.CRITICAL
        if count >= self.consecutive_failures_error:
            return AlertLevel.ERROR
        if count >= self.consecutive_failures_warning:
            return AlertLevel.WARNING
        return AlertLevel.INFO

    def check_heartbeat_delay(self, delay_seconds: float) -> AlertLevel:
        """Determine alert level for heartbeat delay."""
        if delay_seconds >= self.heartbeat_delay_error:
            return AlertLevel.ERROR
        if delay_seconds >= self.heartbeat_delay_warning:
            return AlertLevel.WARNING
        return AlertLevel.INFO

    def check_task_stall(self, stall_seconds: float) -> AlertLevel:
        """Determine alert level for task stall."""
        if stall_seconds >= self.task_stall_critical:
            return AlertLevel.CRITICAL
        if stall_seconds >= self.task_stall_error:
            return AlertLevel.ERROR
        if stall_seconds >= self.task_stall_warning:
            return AlertLevel.WARNING
        return AlertLevel.INFO


@dataclass
class AlertEvent:
    """Alert event for operations console."""

    alert_id: str = field(default_factory=new_id)
    timestamp: datetime = field(default_factory=utc_now)
    level: AlertLevel = AlertLevel.INFO
    scope: str = ""  # "daemon", "runner", "task", "queue", "system"
    reason: str = ""
    message: str = ""
    job_id: Optional[str] = None
    runner_id: Optional[str] = None
    workspace: str = "user_default"
    # Context
    context: dict[str, Any] = field(default_factory=dict)
    # Resolution
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_note: Optional[str] = None

    def acknowledge(self, by: str) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_at = utc_now()
        self.acknowledged_by = by

    def resolve(self, note: Optional[str] = None) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = utc_now()
        if note:
            self.resolution_note = note


@dataclass
class TaskExecutionMetadata:
    """Extended metadata for task execution tracking."""

    job_id: str = ""
    workspace: str = "user_default"
    # Execution tracking
    created_at: datetime = field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    # State tracking
    current_state: TaskState = TaskState.PENDING
    previous_state: Optional[TaskState] = None
    state_history: list[dict[str, Any]] = field(default_factory=list)
    # Runner assignment
    assigned_runner_id: Optional[str] = None
    lease_id: Optional[str] = None
    # Recovery tracking
    retry_count: int = 0
    max_retries: int = 3
    recovery_checkpoint_id: Optional[str] = None
    # Failure tracking
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure_reason: Optional[str] = None
    last_error_details: dict[str, Any] = field(default_factory=dict)
    # Performance
    execution_time_seconds: Optional[float] = None
    total_cpu_time_seconds: Optional[float] = None
    peak_memory_mb: Optional[float] = None

    def transition_to(self, new_state: TaskState, reason: Optional[str] = None) -> None:
        """Record state transition."""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_history.append({
            "from": self.previous_state.value if self.previous_state else None,
            "to": new_state.value,
            "at": utc_now().isoformat(),
            "reason": reason,
        })

    def record_failure(self, reason: str, details: Optional[dict[str, Any]] = None) -> None:
        """Record a failure."""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_reason = reason
        if details:
            self.last_error_details = details

    def record_success(self) -> None:
        """Record successful completion."""
        self.consecutive_failures = 0
        self.completed_at = utc_now()
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.retry_count < self.max_retries and self.consecutive_failures < self.max_retries

    def can_recover(self) -> bool:
        """Check if task can be recovered from checkpoint."""
        return self.recovery_checkpoint_id is not None and self.retry_count < self.max_retries


@dataclass
class ChatInteraction:
    """A single turn in a conversation between user and assistant.

    This represents the raw interaction data that can be analyzed to extract
    implicit signals about user satisfaction with the assistant's response.
    """

    session_id: str
    """Unique identifier for the conversation session."""

    request_id: str
    """Unique identifier for this specific request/response pair."""

    user_message: str
    """The message sent by the user."""

    assistant_message: str
    """The response generated by the assistant."""

    timestamp: datetime = field(default_factory=utc_now)
    """When the interaction occurred."""

    adapter_version: Optional[str] = None
    """Version of the adapter/model used for generation."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the interaction."""

    event_id: str = field(default_factory=new_id)
    """Unique identifier for this interaction event."""

    response_time_seconds: Optional[float] = None
    """Time taken for the user to respond (if available)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "adapter_version": self.adapter_version,
            "metadata": self.metadata,
            "response_time_seconds": self.response_time_seconds,
        }


@dataclass
class InteractionEvent:
    event_id: str = field(default_factory=new_id)
    request_id: str = ""
    source_event_ids: list[str] = field(default_factory=list)
    event_chain_ids: list[str] = field(default_factory=list)
    parent_event_id: Optional[str] = None
    event_type: Literal["chat", "edit", "copy", "interrupt", "regenerate"] = "chat"
    timestamp: datetime = field(default_factory=utc_now)
    session_id: str = ""
    adapter_version: Optional[str] = None
    scenario: Optional[str] = None
    user_input: str = ""
    model_output: str = ""
    user_action: Literal["accept", "reject", "edit", "copy", "interrupt", "regenerate"] = "accept"
    action_detail: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    signal_quality: "SignalQuality" = field(default_factory=lambda: SignalQuality())


@dataclass
class SignalQuality:
    reply_style: Literal["accepted", "rejected", "edited", "other"] = "other"
    confidence: float = 0.0
    conflict: bool = False
    conflict_reason: Optional[str] = None
    confidence_reason: Optional[str] = None
    provenance: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    rolled_back: bool = False
    replay_eligible: bool = False


@dataclass
class RawSignal:
    signal_id: str = field(default_factory=new_id)
    source_event_id: str = ""
    request_id: str = ""
    session_id: str = ""
    parent_event_id: Optional[str] = None
    adapter_version: Optional[str] = None
    event_type: Literal["accept", "reject", "edit", "copy", "interrupt", "regenerate"] = "accept"
    timestamp: datetime = field(default_factory=utc_now)
    context: str = ""
    model_output: str = ""
    user_action: dict[str, Any] = field(default_factory=dict)
    source_event_ids: list[str] = field(default_factory=list)
    event_chain_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    signal_quality: "SignalQuality" = field(default_factory=lambda: SignalQuality())


@dataclass
class ImplicitSignal(RawSignal):
    """An implicit feedback signal extracted from user behavior.

    Unlike explicit feedback where users rate or label responses, implicit
    signals are inferred from actions like editing, deleting, or quickly
    continuing the conversation.
    """

    signal_type: Literal["accept", "reject", "edit", "regenerate"] = "accept"
    """The type of implicit signal detected."""

    confidence: float = 0.0
    """Confidence score for this signal (0-1)."""

    edit_distance: Optional[int] = None
    """Levenshtein distance between original and edited text (for edit signals)."""

    edit_distance_ratio: Optional[float] = None
    """Edit distance as a ratio of the original text length."""

    extraction_rule: str = ""
    """Name of the rule used to extract this signal."""

    response_time_seconds: Optional[float] = None
    """Time between assistant response and user action."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation suitable for storage."""
        return {
            "event_id": self.signal_id,
            "source_event_id": self.source_event_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "parent_event_id": self.parent_event_id,
            "adapter_version": self.adapter_version,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "context": self.context,
            "model_output": self.model_output,
            "user_action": self.user_action,
            "source_event_ids": self.source_event_ids,
            "event_chain_ids": self.event_chain_ids,
            "metadata": {
                **self.metadata,
                "signal_type": self.signal_type,
                "confidence": self.confidence,
                "edit_distance": self.edit_distance,
                "edit_distance_ratio": self.edit_distance_ratio,
                "extraction_rule": self.extraction_rule,
                "response_time_seconds": self.response_time_seconds,
            },
            "signal_quality": {
                "reply_style": self._reply_style_from_signal_type(),
                "confidence": self.confidence,
                "confidence_reason": f"Extracted via {self.extraction_rule}",
                "provenance": {
                    "source": "implicit_extraction",
                    "rule": self.extraction_rule,
                },
            },
        }

    def _reply_style_from_signal_type(self) -> Literal["accepted", "rejected", "edited", "other"]:
        """Map signal type to reply style."""
        mapping = {
            "accept": "accepted",
            "reject": "rejected",
            "edit": "edited",
            "regenerate": "rejected",
        }
        return mapping.get(self.signal_type, "other")


@dataclass
class TrainingSample:
    sample_id: str = field(default_factory=new_id)
    sample_type: Literal["sft", "dpo"] = "sft"
    instruction: str = ""
    chosen: str = ""
    rejected: Optional[str] = None
    score: float = 0.0
    source: Literal["signal", "teacher", "import", "manual"] = "signal"
    source_event_ids: list[str] = field(default_factory=list)
    source_adapter_version: Optional[str] = None
    preference_kind: Optional[str] = None
    preference_source: Optional[str] = None
    preference_pair_id: Optional[str] = None
    preference_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    used_in_version: Optional[str] = None
    signal_quality: "SignalQuality" = field(default_factory=lambda: SignalQuality())


@dataclass
class EvalDetail:
    prompt_id: Optional[str] = None
    prompt: str = ""
    base_output: str = ""
    adapted_output: str = ""
    reference_output: Optional[str] = None
    scores: dict[str, float] = field(default_factory=dict)
    rationale: Optional[str] = None
    passed: Optional[bool] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    adapter_version: str = ""
    base_model: str = ""
    num_test_samples: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    comparison: Literal["improved", "neutral", "degraded"] = "neutral"
    recommendation: Literal["deploy", "keep_previous", "needs_more_data"] = "needs_more_data"
    details: list[EvalDetail] = field(default_factory=list)
    judge_model: Optional[str] = None
    judge_prompt_version: Optional[str] = None
    judge_run_id: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterMeta:
    version: str = ""
    base_model: str = ""
    created_at: datetime = field(default_factory=utc_now)
    num_samples: int = 0
    state: AdapterLifecycleState = AdapterLifecycleState.TRAINING
    artifact_format: AdapterArtifactFormat = AdapterArtifactFormat.PEFT_LORA
    training_config: dict[str, Any] = field(default_factory=dict)
    eval_report: Optional[EvalReport] = None
    manifest_path: Optional[str] = None
    artifact_path: Optional[str] = None
    promoted_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InteractionEventSchema(BaseSchema):
    event_id: str = Field(default_factory=new_id)
    request_id: str = ""
    source_event_ids: list[str] = Field(default_factory=list)
    event_chain_ids: list[str] = Field(default_factory=list)
    parent_event_id: Optional[str] = None
    event_type: Literal["chat", "edit", "copy", "interrupt", "regenerate"] = "chat"
    timestamp: datetime = Field(default_factory=utc_now)
    session_id: str = ""
    adapter_version: Optional[str] = None
    scenario: Optional[str] = None
    user_input: str = ""
    model_output: str = ""
    user_action: Literal["accept", "reject", "edit", "copy", "interrupt", "regenerate"] = "accept"
    action_detail: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    signal_quality: "SignalQualitySchema" = Field(default_factory=lambda: SignalQualitySchema())


class SignalQualitySchema(BaseSchema):
    reply_style: Literal["accepted", "rejected", "edited", "other"] = "other"
    confidence: float = 0.0
    conflict: bool = False
    conflict_reason: Optional[str] = None
    confidence_reason: Optional[str] = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    rolled_back: bool = False
    replay_eligible: bool = False


class RawSignalSchema(BaseSchema):
    signal_id: str = Field(default_factory=new_id)
    source_event_id: str = ""
    request_id: str = ""
    session_id: str = ""
    parent_event_id: Optional[str] = None
    adapter_version: Optional[str] = None
    event_type: Literal["accept", "reject", "edit", "copy", "interrupt", "regenerate"] = "accept"
    timestamp: datetime = Field(default_factory=utc_now)
    context: str = ""
    model_output: str = ""
    user_action: dict[str, Any] = Field(default_factory=dict)
    source_event_ids: list[str] = Field(default_factory=list)
    event_chain_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    signal_quality: SignalQualitySchema = Field(default_factory=SignalQualitySchema)


class TrainingSampleSchema(BaseSchema):
    sample_id: str = Field(default_factory=new_id)
    sample_type: Literal["sft", "dpo"] = "sft"
    instruction: str = ""
    chosen: str = ""
    rejected: Optional[str] = None
    score: float = 0.0
    source: Literal["signal", "teacher", "import", "manual"] = "signal"
    source_event_ids: list[str] = Field(default_factory=list)
    source_adapter_version: Optional[str] = None
    preference_kind: Optional[str] = None
    preference_source: Optional[str] = None
    preference_pair_id: Optional[str] = None
    preference_reason: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    used_in_version: Optional[str] = None
    signal_quality: SignalQualitySchema = Field(default_factory=SignalQualitySchema)


class EvalDetailSchema(BaseSchema):
    prompt_id: Optional[str] = None
    prompt: str = ""
    base_output: str = ""
    adapted_output: str = ""
    reference_output: Optional[str] = None
    scores: dict[str, float] = Field(default_factory=dict)
    rationale: Optional[str] = None
    passed: Optional[bool] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalReportSchema(BaseSchema):
    adapter_version: str = ""
    base_model: str = ""
    num_test_samples: int = 0
    scores: dict[str, float] = Field(default_factory=dict)
    comparison: Literal["improved", "neutral", "degraded"] = "neutral"
    recommendation: Literal["deploy", "keep_previous", "needs_more_data"] = "needs_more_data"
    details: list[EvalDetailSchema] = Field(default_factory=list)
    judge_model: Optional[str] = None
    judge_prompt_version: Optional[str] = None
    judge_run_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdapterMetaSchema(BaseSchema):
    version: str = ""
    base_model: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    num_samples: int = 0
    state: AdapterLifecycleState = AdapterLifecycleState.TRAINING
    artifact_format: AdapterArtifactFormat = AdapterArtifactFormat.PEFT_LORA
    training_config: dict[str, Any] = Field(default_factory=dict)
    eval_report: Optional[EvalReportSchema] = None
    manifest_path: Optional[str] = None
    artifact_path: Optional[str] = None
    promoted_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Pydantic schemas for new reliability models

class RunnerHeartbeatSchema(BaseSchema):
    runner_id: str = Field(default_factory=new_id)
    job_id: Optional[str] = None
    workspace: str = "user_default"
    pid: Optional[int] = None
    timestamp: datetime = Field(default_factory=utc_now)
    sequence_number: int = 0
    state: RunnerState = RunnerState.HEALTHY
    progress_percent: float = 0.0
    current_step: str = ""
    step_details: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    memory_usage_mb: Optional[float] = None
    cpu_percent: Optional[float] = None


class TaskLeaseSchema(BaseSchema):
    lease_id: str = Field(default_factory=new_id)
    job_id: str = ""
    runner_id: str = ""
    workspace: str = "user_default"
    acquired_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime = Field(default_factory=lambda: utc_now() + timedelta(seconds=60))
    renewed_at: Optional[datetime] = None
    renewal_count: int = 0
    state: LeaseState = LeaseState.VALID
    lease_timeout_seconds: float = 60.0
    warning_threshold_seconds: float = 45.0


class RecoveryCheckpointSchema(BaseSchema):
    checkpoint_id: str = Field(default_factory=new_id)
    job_id: str = ""
    workspace: str = "user_default"
    created_at: datetime = Field(default_factory=utc_now)
    epoch: int = 0
    global_step: int = 0
    samples_processed: int = 0
    model_state_path: Optional[str] = None
    optimizer_state_path: Optional[str] = None
    scheduler_state_path: Optional[str] = None
    training_config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    is_resumable: bool = True
    resume_attempt_count: int = 0
    last_resume_at: Optional[datetime] = None


class DeadLetterEntrySchema(BaseSchema):
    entry_id: str = Field(default_factory=new_id)
    job_id: str = ""
    workspace: str = "user_default"
    failed_at: datetime = Field(default_factory=utc_now)
    failure_reason: str = ""
    failure_category: str = ""
    error_details: dict[str, Any] = Field(default_factory=dict)
    stack_trace: Optional[str] = None
    retry_count: int = 0
    retry_history: list[dict[str, Any]] = Field(default_factory=list)
    original_task: dict[str, Any] = Field(default_factory=dict)
    last_checkpoint: Optional[RecoveryCheckpointSchema] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_action: Optional[str] = None


class RestartPolicySchema(BaseSchema):
    max_restart_attempts: int = 3
    base_backoff_seconds: float = 15.0
    max_backoff_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    reset_after_seconds: float = 3600.0
    current_attempt: int = 0
    last_failure_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    next_restart_after: Optional[datetime] = None


class AlertThresholdSchema(BaseSchema):
    consecutive_failures_warning: int = 2
    consecutive_failures_error: int = 3
    consecutive_failures_critical: int = 5
    heartbeat_delay_warning: float = 10.0
    heartbeat_delay_error: float = 30.0
    lease_expiry_warning: float = 15.0
    task_stall_warning: float = 300.0
    task_stall_error: float = 600.0
    task_stall_critical: float = 1800.0
    max_recovery_attempts_warning: int = 2
    dead_letter_queue_size_warning: int = 5
    dead_letter_queue_size_critical: int = 10


class AlertEventSchema(BaseSchema):
    alert_id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)
    level: AlertLevel = AlertLevel.INFO
    scope: str = ""
    reason: str = ""
    message: str = ""
    job_id: Optional[str] = None
    runner_id: Optional[str] = None
    workspace: str = "user_default"
    context: dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_note: Optional[str] = None


class TaskExecutionMetadataSchema(BaseSchema):
    job_id: str = ""
    workspace: str = "user_default"
    created_at: datetime = Field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    current_state: TaskState = TaskState.PENDING
    previous_state: Optional[TaskState] = None
    state_history: list[dict[str, Any]] = Field(default_factory=list)
    assigned_runner_id: Optional[str] = None
    lease_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    recovery_checkpoint_id: Optional[str] = None
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure_reason: Optional[str] = None
    last_error_details: dict[str, Any] = Field(default_factory=dict)
    execution_time_seconds: Optional[float] = None
    total_cpu_time_seconds: Optional[float] = None
    peak_memory_mb: Optional[float] = None
