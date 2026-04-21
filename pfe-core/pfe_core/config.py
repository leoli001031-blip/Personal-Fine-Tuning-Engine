"""Configuration objects and TOML helpers for PFE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, get_type_hints
import json
import math

try:
    import tomllib  # type: ignore[no-redef]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .db.sqlite import resolve_config_path as _resolve_config_path
from .db.sqlite import resolve_home as _resolve_home


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return repr(value)
        raise ValueError("TOML does not support non-finite floats")
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return '""'
    if isinstance(value, str):
        return f'"{_toml_escape(value)}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def _render_toml_section(prefix: list[str], data: dict[str, Any], lines: list[str]) -> None:
    scalar_items: list[tuple[str, Any]] = []
    nested_items: list[tuple[str, dict[str, Any]]] = []
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested_items.append((key, value))
        else:
            scalar_items.append((key, value))

    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")

    for key, value in scalar_items:
        lines.append(f"{key} = {_toml_value(value)}")

    for key, value in nested_items:
        if lines and lines[-1] != "":
            lines.append("")
        _render_toml_section(prefix + [key], value, lines)


def _dataclass_to_nested_dict(instance: Any) -> dict[str, Any]:
    if is_dataclass(instance):
        result: dict[str, Any] = {}
        for field_info in fields(instance):
            value = getattr(instance, field_info.name)
            result[field_info.name] = _dataclass_to_nested_dict(value)
        return result
    if isinstance(instance, dict):
        return {key: _dataclass_to_nested_dict(value) for key, value in instance.items()}
    if isinstance(instance, list):
        return [_dataclass_to_nested_dict(value) for value in instance]
    return instance


def _coerce_nested_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return data

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in data:
            continue
        raw_value = data[field_info.name]
        kwargs[field_info.name] = _coerce_value(type_hints.get(field_info.name, field_info.type), raw_value)
    return cls(**kwargs)


def _coerce_value(annotation: Any, value: Any) -> Any:
    from typing import get_args, get_origin

    origin = get_origin(annotation)
    args = get_args(annotation)

    if value is None:
        return None

    if origin is list:
        item_type = args[0] if args else Any
        return [_coerce_value(item_type, item) for item in value]
    if origin is dict:
        value_type = args[1] if len(args) > 1 else Any
        return {key: _coerce_value(value_type, item) for key, item in value.items()}
    if origin is tuple:
        item_type = args[0] if args else Any
        return tuple(_coerce_value(item_type, item) for item in value)
    if origin is Literal:
        return value
    if origin is not None and type(None) in args:
        non_none = next((arg for arg in args if arg is not type(None)), Any)
        return _coerce_value(non_none, value)

    if isinstance(annotation, type):
        if is_dataclass(annotation) and isinstance(value, dict):
            return _coerce_nested_dataclass(annotation, value)
        if issubclass(annotation, BaseModel) and isinstance(value, dict):
            return annotation.model_validate(value)

    return value


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


@dataclass
class SignalQualityGateConfig:
    """Signal quality filtering thresholds for training eligibility."""

    minimum_confidence: float = 0.65
    reject_conflicted_signal_quality: bool = True
    minimum_signal_length: int = 5
    maximum_signal_length: int = 4096
    require_complete_event_chain: bool = True
    require_user_action: bool = True


@dataclass
class TrainTriggerPolicyConfig:
    """Training trigger thresholds and cooldown policies."""

    enabled: bool = False
    min_new_samples: int = 50
    max_interval_days: int = 7
    min_trigger_interval_minutes: int = 60
    failure_backoff_minutes: int = 30
    consecutive_failure_threshold: int = 3
    consecutive_failure_backoff_multiplier: float = 2.0
    max_queue_depth: int = 10
    pause_on_queue_full: bool = True
    require_holdout_split: bool = True
    signal_quality_gate: SignalQualityGateConfig = field(default_factory=SignalQualityGateConfig)


@dataclass
class EvalGatePolicyConfig:
    """Evaluation trigger policy and quality thresholds."""

    auto_trigger: bool = False
    trigger_delay_seconds: float = 0.0
    eval_split_ratio: float = 0.2
    min_eval_samples: int = 5
    max_eval_samples: int = 200
    eval_frequency_hours: int = 24
    re_evaluate_on_promote: bool = False
    require_holdout_split: bool = True
    forbid_teacher_test_overlap: bool = True
    auto_promote_after_eval: bool = False  # Auto-promote based on eval results


@dataclass
class PromoteGatePolicyConfig:
    """Promotion quality thresholds and comparison policies."""

    auto_promote: bool = False
    min_quality_score: float = 0.7
    min_style_match_score: float = 0.6
    min_preference_alignment_score: float = 0.6
    min_quality_preservation_score: float = 0.8
    require_eval_recommendation_deploy: bool = True
    compare_with_previous: bool = True
    min_improvement_delta: float = 0.05
    require_manual_confirm_on_regression: bool = True
    max_promote_frequency_hours: int = 1
    win_rate_threshold: float = 0.6  # Win rate vs baseline required for auto-promotion


@dataclass
class ConfirmationPolicyConfig:
    """Human confirmation requirements for critical operations."""

    first_training_requires_confirm: bool = True
    quality_regression_requires_confirm: bool = True
    rapid_trigger_requires_confirm: bool = True
    rapid_trigger_threshold_minutes: int = 30
    queue_confirmation_default_approved: bool = False
    auto_approve_below_quality_threshold: bool = False


@dataclass
class QueueReviewPolicyConfig:
    """Queue processing and review policies."""

    default_review_mode: Literal["auto_approve", "manual_review"] = "auto_approve"
    priority_policy: Literal["fifo", "quality_score", "hybrid"] = "hybrid"
    quality_score_weight: float = 0.3
    batch_size: int = 5
    max_concurrent_jobs: int = 1
    auto_retry_failed: bool = False
    max_retry_attempts: int = 2
    retry_backoff_minutes: int = 10


@dataclass
class TrainerTriggerConfig:
    """Unified auto-train/eval/promote trigger configuration.

    Backward-compatible wrapper that delegates to new policy configs.
    """

    enabled: bool = False
    min_new_samples: int = 50
    max_interval_days: int = 7
    min_trigger_interval_minutes: int = 0
    failure_backoff_minutes: int = 30
    queue_mode: Literal["inline", "deferred"] = "inline"
    queue_dedup_scope: Literal["workspace", "base_model", "train_config"] = "train_config"
    queue_priority_policy: Literal["fifo", "source_default", "promotion_bias"] = "source_default"
    queue_process_batch_size: int = 5
    queue_process_until_idle_max: int = 10
    queue_worker_max_cycles: int = 10
    queue_worker_idle_rounds: int = 1
    queue_worker_poll_seconds: float = 0.0
    queue_worker_runner_max_seconds: float = 30.0
    queue_worker_runner_idle_sleep_seconds: float = 1.0
    queue_daemon_auto_restart: bool = True
    queue_daemon_auto_recover: bool = True
    queue_daemon_heartbeat_interval_seconds: float = 2.0
    queue_daemon_lease_timeout_seconds: float = 15.0
    queue_daemon_restart_backoff_seconds: float = 15.0
    queue_daemon_max_restart_attempts: int = 3
    # Reliability configuration
    reliability_enabled: bool = True
    lease_timeout_seconds: float = 60.0
    lease_warning_threshold_ratio: float = 0.75
    heartbeat_stale_threshold_multiplier: float = 2.0
    max_restart_backoff_seconds: float = 300.0
    restart_backoff_multiplier: float = 2.0
    restart_reset_after_seconds: float = 3600.0
    max_task_retries: int = 3
    max_resume_attempts: int = 3
    checkpoint_keep_count: int = 3
    # Alert thresholds
    alert_consecutive_failures_warning: int = 2
    alert_consecutive_failures_error: int = 3
    alert_consecutive_failures_critical: int = 5
    alert_heartbeat_delay_warning_seconds: float = 10.0
    alert_heartbeat_delay_error_seconds: float = 30.0
    alert_task_stall_warning_seconds: float = 300.0
    alert_task_stall_error_seconds: float = 600.0
    alert_task_stall_critical_seconds: float = 1800.0
    require_queue_confirmation: bool = False
    auto_evaluate: bool = False
    auto_promote: bool = False
    eval_num_samples: int = 3
    preference_reinforced_sample_weight: float = 1.5
    train_trigger_policy: TrainTriggerPolicyConfig = field(default_factory=TrainTriggerPolicyConfig)
    eval_gate_policy: EvalGatePolicyConfig = field(default_factory=EvalGatePolicyConfig)
    promote_gate_policy: PromoteGatePolicyConfig = field(default_factory=PromoteGatePolicyConfig)
    confirmation_policy: ConfirmationPolicyConfig = field(default_factory=ConfirmationPolicyConfig)
    queue_review_policy: QueueReviewPolicyConfig = field(default_factory=QueueReviewPolicyConfig)


@dataclass
class DPOConfig:
    """DPO (Direct Preference Optimization) training configuration.

    Attributes:
        beta: DPO temperature parameter controlling preference sensitivity
        label_smoothing: Label smoothing for DPO loss
        loss_type: DPO loss type ("sigmoid", "hinge", "ipo")
        min_preference_confidence: Minimum confidence for preference pairs
        max_prompt_length: Maximum length for prompts
        max_response_length: Maximum length for responses
        use_weighted_loss: Whether to use confidence-weighted loss
    """

    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    min_preference_confidence: float = 0.4
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    use_weighted_loss: bool = True


@dataclass
class TeacherDistillationConfig:
    """Teacher distillation configuration for P2-F."""

    enabled: bool = False
    teacher_model: str = ""
    api_base: str = ""
    api_key: str = ""
    max_teacher_ratio: float = 0.3
    similarity_threshold: float = 0.7
    temperature: float = 0.7


@dataclass
class TrainerConfig:
    method: Literal["lora", "qlora"] = "qlora"
    train_type: Literal["sft", "dpo"] = "sft"
    backend: str = "auto"
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    learning_rate: float = 2e-4
    max_samples: int = 500
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dpo_beta: float = 0.1
    replay_ratio: float = 0.3
    dpo_replay_ratio: float = 0.2
    replay_min_samples: int = 1
    replay_history_limit: int = 20
    incremental_parent_selector: Literal["explicit", "latest", "recent", "promoted_or_latest"] = "promoted_or_latest"
    trigger: TrainerTriggerConfig = field(default_factory=TrainerTriggerConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    teacher_distillation: TeacherDistillationConfig = field(default_factory=TeacherDistillationConfig)


@dataclass
class CuratorConfig:
    signal_collection_enabled: bool = True
    score_threshold: float = 0.3
    dedup_similarity: float = 0.92
    dedup_model: str = "BAAI/bge-small-zh-v1.5"


@dataclass
class DistillationConfig:
    enabled: bool = False
    teacher_model: str = ""
    teacher_prompt_version: str = "v1"
    teacher_temperature: float = 0.7
    teacher_max_samples: int = 200
    rewrite_weak_samples: bool = True
    generate_dpo_pairs: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    prada_similarity_threshold: float = 0.85
    prada_enabled: bool = True
    local_teacher_model: str = ""  # Base model for local teacher inference

    def __post_init__(self) -> None:
        _validate_split_triplet(self.train_split, self.val_split, self.test_split)


@dataclass
class JudgeConfig:
    mode: Literal["local_first", "cloud_only"] = "local_first"
    judge_model: str = ""
    judge_prompt_version: str = "v1"
    require_holdout_split: bool = True
    forbid_teacher_test_overlap: bool = True


@dataclass
class EvaluationConfig:
    judge: JudgeConfig = field(default_factory=JudgeConfig)


@dataclass
class ServerConfig:
    port: int = 8921
    host: str = "127.0.0.1"


@dataclass
class RouterConfig:
    strategy: Literal["local_only", "keyword", "confidence", "hybrid"] = "local_only"
    confidence_threshold: float = 0.7
    cloud_model: str = ""
    cloud_api_key_env: str = "OPENAI_API_KEY"
    # Phase 2: Multi-scenario routing configuration
    enable_scenario_routing: bool = True
    scenario_config_path: str = "~/.pfe/scenarios.json"
    default_scenario: str = "chat"
    routing_cache_size: int = 1000
    min_routing_confidence: float = 0.3
    fallback_to_latest: bool = True


@dataclass
class PIIPolicyConfig:
    """PII detection and anonymization policy configuration."""

    enabled: bool = True
    """Enable PII detection and anonymization."""

    strategy: Literal["replace", "hash", "mask", "remove"] = "replace"
    """Anonymization strategy: replace (tags), hash (irreversible), mask (partial), remove."""

    detect_on_collect: bool = True
    """Detect PII during signal collection."""

    anonymize_on_curate: bool = True
    """Anonymize PII during dataset curation."""

    block_high_risk: bool = False
    """Block samples with high-risk PII (credit cards, SSN, etc.) from training."""

    min_confidence: float = 0.7
    """Minimum confidence threshold for PII detection."""

    pii_types: list[str] = field(default_factory=lambda: [
        "email", "phone", "id_card", "person_name", "address",
        "ip_address", "credit_card", "bank_card"
    ])
    """List of PII types to detect and anonymize."""

    audit_log_path: str = "~/.pfe/logs/pii_audit.jsonl"
    """Path to PII audit log."""

    salt: str | None = None
    """Salt for hashing strategy (should be set via env var)."""


@dataclass
class PrivacyConfig:
    mode: Literal["strict_local", "cloud_assisted"] = "strict_local"
    allow_teacher_cloud: bool = False
    allow_judge_cloud: bool = False
    allow_router_cloud: bool = False
    redact_pii: bool = True
    require_explicit_consent: bool = True
    egress_audit_log: str = "~/.pfe/logs/egress_audit.jsonl"
    pii_policy: PIIPolicyConfig = field(default_factory=PIIPolicyConfig)


@dataclass
class SecurityConfig:
    allow_remote_access: bool = False
    auth_mode: Literal["local_optional", "api_key_required"] = "local_optional"
    api_key_env: str = "PFE_API_KEY"
    allowed_origins: list[str] = field(default_factory=lambda: ["http://127.0.0.1", "http://localhost"])


@dataclass
class StorageConfig:
    signal_retention_days: int = 90
    adapter_keep_versions: int = 10


@dataclass
class LoggingConfig:
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str = "~/.pfe/logs/pfe.log"
    max_size_mb: int = 50
    backup_count: int = 3
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass
class CollectorConfig:
    """Configuration for ChatCollector implicit signal extraction."""

    enabled: bool = True
    """Whether signal collection is enabled."""

    accept_confidence_threshold: float = 0.5
    """Minimum confidence for accept signals to be recorded."""

    edit_confidence_threshold: float = 0.5
    """Minimum confidence for edit signals to be recorded."""

    reject_confidence_threshold: float = 0.5
    """Minimum confidence for reject signals to be recorded."""

    regenerate_confidence_threshold: float = 0.5
    """Minimum confidence for regenerate signals to be recorded."""

    edit_distance_metric: str = "levenshtein"
    """Metric for calculating edit distance (levenshtein or jaro_winkler)."""

    time_decay_enabled: bool = True
    """Whether to apply time-based confidence decay for accept signals."""

    strong_accept_threshold_seconds: float = 5.0
    """Response time below which is considered strong accept."""

    weak_accept_threshold_seconds: float = 60.0
    """Response time above which is considered weak accept."""

    store_interactions: bool = True
    """Whether to store raw ChatInteraction records."""

    max_interaction_history: int = 1000
    """Maximum number of interactions to keep in memory."""

    # PII detection settings
    pii_detection_enabled: bool = True
    """Whether to enable PII detection on signals."""

    detect_pii: bool = True
    """Enable PII detection during signal collection."""

    pii_sensitivity: str = "medium"
    """PII detection sensitivity: low, medium, or high."""

    pii_anonymization_strategy: str = "mask"
    """Anonymization strategy: replace, hash, mask, or remove."""

    pii_action_on_detect: str = "anonymize"
    """Action when PII detected: anonymize, block, or flag."""

    pii_confidence_threshold: float = 0.7
    """Minimum confidence threshold for PII detection."""

    pii_min_confidence: float = 0.7
    """Minimum confidence threshold for PII detection (alias)."""

    pii_audit_enabled: bool = True
    """Whether to enable PII audit logging."""

    explicit_user_data_routing_enabled: bool = True
    """Whether to route explicit user facts/preferences into memory/profile."""

    pii_types_to_detect: list[str] = field(default_factory=lambda: [
        "email", "phone", "id_card", "person_name", "address"
    ])
    """List of PII types to detect during collection."""

    signal_rules: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "accept": {
            "base_confidence": 0.7,
            "strong_multiplier": 1.29,
            "weak_multiplier": 0.57,
        },
        "edit": {
            "slight_threshold": 0.2,
            "moderate_threshold": 0.5,
            "slight_confidence": 0.6,
            "moderate_confidence": 0.8,
            "strong_confidence": 0.9,
        },
        "reject": {
            "base_confidence": 0.95,
        },
        "regenerate": {
            "base_confidence": 0.85,
        },
    })
    """Fine-grained signal extraction rules with configurable thresholds."""


@dataclass
class PFEConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    curator: CuratorConfig = field(default_factory=CuratorConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelConfig):
            self.model = ModelConfig(**(self.model or {}))
        if isinstance(self.trainer, dict):
            trigger_data = self.trainer.get("trigger") or {}
            td_data = self.trainer.get("teacher_distillation") or {}
            self.trainer = TrainerConfig(
                **{
                    **self.trainer,
                    "trigger": TrainerTriggerConfig(
                        **trigger_data,
                        train_trigger_policy=TrainTriggerPolicyConfig(**(trigger_data.get("train_trigger_policy") or {})),
                        eval_gate_policy=EvalGatePolicyConfig(**(trigger_data.get("eval_gate_policy") or {})),
                        promote_gate_policy=PromoteGatePolicyConfig(**(trigger_data.get("promote_gate_policy") or {})),
                        confirmation_policy=ConfirmationPolicyConfig(**(trigger_data.get("confirmation_policy") or {})),
                        queue_review_policy=QueueReviewPolicyConfig(**(trigger_data.get("queue_review_policy") or {})),
                    ),
                    "teacher_distillation": TeacherDistillationConfig(**td_data),
                }
            )
        elif isinstance(self.trainer, TrainerConfig) and not isinstance(self.trainer.trigger, TrainerTriggerConfig):
            trigger_data = self.trainer.trigger or {}
            self.trainer.trigger = TrainerTriggerConfig(
                **trigger_data,
                train_trigger_policy=TrainTriggerPolicyConfig(**(trigger_data.get("train_trigger_policy") or {})),
                eval_gate_policy=EvalGatePolicyConfig(**(trigger_data.get("eval_gate_policy") or {})),
                promote_gate_policy=PromoteGatePolicyConfig(**(trigger_data.get("promote_gate_policy") or {})),
                confirmation_policy=ConfirmationPolicyConfig(**(trigger_data.get("confirmation_policy") or {})),
                queue_review_policy=QueueReviewPolicyConfig(**(trigger_data.get("queue_review_policy") or {})),
            )
        if not isinstance(self.curator, CuratorConfig):
            self.curator = CuratorConfig(**(self.curator or {}))
        if not isinstance(self.distillation, DistillationConfig):
            self.distillation = DistillationConfig(**(self.distillation or {}))
        if isinstance(self.evaluation, dict):
            self.evaluation = EvaluationConfig(judge=JudgeConfig(**(self.evaluation.get("judge") or {})))
        elif isinstance(self.evaluation, EvaluationConfig) and not isinstance(self.evaluation.judge, JudgeConfig):
            self.evaluation.judge = JudgeConfig(**(self.evaluation.judge or {}))
        if not isinstance(self.server, ServerConfig):
            self.server = ServerConfig(**(self.server or {}))
        if not isinstance(self.router, RouterConfig):
            self.router = RouterConfig(**(self.router or {}))
        if not isinstance(self.privacy, PrivacyConfig):
            self.privacy = PrivacyConfig(**(self.privacy or {}))
        if not isinstance(self.security, SecurityConfig):
            self.security = SecurityConfig(**(self.security or {}))
        if not isinstance(self.storage, StorageConfig):
            self.storage = StorageConfig(**(self.storage or {}))
        if not isinstance(self.logging, LoggingConfig):
            self.logging = LoggingConfig(**(self.logging or {}))
        if not isinstance(self.collector, CollectorConfig):
            self.collector = CollectorConfig(**(self.collector or {}))

    @property
    def mode(self) -> str:
        return self.privacy.mode

    @classmethod
    def resolve_home(cls, home: str | Path | None = None) -> Path:
        return _resolve_home(home)

    @classmethod
    def resolve_config_path(cls, path: str | Path | None = None, *, home: str | Path | None = None) -> Path:
        return _resolve_config_path(path, home=home)

    @classmethod
    def default_path(cls, *, home: str | Path | None = None) -> Path:
        return cls.resolve_config_path(home=home)

    @classmethod
    def load(cls, path: str | Path | None = None, *, home: str | Path | None = None) -> "PFEConfig":
        config_path = cls.resolve_config_path(path, home=home)
        if not config_path.exists():
            return cls()
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PFEConfig":
        trainer_data = data.get("trainer") or {}
        trigger_data = trainer_data.get("trigger") or {}
        evaluation_data = data.get("evaluation") or {}

        # Helper to safely get nested dict
        def _nested(cfg: dict, key: str) -> dict:
            val = cfg.get(key) or {}
            return val if isinstance(val, dict) else {}

        # Build trigger config, removing nested policy keys from trigger_data
        # to avoid duplicate keyword arguments
        trigger_data_clean = {k: v for k, v in trigger_data.items()
                              if k not in ("train_trigger_policy", "eval_gate_policy",
                                           "promote_gate_policy", "confirmation_policy",
                                           "queue_review_policy")}

        return cls(
            model=ModelConfig(**(data.get("model") or {})),
            trainer=TrainerConfig(
                **{
                    **{k: v for k, v in trainer_data.items() if k not in ("trigger", "dpo", "teacher_distillation")},
                    "trigger": TrainerTriggerConfig(
                        **trigger_data_clean,
                        train_trigger_policy=TrainTriggerPolicyConfig(
                            **_nested(trigger_data, "train_trigger_policy")
                        ),
                        eval_gate_policy=EvalGatePolicyConfig(
                            **_nested(trigger_data, "eval_gate_policy")
                        ),
                        promote_gate_policy=PromoteGatePolicyConfig(
                            **_nested(trigger_data, "promote_gate_policy")
                        ),
                        confirmation_policy=ConfirmationPolicyConfig(
                            **_nested(trigger_data, "confirmation_policy")
                        ),
                        queue_review_policy=QueueReviewPolicyConfig(
                            **_nested(trigger_data, "queue_review_policy")
                        ),
                    ),
                    "dpo": DPOConfig(**_nested(trainer_data, "dpo")),
                    "teacher_distillation": TeacherDistillationConfig(**_nested(trainer_data, "teacher_distillation")),
                }
            ),
            curator=CuratorConfig(**(data.get("curator") or {})),
            distillation=DistillationConfig(**(data.get("distillation") or {})),
            evaluation=EvaluationConfig(
                judge=JudgeConfig(**(evaluation_data.get("judge") or {}))
            ),
            server=ServerConfig(**(data.get("server") or {})),
            router=RouterConfig(**(data.get("router") or {})),
            privacy=PrivacyConfig(**(data.get("privacy") or {})),
            security=SecurityConfig(**(data.get("security") or {})),
            storage=StorageConfig(**(data.get("storage") or {})),
            logging=LoggingConfig(**(data.get("logging") or {})),
            collector=CollectorConfig(**(data.get("collector") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_nested_dict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_toml(self) -> str:
        lines: list[str] = []
        _render_toml_section([], self.to_dict(), lines)
        return "\n".join(lines).rstrip() + "\n"

    def save(self, path: str | Path | None = None, *, home: str | Path | None = None) -> Path:
        config_path = self.resolve_config_path(path, home=home)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(self.to_toml(), encoding="utf-8")
        return config_path

    @property
    def home(self) -> Path:
        return self.resolve_home().expanduser()

    @property
    def config_path(self) -> Path:
        return self.default_path()


def _validate_split_triplet(train_split: float, val_split: float, test_split: float) -> None:
    splits = (train_split, val_split, test_split)
    if any(split < 0 for split in splits):
        raise ValueError("Dataset split ratios must be non-negative")
    total = sum(splits)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Dataset split ratios must sum to 1.0, got {total}")


class ModelConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class SignalQualityGateConfigSchema(BaseModel):
    """Signal quality filtering thresholds schema."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    minimum_confidence: float = 0.65
    reject_conflicted_signal_quality: bool = True
    minimum_signal_length: int = 5
    maximum_signal_length: int = 4096
    require_complete_event_chain: bool = True
    require_user_action: bool = True


class TrainTriggerPolicyConfigSchema(BaseModel):
    """Training trigger thresholds and cooldown policies schema."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    enabled: bool = False
    min_new_samples: int = 50
    max_interval_days: int = 7
    min_trigger_interval_minutes: int = 60
    failure_backoff_minutes: int = 30
    consecutive_failure_threshold: int = 3
    consecutive_failure_backoff_multiplier: float = 2.0
    max_queue_depth: int = 10
    pause_on_queue_full: bool = True
    require_holdout_split: bool = True
    signal_quality_gate: SignalQualityGateConfigSchema = Field(default_factory=SignalQualityGateConfigSchema)


class EvalGatePolicyConfigSchema(BaseModel):
    """Evaluation trigger policy and quality thresholds schema."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    auto_trigger: bool = False
    trigger_delay_seconds: float = 0.0
    eval_split_ratio: float = 0.2
    min_eval_samples: int = 5
    max_eval_samples: int = 200
    eval_frequency_hours: int = 24
    re_evaluate_on_promote: bool = False
    require_holdout_split: bool = True
    forbid_teacher_test_overlap: bool = True
    auto_promote_after_eval: bool = False


class PromoteGatePolicyConfigSchema(BaseModel):
    """Promotion quality thresholds and comparison policies schema."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    auto_promote: bool = False
    min_quality_score: float = 0.7
    min_style_match_score: float = 0.6
    min_preference_alignment_score: float = 0.6
    min_quality_preservation_score: float = 0.8
    require_eval_recommendation_deploy: bool = True
    compare_with_previous: bool = True
    min_improvement_delta: float = 0.05
    require_manual_confirm_on_regression: bool = True
    max_promote_frequency_hours: int = 1
    win_rate_threshold: float = 0.6


class ConfirmationPolicyConfigSchema(BaseModel):
    """Human confirmation requirements for critical operations schema."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    first_training_requires_confirm: bool = True
    quality_regression_requires_confirm: bool = True
    rapid_trigger_requires_confirm: bool = True
    rapid_trigger_threshold_minutes: int = 30
    queue_confirmation_default_approved: bool = False
    auto_approve_below_quality_threshold: bool = False


class QueueReviewPolicyConfigSchema(BaseModel):
    """Queue processing and review policies schema."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    default_review_mode: Literal["auto_approve", "manual_review"] = "auto_approve"
    priority_policy: Literal["fifo", "quality_score", "hybrid"] = "hybrid"
    quality_score_weight: float = 0.3
    batch_size: int = 5
    max_concurrent_jobs: int = 1
    auto_retry_failed: bool = False
    max_retry_attempts: int = 2
    retry_backoff_minutes: int = 10


class TrainerTriggerConfigSchema(BaseModel):
    """Unified auto-train/eval/promote trigger configuration schema."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    enabled: bool = False
    min_new_samples: int = 50
    max_interval_days: int = 7
    min_trigger_interval_minutes: int = 0
    failure_backoff_minutes: int = 30
    queue_mode: Literal["inline", "deferred"] = "inline"
    queue_dedup_scope: Literal["workspace", "base_model", "train_config"] = "train_config"
    queue_priority_policy: Literal["fifo", "source_default", "promotion_bias"] = "source_default"
    queue_process_batch_size: int = 5
    queue_process_until_idle_max: int = 10
    queue_worker_max_cycles: int = 10
    queue_worker_idle_rounds: int = 1
    queue_worker_poll_seconds: float = 0.0
    queue_worker_runner_max_seconds: float = 30.0
    queue_worker_runner_idle_sleep_seconds: float = 1.0
    queue_daemon_auto_restart: bool = True
    queue_daemon_auto_recover: bool = True
    queue_daemon_heartbeat_interval_seconds: float = 2.0
    queue_daemon_lease_timeout_seconds: float = 15.0
    queue_daemon_restart_backoff_seconds: float = 15.0
    queue_daemon_max_restart_attempts: int = 3
    require_queue_confirmation: bool = False
    auto_evaluate: bool = False
    auto_promote: bool = False
    eval_num_samples: int = 3
    preference_reinforced_sample_weight: float = 1.5
    train_trigger_policy: TrainTriggerPolicyConfigSchema = Field(default_factory=TrainTriggerPolicyConfigSchema)
    eval_gate_policy: EvalGatePolicyConfigSchema = Field(default_factory=EvalGatePolicyConfigSchema)
    promote_gate_policy: PromoteGatePolicyConfigSchema = Field(default_factory=PromoteGatePolicyConfigSchema)
    confirmation_policy: ConfirmationPolicyConfigSchema = Field(default_factory=ConfirmationPolicyConfigSchema)
    queue_review_policy: QueueReviewPolicyConfigSchema = Field(default_factory=QueueReviewPolicyConfigSchema)


class DPOConfigSchema(BaseModel):
    """DPO training configuration schema."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    min_preference_confidence: float = 0.4
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    use_weighted_loss: bool = True


class TeacherDistillationConfigSchema(BaseModel):
    """Teacher distillation configuration schema."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    enabled: bool = False
    teacher_model: str = ""
    api_base: str = ""
    api_key: str = ""
    max_teacher_ratio: float = 0.3
    similarity_threshold: float = 0.7
    temperature: float = 0.7


class TrainerConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    method: Literal["lora", "qlora"] = "qlora"
    train_type: Literal["sft", "dpo"] = "sft"
    backend: str = "auto"
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    learning_rate: float = 2e-4
    max_samples: int = 500
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dpo_beta: float = 0.1
    replay_ratio: float = 0.3
    dpo_replay_ratio: float = 0.2
    replay_min_samples: int = 1
    replay_history_limit: int = 20
    incremental_parent_selector: Literal["explicit", "latest", "recent", "promoted_or_latest"] = "promoted_or_latest"
    trigger: TrainerTriggerConfigSchema = Field(default_factory=TrainerTriggerConfigSchema)
    dpo: DPOConfigSchema = Field(default_factory=DPOConfigSchema)
    teacher_distillation: TeacherDistillationConfigSchema = Field(default_factory=TeacherDistillationConfigSchema)


class CuratorConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    signal_collection_enabled: bool = True
    score_threshold: float = 0.3
    dedup_similarity: float = 0.92
    dedup_model: str = "BAAI/bge-small-zh-v1.5"


class DistillationConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    enabled: bool = False
    teacher_model: str = ""
    teacher_prompt_version: str = "v1"
    teacher_temperature: float = 0.7
    teacher_max_samples: int = 200
    rewrite_weak_samples: bool = True
    generate_dpo_pairs: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    prada_similarity_threshold: float = 0.85
    prada_enabled: bool = True
    local_teacher_model: str = ""

    @model_validator(mode="after")
    def _validate_splits(self) -> "DistillationConfigSchema":
        _validate_split_triplet(self.train_split, self.val_split, self.test_split)
        return self


class JudgeConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    mode: Literal["local_first", "cloud_only"] = "local_first"
    judge_model: str = ""
    judge_prompt_version: str = "v1"
    require_holdout_split: bool = True
    forbid_teacher_test_overlap: bool = True


class EvaluationConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    judge: JudgeConfigSchema = Field(default_factory=JudgeConfigSchema)


class ServerConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    port: int = 8921
    host: str = "127.0.0.1"


class RouterConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    strategy: Literal["local_only", "keyword", "confidence", "hybrid"] = "local_only"
    confidence_threshold: float = 0.7
    cloud_model: str = ""
    cloud_api_key_env: str = "OPENAI_API_KEY"
    # Phase 2: Multi-scenario routing configuration
    enable_scenario_routing: bool = True
    scenario_config_path: str = "~/.pfe/scenarios.json"
    default_scenario: str = "chat"
    routing_cache_size: int = 1000
    min_routing_confidence: float = 0.3
    fallback_to_latest: bool = True


class PrivacyConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    mode: Literal["strict_local", "cloud_assisted"] = "strict_local"
    allow_teacher_cloud: bool = False
    allow_judge_cloud: bool = False
    allow_router_cloud: bool = False
    redact_pii: bool = True
    require_explicit_consent: bool = True
    egress_audit_log: str = "~/.pfe/logs/egress_audit.jsonl"


class SecurityConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    allow_remote_access: bool = False
    auth_mode: Literal["local_optional", "api_key_required"] = "local_optional"
    api_key_env: str = "PFE_API_KEY"
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1", "http://localhost"])


class StorageConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    signal_retention_days: int = 90
    adapter_keep_versions: int = 10


class LoggingConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str = "~/.pfe/logs/pfe.log"
    max_size_mb: int = 50
    backup_count: int = 3
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class PFEConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    model: ModelConfigSchema = Field(default_factory=ModelConfigSchema)
    trainer: TrainerConfigSchema = Field(default_factory=TrainerConfigSchema)
    curator: CuratorConfigSchema = Field(default_factory=CuratorConfigSchema)
    distillation: DistillationConfigSchema = Field(default_factory=DistillationConfigSchema)
    evaluation: EvaluationConfigSchema = Field(default_factory=EvaluationConfigSchema)
    server: ServerConfigSchema = Field(default_factory=ServerConfigSchema)
    router: RouterConfigSchema = Field(default_factory=RouterConfigSchema)
    privacy: PrivacyConfigSchema = Field(default_factory=PrivacyConfigSchema)
    security: SecurityConfigSchema = Field(default_factory=SecurityConfigSchema)
    storage: StorageConfigSchema = Field(default_factory=StorageConfigSchema)
    logging: LoggingConfigSchema = Field(default_factory=LoggingConfigSchema)
