"""Teacher distillation skeleton for PFE."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Iterable, Optional, Sequence

from .datasets import (
    SampleFilterConfig,
    TrainingDataset,
    attach_dataset_split,
    build_training_dataset,
    build_signal_provenance,
    build_signal_quality,
    deduplicate_samples,
    filter_samples,
    normalize_dataset_split,
    normalize_reply_style,
    signal_quality_filter_reasons,
    split_samples,
)
from .prada import (
    compute_text_similarity,
    should_use_teacher_output,
    should_use_teacher_output_for_user_preference,
)
from .teacher_client import TeacherClientConfig, TeacherInferenceClient
from .teacher_fusion import TeacherSignalFusion, TeacherSignalFusionConfig
from ..profile_extractor import get_user_profile_store

try:  # pragma: no cover - optional dependency during early bootstrap
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover
    class BaseModel:  # type: ignore[override]
        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self) -> dict[str, Any]:
            return dict(self.__dict__)

        @classmethod
        def model_validate(cls, data: Any) -> "BaseModel":
            if isinstance(data, dict):
                return cls(**data)
            if hasattr(data, "model_dump"):
                return cls(**data.model_dump())
            return cls(**dict(data))

    class ConfigDict(dict):  # type: ignore[override]
        pass

    def Field(default: Any = None, *, default_factory: Optional[Any] = None, **_: Any) -> Any:  # type: ignore[override]
        if default_factory is not None:
            return default_factory()
        return default


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stable_int(*parts: Any, modulo: int = 10_000) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    return int(sha256(payload).hexdigest(), 16) % modulo


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{sha256(payload).hexdigest()[:16]}"


@dataclass
class TrainingSample:
    sample_id: str
    sample_type: str
    instruction: str
    chosen: str
    rejected: Optional[str]
    score: float
    source: str
    source_event_ids: list[str]
    source_adapter_version: Optional[str] = None
    preference_kind: Optional[str] = None
    preference_source: Optional[str] = None
    preference_pair_id: Optional[str] = None
    preference_reason: Optional[str] = None
    signal_quality: Any = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    teacher_provenance: dict[str, Any] = field(default_factory=dict)


class TrainingSampleSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    sample_id: str
    sample_type: str
    instruction: str
    chosen: str
    rejected: Optional[str] = None
    score: float = 0.0
    source: str
    source_event_ids: list[str] = Field(default_factory=list)
    source_adapter_version: Optional[str] = None
    preference_kind: Optional[str] = None
    preference_source: Optional[str] = None
    preference_pair_id: Optional[str] = None
    preference_reason: Optional[str] = None
    signal_quality: Any = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    teacher_provenance: dict[str, Any] = Field(default_factory=dict)


@dataclass
class DistillationConfig:
    teacher_model: str = "mock-teacher"
    teacher_prompt_version: str = "v1"
    temperature: float = 0.7
    max_samples: int = 200
    rewrite_weak_samples: bool = True
    generate_dpo_pairs: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    score_threshold: float = 0.3
    dedup_similarity: float = 0.92
    seed: int = 42
    scenario: str = "life-coach"
    style: str = "温和、共情"
    # Profile-aware curation settings
    user_id: Optional[str] = None
    enable_profile_prioritization: bool = True
    profile_boost_factor: float = 0.2  # Boost score for profile-matching samples
    # Prada difference-aware filtering
    prada_similarity_threshold: float = 0.85
    prada_enabled: bool = True
    local_teacher_model: str = ""


class DistillationConfigSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    teacher_model: str = "mock-teacher"
    teacher_prompt_version: str = "v1"
    temperature: float = 0.7
    max_samples: int = 200
    rewrite_weak_samples: bool = True
    generate_dpo_pairs: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    score_threshold: float = 0.3
    dedup_similarity: float = 0.92
    seed: int = 42
    scenario: str = "life-coach"
    style: str = "温和、共情"
    prada_similarity_threshold: float = 0.85
    prada_enabled: bool = True
    local_teacher_model: str = ""


@dataclass
class RawSignal:
    signal_id: str
    source_event_id: str
    request_id: str
    session_id: str
    adapter_version: Optional[str]
    event_type: str
    timestamp: datetime
    context: str
    model_output: str
    user_action: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    event_chain_ids: list[str] = field(default_factory=list)


@dataclass
class SignalSampleCurationResult:
    signal_id: str
    sample: Optional[TrainingSample]
    reason: str


@dataclass
class SignalSampleCuratorConfig:
    allowed_event_types: tuple[str, ...] = ("accept", "copy")
    preference_event_types: tuple[str, ...] = ("edit", "reject")
    score_threshold: float = 0.3
    minimum_signal_confidence: float = 0.65
    reject_conflicted_signal_quality: bool = True
    require_complete_chain: bool = True
    explicit_preference_reinforcement_enabled: bool = True
    explicit_preference_score_boost: float = 0.08


def _normalize_event_chain_ids(signal: RawSignal) -> list[str]:
    chain: list[str] = []
    for candidate in [signal.source_event_id, *list(signal.event_chain_ids or [])]:
        candidate = str(candidate or "").strip()
        if not candidate or candidate in chain:
            continue
        chain.append(candidate)
    return chain


def _signal_chain_is_complete(signal: RawSignal) -> bool:
    if not signal.request_id or not signal.session_id or not signal.source_event_id:
        return False
    chain = _normalize_event_chain_ids(signal)
    if len(chain) < 2:
        return False
    return signal.source_event_id in chain


def _signal_event_type_is_reliable(signal: RawSignal, config: SignalSampleCuratorConfig) -> bool:
    return signal.event_type in config.allowed_event_types


def _signal_preference_event_type_is_reliable(signal: RawSignal, config: SignalSampleCuratorConfig) -> bool:
    return signal.event_type in config.preference_event_types


def _signal_dataset_split(signal: RawSignal) -> str:
    metadata_split = normalize_dataset_split(signal.metadata.get("dataset_split"))
    if metadata_split == "test":
        return metadata_split
    return "train"


def _signal_quality_and_reasons(
    signal: RawSignal,
    config: SignalSampleCuratorConfig,
) -> tuple[Any, list[str]]:
    signal_quality = build_signal_quality(signal)
    reasons = signal_quality_filter_reasons(
        signal_quality,
        minimum_confidence=config.minimum_signal_confidence,
        reject_conflicted_signal_quality=config.reject_conflicted_signal_quality,
    )
    return signal_quality, reasons


def _signal_text(value: Any) -> str:
    return str(value or "").strip()


def _explicit_user_data_routing(signal: RawSignal) -> dict[str, Any]:
    routing = signal.metadata.get("explicit_user_data_routing")
    return dict(routing) if isinstance(routing, dict) else {}


def _explicit_response_preference_candidates(signal: RawSignal) -> list[dict[str, Any]]:
    routing = _explicit_user_data_routing(signal)
    candidates = routing.get("candidates") or []
    if not isinstance(candidates, list):
        return []
    return [
        dict(candidate)
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("kind") or "") == "response_preference"
    ]


def _explicit_response_preference_values(signal: RawSignal) -> list[str]:
    values: list[str] = []
    for candidate in _explicit_response_preference_candidates(signal):
        value = str(candidate.get("value") or "").strip()
        if value and value not in values:
            values.append(value)
    return values


def _explicit_response_preference_reinforced(
    signal: RawSignal,
    *,
    sample_type: str,
    reply_style: str,
    config: SignalSampleCuratorConfig,
) -> bool:
    if not config.explicit_preference_reinforcement_enabled:
        return False
    if not _explicit_response_preference_candidates(signal):
        return False
    if sample_type == "sft":
        return reply_style == "accepted"
    if sample_type == "dpo":
        return reply_style in {"edited", "rejected"}
    return False


def _signal_instruction_text(signal: RawSignal) -> str:
    return _signal_text(signal.user_action.get("final_text") or signal.context)


def _signal_sft_texts(signal: RawSignal) -> tuple[str, str]:
    instruction = _signal_instruction_text(signal)
    chosen = _signal_text(signal.user_action.get("accepted_text") or signal.model_output)
    return instruction, chosen


def _signal_preference_texts(signal: RawSignal) -> tuple[str, str, str, str]:
    reply_style = normalize_reply_style(event_type=signal.event_type, user_action=signal.user_action)
    instruction = _signal_instruction_text(signal)
    rejected = _signal_text(
        signal.user_action.get("rejected_text")
        or signal.user_action.get("disliked_text")
        or signal.model_output
    )

    if reply_style == "edited":
        chosen = _signal_text(
            signal.user_action.get("edited_text")
            or signal.user_action.get("final_text")
            or signal.user_action.get("accepted_text")
            or signal.user_action.get("preferred_text")
            or signal.user_action.get("corrected_text")
        )
    elif reply_style == "rejected":
        chosen = _signal_text(
            signal.user_action.get("accepted_text")
            or signal.user_action.get("preferred_text")
            or signal.user_action.get("edited_text")
            or signal.user_action.get("final_text")
            or signal.user_action.get("corrected_text")
        )
    else:
        chosen = ""

    return instruction, chosen, rejected, reply_style


def _signal_curation_reason(signal: RawSignal, config: SignalSampleCuratorConfig, *, sample_type: str) -> str:
    if not signal.request_id or not signal.session_id or not signal.source_event_id:
        return "missing_signal_identity"
    if config.require_complete_chain and not _signal_chain_is_complete(signal):
        return "incomplete_event_chain"
    if sample_type == "sft":
        if not _signal_event_type_is_reliable(signal, config):
            return "unreliable_event_type"
    elif sample_type == "dpo":
        if not _signal_preference_event_type_is_reliable(signal, config):
            return "unreliable_event_type"
    else:
        return "unsupported_sample_type"
    if _signal_dataset_split(signal) == "test":
        return "test_split_not_curated"
    _, quality_reasons = _signal_quality_and_reasons(signal, config)
    if quality_reasons:
        return quality_reasons[0]
    if sample_type == "sft":
        user_text, chosen = _signal_sft_texts(signal)
        if not user_text:
            return "missing_user_text"
        if not chosen:
            return "missing_assistant_text"
    else:
        user_text, chosen, rejected, _reply_style = _signal_preference_texts(signal)
        if not user_text:
            return "missing_user_text"
        if not chosen:
            return "missing_preferred_text"
        if not rejected:
            return "missing_rejected_text"
        if chosen == rejected:
            return "degenerate_preference_pair"
    return "curated_sft" if sample_type == "sft" else "curated_dpo"


def _signal_curation_metadata(
    signal: RawSignal,
    *,
    chain_ids: list[str],
    signal_quality: Any,
    sample_type: str,
    preference_kind: Optional[str] = None,
    preference_source: Optional[str] = None,
    preference_pair_id: Optional[str] = None,
    preference_reason: Optional[str] = None,
    explicit_response_preference_values: Optional[list[str]] = None,
    explicit_response_preference_reinforced: bool = False,
) -> dict[str, Any]:
    metadata = dict(signal.metadata)
    metadata.update(build_signal_provenance(signal))
    metadata.update(
        {
            "source_event_id": signal.source_event_id,
            "source_event_ids": list(chain_ids),
            "request_id": signal.request_id,
            "session_id": signal.session_id,
            "adapter_version": signal.adapter_version,
            "signal_event_type": signal.event_type,
            "signal_timestamp": signal.timestamp.isoformat(),
            "signal_source": "signal",
            "sample_type": sample_type,
            "dataset_split": "train",
            "event_chain_complete": True,
            "event_chain_length": len(chain_ids),
            "event_chain_root_id": chain_ids[0] if chain_ids else signal.source_event_id,
            "event_chain_terminal_id": chain_ids[-1] if chain_ids else signal.source_event_id,
            "signal_quality": asdict(signal_quality),
        }
    )
    if explicit_response_preference_values:
        metadata["explicit_response_preference_values"] = list(explicit_response_preference_values)
        metadata["explicit_response_preference_candidate_count"] = len(explicit_response_preference_values)
    if explicit_response_preference_reinforced:
        metadata["explicit_response_preference_reinforced"] = True
        metadata["training_signal_category"] = "preference_reinforced"
        metadata["training_gate_reason"] = "explicit_response_preference_reinforced_by_feedback"
    if preference_kind:
        metadata["preference_kind"] = preference_kind
    if preference_source:
        metadata["preference_source"] = preference_source
    if preference_pair_id:
        metadata["preference_pair_id"] = preference_pair_id
    if preference_reason:
        metadata["preference_reason"] = preference_reason
    if any(value is not None for value in (preference_kind, preference_source, preference_pair_id, preference_reason)):
        metadata["preference"] = {
            "kind": preference_kind,
            "source": preference_source,
            "pair_id": preference_pair_id,
            "reason": preference_reason,
        }
    return metadata


def _signal_sample_score(
    signal_quality: Any,
    config: SignalSampleCuratorConfig,
    *,
    explicit_response_preference_reinforced: bool = False,
) -> float:
    confidence = float(getattr(signal_quality, "confidence", 0.0) or 0.0)
    if explicit_response_preference_reinforced:
        confidence += config.explicit_preference_score_boost
    return max(config.score_threshold, round(confidence, 3))


def signal_to_sft_sample(
    signal: RawSignal,
    *,
    config: Optional[SignalSampleCuratorConfig] = None,
) -> Optional[TrainingSample]:
    cfg = config or SignalSampleCuratorConfig()
    reason = _signal_curation_reason(signal, cfg, sample_type="sft")
    if reason != "curated_sft":
        return None
    signal_quality, _ = _signal_quality_and_reasons(signal, cfg)
    user_text, chosen = _signal_sft_texts(signal)
    chain_ids = _normalize_event_chain_ids(signal)
    sample_id = _stable_id("signal_sft", signal.signal_id, signal.request_id, signal.session_id)
    explicit_response_preference_values = _explicit_response_preference_values(signal)
    explicit_response_preference_reinforced = _explicit_response_preference_reinforced(
        signal,
        sample_type="sft",
        reply_style="accepted",
        config=cfg,
    )
    metadata = _signal_curation_metadata(
        signal,
        chain_ids=chain_ids,
        signal_quality=signal_quality,
        sample_type="sft",
        preference_kind="accepted",
        preference_source="signal",
        preference_reason="signal_reply_style=accepted",
        explicit_response_preference_values=explicit_response_preference_values,
        explicit_response_preference_reinforced=explicit_response_preference_reinforced,
    )
    return TrainingSample(
        sample_id=sample_id,
        sample_type="sft",
        instruction=user_text,
        chosen=chosen,
        rejected=None,
        score=_signal_sample_score(
            signal_quality,
            cfg,
            explicit_response_preference_reinforced=explicit_response_preference_reinforced,
        ),
        source="signal",
        source_event_ids=list(chain_ids),
        source_adapter_version=signal.adapter_version,
        preference_kind="accepted",
        preference_source="signal",
        preference_reason="signal_reply_style=accepted",
        signal_quality=signal_quality,
        metadata=metadata,
    )


def signal_to_preference_sample(
    signal: RawSignal,
    *,
    config: Optional[SignalSampleCuratorConfig] = None,
) -> Optional[TrainingSample]:
    cfg = config or SignalSampleCuratorConfig()
    reason = _signal_curation_reason(signal, cfg, sample_type="dpo")
    if reason != "curated_dpo":
        return None
    signal_quality, _ = _signal_quality_and_reasons(signal, cfg)
    user_text, chosen, rejected, reply_style = _signal_preference_texts(signal)
    if not user_text or not chosen or not rejected or chosen == rejected:
        return None

    chain_ids = _normalize_event_chain_ids(signal)
    pair_id = _stable_id(
        "signal_dpo",
        signal.signal_id,
        signal.request_id,
        signal.session_id,
        reply_style,
        chosen,
        rejected,
    )
    explicit_response_preference_values = _explicit_response_preference_values(signal)
    explicit_response_preference_reinforced = _explicit_response_preference_reinforced(
        signal,
        sample_type="dpo",
        reply_style=reply_style,
        config=cfg,
    )
    preference_reason = f"signal_reply_style={reply_style}"
    metadata = _signal_curation_metadata(
        signal,
        chain_ids=chain_ids,
        signal_quality=signal_quality,
        sample_type="dpo",
        preference_kind=reply_style,
        preference_source="signal",
        preference_pair_id=pair_id,
        preference_reason=preference_reason,
        explicit_response_preference_values=explicit_response_preference_values,
        explicit_response_preference_reinforced=explicit_response_preference_reinforced,
    )
    return TrainingSample(
        sample_id=pair_id,
        sample_type="dpo",
        instruction=user_text,
        chosen=chosen,
        rejected=rejected,
        score=_signal_sample_score(
            signal_quality,
            cfg,
            explicit_response_preference_reinforced=explicit_response_preference_reinforced,
        ),
        source="signal",
        source_event_ids=list(chain_ids),
        source_adapter_version=signal.adapter_version,
        preference_kind=reply_style,
        preference_source="signal",
        preference_pair_id=pair_id,
        preference_reason=preference_reason,
        signal_quality=signal_quality,
        metadata=metadata,
    )


def curate_signals_to_samples(
    signals: Sequence[RawSignal],
    *,
    config: Optional[SignalSampleCuratorConfig] = None,
) -> tuple[list[TrainingSample], list[SignalSampleCurationResult]]:
    cfg = config or SignalSampleCuratorConfig()
    curated: list[TrainingSample] = []
    results: list[SignalSampleCurationResult] = []

    for signal in signals:
        reason = _signal_curation_reason(signal, cfg, sample_type="sft")
        sample = signal_to_sft_sample(signal, config=cfg)
        if sample is None:
            results.append(
                SignalSampleCurationResult(
                    signal_id=signal.signal_id,
                    sample=None,
                    reason=reason,
                )
            )
            continue

        curated.append(sample)
        results.append(
            SignalSampleCurationResult(
                signal_id=signal.signal_id,
                sample=sample,
                reason=reason,
            )
        )

    return curated, results


def curate_signals_to_preference_samples(
    signals: Sequence[RawSignal],
    *,
    config: Optional[SignalSampleCuratorConfig] = None,
) -> tuple[list[TrainingSample], list[SignalSampleCurationResult]]:
    cfg = config or SignalSampleCuratorConfig()
    curated: list[TrainingSample] = []
    results: list[SignalSampleCurationResult] = []

    for signal in signals:
        reason = _signal_curation_reason(signal, cfg, sample_type="dpo")
        sample = signal_to_preference_sample(signal, config=cfg)
        if sample is None:
            results.append(
                SignalSampleCurationResult(
                    signal_id=signal.signal_id,
                    sample=None,
                    reason=reason,
                )
            )
            continue

        curated.append(sample)
        results.append(
            SignalSampleCurationResult(
                signal_id=signal.signal_id,
                sample=sample,
                reason=reason,
            )
        )

    return curated, results


def _style_keywords(style: str) -> list[str]:
    style = style.lower()
    keywords = ["理解", "感受", "支持", "温和", "共情", "一起", "慢慢", "试着", "可以"]
    if "direct" in style or "直接" in style:
        keywords.extend(["明确", "具体", "步骤"])
    if "empat" in style or "共情" in style:
        keywords.extend(["我能理解", "这听起来"])
    return keywords


def _deterministic_teacher_response(instruction: str, style: str, idx: int) -> str:
    keywords = _style_keywords(style)
    lead = keywords[idx % len(keywords)]
    tone = "温和地"
    return (
        f"{lead}，{tone}回应你的情况："
        f"先承认你现在的感受是合理的，然后给出一个小而具体的下一步。"
        f"你可以先从{instruction[:24] or '当前困扰'}开始，把压力拆小，再慢慢推进。"
    )


def _deterministic_teacher_rejected_response(instruction: str, idx: int) -> str:
    generic = [
        "别想太多，直接行动就行。",
        "这没什么大不了的，继续往前走。",
        "你应该更坚强一点。",
    ]
    return generic[idx % len(generic)] + f" 但这并没有真正回应：{instruction[:18]}。"


def _generation_config(config: DistillationConfig) -> dict[str, Any]:
    return {
        "teacher_model": config.teacher_model,
        "teacher_prompt_version": config.teacher_prompt_version,
        "temperature": config.temperature,
        "max_samples": config.max_samples,
        "rewrite_weak_samples": config.rewrite_weak_samples,
        "generate_dpo_pairs": config.generate_dpo_pairs,
        "scenario": config.scenario,
        "style": config.style,
        "seed": config.seed,
    }


def _build_teacher_provenance(
    *,
    teacher_model: str,
    generation_prompt: str,
    confidence: float,
    was_accepted: bool,
    backend: str = "mock",
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build standardized provenance for teacher-distilled samples.

    All teacher-generated samples must carry this provenance so their origin
    and quality gate status are auditable.
    """
    provenance: dict[str, Any] = {
        "source": "teacher_distillation",
        "teacher_model": teacher_model,
        "generation_prompt": generation_prompt,
        "confidence": round(confidence, 4),
        "was_accepted": was_accepted,
        "backend": backend,
        "generated_at": _now().isoformat(),
    }
    if metadata:
        provenance["extra"] = dict(metadata)
    return provenance


def _teacher_metadata(
    *,
    config: DistillationConfig,
    dataset_split: Optional[str],
    source_event_ids: list[str],
    source_sample_id: Optional[str] = None,
    sample_type: Optional[str] = None,
    preference_kind: Optional[str] = None,
    preference_source: Optional[str] = None,
    preference_pair_id: Optional[str] = None,
    preference_reason: Optional[str] = None,
    teacher_backend: Optional[str] = None,
    local_teacher_similarity: Optional[float] = None,
    prada_filtered: bool = False,
    teacher_provenance: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "teacher_model": config.teacher_model,
        "teacher_prompt_version": config.teacher_prompt_version,
        "generation_config": _generation_config(config),
        "dataset_split": dataset_split,
        "source_event_ids": list(source_event_ids),
        "generated_at": _now().isoformat(),
        "scenario": config.scenario,
        "style": config.style,
    }
    if source_sample_id:
        metadata["source_sample_id"] = source_sample_id
    if sample_type:
        metadata["sample_type"] = sample_type
    if any(value is not None for value in (preference_kind, preference_source, preference_pair_id, preference_reason)):
        metadata["preference"] = {
            "kind": preference_kind,
            "source": preference_source,
            "pair_id": preference_pair_id,
            "reason": preference_reason,
        }
    if preference_kind:
        metadata["preference_kind"] = preference_kind
    if preference_source:
        metadata["preference_source"] = preference_source
    if preference_pair_id:
        metadata["preference_pair_id"] = preference_pair_id
    if preference_reason:
        metadata["preference_reason"] = preference_reason
    if teacher_backend is not None:
        metadata["teacher_backend"] = teacher_backend
    if local_teacher_similarity is not None:
        metadata["local_teacher_similarity"] = round(local_teacher_similarity, 4)
    if prada_filtered:
        metadata["prada_filtered"] = True
    if teacher_provenance is not None:
        metadata["teacher_provenance"] = dict(teacher_provenance)
    return metadata


def _split_bucket_assignments(
    samples: Sequence[TrainingSample],
    *,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> TrainingDataset:
    dataset = split_samples(
        samples,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    dataset.train = [attach_dataset_split(sample, "train") for sample in dataset.train]
    dataset.val = [attach_dataset_split(sample, "val") for sample in dataset.val]
    dataset.test = [attach_dataset_split(sample, "test") for sample in dataset.test]
    return dataset


class TeacherDistiller:
    """Deterministic local teacher used for Phase 0.

    The design keeps the interface ready for a real cloud-backed teacher in
    Phase 1 while providing a local mock that is stable for tests.
    """

    def __init__(
        self,
        config: Optional[DistillationConfig] = None,
        *,
        teacher_client: Optional[Any] = None,
        privacy_config: Optional[Any] = None,
        local_engine: Optional[Any] = None,
    ) -> None:
        self.config = config or DistillationConfig()
        self.teacher_client = teacher_client
        self.privacy_config = privacy_config
        self.local_engine = local_engine

    def _cloud_allowed(self) -> bool:
        if self.privacy_config is None:
            return False
        return (
            getattr(self.privacy_config, "mode", "strict_local") == "cloud_assisted"
            and getattr(self.privacy_config, "allow_teacher_cloud", False) is True
        )

    def _build_teacher_prompt(self, instruction: str, style: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": f"你是一个{style}风格的教练。请用温和、共情的方式回应用户。"},
            {"role": "user", "content": instruction},
        ]

    def _build_rejected_prompt(self, instruction: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "请给出一个简短、直接、忽略情绪背景的回复。"},
            {"role": "user", "content": instruction},
        ]

    def _teacher_response(self, instruction: str, style: str, idx: int) -> str:
        if self.teacher_client is not None:
            backend = getattr(self.teacher_client.config, "backend", "mock")
            if backend == "cloud" and not self._cloud_allowed():
                return _deterministic_teacher_response(instruction, style, idx)
            messages = self._build_teacher_prompt(instruction, style)
            result = self.teacher_client.generate(messages, temperature=self.config.temperature)
            return str(result.get("text", ""))
        return _deterministic_teacher_response(instruction, style, idx)

    def _teacher_rejected_response(self, instruction: str, idx: int) -> str:
        if self.teacher_client is not None:
            backend = getattr(self.teacher_client.config, "backend", "mock")
            if backend == "cloud" and not self._cloud_allowed():
                return _deterministic_teacher_rejected_response(instruction, idx)
            messages = self._build_rejected_prompt(instruction)
            result = self.teacher_client.generate(messages, temperature=self.config.temperature)
            return str(result.get("text", ""))
        return _deterministic_teacher_rejected_response(instruction, idx)

    def _maybe_prada_filter(
        self,
        instruction: str,
        teacher_output: str,
    ) -> tuple[bool, float, str]:
        if not self.config.prada_enabled or self.local_engine is None:
            return True, 0.0, teacher_output
        local_output = self.local_engine.generate(
            [{"role": "user", "content": instruction}],
            temperature=0.7,
            max_new_tokens=128,
        )
        use_teacher, similarity = should_use_teacher_output(
            local_output,
            teacher_output,
            threshold=self.config.prada_similarity_threshold,
        )
        return use_teacher, similarity, teacher_output

    def _difference_driven_gate(
        self,
        teacher_output: str,
        user_edited_text: str,
        similarity_threshold: float = 0.7,
    ) -> tuple[bool, float]:
        """Difference-driven gate for teacher distillation aligned with user preference.

        Only accept the teacher output when it is sufficiently similar to the
        user-edited text (above ``similarity_threshold``). High similarity means
        the teacher understood the user's preference direction; low similarity means
        the teacher is off-target and the sample should be discarded.
        """
        return should_use_teacher_output_for_user_preference(
            teacher_output,
            user_edited_text,
            similarity_threshold=similarity_threshold,
        )

    def distill_from_scenario(self, scenario: str, style: str, num_samples: int) -> list[TrainingSample]:
        limit = min(num_samples, self.config.max_samples)
        samples: list[TrainingSample] = []
        index = 0
        attempts = 0
        max_attempts = limit * 3

        while len(samples) < limit and attempts < max_attempts:
            attempts += 1
            instruction = (
                f"场景[{scenario}] 第{index + 1}条："
                f"请围绕“{style}”风格，回答一个关于日常困扰或情绪压力的问题。"
            )
            sample_id = _stable_id("teacher_sft", scenario, style, index, self.config.seed)
            source_event_ids = [f"{scenario}:{index}:prompt", f"{scenario}:{index}:response"]
            sample_type = "sft"
            rejected = None
            chosen = self._teacher_response(instruction, style, index)
            teacher_backend = "mock"
            local_teacher_similarity: Optional[float] = None
            prada_filtered = False

            if self.teacher_client is not None and self._cloud_allowed():
                teacher_backend = "cloud"
            elif self.teacher_client is not None:
                teacher_backend = "local"

            if self.config.prada_enabled and self.local_engine is not None:
                use_teacher, similarity, _ = self._maybe_prada_filter(instruction, chosen)
                local_teacher_similarity = similarity
                if not use_teacher:
                    prada_filtered = True
                    index += 1
                    continue

            if self.config.generate_dpo_pairs and index % 4 == 0:
                sample_type = "dpo"
                source_event_ids = [
                    f"{scenario}:{index}:rejected",
                    f"{scenario}:{index}:chosen",
                ]
                rejected = self._teacher_rejected_response(instruction, index)
                chosen = self._teacher_response(instruction, style, index + 1)
                score = 0.8
            else:
                score = 0.72 + (index % 5) * 0.05

            provenance = _build_teacher_provenance(
                teacher_model=self.config.teacher_model,
                generation_prompt=instruction,
                confidence=round(local_teacher_similarity or 0.0, 4),
                was_accepted=not prada_filtered,
                backend=teacher_backend,
                metadata={"prada_filtered": prada_filtered, "local_teacher_similarity": local_teacher_similarity},
            )
            samples.append(
                TrainingSample(
                    sample_id=sample_id,
                    sample_type=sample_type,
                    instruction=instruction,
                    chosen=chosen,
                    rejected=rejected,
                    score=score,
                    source="teacher",
                    source_event_ids=source_event_ids,
                    source_adapter_version=None,
                    metadata=_teacher_metadata(
                        config=replace(self.config, scenario=scenario, style=style),
                        dataset_split=None,
                        source_event_ids=source_event_ids,
                        sample_type=sample_type,
                        preference_kind="teacher_dpo" if sample_type == "dpo" else None,
                        preference_source="teacher" if sample_type == "dpo" else None,
                        preference_pair_id=sample_id if sample_type == "dpo" else None,
                        preference_reason="explicit_teacher_pair" if sample_type == "dpo" else None,
                        teacher_backend=teacher_backend,
                        local_teacher_similarity=local_teacher_similarity,
                        prada_filtered=prada_filtered,
                        teacher_provenance=provenance,
                    ),
                    teacher_provenance=provenance,
                )
            )
            index += 1

        dataset = _split_bucket_assignments(
            samples,
            train_split=self.config.train_split,
            val_split=self.config.val_split,
            test_split=self.config.test_split,
            seed=self.config.seed,
        )
        return dataset.all_samples()

    def distill_from_history(
        self,
        samples: Sequence[TrainingSample],
        *,
        source_label: str = "history",
        allow_test_source: bool = False,
    ) -> list[TrainingSample]:
        usable = []
        for sample in samples:
            split = sample.metadata.get("dataset_split")
            if split == "test" and not allow_test_source:
                continue
            if float(sample.score) < self.config.score_threshold:
                continue
            usable.append(sample)

        usable = deduplicate_samples(
            filter_samples(usable, config=SampleFilterConfig(score_threshold=self.config.score_threshold)),
            similarity_threshold=self.config.dedup_similarity,
        )

        distilled: list[TrainingSample] = []
        for index, sample in enumerate(usable):
            instruction = sample.instruction
            rewritten = self._rewrite_text(sample.chosen, index)
            distill_id = _stable_id("teacher_hist", sample.sample_id, index, self.config.seed)
            source_event_ids = list(sample.source_event_ids) or [sample.sample_id]
            teacher_backend = "mock"
            if self.teacher_client is not None and self._cloud_allowed():
                teacher_backend = "cloud"
            elif self.teacher_client is not None:
                teacher_backend = "local"
            metadata = _teacher_metadata(
                config=self.config,
                dataset_split=sample.metadata.get("dataset_split"),
                source_event_ids=source_event_ids,
                source_sample_id=sample.sample_id,
                sample_type=sample.sample_type,
                preference_kind=sample.preference_kind,
                preference_source=sample.preference_source,
                preference_pair_id=sample.preference_pair_id,
                preference_reason=sample.preference_reason,
                teacher_backend=teacher_backend,
            )
            metadata["source_label"] = source_label
            distilled.append(
                TrainingSample(
                    sample_id=distill_id,
                    sample_type=sample.sample_type,
                    instruction=instruction,
                    chosen=rewritten,
                    rejected=sample.rejected,
                    score=min(0.95, float(sample.score) + 0.1),
                    source="teacher",
                    source_event_ids=source_event_ids,
                    source_adapter_version=sample.source_adapter_version,
                    preference_kind=sample.preference_kind,
                    preference_source=sample.preference_source,
                    preference_pair_id=sample.preference_pair_id,
                    preference_reason=sample.preference_reason,
                    signal_quality=sample.signal_quality,
                    metadata=metadata,
                )
            )

        if not distilled:
            return []

        dataset = _split_bucket_assignments(
            distilled,
            train_split=self.config.train_split,
            val_split=self.config.val_split,
            test_split=self.config.test_split,
            seed=self.config.seed,
        )
        return dataset.all_samples()

    def rewrite_weak_sample(self, sample: TrainingSample) -> TrainingSample:
        rewritten = self._rewrite_text(sample.chosen, 0)
        metadata = deepcopy(sample.metadata)
        teacher_backend = "mock"
        if self.teacher_client is not None and self._cloud_allowed():
            teacher_backend = "cloud"
        elif self.teacher_client is not None:
            teacher_backend = "local"
        metadata.update(
            _teacher_metadata(
                config=self.config,
                dataset_split=sample.metadata.get("dataset_split"),
                source_event_ids=list(sample.source_event_ids) or [sample.sample_id],
                source_sample_id=sample.sample_id,
                teacher_backend=teacher_backend,
            )
        )
        metadata["rewritten_from"] = sample.sample_id
        return replace(
            sample,
            chosen=rewritten,
            score=min(0.98, max(sample.score, 0.55)),
            source="teacher",
            metadata=metadata,
        )

    def build_dpo_pair(self, prompt: str, chosen: str, rejected: str) -> TrainingSample:
        if not prompt.strip() or not chosen.strip() or not rejected.strip():
            raise ValueError("dpo pair requires explicit prompt, chosen, and rejected texts")
        pair_id = _stable_id("teacher_dpo", prompt, chosen, rejected, self.config.seed)
        source_event_ids = [
            f"{pair_id}:prompt",
            f"{pair_id}:rejected",
            f"{pair_id}:chosen",
        ]
        teacher_backend = "mock"
        if self.teacher_client is not None and self._cloud_allowed():
            teacher_backend = "cloud"
        elif self.teacher_client is not None:
            teacher_backend = "local"
        return TrainingSample(
            sample_id=pair_id,
            sample_type="dpo",
            instruction=prompt,
            chosen=chosen,
            rejected=rejected,
            score=0.85,
            source="teacher",
            source_event_ids=source_event_ids,
            source_adapter_version=None,
            preference_kind="teacher_dpo",
            preference_source="teacher",
            preference_pair_id=pair_id,
            preference_reason="explicit_teacher_pair",
            metadata=_teacher_metadata(
                config=self.config,
                dataset_split=None,
                source_event_ids=source_event_ids,
                sample_type="dpo",
                preference_kind="teacher_dpo",
                preference_source="teacher",
                preference_pair_id=pair_id,
                preference_reason="explicit_teacher_pair",
                teacher_backend=teacher_backend,
            ),
        )

    def _rewrite_text(self, text: str, salt: int) -> str:
        if self.teacher_client is not None:
            backend = getattr(self.teacher_client.config, "backend", "mock")
            if backend == "cloud" and not self._cloud_allowed():
                pass
            else:
                messages = [
                    {"role": "system", "content": "请用温和、共情的风格改写以下句子，保持原意但提升表达质量。"},
                    {"role": "user", "content": text},
                ]
                result = self.teacher_client.generate(messages, temperature=self.config.temperature)
                return str(result.get("text", "")).strip() or text
        variants = [
            "我理解你的处境，这里可以先慢一点，给自己一点空间。",
            "先承认自己的感受，再选择一个小步骤去推进，会更稳妥。",
            "你不需要一次解决全部问题，先把当下最重要的一点处理好。",
        ]
        base = variants[salt % len(variants)]
        if text.strip():
            return f"{base} 原句关注点是：{text[:48]}。"
        return base


def _compute_profile_match_score(
    sample: TrainingSample,
    profile_priorities: dict[str, list[str]],
) -> float:
    """Compute how well a sample matches user profile priorities.

    Args:
        sample: Training sample to evaluate
        profile_priorities: Dict with priority_domains, priority_styles, priority_patterns

    Returns:
        Match score between 0 and 1
    """
    instruction = sample.instruction.lower()
    chosen = sample.chosen.lower()
    text = f"{instruction} {chosen}"

    score = 0.0
    total_weights = 0.0

    # Domain matching
    priority_domains = profile_priorities.get("priority_domains", [])
    if priority_domains:
        domain_keywords = {
            "programming": ["code", "programming", "function", "class", "variable", "bug", "debug", "python", "javascript", "代码", "编程"],
            "writing": ["write", "essay", "story", "article", "blog", "写作", "文章", "故事"],
            "learning": ["learn", "tutorial", "course", "study", "学习", "教程", "课程"],
            "analysis": ["analyze", "data", "statistics", "分析", "数据", "统计"],
            "creative": ["creative", "design", "idea", "创意", "设计", "想法"],
            "business": ["business", "product", "market", "商业", "产品", "市场"],
        }
        domain_matches = 0
        for domain in priority_domains:
            keywords = domain_keywords.get(domain, [domain])
            if any(kw in text for kw in keywords):
                domain_matches += 1
        if priority_domains:
            score += (domain_matches / len(priority_domains)) * 0.4
            total_weights += 0.4

    # Style matching (simplified - check for style indicators)
    priority_styles = profile_priorities.get("priority_styles", [])
    if priority_styles:
        style_indicators = {
            "formal": ["please", "thank you", "would you", "could you", "请", "您好", "谢谢"],
            "casual": ["hey", "hi", "yeah", "cool", "嘿", "嗨", "咋"],
            "concise": ["brief", "short", "summary", "简短", "简洁", "概要"],
            "detailed": ["detailed", "comprehensive", "详细", "具体", "深入"],
            "technical": ["technical", "implementation", "技术", "实现", "架构"],
            "non_technical": ["simple", "plain", "简单", "通俗", "易懂"],
        }
        style_matches = 0
        for style in priority_styles:
            indicators = style_indicators.get(style, [style])
            if any(ind in text for ind in indicators):
                style_matches += 1
        if priority_styles:
            score += (style_matches / len(priority_styles)) * 0.3
            total_weights += 0.3

    # Pattern matching
    priority_patterns = profile_priorities.get("priority_patterns", [])
    if priority_patterns:
        pattern_indicators = {
            "likes_examples": ["example", "for instance", "such as", "例子", "示例", "比如"],
            "prefers_direct": ["direct", "straight", "to the point", "直接", "干脆"],
            "wants_reasoning": ["why", "reason", "logic", "为什么", "原因", "逻辑"],
            "prefers_code": ["code", "implement", "function", "代码", "实现"],
            "wants_alternatives": ["alternative", "other", "different", "其他", "另外", "备选"],
            "seeks_validation": ["right", "correct", "对吗", "是否正确", "确认"],
        }
        pattern_matches = 0
        for pattern in priority_patterns:
            indicators = pattern_indicators.get(pattern, [pattern])
            if any(ind in text for ind in indicators):
                pattern_matches += 1
        if priority_patterns:
            score += (pattern_matches / len(priority_patterns)) * 0.3
            total_weights += 0.3

    return score / total_weights if total_weights > 0 else 0.0


def _apply_profile_prioritization(
    samples: list[TrainingSample],
    profile_priorities: dict[str, list[str]],
    boost_factor: float = 0.2,
) -> list[TrainingSample]:
    """Boost scores of samples matching user profile.

    Args:
        samples: List of training samples
        profile_priorities: Profile priorities from user profile
        boost_factor: Score boost factor for matching samples

    Returns:
        Samples with updated scores
    """
    boosted_samples = []
    for sample in samples:
        match_score = _compute_profile_match_score(sample, profile_priorities)
        if match_score > 0:
            # Boost sample score based on profile match
            new_score = min(1.0, sample.score + (match_score * boost_factor))
            boosted_sample = replace(sample, score=new_score)
            # Update metadata to track the boost
            metadata = dict(sample.metadata)
            metadata["profile_match_score"] = round(match_score, 4)
            metadata["profile_boosted"] = True
            boosted_sample = replace(boosted_sample, metadata=metadata)
            boosted_samples.append(boosted_sample)
        else:
            boosted_samples.append(sample)
    return boosted_samples


class TeacherCurator:
    """High-level facade for generating curated teacher data."""

    def __init__(self, config: Optional[DistillationConfig] = None, *, teacher_client: Optional[Any] = None, home: Optional[str] = None) -> None:
        self.config = config or DistillationConfig()
        self.distiller = TeacherDistiller(self.config, teacher_client=teacher_client)
        self.home = home
        self._profile_priorities: Optional[dict[str, list[str]]] = None

    def _load_profile_priorities(self) -> dict[str, list[str]]:
        """Load profile priorities if user_id is configured."""
        if self._profile_priorities is not None:
            return self._profile_priorities

        if not self.config.user_id or not self.config.enable_profile_prioritization:
            return {"priority_domains": [], "priority_styles": [], "priority_patterns": []}

        try:
            store = get_user_profile_store(self.home)
            profile = store.get_profile(self.config.user_id)
            self._profile_priorities = {
                "priority_domains": [d[0] for d in profile.get_top_domains(5)],
                "priority_styles": [s[0] for s in profile.get_top_style_preferences(3)],
                "priority_patterns": [p[0] for p in profile.get_top_interaction_patterns(3)],
            }
        except Exception:
            self._profile_priorities = {"priority_domains": [], "priority_styles": [], "priority_patterns": []}

        return self._profile_priorities

    def generate(self, scenario_template: str, style_description: str, num_samples: int = 200) -> list[TrainingSample]:
        scenario = scenario_template.strip() or self.config.scenario
        style = style_description.strip() or self.config.style
        raw_samples = self.distiller.distill_from_scenario(scenario, style, num_samples)

        # Apply profile prioritization if enabled
        if self.config.enable_profile_prioritization and self.config.user_id:
            profile_priorities = self._load_profile_priorities()
            if any(profile_priorities.values()):  # Only if profile has data
                raw_samples = _apply_profile_prioritization(
                    raw_samples,
                    profile_priorities,
                    boost_factor=self.config.profile_boost_factor,
                )

        filtered = filter_samples(
            raw_samples,
            config=SampleFilterConfig(
                score_threshold=self.config.score_threshold,
                dedup_similarity=self.config.dedup_similarity,
            ),
        )
        unique = deduplicate_samples(filtered, similarity_threshold=self.config.dedup_similarity)
        dataset = build_training_dataset(
            unique,
            train_split=self.config.train_split,
            val_split=self.config.val_split,
            test_split=self.config.test_split,
            similarity_threshold=self.config.dedup_similarity,
            score_threshold=self.config.score_threshold,
            seed=self.config.seed,
        )
        return dataset.all_samples()
