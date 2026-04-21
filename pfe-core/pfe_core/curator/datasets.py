"""Dataset helpers for PFE distillation and training.

These helpers intentionally avoid a hard dependency on the eventual core model
module so the package stays importable while the project skeleton is being
built out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from difflib import SequenceMatcher
from hashlib import sha256
import random
from typing import Any, Iterable, Iterator, Optional, Sequence

from ..pii_detector import PIIDetector, PIIDetectionResult, PIIType
from ..anonymizer import Anonymizer, AnonymizationConfig, AnonymizationStrategy

ALLOWED_DATASET_SPLITS = ("train", "val", "test")
ALLOWED_REPLY_STYLES = ("accepted", "rejected", "edited", "other")


def _get_attr(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _set_metadata(sample: Any, metadata: dict[str, Any]) -> Any:
    if isinstance(sample, dict):
        new_sample = dict(sample)
        new_sample["metadata"] = metadata
        return new_sample
    return replace(sample, metadata=metadata)


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().split())


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _sample_key(sample: Any) -> str:
    instruction = _normalize_text(_get_attr(sample, "instruction", ""))
    chosen = _normalize_text(_get_attr(sample, "chosen", ""))
    rejected = _normalize_text(_get_attr(sample, "rejected", ""))
    sample_type = _get_attr(sample, "sample_type", "")
    source = _get_attr(sample, "source", "")
    return f"{sample_type}|{source}|{instruction}|{chosen}|{rejected}"


def _similarity(left: Any, right: Any) -> float:
    return SequenceMatcher(None, _sample_key(left), _sample_key(right)).ratio()


def _metadata(sample: Any) -> dict[str, Any]:
    metadata = _get_attr(sample, "metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _signal_quality_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            payload = value.model_dump()  # type: ignore[no-untyped-call]
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            return dict(payload)
    if hasattr(value, "__dataclass_fields__"):
        try:
            payload = asdict(value)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            return dict(payload)
    if hasattr(value, "__dict__"):
        return {key: val for key, val in vars(value).items() if not key.startswith("_")}
    return {}


def _sample_signal_quality(sample: Any) -> dict[str, Any]:
    quality = _get_attr(sample, "signal_quality", None)
    if quality is None:
        quality = _metadata(sample).get("signal_quality")
    return _signal_quality_payload(quality)


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, round(value, 3)))


def normalize_dataset_split(value: Any) -> Optional[str]:
    if value is None:
        return None
    split = str(value).strip().lower()
    if split not in ALLOWED_DATASET_SPLITS:
        raise ValueError(f"invalid dataset split: {value!r}")
    return split


def sample_dataset_split(sample: Any) -> Optional[str]:
    metadata = _metadata(sample)
    split = metadata.get("dataset_split")
    if split is None:
        split = _get_attr(sample, "dataset_split", None)
    return normalize_dataset_split(split)


def attach_dataset_split(sample: Any, split: str) -> Any:
    normalized = normalize_dataset_split(split)
    if normalized is None:
        raise ValueError("dataset split cannot be None")

    metadata = _metadata(sample)
    metadata["dataset_split"] = normalized
    return _set_metadata(sample, metadata)


def signal_quality_filter_reasons(
    quality: Any,
    *,
    minimum_confidence: float = 0.7,
    reject_conflicted_signal_quality: bool = True,
    reject_rolled_back_signals: bool = True,
    require_complete_event_chain: bool = True,
) -> list[str]:
    payload = _signal_quality_payload(quality)
    if not payload:
        return []

    reasons: list[str] = []
    try:
        confidence_value = float(payload.get("confidence") or 0.0)
    except Exception:
        confidence_value = 0.0

    if reject_conflicted_signal_quality and bool(payload.get("conflict")):
        conflict_reason = str(payload.get("conflict_reason") or "").strip() or "signal_conflict"
        if conflict_reason == "incomplete_event_chain":
            conflict_reason = ""
        if not conflict_reason:
            conflict_reason = ""
        if conflict_reason:
            reasons.append(conflict_reason)
    if reject_rolled_back_signals and bool(payload.get("rolled_back")):
        reasons.append("rolled_back_signal")
    if require_complete_event_chain and not bool(payload.get("provenance", {}).get("event_chain_complete")):
        reasons.append("incomplete_event_chain")
    if confidence_value < minimum_confidence:
        reasons.append("low_signal_confidence")
    return reasons


def summarize_signal_quality_filters(
    samples: Iterable[Any],
    *,
    minimum_confidence: float = 0.7,
    reject_conflicted_signal_quality: bool = True,
    reject_rolled_back_signals: bool = True,
    require_complete_event_chain: bool = True,
) -> dict[str, Any]:
    evaluated = 0
    passed = 0
    filtered = 0
    filtered_reasons: dict[str, int] = {}
    reply_style_counts: dict[str, int] = {}

    for sample in samples:
        quality = _sample_signal_quality(sample)
        if not quality:
            continue
        evaluated += 1
        reply_style = str(quality.get("reply_style") or "other")
        reply_style_counts[reply_style] = reply_style_counts.get(reply_style, 0) + 1
        reasons = signal_quality_filter_reasons(
            quality,
            minimum_confidence=minimum_confidence,
            reject_conflicted_signal_quality=reject_conflicted_signal_quality,
            reject_rolled_back_signals=reject_rolled_back_signals,
            require_complete_event_chain=require_complete_event_chain,
        )
        if reasons:
            filtered += 1
            for reason in reasons:
                filtered_reasons[reason] = filtered_reasons.get(reason, 0) + 1
            continue
        passed += 1

    return {
        "evaluated_count": evaluated,
        "passed_count": passed,
        "filtered_count": filtered,
        "filtered_reasons": filtered_reasons,
        "reply_style_counts": reply_style_counts,
        "minimum_confidence": minimum_confidence,
        "reject_conflicted_signal_quality": reject_conflicted_signal_quality,
        "reject_rolled_back_signals": reject_rolled_back_signals,
        "require_complete_event_chain": require_complete_event_chain,
    }


def normalize_reply_style(
    value: Any = None,
    *,
    event_type: Any = None,
    user_action: Any = None,
) -> str:
    candidate = _normalize_text(value)
    action = _as_mapping(user_action)
    if not candidate:
        candidate = _normalize_text(action.get("reply_style") or action.get("type") or action.get("action"))
    if not candidate:
        candidate = _normalize_text(event_type)

    candidate = candidate.replace("-", "_")
    mapping = {
        "accept": "accepted",
        "accepted": "accepted",
        "accepted_reply": "accepted",
        "copy": "accepted",
        "reject": "rejected",
        "rejected": "rejected",
        "rejected_reply": "rejected",
        "edit": "edited",
        "edited": "edited",
        "edited_reply": "edited",
        "update": "edited",
    }
    return mapping.get(candidate, "other")


def build_signal_provenance(signal: Any) -> dict[str, Any]:
    source_event_id = str(_get_attr(signal, "source_event_id", None) or _get_attr(signal, "event_id", ""))
    raw_chain = _get_attr(signal, "source_event_ids", None) or _get_attr(signal, "event_chain_ids", None) or []
    chain: list[str] = []
    for candidate in [*list(raw_chain), source_event_id, str(_get_attr(signal, "event_id", "") or "")]:
        candidate = str(candidate or "").strip()
        if not candidate or candidate in chain:
            continue
        chain.append(candidate)

    request_id = str(_get_attr(signal, "request_id", "") or "")
    session_id = str(_get_attr(signal, "session_id", "") or "")
    parent_event_id = _get_attr(signal, "parent_event_id", None)
    adapter_version = _get_attr(signal, "adapter_version", None)
    event_chain_complete = bool(request_id and session_id and source_event_id and len(chain) >= 2 and source_event_id in chain)

    return {
        "request_id": request_id,
        "session_id": session_id,
        "source_event_id": source_event_id,
        "source_event_ids": list(chain),
        "event_chain_ids": list(chain),
        "parent_event_id": parent_event_id,
        "adapter_version": adapter_version,
        "event_chain_root_id": chain[0] if chain else source_event_id,
        "event_chain_terminal_id": chain[-1] if chain else source_event_id,
        "event_chain_length": len(chain),
        "event_chain_complete": event_chain_complete,
    }


def build_signal_quality(
    signal: Any,
    *,
    confidence: float | None = None,
    conflict: bool | None = None,
) -> "SignalQuality":
    from ..models import SignalQuality

    user_action = _as_mapping(_get_attr(signal, "user_action", None))
    event_type = _get_attr(signal, "event_type", None)
    reply_style = normalize_reply_style(
        _get_attr(signal, "reply_style", None),
        event_type=event_type,
        user_action=user_action,
    )
    provenance = build_signal_provenance(signal)
    context = _normalize_text(_get_attr(signal, "context", ""))
    model_output = _normalize_text(_get_attr(signal, "model_output", ""))
    user_input = _normalize_text(_get_attr(signal, "user_input", ""))
    final_text = _normalize_text(user_action.get("final_text") or user_action.get("edited_text") or user_action.get("accepted_text"))
    accepted_text = _normalize_text(user_action.get("accepted_text") or model_output)
    rejected_text = _normalize_text(user_action.get("rejected_text") or user_action.get("disliked_text"))

    confidence_components = {
        "reply_style": reply_style,
        "event_chain_complete": provenance["event_chain_complete"],
        "has_context": bool(context),
        "has_model_output": bool(model_output),
        "has_user_input": bool(user_input),
        "has_final_text": bool(final_text),
        "has_accepted_text": bool(accepted_text),
        "has_rejected_text": bool(rejected_text),
    }

    if confidence is None:
        base = {
            "accepted": 0.88,
            "edited": 0.76,
            "rejected": 0.56,
            "other": 0.5,
        }.get(reply_style, 0.5)
        if provenance["event_chain_complete"]:
            base += 0.08
        else:
            base -= 0.18
        if reply_style == "accepted" and accepted_text:
            base += 0.04
        elif reply_style == "edited" and final_text:
            base += 0.04
        elif reply_style == "rejected" and rejected_text:
            base += 0.04
        if not model_output:
            base -= 0.1
        if not (context or user_input):
            base -= 0.05
        confidence = base

    conflict_reasons: list[str] = []
    if not provenance["event_chain_complete"]:
        conflict_reasons.append("incomplete_event_chain")
    if not provenance["request_id"] or not provenance["session_id"]:
        conflict_reasons.append("missing_identity")
    if not model_output:
        conflict_reasons.append("missing_model_output")
    if reply_style == "edited" and not final_text:
        conflict_reasons.append("missing_edited_text")
    if reply_style == "rejected" and not rejected_text:
        conflict_reasons.append("missing_rejected_text")
    if reply_style == "accepted" and not accepted_text:
        conflict_reasons.append("missing_accepted_text")

    inferred_conflict = bool(conflict_reasons)
    if conflict is not None:
        inferred_conflict = conflict

    return SignalQuality(
        reply_style=reply_style,
        confidence=_clamp_confidence(confidence),
        conflict=inferred_conflict,
        conflict_reason=conflict_reasons[0] if conflict_reasons else None,
        confidence_reason=(
            "accepted_reply" if reply_style == "accepted" else
            "edited_reply" if reply_style == "edited" else
            "rejected_reply" if reply_style == "rejected" else
            "other_reply"
        )
        + ("_with_complete_chain" if provenance["event_chain_complete"] else "_with_incomplete_chain"),
        provenance=provenance,
        details={
            "confidence_components": confidence_components,
            "conflict_reasons": conflict_reasons,
        },
    )


def attach_signal_quality(sample: Any, signal_quality: Any) -> Any:
    if isinstance(sample, dict):
        updated = dict(sample)
        updated["signal_quality"] = signal_quality
        return updated
    if hasattr(sample, "__dataclass_fields__") and "signal_quality" in getattr(sample, "__dataclass_fields__", {}):
        return replace(sample, signal_quality=signal_quality)
    return sample


@dataclass
class SampleFilterConfig:
    score_threshold: float = 0.3
    dedup_similarity: float = 0.92
    allow_test_split: bool = True
    allow_missing_provenance: bool = False
    allow_missing_event_chain: bool = False
    minimum_signal_confidence: float = 0.7
    reject_conflicted_signal_quality: bool = True
    reject_rolled_back_signals: bool = True
    require_complete_event_chain: bool = True


@dataclass
class TrainingDataset:
    train: list[Any] = field(default_factory=list)
    val: list[Any] = field(default_factory=list)
    test: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def all_samples(self) -> list[Any]:
        return [*self.train, *self.val, *self.test]

    def iter_train_samples(self) -> Iterator[Any]:
        yield from self.train

    def replay_buffer_candidates(self) -> list[Any]:
        """Return samples that may be reused for replay.

        Test split is always excluded.
        """

        return replay_buffer_candidates(self)

    def summary(self) -> dict[str, Any]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "total": len(self.all_samples()),
        }


def _valid_teacher_provenance(sample: Any) -> bool:
    if _get_attr(sample, "source", None) != "teacher":
        return True

    metadata = _metadata(sample)
    required = ("teacher_model", "teacher_prompt_version", "generation_config", "dataset_split")
    if not all(metadata.get(key) not in (None, "") for key in required):
        return False

    source_event_ids = metadata.get("source_event_ids") or _get_attr(sample, "source_event_ids", None)
    if not isinstance(source_event_ids, list) or not source_event_ids:
        return False

    split = normalize_dataset_split(metadata.get("dataset_split"))
    return split in {"train", "val", "test"}


def filter_samples(
    samples: Iterable[Any],
    *,
    config: Optional[SampleFilterConfig] = None,
) -> list[Any]:
    cfg = config or SampleFilterConfig()
    filtered: list[Any] = []

    for sample in samples:
        score = float(_get_attr(sample, "score", 0.0) or 0.0)
        if score < cfg.score_threshold:
            continue
        signal_quality = _sample_signal_quality(sample)
        if signal_quality:
            reasons = signal_quality_filter_reasons(
                signal_quality,
                minimum_confidence=cfg.minimum_signal_confidence,
                reject_conflicted_signal_quality=cfg.reject_conflicted_signal_quality,
                reject_rolled_back_signals=cfg.reject_rolled_back_signals,
                require_complete_event_chain=cfg.require_complete_event_chain,
            )
            if reasons:
                continue
        if not cfg.allow_missing_provenance and not _valid_teacher_provenance(sample):
            continue
        source_event_ids = _get_attr(sample, "source_event_ids", None)
        if not cfg.allow_missing_event_chain and (not isinstance(source_event_ids, list) or not source_event_ids):
            continue
        split = sample_dataset_split(sample)
        if split == "test" and not cfg.allow_test_split:
            continue
        filtered.append(sample)

    return filtered


def deduplicate_samples(
    samples: Iterable[Any],
    *,
    similarity_threshold: float = 0.92,
) -> list[Any]:
    unique: list[Any] = []
    seen_keys: set[str] = set()

    for sample in samples:
        key = _sample_key(sample)
        if key in seen_keys:
            continue

        duplicate = False
        for existing in unique:
            if _similarity(sample, existing) >= similarity_threshold:
                duplicate = True
                break
        if duplicate:
            continue

        seen_keys.add(key)
        unique.append(sample)

    return unique


def _split_indices(count: int, train_split: float, val_split: float, test_split: float) -> tuple[int, int, int]:
    if count <= 0:
        return 0, 0, 0

    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError("dataset split ratios must sum to 1.0")

    train_count = int(count * train_split)
    val_count = int(count * val_split)
    test_count = count - train_count - val_count
    return train_count, val_count, test_count


def split_samples(
    samples: Sequence[Any],
    *,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> TrainingDataset:
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)

    train_count, val_count, test_count = _split_indices(len(ordered), train_split, val_split, test_split)
    train = ordered[:train_count]
    val = ordered[train_count : train_count + val_count]
    test = ordered[train_count + val_count : train_count + val_count + test_count]

    return TrainingDataset(train=train, val=val, test=test)


def select_samples_by_split(
    samples: Sequence[Any],
    *,
    allowed_splits: Sequence[str],
) -> list[Any]:
    normalized_allowed = {normalize_dataset_split(split) for split in allowed_splits}
    return [
        sample
        for sample in samples
        if sample_dataset_split(sample) in normalized_allowed
    ]


def select_holdout_samples(
    samples: Sequence[Any],
    *,
    preferred_splits: Sequence[str] = ("test", "val"),
    max_samples: Optional[int] = None,
) -> list[Any]:
    selected: list[Any] = []
    for split in preferred_splits:
        bucket = select_samples_by_split(samples, allowed_splits=[split])
        if bucket:
            selected.extend(bucket)
        if max_samples is not None and len(selected) >= max_samples:
            return selected[:max_samples]
    return selected[:max_samples] if max_samples is not None else selected


def _annotate_split_bucket(bucket: Iterable[Any], split: str) -> list[Any]:
    return [attach_dataset_split(sample, split) for sample in bucket]


def build_training_dataset(
    samples: Sequence[Any],
    *,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    similarity_threshold: float = 0.92,
    score_threshold: float = 0.3,
    seed: int = 42,
    teacher_distillation_config: Optional[Any] = None,
) -> TrainingDataset:
    """Build a training dataset from samples with optional teacher distillation fusion.

    If ``teacher_distillation_config`` is provided and ``enabled`` is True,
    user signals and teacher signals are fused via :class:`TeacherSignalFusion`
    before dataset construction. The fusion plan is recorded in the returned
    dataset metadata.
    """
    from .teacher_fusion import TeacherSignalFusion, TeacherSignalFusionConfig

    working_samples = list(samples)
    fusion_plan: dict[str, Any] = {"enabled": False}

    if teacher_distillation_config is not None:
        enabled = getattr(teacher_distillation_config, "enabled", False)
        if enabled:
            max_teacher_ratio = getattr(teacher_distillation_config, "max_teacher_ratio", 0.3)
            similarity_threshold_td = getattr(teacher_distillation_config, "similarity_threshold", 0.7)
            fusion_cfg = TeacherSignalFusionConfig(
                max_teacher_ratio=max_teacher_ratio,
                similarity_threshold=similarity_threshold_td,
                user_signal_priority=True,
            )
            fuser = TeacherSignalFusion(config=fusion_cfg)
            # Partition samples by source
            user_signals = [s for s in working_samples if _get_attr(s, "source", None) != "teacher"]
            teacher_signals = [s for s in working_samples if _get_attr(s, "source", None) == "teacher"]
            working_samples = fuser.fuse_signals(user_signals, teacher_signals)
            fusion_plan = fuser.get_last_plan()
            fusion_plan["enabled"] = True

    filtered = filter_samples(
        working_samples,
        config=SampleFilterConfig(score_threshold=score_threshold, dedup_similarity=similarity_threshold),
    )
    unique = deduplicate_samples(filtered, similarity_threshold=similarity_threshold)

    already_split = {
        "train": [],
        "val": [],
        "test": [],
    }
    unsplit: list[Any] = []
    for sample in unique:
        split = sample_dataset_split(sample)
        if split in already_split:
            already_split[split].append(sample)
        else:
            unsplit.append(sample)

    if unsplit:
        resplit = split_samples(
            unsplit,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )
        already_split["train"].extend(resplit.train)
        already_split["val"].extend(resplit.val)
        already_split["test"].extend(resplit.test)

    metadata: dict[str, Any] = {
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "dedup_similarity": similarity_threshold,
        "score_threshold": score_threshold,
        "seed": seed,
        "teacher_fusion": fusion_plan,
    }

    return TrainingDataset(
        train=_annotate_split_bucket(already_split["train"], "train"),
        val=_annotate_split_bucket(already_split["val"], "val"),
        test=_annotate_split_bucket(already_split["test"], "test"),
        metadata=metadata,
    )


def replay_buffer_candidates(dataset: TrainingDataset) -> list[Any]:
    """Return train-only samples eligible for replay.

    The `test` split never participates, and only samples explicitly tagged as
    train are eligible even if the in-memory bucket is polluted.
    """

    return [sample for sample in dataset.train if sample_dataset_split(sample) == "train"]


@dataclass
class PIIAnonymizationResult:
    """Result of PII anonymization on a dataset."""
    original_sample: Any
    anonymized_sample: Any
    detection_result: PIIDetectionResult
    had_pii: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "had_pii": self.had_pii,
            "finding_count": len(self.detection_result.findings) if self.detection_result else 0,
            "pii_types": [t.value for t in self.detection_result.pii_types_found] if self.detection_result else [],
        }


class DatasetPIIAnonymizer:
    """Anonymizes PII in training datasets."""

    def __init__(
        self,
        strategy: AnonymizationStrategy = AnonymizationStrategy.REPLACE,
        min_confidence: float = 0.7,
        pii_types: list[PIIType] | None = None,
        salt: str | None = None,
    ):
        """Initialize the dataset PII anonymizer.

        Args:
            strategy: Anonymization strategy to use
            min_confidence: Minimum confidence for PII detection
            pii_types: Specific PII types to detect, or None for all
            salt: Salt for hashing strategy
        """
        self.detector = PIIDetector()
        # Convert string strategy to enum if needed
        if isinstance(strategy, str):
            strategy = AnonymizationStrategy(strategy.lower())
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.pii_types = pii_types
        self.anonymizer = Anonymizer(
            AnonymizationConfig(strategy=strategy, salt=salt)
        )

    def anonymize_sample(
        self,
        sample: Any,
        text_fields: list[str] | None = None,
    ) -> PIIAnonymizationResult:
        """Anonymize PII in a single sample.

        Args:
            sample: The sample to anonymize
            text_fields: Fields to scan for PII (default: ['instruction', 'input', 'output'])

        Returns:
            PIIAnonymizationResult with original and anonymized samples
        """
        fields = text_fields or ['instruction', 'input', 'output', 'chosen', 'rejected']
        sample_dict = dict(sample) if isinstance(sample, dict) else vars(sample)

        anonymized_dict = dict(sample_dict)
        had_pii = False
        combined_findings = []

        for field_name in fields:
            if field_name not in sample_dict:
                continue

            text = sample_dict[field_name]
            if not isinstance(text, str):
                continue

            detection_result = self.detector.detect(
                text,
                pii_types=self.pii_types,
                min_confidence=self.min_confidence,
            )

            if detection_result.has_pii:
                had_pii = True
                combined_findings.extend(detection_result.findings)
                anonymized_text = self.anonymizer.anonymize(text, detection_result)
                anonymized_dict[field_name] = anonymized_text

        # Create detection result with combined findings
        combined_result = PIIDetectionResult(
            text_length=sum(len(anonymized_dict.get(f, '')) for f in fields),
            findings=combined_findings,
            has_pii=had_pii,
            pii_types_found=set(f.pii_type for f in combined_findings),
        )

        # Preserve original type if possible
        if isinstance(sample, dict):
            anonymized_sample = anonymized_dict
        else:
            try:
                anonymized_sample = replace(sample, **anonymized_dict)
            except (TypeError, ValueError):
                anonymized_sample = anonymized_dict

        return PIIAnonymizationResult(
            original_sample=sample,
            anonymized_sample=anonymized_sample,
            detection_result=combined_result,
            had_pii=had_pii,
        )

    def anonymize_dataset(
        self,
        samples: Sequence[Any],
        text_fields: list[str] | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Anonymize PII in an entire dataset.

        Args:
            samples: List of samples to anonymize
            text_fields: Fields to scan for PII

        Returns:
            Tuple of (anonymized_samples, statistics)
        """
        anonymized_samples = []
        stats = {
            "total_samples": len(samples),
            "samples_with_pii": 0,
            "total_findings": 0,
            "pii_type_counts": {},
        }

        for sample in samples:
            result = self.anonymize_sample(sample, text_fields)
            anonymized_samples.append(result.anonymized_sample)

            if result.had_pii:
                stats["samples_with_pii"] += 1
                stats["total_findings"] += len(result.detection_result.findings)

                for pii_type in result.detection_result.pii_types_found:
                    type_key = pii_type.value
                    stats["pii_type_counts"][type_key] = stats["pii_type_counts"].get(type_key, 0) + 1

        return anonymized_samples, stats


def anonymize_training_dataset(
    dataset: TrainingDataset,
    strategy: str = "replace",
    min_confidence: float = 0.7,
) -> tuple[TrainingDataset, dict[str, Any]]:
    """Anonymize PII in a complete training dataset.

    Args:
        dataset: The training dataset to anonymize
        strategy: Anonymization strategy ("replace", "hash", "mask", "remove")
        min_confidence: Minimum confidence for PII detection

    Returns:
        Tuple of (anonymized_dataset, statistics)
    """
    strategy_enum = AnonymizationStrategy(strategy.lower())
    anonymizer = DatasetPIIAnonymizer(
        strategy=strategy_enum,
        min_confidence=min_confidence,
    )

    # Anonymize each split
    train_anon, train_stats = anonymizer.anonymize_dataset(dataset.train)
    val_anon, val_stats = anonymizer.anonymize_dataset(dataset.val)
    test_anon, test_stats = anonymizer.anonymize_dataset(dataset.test)

    # Combine statistics
    combined_stats = {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "total_samples": train_stats["total_samples"] + val_stats["total_samples"] + test_stats["total_samples"],
        "total_with_pii": train_stats["samples_with_pii"] + val_stats["samples_with_pii"] + test_stats["samples_with_pii"],
        "strategy": strategy,
    }

    # Create new dataset with anonymized samples
    anonymized_dataset = TrainingDataset(
        train=train_anon,
        val=val_anon,
        test=test_anon,
        metadata={
            **dataset.metadata,
            "pii_anonymized": True,
            "pii_anonymization_stats": combined_stats,
        },
    )

    return anonymized_dataset, combined_stats


def apply_conflict_detection_to_samples(
    samples: Iterable[Any],
    *,
    similarity_threshold: float = 0.75,
    lookback_window_hours: float = 168.0,
) -> list[Any]:
    """Run semantic conflict detection over samples and update their signal_quality.

    This is a thin wrapper around
    ``pfe_core.signal.conflict_detector.apply_conflict_detection`` so that
    callers in the curation pipeline do not need to import the signal module
    directly. It preserves backward compatibility: existing functions such as
    ``signal_quality_filter_reasons`` and ``summarize_signal_quality_filters``
    are unchanged.

    Typical usage after ``build_signal_quality``:

        samples = [build_signal_quality(sig) for sig in raw_signals]
        samples = apply_conflict_detection_to_samples(samples)

    Args:
        samples: Iterable of samples (dicts or dataclasses) that expose
            ``signal_quality`` and signal fields such as ``context``,
            ``timestamp``, and ``event_id``.
        similarity_threshold: Minimum context similarity [0, 1] to treat two
            signals as the same question.
        lookback_window_hours: Only compare against older signals within this
            window. Default is 7 days.

    Returns:
        List of samples with ``signal_quality.conflict`` and
        ``signal_quality.conflict_reason`` updated where semantic conflicts
        were detected.
    """
    from ..signal.conflict_detector import apply_conflict_detection

    return apply_conflict_detection(
        list(samples),
        similarity_threshold=similarity_threshold,
        lookback_window_hours=lookback_window_hours,
    )


__all__ = [
    "ALLOWED_DATASET_SPLITS",
    "ALLOWED_REPLY_STYLES",
    "SampleFilterConfig",
    "SplitResult",
    "TrainingDataset",
    "PIIAnonymizationResult",
    "DatasetPIIAnonymizer",
    "build_signal_quality",
    "filter_samples",
    "deduplicate_samples",
    "split_samples",
    "build_training_dataset",
    "anonymize_training_dataset",
    "replay_buffer_candidates",
    "sample_dataset_split",
    "signal_quality_filter_reasons",
    "summarize_signal_quality_filters",
    "apply_conflict_detection_to_samples",
]
