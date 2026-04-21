"""Data routing policy for user facts, preferences, and feedback signals.

This module defines a lightweight policy layer that answers a practical
question for PFE:

"A piece of user-provided information or a feedback signal arrives. Should it
go to memory, profile, prompt context, SFT, DPO, or nowhere?"

The policy intentionally starts conservative:
- stable user facts go to memory/profile first
- response-style preferences are profile-first and become trainable only after
  repeated reinforcement
- ephemeral task context stays in prompt context
- high-risk PII and secrets are never trainable
- reject/regenerate signals are not directly trainable until paired
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Optional


DatumKind = Literal[
    "identity_fact",
    "role_fact",
    "stable_preference",
    "response_preference",
    "ephemeral_context",
    "sensitive_pii",
    "unknown",
]

DataLane = Literal["memory", "profile", "prompt_context", "signal", "discard"]
TrainingTarget = Literal["none", "sft", "dpo", "preference_only", "blocked"]
SignalTarget = Literal["none", "sft_candidate", "dpo_candidate", "dpo_rejected_only", "blocked"]


HIGH_RISK_PII_TYPES = {
    "credit_card",
    "bank_card",
    "id_card",
    "phone",
    "email",
    "address",
    "password",
    "private_key",
    "secret",
    "token",
    "api_key",
    "biometric",
    "health_record",
    "financial_account",
}


_NAME_PATTERNS = (
    re.compile(r"(?:我叫|我的名字是)\s*([^\s,，。.!！?？]+)"),
    re.compile(r"(?:my name is)\s+([A-Za-z][A-Za-z\-'_]{0,31})", re.IGNORECASE),
)

_ROLE_PATTERNS = (
    re.compile(r"(?:我是|我的工作是)\s*([^\s,，。.!！?？]{1,24})"),
    re.compile(
        r"(?:i am|i work as|my job is)\s+(?:an?\s+)?([A-Za-z][A-Za-z\s\-_]{0,31})",
        re.IGNORECASE,
    ),
)

_PREFERENCE_PATTERNS = (
    re.compile(r"(?:我喜欢|我偏好)\s*([^。.!！?？]{1,80})"),
    re.compile(r"(?:我不喜欢|我讨厌)\s*([^。.!！?？]{1,80})"),
)

_RESPONSE_PREFERENCE_PATTERNS = (
    re.compile(r"(?:我希望你|希望你|请你以后)\s*([^。.!！?？]{1,120})"),
    re.compile(r"(?:希望模型|希望助手)\s*([^。.!！?？]{1,120})"),
    re.compile(r"(?:please be|i want you to be)\s*([^.!?]{1,120})", re.IGNORECASE),
)


@dataclass(frozen=True)
class UserDatum:
    """Structured piece of user-provided information."""

    key: str
    value: str
    kind: DatumKind
    source: Literal["explicit", "implicit", "inference"] = "explicit"
    confidence: float = 1.0
    pii_types: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataRoutingDecision:
    """Decision for routing explicit user data."""

    kind: DatumKind
    primary_lane: DataLane
    additional_lanes: list[DataLane] = field(default_factory=list)
    should_persist_memory: bool = False
    should_update_profile: bool = False
    eligible_for_training: bool = False
    training_target: TrainingTarget = "none"
    requires_human_review: bool = False
    pii_blocked: bool = False
    reason: str = ""


@dataclass(frozen=True)
class SignalTrainingDecision:
    """Decision for routing implicit feedback into training datasets."""

    eligible: bool
    primary_target: SignalTarget
    recommended_sample_type: Optional[Literal["sft", "dpo"]] = None
    requires_pairing: bool = False
    requires_complete_event_chain: bool = False
    reason: str = ""


@dataclass
class PIIAuditReport:
    """Structured report for PII exposure audit over a set of samples."""

    total_samples: int = 0
    pii_detected_count: int = 0
    pii_types_found: dict[str, int] = field(default_factory=dict)
    redacted_samples: list[str] = field(default_factory=list)
    severity: str = "low"  # "low" | "medium" | "high" | "critical"
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "pii_detected_count": self.pii_detected_count,
            "pii_types_found": dict(self.pii_types_found),
            "redacted_samples": list(self.redacted_samples),
            "severity": self.severity,
            "recommendations": list(self.recommendations),
        }


def _compile_pii_patterns() -> dict[str, re.Pattern[str]]:
    """Compile regex patterns for PII detection used by sanitize_for_training."""
    return {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"(?:\+86[\s-]?)?1[3-9]\d{9}"),
        "id_card": re.compile(r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"),
        "credit_card": re.compile(r"(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2})[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"),
        "bank_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4,7}\b"),
        "address": re.compile(r"(?:省|市|区|县|镇|乡|村|街道|路|街|号|楼|单元|室|栋|层|小区|花园|公寓)[\u4e00-\u9fa5\d\s]+"),
        "biometric": re.compile(r"(?:指纹|虹膜|面部识别|人脸识别|声纹|掌纹|DNA|基因序列|biometric|fingerprint|iris|face\s*recognition|voiceprint|palm\s*print)", re.IGNORECASE),
        "health_record": re.compile(r"(?:病历|诊断|处方|医保号|医保卡|社保卡|住院号|门诊号|体检报告|检验单|medical\s*record|diagnosis|prescription|health\s*record)", re.IGNORECASE),
        "financial_account": re.compile(r"(?:银行卡|账户|账号|开户行|支行|支付宝|微信支付|余额|转账记录|bank\s*account|account\s*number|financial\s*account|alipay|wechat\s*pay)", re.IGNORECASE),
        "password": re.compile(r"(?:password|密码|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
        "private_key": re.compile(r"(?:private[_-]?key|secret[_-]?key|api[_-]?key|token)\s*[:=]\s*[a-zA-Z0-9+/=]+", re.IGNORECASE),
    }


_PII_PATTERNS = _compile_pii_patterns()


def sanitize_for_training(text: str) -> str:
    """Sanitize text for training by replacing detected PII with [REDACTED_{type}].

    Uses rule-based regex matching.  Patterns are applied in reverse position
    order so replacements do not shift subsequent match positions.
    """
    if not text:
        return text

    findings: list[tuple[int, int, str]] = []
    for pii_type, pattern in _PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append((match.start(), match.end(), pii_type))

    if not findings:
        return text

    # Sort by start position descending so we can replace in-place safely
    findings.sort(key=lambda f: f[0], reverse=True)
    result = text
    for start, end, pii_type in findings:
        result = result[:start] + f"[REDACTED_{pii_type}]" + result[end:]
    return result


def audit_pii_exposure(samples: list[dict[str, Any]]) -> PIIAuditReport:
    """Audit a list of training samples for PII exposure.

    Each sample is expected to contain text fields such as ``instruction``,
    ``input``, ``output``, ``chosen``, ``rejected``, or ``messages``.
    """
    report = PIIAuditReport(total_samples=len(samples))
    if not samples:
        return report

    text_fields = {"instruction", "input", "output", "chosen", "rejected", "conversation", "context", "model_output", "messages"}

    for sample in samples:
        sample_id = str(sample.get("sample_id") or sample.get("id") or "")
        texts: list[str] = []
        for field in text_fields:
            val = sample.get(field)
            if field == "messages" and isinstance(val, list):
                for msg in val:
                    if isinstance(msg, dict):
                        content = msg.get("content") or ""
                        if isinstance(content, str) and content:
                            texts.append(content)
            elif isinstance(val, str) and val:
                texts.append(val)

        detected_types: set[str] = set()
        for text in texts:
            for pii_type, pattern in _PII_PATTERNS.items():
                if pattern.search(text):
                    detected_types.add(pii_type)

        if detected_types:
            report.pii_detected_count += 1
            for dt in detected_types:
                report.pii_types_found[dt] = report.pii_types_found.get(dt, 0) + 1
            if sample_id:
                report.redacted_samples.append(sample_id)

    # Determine severity
    high_risk_found = set(report.pii_types_found.keys()) & HIGH_RISK_PII_TYPES
    if report.pii_detected_count == 0:
        report.severity = "low"
    elif len(high_risk_found) >= 3 or report.pii_detected_count >= 10:
        report.severity = "critical"
    elif len(high_risk_found) >= 1 or report.pii_detected_count >= 5:
        report.severity = "high"
    else:
        report.severity = "medium"

    # Generate recommendations
    if report.severity == "critical":
        report.recommendations.append("Block training until all high-risk PII is removed or redacted.")
    if high_risk_found:
        report.recommendations.append(f"Review high-risk PII types: {sorted(high_risk_found)}.")
    if report.pii_detected_count > 0:
        report.recommendations.append("Run sanitize_for_training() on affected samples before training.")
    if "email" in report.pii_types_found or "phone" in report.pii_types_found:
        report.recommendations.append("Consider stricter collection-time filtering for contact information.")

    return report


def extract_user_data_candidates(message: str) -> list[UserDatum]:
    """Extract explicit user facts and preferences from a message.

    This stays intentionally heuristic and conservative. The goal is to
    surface obvious candidate data points that a higher-level curator can
    inspect, store, or monitor for reinforcement.
    """

    text = (message or "").strip()
    if not text:
        return []

    candidates: list[UserDatum] = []

    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            candidates.append(
                UserDatum(
                    key="名字",
                    value=match.group(1).strip(),
                    kind="identity_fact",
                    source="explicit",
                    confidence=0.9,
                    pii_types=["person_name"],
                )
            )
            break

    for pattern in _ROLE_PATTERNS:
        match = pattern.search(text)
        if match:
            role = match.group(1).strip(" ，,。.!！?？")
            if role:
                candidates.append(
                    UserDatum(
                        key="职业",
                        value=role,
                        kind="role_fact",
                        source="explicit",
                        confidence=0.75,
                    )
                )
                break

    for pattern in _PREFERENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            candidates.append(
                UserDatum(
                    key="长期偏好",
                    value=match.group(1).strip(),
                    kind="stable_preference",
                    source="explicit",
                    confidence=0.75,
                )
            )

    for pattern in _RESPONSE_PREFERENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            preference = match.group(1).strip()
            if preference:
                candidates.append(
                    UserDatum(
                        key="回答风格偏好",
                        value=preference,
                        kind="response_preference",
                        source="explicit",
                        confidence=0.8,
                    )
                )
                break

    return candidates


def route_user_datum(datum: UserDatum) -> DataRoutingDecision:
    """Route a structured user datum into the correct lane."""

    pii_types = {str(value).strip().lower() for value in datum.pii_types if str(value).strip()}
    if datum.kind == "sensitive_pii" or pii_types.intersection(HIGH_RISK_PII_TYPES):
        return DataRoutingDecision(
            kind=datum.kind,
            primary_lane="discard",
            training_target="blocked",
            pii_blocked=True,
            reason="high_risk_pii_or_secret_must_not_enter_training_or_memory",
        )

    if datum.kind in {"identity_fact", "role_fact"}:
        return DataRoutingDecision(
            kind=datum.kind,
            primary_lane="memory",
            additional_lanes=["prompt_context"],
            should_persist_memory=True,
            eligible_for_training=False,
            training_target="none",
            reason="stable_user_fact_belongs_to_memory_not_model_weights",
        )

    if datum.kind == "stable_preference":
        return DataRoutingDecision(
            kind=datum.kind,
            primary_lane="profile",
            additional_lanes=["prompt_context", "signal"],
            should_update_profile=True,
            eligible_for_training=False,
            training_target="preference_only",
            reason="domain_or_taste_preference_should_be_tracked_before_training",
        )

    if datum.kind == "response_preference":
        reinforced = bool(datum.metadata.get("confirmed_by_feedback") or datum.metadata.get("repeated"))
        return DataRoutingDecision(
            kind=datum.kind,
            primary_lane="profile",
            additional_lanes=["prompt_context", "signal"],
            should_update_profile=True,
            eligible_for_training=reinforced,
            training_target="sft" if reinforced else "preference_only",
            reason=(
                "response_style_preference_can_become_trainable_after_repeated_feedback"
                if reinforced
                else "response_style_preference_is_profile_first_until_reinforced"
            ),
        )

    if datum.kind == "ephemeral_context":
        return DataRoutingDecision(
            kind=datum.kind,
            primary_lane="prompt_context",
            additional_lanes=[],
            eligible_for_training=False,
            training_target="none",
            reason="one_off_task_context_should_not_be_persisted_or_trained",
        )

    return DataRoutingDecision(
        kind=datum.kind,
        primary_lane="prompt_context",
        additional_lanes=["signal"],
        eligible_for_training=False,
        training_target="none",
        reason="unknown_data_defaults_to_context_until_more_evidence_arrives",
    )


def route_signal_for_training(signal: Any, *, minimum_confidence: float = 0.7) -> SignalTrainingDecision:
    """Route an implicit signal into SFT/DPO eligibility buckets."""

    signal_type = _signal_type(signal)
    confidence = _signal_confidence(signal)
    if confidence < minimum_confidence:
        return SignalTrainingDecision(
            eligible=False,
            primary_target="none",
            reason="signal_confidence_below_training_threshold",
        )

    if _signal_contains_high_risk_pii(signal):
        return SignalTrainingDecision(
            eligible=False,
            primary_target="blocked",
            reason="high_risk_pii_signal_must_not_enter_training",
        )

    context = _get_value(signal, "context")
    model_output = _get_value(signal, "model_output")
    edited_text = _signal_edited_text(signal)
    event_chain_complete = _signal_event_chain_complete(signal)

    if signal_type in {"accept", "copy"}:
        if context and model_output:
            return SignalTrainingDecision(
                eligible=True,
                primary_target="sft_candidate",
                recommended_sample_type="sft",
                reason="accepted_or_copied_response_is_a_positive_sft_candidate",
            )
        return SignalTrainingDecision(
            eligible=False,
            primary_target="none",
            reason="accepted_signal_missing_context_or_model_output",
        )

    if signal_type == "edit":
        if edited_text and context and model_output and event_chain_complete:
            return SignalTrainingDecision(
                eligible=True,
                primary_target="dpo_candidate",
                recommended_sample_type="dpo",
                requires_complete_event_chain=True,
                reason="edited_response_with_complete_event_chain_is_a_dpo_candidate",
            )
        if edited_text and context:
            return SignalTrainingDecision(
                eligible=True,
                primary_target="sft_candidate",
                recommended_sample_type="sft",
                reason="edited_response_without_complete_pairing_can_fallback_to_sft",
            )
        return SignalTrainingDecision(
            eligible=False,
            primary_target="none",
            reason="edit_signal_missing_edited_text_or_context",
        )

    if signal_type in {"reject", "regenerate"}:
        if context and model_output and event_chain_complete:
            return SignalTrainingDecision(
                eligible=False,
                primary_target="dpo_rejected_only",
                recommended_sample_type="dpo",
                requires_pairing=True,
                requires_complete_event_chain=True,
                reason="reject_or_regenerate_signal_is_negative_evidence_until_paired",
            )
        return SignalTrainingDecision(
            eligible=False,
            primary_target="none",
            requires_pairing=True,
            reason="reject_or_regenerate_signal_without_complete_event_chain_cannot_be_paired",
        )

    return SignalTrainingDecision(
        eligible=False,
        primary_target="none",
        reason="unsupported_signal_type_for_training",
    )


def _signal_type(signal: Any) -> str:
    signal_type = str(_get_value(signal, "signal_type") or "").strip().lower()
    if signal_type:
        return signal_type
    event_type = str(_get_value(signal, "event_type") or "").strip().lower()
    if event_type:
        return event_type
    user_action = _as_mapping(_get_value(signal, "user_action"))
    return str(user_action.get("type") or user_action.get("action") or "").strip().lower()


def _signal_confidence(signal: Any) -> float:
    direct_confidence = _get_value(signal, "confidence")
    if direct_confidence is not None:
        try:
            return float(direct_confidence)
        except Exception:
            pass

    quality = _get_value(signal, "signal_quality")
    if quality is not None:
        confidence = _get_value(quality, "confidence")
        if confidence is not None:
            try:
                return float(confidence)
            except Exception:
                pass

    metadata = _as_mapping(_get_value(signal, "metadata"))
    try:
        return float(metadata.get("confidence") or 0.0)
    except Exception:
        return 0.0


def _signal_event_chain_complete(signal: Any) -> bool:
    quality = _get_value(signal, "signal_quality")
    provenance = _as_mapping(_get_value(quality, "provenance")) if quality is not None else {}
    if provenance:
        return bool(provenance.get("event_chain_complete"))

    request_id = str(_get_value(signal, "request_id") or "").strip()
    session_id = str(_get_value(signal, "session_id") or "").strip()
    source_event_ids = _get_value(signal, "source_event_ids") or []
    return bool(request_id and session_id and len(list(source_event_ids)) >= 2)


def _signal_edited_text(signal: Any) -> str:
    user_action = _as_mapping(_get_value(signal, "user_action"))
    return str(user_action.get("edited_text") or "").strip()


def _signal_contains_high_risk_pii(signal: Any) -> bool:
    metadata = _as_mapping(_get_value(signal, "metadata"))
    pii_types = metadata.get("pii_types") or []
    pii_set = {str(value).strip().lower() for value in pii_types if str(value).strip()}
    return bool(pii_set.intersection(HIGH_RISK_PII_TYPES))


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {key: val for key, val in vars(value).items() if not key.startswith("_")}
    return {}


def _get_value(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
