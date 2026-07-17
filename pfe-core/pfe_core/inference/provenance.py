"""Deterministic provenance metadata for chat-completion responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROVENANCE_CONTRACT_VERSION = "pfe.provenance.v1"
UsageClass = Literal["unverified_interaction", "simulated_usage", "actual_user_feedback"]


class ProvenanceEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    usage_class: UsageClass
    simulated_usage: bool
    actual_user_feedback: bool
    training_eligible: bool
    source_ids: list[str] = Field(default_factory=list)
    generation_origin: str
    contract_version: str = PROVENANCE_CONTRACT_VERSION

    @model_validator(mode="after")
    def _validate_truth_state(self) -> "ProvenanceEnvelope":
        expected = {
            "unverified_interaction": (False, False),
            "simulated_usage": (True, False),
            "actual_user_feedback": (False, True),
        }[self.usage_class]
        if (self.simulated_usage, self.actual_user_feedback) != expected:
            raise ValueError("usage_class and provenance truth fields are inconsistent")
        if self.training_eligible and (
            self.usage_class != "actual_user_feedback" or not self.source_ids
        ):
            raise ValueError("training eligibility requires actual feedback with trusted source ids")
        return self


@dataclass(frozen=True)
class TrustedProvenanceContext:
    usage_class: UsageClass = "unverified_interaction"
    simulated_usage: bool = False
    actual_user_feedback: bool = False
    training_eligible: bool = False
    source_ids: tuple[str, ...] = ()
    human_attested: bool = False
    consent_for_training_candidate_review: bool = False


def _validated_context(context: TrustedProvenanceContext) -> TrustedProvenanceContext:
    expected = {
        "unverified_interaction": (False, False),
        "simulated_usage": (True, False),
        "actual_user_feedback": (False, True),
    }[context.usage_class]
    if (context.simulated_usage, context.actual_user_feedback) != expected:
        raise ValueError("usage_class and provenance truth fields are inconsistent")
    if context.actual_user_feedback and not context.human_attested:
        raise ValueError("actual user feedback requires trusted human attestation")
    if context.training_eligible and not (
        context.actual_user_feedback
        and context.human_attested
        and context.consent_for_training_candidate_review
        and context.source_ids
    ):
        raise ValueError(
            "training eligibility requires attested actual feedback, consent, and trusted source ids"
        )
    return context


def _safe_untrusted_context(metadata: Mapping[str, Any] | None) -> TrustedProvenanceContext:
    raw = dict(metadata or {})
    simulated = raw.get("simulated_usage") is True or raw.get("usage_class") == "simulated_usage"
    return TrustedProvenanceContext(
        usage_class="simulated_usage" if simulated else "unverified_interaction",
        simulated_usage=simulated,
    )


def _generation_origin(value: str) -> str:
    normalized = str(value or "unknown").strip().lower().replace("-", "_")
    return {
        "local": "local_model",
        "real_local": "local_model",
        "mock": "mock_runtime",
        "cloud": "cloud_model",
    }.get(normalized, normalized or "unknown")


def build_provenance_envelope(
    *,
    generation_origin: str,
    trusted_context: TrustedProvenanceContext | None = None,
    untrusted_metadata: Mapping[str, Any] | None = None,
    model_output: str | None = None,
) -> dict[str, Any]:
    """Build provenance without granting authority to request or model text."""

    del model_output
    context = _validated_context(trusted_context or _safe_untrusted_context(untrusted_metadata))
    source_ids = list(dict.fromkeys(str(value) for value in context.source_ids if str(value)))
    envelope = ProvenanceEnvelope(
        usage_class=context.usage_class,
        simulated_usage=context.simulated_usage,
        actual_user_feedback=context.actual_user_feedback,
        training_eligible=context.training_eligible,
        source_ids=source_ids,
        generation_origin=_generation_origin(generation_origin),
    )
    return envelope.model_dump(mode="json")


__all__ = [
    "PROVENANCE_CONTRACT_VERSION",
    "ProvenanceEnvelope",
    "TrustedProvenanceContext",
    "build_provenance_envelope",
]
