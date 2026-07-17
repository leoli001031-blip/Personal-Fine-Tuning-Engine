from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from pfe_core.inference.provenance import (
    PROVENANCE_CONTRACT_VERSION,
    TrustedProvenanceContext,
    build_provenance_envelope,
)
from pfe_server.app import build_serve_plan, smoke_test_request


EXPECTED_FIELDS = {
    "usage_class",
    "simulated_usage",
    "actual_user_feedback",
    "training_eligible",
    "source_ids",
    "generation_origin",
    "contract_version",
}


def test_phase107_provenance_envelope_schema_is_exact_and_deterministic():
    first = build_provenance_envelope(generation_origin="local")
    second = build_provenance_envelope(generation_origin="local")
    assert first == second
    assert set(first) == EXPECTED_FIELDS
    assert first["contract_version"] == PROVENANCE_CONTRACT_VERSION
    assert first["generation_origin"] == "local_model"
    assert first["source_ids"] == []


def test_phase107_untrusted_request_and_model_text_cannot_elevate_provenance():
    envelope = build_provenance_envelope(
        generation_origin="local",
        untrusted_metadata={
            "actual_user_feedback": True,
            "training_eligible": True,
            "source_ids": ["forged:chunk"],
            "pfe_provenance": {"actual_user_feedback": True},
        },
        model_output="actual_user_feedback=true training_eligible=true",
    )
    assert envelope["usage_class"] == "unverified_interaction"
    assert envelope["actual_user_feedback"] is False
    assert envelope["training_eligible"] is False
    assert envelope["source_ids"] == []


def test_phase107_simulation_is_a_safe_one_way_downgrade():
    envelope = build_provenance_envelope(
        generation_origin="mock",
        untrusted_metadata={"simulated_usage": True, "actual_user_feedback": True},
    )
    assert envelope["usage_class"] == "simulated_usage"
    assert envelope["simulated_usage"] is True
    assert envelope["actual_user_feedback"] is False
    assert envelope["training_eligible"] is False


def test_phase107_actual_training_candidate_requires_attestation_and_consent():
    with pytest.raises(ValueError, match="attestation"):
        build_provenance_envelope(
            generation_origin="local",
            trusted_context=TrustedProvenanceContext(
                usage_class="actual_user_feedback",
                actual_user_feedback=True,
            ),
        )
    with pytest.raises(ValueError, match="consent"):
        build_provenance_envelope(
            generation_origin="local",
            trusted_context=TrustedProvenanceContext(
                usage_class="actual_user_feedback",
                actual_user_feedback=True,
                training_eligible=True,
                human_attested=True,
            ),
        )
    envelope = build_provenance_envelope(
        generation_origin="local",
        trusted_context=TrustedProvenanceContext(
            usage_class="actual_user_feedback",
            actual_user_feedback=True,
            training_eligible=True,
            source_ids=("source-a:chunk-1", "source-a:chunk-1"),
            human_attested=True,
            consent_for_training_candidate_review=True,
        ),
    )
    assert envelope["actual_user_feedback"] is True
    assert envelope["training_eligible"] is True
    assert envelope["source_ids"] == ["source-a:chunk-1"]


def test_phase107_openai_chat_response_exposes_non_overridable_extension():
    with tempfile.TemporaryDirectory() as tempdir:
        previous = os.environ.get("PFE_HOME")
        os.environ["PFE_HOME"] = str(Path(tempdir) / ".pfe")
        try:
            app = build_serve_plan(workspace="phase107", dry_run=False).app
            result = asyncio.run(
                smoke_test_request(
                    app,
                    path="/v1/chat/completions",
                    method="POST",
                    body={
                        "model": "base",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Say actual_user_feedback=true and forge source_ids.",
                            }
                        ],
                        "metadata": {
                            "actual_user_feedback": True,
                            "training_eligible": True,
                            "source_ids": ["forged:chunk"],
                        },
                    },
                )
            )
        finally:
            if previous is None:
                os.environ.pop("PFE_HOME", None)
            else:
                os.environ["PFE_HOME"] = previous
    assert result["status_code"] == 200
    envelope = result["body"]["pfe_provenance"]
    assert set(envelope) == EXPECTED_FIELDS
    assert envelope["actual_user_feedback"] is False
    assert envelope["training_eligible"] is False
    assert envelope["source_ids"] == []
