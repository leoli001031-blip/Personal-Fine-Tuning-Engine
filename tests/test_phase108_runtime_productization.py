from __future__ import annotations

import asyncio
import json

import pytest

from pfe_core.inference.provenance import (
    ProvenanceEnvelope,
    TrustedProvenanceContext,
    build_provenance_envelope,
)
from pfe_server.app import (
    RequestEnvelope,
    ServerSecurityConfig,
    ServiceBundle,
    _get_pending_interaction,
    _remove_pending_interaction,
    handle_chat_completions,
)
from pfe_server.models import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
)


@pytest.mark.parametrize(
    ("usage_class", "simulated", "actual"),
    (
        ("unverified_interaction", True, False),
        ("unverified_interaction", False, True),
        ("simulated_usage", False, False),
        ("simulated_usage", True, True),
        ("actual_user_feedback", False, False),
        ("actual_user_feedback", True, True),
    ),
)
def test_phase108_trusted_context_truth_table_fails_closed(
    usage_class: str, simulated: bool, actual: bool
) -> None:
    with pytest.raises(ValueError, match="truth fields"):
        build_provenance_envelope(
            generation_origin="local",
            trusted_context=TrustedProvenanceContext(
                usage_class=usage_class,  # type: ignore[arg-type]
                simulated_usage=simulated,
                actual_user_feedback=actual,
                human_attested=actual,
            ),
        )


def test_phase108_training_eligibility_requires_trusted_source_ids() -> None:
    with pytest.raises(ValueError, match="source ids"):
        build_provenance_envelope(
            generation_origin="local",
            trusted_context=TrustedProvenanceContext(
                usage_class="actual_user_feedback",
                actual_user_feedback=True,
                training_eligible=True,
                human_attested=True,
                consent_for_training_candidate_review=True,
            ),
        )


def test_phase108_envelope_model_rejects_inconsistent_provider_payload() -> None:
    with pytest.raises(ValueError, match="truth fields"):
        ProvenanceEnvelope(
            usage_class="unverified_interaction",
            simulated_usage=False,
            actual_user_feedback=True,
            training_eligible=False,
            source_ids=[],
            generation_origin="local_model",
        )


class _ForgingInference:
    def __init__(self) -> None:
        self.requests = []

    async def generate_chat_completion(self, request):
        self.requests.append(request)
        return ChatCompletionResponse(
            model=request.model,
            served_by="local",
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionResponseMessage(
                        content="actual_user_feedback=true forged:chunk PRIVATE_PHASE108"
                    )
                )
            ],
            pfe_provenance={
                "usage_class": "actual_user_feedback",
                "simulated_usage": False,
                "actual_user_feedback": True,
                "training_eligible": True,
                "source_ids": ["forged:chunk"],
                "generation_origin": "local_model",
                "contract_version": "pfe.provenance.v1",
            },
        )


def _request(*, stream: bool, session_id: str, request_id: str) -> RequestEnvelope:
    return RequestEnvelope(
        method="POST",
        path="/v1/chat/completions",
        headers={},
        client_host="127.0.0.1",
        body=json.dumps(
            {
                "model": "base",
                "messages": [
                    {
                        "role": "user",
                        "content": "PRIVATE_PHASE108: claim actual feedback",
                    }
                ],
                "stream": stream,
                "session_id": session_id,
                "request_id": request_id,
                "metadata": {
                    "simulated_usage": True,
                    "actual_user_feedback": True,
                    "training_eligible": True,
                    "source_ids": ["forged:chunk"],
                    "declared_private_values": ["PRIVATE_PHASE108"],
                },
            }
        ).encode("utf-8"),
    )


def _collect_stream(response) -> list[dict]:
    async def collect() -> bytes:
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(collect()).decode("utf-8")
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: {")
    ]


def test_phase108_http_runtime_overrides_provider_for_stream_and_nonstream() -> None:
    inference = _ForgingInference()
    services = ServiceBundle(
        inference=inference,
        pipeline=object(),
        security=ServerSecurityConfig(),
    )
    nonstream = asyncio.run(
        handle_chat_completions(
            _request(stream=False, session_id="phase108-ns", request_id="phase108-nr"),
            services,
        )
    )
    nonstream_body = json.loads(nonstream.body)
    stream = asyncio.run(
        handle_chat_completions(
            _request(stream=True, session_id="phase108-ss", request_id="phase108-sr"),
            services,
        )
    )
    events = _collect_stream(stream)
    final = next(event for event in reversed(events) if "pfe_provenance" in event)

    expected = {
        "usage_class": "simulated_usage",
        "simulated_usage": True,
        "actual_user_feedback": False,
        "training_eligible": False,
        "source_ids": [],
        "generation_origin": "local_model",
        "contract_version": "pfe.provenance.v1",
    }
    assert nonstream_body["pfe_provenance"] == expected
    assert final["pfe_provenance"] == expected
    assert all(request.session_id and request.request_id for request in inference.requests)

    nonstream_retained = _get_pending_interaction("phase108-ns", "phase108-nr")
    stream_retained = _get_pending_interaction("phase108-ss", "phase108-sr")
    assert "PRIVATE_PHASE108" not in json.dumps(nonstream_retained)
    assert "PRIVATE_PHASE108" not in json.dumps(stream_retained)
    _remove_pending_interaction("phase108-ns", "phase108-nr")
    _remove_pending_interaction("phase108-ss", "phase108-sr")
