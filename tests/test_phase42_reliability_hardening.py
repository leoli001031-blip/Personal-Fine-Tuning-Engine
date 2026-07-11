from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.candidate_quality import assess_preference_candidate_quality
from pfe_core.inference.engine import InferenceConfig, InferenceEngine
from pfe_core.server_services import InferenceServiceAdapter
from pfe_core.phase42_reliability_hardening import (
    PHASE42_GENERIC_HOLDOUTS,
    build_phase41_v2_simulated_candidates,
    build_phase42_final_decision,
    evaluate_adapter_generic_holdout,
)
from pfe_core.security.identifiers import safe_user_storage_id
from pfe_core.user_memory import UserMemoryStore
from pfe_core.user_profile import UserProfileStore
from pfe_server.app import (
    RequestEnvelope,
    ServiceBundle,
    handle_chat_completions,
    handle_dashboard_metrics,
    handle_dashboard_signals,
    handle_dashboard_adapters,
    handle_dashboard_health,
    handle_dashboard_training,
)
from pfe_server.auth import ServerSecurityConfig
from pfe_server.models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionUsage,
)


def _quality_report(*, passed: bool) -> dict[str, object]:
    return {
        "kind": "phase42_adapter_holdout_report",
        "passed": passed,
        "holdout": {"count": 20, "passed": passed},
        "training_leakage_detected": not passed,
    }


def _promoted_adapter(tmp_path: Path) -> tuple[AdapterStore, str]:
    store = AdapterStore(home=tmp_path, workspace="phase42")
    created = store.create_training_version(base_model="tiny", training_config={"backend": "peft"})
    version = created["version"]
    save_file(
        {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 2)},
        str(Path(created["path"]) / "adapter_model.safetensors"),
    )
    store.mark_pending_eval(version, num_samples=20)
    store.attach_eval_report(version, {"recommendation": "deploy", "comparison": "improved", "scores": {}})
    store.promote(version)
    return store, version


def test_latest_adapter_is_blocked_until_strict_serving_quality_passes(tmp_path: Path) -> None:
    store, version = _promoted_adapter(tmp_path)

    missing = store.serving_quality_gate(version)
    assert missing["passed"] is False
    assert "serving_quality_report_missing" in missing["reasons"]

    attached = store.attach_serving_quality_report(version, _quality_report(passed=True))
    assert attached["passed"] is True
    assert store.load("latest") == str(store.root / version)


def test_failed_latest_adapter_can_be_archived_without_deleting_artifact(tmp_path: Path) -> None:
    store, version = _promoted_adapter(tmp_path)
    artifact = store.root / version / "adapter_model.safetensors"

    message = store.archive_failed_serving_quality(version, _quality_report(passed=False))

    assert "retained" in message
    assert artifact.exists()
    assert store.current_latest_version() is None
    assert next(row for row in store.list_version_records() if row["version"] == version)["state"] == "archived"


def test_duplicate_phase41_preferences_fail_candidate_quality_gate() -> None:
    rows = [
        {
            "instruction": f"scenario {index}",
            "chosen": "This preferred answer is exactly the same for every scenario.",
            "rejected": "This rejected answer is also repeated.",
        }
        for index in range(24)
    ]

    report = assess_preference_candidate_quality(rows)

    assert report["passed"] is False
    assert report["chosen_unique_ratio"] < 0.7
    assert report["rejected_unique_ratio"] < 0.7


def test_user_storage_ids_cannot_escape_profile_directory(tmp_path: Path) -> None:
    malicious = "../../escaped-profile"
    assert safe_user_storage_id(malicious).startswith("user-")
    memory_path = UserMemoryStore(home=tmp_path)._profile_path(malicious)
    profile_path = UserProfileStore(home=tmp_path)._profile_path(malicious)
    assert memory_path.parent == tmp_path / "profiles"
    assert profile_path.parent == tmp_path / "profiles"


def test_dashboard_data_endpoints_reject_remote_clients() -> None:
    envelope = RequestEnvelope(
        method="GET",
        path="/pfe/dashboard/signals",
        headers={},
        client_host="192.168.1.88",
        body=b"",
    )
    bundle = ServiceBundle(
        inference=object(),
        pipeline=object(),
        security=ServerSecurityConfig(),
    )

    for handler in (
        handle_dashboard_metrics,
        handle_dashboard_training,
        handle_dashboard_signals,
        handle_dashboard_adapters,
        handle_dashboard_health,
    ):
        response = asyncio.run(handler(envelope, bundle))
        assert response.status_code == 403


def test_dashboard_data_endpoints_reject_wrong_api_key(monkeypatch) -> None:
    monkeypatch.setenv("PFE_API_KEY", "correct-key")
    envelope = RequestEnvelope(
        method="GET",
        path="/pfe/dashboard/metrics",
        headers={"authorization": "Bearer wrong-key"},
        client_host="127.0.0.1",
        body=b"",
    )
    bundle = ServiceBundle(
        inference=object(),
        pipeline=object(),
        security=ServerSecurityConfig(auth_mode="api_key_required"),
    )

    response = asyncio.run(handle_dashboard_metrics(envelope, bundle))

    assert response.status_code == 401


def test_openai_stream_returns_role_deltas_finish_reason_and_done() -> None:
    class _Inference:
        async def generate_chat_completion(self, request):
            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionResponseMessage(content="streamed answer"),
                        finish_reason="length",
                    )
                ],
                usage=ChatCompletionUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            )

    envelope = RequestEnvelope(
        method="POST",
        path="/v1/chat/completions",
        headers={},
        client_host="127.0.0.1",
        body=json.dumps(
            {
                "model": "base",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
        ).encode("utf-8"),
    )
    bundle = ServiceBundle(
        inference=_Inference(),
        pipeline=object(),
        security=ServerSecurityConfig(),
    )

    response = asyncio.run(handle_chat_completions(envelope, bundle))

    async def _collect() -> bytes:
        return b"".join([chunk async for chunk in response.body_iterator])

    body = asyncio.run(_collect()).decode("utf-8")
    assert response.media_type == "text/event-stream"
    assert '"role":"assistant"' in body
    assert '"content":"streamed answer"' in body
    assert '"finish_reason":"length"' in body
    assert '"streaming_mode":"buffered_backend"' in body
    assert body.endswith("data: [DONE]\n\n")


def test_openai_stream_cancels_upstream_when_client_disconnects() -> None:
    class _SlowInference:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = False

        async def generate_chat_completion(self, request):
            del request
            self.started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    inference = _SlowInference()
    envelope = RequestEnvelope(
        method="POST",
        path="/v1/chat/completions",
        headers={},
        client_host="127.0.0.1",
        body=json.dumps(
            {
                "model": "base",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
        ).encode("utf-8"),
    )
    bundle = ServiceBundle(
        inference=inference,
        pipeline=object(),
        security=ServerSecurityConfig(),
    )

    async def _disconnect() -> bool:
        response = await handle_chat_completions(envelope, bundle)
        iterator = response.body_iterator
        first = await anext(iterator)
        assert b'"role":"assistant"' in first
        pending = asyncio.create_task(anext(iterator))
        await inference.started.wait()
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)
        return inference.cancelled

    assert asyncio.run(_disconnect()) is True


def test_real_inference_uses_tokenizer_budget_without_128_token_clamp(monkeypatch) -> None:
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 2
        model_max_length = 4096
        truncation_side = "right"

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return " ".join(str(message.get("content") or "") for message in messages)

        def __call__(self, text, *, return_tensors, truncation=False, max_length=None):
            del return_tensors
            count = max(1, len(str(text).split()))
            if truncation and max_length is not None:
                count = min(count, max_length)
            ids = torch.arange(count, dtype=torch.long).unsqueeze(0)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def decode(self, token_ids, **kwargs):
            del token_ids, kwargs
            return "budgeted response"

    class _Model:
        config = SimpleNamespace(max_position_embeddings=4096)

        def __init__(self):
            self.max_new_tokens = None

        def generate(self, input_ids, **kwargs):
            self.max_new_tokens = kwargs["max_new_tokens"]
            generated = torch.ones((1, self.max_new_tokens), dtype=torch.long)
            return torch.cat([input_ids, generated], dim=1)

    monkeypatch.setenv("PFE_MAX_CONTEXT_TOKENS", "4096")
    monkeypatch.setenv("PFE_MAX_OUTPUT_TOKENS", "512")
    model = _Model()
    engine = InferenceEngine(InferenceConfig(base_model="unused"))

    result = engine._generate_real_response(
        [{"role": "user", "content": "token " * 4500}],
        runtime_bundle={
            "tokenizer": _Tokenizer(),
            "model": model,
            "torch": torch,
            "device": "cpu",
            "adapter_loaded": False,
            "adapter_path": None,
            "adapter_reason": None,
        },
        resolved_base_model="unused",
        max_tokens=300,
    )

    assert model.max_new_tokens == 300
    assert result["finish_reason"] == "length"
    assert result["token_budget"]["original_prompt_tokens"] == 4500
    assert result["token_budget"]["prompt_tokens"] == 3796
    assert result["token_budget"]["input_truncated"] is True


def test_long_term_memory_requires_explicit_request_consent() -> None:
    class _Memory:
        def __init__(self):
            self.reads = 0
            self.writes = 0

        def get_profile_for_prompt(self, user_id):
            del user_id
            self.reads += 1
            return "saved preference"

        def extract_facts_from_conversation(self, **kwargs):
            del kwargs
            self.writes += 1

    class _Pipeline:
        def chat_completion(self, **kwargs):
            return {
                "id": "chatcmpl-memory",
                "object": "chat.completion",
                "model": kwargs["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "metadata": {},
            }

    service = InferenceServiceAdapter.__new__(InferenceServiceAdapter)
    service.pipeline = _Pipeline()
    service.user_memory = _Memory()
    service._pending_interactions = {}
    service._get_adapter_for_request = lambda request: (None, {})

    without_consent = ChatCompletionRequest(
        model="base",
        session_id="../../unsafe-session",
        messages=[{"role": "user", "content": "remember this"}],
    )
    response = asyncio.run(service.generate_chat_completion(without_consent))
    assert service.user_memory.reads == 0
    assert service.user_memory.writes == 0
    assert response.metadata["memory"]["explicit_consent"] is False
    assert response.metadata["memory"]["temporary_interaction_retained"] is True

    with_consent = without_consent.model_copy(update={"metadata": {"memory_consent": True}})
    response = asyncio.run(service.generate_chat_completion(with_consent))
    assert service.user_memory.reads == 1
    assert service.user_memory.writes == 1
    assert response.metadata["memory"]["long_term_memory_written"] is True


def test_bad_promoted_adapter_outputs_fail_twenty_prompt_holdout() -> None:
    outputs = [
        {
            "holdout_id": item["id"],
            "response": "未量化零点五B-002",
            "expected_keywords": item["keywords"],
        }
        for item in PHASE42_GENERIC_HOLDOUTS
    ]

    report = evaluate_adapter_generic_holdout(outputs)

    assert report["holdout_count"] == 20
    assert report["passed"] is False
    assert report["training_leakage_detected"] is True
    assert "response_unique_ratio_below_threshold" in report["reasons"]


def test_phase41_v2_candidates_are_diverse_but_remain_simulated_only() -> None:
    review_items = []
    decisions = []
    for index in range(24):
        review_id = f"review-{index:03d}"
        review_items.append(
            {
                "review_item_id": review_id,
                "scenario_id": f"scenario-{index:03d}",
                "review_payload": {
                    "user_goal": f"核对项目状态类型 {index % 4}",
                    "user_correction": "先检查证据",
                    "continuation_request": "继续到明确 decision",
                },
            }
        )
        decisions.append({"review_item_id": review_id, "decision": "prefer_a"})

    manifest = build_phase41_v2_simulated_candidates(
        review_items=review_items,
        review_decisions=decisions,
    )

    assert manifest["status"] == "quality_ready_for_future_manual_review"
    assert manifest["candidate_quality"]["passed"] is True
    assert manifest["candidate_quality"]["chosen_unique_ratio"] >= 0.7
    assert manifest["candidate_quality"]["rejected_unique_ratio"] >= 0.7
    assert manifest["actual_user_feedback_count"] == 0
    assert manifest["actual_product_benefit_claim_allowed"] is False
    assert manifest["manual_training_probe_allowed"] is False


def test_phase42_final_decision_never_claims_product_benefit_or_auto_promotes() -> None:
    decision = build_phase42_final_decision(
        adapter_report={"passed": False, "version": "bad-adapter"},
        lifecycle_decision={"action": "archived", "artifact_retained": True},
        training_attempt={
            "real_training": True,
            "adapter_validation": {"valid": True},
            "execution": {"parameters_updated": True},
        },
        context_smoke={"passed": True},
        hermes_streaming_passed=True,
        security_tests_passed=True,
        full_validation_passed=True,
        phase41_current={"training_candidate_status": "blocked"},
        phase41_v2={
            "status": "quality_ready_for_future_manual_review",
            "candidate_quality": {"passed": True},
        },
    )

    assert decision["status"] == "reliability_gate_passed"
    assert decision["actual_product_benefit_claim_allowed"] is False
    assert decision["auto_training_allowed"] is False
    assert decision["auto_promotion_allowed"] is False
