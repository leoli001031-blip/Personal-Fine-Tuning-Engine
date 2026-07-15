from __future__ import annotations

import asyncio
from unittest.mock import patch

from pfe_core.inference.contracts import (
    PERSONA_CONTRACT_ID,
    apply_response_contract,
    resolve_response_contract,
)
from pfe_core.inference.engine import InferenceConfig, InferenceEngine
from pfe_core.pipeline import PipelineService
from pfe_core.phase77_private_value_guarded_runtime import build_phase77_holdout
from pfe_core.phase78_persona_internalization_training import build_phase78_holdout
from pfe_core.phase79_cpu_feasible_persona_probe import build_phase79_holdout
from pfe_core.phase80_small_model_failure_taxonomy import build_phase80_holdout
from pfe_core.phase81_trainable_mid_model_selection import build_phase81_holdout
from pfe_core.phase82_mid_model_runtime_contract import (
    PHASE82_VARIANTS,
    audit_phase82_isolation,
    build_phase82_decision,
    build_phase82_holdout,
)
from pfe_server.app import MockInferenceService
from pfe_server.models import ChatCompletionRequest


def test_persona_contract_resolves_aliases_and_routes_workflow() -> None:
    messages = [{"role": "user", "content": "继续检查 adapter 证据，不要等待确认。"}]

    contracted, info = apply_response_contract(
        messages,
        {"response_contract": "conditional-persona-runtime"},
    )

    assert resolve_response_contract(metadata={"response_contract": "persona_guarded"}) == PERSONA_CONTRACT_ID
    assert info["applied"] is True
    assert info["route"]["routed"] is True
    assert contracted[0]["role"] == "system"
    assert "结论：" in str(contracted[0]["content"])


def test_persona_contract_keeps_explicit_ordinary_task_out_of_persona_route() -> None:
    contracted, info = apply_response_contract(
        [{"role": "user", "content": "把 adapter status 翻译成中文，只给译文。"}],
        {"response_contract": "contract_persona_guarded"},
    )

    assert info["route"]["routed"] is False
    assert contracted[0]["role"] == "system"


def test_persona_contract_guards_private_input_and_output_and_freezes_generation() -> None:
    secret = "PHASE82_PRIVATE_VALUE_DO_NOT_ECHO"
    captured: dict[str, object] = {}
    engine = InferenceEngine(InferenceConfig(base_model="local-default"))

    def fake_generate(messages, **kwargs):  # type: ignore[no-untyped-def]
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return {
            "text": f"<think>hidden</think>结论：不要回显 {secret}",
            "served_by": "local",
            "runtime_path": "real_local",
        }

    with patch.object(engine, "_generate_real_response", side_effect=fake_generate):
        output = engine.generate(
            [{"role": "user", "content": f"检查服务，临时口令是 {secret}"}],
            max_tokens=300,
            temperature=0.0,
            metadata={
                "enable_real_local": True,
                "response_contract": "contract_persona_guarded",
                "declared_private_values": [secret],
            },
        )

    serialized_messages = str(captured["messages"])
    kwargs = captured["kwargs"]
    assert secret not in serialized_messages
    assert "<PRIVATE_VALUE_REDACTED>" in serialized_messages
    assert kwargs["max_tokens"] == 128  # type: ignore[index]
    assert kwargs["repetition_penalty"] == 1.15  # type: ignore[index]
    assert kwargs["no_repeat_ngram_size"] == 4  # type: ignore[index]
    assert secret not in output
    assert "<PRIVATE_VALUE_REDACTED>" in output
    assert "<think>" not in output
    generation = engine.status()["generation"]
    assert generation["response_contract"]["route"]["routed"] is True
    assert generation["contract_output"]["output_guard"]["passed"] is True


def test_pipeline_passes_persona_contract_to_engine() -> None:
    service = PipelineService()
    captured: dict[str, object] = {}

    def fake_generate(_engine, messages, **kwargs):  # type: ignore[no-untyped-def]
        captured["messages"] = messages
        captured["metadata"] = kwargs.get("metadata")
        return "结论：继续检查。\n依据：尚未获得完整证据。\n下一步：核对日志。"

    with patch("pfe_core.pipeline.InferenceEngine.generate", new=fake_generate), patch(
        "pfe_core.pipeline.InferenceEngine.status",
        return_value={"served_by": "mock"},
    ):
        payload = service.chat_completion(
            messages=[{"role": "user", "content": "继续检查日志。"}],
            model="base",
            metadata={"response_contract": "contract_persona_guarded"},
        )

    assert captured["metadata"]["response_contract"] == "contract_persona_guarded"  # type: ignore[index]
    assert "继续检查" in payload["choices"][0]["message"]["content"]


def test_mock_service_supports_persona_contract_without_private_echo() -> None:
    secret = "PHASE82_MOCK_PRIVATE_DO_NOT_ECHO"
    request = ChatCompletionRequest(
        model="local",
        response_contract="contract_persona_guarded",
        messages=[{"role": "user", "content": f"检查服务，口令 {secret}"}],
        metadata={"declared_private_values": [secret]},
    )

    response = asyncio.run(MockInferenceService().generate_chat_completion(request))

    assert secret not in response.choices[0].message.content
    assert response.metadata["request_metadata"]["response_contract"] == "contract_persona_guarded"


def _metrics(
    target: float,
    *,
    ordinary: float = 0.9,
    truncated: float = 0.0,
    route_accuracy: float = 1.0,
) -> dict[str, object]:
    categories = {
        name: {"composite_personalization_score": target, "hard_gate_pass_rate": 1.0}
        for name in (
            "evidence_truthfulness",
            "latest_action_switch",
            "provenance_labeling",
            "autonomous_execution",
            "concise_workstyle",
            "privacy_non_echo",
        )
    }
    categories["ordinary_direct"] = {
        "composite_personalization_score": ordinary,
        "hard_gate_pass_rate": 1.0,
    }
    return {
        "actual_model_calls": True,
        "session_count": 30,
        "category_metrics": categories,
        "hard_gate_pass_rate": 1.0,
        "truncated_session_rate": truncated,
        "privacy_canary_echo_rate": 0.0,
        "think_leak_rate": 0.0,
        "route_accuracy": route_accuracy,
    }


def test_phase82_holdout_is_fresh_simulated_and_isolated() -> None:
    holdout = build_phase82_holdout()
    previous = (
        build_phase77_holdout()["sessions"]
        + build_phase78_holdout()["sessions"]
        + build_phase79_holdout()["sessions"]
        + build_phase80_holdout()["sessions"]
        + build_phase81_holdout()["sessions"]
    )
    audit = audit_phase82_isolation(holdout["sessions"], previous)

    assert holdout["session_count"] == 30
    assert holdout["persona_target_count"] == 24
    assert holdout["ordinary_control_count"] == 6
    assert audit["passed"] is True
    assert audit["training_text_overlap"] == []
    assert audit["previous_holdout_text_overlap"] == []
    assert all(row["simulated_usage"] is True for row in holdout["sessions"])
    assert all(row["actual_user_feedback"] is False for row in holdout["sessions"])


def test_phase82_decision_qualifies_only_simulated_runtime_benefit() -> None:
    metrics = {
        "base_api_length_control": _metrics(0.5, truncated=0.3),
        "persona_api_contract": _metrics(0.56, truncated=0.1),
    }
    decision = build_phase82_decision(
        metrics=metrics,
        isolation_audit={"passed": True},
        api_smoke={"passed": True},
        public_private_audit={"passed": True},
    )

    assert set(metrics) == set(PHASE82_VARIANTS)
    assert decision["status"] == "qualified_simulated_persona_runtime_contract"
    assert decision["recommendation"] == "phase83_manual_runtime_contract_trial_pack"
    assert decision["simulated_lab_runtime_benefit"] is True
    assert decision["actual_product_benefit_claim_allowed"] is False
    assert decision["automatic_deployment_allowed"] is False


def test_phase82_decision_holds_when_runtime_gain_does_not_reproduce() -> None:
    metrics = {
        "base_api_length_control": _metrics(0.5, truncated=0.1),
        "persona_api_contract": _metrics(0.52, truncated=0.2),
    }
    decision = build_phase82_decision(
        metrics=metrics,
        isolation_audit={"passed": True},
        api_smoke={"passed": True},
        public_private_audit={"passed": True},
    )

    assert decision["status"] == "hold_runtime_contract_not_reproduced"
    assert "runtime_gain_at_least_0_04" in decision["failed_benefit_checks"]
    assert "runtime_truncation_at_most_0_15" in decision["failed_benefit_checks"]
