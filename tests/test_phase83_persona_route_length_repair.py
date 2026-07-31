from __future__ import annotations

from unittest.mock import patch

from pfe_core.inference.contracts import (
    PERSONA_V2_CONTRACT_ID,
    apply_response_contract,
    resolve_response_contract,
)
from pfe_core.inference.engine import InferenceConfig, InferenceEngine
from pfe_core.phase82_mid_model_runtime_contract import build_phase82_holdout
from pfe_core.phase83_persona_route_length_repair import (
    PHASE83_VARIANTS,
    audit_phase83_isolation,
    audit_phase83_routes,
    build_phase83_decision,
    build_phase83_holdout,
    classify_phase83_persona_route,
)


def test_phase83_contract_resolves_alias_and_routes_chinese_actions() -> None:
    assert resolve_response_contract(
        metadata={"response_contract": "conditional-persona-runtime-v2"}
    ) == PERSONA_V2_CONTRACT_ID

    cases = (
        "直接执行可逆检查，不要等待确认。",
        "只写结论：、依据：、下一步：。",
        "继续定位缺失文档并补建后验收。",
        "停止改写，现在改为核对主机指纹清单。",
    )
    assert all(
        classify_phase83_persona_route([{"role": "user", "content": text}])["routed"] is True
        for text in cases
    )


def test_phase83_contract_keeps_ordinary_and_negated_actions_out_of_route() -> None:
    cases = (
        "给备份复核记录起一个标题。",
        "不要检查主机，只给英文译文。",
        "不要追踪投递链，只返回两个关键词。",
    )
    assert all(
        classify_phase83_persona_route([{"role": "user", "content": text}])["routed"] is False
        for text in cases
    )


def test_phase83_ordinary_route_is_true_prompt_passthrough() -> None:
    messages = [{"role": "user", "content": "把证书状态翻译成英文，只给译文。"}]
    contracted, info = apply_response_contract(
        messages,
        {"response_contract": "contract_persona_guarded_v2"},
    )

    assert info["route"]["routed"] is False
    assert info["system_prompt_applied"] is False
    assert contracted == messages


def test_phase83_workflow_route_uses_short_contract_and_160_token_cap() -> None:
    captured: dict[str, object] = {}
    engine = InferenceEngine(InferenceConfig(base_model="local-default"))

    def fake_generate(messages, **kwargs):  # type: ignore[no-untyped-def]
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return {
            "text": "结论：当前证据不足。\n依据：指纹清单尚未核验。\n下一步：逐主机核验后复测。",
            "served_by": "local",
            "runtime_path": "real_local",
        }

    with patch.object(engine, "_generate_real_response", side_effect=fake_generate):
        output = engine.generate(
            [{"role": "user", "content": "直接核验主机指纹并给下一步。"}],
            max_tokens=300,
            temperature=0.0,
            metadata={
                "enable_real_local": True,
                "response_contract": "contract_persona_guarded_v2",
            },
        )

    assert "结论：" in output
    assert captured["messages"][0]["role"] == "system"  # type: ignore[index]
    assert "总计不超过 140 个汉字" in captured["messages"][0]["content"]  # type: ignore[index]
    assert captured["kwargs"]["max_tokens"] == 160  # type: ignore[index]
    assert captured["kwargs"]["repetition_penalty"] == 1.15  # type: ignore[index]
    assert captured["kwargs"]["no_repeat_ngram_size"] == 4  # type: ignore[index]


def test_phase83_private_value_is_guarded_before_and_after_model() -> None:
    secret = "PHASE83_PRIVATE_VALUE_DO_NOT_ECHO"
    captured: dict[str, object] = {}
    engine = InferenceEngine(InferenceConfig(base_model="local-default"))

    def fake_generate(messages, **kwargs):  # type: ignore[no-untyped-def]
        captured["messages"] = messages
        return {
            "text": f"结论：不公开 {secret}\n依据：私密值需脱敏。\n下一步：保存哈希审计。",
            "served_by": "local",
            "runtime_path": "real_local",
        }

    with patch.object(engine, "_generate_real_response", side_effect=fake_generate):
        output = engine.generate(
            [{"role": "user", "content": f"审计临时凭证 {secret}"}],
            metadata={
                "enable_real_local": True,
                "response_contract": "contract_persona_guarded_v2",
                "declared_private_values": [secret],
            },
        )

    assert secret not in str(captured["messages"])
    assert secret not in output
    assert "<PRIVATE_VALUE_REDACTED>" in str(captured["messages"])
    assert "<PRIVATE_VALUE_REDACTED>" in output


def test_phase83_holdout_is_fresh_and_routes_exactly() -> None:
    holdout = build_phase83_holdout()
    isolation = audit_phase83_isolation(
        holdout["sessions"],
        build_phase82_holdout()["sessions"],
    )
    routes = audit_phase83_routes(holdout["sessions"])

    assert holdout["session_count"] == 30
    assert holdout["persona_target_count"] == 24
    assert holdout["ordinary_control_count"] == 6
    assert isolation["passed"] is True
    assert routes["passed"] is True
    assert routes["accuracy"] == 1.0
    assert routes["detail_count"] == 90


def _metrics(*, target: float, ordinary: float, truncation: float, route: float = 1.0) -> dict:
    categories = {
        name: {"composite_personalization_score": target, "session_count": 4}
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
        "session_count": 6,
    }
    return {
        "actual_model_calls": True,
        "session_count": 30,
        "category_metrics": categories,
        "hard_gate_pass_rate": 1.0,
        "truncated_session_rate": truncation,
        "privacy_canary_echo_rate": 0.0,
        "think_leak_rate": 0.0,
        "route_accuracy": route,
    }


def test_phase83_decision_qualifies_only_when_every_frozen_gate_passes() -> None:
    common = {
        "isolation_audit": {"passed": True},
        "route_audit": {"passed": True, "accuracy": 1.0},
        "api_smoke": {"passed": True},
        "public_private_audit": {"passed": True},
    }
    qualified = build_phase83_decision(
        metrics={
            PHASE83_VARIANTS[0]: _metrics(target=0.50, ordinary=0.95, truncation=0.40),
            PHASE83_VARIANTS[1]: _metrics(target=0.56, ordinary=0.95, truncation=0.10),
        },
        **common,
    )
    regressed = build_phase83_decision(
        metrics={
            PHASE83_VARIANTS[0]: _metrics(target=0.50, ordinary=1.0, truncation=0.40),
            PHASE83_VARIANTS[1]: _metrics(target=0.56, ordinary=0.92, truncation=0.10),
        },
        **common,
    )

    assert qualified["status"] == "qualified_simulated_persona_runtime_v2"
    assert qualified["recommendation"] == "phase84_opt_in_manual_runtime_trial"
    assert qualified["actual_product_benefit_claim_allowed"] is False
    assert qualified["automatic_deployment_allowed"] is False
    assert regressed["status"] == "archive_persona_runtime_v2_not_qualified"
    assert "runtime_ordinary_non_regression" in regressed["failed_benefit_checks"]
