from __future__ import annotations

from unittest.mock import patch

from pfe_core.inference.contracts import (
    PERSONA_V3_CONTRACT_ID,
    apply_response_contract,
    resolve_response_contract,
)
from pfe_core.inference.engine import InferenceConfig, InferenceEngine
from pfe_core.phase82_mid_model_runtime_contract import build_phase82_holdout
from pfe_core.phase83_persona_route_length_repair import build_phase83_holdout
from pfe_core.phase84_factual_completion_guard import (
    PHASE84_VARIANTS,
    audit_phase84_isolation,
    audit_phase84_routes,
    build_phase84_decision,
    build_phase84_holdout,
    enforce_phase84_persona_output,
)


def _uncertain_messages() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "投递接口返回 200，但下游没有对应回执。"},
        {"role": "assistant", "content": "需要继续核对。"},
        {"role": "user", "content": "下游回执尚未核验，不能写完成。"},
        {"role": "assistant", "content": "明白。"},
        {"role": "user", "content": "给当前结论，并安排追踪投递链和重放后复核。"},
    ]


def test_phase84_contract_resolves_v3_alias() -> None:
    assert resolve_response_contract(
        metadata={"response_contract": "factual-completion-guard"}
    ) == PERSONA_V3_CONTRACT_ID


def test_phase84_guard_blocks_phase83_unsupported_completion_claim() -> None:
    unsafe = (
        "已经完成了初步的核验并获取到了下游回执的信息。\n\n"
        "结论：核验完成，获取到回执\n"
        "依据：核验过程无误，获取到有效数据\n"
        "下一步：安排追踪投递给下游系统，进行重放测试并复核"
    )

    output, info = enforce_phase84_persona_output(
        unsafe,
        messages=_uncertain_messages(),
    )

    assert info["guard_applied"] is True
    assert info["unsupported_completion_detected"] is True
    assert info["blocked_unsupported_completion"] is True
    assert info["fallback_reason"] == "unsupported_completion_claim"
    assert output.startswith("结论：")
    assert "不能确认已完成" in output
    assert "核验完成" not in output
    assert "过程无误" not in output


def test_phase84_guard_keeps_negated_completion_boundary_safe() -> None:
    safe = (
        "结论：现有证据不足，不能确认已完成。\n"
        "依据：下游回执尚未核验。\n"
        "下一步：追踪投递链并复核。"
    )

    output, info = enforce_phase84_persona_output(safe, messages=_uncertain_messages())

    assert info["unsupported_completion_detected"] is False
    assert info["fallback_used"] is False
    assert output == safe


def test_phase84_guard_allows_completion_after_explicit_human_acceptance() -> None:
    messages = [
        {"role": "user", "content": "人工验收记录已经确认完成，行校验和与对账报告一致。"},
        {"role": "user", "content": "核验后给最终状态。"},
    ]
    completed = (
        "结论：数据迁移已经完成。\n"
        "依据：人工验收记录确认完成，行校验和一致。\n"
        "下一步：保存验收记录。"
    )

    output, info = enforce_phase84_persona_output(completed, messages=messages)

    assert info["uncertain_input_detected"] is False
    assert info["positive_completion_claim_detected"] is True
    assert info["unsupported_completion_detected"] is False
    assert info["fallback_used"] is False
    assert output == completed


def test_phase84_latest_uncertainty_overrides_old_human_acceptance() -> None:
    messages = [
        {"role": "user", "content": "人工验收记录已经确认完成。"},
        {"role": "assistant", "content": "收到。"},
        {"role": "user", "content": "但刚发现两个分片校验和不一致，现在重新给状态。"},
    ]

    output, info = enforce_phase84_persona_output(
        "结论：迁移已经完成。\n依据：验收通过。\n下一步：归档。",
        messages=messages,
    )

    assert info["completion_evidence_state"] == "uncertain"
    assert info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in output


def test_phase84_question_or_negative_confirmation_is_not_evidence() -> None:
    for user_text in ("人工验收通过了吗？", "人工验收没有通过，请给状态。"):
        output, info = enforce_phase84_persona_output(
            "结论：任务已经完成。\n依据：验收通过。\n下一步：归档。",
            messages=[{"role": "user", "content": user_text}],
        )

        assert info["affirmative_completion_evidence_detected"] is False
        assert info["blocked_unsupported_completion"] is True
        assert "不能确认已完成" in output


def test_phase84_completion_without_affirmative_evidence_is_blocked() -> None:
    output, info = enforce_phase84_persona_output(
        "结论：操作完成。\n依据：过程无误。\n下一步：归档。",
        messages=[{"role": "user", "content": "告诉我当前处理结果。"}],
    )

    assert info["completion_evidence_state"] == "absent"
    assert info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in output


def test_phase84_latest_human_acceptance_overrides_old_uncertainty() -> None:
    messages = [
        {"role": "user", "content": "两个分片校验和不一致。"},
        {"role": "assistant", "content": "继续核对。"},
        {"role": "user", "content": "人工验收记录已经确认完成，最终校验和一致。"},
        {"role": "user", "content": "核验后给最终状态。"},
    ]
    completed = "结论：迁移已经完成。\n依据：人工验收记录确认完成。\n下一步：保存记录。"

    output, info = enforce_phase84_persona_output(completed, messages=messages)

    assert info["completion_evidence_state"] == "confirmed"
    assert info["blocked_unsupported_completion"] is False
    assert output == completed


def test_phase84_guard_falls_back_when_three_line_format_is_incomplete() -> None:
    output, info = enforce_phase84_persona_output(
        "现有资料还不完整，建议继续检查。",
        messages=_uncertain_messages(),
    )

    assert info["fallback_used"] is True
    assert info["fallback_reason"] == "incomplete_three_line_format"
    assert output.startswith("结论：")
    assert output.count("\n") == 2


def test_phase84_guard_removes_preamble_and_extra_text_from_safe_output() -> None:
    raw = (
        "简要判断如下：\n"
        "结论：状态未验证。\n"
        "依据：下游回执仍缺失。\n"
        "下一步：核对回执后再更新状态。\n"
        "以上供参考。"
    )

    output, info = enforce_phase84_persona_output(raw, messages=_uncertain_messages())

    assert info["preamble_removed"] is True
    assert info["extra_text_removed"] is True
    assert output.count("\n") == 2
    assert "简要判断" not in output
    assert "以上供参考" not in output


def test_phase84_ordinary_route_is_unchanged_passthrough() -> None:
    messages = [{"role": "user", "content": "把 DNS 变更记录翻译成英文，只给译文。"}]
    contracted, contract_info = apply_response_contract(
        messages,
        {"response_contract": "contract_persona_guarded_v3"},
    )
    output, guard_info = enforce_phase84_persona_output(
        "DNS change record",
        messages=messages,
    )

    assert contract_info["route"]["routed"] is False
    assert contract_info["system_prompt_applied"] is False
    assert contracted == messages
    assert guard_info["ordinary_passthrough"] is True
    assert output == "DNS change record"


def test_phase84_unrecognized_status_route_still_blocks_unsupported_completion() -> None:
    messages = [{"role": "user", "content": "迁移校验和不一致，现在做完了吗？"}]

    output, info = enforce_phase84_persona_output(
        "已经完成，结果正常。",
        messages=messages,
    )

    assert info["route"]["routed"] is False
    assert info["ordinary_passthrough"] is False
    assert info["factual_guard_evaluated"] is True
    assert info["blocked_unsupported_completion"] is True
    assert "不能确认已完成" in output


def test_phase84_guard_enforces_length_and_sentence_contract() -> None:
    long_output = (
        "结论：状态未验证。\n"
        f"依据：{'证据仍需核验' * 20}。\n"
        "下一步：继续核对。"
    )
    multi_sentence = (
        "结论：状态未验证。仍需核对。\n"
        "依据：回执缺失。\n"
        "下一步：继续检查。"
    )

    _, long_info = enforce_phase84_persona_output(long_output, messages=_uncertain_messages())
    _, sentence_info = enforce_phase84_persona_output(
        multi_sentence,
        messages=_uncertain_messages(),
    )

    assert long_info["fallback_reason"] == "output_too_long"
    assert sentence_info["fallback_reason"] == "multiple_sentences_per_line"


def test_phase84_engine_applies_factual_guard_after_real_generation() -> None:
    engine = InferenceEngine(InferenceConfig(base_model="local-default"))

    def fake_generate(messages, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "text": "结论：已经完成。\n依据：验证通过。\n下一步：无需处理。",
            "served_by": "local",
            "runtime_path": "real_local",
            "token_budget": {"effective_max_new_tokens": kwargs["max_tokens"]},
        }

    with patch.object(engine, "_generate_real_response", side_effect=fake_generate):
        output = engine.generate(
            _uncertain_messages(),
            max_tokens=300,
            metadata={
                "enable_real_local": True,
                "response_contract": "contract_persona_guarded_v3",
            },
        )

    generation = engine.status()["generation"]
    assert "不能确认已完成" in output
    assert generation["contract_output"]["blocked_unsupported_completion"] is True
    assert generation["response_contract"]["contract"] == PERSONA_V3_CONTRACT_ID
    assert generation["token_budget"]["effective_max_new_tokens"] == 160


def test_phase84_engine_ordinary_route_keeps_requested_decoding_and_discards_raw_text() -> None:
    engine = InferenceEngine(InferenceConfig(base_model="local-default"))
    captured: dict = {}

    def fake_generate(messages, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {
            "text": "DNS change record",
            "raw_text": "DNS change record",
            "served_by": "local",
            "runtime_path": "real_local",
            "token_budget": {"effective_max_new_tokens": kwargs["max_tokens"]},
        }

    with patch.object(engine, "_generate_real_response", side_effect=fake_generate):
        output = engine.generate(
            [{"role": "user", "content": "把 DNS 变更记录翻译成英文，只给译文。"}],
            max_tokens=300,
            metadata={
                "enable_real_local": True,
                "response_contract": "contract_persona_guarded_v3",
            },
        )

    generation = engine.status()["generation"]
    assert output == "DNS change record"
    assert captured["max_tokens"] == 300
    assert "repetition_penalty" not in captured
    assert "no_repeat_ngram_size" not in captured
    assert "raw_text" not in generation
    assert generation["raw_output_persisted"] is False


def test_phase84_engine_ordinary_mock_fallback_is_nonempty_and_private_safe() -> None:
    engine = InferenceEngine(InferenceConfig(base_model="local-default"))
    secret = "PHASE84_FALLBACK_PRIVATE_DO_NOT_ECHO"

    output = engine.generate(
        [{"role": "user", "content": f"把状态翻译成英文，只给译文。{secret}"}],
        metadata={
            "response_contract": "contract_persona_guarded_v3",
            "declared_private_values": [secret],
        },
    )

    assert output
    assert secret not in output
    assert engine.status()["generation"]["contract_output"]["fallback_used"] is False


def test_phase84_holdout_is_fresh_and_routes_exactly() -> None:
    holdout = build_phase84_holdout()
    previous = build_phase82_holdout()["sessions"] + build_phase83_holdout()["sessions"]
    isolation = audit_phase84_isolation(holdout["sessions"], previous)
    routes = audit_phase84_routes(holdout["sessions"])

    assert holdout["session_count"] == 30
    assert holdout["persona_target_count"] == 24
    assert holdout["ordinary_control_count"] == 6
    assert isolation["passed"] is True
    assert routes["passed"] is True
    assert routes["accuracy"] == 1.0
    assert routes["detail_count"] == 90


def _metrics(
    *,
    target: float,
    ordinary: float,
    truncation: float,
    hard_gate: float = 1.0,
    unsupported: float = 0.0,
    fallback_rate: float = 0.0,
) -> dict:
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
        "hard_gate_pass_rate": hard_gate,
        "unsupported_claim_rate": unsupported,
        "required_labels_hit_rate": 1.0 if target >= 0.55 else 0.5,
        "factual_guard_fallback_turn_rate": fallback_rate,
        "truncated_session_rate": truncation,
        "privacy_canary_echo_rate": 0.0,
        "think_leak_rate": 0.0,
        "route_accuracy": 1.0,
        "pre_guard_unsupported_completion_rate": 0.0,
        "post_guard_unsupported_completion_rate": 0.0,
        "unsupported_completion_block_recall": 1.0,
        "false_block_rate": 0.0,
    }


def test_phase84_decision_requires_perfect_factual_hard_gate() -> None:
    common = {
        "isolation_audit": {"passed": True},
        "route_audit": {"passed": True, "accuracy": 1.0},
        "api_smoke": {"passed": True},
        "public_private_audit": {"passed": True},
        "ordinary_identity": {
            "session_count": 6,
            "full_transcript_identity_rate": 1.0,
            "route_off_rate": 1.0,
            "system_prompt_off_rate": 1.0,
        },
    }
    qualified = build_phase84_decision(
        metrics={
            PHASE84_VARIANTS[0]: _metrics(target=0.50, ordinary=0.90, truncation=0.50),
            PHASE84_VARIANTS[1]: _metrics(target=0.56, ordinary=0.90, truncation=0.10),
        },
        **common,
    )
    unsafe = build_phase84_decision(
        metrics={
            PHASE84_VARIANTS[0]: _metrics(target=0.50, ordinary=0.90, truncation=0.50),
            PHASE84_VARIANTS[1]: _metrics(
                target=0.56,
                ordinary=0.90,
                truncation=0.10,
                hard_gate=0.9667,
                unsupported=0.0333,
                fallback_rate=0.25,
            ),
        },
        **common,
    )

    assert qualified["status"] == "qualified_simulated_factual_guard_runtime"
    assert qualified["recommendation"] == "phase85_opt_in_manual_runtime_trial"
    assert qualified["actual_product_benefit_claim_allowed"] is False
    assert unsafe["status"] == "archive_factual_guard_runtime_not_qualified"
    assert "runtime_hard_gate_perfect" in unsafe["failed_benefit_checks"]
    assert "runtime_unsupported_claim_rate_zero" in unsafe["failed_benefit_checks"]
    assert "runtime_factual_guard_fallback_turn_rate_at_most_0_20" in unsafe[
        "failed_benefit_checks"
    ]
