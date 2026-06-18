from __future__ import annotations

from pathlib import Path

import pytest

from pfe_core.phase3_signal_loop import (
    DEFAULT_PERSONA,
    DEFAULT_SCENARIO,
    Phase3SignalLoopStore,
    PersonaSpec,
    ScenarioSpec,
    SignalInboxItem,
    route_signal_item,
    signal_item_from_feedback,
    training_sample_from_signal,
)


def test_persona_and_scenario_schema_validate_required_fields() -> None:
    persona = PersonaSpec.from_dict(DEFAULT_PERSONA.to_dict())
    scenario = ScenarioSpec.from_dict(DEFAULT_SCENARIO.to_dict())

    assert persona.persona_id == "ops-analyst"
    assert persona.goals
    assert persona.evaluation_criteria
    assert scenario.scenario_id == "contract-risk-summary"
    assert scenario.human_review_required is True
    assert "法律结论" in " ".join(scenario.risk_boundaries)

    with pytest.raises(ValueError):
        PersonaSpec.from_dict({**DEFAULT_PERSONA.to_dict(), "goals": []})

    with pytest.raises(ValueError):
        ScenarioSpec.from_dict({**DEFAULT_SCENARIO.to_dict(), "risk_boundaries": []})


def test_signal_routing_marks_accept_and_correction_as_training_candidates() -> None:
    accept = signal_item_from_feedback(
        action="accept",
        user_input="请按资料整理合同交付条款。",
        model_output="摘要：交付期为 7 日；风险：违约金较高，需人工确认。",
        confidence=0.9,
    )
    correction = signal_item_from_feedback(
        action="edit",
        user_input="请整理竞业限制条款。",
        model_output="这条款完全合法。",
        edited_text="摘要：涉及竞业限制；风险：期限和补偿需人工确认。本输出不是法律结论。",
        confidence=0.88,
    )

    assert accept.eligible_for_training is True
    assert accept.route is not None
    assert accept.route.training_target == "sft_candidate"
    assert correction.signal_type == "correction"
    assert correction.eligible_for_training is True
    assert correction.route is not None
    assert "memory" in correction.route.lanes
    assert "training_candidate" in correction.route.lanes


def test_signal_routing_keeps_preference_profile_first_and_reject_review_only() -> None:
    preference = signal_item_from_feedback(
        action="preference",
        user_input="以后整理合同时先给摘要，再给风险点。",
        model_output="",
        user_feedback="先摘要，再风险点，最后人工确认项。",
        confidence=0.9,
    )
    reject = signal_item_from_feedback(
        action="reject",
        user_input="请整理合同。",
        model_output="可以直接签，没有风险。",
        confidence=0.95,
    )

    assert preference.eligible_for_training is False
    assert preference.route is not None
    assert preference.route.lanes == ["profile"]
    assert preference.route.training_target == "preference_only"
    assert reject.eligible_for_training is False
    assert reject.route is not None
    assert reject.route.training_target == "dpo_rejected_only"
    assert reject.route.excluded_reason == "requires_positive_pair"


def test_signal_routing_blocks_safety_high_risk_domain_and_pii() -> None:
    safety = signal_item_from_feedback(
        action="safety_block",
        user_input="请给出确定法律意见。",
        model_output="",
        confidence=0.99,
    )
    high_risk = signal_item_from_feedback(
        action="accept",
        user_input="这个合同能不能稳赢？",
        model_output="一定稳赢。",
        metadata={"risk_flags": ["legal_advice"]},
        confidence=0.9,
    )
    pii = signal_item_from_feedback(
        action="accept",
        user_input="我的手机号是 13800000000，请写进合同摘要。",
        model_output="手机号 13800000000 已记录。",
        confidence=0.9,
    )

    assert safety.route is not None
    assert safety.route.excluded_reason == "safety_block"
    assert safety.route.requires_human_review is True
    assert high_risk.route is not None
    assert high_risk.route.excluded_reason == "high_risk_domain_decision"
    assert high_risk.route.requires_human_review is True
    assert pii.route is not None
    assert pii.route.excluded_reason == "detected_high_risk_pii"
    assert pii.eligible_for_training is False


def test_candidate_sample_generation_sanitizes_and_preserves_source() -> None:
    item = signal_item_from_feedback(
        action="edit",
        user_input="请整理合同。邮箱 buyer@example.com 需要隐藏。",
        model_output="原文保留邮箱 buyer@example.com。",
        edited_text="摘要：邮箱 [REDACTED_email] 需隐藏；风险：人工确认。",
        confidence=0.9,
    )
    sample = training_sample_from_signal(item)

    assert sample["source_signal_id"] == item.signal_id
    assert sample["sample_type"] == "sft"
    assert "buyer@example.com" not in sample["input"]
    assert sample["rejected"]


def test_phase3_store_lists_filters_and_builds_candidate_plan(tmp_path: Path) -> None:
    store = Phase3SignalLoopStore(home=tmp_path, workspace="demo")
    accepted = store.ingest_feedback(
        action="accept",
        user_input="请整理合同付款条款。",
        model_output="摘要：付款节点明确；风险：逾期责任需人工确认。",
        confidence=0.9,
    )
    store.ingest_feedback(
        action="preference",
        user_input="我希望先摘要再风险点。",
        model_output="",
        user_feedback="先摘要，再风险点。",
        confidence=0.9,
    )

    all_signals = store.list_signals(limit=10)
    candidates = store.list_signals(eligible_for_training=True, limit=10)
    plan = store.build_candidate_plan(limit=10)
    summary = store.summary()

    assert len(all_signals) == 2
    assert len(candidates) == 1
    assert candidates[0]["signal_id"] == accepted["signal_id"]
    assert plan["kind"] == "phase3_candidate_training_plan"
    assert plan["sample_count"] == 1
    assert plan["candidate_adapter"]["state"] == "planned"
    assert plan["eval_gate"]["required"] is True
    assert "/pfe/candidate/promote" == plan["handoff"]["promote_endpoint"]
    assert summary["eligible_training_count"] == 1
    assert summary["route_counts"]["training_candidate"] == 1


def test_route_signal_item_handles_loaded_records_without_route() -> None:
    item = SignalInboxItem.from_dict(
        {
            "signal_id": "sig-loaded",
            "signal_type": "accept",
            "persona_id": DEFAULT_PERSONA.persona_id,
            "scenario_id": DEFAULT_SCENARIO.scenario_id,
            "user_input": "请整理合同。",
            "model_output": "摘要和风险点。",
            "confidence": 0.9,
        }
    )
    route = route_signal_item(item)

    assert item.eligible_for_training is True
    assert route.eligible_for_training is True
