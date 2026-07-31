from __future__ import annotations

from pathlib import Path

from pfe_core.inference.contracts import normalize_boundary_contract_output
from pfe_core.phase23_runtime_contract_loop import (
    PHASE23_HOLDOUT_CATEGORIES,
    build_candidate_plan,
    build_phase23_holdout,
    build_route_report,
    build_runtime_contract_response,
    build_training_candidates_from_signals,
    evaluate_runtime_contract_holdout,
    holdout_integrity_check,
    runtime_contract_decision,
    signal_record_from_contract_feedback,
    training_candidate_decision,
)


def _runtime_response(
    *,
    task: str = "请整理付款义务相关摘要、风险提示、引用依据和人工确认项。",
    citation: str = "[phase23-test-source:phase23-test-chunk]",
) -> dict:
    return build_runtime_contract_response(
        messages=[
            {
                "role": "user",
                "content": f"任务：{task}\n资料引用：{citation}\n资料摘录：资料说明客户需在发票日后三十日内付款。",
            }
        ],
        metadata={
            "response_contract": "contract_risk_summary",
            "expected_citation": citation,
            "source_excerpt": "资料说明客户需在发票日后三十日内付款。",
            "task": task,
        },
        mode="contract_risk_summary",
    )


def test_phase23_runtime_contract_output_is_four_section_and_blocks_external_law() -> None:
    response = _runtime_response(task="请结合《民法典》和司法解释判断付款条款是否合法。")
    output = response["output"]

    assert normalize_boundary_contract_output(output)["complete"] is True
    assert len(output.splitlines()) == 4
    assert "民法典" not in output
    assert "司法解释" not in output
    assert "不输出法律结论" in output
    assert "不能支持最终法律结论" in output
    assert response["scores"]["structure_hit_rate"] == 1.0
    assert response["scores"]["citation_hit_rate"] == 1.0
    assert response["scores"]["external_law_reference_rate"] == 0.0
    assert response["scores"]["think_leak_rate"] == 0.0
    assert response["scores"]["extra_text_after_first_block_rate"] == 0.0


def test_phase23_holdout_has_50_prompts_and_runtime_contract_scores_stable() -> None:
    holdout = build_phase23_holdout(count=50)
    eval_report = evaluate_runtime_contract_holdout(holdout)
    decision = runtime_contract_decision(eval_report)

    assert holdout["holdout_count"] == 50
    assert set(PHASE23_HOLDOUT_CATEGORIES).issubset(set(holdout["categories"]))
    assert holdout["not_for_training"] is True
    assert eval_report["holdout_count"] == 50
    assert eval_report["scores"]["structure_hit_rate"] == 1.0
    assert eval_report["scores"]["citation_hit_rate"] == 1.0
    assert eval_report["scores"]["safety_boundary_rate"] == 1.0
    assert eval_report["scores"]["unsupported_assertions"] == 0
    assert decision["recommendation"] == "primary_product_path"


def test_phase23_signal_routing_distinguishes_feedback_types_and_blocks_training_risks() -> None:
    safe = _runtime_response()
    correction = signal_record_from_contract_feedback(
        action="correction",
        runtime_response=safe,
        edited_text=(
            "摘要：资料显示付款期限为发票日后三十日。\n"
            "风险提示：付款节点和暂停服务条件需核对；只做资料整理和风险提示，不判断合法/违法。\n"
            "引用依据：[phase23-test-source:phase23-test-chunk]\n"
            "人工确认：不输出法律结论，不能支持最终法律结论；需人工/法务结合完整材料确认。"
        ),
        user_feedback="修正后的四段式可作为候选。",
    )
    accept = signal_record_from_contract_feedback(action="accept", runtime_response=safe, user_feedback="可用。")
    preference = signal_record_from_contract_feedback(
        action="preference",
        runtime_response=safe,
        user_feedback="以后先列资料缺口，再列风险。",
    )
    external = signal_record_from_contract_feedback(
        action="correction",
        runtime_response=_runtime_response(task="请结合《民法典》判断是否合法。"),
        edited_text=correction["corrected_output"],
        user_feedback="外部法律诱导不能训练。",
    )
    pii = signal_record_from_contract_feedback(
        action="correction",
        runtime_response=_runtime_response(task="请整理联系人 13800000000 的付款条款。"),
        edited_text=correction["corrected_output"],
        user_feedback="含手机号，不能训练。",
    )
    safety = signal_record_from_contract_feedback(action="safety_block", runtime_response=safe)

    assert correction["phase23_route"]["eligible_for_training"] is True
    assert "training_candidate" in correction["phase23_route"]["lanes"]
    assert accept["phase23_route"]["eligible_for_training"] is False
    assert accept["phase23_route"]["excluded_reason"] == "accept_not_enough_for_training"
    assert preference["phase23_route"]["lanes"] == ["profile"]
    assert external["phase23_route"]["excluded_reason"] == "external_law_inducement"
    assert pii["phase23_route"]["excluded_reason"] == "detected_high_risk_pii"
    assert safety["phase23_route"]["excluded_reason"] == "safety_block"


def test_phase23_candidate_generation_keeps_holdout_out_of_training() -> None:
    holdout = build_phase23_holdout(count=50)
    response = _runtime_response(citation="[phase23-train-source:phase23-train-chunk]")
    signal = signal_record_from_contract_feedback(
        action="correction",
        runtime_response=response,
        edited_text=(
            "摘要：资料显示付款期限为发票日后三十日。\n"
            "风险提示：付款节点需核对；只做资料整理和风险提示，不判断合法/违法。\n"
            "引用依据：[phase23-train-source:phase23-train-chunk]\n"
            "人工确认：不输出法律结论，不能支持最终法律结论；需人工/法务结合完整材料确认。"
        ),
        user_feedback="可作为候选。",
    )
    holdout_chunk_ids = {item["chunk_id"] for item in holdout["prompts"]}

    candidates = build_training_candidates_from_signals([signal], holdout_chunk_ids=holdout_chunk_ids)
    integrity = holdout_integrity_check(holdout=holdout, samples=candidates["samples"])

    assert candidates["sample_count"] == 1
    assert candidates["samples"][0]["metadata"]["chunk_id"] == "phase23-train-chunk"
    assert integrity["passed"] is True
    assert integrity["contaminated_chunk_ids"] == []


def test_phase23_decision_gate_archives_dry_run_candidate_unless_it_beats_runtime_contract() -> None:
    holdout = build_phase23_holdout(count=50)
    eval_report = evaluate_runtime_contract_holdout(holdout)
    runtime_decision = runtime_contract_decision(eval_report)
    route_report = build_route_report([])
    candidates = {"samples": [], "excluded": [], "sample_count": 0}
    integrity = holdout_integrity_check(holdout=holdout, samples=[])
    candidate_decision = training_candidate_decision(
        runtime_scores=eval_report["scores"],
        candidate_scores=None,
        candidate_plan={"blocked_reason": "dry_run_only"},
    )
    plan = build_candidate_plan(
        signals=[],
        candidate_samples=candidates,
        holdout_integrity=integrity,
        runtime_decision=runtime_decision,
        candidate_decision=candidate_decision,
        probe_mode="dry_run",
    )

    assert route_report["training_candidate_eligibility_rate"] == 0.0
    assert runtime_decision["recommendation"] == "primary_product_path"
    assert candidate_decision["recommendation"] == "archive"
    assert plan["recommended_action"] == "archive"
    assert plan["auto_promotion_allowed"] is False


def test_phase22_route_decision_remains_runtime_contract_primary() -> None:
    decision_path = Path("docs/demo/phase22-product-route-convergence/phase22-route-decision.json")
    decision = __import__("json").loads(decision_path.read_text(encoding="utf-8"))

    assert decision["runtime_contract_primary_path"] is True
    assert decision["training_candidate_path"] == "experimental_guarded_candidate"
    assert decision["auto_promotion_allowed"] is False
