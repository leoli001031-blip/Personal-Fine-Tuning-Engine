from __future__ import annotations

from pfe_core.phase93_95_dpo_product_proof import (
    aggregate_phase94_scores,
    build_phase93_94_holdouts,
    build_phase93_sanity_decision,
    build_phase95_product_decision,
    has_repeated_output,
)


def test_holdout_splits_are_fresh_disjoint_and_simulated() -> None:
    payload = build_phase93_94_holdouts()
    sanity = payload["sanity_sessions"]
    product = payload["product_sessions"]

    assert len(sanity) == 4
    assert len(product) == 12
    assert {row["session_id"] for row in sanity}.isdisjoint({row["session_id"] for row in product})
    assert all(row["not_for_training"] is True for row in sanity + product)
    assert all(row["simulated_usage"] is True and row["actual_user_feedback"] is False for row in sanity + product)


def test_repetition_detector_catches_duplicate_line_and_second_three_line_block() -> None:
    assert has_repeated_output("结论：完成\n依据：已确认\n下一步：归档") is False
    assert has_repeated_output("结论：完成\n结论：完成\n下一步：归档") is True
    assert has_repeated_output("结论：一\n依据：一\n下一步：一\n结论：二") is True


def test_aggregate_uses_category_specific_denominators() -> None:
    rows = [
        {"category": "exact_three_line", "native_format": True, "category_correct": True},
        {"category": "false_block", "false_block": False, "category_correct": True},
        {"category": "provenance", "provenance_correct": True, "category_correct": True},
        {"category": "ordinary_control", "category_correct": True},
    ]
    metrics = aggregate_phase94_scores(rows)

    assert metrics["exact_three_line_rate"] == 1.0
    assert metrics["false_block_avoidance_rate"] == 1.0
    assert metrics["provenance_correct_rate"] == 1.0


def test_sanity_requires_strict_improvement_without_regression() -> None:
    phase89 = {
        "session_count": 4,
        "exact_three_line_rate": 0.0,
        "false_block_avoidance_rate": 1.0,
        "provenance_correct_rate": 0.0,
    }
    candidate = {**phase89, "exact_three_line_rate": 1.0}

    assert build_phase93_sanity_decision(phase89, candidate)["passed"] is True
    assert build_phase93_sanity_decision(phase89, dict(phase89))["passed"] is False


def test_final_decision_never_promotes_even_when_simulated_gate_passes() -> None:
    base = {"session_count": 12, "exact_three_line_rate": 0.0, "false_block_avoidance_rate": 0.0, "provenance_correct_rate": 0.0}
    phase89 = {**base, "false_block_avoidance_rate": 1.0}
    candidate = {
        **phase89,
        "exact_three_line_rate": 1.0,
        "unsupported_assertion_rate": 0.0,
        "repeated_output_rate": 0.0,
        "think_leak_rate": 0.0,
        "privacy_echo_rate": 0.0,
    }
    decision = build_phase95_product_decision({"base": base, "phase89": phase89, "candidate": candidate})

    assert decision["status"] == "qualified_for_manual_review"
    assert decision["promotion_allowed"] is False
    assert decision["actual_product_benefit_claim_allowed"] is False
