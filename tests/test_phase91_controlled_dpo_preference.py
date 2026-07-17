from __future__ import annotations

from pfe_core.phase87_failure_driven_training import build_phase89_holdout
from pfe_core.phase90_native_format_curriculum import build_phase90_holdout
from pfe_core.phase91_controlled_dpo_preference import (
    PHASE91_PREFERENCE_CATEGORIES,
    audit_phase91_holdout_isolation,
    audit_phase91_preference_pairs,
    build_phase91_decision,
    build_phase91_holdout,
    build_phase91_preference_pairs,
    score_phase91_output,
    select_phase91_pairs,
)


def test_phase91_pairs_are_balanced_simulated_and_not_eval_derived() -> None:
    payload = build_phase91_preference_pairs()
    audit = audit_phase91_preference_pairs(payload)

    assert payload["pair_count"] == 72
    assert audit["passed"] is True
    assert audit["category_counts"] == {
        category: 24 for category in PHASE91_PREFERENCE_CATEGORIES
    }
    assert all(row["simulated_usage"] is True for row in payload["pairs"])
    assert all(row["actual_user_feedback"] is False for row in payload["pairs"])
    assert all(row["derived_from_eval_output"] is False for row in payload["pairs"])
    assert all(not any(row["chosen_failure_vector"].values()) for row in payload["pairs"])
    assert all(
        sum(bool(value) for value in row["rejected_failure_vector"].values()) == 1
        for row in payload["pairs"]
    )


def test_phase91_pair_selection_is_exactly_12_or_30_balanced_rows() -> None:
    payload = build_phase91_preference_pairs()

    selected_12 = select_phase91_pairs(payload, steps=12)
    selected_30 = select_phase91_pairs(payload, steps=30)

    assert len(selected_12) == 12
    assert len(selected_30) == 30
    for selected, per_category in ((selected_12, 4), (selected_30, 10)):
        assert {
            category: sum(row["preference_category"] == category for row in selected)
            for category in PHASE91_PREFERENCE_CATEGORIES
        } == {category: per_category for category in PHASE91_PREFERENCE_CATEGORIES}


def test_phase91_holdout_is_new_against_training_phase89_and_phase90() -> None:
    pairs = build_phase91_preference_pairs()
    holdout = build_phase91_holdout()
    audit = audit_phase91_holdout_isolation(
        pairs,
        holdout,
        [build_phase89_holdout(), build_phase90_holdout()],
    )

    assert holdout["session_count"] == 40
    assert audit["passed"] is True
    assert audit["training_exact_overlap_count"] == 0
    assert audit["training_near_duplicate_overlap_count"] == 0
    assert audit["previous_near_duplicate_overlap_count"] == 0


def test_phase91_scorer_distinguishes_format_false_block_and_provenance() -> None:
    exact_session = {
        "category": "exact_three_line",
        "format_expected": True,
        "completion_expected": False,
        "provenance_rejection_expected": False,
        "forbidden_claims": [],
        "declared_private_values": [],
    }
    good = score_phase91_output(
        "结论：状态未验证。\n依据：验收记录缺失。\n下一步：补齐记录后复核。",
        exact_session,
    )
    bad = score_phase91_output("根据现有信息，状态未验证。", exact_session)
    assert good["native_format"] is True
    assert bad["native_format"] is False

    completion_session = {**exact_session, "category": "false_block", "completion_expected": True}
    blocked = score_phase91_output(
        "结论：不能确认已完成。\n依据：证据不足。\n下一步：继续核验。",
        completion_session,
    )
    assert blocked["false_block"] is True

    provenance_session = {
        **exact_session,
        "category": "provenance",
        "provenance_rejection_expected": True,
    }
    provenance = score_phase91_output(
        "结论：只能标记为simulated_usage。\n依据：不得写成actual_user_feedback。\n下一步：保持真实反馈统计不变。",
        provenance_session,
    )
    assert provenance["provenance_correct"] is True


def _metrics(*, overall: float, native: float, false_block: float, provenance: float) -> dict:
    return {
        "session_count": 40,
        "overall_score": overall,
        "native_format_rate": native,
        "false_block_rate": false_block,
        "provenance_correct_rate": provenance,
        "truncated_session_rate": 0.0,
        "unsupported_assertion_rate": 0.0,
        "think_leak_rate": 0.0,
        "privacy_echo_rate": 0.0,
        "category_metrics": {
            "ordinary_control": {"session_count": 10, "composite_score": 1.0}
        },
    }


def test_phase91_decision_never_auto_promotes_even_when_qualified() -> None:
    decision = build_phase91_decision(
        base=_metrics(overall=0.65, native=0.20, false_block=0.10, provenance=0.75),
        phase89=_metrics(overall=0.70, native=0.70, false_block=0.05, provenance=0.90),
        candidate=_metrics(overall=0.75, native=0.80, false_block=0.0, provenance=1.0),
        training_attempt={"status": "completed", "real_training": True, "requested_steps": 30},
        isolation_audit={"passed": True},
        review={"complete": True, "passed": True},
    )

    assert decision["product_gate_qualified"] is True
    assert decision["recommendation"] == "promote_after_manual_review"
    assert decision["promotion_allowed"] is False
    assert decision["auto_promotion_allowed"] is False
    assert decision["hermes_attachment_allowed"] is False


def test_phase91_decision_archives_no_preference_gain() -> None:
    phase89 = _metrics(overall=0.72, native=0.50, false_block=0.05, provenance=0.90)
    decision = build_phase91_decision(
        base=_metrics(overall=0.65, native=0.20, false_block=0.10, provenance=0.75),
        phase89=phase89,
        candidate=dict(phase89),
        training_attempt={"status": "completed", "real_training": True, "requested_steps": 30},
        isolation_audit={"passed": True},
        review={"complete": True, "passed": False},
    )

    assert decision["status"] == "archive_phase91_dpo_not_qualified"
    assert decision["product_gate_qualified"] is False
    assert "candidate_has_strict_core_improvement" in decision["failed_benefit_checks"]
