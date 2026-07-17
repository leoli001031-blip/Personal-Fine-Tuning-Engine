from __future__ import annotations

from pfe_core.phase85_low_fallback_semantic_guard import PHASE85_PERSONA_CONTRACT
from pfe_core.phase87_failure_driven_training import PHASE87_TARGET_CATEGORIES
from pfe_core.phase87_failure_driven_training import build_phase89_holdout
from pfe_core.phase90_native_format_curriculum import (
    PHASE90_CURRICULA,
    audit_phase90_curriculum_candidates,
    audit_phase90_holdout_isolation,
    build_phase90_curriculum_candidates,
    build_phase90_decision,
    build_phase90_holdout,
    build_phase90_training_plan,
    select_phase90_training_samples,
)


def test_phase90_candidates_align_target_prompts_with_runtime_contract() -> None:
    candidates = build_phase90_curriculum_candidates()
    audit = audit_phase90_curriculum_candidates(candidates)

    assert audit["passed"] is True
    assert candidates["sample_count"] == 120
    for row in candidates["samples"]:
        messages = row["messages"]
        if row["taxonomy_dimension"] in PHASE87_TARGET_CATEGORIES:
            assert messages[0] == {
                "role": "system",
                "content": PHASE85_PERSONA_CONTRACT,
            }
        else:
            assert all(message["role"] != "system" for message in messages)
        assert row["simulated_usage"] is True
        assert row["actual_user_feedback"] is False


def test_phase90_training_plan_freezes_both_curricula_and_steps() -> None:
    candidates = build_phase90_curriculum_candidates()
    plan = build_phase90_training_plan(candidates)

    assert set(plan["curricula"]) == set(PHASE90_CURRICULA)
    for curriculum in PHASE90_CURRICULA:
        assert plan["curricula"][curriculum]["5"]["sample_count"] == 5
        assert plan["curricula"][curriculum]["25"]["sample_count"] == 25
    assert plan["curricula"]["format_first"]["25"]["category_counts"].get(
        "ordinary_direct", 0
    ) == 0
    assert plan["curricula"]["balanced"]["25"]["category_counts"][
        "ordinary_direct"
    ] == 4


def test_phase90_selection_rejects_unsupported_matrix_entries() -> None:
    candidates = build_phase90_curriculum_candidates()

    try:
        select_phase90_training_samples(candidates, curriculum="unknown", steps=5)
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("unsupported curriculum should fail")

    try:
        select_phase90_training_samples(candidates, curriculum="balanced", steps=12)
    except ValueError as exc:
        assert "5-step and 25-step" in str(exc)
    else:
        raise AssertionError("unsupported step count should fail")


def test_phase90_holdout_is_fresh_and_isolated() -> None:
    candidates = build_phase90_curriculum_candidates()
    holdout = build_phase90_holdout()
    audit = audit_phase90_holdout_isolation(
        candidates, holdout, build_phase89_holdout()
    )

    assert holdout["session_count"] == 40
    assert audit["passed"] is True
    assert audit["exact_overlap_count"] == 0
    assert audit["near_duplicate_overlap_count"] == 0
    assert audit["previous_holdout_near_duplicate_overlap_count"] == 0
    assert all(row["not_for_training"] is True for row in holdout["sessions"])


def _metrics(
    *,
    overall: float,
    native: float,
    false_block: float = 0.0,
    unsupported: float = 0.0,
    think: float = 0.0,
    privacy: float = 0.0,
    truncated: float = 0.0,
) -> dict:
    return {
        "session_count": 40,
        "overall_score": overall,
        "native_format_rate": native,
        "false_block_rate": false_block,
        "unsupported_assertion_rate": unsupported,
        "think_leak_rate": think,
        "privacy_echo_rate": privacy,
        "truncated_session_rate": truncated,
        "category_metrics": {
            category: {"session_count": 8, "composite_score": 0.8}
            for category in (
                "verified_completion_positive",
                "confirmation_reversal",
                "provenance_truthfulness",
                "grounded_no_invention",
                "ordinary_direct",
            )
        },
    }


def test_phase90_decision_never_allows_automatic_promotion() -> None:
    decision = build_phase90_decision(
        base_raw=_metrics(overall=0.70, native=0.20),
        phase89_raw=_metrics(overall=0.74, native=0.30),
        candidate_raw=_metrics(overall=0.82, native=0.80),
        base_runtime={"fallback_rate": 0.80},
        candidate_runtime={"fallback_rate": 0.05},
        training_attempt={"status": "completed", "real_training": True},
        isolation_audit={"passed": True},
        manual_review={"complete": True, "passed": True},
    )

    assert decision["product_gate_qualified"] is True
    assert decision["recommendation"] == "promote_after_manual_review"
    assert decision["promotion_allowed"] is False
    assert decision["auto_promotion_allowed"] is False
    assert decision["automatic_deployment_allowed"] is False


def test_phase90_decision_archives_low_native_or_high_fallback() -> None:
    decision = build_phase90_decision(
        base_raw=_metrics(overall=0.70, native=0.20),
        phase89_raw=_metrics(overall=0.72, native=0.20),
        candidate_raw=_metrics(overall=0.74, native=0.30, false_block=0.10),
        base_runtime={"fallback_rate": 0.80},
        candidate_runtime={"fallback_rate": 0.70},
        training_attempt={"status": "completed", "real_training": True},
        isolation_audit={"passed": True},
        manual_review={"complete": True, "passed": False},
    )

    assert decision["status"] == "archive_phase90_native_format_not_qualified"
    assert decision["product_gate_qualified"] is False
    assert "candidate_raw_native_at_least_0_75" in decision["failed_benefit_checks"]
    assert "candidate_runtime_fallback_at_most_0_10" in decision[
        "failed_benefit_checks"
    ]
