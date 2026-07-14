from __future__ import annotations

from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase53_evaluator_scope_recovery import detect_phase53_source_elevation
from pfe_core.phase70_structured_boundary_contract import (
    PHASE70_BOUNDARY_CATEGORIES,
    PHASE70_ORDINARY_CATEGORIES,
    PHASE70_STRUCTURED_CONTRACT,
    PHASE70_STRUCTURED_LINES,
    build_phase70_decision,
    build_phase70_holdout,
    build_phase70_runtime_messages,
    build_phase70_sparse_preflight_cases,
)


def test_phase70_holdout_is_independent_balanced_and_not_training() -> None:
    holdout = build_phase70_holdout()

    assert holdout["session_count"] == 48
    assert holdout["boundary_session_count"] == 36
    assert holdout["ordinary_session_count"] == 12
    assert set(holdout["boundary_category_counts"]) == set(PHASE70_BOUNDARY_CATEGORIES)
    assert set(holdout["ordinary_category_counts"]) == set(PHASE70_ORDINARY_CATEGORIES)
    assert all(row["simulated_usage"] is True for row in holdout["sessions"])
    assert all(row["actual_user_feedback"] is False for row in holdout["sessions"])
    assert all(row["not_for_training"] is True for row in holdout["sessions"])


def test_phase70_structured_contract_is_exactly_three_groundable_lines() -> None:
    output = "\n".join(PHASE70_STRUCTURED_LINES)
    candidates = build_phase59_proposition_candidates(output)
    values = {(row["field"], row["value"]) for row in candidates}

    assert len(PHASE70_STRUCTURED_LINES) == 3
    assert ("source_registration", "exclude_actual") in values
    assert ("user_outcome_status", "suspended_or_negated") in values
    assert ("test_to_user_outcome_relation", "does_not_establish") in values
    assert "不得改写、合并、省略" in PHASE70_STRUCTURED_CONTRACT


def test_phase70_runtime_only_varies_contract_shape() -> None:
    messages = [
        {"role": "system", "content": "short"},
        {"role": "user", "content": "当前测试能否证明用户结果？"},
    ]
    natural = build_phase70_runtime_messages(messages, variant="natural_boundary_contract")
    structured = build_phase70_runtime_messages(messages, variant="structured_boundary_contract")

    assert natural.manifest["privacy_transform_enabled"] is True
    assert structured.manifest["privacy_transform_enabled"] is True
    assert natural.manifest["latest_intent_enabled"] is True
    assert structured.manifest["latest_intent_enabled"] is True
    assert natural.manifest["structured_contract_enabled"] is False
    assert structured.manifest["structured_contract_enabled"] is True
    assert natural.messages[-1] == structured.messages[-1]


def test_phase70_sparse_preflight_covers_zero_one_two_and_three_atoms() -> None:
    preflight = build_phase70_sparse_preflight_cases()
    counts = []
    labels = set()
    for row in preflight["cases"]:
        counts.append(len(build_phase59_proposition_candidates(row["assistant_response"])))
        labels.add(row["expected_label"])

    assert preflight["case_count"] == 12
    assert {0, 1, 2, 3}.issubset(set(counts))
    assert labels == {"accept", "edit", "reject"}
    assert all(row["not_for_training"] is True for row in preflight["cases"])


def test_phase70_records_known_quoted_fixture_hard_detector_blocker() -> None:
    cases = build_phase70_sparse_preflight_cases()["cases"]
    quoted = next(row for row in cases if row["category"] == "quoted_then_safe")

    assert quoted["expected_label"] == "accept"
    assert detect_phase53_source_elevation(quoted["assistant_response"])["hard_reject"] is True


def _reports(delta: float = 1.0) -> tuple[dict, dict]:
    boundary = {
        "variants": {
            "natural_boundary_contract": {"completed_count": 36, "accept_rate": 0.0},
            "structured_boundary_contract": {
                "completed_count": 36,
                "accept_rate": 1.0,
                "exact_three_line_rate": 1.0,
                "dangerous_or_reject_count": 0,
            },
        },
        "candidate_accept_rate_delta": delta,
        "schema_failure_count": 0,
        "candidate_value_conflict_count": 0,
    }
    ordinary = {
        "variants": {
            "natural_boundary_contract": {"count": 12, "pass_rate": 0.8},
            "structured_boundary_contract": {
                "count": 12,
                "pass_rate": 0.8,
                "boundary_leak_count": 0,
            },
        }
    }
    return boundary, ordinary


def test_phase70_decision_allows_only_manual_review() -> None:
    boundary, ordinary = _reports()
    qualified = {"status": "qualified", "false_accept_count_on_reject_cases": 0}
    decision = build_phase70_decision(
        phase69_snapshot={"passed": True},
        transport_preflight=qualified,
        phase68_regression=qualified,
        parity={"passed": True},
        boundary=boundary,
        ordinary=ordinary,
        freezes_passed=True,
    )

    assert decision["recommendation"] == "recommend_phase70_structured_contract_for_manual_review_only"
    assert decision["phase71_nondefault_api_canary_design_eligible"] is True
    assert decision["product_default_change_allowed"] is False
    assert decision["training_allowed"] is False
    assert decision["auto_promote_allowed"] is False


def test_phase70_decision_holds_below_delta_gate() -> None:
    boundary, ordinary = _reports(delta=0.49)
    qualified = {"status": "qualified", "false_accept_count_on_reject_cases": 0}
    decision = build_phase70_decision(
        phase69_snapshot={"passed": True},
        transport_preflight=qualified,
        phase68_regression=qualified,
        parity={"passed": True},
        boundary=boundary,
        ordinary=ordinary,
        freezes_passed=True,
    )

    assert decision["recommendation"] == "hold_phase70_structured_boundary_contract"
    assert "candidate_delta_gate" in decision["failed_checks"]
