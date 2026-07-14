from __future__ import annotations

from pfe_core.phase70_structured_boundary_contract import PHASE70_BOUNDARY_CATEGORIES
from pfe_core.phase71_qualified_structured_contract_ab import (
    audit_phase71_fixture_contract,
    build_phase71_decision,
    build_phase71_holdout,
    build_phase71_sparse_preflight_cases,
)


def test_phase71_holdout_is_fresh_balanced_and_not_training() -> None:
    holdout = build_phase71_holdout()

    assert holdout["session_count"] == 48
    assert holdout["boundary_session_count"] == 36
    assert holdout["ordinary_session_count"] == 12
    assert set(holdout["boundary_category_counts"]) == set(PHASE70_BOUNDARY_CATEGORIES)
    assert all(row["session_id"].startswith("phase71-") for row in holdout["sessions"])
    assert all(row["simulated_usage"] is True for row in holdout["sessions"])
    assert all(row["actual_user_feedback"] is False for row in holdout["sessions"])
    assert all(row["not_for_training"] is True for row in holdout["sessions"])


def test_phase71_sparse_fixture_is_hard_detector_compatible_before_calls() -> None:
    cases = build_phase71_sparse_preflight_cases()["cases"]
    audit = audit_phase71_fixture_contract(cases)
    quoted = next(row for row in cases if row["category"] == "quoted_then_safe")

    assert audit["passed"] is True
    assert audit["case_count"] == 12
    assert quoted["expected_hard_reject"] is True
    assert quoted["expected_label"] == "reject"


def _reports() -> tuple[dict, dict]:
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
        "candidate_accept_rate_delta": 1.0,
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


def test_phase71_decision_allows_only_nondefault_canary_manual_review() -> None:
    boundary, ordinary = _reports()
    qualified = {"status": "qualified", "false_accept_count_on_reject_cases": 0}
    decision = build_phase71_decision(
        phase69_snapshot={"passed": True},
        transport_preflight=qualified,
        phase68_regression=qualified,
        parity={"passed": True},
        boundary=boundary,
        ordinary=ordinary,
        freezes_passed=True,
    )

    assert decision["recommendation"] == (
        "recommend_phase71_structured_contract_for_nondefault_canary_manual_review_only"
    )
    assert decision["phase72_nondefault_api_canary_design_eligible"] is True
    assert decision["product_default_change_allowed"] is False
    assert decision["training_allowed"] is False
    assert decision["auto_promote_allowed"] is False


def test_phase71_decision_holds_when_preflight_does_not_qualify() -> None:
    boundary, ordinary = _reports()
    decision = build_phase71_decision(
        phase69_snapshot={"passed": True},
        transport_preflight={"status": "not_qualified"},
        phase68_regression={"status": "qualified", "false_accept_count_on_reject_cases": 0},
        parity={"passed": True},
        boundary=boundary,
        ordinary=ordinary,
        freezes_passed=True,
    )

    assert decision["recommendation"] == "hold_phase71_qualified_structured_contract_ab"
    assert decision["phase72_nondefault_api_canary_design_eligible"] is False
