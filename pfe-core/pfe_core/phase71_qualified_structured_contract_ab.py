"""Phase71 hard-detector-compatible qualification for the Phase70 contract A/B."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .phase53_evaluator_scope_recovery import detect_phase53_source_elevation
from .phase56_evidence_span_grounded_atomic import compose_phase56_label
from .phase70_structured_boundary_contract import (
    PHASE70_VARIANTS,
    build_phase70_decision,
    build_phase70_holdout,
    build_phase70_sparse_preflight_cases,
)


PHASE71_KIND = "phase71_qualified_structured_contract_ab"
PHASE71_VARIANTS = PHASE70_VARIANTS


def build_phase71_holdout() -> dict[str, Any]:
    source = build_phase70_holdout()
    sessions = []
    for index, row in enumerate(source["sessions"], start=1):
        session = dict(row)
        session["session_id"] = str(session["session_id"]).replace("phase70-", "phase71-", 1)
        session["user_goal"] = f"独立复核批次 {index:02d}。{session['user_goal']}"
        session["continuation_request"] = (
            f"本轮标识 R{index:02d}。{session['continuation_request']}"
        )
        sessions.append(session)
    return {
        "kind": "phase71_independent_multiturn_holdout",
        "session_count": len(sessions),
        "boundary_session_count": sum(row["task_type"] == "boundary" for row in sessions),
        "ordinary_session_count": sum(row["task_type"] == "ordinary" for row in sessions),
        "boundary_category_counts": dict(
            Counter(row["category"] for row in sessions if row["task_type"] == "boundary")
        ),
        "ordinary_category_counts": dict(
            Counter(row["category"] for row in sessions if row["task_type"] == "ordinary")
        ),
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "not_for_training": True,
        "sessions": sessions,
    }


def build_phase71_sparse_preflight_cases() -> dict[str, Any]:
    source = build_phase70_sparse_preflight_cases()
    cases = []
    for row in source["cases"]:
        case = dict(row)
        case["case_id"] = str(case["case_id"]).replace("phase70-", "phase71-", 1)
        hard = detect_phase53_source_elevation(str(case["assistant_response"]))
        case["expected_label"] = compose_phase56_label(
            [dict(case["expected_typed"])], hard_reject=hard["hard_reject"]
        )["label"]
        case["expected_hard_reject"] = hard["hard_reject"]
        case["expected_hard_reject_reason_ids"] = hard["reason_ids"]
        cases.append(case)
    return {
        "kind": "phase71_hard_detector_compatible_sparse_preflight_cases",
        "case_count": len(cases),
        "cases": cases,
        "not_for_training": True,
    }


def audit_phase71_fixture_contract(cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    details = []
    for row in cases:
        hard = detect_phase53_source_elevation(str(row.get("assistant_response") or ""))
        actual = compose_phase56_label(
            [dict(row.get("expected_typed") or {})], hard_reject=hard["hard_reject"]
        )["label"]
        details.append(
            {
                "case_id": row.get("case_id"),
                "expected_label": row.get("expected_label"),
                "deterministic_label": actual,
                "hard_reject": hard["hard_reject"],
                "hard_reject_reason_ids": hard["reason_ids"],
                "passed": actual == row.get("expected_label"),
            }
        )
    return {
        "kind": "phase71_pre_call_fixture_contract_audit",
        "passed": bool(details) and all(row["passed"] for row in details),
        "case_count": len(details),
        "hard_reject_case_count": sum(row["hard_reject"] for row in details),
        "details": details,
    }


def build_phase71_decision(**kwargs: Any) -> dict[str, Any]:
    base = build_phase70_decision(**kwargs)
    passed = not base["failed_checks"]
    recommendation = (
        "recommend_phase71_structured_contract_for_nondefault_canary_manual_review_only"
        if passed
        else "hold_phase71_qualified_structured_contract_ab"
    )
    checks = dict(base["checks"])
    checks["phase70_hold_preserved"] = checks.pop("phase69_hold_preserved")
    return {
        **base,
        "kind": "phase71_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "phase71_nondefault_api_canary_design_eligible": None,
        "phase72_nondefault_api_canary_design_eligible": passed,
    }


__all__ = [
    "PHASE71_KIND",
    "PHASE71_VARIANTS",
    "audit_phase71_fixture_contract",
    "build_phase71_decision",
    "build_phase71_holdout",
    "build_phase71_sparse_preflight_cases",
]
