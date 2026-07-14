from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase55_atomic_boundary_composition import build_phase55_holdout_cases
from pfe_core.phase68_aligned_candidate_scope_recovery import (
    PHASE68_CATEGORIES,
    build_phase68_calibration_cases,
    build_phase68_candidate_audit,
    build_phase68_decision,
    build_phase68_holdout_cases,
    build_phase68_preflight_items,
)


PHASE67_ROOT = ROOT / "docs/demo/phase67-historical-contract-compatibility-audit"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase68_fresh_cases_are_balanced_and_isolated() -> None:
    calibration = build_phase68_calibration_cases()
    holdout = build_phase68_holdout_cases()

    assert calibration["case_count"] == 60
    assert holdout["case_count"] == 120
    assert set(calibration["category_counts"]) == set(PHASE68_CATEGORIES)
    assert calibration["label_counts"] == {"accept": 20, "edit": 20, "reject": 20}
    assert holdout["label_counts"] == {"accept": 40, "edit": 40, "reject": 40}
    assert {row["assistant_response"] for row in calibration["cases"]}.isdisjoint(
        row["assistant_response"] for row in holdout["cases"]
    )


def test_phase68_negation_first_candidates_cover_fresh_cases() -> None:
    cases = build_phase68_calibration_cases()["cases"] + build_phase68_holdout_cases()["cases"]
    audit = build_phase68_candidate_audit(cases)

    assert audit["status"] == "passed"
    assert audit["failed_case_count"] == 0


def test_phase68_candidates_cover_aligned_phase55_without_relabeling() -> None:
    audit = build_phase68_candidate_audit(
        build_phase55_holdout_cases()["cases"],
        include_details=False,
        require_typed_exact=False,
    )

    assert audit["status"] == "passed"
    assert audit["case_count"] == 150
    assert audit["failed_case_count"] == 0
    assert audit["audit_mode"] == "label_compatible"


def test_phase68_preflight_is_not_training_data() -> None:
    preflight = build_phase68_preflight_items()

    assert preflight["item_count"] == 6
    assert preflight["scored_as_calibration"] is False
    assert all(row["not_for_training"] is True for row in preflight["items"])
    assert all(row["actual_user_feedback"] is False for row in preflight["items"])


def test_phase68_decision_requires_fresh_and_aligned_gates() -> None:
    qualified = {
        "status": "qualified",
        "accuracy": 1.0,
        "false_accept_count_on_reject_cases": 0,
        "schema_failure_count": 0,
        "candidate_value_conflict_count": 0,
    }
    decision = build_phase68_decision(
        phase67_snapshot={"passed": True},
        aggregate_failure_audit={"passed": True},
        fresh_calibration_report=qualified,
        fresh_holdout_report=qualified,
        aligned_phase55_report=qualified,
        fresh_candidate_audit={"status": "passed"},
        aligned_candidate_audit={"status": "passed"},
        split_integrity={"passed": True},
    )

    assert decision["recommendation"] == (
        "recommend_phase68_evaluator_qualification_for_manual_review_only"
    )
    assert decision["phase69_minimal_runtime_ab_design_eligible"] is True
    assert decision["runtime_ab_allowed_in_phase68"] is False
    assert decision["training_allowed"] is False


def test_phase68_below_aligned_gate_holds() -> None:
    qualified = {
        "status": "qualified",
        "accuracy": 1.0,
        "false_accept_count_on_reject_cases": 0,
        "schema_failure_count": 0,
        "candidate_value_conflict_count": 0,
    }
    aligned = {**qualified, "accuracy": 0.9499}
    decision = build_phase68_decision(
        phase67_snapshot={"passed": True},
        aggregate_failure_audit={"passed": True},
        fresh_calibration_report=qualified,
        fresh_holdout_report=qualified,
        aligned_phase55_report=aligned,
        fresh_candidate_audit={"status": "passed"},
        aligned_candidate_audit={"status": "passed"},
        split_integrity={"passed": True},
    )

    assert decision["recommendation"] == "hold_phase68_aligned_candidate_scope_recovery"
    assert decision["phase69_minimal_runtime_ab_design_eligible"] is False


def test_phase67_decision_remains_manual_review_only() -> None:
    decision = _read(PHASE67_ROOT / "phase67-final-decision.json")
    assert decision["recommendation"] == (
        "recommend_phase67_contract_aware_partition_for_manual_review_only"
    )
    assert decision["runtime_ab_allowed"] is False
