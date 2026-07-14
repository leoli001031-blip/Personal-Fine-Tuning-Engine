from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase65_aggregate_safe_boundary_coverage import (
    PHASE65_AGGREGATE_FAILURE_CLASSES,
    build_phase65_calibration_cases,
    build_phase65_decision,
    build_phase65_holdout_cases,
    build_phase65_preflight_items,
    build_phase65_scope_rule_audit,
    build_phase65_split_integrity,
)


PHASE64_ROOT = ROOT / "docs/demo/phase64-field-typed-historical-replay"


def test_phase65_only_suppresses_relation_scoped_dangerous_outcome() -> None:
    response = (
        "当前内部巡检记录不得计入 actual_user_feedback，"
        "现有测试不能证明产品收益已经成立，关于产品收益已经成立的用户结果尚未确认。"
    )
    candidates = build_phase59_proposition_candidates(response)
    values = {(row["field"], row["value"]) for row in candidates}

    assert ("source_registration", "exclude_actual") in values
    assert ("user_outcome_status", "suspended_or_negated") in values
    assert ("test_to_user_outcome_relation", "does_not_establish") in values
    assert ("user_outcome_status", "asserted_current") not in values


def test_phase65_fresh_fixture_schema_and_scope_audits_pass() -> None:
    calibration = build_phase65_calibration_cases()
    holdout = build_phase65_holdout_cases()

    assert calibration["case_count"] == 60
    assert holdout["case_count"] == 120
    assert build_phase65_scope_rule_audit(calibration["cases"])["status"] == "passed"
    assert build_phase65_scope_rule_audit(holdout["cases"])["status"] == "passed"
    assert all(row["actual_user_feedback"] is False for row in calibration["cases"])
    assert all(row["not_for_training"] is True for row in holdout["cases"])


def test_phase65_splits_are_isolated_from_phase64_history() -> None:
    calibration = build_phase65_calibration_cases()["cases"]
    holdout = build_phase65_holdout_cases()["cases"]
    preflight = build_phase65_preflight_items()["items"]
    historical = []
    source_manifest = json.loads(
        (PHASE64_ROOT / "source_manifest.json").read_text(encoding="utf-8")
    )
    for source in source_manifest["sources"]:
        payload = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))
        historical.extend(payload["cases"])
    integrity = build_phase65_split_integrity(
        calibration,
        holdout,
        preflight_items=preflight,
        historical_cases=historical,
    )

    assert integrity["passed"] is True
    assert integrity["historical_fixture_count"] == 558
    assert integrity["preflight_count"] == 6


def test_phase65_decision_requires_fresh_holdout_and_scope_audits() -> None:
    qualified = {
        "status": "qualified",
        "accuracy": 1.0,
        "typed_exact_match_rate": 1.0,
        "candidate_selection_exact_match_rate": 1.0,
        "per_field": {
            field: {"accuracy": 1.0}
            for field in (
                "source_registration",
                "user_outcome_status",
                "test_to_user_outcome_relation",
            )
        },
        "per_category": {
            category: {"accuracy": 1.0}
            for category in (
                "separate_clause_ids",
                "shared_clause_id",
                "postposed_negation_scope",
                "quoted_distractor",
                "nearby_test_fact",
            )
        },
        "false_accept_count_on_reject_cases": 0,
        "candidate_value_conflict_count": 0,
    }
    decision = build_phase65_decision(
        phase64_snapshot={"passed": True},
        aggregate_failure_taxonomy={"passed": True},
        preflight_report={"status": "passed"},
        calibration_report=qualified,
        holdout_report=qualified,
        calibration_audit={"status": "passed"},
        holdout_audit={"status": "passed"},
        scope_calibration={"status": "passed"},
        scope_holdout={"status": "passed"},
        hard_calibration={"status": "passed"},
        hard_holdout={"status": "passed"},
        split_integrity={"passed": True},
    )

    assert decision["recommendation"] == (
        "recommend_phase65_scope_aware_candidates_for_manual_review_only"
    )
    assert decision["phase66_external_regression_design_eligible"] is True
    assert decision["runtime_replay_allowed_in_phase65"] is False
    assert decision["new_training_allowed"] is False


def test_phase64_hold_decision_remains_unchanged() -> None:
    decision = json.loads(
        (PHASE64_ROOT / "phase64-final-decision.json").read_text(encoding="utf-8")
    )
    assert decision["recommendation"] == "hold_phase64_field_typed_historical_replay"
    assert decision["phase65_minimal_runtime_ab_design_eligible"] is False


def test_phase65_failure_taxonomy_is_aggregate_only() -> None:
    assert PHASE65_AGGREGATE_FAILURE_CLASSES == (
        "safe_outcome_removed_when_relation_candidate_present",
        "assertion_shaped_outcome_embedded_in_negated_relation",
        "rejected_quote_dangerous_text",
        "multi_atom_single_clause",
    )
