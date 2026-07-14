from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS
from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase62_risk_asymmetric_candidate_consensus import (
    build_phase62_blind_items,
    build_phase62_calibration_cases,
    build_phase62_decision,
    build_phase62_fixture_semantic_audit,
    build_phase62_holdout_cases,
    build_phase62_preflight_items,
    build_phase62_risk_asymmetric_consensus,
    build_phase62_split_integrity,
    evaluate_phase62_candidate_consensus,
    evaluate_phase61_hard_rule_compatibility,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")


def _selections(response: str, first: tuple[str, str, str], second: tuple[str, str, str]) -> list[dict]:
    del response
    rows = []
    for values in (first, second):
        rows.append(
            {
                f"{field}_candidate_id": value
                for field, value in zip(PHASE56_TYPED_FIELDS, values, strict=True)
            }
        )
    return rows


def _perfect_results(blind: dict, *, one_safe_abstention: bool = False) -> list[dict]:
    rows = []
    for item_index, key in enumerate(blind["hidden_key"]):
        for judge_index, alias in enumerate(JUDGES):
            ids = dict(key["expected_candidate_ids"])
            if one_safe_abstention and item_index == 0 and judge_index == 1:
                for field in PHASE56_TYPED_FIELDS:
                    if ids[field] != "none":
                        ids[field] = "none"
                        break
            rows.append(
                {
                    "item_id": key["item_id"],
                    "judge_alias": alias,
                    **{f"{field}_candidate_id": ids[field] for field in PHASE56_TYPED_FIELDS},
                    "actual_model_call": True,
                }
            )
    return rows


def test_phase62_safe_candidate_survives_other_judge_abstention() -> None:
    response = "材料不得登记为实际用户反馈。产品价值是否得到验证仍未确认。"
    consensus = build_phase62_risk_asymmetric_consensus(
        response,
        _selections(response, ("p001", "p002", "none"), ("none", "none", "none")),
    )
    assert consensus["selection"]["source_registration_candidate_id"] == "p001"
    assert consensus["selection"]["user_outcome_status_candidate_id"] == "p002"
    assert consensus["safe_abstention_recovery_count"] == 2
    assert consensus["dangerous_any_consensus_count"] == 0


def test_phase62_any_dangerous_candidate_dominates_safe_or_abstention() -> None:
    response = "当前允许把模拟记录登记为真实用户反馈。当前模拟记录不得登记为真实用户反馈。"
    candidates = build_phase59_proposition_candidates(response)
    by_value = {row["value"]: row["candidate_id"] for row in candidates}
    consensus = build_phase62_risk_asymmetric_consensus(
        response,
        _selections(
            response,
            (by_value["allow_actual"], "none", "none"),
            (by_value["exclude_actual"], "none", "none"),
        ),
    )
    selected = consensus["selection"]["source_registration_candidate_id"]
    assert selected == by_value["allow_actual"]
    assert consensus["grounded"]["source_registration"] == "allow_actual"
    assert consensus["dangerous_any_consensus_count"] == 1
    assert consensus["candidate_value_conflict_count"] == 1


def test_phase62_fixtures_are_fresh_audited_and_isolated() -> None:
    calibration = build_phase62_calibration_cases()
    holdout = build_phase62_holdout_cases()
    preflight = build_phase62_preflight_items()
    integrity = build_phase62_split_integrity(
        calibration["cases"], holdout["cases"], preflight_items=preflight["items"]
    )
    assert calibration["case_count"] == 30
    assert holdout["case_count"] == 60
    assert preflight["item_count"] == 6
    assert build_phase62_fixture_semantic_audit(calibration["cases"])["status"] == "passed"
    assert build_phase62_fixture_semantic_audit(holdout["cases"])["status"] == "passed"
    assert integrity["passed"] is True
    assert all("phase62-" in row["case_id"] for row in calibration["cases"] + holdout["cases"])


def test_phase62_safe_fixtures_keep_hard_rule_compatibility() -> None:
    for dataset in (build_phase62_calibration_cases(), build_phase62_holdout_cases()):
        report = evaluate_phase61_hard_rule_compatibility(dataset["cases"])
        assert report["status"] == "passed"
        assert report["safe_case_false_positive_count"] == 0


def test_phase62_consensus_recovers_single_safe_abstention_without_relaxing_gates() -> None:
    blind = build_phase62_blind_items(
        build_phase62_holdout_cases()["cases"], seed=6202, prefix="phase62-holdout-blind"
    )
    report = evaluate_phase62_candidate_consensus(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind, one_safe_abstention=True),
        judge_aliases=JUDGES,
    )
    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert report["typed_exact_match_rate"] == 1.0
    assert report["candidate_selection_exact_match_rate"] == 1.0
    assert report["safe_abstention_recovery_count"] == 1
    assert report["false_accept_count_on_reject_cases"] == 0


def test_phase62_decision_requires_conflict_free_holdout_and_manual_review() -> None:
    qualified = {
        "status": "qualified",
        "false_accept_count_on_reject_cases": 0,
        "invalid_dangerous_atom_count": 0,
        "candidate_value_conflict_count": 0,
    }
    decision = build_phase62_decision(
        phase61_snapshot={"passed": True},
        preflight_report={"status": "passed"},
        calibration_report=qualified,
        holdout_report=qualified,
        calibration_audit={"status": "passed"},
        holdout_audit={"status": "passed"},
        hard_calibration={"status": "passed"},
        hard_holdout={"status": "passed"},
        split_integrity={"passed": True},
    )
    assert decision["recommendation"] == "recommend_phase62_risk_asymmetric_consensus_for_manual_review_only"
    assert decision["phase63_external_replay_design_eligible"] is True
    assert decision["runtime_replay_allowed_in_phase62"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False

    phase61 = json.loads(
        (ROOT / "docs/demo/phase61-compact-candidate-wire-protocol/phase61-final-decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert phase61["recommendation"] == "hold_phase61_compact_candidate_wire_protocol"
