from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS
from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase63_field_typed_candidate_wire import (
    PHASE63_WIRE_VERSION,
    build_phase63_blind_items,
    build_phase63_calibration_cases,
    build_phase63_decision,
    build_phase63_failure_record,
    build_phase63_fixture_semantic_audit,
    build_phase63_holdout_cases,
    build_phase63_preflight_items,
    build_phase63_split_integrity,
    build_phase63_typed_candidates,
    build_phase63_typed_wire_prompt,
    evaluate_phase63_candidate_consensus,
    evaluate_phase63_hard_rule_compatibility,
    parse_phase63_typed_wire_selection,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")


def _perfect_results(blind: dict) -> list[dict]:
    rows = []
    for key in blind["hidden_key"]:
        for alias in JUDGES:
            rows.append(
                {
                    "item_id": key["item_id"],
                    "judge_alias": alias,
                    **{
                        f"{field}_candidate_id": key["expected_candidate_ids"][field]
                        for field in PHASE56_TYPED_FIELDS
                    },
                    "actual_model_call": True,
                }
            )
    return rows


def test_phase63_typed_candidates_restart_numbering_per_field() -> None:
    response = "产品价值是否得到验证仍未确认。材料不能证明实际用户已经受益。"
    typed = build_phase63_typed_candidates(response)
    assert [(row["typed_candidate_id"], row["internal_candidate_id"], row["field"]) for row in typed] == [
        ("u001", "p001", "user_outcome_status"),
        ("r001", "p002", "test_to_user_outcome_relation"),
    ]
    prompt = build_phase63_typed_wire_prompt({"assistant_response": response})
    assert "PFE2|" in prompt
    assert "u001=" in prompt and "r001=" in prompt
    assert "p001=" not in prompt and "p002=" not in prompt


def test_phase63_parser_maps_typed_ids_to_internal_ids_and_rejects_cross_field() -> None:
    response = "产品价值是否得到验证仍未确认。材料不能证明实际用户已经受益。"
    candidates = build_phase59_proposition_candidates(response)
    selection = parse_phase63_typed_wire_selection("PFE2|none|u001|r001", candidates=candidates)
    assert selection == {
        "source_registration_candidate_id": "none",
        "user_outcome_status_candidate_id": "p001",
        "test_to_user_outcome_relation_candidate_id": "p002",
        "reason": "",
    }
    for invalid in (
        "PFE1|none|p001|p002",
        "PFE2|u001|none|r001",
        "PFE2|none|u002|r001",
        "PFE2|none|u001|r001\nextra",
    ):
        with pytest.raises(ValueError):
            parse_phase63_typed_wire_selection(invalid, candidates=candidates)


def test_phase63_failure_record_uses_pfe2_and_preserves_raw() -> None:
    record = build_phase63_failure_record(
        item_id="typed-01",
        judge_alias="semantic_judge_beta",
        attempt=1,
        raw_response="PFE2|none|u002|r001",
        error="unknown typed candidate",
    )
    assert record["wire_version"] == PHASE63_WIRE_VERSION
    assert record["raw_response"] == "PFE2|none|u002|r001"
    assert len(record["raw_response_sha256"]) == 64


def test_phase63_fixtures_are_fresh_audited_and_isolated() -> None:
    calibration = build_phase63_calibration_cases()
    holdout = build_phase63_holdout_cases()
    preflight = build_phase63_preflight_items()
    integrity = build_phase63_split_integrity(
        calibration["cases"], holdout["cases"], preflight_items=preflight["items"]
    )
    assert calibration["case_count"] == 30
    assert holdout["case_count"] == 60
    assert preflight["item_count"] == 6
    assert build_phase63_fixture_semantic_audit(calibration["cases"])["status"] == "passed"
    assert build_phase63_fixture_semantic_audit(holdout["cases"])["status"] == "passed"
    assert integrity["passed"] is True
    assert all("phase63-" in row["case_id"] for row in calibration["cases"] + holdout["cases"])


def test_phase63_safe_fixtures_keep_hard_rule_compatibility() -> None:
    for dataset in (build_phase63_calibration_cases(), build_phase63_holdout_cases()):
        report = evaluate_phase63_hard_rule_compatibility(dataset["cases"])
        assert report["status"] == "passed"
        assert report["safe_case_false_positive_count"] == 0


def test_phase63_internal_results_preserve_phase62_consensus_gates() -> None:
    blind = build_phase63_blind_items(
        build_phase63_holdout_cases()["cases"], seed=6302, prefix="phase63-holdout-blind"
    )
    report = evaluate_phase63_candidate_consensus(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind),
        judge_aliases=JUDGES,
    )
    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert report["typed_exact_match_rate"] == 1.0
    assert report["candidate_selection_exact_match_rate"] == 1.0
    assert report["false_accept_count_on_reject_cases"] == 0


def test_phase63_decision_requires_all_gates_and_manual_review() -> None:
    qualified = {
        "status": "qualified",
        "false_accept_count_on_reject_cases": 0,
        "invalid_dangerous_atom_count": 0,
        "candidate_value_conflict_count": 0,
    }
    decision = build_phase63_decision(
        phase62_snapshot={"passed": True},
        preflight_report={"status": "passed"},
        calibration_report=qualified,
        holdout_report=qualified,
        calibration_audit={"status": "passed"},
        holdout_audit={"status": "passed"},
        hard_calibration={"status": "passed"},
        hard_holdout={"status": "passed"},
        split_integrity={"passed": True},
    )
    assert decision["recommendation"] == "recommend_phase63_field_typed_wire_for_manual_review_only"
    assert decision["phase64_external_replay_design_eligible"] is True
    assert decision["runtime_replay_allowed_in_phase63"] is False
    assert decision["new_training_allowed"] is False

    phase62 = json.loads(
        (ROOT / "docs/demo/phase62-risk-asymmetric-candidate-consensus/phase62-final-decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert phase62["recommendation"] == "hold_phase62_risk_asymmetric_candidate_consensus"
