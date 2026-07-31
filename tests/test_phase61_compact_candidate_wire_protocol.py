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
from pfe_core.phase61_compact_candidate_wire_protocol import (
    PHASE61_WIRE_VERSION,
    build_phase61_blind_items,
    build_phase61_calibration_cases,
    build_phase61_decision,
    build_phase61_failure_record,
    build_phase61_fixture_semantic_audit,
    build_phase61_holdout_cases,
    build_phase61_preflight_items,
    build_phase61_split_integrity,
    build_phase61_wire_judge_prompt,
    evaluate_phase61_candidate_evaluator,
    evaluate_phase61_hard_rule_compatibility,
    parse_phase61_wire_selection,
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


def test_phase61_prompt_uses_one_fixed_order_ascii_line() -> None:
    response = "材料不得登记为实际用户反馈。产品价值是否得到验证仍未确认。"
    prompt = build_phase61_wire_judge_prompt({"assistant_response": response})

    assert "PFE1|" in prompt
    assert "三个位置顺序固定" in prompt
    assert "不得输出 JSON" in prompt
    assert "只输出 schema 要求" not in prompt


def test_phase61_parser_accepts_only_exact_wire_and_field_valid_ids() -> None:
    response = "材料不得登记为实际用户反馈。产品价值是否得到验证仍未确认。"
    candidates = build_phase59_proposition_candidates(response)
    expected = {
        "source_registration_candidate_id": "p001",
        "user_outcome_status_candidate_id": "p002",
        "test_to_user_outcome_relation_candidate_id": "none",
        "reason": "",
    }
    assert parse_phase61_wire_selection("PFE1|p001|p002|none", candidates=candidates) == expected
    for invalid in (
        '{"source_registration_candidate_id":"p001"}',
        "source_registration_candidate_id: p001",
        "PFE1|p001=exclude_actual@c001|p002|none",
        "PFE1|p001|p002|none\nextra",
        " PFE1|p001|p002|none",
    ):
        with pytest.raises(ValueError, match="wire envelope"):
            parse_phase61_wire_selection(invalid, candidates=candidates)
    with pytest.raises(ValueError, match="invalid user_outcome_status"):
        parse_phase61_wire_selection("PFE1|p001|p001|none", candidates=candidates)


def test_phase61_failure_record_preserves_raw_wire_failure() -> None:
    record = build_phase61_failure_record(
        item_id="preflight-01",
        judge_alias="semantic_judge_beta",
        attempt=1,
        raw_response="PFE1|p001=exclude_actual@c001|none|none",
        error="invalid Phase61 compact wire envelope",
    )
    assert record["wire_version"] == PHASE61_WIRE_VERSION
    assert record["raw_response"].startswith("PFE1|")
    assert len(record["raw_response_sha256"]) == 64


def test_phase61_fixtures_are_fresh_audited_and_preflight_isolated() -> None:
    calibration = build_phase61_calibration_cases()
    holdout = build_phase61_holdout_cases()
    preflight = build_phase61_preflight_items()
    calibration_audit = build_phase61_fixture_semantic_audit(calibration["cases"])
    holdout_audit = build_phase61_fixture_semantic_audit(holdout["cases"])
    integrity = build_phase61_split_integrity(
        calibration["cases"], holdout["cases"], preflight_items=preflight["items"]
    )

    assert calibration["case_count"] == 30
    assert holdout["case_count"] == 60
    assert preflight["item_count"] == 6
    assert calibration_audit["status"] == "passed"
    assert holdout_audit["status"] == "passed"
    assert integrity["passed"] is True
    assert all("phase61-" in row["case_id"] for row in calibration["cases"] + holdout["cases"])


def test_phase61_safe_fixtures_have_no_hard_rule_false_positive() -> None:
    for dataset in (build_phase61_calibration_cases(), build_phase61_holdout_cases()):
        report = evaluate_phase61_hard_rule_compatibility(dataset["cases"])
        assert report["status"] == "passed"
        assert report["safe_case_false_positive_count"] == 0


def test_phase61_blinding_and_perfect_results_qualify() -> None:
    blind = build_phase61_blind_items(
        build_phase61_holdout_cases()["cases"], seed=6102, prefix="phase61-holdout-blind"
    )
    report = evaluate_phase61_candidate_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind),
        judge_aliases=JUDGES,
    )
    assert report["kind"] == "phase61_compact_wire_candidate_evaluator_report"
    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert report["candidate_selection_exact_match_rate"] == 1.0
    assert report["invalid_dangerous_atom_count"] == 0


def test_phase61_decision_requires_all_gates_and_remains_manual_review_only() -> None:
    qualified = {
        "status": "qualified",
        "false_accept_count_on_reject_cases": 0,
        "invalid_dangerous_atom_count": 0,
    }
    decision = build_phase61_decision(
        phase60_snapshot={"passed": True},
        preflight_report={"status": "passed"},
        calibration_report=qualified,
        holdout_report=qualified,
        calibration_audit={"status": "passed"},
        holdout_audit={"status": "passed"},
        hard_calibration={"status": "passed"},
        hard_holdout={"status": "passed"},
        split_integrity={"passed": True},
    )
    assert decision["recommendation"] == "recommend_phase61_compact_wire_evaluator_for_manual_review_only"
    assert decision["phase62_external_replay_design_eligible"] is True
    assert decision["runtime_replay_allowed_in_phase61"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False

    phase60 = json.loads(
        (ROOT / "docs/demo/phase60-flat-schema-compatibility-recovery/phase60-final-decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert phase60["recommendation"] == "hold_phase60_flat_schema_compatibility_recovery"
