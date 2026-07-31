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
from pfe_core.phase59_proposition_addressed_grounding import (
    build_phase59_proposition_candidates,
    evaluate_phase59_hard_rule_compatibility,
)
from pfe_core.phase60_flat_schema_compatibility import (
    PHASE60_OUTPUT_FIELDS,
    build_phase60_blind_items,
    build_phase60_calibration_cases,
    build_phase60_decision,
    build_phase60_failure_record,
    build_phase60_fixture_semantic_audit,
    build_phase60_flat_judge_prompt,
    build_phase60_holdout_cases,
    build_phase60_preflight_items,
    build_phase60_split_integrity,
    evaluate_phase60_candidate_evaluator,
    phase60_flat_json_schema,
    validate_phase60_flat_selection,
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


def test_phase60_schema_is_flat_and_model_specific_fields_are_required() -> None:
    response = "材料不得登记为实际用户反馈。产品价值是否得到验证仍未确认。"
    candidates = build_phase59_proposition_candidates(response)
    schema = phase60_flat_json_schema(candidates)
    prompt = build_phase60_flat_judge_prompt({"assistant_response": response})

    assert tuple(schema["required"]) == PHASE60_OUTPUT_FIELDS
    assert schema["properties"]["source_registration_candidate_id"]["type"] == "string"
    assert schema["properties"]["source_registration_candidate_id"]["enum"] == ["none", "p001"]
    assert "不要嵌套对象" in prompt
    assert "source_registration_candidate_id" in prompt
    assert "label" not in schema["properties"]


def test_phase60_validator_accepts_flat_and_rejects_nested_or_direct_label() -> None:
    response = "材料不得登记为实际用户反馈。产品价值是否得到验证仍未确认。"
    candidates = build_phase59_proposition_candidates(response)
    flat = {
        "source_registration_candidate_id": "p001",
        "user_outcome_status_candidate_id": "p002",
        "test_to_user_outcome_relation_candidate_id": "none",
    }
    assert validate_phase60_flat_selection(flat, candidates=candidates) == {**flat, "reason": ""}
    nested = {
        "source_registration": {"candidate_id": "p001"},
        **flat,
    }
    with pytest.raises(ValueError, match="nested"):
        validate_phase60_flat_selection(nested, candidates=candidates)
    with pytest.raises(ValueError, match="direct label"):
        validate_phase60_flat_selection({**flat, "label": "accept"}, candidates=candidates)


def test_phase60_failure_record_preserves_raw_invalid_response() -> None:
    record = build_phase60_failure_record(
        item_id="preflight-01",
        judge_alias="semantic_judge_beta",
        attempt=1,
        raw_response='{"source_registration":"p001"}',
        error="nested schema mismatch",
    )
    assert record["raw_response"] == '{"source_registration":"p001"}'
    assert len(record["raw_response_sha256"]) == 64
    assert record["schema_valid"] is False
    assert record["actual_model_call"] is True


def test_phase60_fixtures_are_fresh_audited_and_preflight_is_isolated() -> None:
    calibration = build_phase60_calibration_cases()
    holdout = build_phase60_holdout_cases()
    preflight = build_phase60_preflight_items()
    calibration_audit = build_phase60_fixture_semantic_audit(calibration["cases"])
    holdout_audit = build_phase60_fixture_semantic_audit(holdout["cases"])
    integrity = build_phase60_split_integrity(
        calibration["cases"], holdout["cases"], preflight_items=preflight["items"]
    )

    assert calibration["case_count"] == 30
    assert holdout["case_count"] == 60
    assert preflight["item_count"] == 6
    assert calibration_audit["status"] == "passed"
    assert holdout_audit["status"] == "passed"
    assert integrity["passed"] is True
    assert all("phase60-" in row["case_id"] for row in calibration["cases"] + holdout["cases"])


def test_phase60_safe_fixtures_have_no_hard_rule_false_positive() -> None:
    for dataset in (build_phase60_calibration_cases(), build_phase60_holdout_cases()):
        report = evaluate_phase59_hard_rule_compatibility(dataset["cases"])
        assert report["status"] == "passed"
        assert report["safe_case_false_positive_count"] == 0


def test_phase60_blinding_and_perfect_flat_results_qualify() -> None:
    blind = build_phase60_blind_items(
        build_phase60_holdout_cases()["cases"], seed=6002, prefix="phase60-holdout-blind"
    )
    report = evaluate_phase60_candidate_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind),
        judge_aliases=JUDGES,
    )
    assert report["kind"] == "phase60_flat_candidate_evaluator_report"
    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert report["candidate_selection_exact_match_rate"] == 1.0
    assert report["invalid_dangerous_atom_count"] == 0
    assert all("expected_label" not in row for row in blind["public_items"])


def test_phase60_decision_requires_preflight_and_remains_manual_review_only() -> None:
    qualified = {
        "status": "qualified",
        "false_accept_count_on_reject_cases": 0,
        "invalid_dangerous_atom_count": 0,
    }
    decision = build_phase60_decision(
        phase59_snapshot={"passed": True},
        preflight_report={"status": "passed"},
        calibration_report=qualified,
        holdout_report=qualified,
        calibration_audit={"status": "passed"},
        holdout_audit={"status": "passed"},
        hard_calibration={"status": "passed"},
        hard_holdout={"status": "passed"},
        split_integrity={"passed": True},
    )
    assert decision["recommendation"] == "recommend_phase60_flat_candidate_evaluator_for_manual_review_only"
    assert decision["phase61_external_replay_design_eligible"] is True
    assert decision["runtime_replay_allowed_in_phase60"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False

    phase59 = json.loads(
        (ROOT / "docs/demo/phase59-proposition-addressed-grounding/phase59-final-decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert phase59["recommendation"] == "hold_phase59_proposition_addressed_grounding"
