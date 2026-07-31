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
    PHASE59_CATEGORIES,
    build_phase59_blind_items,
    build_phase59_calibration_cases,
    build_phase59_candidate_judge_prompt,
    build_phase59_decision,
    build_phase59_fixture_semantic_audit,
    build_phase59_holdout_cases,
    build_phase59_proposition_candidates,
    build_phase59_split_integrity,
    evaluate_phase59_candidate_evaluator,
    evaluate_phase59_hard_rule_compatibility,
    ground_phase59_selection,
    phase59_ollama_json_schema,
    validate_phase59_raw_selection,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")


def _perfect_results(blind: dict) -> list[dict]:
    results = []
    for key in blind["hidden_key"]:
        for alias in JUDGES:
            results.append(
                {
                    "item_id": key["item_id"],
                    "judge_alias": alias,
                    **{
                        field: {"candidate_id": key["expected_candidate_ids"][field]}
                        for field in PHASE56_TYPED_FIELDS
                    },
                    "actual_model_call": True,
                }
            )
    return results


def test_phase59_candidates_are_pre_grounded_and_stable() -> None:
    response = (
        "材料不得登记为实际用户反馈。产品价值是否得到验证仍未确认。"
        "材料不能证明实际用户已经受益。"
    )
    first = build_phase59_proposition_candidates(response)
    second = build_phase59_proposition_candidates(response)

    assert first == second
    assert [row["candidate_id"] for row in first] == ["p001", "p002", "p003"]
    assert [row["value"] for row in first] == [
        "exclude_actual", "suspended_or_negated", "does_not_establish"
    ]
    assert all(row["pre_grounded"] is True for row in first)


def test_phase59_quotes_and_operational_facts_do_not_create_candidates() -> None:
    response = (
        "报告转述‘材料可以登记为实际用户反馈，产品价值已经得到验证’，该引文不代表认可。"
        "机器检查已经完成，耗时下降。"
    )
    assert build_phase59_proposition_candidates(response) == []


def test_phase59_prompt_and_schema_offer_only_field_candidates_or_none() -> None:
    item = {"assistant_response": "材料不得登记为实际用户反馈。产品价值是否得到验证仍未确认。"}
    candidates = build_phase59_proposition_candidates(item["assistant_response"])
    prompt = build_phase59_candidate_judge_prompt(item)
    schema = phase59_ollama_json_schema(candidates)

    assert "预验证候选" in prompt
    assert "不要输出 label" in prompt
    assert "p001" in prompt and "p002" in prompt
    source_enum = schema["properties"]["source_registration"]["properties"]["candidate_id"]["enum"]
    assert source_enum == ["none", "p001"]
    assert "p002" not in source_enum
    assert "label" not in schema["properties"]


def test_phase59_validation_rejects_cross_field_or_direct_label() -> None:
    response = "材料不得登记为实际用户反馈。产品价值是否得到验证仍未确认。"
    candidates = build_phase59_proposition_candidates(response)
    valid = {
        "source_registration": {"candidate_id": "p001"},
        "user_outcome_status": {"candidate_id": "p002"},
        "test_to_user_outcome_relation": {"candidate_id": "none"},
    }
    assert validate_phase59_raw_selection(valid, candidates=candidates)["source_registration_candidate_id"] == "p001"
    invalid = {**valid, "source_registration": {"candidate_id": "p002"}}
    with pytest.raises(ValueError, match="source_registration"):
        validate_phase59_raw_selection(invalid, candidates=candidates)
    with pytest.raises(ValueError, match="direct label"):
        validate_phase59_raw_selection({**valid, "label": "accept"}, candidates=candidates)


def test_phase59_grounding_attaches_prevalidated_evidence() -> None:
    response = "材料不得登记为实际用户反馈。产品价值是否得到验证仍未确认。"
    raw = {
        "source_registration": {"candidate_id": "p001"},
        "user_outcome_status": {"candidate_id": "p002"},
        "test_to_user_outcome_relation": {"candidate_id": "none"},
    }
    grounded = ground_phase59_selection(raw, response)
    assert grounded["source_registration"] == "exclude_actual"
    assert grounded["user_outcome_status"] == "suspended_or_negated"
    assert grounded["test_to_user_outcome_relation"] == "unstated"
    assert grounded["invalid_atom_count"] == 0
    assert grounded["invalid_dangerous_atom_count"] == 0
    assert grounded["source_registration_evidence_candidate"]["clause_id"] == "c001"


def test_phase59_fixture_semantic_audit_catches_phase58_style_cross_field_ambiguity() -> None:
    ambiguous = {
        "case_id": "ambiguous",
        "category": "fixture_audit",
        "assistant_response": (
            "材料不得登记为实际用户反馈。另一份材料也不得登记为实际用户反馈。"
            "材料不能证明实际用户已经受益。"
        ),
        "expected_typed": {
            "source_registration": "exclude_actual",
            "user_outcome_status": "unstated",
            "test_to_user_outcome_relation": "does_not_establish",
        },
        "expected_candidate_ids": {
            "source_registration": "p001",
            "user_outcome_status": "none",
            "test_to_user_outcome_relation": "p003",
        },
    }
    audit = build_phase59_fixture_semantic_audit([ambiguous])
    assert audit["status"] == "failed"
    assert audit["ambiguous_case_count"] == 1


def test_phase59_splits_are_balanced_fresh_and_semantically_audited() -> None:
    calibration = build_phase59_calibration_cases()
    holdout = build_phase59_holdout_cases()
    calibration_audit = build_phase59_fixture_semantic_audit(calibration["cases"])
    holdout_audit = build_phase59_fixture_semantic_audit(holdout["cases"])
    integrity = build_phase59_split_integrity(calibration["cases"], holdout["cases"])

    assert calibration["case_count"] == 30
    assert calibration["label_counts"] == {"accept": 10, "edit": 10, "reject": 10}
    assert holdout["case_count"] == 60
    assert holdout["label_counts"] == {"accept": 20, "edit": 20, "reject": 20}
    assert set(calibration["category_counts"]) == set(PHASE59_CATEGORIES)
    assert calibration_audit["status"] == "passed"
    assert holdout_audit["status"] == "passed"
    assert integrity["passed"] is True


def test_phase59_blinding_hides_gold_but_keeps_public_candidates() -> None:
    blind = build_phase59_blind_items(
        build_phase59_calibration_cases()["cases"], seed=5901, prefix="phase59-calibration-blind"
    )
    assert len(blind["public_items"]) == 30
    assert all("proposition_candidates" in row for row in blind["public_items"])
    assert all(
        "expected_label" not in row
        and "expected_typed" not in row
        and "expected_candidate_ids" not in row
        and "category" not in row
        for row in blind["public_items"]
    )


def test_phase59_frozen_hard_detector_matches_new_cases() -> None:
    for dataset in (build_phase59_calibration_cases(), build_phase59_holdout_cases()):
        report = evaluate_phase59_hard_rule_compatibility(dataset["cases"])
        assert report["status"] == "passed", [
            row for row in report["details"] if row["safe_case_false_positive"]
        ]
        assert report["safe_case_false_positive_count"] == 0
        assert report["typed_composer_remains_responsible_for_non_hard_rejects"] is True


def test_phase59_perfect_candidate_results_qualify() -> None:
    blind = build_phase59_blind_items(
        build_phase59_holdout_cases()["cases"], seed=5902, prefix="phase59-holdout-blind"
    )
    report = evaluate_phase59_candidate_evaluator(
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
    assert report["invalid_dangerous_atom_count"] == 0
    assert report["composer_received_ungrounded_atom_count"] == 0


def test_phase59_wrong_candidate_selection_fails_field_gate() -> None:
    blind = build_phase59_blind_items(
        build_phase59_calibration_cases()["cases"], seed=5901, prefix="phase59-calibration-blind"
    )
    results = _perfect_results(blind)
    changed = 0
    for row in results:
        if row["source_registration"]["candidate_id"] != "none" and changed < 8:
            row["source_registration"]["candidate_id"] = "none"
            changed += 1
    report = evaluate_phase59_candidate_evaluator(
        split="calibration",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )
    assert report["per_field"]["source_registration"]["accuracy"] < 0.95
    assert report["status"] == "not_qualified"


def test_phase59_decision_remains_manual_review_only_and_phase58_stays_held() -> None:
    qualified = {
        "status": "qualified",
        "false_accept_count_on_reject_cases": 0,
        "invalid_dangerous_atom_count": 0,
        "composer_received_ungrounded_atom_count": 0,
    }
    decision = build_phase59_decision(
        phase58_snapshot={"passed": True},
        calibration_report=qualified,
        holdout_report=qualified,
        calibration_audit={"status": "passed"},
        holdout_audit={"status": "passed"},
        hard_calibration={"status": "passed"},
        hard_holdout={"status": "passed"},
        split_integrity={"passed": True},
    )
    assert decision["recommendation"] == "recommend_phase59_proposition_evaluator_for_manual_review_only"
    assert decision["phase60_external_replay_design_eligible"] is True
    assert decision["runtime_replay_allowed_in_phase59"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False

    phase58 = json.loads(
        (ROOT / "docs/demo/phase58-clause-addressed-grounding/phase58-final-decision.json").read_text(encoding="utf-8")
    )
    assert phase58["recommendation"] == "hold_phase58_clause_addressed_grounding"
    assert phase58["phase59_minimal_runtime_ab_design_eligible"] is False
