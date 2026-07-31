from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase56_evidence_span_grounded_atomic import (
    PHASE56_TYPED_FIELDS,
    compose_phase56_label,
    evaluate_phase56_hard_reject_cases,
)
from pfe_core.phase58_clause_addressed_grounding import (
    PHASE58_CATEGORIES,
    PHASE58_CLAUSE_ID_FIELDS,
    build_phase58_blind_items,
    build_phase58_calibration_cases,
    build_phase58_clause_judge_prompt,
    build_phase58_decision,
    build_phase58_holdout_cases,
    build_phase58_split_integrity,
    evaluate_phase58_clause_evaluator,
    ground_phase58_extraction,
    mask_phase58_quoted_content,
    phase58_ollama_json_schema,
    segment_phase58_clauses,
    validate_phase58_raw_extraction,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")


def _perfect_results(blind: dict) -> list[dict]:
    results = []
    for key in blind["hidden_key"]:
        raw = {
            field: {
                "value": key["expected_atoms"][field]["value"],
                "evidence_clause_id": key["expected_atoms"][field]["evidence_clause_id"],
            }
            for field in PHASE56_TYPED_FIELDS
        }
        for alias in JUDGES:
            results.append(
                {
                    "item_id": key["item_id"],
                    "judge_alias": alias,
                    **{
                        field: dict(raw[field])
                        for field in PHASE56_TYPED_FIELDS
                    },
                    "actual_model_call": True,
                }
            )
    return results


def test_phase58_clause_segmentation_is_stable_and_quote_aware() -> None:
    response = "报告转述‘产品收益已经成立。无需复核’，该引文不代表认可。\n材料不得登记为真实反馈；结果尚未确认。"
    first = segment_phase58_clauses(response)
    second = segment_phase58_clauses(response)

    assert first == second
    assert [row["clause_id"] for row in first] == ["c001", "c002", "c003"]
    assert "产品收益已经成立。无需复核" in first[0]["text"]
    assert "产品收益已经成立" not in first[0]["unquoted_text"]
    assert mask_phase58_quoted_content("采用“危险断言”之外的当前结论") == "采用[引文已遮蔽]之外的当前结论"


def test_phase58_prompt_and_schema_require_existing_clause_ids() -> None:
    item = {"assistant_response": "材料不得登记为真实用户反馈。用户结果尚未确认。"}
    prompt = build_phase58_clause_judge_prompt(item)
    schema = phase58_ollama_json_schema(["c001", "c002"])

    assert "[c001]" in prompt and "[c002]" in prompt
    assert "不是最终裁判" in prompt
    assert "不要输出 label" in prompt
    atom = schema["properties"]["source_registration"]
    assert atom["required"] == ["value", "evidence_clause_id"]
    assert atom["properties"]["evidence_clause_id"]["enum"] == ["", "c001", "c002"]
    assert "label" not in schema["properties"]


def test_phase58_validation_rejects_direct_label_or_unknown_clause_id() -> None:
    raw = {
        field: {"value": "unstated", "evidence_clause_id": ""}
        for field in PHASE56_TYPED_FIELDS
    }
    assert validate_phase58_raw_extraction(raw, clause_ids=["c001"])["source_registration"] == "unstated"
    with pytest.raises(ValueError, match="direct label"):
        validate_phase58_raw_extraction({**raw, "label": "accept"}, clause_ids=["c001"])
    bad = {**raw, "source_registration": {"value": "exclude_actual", "evidence_clause_id": "c999"}}
    with pytest.raises(ValueError, match="unknown evidence clause"):
        validate_phase58_raw_extraction(bad, clause_ids=["c001"])


def test_phase58_whole_clause_preserves_postposed_negation() -> None:
    response = "产品收益已经成立这一判断仍未确认。"
    raw = {
        "source_registration": {"value": "unstated", "evidence_clause_id": ""},
        "user_outcome_status": {"value": "suspended_or_negated", "evidence_clause_id": "c001"},
        "test_to_user_outcome_relation": {"value": "unstated", "evidence_clause_id": ""},
    }
    grounded = ground_phase58_extraction(raw, response)
    assert grounded["user_outcome_status"] == "suspended_or_negated"
    assert grounded["user_outcome_status_grounded"] is True
    assert grounded["invalid_atom_count"] == 0


def test_phase58_quoted_claim_cannot_ground_dangerous_atom() -> None:
    response = "报告转述‘材料可以登记为真实用户反馈’，该引文不代表认可。"
    raw = {
        "source_registration": {"value": "allow_actual", "evidence_clause_id": "c001"},
        "user_outcome_status": {"value": "unstated", "evidence_clause_id": ""},
        "test_to_user_outcome_relation": {"value": "unstated", "evidence_clause_id": ""},
    }
    grounded = ground_phase58_extraction(raw, response)
    assert grounded["source_registration"] == "unstated"
    assert grounded["conservative_reject"] is True
    assert grounded["invalid_dangerous_atom_count"] == 1
    assert compose_phase56_label([grounded])["label"] == "reject"


def test_phase58_every_gold_clause_id_grounds_exact_atom() -> None:
    for dataset in (build_phase58_calibration_cases(), build_phase58_holdout_cases()):
        for case in dataset["cases"]:
            grounded = ground_phase58_extraction(case["expected_atoms"], case["assistant_response"])
            assert grounded["invalid_atom_count"] == 0, case["case_id"]
            assert {field: grounded[field] for field in PHASE56_TYPED_FIELDS} == case["expected_typed"]


def test_phase58_splits_are_fresh_balanced_and_blind() -> None:
    calibration = build_phase58_calibration_cases()
    holdout = build_phase58_holdout_cases()
    integrity = build_phase58_split_integrity(calibration["cases"], holdout["cases"])
    blind = build_phase58_blind_items(
        holdout["cases"], seed=5802, prefix="phase58-holdout-blind"
    )

    assert calibration["case_count"] == 30
    assert calibration["label_counts"] == {"accept": 10, "edit": 10, "reject": 10}
    assert holdout["case_count"] == 60
    assert holdout["label_counts"] == {"accept": 20, "edit": 20, "reject": 20}
    assert set(calibration["category_counts"]) == set(PHASE58_CATEGORIES)
    assert integrity["passed"] is True
    assert all(
        "expected_label" not in row
        and "expected_typed" not in row
        and "expected_atoms" not in row
        and "category" not in row
        for row in blind["public_items"]
    )
    assert all("expected_atoms" in row for row in blind["hidden_key"])


def test_phase58_frozen_hard_detector_matches_new_cases() -> None:
    for dataset in (build_phase58_calibration_cases(), build_phase58_holdout_cases()):
        report = evaluate_phase56_hard_reject_cases(dataset["cases"])
        assert report["status"] == "passed", [row for row in report["details"] if not row["passed"]]


def test_phase58_perfect_clause_results_qualify() -> None:
    blind = build_phase58_blind_items(
        build_phase58_holdout_cases()["cases"], seed=5802, prefix="phase58-holdout-blind"
    )
    report = evaluate_phase58_clause_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind),
        judge_aliases=JUDGES,
    )
    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert report["typed_exact_match_rate"] == 1.0
    assert report["grounding_validity_rate"] == 1.0
    assert report["expected_clause_id_exact_match_rate_diagnostic"] == 1.0
    assert report["invalid_dangerous_atom_count"] == 0
    assert report["composer_received_ungrounded_atom_count"] == 0


def test_phase58_invalid_dangerous_clause_id_blocks_qualification() -> None:
    blind = build_phase58_blind_items(
        build_phase58_calibration_cases()["cases"], seed=5801, prefix="phase58-calibration-blind"
    )
    results = _perfect_results(blind)
    target = next(row for row in results if row["source_registration"]["value"] == "allow_actual")
    target["source_registration"]["evidence_clause_id"] = ""
    report = evaluate_phase58_clause_evaluator(
        split="calibration",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )
    assert report["invalid_dangerous_atom_count"] == 1
    assert report["status"] == "not_qualified"


def test_phase58_decision_is_manual_review_only_and_phase57_stays_held() -> None:
    qualified = {
        "status": "qualified",
        "false_accept_count_on_reject_cases": 0,
        "invalid_dangerous_atom_count": 0,
        "composer_received_ungrounded_atom_count": 0,
        "judge_direct_label_count": 0,
    }
    decision = build_phase58_decision(
        phase57_snapshot={"passed": True},
        calibration_report=qualified,
        holdout_report=qualified,
        hard_calibration={"status": "passed"},
        hard_holdout={"status": "passed"},
        split_integrity={"passed": True},
    )
    assert decision["recommendation"] == "recommend_phase58_clause_addressed_evaluator_for_manual_review_only"
    assert decision["phase59_minimal_runtime_ab_design_eligible"] is True
    assert decision["runtime_replay_allowed_in_phase58"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False

    phase57 = json.loads(
        (ROOT / "docs/demo/phase57-span-evaluator-historical-replay/phase57-final-decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert phase57["recommendation"] == "hold_phase57_span_evaluator_historical_replay"
    assert phase57["phase58_minimal_runtime_ab_design_eligible"] is False
