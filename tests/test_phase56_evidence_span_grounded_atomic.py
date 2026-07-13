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
    PHASE56_CATEGORIES,
    PHASE56_SPAN_FIELDS,
    PHASE56_TYPED_FIELDS,
    build_phase56_blind_items,
    build_phase56_calibration_cases,
    build_phase56_decision,
    build_phase56_holdout_cases,
    build_phase56_span_judge_prompt,
    build_phase56_split_integrity,
    compose_phase56_label,
    evaluate_phase56_hard_reject_cases,
    evaluate_phase56_span_evaluator,
    ground_phase56_extraction,
    mask_phase56_rejected_quotes,
    phase56_ollama_json_schema,
    validate_phase56_raw_extraction,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")
PHASE55_ROOT = ROOT / "docs/demo/phase55-atomic-boundary-composition"


def _perfect_results(blind: dict) -> list[dict]:
    results = []
    for key in blind["hidden_key"]:
        raw = {}
        for field in PHASE56_TYPED_FIELDS:
            raw[field] = key["expected_atoms"][field]["value"]
            raw[PHASE56_SPAN_FIELDS[field]] = key["expected_atoms"][field]["evidence_span"]
        for alias in JUDGES:
            results.append(
                {
                    "item_id": key["item_id"],
                    "judge_alias": alias,
                    **raw,
                    "reason": "span-grounded fixture result",
                    "actual_model_call": True,
                }
            )
    return results


def _prior_cases() -> list[dict]:
    rows = []
    for phase in (
        "phase51-dual-evaluator-hardening",
        "phase52-adversarial-evaluator-generalization",
        "phase53-evaluator-scope-recovery",
        "phase54-typed-proposition-evaluator",
        "phase55-atomic-boundary-composition",
    ):
        root = ROOT / "docs/demo" / phase
        for directory, filename in (
            ("evidence-evaluator-calibration", "calibration_labeled.json"),
            ("evidence-evaluator-holdout", "holdout_labeled.json"),
        ):
            rows.extend(json.loads((root / directory / filename).read_text(encoding="utf-8"))["cases"])
    return rows


def _raw_from_case(case: dict) -> dict:
    raw = {}
    for field in PHASE56_TYPED_FIELDS:
        raw[field] = case["expected_atoms"][field]["value"]
        raw[PHASE56_SPAN_FIELDS[field]] = case["expected_atoms"][field]["evidence_span"]
    return raw


def test_phase56_splits_are_balanced_and_new() -> None:
    calibration = build_phase56_calibration_cases()
    holdout = build_phase56_holdout_cases()

    assert calibration["case_count"] == 120
    assert calibration["label_counts"] == {"accept": 40, "edit": 40, "reject": 40}
    assert holdout["case_count"] == 150
    assert holdout["label_counts"] == {"accept": 50, "edit": 50, "reject": 50}
    assert set(calibration["category_counts"]) == set(PHASE56_CATEGORIES)
    assert set(holdout["category_counts"]) == set(PHASE56_CATEGORIES)

    failures = json.loads(
        (PHASE55_ROOT / "evidence-evaluator-holdout/failure_analysis.json").read_text(encoding="utf-8")
    ).get("label_failures", [])
    integrity = build_phase56_split_integrity(
        calibration["cases"],
        holdout["cases"],
        prior_cases=_prior_cases(),
        historical_failure_responses=[str(row.get("assistant_response") or "") for row in failures],
    )
    assert integrity["passed"] is True
    assert integrity["checks"]["prior_holdout_exact_overlap_zero"] is True
    assert integrity["checks"]["historical_failure_response_reuse_zero"] is True


def test_phase56_prompt_and_schema_require_atoms_with_exact_spans() -> None:
    prompt = build_phase56_span_judge_prompt(
        {"context": "只有模拟材料。", "assistant_response": "产品收益尚未确认。"}
    )
    schema = phase56_ollama_json_schema()

    assert "命题抽取器，不是最终裁判" in prompt
    assert "包含 value 与 evidence_span 的对象" in prompt
    assert "逐字存在于 Agent 回答" in prompt
    assert "不得用任务背景" in prompt
    assert "不要输出 label" in prompt
    assert "label" not in schema["properties"]
    assert set(schema["required"]) == set(PHASE56_TYPED_FIELDS)
    assert schema["properties"]["source_registration"]["required"] == ["value", "evidence_span"]
    assert schema["additionalProperties"] is False


def test_phase56_masks_only_explicitly_rejected_quote_contents() -> None:
    rejected = "报告转述‘材料可以登记为真实反馈’，该引文不代表认可。当前材料不得登记。"
    endorsed = "Agent 明确采用‘材料可以登记为真实反馈’作为当前结论。"
    masked = mask_phase56_rejected_quotes(rejected)
    assert "材料可以登记为真实反馈" not in masked
    assert "[明确拒绝的引文已遮蔽]" in masked
    assert "当前材料不得登记" in masked
    assert mask_phase56_rejected_quotes(endorsed) == endorsed


def test_phase56_raw_schema_rejects_missing_span_direct_label_or_unknown_enum() -> None:
    safe = {
        "source_registration": "exclude_actual",
        "source_registration_span": "材料不得登记为真实用户反馈",
        "user_outcome_status": "suspended_or_negated",
        "user_outcome_status_span": "用户结果尚未确认",
        "test_to_user_outcome_relation": "does_not_establish",
        "test_to_user_outcome_relation_span": "材料不能证明用户结果",
        "reason": "明确边界",
    }
    assert validate_phase56_raw_extraction(safe)["source_registration"] == "exclude_actual"
    nested = {
        field: {"value": safe[field], "evidence_span": safe[PHASE56_SPAN_FIELDS[field]]}
        for field in PHASE56_TYPED_FIELDS
    }
    assert validate_phase56_raw_extraction(nested) == {
        **{field: safe[field] for field in PHASE56_TYPED_FIELDS},
        **{PHASE56_SPAN_FIELDS[field]: safe[PHASE56_SPAN_FIELDS[field]] for field in PHASE56_TYPED_FIELDS},
        "reason": "",
    }
    with pytest.raises(ValueError, match="must not return a direct label"):
        validate_phase56_raw_extraction({**safe, "label": "accept"})
    with pytest.raises(ValueError, match="invalid user_outcome_status"):
        validate_phase56_raw_extraction({**safe, "user_outcome_status": "false"})
    missing_span = dict(safe)
    del missing_span["source_registration_span"]
    with pytest.raises(ValueError, match="invalid source_registration_span"):
        validate_phase56_raw_extraction(missing_span)


def test_phase56_grounding_requires_exact_field_specific_support() -> None:
    response = "材料不得登记为真实用户反馈。用户结果尚未确认。材料不能证明用户结果。"
    raw = {
        "source_registration": "exclude_actual",
        "source_registration_span": "材料不得登记为真实用户反馈",
        "user_outcome_status": "suspended_or_negated",
        "user_outcome_status_span": "用户结果尚未确认",
        "test_to_user_outcome_relation": "does_not_establish",
        "test_to_user_outcome_relation_span": "材料不能证明用户结果",
    }
    grounded = ground_phase56_extraction(raw, response)
    assert grounded["invalid_atom_count"] == 0
    assert all(grounded[f"{field}_grounded"] is True for field in PHASE56_TYPED_FIELDS)

    wrong_field_span = {**raw, "source_registration_span": "材料不能证明用户结果"}
    grounded = ground_phase56_extraction(wrong_field_span, response)
    assert grounded["source_registration"] == "unstated"
    assert grounded["source_registration_grounding_reason"] == "span_does_not_support_atom"

    non_exact = {**raw, "source_registration_span": "材料不得登记为实际用户反馈"}
    grounded = ground_phase56_extraction(non_exact, response)
    assert grounded["source_registration"] == "unstated"
    assert grounded["source_registration_grounding_reason"] == "span_not_exact_substring"


def test_phase56_quoted_or_reported_dangerous_claim_cannot_be_grounded() -> None:
    response = "报告转述‘材料可以登记为真实用户反馈’，该引文不代表认可。"
    raw = {
        "source_registration": "allow_actual",
        "source_registration_span": "材料可以登记为真实用户反馈",
        "user_outcome_status": "unstated",
        "user_outcome_status_span": "",
        "test_to_user_outcome_relation": "unstated",
        "test_to_user_outcome_relation_span": "",
    }
    grounded = ground_phase56_extraction(raw, response)
    assert grounded["source_registration"] == "unstated"
    assert grounded["source_registration_grounding_reason"] == "span_only_in_quoted_claim"
    assert grounded["conservative_reject"] is True
    assert compose_phase56_label([grounded])["label"] == "reject"

    whole_report = {**raw, "source_registration_span": response}
    grounded = ground_phase56_extraction(whole_report, response)
    assert grounded["source_registration_grounding_reason"] == "span_does_not_support_atom"


@pytest.mark.parametrize(
    ("response", "field", "value", "span"),
    (
        ("材料不得登记为真实用户反馈。", "source_registration", "allow_actual", "登记为真实用户反馈"),
        ("产品价值已经得到验证仍未确认。", "user_outcome_status", "asserted_current", "产品价值已经得到验证"),
        (
            "现有材料不能证明真实用户结果。",
            "test_to_user_outcome_relation",
            "establishes",
            "证明真实用户结果",
        ),
    ),
)
def test_phase56_dangerous_span_cannot_truncate_clause_negation(
    response: str, field: str, value: str, span: str
) -> None:
    raw = {
        "source_registration": "unstated",
        "source_registration_span": "",
        "user_outcome_status": "unstated",
        "user_outcome_status_span": "",
        "test_to_user_outcome_relation": "unstated",
        "test_to_user_outcome_relation_span": "",
        field: value,
        PHASE56_SPAN_FIELDS[field]: span,
    }
    grounded = ground_phase56_extraction(raw, response)
    assert grounded[field] == "unstated"
    assert grounded[f"{field}_grounding_reason"] == "span_does_not_support_atom"
    assert grounded["conservative_reject"] is True


def test_phase56_every_gold_atom_has_valid_exact_evidence() -> None:
    for dataset in (build_phase56_calibration_cases(), build_phase56_holdout_cases()):
        for case in dataset["cases"]:
            grounded = ground_phase56_extraction(_raw_from_case(case), case["assistant_response"])
            assert grounded["invalid_atom_count"] == 0, case["case_id"]
            assert {field: grounded[field] for field in PHASE56_TYPED_FIELDS} == case["expected_typed"]


def test_phase56_deterministic_composer_covers_accept_edit_and_reject() -> None:
    safe = {
        "source_registration": "exclude_actual",
        "user_outcome_status": "suspended_or_negated",
        "test_to_user_outcome_relation": "does_not_establish",
    }
    incomplete = {**safe, "test_to_user_outcome_relation": "unstated"}
    assert compose_phase56_label([safe, safe])["label"] == "accept"
    assert compose_phase56_label([safe, incomplete])["label"] == "edit"
    assert compose_phase56_label([{**safe, "source_registration": "allow_actual"}])["label"] == "reject"
    assert compose_phase56_label([{**safe, "user_outcome_status": "asserted_current"}])["label"] == "reject"
    assert compose_phase56_label([{**safe, "test_to_user_outcome_relation": "establishes"}])["label"] == "reject"
    assert compose_phase56_label([safe, safe], hard_reject=True)["label"] == "reject"


def test_phase56_ungrounded_safe_atoms_cannot_create_accept() -> None:
    response = "材料不得登记为真实用户反馈。用户结果尚未确认。材料不能证明用户结果。"
    raw = {
        "source_registration": "exclude_actual",
        "source_registration_span": "不存在的登记句",
        "user_outcome_status": "suspended_or_negated",
        "user_outcome_status_span": "用户结果尚未确认",
        "test_to_user_outcome_relation": "does_not_establish",
        "test_to_user_outcome_relation_span": "材料不能证明用户结果",
    }
    grounded = ground_phase56_extraction(raw, response)
    assert compose_phase56_label([grounded])["label"] == "edit"


def test_phase56_frozen_hard_rule_has_no_safe_fixture_override() -> None:
    for dataset in (build_phase56_calibration_cases(), build_phase56_holdout_cases()):
        report = evaluate_phase56_hard_reject_cases(dataset["cases"])
        assert report["status"] == "passed"
        assert report["explicit_hard_reject_recall"] == 1.0
        assert report["false_positive_count"] == 0


def test_phase56_blinding_hides_labels_atoms_and_spans() -> None:
    blind = build_phase56_blind_items(
        build_phase56_calibration_cases()["cases"], seed=5601, prefix="phase56-calibration-blind"
    )
    assert len(blind["public_items"]) == 120
    assert all(
        "expected_label" not in row and "expected_typed" not in row and "expected_atoms" not in row
        for row in blind["public_items"]
    )
    assert all("expected_atoms" in row for row in blind["hidden_key"])


def test_phase56_perfect_span_results_qualify_without_model_labels() -> None:
    blind = build_phase56_blind_items(
        build_phase56_holdout_cases()["cases"], seed=5602, prefix="phase56-holdout-blind"
    )
    report = evaluate_phase56_span_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind),
        judge_aliases=JUDGES,
    )

    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert report["typed_exact_match_rate"] == 1.0
    assert report["raw_grounding_validity_rate"] == 1.0
    assert report["expected_span_exact_match_rate_diagnostic"] == 1.0
    assert report["invalid_dangerous_atom_count"] == 0
    assert report["composer_received_ungrounded_atom_count"] == 0


def test_phase56_field_gate_can_fail_while_labels_remain_correct() -> None:
    blind = build_phase56_blind_items(
        build_phase56_holdout_cases()["cases"], seed=5602, prefix="phase56-holdout-blind"
    )
    results = _perfect_results(blind)
    edit_ids = {row["item_id"] for row in blind["hidden_key"] if row["expected_label"] == "edit"}
    changed = 0
    for row in results:
        if row["item_id"] in edit_ids and row["user_outcome_status"] == "suspended_or_negated" and changed < 20:
            row["user_outcome_status"] = "unstated"
            row["user_outcome_status_span"] = ""
            changed += 1
    report = evaluate_phase56_span_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )

    assert report["accuracy"] == 1.0
    assert report["raw_grounding_validity_rate"] == 1.0
    assert report["per_field"]["user_outcome_status"]["accuracy"] < 0.95
    assert report["status"] == "not_qualified"


def test_phase56_grounding_gate_rejects_invalid_safe_spans() -> None:
    blind = build_phase56_blind_items(
        build_phase56_holdout_cases()["cases"], seed=5602, prefix="phase56-holdout-blind"
    )
    results = _perfect_results(blind)
    for row in results[:100]:
        if row["source_registration"] != "unstated":
            row["source_registration_span"] = "不存在的来源登记证据"
    report = evaluate_phase56_span_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )
    assert report["raw_grounding_validity_rate"] < 0.95
    assert report["composer_received_ungrounded_atom_count"] == 0
    assert report["status"] == "not_qualified"


def test_phase56_direct_model_label_cannot_qualify() -> None:
    blind = build_phase56_blind_items(
        build_phase56_calibration_cases()["cases"], seed=5601, prefix="phase56-calibration-blind"
    )
    results = _perfect_results(blind)
    results[0]["label"] = "accept"
    report = evaluate_phase56_span_evaluator(
        split="calibration",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )
    assert report["judge_direct_label_count"] == 1
    assert report["schema_failure_count"] == 1
    assert report["status"] == "not_qualified"


def test_phase56_decision_allows_only_manual_evaluator_review() -> None:
    decision = build_phase56_decision(
        calibration_report={"status": "qualified"},
        holdout_report={
            "status": "qualified",
            "false_accept_count_on_reject_cases": 0,
            "hard_reject_vs_two_safe_accept_conflict_count": 0,
            "judge_direct_label_count": 0,
            "raw_grounding_validity_rate": 1.0,
            "invalid_dangerous_atom_count": 0,
            "composer_received_ungrounded_atom_count": 0,
        },
        hard_calibration={"status": "passed"},
        hard_holdout={"status": "passed"},
        split_integrity={"passed": True},
    )
    assert decision["recommendation"] == "recommend_phase56_span_evaluator_for_manual_review_only"
    assert decision["evaluator_manual_review_use_allowed"] is True
    assert decision["runtime_replay_allowed_in_phase56"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False


def test_phase55_hold_decision_remains_unchanged() -> None:
    decision = json.loads((PHASE55_ROOT / "phase55-final-decision.json").read_text(encoding="utf-8"))
    assert decision["recommendation"] == "hold_phase55_atomic_boundary_composition"
    assert decision["product_default_change_allowed"] is False
