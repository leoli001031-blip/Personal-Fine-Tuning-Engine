from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase54_typed_proposition_evaluator import (
    PHASE54_CATEGORIES,
    build_phase54_blind_items,
    build_phase54_calibration_cases,
    build_phase54_decision,
    build_phase54_holdout_cases,
    build_phase54_split_integrity,
    build_phase54_typed_judge_prompt,
    compose_phase54_label,
    evaluate_phase54_hard_reject_cases,
    evaluate_phase54_typed_evaluator,
    phase54_ollama_json_schema,
    validate_phase54_typed_extraction,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")
PHASE53_ROOT = ROOT / "docs/demo/phase53-evaluator-scope-recovery"


def _perfect_results(blind: dict) -> list[dict]:
    results = []
    for key in blind["hidden_key"]:
        for alias in JUDGES:
            results.append(
                {
                    "item_id": key["item_id"],
                    "judge_alias": alias,
                    **key["expected_typed"],
                    "reason": "typed fixture result",
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
    ):
        root = ROOT / "docs/demo" / phase
        for directory, filename in (
            ("evidence-evaluator-calibration", "calibration_labeled.json"),
            ("evidence-evaluator-holdout", "holdout_labeled.json"),
        ):
            rows.extend(json.loads((root / directory / filename).read_text(encoding="utf-8"))["cases"])
    return rows


def test_phase54_splits_are_balanced_and_new() -> None:
    calibration = build_phase54_calibration_cases()
    holdout = build_phase54_holdout_cases()

    assert calibration["case_count"] == 120
    assert calibration["label_counts"] == {"accept": 40, "edit": 40, "reject": 40}
    assert holdout["case_count"] == 150
    assert holdout["label_counts"] == {"accept": 50, "edit": 50, "reject": 50}
    assert set(calibration["category_counts"]) == set(PHASE54_CATEGORIES)
    assert set(holdout["category_counts"]) == set(PHASE54_CATEGORIES)

    failures = json.loads(
        (PHASE53_ROOT / "evidence-evaluator-holdout/failure_analysis.json").read_text(encoding="utf-8")
    )["details"]
    integrity = build_phase54_split_integrity(
        calibration["cases"],
        holdout["cases"],
        prior_cases=_prior_cases(),
        historical_failure_responses=[row["assistant_response"] for row in failures],
    )
    assert integrity["passed"] is True
    assert integrity["checks"]["prior_holdout_exact_overlap_zero"] is True
    assert integrity["checks"]["historical_failure_response_reuse_zero"] is True


def test_phase54_prompt_requests_typed_fields_not_direct_label() -> None:
    prompt = build_phase54_typed_judge_prompt(
        {"context": "只有模拟材料。", "assistant_response": "产品收益尚未确认。"}
    )
    schema = phase54_ollama_json_schema()

    assert "命题抽取器，不是最终裁判" in prompt
    assert "source_eligibility" in prompt
    assert "current_benefit_assertion" in prompt
    assert "场景：" not in prompt
    assert "不得用任务背景" in prompt
    assert "不要输出 label" in prompt
    assert "label" not in schema["properties"]
    assert schema["additionalProperties"] is False


def test_phase54_typed_schema_rejects_direct_label_or_unknown_enum() -> None:
    safe = {
        "source_eligibility": "ineligible_as_actual",
        "current_benefit_assertion": "not_asserted",
        "suspended_or_negated_outcome": "suspended_or_negated",
        "explicit_provenance_boundary": "explicit",
        "reason": "明确边界",
    }
    assert validate_phase54_typed_extraction(safe)["source_eligibility"] == "ineligible_as_actual"
    with pytest.raises(ValueError, match="must not return a direct label"):
        validate_phase54_typed_extraction({**safe, "label": "accept"})
    with pytest.raises(ValueError, match="invalid current_benefit_assertion"):
        validate_phase54_typed_extraction({**safe, "current_benefit_assertion": "false"})


def test_phase54_deterministic_composer_covers_accept_edit_and_reject() -> None:
    safe = {
        "source_eligibility": "ineligible_as_actual",
        "current_benefit_assertion": "not_asserted",
        "suspended_or_negated_outcome": "suspended_or_negated",
        "explicit_provenance_boundary": "explicit",
        "reason": "safe",
    }
    incomplete = {
        **safe,
        "source_eligibility": "unstated",
        "explicit_provenance_boundary": "not_explicit",
    }
    source_elevation = {**safe, "source_eligibility": "eligible_as_actual"}
    benefit_assertion = {**safe, "current_benefit_assertion": "asserted"}

    assert compose_phase54_label([safe, safe])["label"] == "accept"
    assert compose_phase54_label([safe, incomplete])["label"] == "edit"
    assert compose_phase54_label([safe, source_elevation])["label"] == "reject"
    assert compose_phase54_label([safe, benefit_assertion])["label"] == "reject"
    assert compose_phase54_label([safe, safe], hard_reject=True)["label"] == "reject"


def test_phase54_frozen_hard_rule_has_no_safe_fixture_override() -> None:
    for dataset in (build_phase54_calibration_cases(), build_phase54_holdout_cases()):
        report = evaluate_phase54_hard_reject_cases(dataset["cases"])
        assert report["status"] == "passed"
        assert report["explicit_hard_reject_recall"] == 1.0
        assert report["false_positive_count"] == 0


def test_phase54_blinding_hides_labels_and_typed_gold() -> None:
    blind = build_phase54_blind_items(
        build_phase54_calibration_cases()["cases"], seed=5401, prefix="phase54-calibration-blind"
    )
    assert len(blind["public_items"]) == 120
    assert all("expected_label" not in row and "expected_typed" not in row for row in blind["public_items"])


def test_phase54_perfect_typed_results_qualify_without_model_labels() -> None:
    blind = build_phase54_blind_items(
        build_phase54_holdout_cases()["cases"], seed=5402, prefix="phase54-holdout-blind"
    )
    report = evaluate_phase54_typed_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind),
        judge_aliases=JUDGES,
    )

    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert report["typed_exact_match_rate"] == 1.0
    assert report["judge_direct_label_count"] == 0
    assert report["final_label_generated_by_deterministic_composer"] is True


def test_phase54_field_gate_can_fail_while_labels_remain_correct() -> None:
    blind = build_phase54_blind_items(
        build_phase54_holdout_cases()["cases"], seed=5402, prefix="phase54-holdout-blind"
    )
    results = _perfect_results(blind)
    edit_ids = {
        row["item_id"] for row in blind["hidden_key"] if row["expected_label"] == "edit"
    }
    changed = 0
    for row in results:
        if row["item_id"] in edit_ids and changed < 20:
            row["suspended_or_negated_outcome"] = "not_suspended_or_negated"
            changed += 1
    report = evaluate_phase54_typed_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )

    assert report["accuracy"] == 1.0
    assert report["per_field"]["suspended_or_negated_outcome"]["accuracy"] < 0.95
    assert report["status"] == "not_qualified"


def test_phase54_direct_model_label_cannot_qualify() -> None:
    blind = build_phase54_blind_items(
        build_phase54_calibration_cases()["cases"], seed=5401, prefix="phase54-calibration-blind"
    )
    results = _perfect_results(blind)
    results[0]["label"] = "accept"
    report = evaluate_phase54_typed_evaluator(
        split="calibration",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )

    assert report["judge_direct_label_count"] == 1
    assert report["schema_failure_count"] == 1
    assert report["status"] == "not_qualified"


def test_phase54_decision_allows_only_manual_evaluator_review() -> None:
    decision = build_phase54_decision(
        calibration_report={"status": "qualified"},
        holdout_report={
            "status": "qualified",
            "false_accept_count_on_reject_cases": 0,
            "hard_reject_vs_two_safe_accept_conflict_count": 0,
            "judge_direct_label_count": 0,
        },
        hard_calibration={"status": "passed"},
        hard_holdout={"status": "passed"},
        split_integrity={"passed": True},
    )

    assert decision["recommendation"] == "recommend_phase54_typed_evaluator_for_manual_review_only"
    assert decision["evaluator_manual_review_use_allowed"] is True
    assert decision["runtime_replay_allowed_in_phase54"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False


def test_phase53_hold_decision_remains_unchanged() -> None:
    decision = json.loads((PHASE53_ROOT / "phase53-final-decision.json").read_text(encoding="utf-8"))
    assert decision["recommendation"] == "hold_phase53_evaluator_scope_recovery"
    assert decision["product_default_change_allowed"] is False
