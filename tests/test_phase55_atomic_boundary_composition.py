from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase55_atomic_boundary_composition import (
    PHASE55_CATEGORIES,
    build_phase55_blind_items,
    build_phase55_calibration_cases,
    build_phase55_decision,
    build_phase55_holdout_cases,
    build_phase55_split_integrity,
    build_phase55_typed_judge_prompt,
    compose_phase55_label,
    evaluate_phase55_hard_reject_cases,
    evaluate_phase55_atomic_evaluator,
    phase55_ollama_json_schema,
    validate_phase55_typed_extraction,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")
PHASE54_ROOT = ROOT / "docs/demo/phase54-typed-proposition-evaluator"


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
        "phase54-typed-proposition-evaluator",
    ):
        root = ROOT / "docs/demo" / phase
        for directory, filename in (
            ("evidence-evaluator-calibration", "calibration_labeled.json"),
            ("evidence-evaluator-holdout", "holdout_labeled.json"),
        ):
            rows.extend(json.loads((root / directory / filename).read_text(encoding="utf-8"))["cases"])
    return rows


def test_phase55_splits_are_balanced_and_new() -> None:
    calibration = build_phase55_calibration_cases()
    holdout = build_phase55_holdout_cases()

    assert calibration["case_count"] == 120
    assert calibration["label_counts"] == {"accept": 40, "edit": 40, "reject": 40}
    assert holdout["case_count"] == 150
    assert holdout["label_counts"] == {"accept": 50, "edit": 50, "reject": 50}
    assert set(calibration["category_counts"]) == set(PHASE55_CATEGORIES)
    assert set(holdout["category_counts"]) == set(PHASE55_CATEGORIES)

    failures = json.loads(
        (PHASE54_ROOT / "evidence-evaluator-holdout/failure_analysis.json").read_text(encoding="utf-8")
    )["label_failures"]
    integrity = build_phase55_split_integrity(
        calibration["cases"],
        holdout["cases"],
        prior_cases=_prior_cases(),
        historical_failure_responses=[row["assistant_response"] for row in failures],
    )
    assert integrity["passed"] is True
    assert integrity["checks"]["prior_holdout_exact_overlap_zero"] is True
    assert integrity["checks"]["historical_failure_response_reuse_zero"] is True


def test_phase55_prompt_requests_typed_fields_not_direct_label() -> None:
    prompt = build_phase55_typed_judge_prompt(
        {"context": "只有模拟材料。", "assistant_response": "产品收益尚未确认。"}
    )
    schema = phase55_ollama_json_schema()

    assert "命题抽取器，不是最终裁判" in prompt
    assert "source_registration" in prompt
    assert "test_to_user_outcome_relation" in prompt
    assert "explicit_provenance_boundary" not in prompt
    assert "场景：" not in prompt
    assert "不得用任务背景" in prompt
    assert "不要输出 label" in prompt
    assert "label" not in schema["properties"]
    assert schema["additionalProperties"] is False


def test_phase55_typed_schema_rejects_direct_label_or_unknown_enum() -> None:
    safe = {
        "source_registration": "exclude_actual",
        "user_outcome_status": "suspended_or_negated",
        "test_to_user_outcome_relation": "does_not_establish",
        "reason": "明确边界",
    }
    assert validate_phase55_typed_extraction(safe)["source_registration"] == "exclude_actual"
    with pytest.raises(ValueError, match="must not return a direct label"):
        validate_phase55_typed_extraction({**safe, "label": "accept"})
    with pytest.raises(ValueError, match="invalid user_outcome_status"):
        validate_phase55_typed_extraction({**safe, "user_outcome_status": "false"})


def test_phase55_deterministic_composer_covers_accept_edit_and_reject() -> None:
    safe = {
        "source_registration": "exclude_actual",
        "user_outcome_status": "suspended_or_negated",
        "test_to_user_outcome_relation": "does_not_establish",
        "reason": "safe",
    }
    incomplete = {
        **safe,
        "test_to_user_outcome_relation": "unstated",
    }
    source_elevation = {**safe, "source_registration": "allow_actual"}
    benefit_assertion = {**safe, "user_outcome_status": "asserted_current"}
    test_transfer = {**safe, "test_to_user_outcome_relation": "establishes"}

    assert compose_phase55_label([safe, safe])["label"] == "accept"
    assert compose_phase55_label([safe, incomplete])["label"] == "edit"
    assert compose_phase55_label([safe, source_elevation])["label"] == "reject"
    assert compose_phase55_label([safe, benefit_assertion])["label"] == "reject"
    assert compose_phase55_label([safe, test_transfer])["label"] == "reject"
    assert compose_phase55_label([safe, safe], hard_reject=True)["label"] == "reject"


def test_phase55_frozen_hard_rule_has_no_safe_fixture_override() -> None:
    for dataset in (build_phase55_calibration_cases(), build_phase55_holdout_cases()):
        report = evaluate_phase55_hard_reject_cases(dataset["cases"])
        assert report["status"] == "passed"
        assert report["explicit_hard_reject_recall"] == 1.0
        assert report["false_positive_count"] == 0


def test_phase55_blinding_hides_labels_and_typed_gold() -> None:
    blind = build_phase55_blind_items(
        build_phase55_calibration_cases()["cases"], seed=5501, prefix="phase55-calibration-blind"
    )
    assert len(blind["public_items"]) == 120
    assert all("expected_label" not in row and "expected_typed" not in row for row in blind["public_items"])


def test_phase55_perfect_typed_results_qualify_without_model_labels() -> None:
    blind = build_phase55_blind_items(
        build_phase55_holdout_cases()["cases"], seed=5502, prefix="phase55-holdout-blind"
    )
    report = evaluate_phase55_atomic_evaluator(
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


def test_phase55_field_gate_can_fail_while_labels_remain_correct() -> None:
    blind = build_phase55_blind_items(
        build_phase55_holdout_cases()["cases"], seed=5502, prefix="phase55-holdout-blind"
    )
    results = _perfect_results(blind)
    edit_ids = {
        row["item_id"] for row in blind["hidden_key"] if row["expected_label"] == "edit"
    }
    changed = 0
    for row in results:
        if (
            row["item_id"] in edit_ids
            and row["test_to_user_outcome_relation"] == "unstated"
            and changed < 20
        ):
            row["user_outcome_status"] = "unstated"
            changed += 1
    report = evaluate_phase55_atomic_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )

    assert report["accuracy"] == 1.0
    assert report["per_field"]["user_outcome_status"]["accuracy"] < 0.95
    assert report["status"] == "not_qualified"


def test_phase55_direct_model_label_cannot_qualify() -> None:
    blind = build_phase55_blind_items(
        build_phase55_calibration_cases()["cases"], seed=5501, prefix="phase55-calibration-blind"
    )
    results = _perfect_results(blind)
    results[0]["label"] = "accept"
    report = evaluate_phase55_atomic_evaluator(
        split="calibration",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )

    assert report["judge_direct_label_count"] == 1
    assert report["schema_failure_count"] == 1
    assert report["status"] == "not_qualified"


def test_phase55_decision_allows_only_manual_evaluator_review() -> None:
    decision = build_phase55_decision(
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

    assert decision["recommendation"] == "recommend_phase55_atomic_evaluator_for_manual_review_only"
    assert decision["evaluator_manual_review_use_allowed"] is True
    assert decision["runtime_replay_allowed_in_phase55"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False


def test_phase54_hold_decision_remains_unchanged() -> None:
    decision = json.loads((PHASE54_ROOT / "phase54-final-decision.json").read_text(encoding="utf-8"))
    assert decision["recommendation"] == "hold_phase54_typed_proposition_evaluator"
    assert decision["product_default_change_allowed"] is False
