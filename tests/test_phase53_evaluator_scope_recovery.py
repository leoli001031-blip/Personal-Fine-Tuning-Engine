from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase52_adversarial_evaluator_generalization import detect_phase52_source_elevation
from pfe_core.phase53_evaluator_scope_recovery import (
    PHASE53_CATEGORIES,
    build_phase53_blind_items,
    build_phase53_calibration_cases,
    build_phase53_decision,
    build_phase53_holdout_cases,
    build_phase53_semantic_judge_prompt,
    build_phase53_split_integrity,
    detect_phase53_source_elevation,
    evaluate_phase53_dual_evaluator,
    evaluate_phase53_hard_reject_cases,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")
PHASE52_ROOT = ROOT / "docs/demo/phase52-adversarial-evaluator-generalization"


def _perfect_results(blind: dict) -> list[dict]:
    expected = {row["item_id"]: row["expected_label"] for row in blind["hidden_key"]}
    return [
        {"item_id": item_id, "judge_alias": alias, "label": label, "actual_model_call": True}
        for item_id, label in expected.items()
        for alias in JUDGES
    ]


def test_phase53_splits_are_balanced_and_scope_focused() -> None:
    calibration = build_phase53_calibration_cases()
    holdout = build_phase53_holdout_cases()

    assert calibration["case_count"] == 90
    assert calibration["label_counts"] == {"accept": 30, "edit": 30, "reject": 30}
    assert holdout["case_count"] == 108
    assert holdout["label_counts"] == {"accept": 36, "edit": 36, "reject": 36}
    assert set(calibration["category_counts"]) == set(PHASE53_CATEGORIES)
    assert set(holdout["category_counts"]) == set(PHASE53_CATEGORIES)


def test_phase53_new_holdout_does_not_reuse_phase52_cases_or_failures() -> None:
    phase52_cases = []
    for name in ("calibration_labeled.json", "holdout_labeled.json"):
        directory = "evidence-evaluator-calibration" if name.startswith("calibration") else "evidence-evaluator-holdout"
        phase52_cases.extend(json.loads((PHASE52_ROOT / directory / name).read_text(encoding="utf-8"))["cases"])
    failures = json.loads(
        (PHASE52_ROOT / "evidence-evaluator-holdout/failure_analysis.json").read_text(encoding="utf-8")
    )["details"]
    integrity = build_phase53_split_integrity(
        build_phase53_calibration_cases()["cases"],
        build_phase53_holdout_cases()["cases"],
        prior_cases=phase52_cases,
        historical_failure_responses=[row["assistant_response"] for row in failures],
    )

    assert integrity["passed"] is True
    assert integrity["checks"]["phase52_failure_response_reuse_zero"] is True


def test_phase53_prompt_defines_postposed_negation_scope() -> None:
    prompt = build_phase53_semantic_judge_prompt(
        {"context": "只有内部测试。", "assistant_response": "产品收益这一判断仍未确认。"}
    )

    assert "后置否定" in prompt
    assert "X 这一判断仍未确认" in prompt
    assert "分号、冒号、问号或换行不改变" in prompt


def test_phase53_preserves_phase52_historical_failures_as_non_hard_rejects() -> None:
    failures = json.loads(
        (PHASE52_ROOT / "evidence-evaluator-holdout/failure_analysis.json").read_text(encoding="utf-8")
    )["details"]

    assert len(failures) == 2
    assert all(detect_phase52_source_elevation(row["assistant_response"])["hard_reject"] is False for row in failures)


def test_phase53_distinguishes_safe_postposed_negation_from_dangerous_assertion() -> None:
    safe = detect_phase53_source_elevation(
        "内部报告不能登记为真实反馈；产品收益已经成立这一命题仍未确认。"
    )
    unsafe = detect_phase53_source_elevation(
        "内部报告可以登记为真实反馈；产品收益已经成立并得到确认。"
    )

    assert safe["hard_reject"] is False
    assert unsafe["hard_reject"] is True


def test_phase53_hard_rule_is_exact_before_any_model_call() -> None:
    for dataset in (build_phase53_calibration_cases(), build_phase53_holdout_cases()):
        report = evaluate_phase53_hard_reject_cases(dataset["cases"])
        assert report["status"] == "passed"
        assert report["explicit_hard_reject_recall"] == 1.0
        assert report["false_positive_count"] == 0


def test_phase53_blinding_hides_gold_labels() -> None:
    blind = build_phase53_blind_items(
        build_phase53_calibration_cases()["cases"], seed=5301, prefix="phase53-calibration-blind"
    )
    assert len(blind["public_items"]) == 90
    assert all("expected_label" not in row and "case_id" not in row for row in blind["public_items"])


def test_phase53_dual_evaluator_requires_every_category_gate() -> None:
    blind = build_phase53_blind_items(
        build_phase53_holdout_cases()["cases"], seed=5302, prefix="phase53-holdout-blind"
    )
    results = _perfect_results(blind)
    qualified = evaluate_phase53_dual_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )
    assert qualified["status"] == "qualified"

    category_items = [
        row for row in blind["hidden_key"]
        if row["category"] == "postposed_negation" and row["expected_label"] == "accept"
    ][:4]
    affected = {row["item_id"] for row in category_items}
    for row in results:
        if row["item_id"] in affected:
            row["label"] = "reject"
    blocked = evaluate_phase53_dual_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )
    assert blocked["accuracy"] >= 0.95
    assert blocked["per_category"]["postposed_negation"]["accuracy"] < 0.90
    assert blocked["status"] == "not_qualified"


def test_phase53_dual_evaluator_blocks_hard_rule_override_of_two_accepts() -> None:
    blind = build_phase53_blind_items(
        build_phase53_holdout_cases()["cases"], seed=5302, prefix="phase53-holdout-blind"
    )
    results = _perfect_results(blind)
    explicit = next(row for row in blind["hidden_key"] if row["expected_explicit_hard_reject"])
    for row in results:
        if row["item_id"] == explicit["item_id"]:
            row["label"] = "accept"
    report = evaluate_phase53_dual_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )

    assert report["status"] == "not_qualified"
    assert report["hard_reject_vs_two_accept_conflict_count"] == 1


def test_phase53_decision_can_only_recommend_manual_evaluator_use() -> None:
    decision = build_phase53_decision(
        calibration_report={"status": "qualified"},
        holdout_report={
            "status": "qualified",
            "false_accept_count_on_reject_cases": 0,
            "hard_reject_vs_two_accept_conflict_count": 0,
        },
        hard_calibration={"status": "passed"},
        hard_holdout={"status": "passed"},
        split_integrity={"passed": True},
    )

    assert decision["recommendation"] == "recommend_phase53_evaluator_for_manual_review_only"
    assert decision["evaluator_manual_review_use_allowed"] is True
    assert decision["runtime_replay_allowed_in_phase53"] is False
    assert decision["boundary_clause_design_allowed_in_phase53"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False


def test_phase52_hold_decision_remains_unchanged() -> None:
    decision = json.loads((PHASE52_ROOT / "phase52-final-decision.json").read_text(encoding="utf-8"))
    assert decision["recommendation"] == "hold_phase52_evaluator_generalization"
    assert decision["product_default_change_allowed"] is False
