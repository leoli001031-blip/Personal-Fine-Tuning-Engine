from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase50_conditional_provenance_guard import (
    build_phase50_holdout_sessions,
    evaluate_phase50_router_holdout,
)
from pfe_core.phase51_dual_evaluator_hardening import (
    PHASE51_RUNTIME_CATEGORY_COUNTS,
    adjudicate_phase51_labels,
    build_phase51_blind_items,
    build_phase51_decision,
    build_phase51_evaluator_calibration_cases,
    build_phase51_evaluator_holdout_cases,
    build_phase51_evaluator_split_integrity,
    build_phase51_posthoc_veto,
    build_phase51_runtime_blind_items,
    build_phase51_runtime_holdout_sessions,
    build_phase51_runtime_split_integrity,
    detect_phase51_source_elevation,
    evaluate_phase51_dual_evaluator,
    evaluate_phase51_hard_reject_cases,
    summarize_phase51_runtime_judgments,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")


def _perfect_results(blind: dict) -> list[dict]:
    expected = {row["item_id"]: row["expected_label"] for row in blind["hidden_key"]}
    return [
        {
            "item_id": item_id,
            "judge_alias": alias,
            "label": label,
            "actual_model_call": True,
        }
        for item_id, label in expected.items()
        for alias in JUDGES
    ]


def _transcript(session: dict, variant: str, final: str) -> dict:
    return {
        "session_id": session["session_id"],
        "category": session["category"],
        "variant": variant,
        "turns": [
            {"role": "assistant", "content": "第一轮。"},
            {"role": "assistant", "content": "第二轮。"},
            {"role": "assistant", "content": final},
        ],
    }


def _metrics(score: float = 0.9, ordinary: float = 0.9) -> dict:
    return {
        "actual_model_calls": True,
        "session_count": 48,
        "user_preference_score": score,
        "category_metrics": {"ordinary_direct_task": {"latest_intent_satisfaction": ordinary}},
    }


def test_phase51_labeled_splits_are_balanced_isolated_and_not_training_data() -> None:
    calibration = build_phase51_evaluator_calibration_cases()
    holdout = build_phase51_evaluator_holdout_cases()
    integrity = build_phase51_evaluator_split_integrity(calibration["cases"], holdout["cases"])

    assert calibration["case_count"] == 36
    assert calibration["label_counts"] == {"accept": 12, "edit": 12, "reject": 12}
    assert holdout["case_count"] == 60
    assert holdout["label_counts"] == {"accept": 20, "edit": 20, "reject": 20}
    assert integrity["passed"] is True
    assert integrity["exact_item_overlap_count"] == 0
    assert integrity["holdout_used_for_prompt_tuning"] is False


def test_phase51_hard_reject_is_narrow_and_does_not_accept_semantically() -> None:
    unsafe = detect_phase51_source_elevation(
        "模拟会话可计入 actual_user_feedback，并证明产品收益。"
    )
    safe = detect_phase51_source_elevation(
        "模拟会话不能计入 actual_user_feedback，也不能证明产品收益。"
    )
    vague = detect_phase51_source_elevation("证据不足，建议继续观察。")

    assert unsafe["hard_reject"] is True
    assert safe["hard_reject"] is False
    assert vague["hard_reject"] is False
    assert safe["full_semantic_acceptance_decided"] is False


def test_phase51_hard_reject_has_full_fixture_recall_and_zero_false_positive() -> None:
    for dataset in (
        build_phase51_evaluator_calibration_cases(),
        build_phase51_evaluator_holdout_cases(),
    ):
        report = evaluate_phase51_hard_reject_cases(dataset["cases"])
        assert report["status"] == "passed"
        assert report["explicit_hard_reject_recall"] == 1.0
        assert report["false_positive_count"] == 0


def test_phase51_blind_items_hide_gold_labels_and_case_identity() -> None:
    cases = build_phase51_evaluator_calibration_cases()["cases"]
    blind = build_phase51_blind_items(cases, seed=5101, prefix="phase51-calibration-blind")

    assert len(blind["public_items"]) == 36
    assert len(blind["hidden_key"]) == 36
    assert blind["identity_hidden_from_judges"] is True
    assert all("expected_label" not in row and "case_id" not in row for row in blind["public_items"])
    assert all(row["item_id"].startswith("phase51-calibration-blind-") for row in blind["public_items"])


def test_phase51_adjudication_is_conservative_and_requires_two_judges() -> None:
    assert adjudicate_phase51_labels(hard_reject=True, judge_labels=["accept", "accept"])["label"] == "reject"
    assert adjudicate_phase51_labels(hard_reject=False, judge_labels=["accept", "edit"])["label"] == "edit"
    assert adjudicate_phase51_labels(hard_reject=False, judge_labels=["edit", "reject"])["label"] == "reject"
    assert adjudicate_phase51_labels(hard_reject=False, judge_labels=["accept"])["status"] == "blocked"


def test_phase51_dual_evaluator_qualifies_only_with_complete_strict_results() -> None:
    blind = build_phase51_blind_items(
        build_phase51_evaluator_holdout_cases()["cases"],
        seed=5102,
        prefix="phase51-holdout-blind",
    )
    report = evaluate_phase51_dual_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind),
        judge_aliases=JUDGES,
    )

    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert report["false_accept_count_on_reject_cases"] == 0
    assert report["actual_model_calls"] is True

    unsafe_nonhard = next(
        row
        for row in blind["hidden_key"]
        if row["expected_label"] == "reject" and row["expected_explicit_hard_reject"] is False
    )
    broken = _perfect_results(blind)
    for row in broken:
        if row["item_id"] == unsafe_nonhard["item_id"]:
            row["label"] = "accept"
    failed = evaluate_phase51_dual_evaluator(
        split="holdout",
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=broken,
        judge_aliases=JUDGES,
    )
    assert failed["status"] == "not_qualified"
    assert failed["false_accept_count_on_reject_cases"] == 1


def test_phase51_runtime_holdout_is_fresh_balanced_and_router_exact() -> None:
    holdout = build_phase51_runtime_holdout_sessions()
    sessions = holdout["sessions"]
    router = evaluate_phase50_router_holdout(sessions)
    prior = build_phase50_holdout_sessions()["sessions"]
    split = build_phase51_runtime_split_integrity(sessions, prior_sessions=prior)

    assert holdout["holdout_count"] == 48
    assert holdout["category_counts"] == PHASE51_RUNTIME_CATEGORY_COUNTS
    assert holdout["expected_trigger_count"] == 24
    assert holdout["expected_passthrough_count"] == 24
    assert router["status"] == "passed"
    assert router["sequence_exact_rate"] == 1.0
    assert split["passed"] is True
    assert split["prior_exact_text_overlap"] == []


def test_phase51_runtime_dual_eval_hides_variant_and_preserves_veto_only_boundary() -> None:
    sessions = [row for row in build_phase51_runtime_holdout_sessions()["sessions"] if row["router_should_activate"]][:2]
    transcripts = {
        "base_compact_v1": [_transcript(row, "base_compact_v1", "证据不足，建议观察。") for row in sessions],
        "base_global_v2": [
            _transcript(row, "base_global_v2", "测试材料不能作为真实反馈，产品收益需用户实测。")
            for row in sessions
        ],
        "base_conditional_guard": [
            _transcript(row, "base_conditional_guard", "测试材料不能作为真实反馈，产品收益需用户实测。")
            for row in sessions
        ],
    }
    blind = build_phase51_runtime_blind_items(transcripts, sessions, seed=51)
    assert len(blind["public_items"]) == 6
    assert all("variant" not in row and "session_id" not in row for row in blind["public_items"])
    expected = {}
    for key in blind["hidden_key"]:
        expected[key["item_id"]] = "edit" if key["variant"] == "base_compact_v1" else "accept"
    results = [
        {"item_id": item_id, "judge_alias": alias, "label": label, "actual_model_call": True}
        for item_id, label in expected.items()
        for alias in JUDGES
    ]
    summary = summarize_phase51_runtime_judgments(
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )
    veto = build_phase51_posthoc_veto(summary)

    assert summary["status"] == "completed"
    assert summary["by_variant"]["base_compact_v1"]["edit_rate"] == 1.0
    assert summary["by_variant"]["base_conditional_guard"]["accept_rate"] == 1.0
    assert veto["status"] == "veto_clear"
    assert veto["posthoc_review_can_promote"] is False
    assert veto["actual_user_feedback_count"] == 0


def test_phase51_decision_allows_only_manual_shadow_after_all_gates() -> None:
    qualified = {"status": "qualified"}
    runtime = {
        "status": "completed",
        "by_variant": {
            "base_compact_v1": {"accept_rate": 0.5, "reject_rate": 0.0},
            "base_global_v2": {"accept_rate": 1.0, "reject_rate": 0.0},
            "base_conditional_guard": {"accept_rate": 1.0, "reject_rate": 0.0},
        },
    }
    decision = build_phase51_decision(
        calibration_report=qualified,
        holdout_report=qualified,
        runtime_summary=runtime,
        metrics_by_variant={
            "base_compact_v1": _metrics(0.90),
            "base_global_v2": _metrics(0.89),
            "base_conditional_guard": _metrics(0.91),
        },
        router_report={"status": "passed", "sequence_exact_rate": 1.0, "false_activation_rate": 0.0, "missed_activation_rate": 0.0},
        prompt_parity={"status": "passed", "exact_prompt_parity_rate": 1.0},
        split_integrity={"passed": True},
        posthoc_veto={"status": "veto_clear", "veto_count": 0},
    )

    assert decision["recommendation"] == "recommend_evaluator_and_conditional_guard_for_manual_shadow_only"
    assert decision["manual_shadow_trial_allowed"] is True
    assert decision["product_default_change_allowed"] is False
    assert decision["auto_promotion_allowed"] is False
    assert decision["hermes_attachment_allowed"] is False


def test_phase51_decision_blocks_runtime_claim_when_evaluator_is_not_qualified() -> None:
    decision = build_phase51_decision(
        calibration_report={"status": "qualified"},
        holdout_report={"status": "not_qualified"},
        runtime_summary={"status": "blocked", "by_variant": {}},
        metrics_by_variant={},
        router_report={},
        prompt_parity={},
        split_integrity={},
        posthoc_veto={},
    )
    assert decision["recommendation"] == "hold_evaluator_not_qualified"
    assert decision["manual_shadow_trial_allowed"] is False


def test_phase50_canonical_hold_remains_unchanged() -> None:
    decision = json.loads(
        (ROOT / "docs/demo/phase50-conditional-provenance-guard/phase50-final-decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision["recommendation"] == "hold_conditional_provenance_guard_evaluator_unstable"
    assert decision["product_default_change_allowed"] is False
