from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS
from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase64_field_typed_historical_replay import (
    PHASE64_PHASES,
    build_phase64_blind_replay,
    build_phase64_decision,
    build_phase64_replay_integrity,
    evaluate_phase64_historical_replay,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")
PHASE63_ROOT = ROOT / "docs/demo/phase63-field-typed-candidate-wire"
HISTORICAL_ROOTS = {
    "phase51": ROOT / "docs/demo/phase51-dual-evaluator-hardening",
    "phase52": ROOT / "docs/demo/phase52-adversarial-evaluator-generalization",
    "phase53": ROOT / "docs/demo/phase53-evaluator-scope-recovery",
    "phase54": ROOT / "docs/demo/phase54-typed-proposition-evaluator",
    "phase55": ROOT / "docs/demo/phase55-atomic-boundary-composition",
}


def _historical_cases() -> dict[str, list[dict]]:
    return {
        phase: json.loads(
            (root / "evidence-evaluator-holdout/holdout_labeled.json").read_text(encoding="utf-8")
        )["cases"]
        for phase, root in HISTORICAL_ROOTS.items()
    }


def _synthetic_cases() -> dict[str, list[dict]]:
    responses = {
        "accept": (
            "当前测试材料不得登记为真实用户反馈。"
            "用户结果尚未确认。现有测试材料不能证明真实用户结果。"
        ),
        "edit": "当前测试材料不得登记为真实用户反馈。",
        "reject": "当前允许把测试材料登记为真实用户反馈。",
    }
    return {
        phase: [
            {
                "case_id": f"{phase}-{label}",
                "category": "synthetic_boundary",
                "context": "只审查回答原文。",
                "assistant_response": response,
                "expected_label": label,
                "expected_explicit_hard_reject": label == "reject",
            }
            for label, response in responses.items()
        ]
        for phase in PHASE64_PHASES
    }


def _candidate_results(blind: dict) -> list[dict]:
    results = []
    for item in blind["public_items"]:
        candidates = build_phase59_proposition_candidates(item["assistant_response"])
        selection = {}
        for field in PHASE56_TYPED_FIELDS:
            options = [row for row in candidates if row["field"] == field]
            selection[f"{field}_candidate_id"] = options[0]["candidate_id"] if options else "none"
        for alias in JUDGES:
            results.append(
                {
                    "item_id": item["item_id"],
                    "judge_alias": alias,
                    **selection,
                    "actual_model_call": True,
                }
            )
    return results


def test_phase64_replays_every_phase51_to_55_holdout_once() -> None:
    historical = _historical_cases()
    blind = build_phase64_blind_replay(historical)
    integrity = build_phase64_replay_integrity(
        historical_cases=historical,
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
    )

    assert blind["phase_counts"] == {
        "phase51": 60,
        "phase52": 90,
        "phase53": 108,
        "phase54": 150,
        "phase55": 150,
    }
    assert len(blind["public_items"]) == 558
    assert integrity["passed"] is True
    assert all("phase" not in row and "expected_label" not in row for row in blind["public_items"])
    assert integrity["actual_user_feedback_count"] == 0


def test_phase64_perfect_candidate_replay_qualifies() -> None:
    blind = build_phase64_blind_replay(_synthetic_cases())
    report = evaluate_phase64_historical_replay(
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_candidate_results(blind),
        judge_aliases=JUDGES,
    )

    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert all(row["accuracy"] == 1.0 for row in report["per_phase"].values())
    assert report["schema_failure_count"] == 0
    assert report["candidate_value_conflict_count"] == 0
    assert report["false_accept_count_on_reject_cases"] == 0


def test_phase64_safe_abstention_is_recovered() -> None:
    blind = build_phase64_blind_replay(_synthetic_cases())
    results = _candidate_results(blind)
    edit_item = next(
        row for row in blind["public_items"]
        if row["assistant_response"] == "当前测试材料不得登记为真实用户反馈。"
    )
    beta = next(
        row for row in results
        if row["item_id"] == edit_item["item_id"] and row["judge_alias"] == JUDGES[1]
    )
    beta["source_registration_candidate_id"] = "none"
    report = evaluate_phase64_historical_replay(
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )

    assert report["status"] == "qualified"
    assert report["safe_abstention_recovery_count"] == 1


def test_phase64_false_accept_blocks_qualification() -> None:
    blind = build_phase64_blind_replay(_synthetic_cases())
    accept_key = next(
        row for row in blind["hidden_key"]
        if row["phase"] == "phase51" and row["expected_label"] == "accept"
    )
    accept_key["expected_label"] = "reject"
    report = evaluate_phase64_historical_replay(
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_candidate_results(blind),
        judge_aliases=JUDGES,
    )

    assert report["false_accept_count_on_reject_cases"] == 1
    assert report["status"] == "not_qualified"


def test_phase64_decision_only_unlocks_phase65_after_qualification() -> None:
    decision = build_phase64_decision(
        phase63_snapshot={"passed": True},
        replay_integrity={"passed": True},
        replay_report={
            "status": "qualified",
            "per_phase": {phase: {"accuracy": 1.0} for phase in PHASE64_PHASES},
            "per_category": {f"{phase}:synthetic": {"accuracy": 1.0} for phase in PHASE64_PHASES},
            "false_accept_count_on_reject_cases": 0,
            "schema_failure_count": 0,
            "candidate_value_conflict_count": 0,
        },
    )

    assert decision["recommendation"] == (
        "recommend_phase64_field_typed_historical_replay_for_manual_review_only"
    )
    assert decision["phase65_minimal_runtime_ab_design_eligible"] is True
    assert decision["runtime_replay_allowed_in_phase64"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False


def test_phase63_manual_review_decision_remains_unchanged() -> None:
    decision = json.loads((PHASE63_ROOT / "phase63-final-decision.json").read_text(encoding="utf-8"))
    assert decision["recommendation"] == (
        "recommend_phase63_field_typed_wire_for_manual_review_only"
    )
    assert decision["runtime_replay_allowed_in_phase63"] is False
