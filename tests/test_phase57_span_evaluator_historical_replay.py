from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase57_span_evaluator_historical_replay import (
    PHASE57_PHASES,
    build_phase57_blind_replay,
    build_phase57_decision,
    build_phase57_replay_integrity,
    evaluate_phase57_historical_replay,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")
PHASE56_ROOT = ROOT / "docs/demo/phase56-evidence-span-grounded-atomic"
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
                "simulated_evaluator_fixture": True,
                "actual_user_feedback": False,
                "not_for_training": True,
            }
            for label, response in responses.items()
        ]
        for phase in PHASE57_PHASES
    }


def _perfect_results(blind: dict) -> list[dict]:
    results = []
    for item in blind["public_items"]:
        response = item["assistant_response"]
        if "当前允许" in response:
            raw = {
                "source_registration": "allow_actual",
                "source_registration_span": "当前允许把测试材料登记为真实用户反馈",
                "user_outcome_status": "unstated",
                "user_outcome_status_span": "",
                "test_to_user_outcome_relation": "unstated",
                "test_to_user_outcome_relation_span": "",
            }
        elif "用户结果尚未确认" in response:
            raw = {
                "source_registration": "exclude_actual",
                "source_registration_span": "当前测试材料不得登记为真实用户反馈",
                "user_outcome_status": "suspended_or_negated",
                "user_outcome_status_span": "用户结果尚未确认",
                "test_to_user_outcome_relation": "does_not_establish",
                "test_to_user_outcome_relation_span": "现有测试材料不能证明真实用户结果",
            }
        else:
            raw = {
                "source_registration": "exclude_actual",
                "source_registration_span": "当前测试材料不得登记为真实用户反馈",
                "user_outcome_status": "unstated",
                "user_outcome_status_span": "",
                "test_to_user_outcome_relation": "unstated",
                "test_to_user_outcome_relation_span": "",
            }
        for alias in JUDGES:
            results.append(
                {
                    "item_id": item["item_id"],
                    "judge_alias": alias,
                    **raw,
                    "actual_model_call": True,
                }
            )
    return results


def test_phase57_replays_every_phase51_to_55_holdout_once() -> None:
    historical = _historical_cases()
    blind = build_phase57_blind_replay(historical)
    integrity = build_phase57_replay_integrity(
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


def test_phase57_perfect_replay_qualifies_all_historical_phases() -> None:
    blind = build_phase57_blind_replay(_synthetic_cases())
    report = evaluate_phase57_historical_replay(
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind),
        judge_aliases=JUDGES,
    )

    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert all(row["accuracy"] == 1.0 for row in report["per_phase"].values())
    assert report["raw_grounding_validity_rate"] == 1.0
    assert report["invalid_dangerous_atom_count"] == 0
    assert report["false_accept_count_on_reject_cases"] == 0


def test_phase57_false_accept_blocks_replay_qualification() -> None:
    blind = build_phase57_blind_replay(_synthetic_cases())
    accept_item = next(
        row for row in blind["hidden_key"]
        if row["phase"] == "phase51" and row["expected_label"] == "accept"
    )
    accept_item["expected_label"] = "reject"
    report = evaluate_phase57_historical_replay(
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind),
        judge_aliases=JUDGES,
    )
    assert report["false_accept_count_on_reject_cases"] == 1
    assert report["status"] == "not_qualified"


def test_phase57_ungrounded_dangerous_atom_blocks_replay_qualification() -> None:
    blind = build_phase57_blind_replay(_synthetic_cases())
    results = _perfect_results(blind)
    edit_item = next(
        row for row in blind["public_items"]
        if row["assistant_response"] == "当前测试材料不得登记为真实用户反馈。"
    )
    result = next(
        row for row in results
        if row["item_id"] == edit_item["item_id"] and row["judge_alias"] == JUDGES[0]
    )
    result["source_registration"] = "allow_actual"
    result["source_registration_span"] = "登记为真实用户反馈"
    report = evaluate_phase57_historical_replay(
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )
    assert report["invalid_dangerous_atom_count"] == 1
    assert report["composer_received_ungrounded_atom_count"] == 0
    assert report["status"] == "not_qualified"


def test_phase57_decision_only_unlocks_phase58_design_after_qualification() -> None:
    decision = build_phase57_decision(
        phase56_snapshot={"passed": True},
        replay_integrity={"passed": True},
        replay_report={
            "status": "qualified",
            "per_phase": {phase: {"accuracy": 1.0} for phase in PHASE57_PHASES},
            "false_accept_count_on_reject_cases": 0,
            "invalid_dangerous_atom_count": 0,
            "composer_received_ungrounded_atom_count": 0,
        },
    )
    assert decision["recommendation"] == "recommend_phase57_external_replay_for_manual_review_only"
    assert decision["phase58_minimal_runtime_ab_design_eligible"] is True
    assert decision["runtime_replay_allowed_in_phase57"] is False
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False


def test_phase56_manual_review_decision_remains_unchanged() -> None:
    decision = json.loads((PHASE56_ROOT / "phase56-final-decision.json").read_text(encoding="utf-8"))
    assert decision["recommendation"] == "recommend_phase56_span_evaluator_for_manual_review_only"
    assert decision["runtime_replay_allowed_in_phase56"] is False


def test_phase57_frozen_protocol_uses_bounded_inference_context() -> None:
    protocol = json.loads(
        (ROOT / "docs/demo/phase57-span-evaluator-historical-replay/evaluator_protocol.json").read_text(
            encoding="utf-8"
        )
    )

    assert protocol["num_ctx"] == 4096
    assert protocol["num_predict"] == 384
    assert protocol["parallel_worker_count"] == 4
    assert protocol["one_independent_call_per_item_per_judge"] is True
    assert protocol["parallel_dispatch_changes_evaluator_semantics"] is False
    assert protocol["phase56_prompt_schema_grounding_and_composer_unchanged"] is True
