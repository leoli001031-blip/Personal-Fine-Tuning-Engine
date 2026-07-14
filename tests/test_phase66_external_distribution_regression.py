from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS
from pfe_core.phase59_proposition_addressed_grounding import (
    PHASE59_CATEGORIES,
    build_phase59_proposition_candidates,
)
from pfe_core.phase66_external_distribution_regression import (
    PHASE66_EXTERNAL_HOLDOUT_COUNT,
    PHASE66_HISTORICAL_REPLAY_COUNT,
    build_phase66_decision,
    build_phase66_external_blind_items,
    build_phase66_external_holdout_cases,
    build_phase66_external_integrity,
    build_phase66_historical_blind_replay,
    build_phase66_historical_integrity,
    build_phase66_preflight_items,
    evaluate_phase66_external_holdout,
    evaluate_phase66_historical_replay,
)


JUDGES = ("semantic_judge_alpha", "semantic_judge_beta")
PHASE65_ROOT = ROOT / "docs/demo/phase65-aggregate-safe-boundary-coverage"
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
            (root / "evidence-evaluator-holdout/holdout_labeled.json").read_text(
                encoding="utf-8"
            )
        )["cases"]
        for phase, root in HISTORICAL_ROOTS.items()
    }


def _phase65_cases() -> list[dict]:
    rows = []
    for split in ("calibration", "holdout"):
        rows.extend(
            json.loads(
                (
                    PHASE65_ROOT
                    / f"evidence-evaluator-{split}/{split}_labeled.json"
                ).read_text(encoding="utf-8")
            )["cases"]
        )
    return rows


def _perfect_results(public_items: list[dict], hidden_key: list[dict]) -> list[dict]:
    hidden = {row["item_id"]: row for row in hidden_key}
    results = []
    for item in public_items:
        candidates = build_phase59_proposition_candidates(item["assistant_response"])
        key = hidden[item["item_id"]]
        selection = {
            f"{field}_candidate_id": key["expected_candidate_ids"][field]
            for field in PHASE56_TYPED_FIELDS
        }
        assert all(
            selection[f"{field}_candidate_id"] == "none"
            or any(
                row["candidate_id"] == selection[f"{field}_candidate_id"]
                and row["field"] == field
                for row in candidates
            )
            for field in PHASE56_TYPED_FIELDS
        )
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


def test_phase66_external_holdout_is_fresh_balanced_and_non_training() -> None:
    payload = build_phase66_external_holdout_cases()

    assert payload["case_count"] == PHASE66_EXTERNAL_HOLDOUT_COUNT
    assert payload["category_counts"] == {
        category: 30 for category in PHASE59_CATEGORIES
    }
    assert payload["label_counts"] == {"accept": 50, "edit": 50, "reject": 50}
    assert all(row["actual_user_feedback"] is False for row in payload["cases"])
    assert all(row["not_for_training"] is True for row in payload["cases"])


def test_phase66_external_integrity_excludes_phase65_and_history() -> None:
    external = build_phase66_external_holdout_cases()["cases"]
    historical = _historical_cases()
    integrity = build_phase66_external_integrity(
        external,
        historical_cases=[row for rows in historical.values() for row in rows],
        phase65_cases=_phase65_cases(),
        preflight_items=build_phase66_preflight_items()["items"],
    )

    assert integrity["passed"] is True
    assert integrity["external_holdout_count"] == PHASE66_EXTERNAL_HOLDOUT_COUNT
    assert integrity["historical_replay_count"] == PHASE66_HISTORICAL_REPLAY_COUNT


def test_phase66_perfect_external_candidate_selection_qualifies() -> None:
    cases = build_phase66_external_holdout_cases()["cases"]
    blind = build_phase66_external_blind_items(cases)
    report = evaluate_phase66_external_holdout(
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=_perfect_results(blind["public_items"], blind["hidden_key"]),
        judge_aliases=JUDGES,
    )

    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert report["candidate_value_conflict_count"] == 0
    assert report["false_accept_count_on_reject_cases"] == 0


def test_phase66_historical_replay_preserves_all_558_cases() -> None:
    historical = _historical_cases()
    blind = build_phase66_historical_blind_replay(historical)
    integrity = build_phase66_historical_integrity(
        historical_cases=historical,
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
    )

    assert len(blind["public_items"]) == PHASE66_HISTORICAL_REPLAY_COUNT
    assert integrity["passed"] is True


def test_phase66_historical_wrapper_preserves_strict_phase64_gates() -> None:
    synthetic = {
        phase: [
            {
                "case_id": f"{phase}-{label}",
                "category": "synthetic_boundary",
                "context": "只审查回答原文。",
                "assistant_response": response,
                "expected_label": label,
                "expected_explicit_hard_reject": label == "reject",
            }
            for label, response in {
                "accept": (
                    "当前测试材料不得登记为真实用户反馈。用户结果尚未确认。"
                    "现有测试材料不能证明真实用户结果。"
                ),
                "edit": "当前测试材料不得登记为真实用户反馈。",
                "reject": "当前允许把测试材料登记为真实用户反馈。",
            }.items()
        ]
        for phase in HISTORICAL_ROOTS
    }
    blind = build_phase66_historical_blind_replay(synthetic)
    hidden_with_typed = []
    for key in blind["hidden_key"]:
        item = next(row for row in blind["public_items"] if row["item_id"] == key["item_id"])
        expected_ids = {}
        for field in PHASE56_TYPED_FIELDS:
            options = [
                row
                for row in build_phase59_proposition_candidates(item["assistant_response"])
                if row["field"] == field
            ]
            expected_ids[field] = options[0]["candidate_id"] if options else "none"
        hidden_with_typed.append({**key, "expected_candidate_ids": expected_ids})
    results = _perfect_results(blind["public_items"], hidden_with_typed)
    report = evaluate_phase66_historical_replay(
        public_items=blind["public_items"],
        hidden_key=blind["hidden_key"],
        judge_results=results,
        judge_aliases=JUDGES,
    )

    assert report["status"] == "qualified"
    assert report["accuracy"] == 1.0
    assert report["material_accuracy_improvement"] is True
    assert report["overall_accuracy_gate"] == 0.95


def test_phase66_decision_requires_both_external_distributions() -> None:
    clean_report = {
        "status": "qualified",
        "false_accept_count_on_reject_cases": 0,
        "schema_failure_count": 0,
        "candidate_value_conflict_count": 0,
    }
    decision = build_phase66_decision(
        phase65_snapshot={"passed": True},
        external_integrity={"passed": True},
        historical_integrity={"passed": True},
        preflight_report={"status": "passed"},
        external_report=clean_report,
        historical_report={**clean_report, "material_accuracy_improvement": True},
        external_audit={"status": "passed"},
        external_hard_compatibility={"status": "passed"},
    )

    assert decision["recommendation"] == (
        "recommend_phase66_external_distribution_regression_for_manual_review_only"
    )
    assert decision["phase67_minimal_runtime_ab_design_eligible"] is True
    assert decision["runtime_replay_allowed_in_phase66"] is False
    assert decision["new_training_allowed"] is False

    held = build_phase66_decision(
        phase65_snapshot={"passed": True},
        external_integrity={"passed": True},
        historical_integrity={"passed": True},
        preflight_report={"status": "passed"},
        external_report={**clean_report, "candidate_value_conflict_count": 1},
        historical_report={**clean_report, "material_accuracy_improvement": True},
        external_audit={"status": "passed"},
        external_hard_compatibility={"status": "passed"},
    )
    assert held["recommendation"] == "hold_phase66_external_distribution_regression"


def test_phase65_manual_review_decision_remains_unchanged() -> None:
    decision = json.loads(
        (PHASE65_ROOT / "phase65-final-decision.json").read_text(encoding="utf-8")
    )
    assert decision["recommendation"] == (
        "recommend_phase65_scope_aware_candidates_for_manual_review_only"
    )
    assert decision["phase66_external_regression_design_eligible"] is True
