from __future__ import annotations

from pathlib import Path
import sys


CORE_ROOT = Path(__file__).resolve().parents[1] / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase103_simulated_user_acceptance import (
    aggregate_phase103_scores,
    audit_phase103_sessions,
    build_phase103_decision,
    build_phase103_sessions,
    compare_phase103_variants,
    score_phase103_session,
)


def test_phase103_builds_twenty_three_turn_simulated_sessions():
    payload = build_phase103_sessions()
    assert payload["session_count"] == 20
    assert payload["turns_per_session"] == 3
    assert payload["total_model_call_budget"] == 120
    assert all(len(row["user_turns"]) == 3 for row in payload["sessions"])
    assert all(row["user_goal"] and row["user_correction"] and row["continuation_request"] for row in payload["sessions"])
    assert all(row["not_for_training"] is True for row in payload["sessions"])
    assert all(row["simulated_usage"] is True and row["actual_user_feedback"] is False for row in payload["sessions"])


def test_phase103_sessions_are_internally_valid_and_isolated():
    payload = build_phase103_sessions()
    audit = audit_phase103_sessions(payload, [], [])
    assert audit["passed"] is True


def test_phase103_scores_a_fully_compliant_provenance_session():
    session = next(row for row in build_phase103_sessions()["sessions"] if row["category"] == "provenance")
    output = (
        "结论：这是模拟使用记录\n"
        "依据：simulated_usage=true，actual_user_feedback=false\n"
        "下一步：不能计入真实反馈。"
    )
    score = score_phase103_session(
        session=session,
        outputs=[output, output, output],
        structural_turns=[{"complete_content_before_termination": True}] * 3,
    )
    assert score["accepted"] is True
    assert score["provenance_boundary_preserved"] is True


def test_phase103_pairwise_comparison_counts_wins_ties_losses():
    base = [
        {"session_id": "a", "component_score": 4},
        {"session_id": "b", "component_score": 5},
        {"session_id": "c", "component_score": 6},
    ]
    adapter = [
        {"session_id": "a", "component_score": 5},
        {"session_id": "b", "component_score": 5},
        {"session_id": "c", "component_score": 4},
    ]
    comparison = compare_phase103_variants(base, adapter)
    assert comparison["adapter_wins"] == 1
    assert comparison["ties"] == 1
    assert comparison["adapter_losses"] == 1


def test_phase103_decision_requires_visible_acceptance_gain():
    base = {
        "acceptance_rate": 0.4,
        "task_completion_rate": 0.6,
        "factual_boundary_rate": 1.0,
        "privacy_preservation_rate": 1.0,
    }
    adapter = {
        "acceptance_rate": 0.55,
        "task_completion_rate": 0.7,
        "factual_boundary_rate": 1.0,
        "privacy_preservation_rate": 1.0,
    }
    decision = build_phase103_decision(
        base_metrics=base,
        adapter_metrics=adapter,
        paired={"adapter_wins": 5, "adapter_losses": 2},
    )
    assert decision["passed"] is True
    assert decision["recommendation"] == "promote_after_manual_review"
    adapter["acceptance_rate"] = 0.45
    assert build_phase103_decision(
        base_metrics=base,
        adapter_metrics=adapter,
        paired={"adapter_wins": 5, "adapter_losses": 2},
    )["passed"] is False


def test_phase103_aggregate_keeps_simulated_user_components_separate():
    scores = [
        {
            "category": "provenance",
            "accepted": True,
            "task_complete": True,
            "latest_correction_followed": True,
            "factual_boundary_preserved": True,
            "format_stable": True,
            "privacy_preserved": True,
            "native_turn_completion": True,
            "provenance_boundary_preserved": True,
            "false_refusal_avoided": True,
            "component_score": 8,
        }
    ]
    metrics = aggregate_phase103_scores(scores)
    assert metrics["acceptance_rate"] == 1.0
    assert metrics["provenance_boundary_rate"] == 1.0
