from __future__ import annotations

from pfe_core.phase96_98_qwen3_4b_capacity_ladder import (
    build_phase96_capacity_decision,
    build_phase96_capacity_holdout,
)


def test_capacity_holdout_has_two_fresh_sessions_per_category() -> None:
    payload = build_phase96_capacity_holdout()
    sessions = payload["sessions"]

    assert len(sessions) == 8
    assert {row["category"] for row in sessions} == {
        "exact_three_line",
        "false_block",
        "provenance",
        "ordinary_control",
    }
    assert all(row["not_for_training"] is True for row in sessions)
    assert all(row["simulated_usage"] is True and row["actual_user_feedback"] is False for row in sessions)


def test_capacity_gate_requires_strict_core_gain_without_regression() -> None:
    smaller = {
        "session_count": 8,
        "exact_three_line_rate": 0.0,
        "false_block_avoidance_rate": 1.0,
        "provenance_correct_rate": 0.0,
        "ordinary_control_rate": 1.0,
    }
    larger = {**smaller, "exact_three_line_rate": 0.5}

    assert build_phase96_capacity_decision(smaller, larger)["passed"] is True
    assert build_phase96_capacity_decision(smaller, dict(smaller))["passed"] is False


def test_capacity_gate_never_promotes() -> None:
    smaller = {
        "session_count": 8,
        "exact_three_line_rate": 0.0,
        "false_block_avoidance_rate": 0.0,
        "provenance_correct_rate": 0.0,
        "ordinary_control_rate": 1.0,
    }
    larger = {**smaller, "provenance_correct_rate": 1.0}
    decision = build_phase96_capacity_decision(smaller, larger)

    assert decision["automatic_promotion_allowed"] is False
    assert decision["product_gate_qualified"] is False
