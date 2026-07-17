from __future__ import annotations

from pathlib import Path
import sys


CORE_ROOT = Path(__file__).resolve().parents[1] / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase105_qwen3_curriculum_alignment import (
    audit_phase105_curriculum,
    build_phase105_curriculum,
    build_phase105_decision,
    build_phase105_holdout,
)


def test_phase105_curriculum_is_balanced_diverse_and_multiturn():
    rows = build_phase105_curriculum()
    assert len(rows) == 240
    categories = {row["category"] for row in rows}
    assert categories == {
        "exact_three_line",
        "false_block",
        "provenance",
        "correction_following",
        "ordinary_control",
    }
    assert all(sum(row["category"] == category for row in rows) == 48 for category in categories)
    assert len({row["chosen"] for row in rows}) == 240
    assert all([message["role"] for message in row["messages"]] == ["system", "user", "assistant", "user"] for row in rows)
    assert all(row["simulated_usage"] is True and row["actual_user_feedback"] is False for row in rows)


def test_phase105_provenance_targets_preserve_exact_identifiers_with_diversity():
    rows = [row for row in build_phase105_curriculum() if row["category"] == "provenance"]
    assert len(rows) == 48
    assert len({row["chosen"] for row in rows}) == 48
    assert all("simulated_usage=true" in row["chosen"] for row in rows)
    assert all("actual_user_feedback=false" in row["chosen"] for row in rows)
    assert all("不能计入真实反馈" in row["chosen"] for row in rows)


def test_phase105_holdout_is_fresh_balanced_and_not_training_data():
    holdout = build_phase105_holdout()
    assert holdout["session_count"] == 10
    assert holdout["total_model_call_budget"] == 60
    assert all(row["not_for_training"] is True for row in holdout["sessions"])
    audit = audit_phase105_curriculum(build_phase105_curriculum(), holdout, [])
    assert audit["passed"] is True


def _metrics(*, acceptance: float, provenance: float, format_rate: float) -> dict[str, float]:
    return {
        "acceptance_rate": acceptance,
        "task_completion_rate": acceptance,
        "correction_following_rate": acceptance,
        "format_stability_rate": format_rate,
        "factual_boundary_rate": 1.0,
        "privacy_preservation_rate": 1.0,
        "provenance_boundary_rate": provenance,
    }


def test_phase105_gate_requires_visible_gain_and_provenance_improvement():
    base = _metrics(acceptance=0.4, provenance=0.0, format_rate=0.5)
    candidate = _metrics(acceptance=0.6, provenance=0.5, format_rate=0.7)
    decision = build_phase105_decision(base_metrics=base, candidate_metrics=candidate, training_completed=True)
    assert decision["passed"] is True
    candidate["provenance_boundary_rate"] = 0.0
    assert build_phase105_decision(base_metrics=base, candidate_metrics=candidate, training_completed=True)["passed"] is False
