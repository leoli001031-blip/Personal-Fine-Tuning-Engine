from __future__ import annotations

from pathlib import Path
import sys


CORE_ROOT = Path(__file__).resolve().parents[1] / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase101_failure_targeted_sft import (
    audit_phase101_training_and_holdout,
    build_phase101_holdout,
    build_phase101_sft_decision,
    build_phase101_training_candidates,
)


def test_phase101_candidates_are_failure_targeted_and_simulated_only():
    rows = build_phase101_training_candidates()
    assert len(rows) == 32
    assert {row["category"] for row in rows} == {"exact_three_line", "false_block", "provenance"}
    assert all(row["eligible_for_training"] is True for row in rows)
    assert all(row["simulated_usage"] is True for row in rows)
    assert all(row["actual_user_feedback"] is False for row in rows)
    assert all(row["chosen"] != row["rejected"] for row in rows)


def test_phase101_holdout_is_fresh_and_not_for_training():
    holdout = build_phase101_holdout()
    assert holdout["session_count"] == 8
    assert holdout["model_calls_per_variant"] == 24
    assert all(row["not_for_training"] is True for row in holdout["sessions"])
    audit = audit_phase101_training_and_holdout(build_phase101_training_candidates(), holdout, [])
    assert audit["passed"] is True


def test_phase101_provenance_targets_keep_exact_boundary():
    rows = [row for row in build_phase101_training_candidates() if row["category"] == "provenance"]
    assert len(rows) == 10
    assert all("simulated_usage=true" in row["chosen"] for row in rows)
    assert all("actual_user_feedback=false" in row["chosen"] for row in rows)
    assert all(row["chosen"].endswith("不能计入真实反馈。") for row in rows)


def _perfect_metrics(*, dependency: float) -> dict[str, float]:
    return {
        "exact_three_line_rate": 1.0,
        "false_block_avoidance_rate": 1.0,
        "provenance_correct_rate": 1.0,
        "ordinary_control_rate": 1.0,
        "complete_content_before_termination_rate": 1.0,
        "native_termination_rate": 1.0,
        "unsupported_assertion_rate": 0.0,
        "think_leak_rate": 0.0,
        "privacy_echo_rate": 0.0,
        "repeated_output_rate": 0.0,
        "extra_text_after_first_answer_rate": 0.0,
        "forbidden_generation_rate": 0.0,
        "runtime_control_dependency_rate": dependency,
    }


def test_phase101_gate_requires_runtime_parity_base_gain_and_lower_dependency():
    runtime = _perfect_metrics(dependency=0.25)
    candidate = _perfect_metrics(dependency=0.0)
    base = _perfect_metrics(dependency=0.0)
    base["provenance_correct_rate"] = 0.0
    decision = build_phase101_sft_decision(
        base_metrics=base,
        runtime_metrics=runtime,
        candidate_metrics=candidate,
        training_completed=True,
    )
    assert decision["passed"] is True
    candidate["provenance_correct_rate"] = 0.5
    assert build_phase101_sft_decision(
        base_metrics=base,
        runtime_metrics=runtime,
        candidate_metrics=candidate,
        training_completed=True,
    )["passed"] is False
