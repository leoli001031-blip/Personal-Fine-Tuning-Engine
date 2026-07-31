from __future__ import annotations

from pathlib import Path
import sys


CORE_ROOT = Path(__file__).resolve().parents[1] / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase101_failure_targeted_sft import build_phase101_holdout, build_phase101_training_candidates
from pfe_core.phase102_failure_targeted_dpo import (
    audit_phase102_pairs,
    build_phase102_dpo_decision,
    select_phase102_dpo_pairs,
)


def test_phase102_selects_balanced_failure_pairs():
    pairs = select_phase102_dpo_pairs(build_phase101_training_candidates())
    assert len(pairs) == 24
    counts = {
        category: sum(row["preference_category"] == category for row in pairs)
        for category in ("exact_three_line", "false_block", "provenance")
    }
    assert counts == {"exact_three_line": 8, "false_block": 8, "provenance": 8}
    assert all(row["chosen"] != row["rejected"] for row in pairs)
    assert all(row["simulated_usage"] is True for row in pairs)
    assert all(row["actual_user_feedback"] is False for row in pairs)


def test_phase102_pairs_remain_isolated_from_phase101_holdout():
    pairs = select_phase102_dpo_pairs(build_phase101_training_candidates())
    audit = audit_phase102_pairs(pairs, build_phase101_holdout())
    assert audit["passed"] is True


def _metrics(value: float, dependency: float) -> dict[str, float]:
    return {
        "exact_three_line_rate": value,
        "false_block_avoidance_rate": 1.0,
        "provenance_correct_rate": value,
        "ordinary_control_rate": 1.0,
        "complete_content_before_termination_rate": value,
        "native_termination_rate": value,
        "unsupported_assertion_rate": 0.0,
        "think_leak_rate": 0.0,
        "privacy_echo_rate": 0.0,
        "repeated_output_rate": 0.0,
        "extra_text_after_first_answer_rate": 0.0,
        "forbidden_generation_rate": 0.0,
        "runtime_control_dependency_rate": dependency,
    }


def test_phase102_gate_requires_runtime_parity_and_real_training():
    decision = build_phase102_dpo_decision(
        base_metrics=_metrics(0.5, 0.0),
        sft_metrics=_metrics(0.25, 0.0),
        runtime_metrics=_metrics(1.0, 0.25),
        candidate_metrics=_metrics(1.0, 0.0),
        training_completed=True,
    )
    assert decision["passed"] is True
    assert build_phase102_dpo_decision(
        base_metrics=_metrics(0.5, 0.0),
        sft_metrics=_metrics(0.25, 0.0),
        runtime_metrics=_metrics(1.0, 0.25),
        candidate_metrics=_metrics(0.99, 0.0),
        training_completed=True,
    )["passed"] is False
