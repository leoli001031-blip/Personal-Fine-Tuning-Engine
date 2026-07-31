from __future__ import annotations

from pathlib import Path
import sys


CORE_ROOT = Path(__file__).resolve().parents[1] / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase103_simulated_user_acceptance import PHASE103_CATEGORIES
from pfe_core.phase105_qwen3_curriculum_alignment import build_phase105_curriculum
from pfe_core.phase106_stratified_curriculum_repair import (
    audit_phase106_holdout,
    build_phase106_decision,
    build_phase106_holdout,
    summarize_phase106_exposure,
)
from pfe_core.trainer.executors import _build_seeded_stratified_training_order


def test_phase106_holdout_is_fresh_balanced_and_not_training_data():
    holdout = build_phase106_holdout()
    assert holdout["session_count"] == 10
    assert holdout["total_model_call_budget"] == 60
    assert all(row["not_for_training"] is True for row in holdout["sessions"])
    audit = audit_phase106_holdout(build_phase105_curriculum(), holdout, [])
    assert audit["passed"] is True


def test_phase106_thirty_step_order_exposes_each_category_six_times():
    rows = build_phase105_curriculum()
    order = _build_seeded_stratified_training_order(rows, seed=106, cycle=0)[:30]
    exposure = summarize_phase106_exposure(rows, order)
    assert exposure["passed"] is True
    assert exposure["category_exposure_counts"] == {
        category: 6 for category in sorted(PHASE103_CATEGORIES)
    }


def _metrics(*, acceptance: float, provenance: float, native: float) -> dict[str, float]:
    return {
        "acceptance_rate": acceptance,
        "task_completion_rate": acceptance,
        "correction_following_rate": acceptance,
        "format_stability_rate": native,
        "native_turn_completion_rate": native,
        "factual_boundary_rate": 1.0,
        "privacy_preservation_rate": 1.0,
        "provenance_boundary_rate": provenance,
    }


def test_phase106_gate_requires_balanced_exposure_and_no_native_regression():
    base = _metrics(acceptance=0.4, provenance=0.0, native=0.7)
    candidate = _metrics(acceptance=0.6, provenance=0.5, native=0.8)
    decision = build_phase106_decision(
        base_metrics=base,
        candidate_metrics=candidate,
        training_completed=True,
        exposure_balanced=True,
    )
    assert decision["passed"] is True
    candidate["native_turn_completion_rate"] = 0.6
    assert build_phase106_decision(
        base_metrics=base,
        candidate_metrics=candidate,
        training_completed=True,
        exposure_balanced=True,
    )["passed"] is False


def test_phase106_gate_treats_exact_tenth_gain_as_threshold_pass():
    base = _metrics(acceptance=0.6, provenance=0.0, native=0.7)
    candidate = _metrics(acceptance=0.7, provenance=0.5, native=0.8)
    decision = build_phase106_decision(
        base_metrics=base,
        candidate_metrics=candidate,
        training_completed=True,
        exposure_balanced=True,
    )
    assert decision["checks"]["acceptance_gain_at_least_0_10"] is True
