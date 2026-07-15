from __future__ import annotations

from pfe_core.phase77_private_value_guarded_runtime import build_phase77_holdout
from pfe_core.phase78_persona_internalization_training import build_phase78_holdout
from pfe_core.phase79_cpu_feasible_persona_probe import build_phase79_holdout
from pfe_core.phase80_small_model_failure_taxonomy import (
    PHASE80_VARIANTS,
    audit_phase80_isolation,
    build_phase80_decision,
    build_phase80_holdout,
)


def _metrics(target: float, *, ordinary: float = 0.9, truncated: float = 0.0) -> dict[str, object]:
    categories = {
        name: {
            "composite_personalization_score": target,
            "hard_gate_pass_rate": 1.0,
        }
        for name in (
            "evidence_truthfulness",
            "latest_action_switch",
            "provenance_labeling",
            "autonomous_execution",
            "concise_workstyle",
            "privacy_non_echo",
        )
    }
    categories["ordinary_direct"] = {
        "composite_personalization_score": ordinary,
        "hard_gate_pass_rate": 1.0,
    }
    return {
        "actual_model_calls": True,
        "session_count": 21,
        "category_metrics": categories,
        "truncated_session_rate": truncated,
        "privacy_canary_echo_rate": 0.0,
    }


def _attempt() -> dict[str, object]:
    return {
        "status": "completed",
        "real_training": True,
        "requested_steps": 12,
        "historical_adapter_reused": False,
        "adapter_validation": {"valid": True},
    }


def test_phase80_holdout_is_fresh_simulated_and_isolated() -> None:
    holdout = build_phase80_holdout()
    previous = (
        build_phase77_holdout()["sessions"]
        + build_phase78_holdout()["sessions"]
        + build_phase79_holdout()["sessions"]
    )
    audit = audit_phase80_isolation(holdout["sessions"], previous)

    assert holdout["session_count"] == 21
    assert holdout["persona_target_count"] == 18
    assert holdout["ordinary_control_count"] == 3
    assert audit["passed"] is True
    assert audit["training_text_overlap"] == []
    assert audit["previous_holdout_text_overlap"] == []
    assert all(row["simulated_usage"] is True for row in holdout["sessions"])
    assert all(row["actual_user_feedback"] is False for row in holdout["sessions"])


def test_phase80_decision_classifies_recoverable_learning_rate_instability() -> None:
    metrics = {
        "base_0_5b_minimal": _metrics(0.50),
        "runtime_0_5b": _metrics(0.57),
        "phase79_high_lr_adapter": _metrics(0.46, truncated=0.15),
        "phase80_low_lr_adapter": _metrics(0.60),
        "phase79_high_lr_stop_control": _metrics(0.50),
        "base_4b_minimal": _metrics(0.62),
        "runtime_4b": _metrics(0.72),
    }
    decision = build_phase80_decision(
        metrics=metrics,
        low_lr_training_attempt=_attempt(),
        isolation_audit={"passed": True},
        public_private_audit={"passed": True},
    )

    assert decision["status"] == "diagnosis_completed"
    assert decision["failure_classification"] == "optimization_instability_recoverable"
    assert decision["recommendation"] == "phase81_low_lr_full_coverage_probe"
    assert decision["actual_product_benefit_claim_allowed"] is False
    assert decision["auto_promotion_allowed"] is False


def test_phase80_decision_classifies_small_model_capacity_when_low_lr_does_not_help() -> None:
    metrics = {
        "base_0_5b_minimal": _metrics(0.50),
        "runtime_0_5b": _metrics(0.56),
        "phase79_high_lr_adapter": _metrics(0.46, truncated=0.15),
        "phase80_low_lr_adapter": _metrics(0.52),
        "phase79_high_lr_stop_control": _metrics(0.49),
        "base_4b_minimal": _metrics(0.64),
        "runtime_4b": _metrics(0.72, truncated=0.2),
    }
    decision = build_phase80_decision(
        metrics=metrics,
        low_lr_training_attempt=_attempt(),
        isolation_audit={"passed": True},
        public_private_audit={"passed": True},
    )

    assert decision["failure_classification"] == "small_model_capacity_dominant_with_length_cost"
    assert decision["recommendation"] == "phase81_trainable_mid_model_selection"
    assert decision["four_b_runtime_gap_vs_zero_point_five_b"] == 0.16


def test_phase80_decision_blocks_incomplete_or_reused_training_evidence() -> None:
    metrics = {name: _metrics(0.5) for name in PHASE80_VARIANTS}
    attempt = _attempt()
    attempt["historical_adapter_reused"] = True
    decision = build_phase80_decision(
        metrics=metrics,
        low_lr_training_attempt=attempt,
        isolation_audit={"passed": True},
        public_private_audit={"passed": True},
    )

    assert decision["status"] == "archive_incomplete_diagnosis"
    assert "phase79_adapter_not_reused_as_new_candidate" in decision["failed_checks"]
    assert decision["simulated_lab_benefit_claim_allowed"] is False
