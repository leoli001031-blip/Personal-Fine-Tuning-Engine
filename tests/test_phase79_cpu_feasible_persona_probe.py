from __future__ import annotations

import copy

from pfe_core.phase77_private_value_guarded_runtime import build_phase77_holdout
from pfe_core.phase78_persona_internalization_training import (
    PHASE78_PERSONA_CATEGORIES,
    build_phase78_holdout,
)
from pfe_core.phase79_cpu_feasible_persona_probe import (
    PHASE79_MODEL_NAME,
    audit_phase79_isolation,
    build_phase79_decision,
    build_phase79_holdout,
    build_phase79_sanity_blocked_decision,
)


def _metrics(target: float, ordinary: float = 0.9) -> dict[str, object]:
    categories = {
        name: {
            "composite_personalization_score": target,
            "hard_gate_pass_rate": 1.0,
        }
        for name in PHASE78_PERSONA_CATEGORIES
    }
    categories["ordinary_direct"] = {
        "composite_personalization_score": ordinary,
        "hard_gate_pass_rate": 1.0,
    }
    return {
        "actual_model_calls": True,
        "session_count": 48,
        "category_metrics": categories,
        "privacy_canary_echo_rate": 0.0,
        "unsupported_claim_rate": 0.0,
    }


def _judge_summary(base_win: float = 0.7, runtime_win: float = 0.4) -> dict[str, object]:
    return {
        "status": "completed",
        "actual_model_calls": True,
        "completed_pair_count": 96,
        "failure_count": 0,
        "invalid_result_count": 0,
        "comparisons": {
            "adapter_vs_base": {
                "slices": {
                    "persona_target": {
                        "candidate_win_rate": base_win,
                        "tie_rate": 0.1,
                    }
                }
            },
            "adapter_vs_runtime": {
                "slices": {
                    "persona_target": {
                        "candidate_win_rate": runtime_win,
                        "tie_rate": 0.3,
                    }
                }
            },
        },
    }


def _training_attempt() -> dict[str, object]:
    return {
        "status": "completed",
        "real_training": True,
        "requested_steps": 120,
        "selected_model": PHASE79_MODEL_NAME,
        "historical_adapter_reused": False,
        "adapter_validation": {"valid": True},
        "execution": {"parameters_updated": True},
        "exposure": {"full_coverage": True},
    }


def _decision(metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    judge = _judge_summary()
    return build_phase79_decision(
        metrics=metrics,
        training_attempt=_training_attempt(),
        quality_audit={"passed": True},
        isolation_audit={"passed": True},
        completion_boundary={"passed": True},
        public_private_audit={"passed": True},
        deterministic=judge,
        independent={"gemma4:31b": judge, "qwen3.6": judge},
        phase78_archive={"status": "archive_execution_environment_blocked"},
        phase32_audit={"passed": True},
    )


def test_phase79_holdout_is_fresh_simulated_and_never_training_data() -> None:
    holdout = build_phase79_holdout()
    audit = audit_phase79_isolation(
        holdout["sessions"],
        build_phase78_holdout()["sessions"],
        build_phase77_holdout()["sessions"],
    )

    assert holdout["session_count"] == 48
    assert holdout["persona_target_count"] == 36
    assert holdout["ordinary_control_count"] == 12
    assert audit["passed"] is True
    assert audit["training_text_overlap"] == []
    assert audit["previous_holdout_text_overlap"] == []
    assert all(row["not_for_training"] is True for row in holdout["sessions"])
    assert all(row["simulated_usage"] is True for row in holdout["sessions"])
    assert all(row["actual_user_feedback"] is False for row in holdout["sessions"])


def test_phase79_private_canaries_exist_only_in_privacy_holdout_rows() -> None:
    sessions = build_phase79_holdout()["sessions"]
    privacy = [row for row in sessions if row["category"] == "privacy_non_echo"]
    ordinary = [row for row in sessions if row["category"] != "privacy_non_echo"]

    assert len(privacy) == 6
    assert all("SYNTHETIC_PHASE79_PRIVATE_" in str(row) for row in privacy)
    assert all("SYNTHETIC_PHASE79_PRIVATE_" not in str(row) for row in ordinary)


def test_phase79_decision_requires_benefit_beyond_training_completion() -> None:
    metrics = {
        "base_minimal_guarded": _metrics(0.62),
        "adapter_minimal_guarded": _metrics(0.74),
        "runtime_reference": _metrics(0.76),
    }
    qualified = _decision(metrics)

    assert qualified["status"] == "qualified_simulated_cpu_persona_adapter"
    assert qualified["recommendation"] == "manual_review_then_limited_actual_usage_design"
    assert qualified["actual_product_benefit_claim_allowed"] is False
    assert qualified["auto_promotion_allowed"] is False
    assert qualified["checks"]["phase78_environment_archive_acknowledged"] is True
    assert qualified["checks"]["phase32_overclaim_not_inherited"] is True

    no_benefit = copy.deepcopy(metrics)
    no_benefit["adapter_minimal_guarded"] = _metrics(0.63)
    archived = _decision(no_benefit)
    assert archived["status"] == "archive"
    assert "adapter_target_gain_at_least_0_08" in archived["failed_checks"]
    assert archived["simulated_lab_benefit_claim_allowed"] is False


def test_phase79_decision_rejects_historical_adapter_reuse_or_phase32_overclaim() -> None:
    metrics = {
        "base_minimal_guarded": _metrics(0.62),
        "adapter_minimal_guarded": _metrics(0.74),
        "runtime_reference": _metrics(0.76),
    }
    judge = _judge_summary()
    reused = _training_attempt()
    reused["historical_adapter_reused"] = True
    decision = build_phase79_decision(
        metrics=metrics,
        training_attempt=reused,
        quality_audit={"passed": True},
        isolation_audit={"passed": True},
        completion_boundary={"passed": True},
        public_private_audit={"passed": True},
        deterministic=judge,
        independent={"gemma4:31b": judge, "qwen3.6": judge},
        phase78_archive={"status": "archive_execution_environment_blocked"},
        phase32_audit={"passed": False},
    )

    assert decision["status"] == "archive"
    assert "historical_adapter_not_reused" in decision["failed_checks"]
    assert "phase32_overclaim_not_inherited" in decision["failed_checks"]


def test_phase79_sanity_failure_blocks_full_training_without_erasing_training_proof() -> None:
    attempt = _training_attempt()
    attempt["requested_steps"] = 12
    decision = build_phase79_sanity_blocked_decision(
        training_attempt=attempt,
        sanity_report={
            "passed": False,
            "checks": {
                "seven_sessions_completed": True,
                "all_real_model_calls": True,
                "no_truncation": False,
            },
        },
        sanity_diagnostic={"passed": True, "full_training_started": False},
    )

    assert decision["status"] == "archive_12_step_sanity_failed"
    assert decision["training_success"] is True
    assert decision["adapter_benefit"] == "not_evaluated_on_full_holdout"
    assert decision["full_training_blocked_by_frozen_sanity_gate"] is True
    assert decision["actual_product_benefit_claim_allowed"] is False
    assert decision["auto_promotion_allowed"] is False
