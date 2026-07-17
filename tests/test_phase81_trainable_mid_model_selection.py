from __future__ import annotations

from pfe_core.phase77_private_value_guarded_runtime import build_phase77_holdout
from pfe_core.phase78_persona_internalization_training import build_phase78_holdout
from pfe_core.phase79_cpu_feasible_persona_probe import build_phase79_holdout
from pfe_core.phase80_small_model_failure_taxonomy import build_phase80_holdout
from pfe_core.phase81_trainable_mid_model_selection import (
    PHASE81_SANITY_VARIANTS,
    PHASE81_VARIANTS,
    audit_phase81_isolation,
    build_phase81_final_decision,
    build_phase81_holdout,
    build_phase81_model_selection,
    build_phase81_sanity_decision,
    build_phase81_sanity_holdout,
)


def _metrics(
    target: float,
    *,
    sessions: int = 21,
    ordinary: float = 0.9,
    truncated: float = 0.0,
) -> dict[str, object]:
    categories = {
        name: {"composite_personalization_score": target, "hard_gate_pass_rate": 1.0}
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
        "session_count": sessions,
        "category_metrics": categories,
        "hard_gate_pass_rate": 1.0,
        "truncated_session_rate": truncated,
        "privacy_canary_echo_rate": 0.0,
        "think_leak_rate": 0.0,
    }


def _attempt(steps: int) -> dict[str, object]:
    return {
        "status": "completed",
        "real_training": True,
        "requested_steps": steps,
        "duration_seconds": 1800.0,
        "adapter_validation": {"valid": True},
    }


def _selection() -> dict[str, object]:
    return {"status": "selected", "selected_model": "Qwen2.5-1.5B-Instruct"}


def test_phase81_holdout_is_fresh_simulated_and_isolated() -> None:
    holdout = build_phase81_holdout()
    previous = (
        build_phase77_holdout()["sessions"]
        + build_phase78_holdout()["sessions"]
        + build_phase79_holdout()["sessions"]
        + build_phase80_holdout()["sessions"]
    )
    audit = audit_phase81_isolation(holdout["sessions"], previous)
    sanity = build_phase81_sanity_holdout(holdout)

    assert holdout["session_count"] == 21
    assert holdout["persona_target_count"] == 18
    assert holdout["ordinary_control_count"] == 3
    assert sanity["session_count"] == 7
    assert audit["passed"] is True
    assert audit["training_text_overlap"] == []
    assert audit["previous_holdout_text_overlap"] == []
    assert all(row["simulated_usage"] is True for row in holdout["sessions"])
    assert all(row["actual_user_feedback"] is False for row in holdout["sessions"])


def test_phase81_model_selection_chooses_smallest_eligible_mid_model() -> None:
    candidates = [
        {
            "model_id": "Qwen2.5-1.5B-Instruct",
            "local_path": "models/Qwen2.5-1.5B-Instruct",
            "parameter_billions": 1.54,
            "download_bytes": 3_100_000_000,
            "estimated_training_memory_bytes": 12_000_000_000,
            "official_qwen": True,
            "architecture_compatible": True,
            "download_complete": True,
        },
        {
            "model_id": "Qwen3-1.7B",
            "local_path": "models/Qwen3-1.7B",
            "parameter_billions": 1.7,
            "download_bytes": 4_100_000_000,
            "estimated_training_memory_bytes": 14_000_000_000,
            "official_qwen": True,
            "architecture_compatible": True,
            "download_complete": True,
        },
    ]
    result = build_phase81_model_selection(
        candidates,
        available_disk_bytes=900_000_000_000,
        system_memory_bytes=128_000_000_000,
        mps_available=False,
    )

    assert result["status"] == "selected"
    assert result["selected_model"] == "Qwen2.5-1.5B-Instruct"
    assert result["execution_device"] == "cpu"
    assert result["automatic_training_allowed"] is False


def test_phase81_sanity_allows_12_step_only_without_catastrophic_regression() -> None:
    metrics = {
        "base_mid_4step_sanity": _metrics(0.58, sessions=7),
        "adapter_mid_4step_sanity": _metrics(0.53, sessions=7, truncated=0.1),
    }
    decision = build_phase81_sanity_decision(metrics=metrics, training_attempt=_attempt(4))

    assert set(metrics) == set(PHASE81_SANITY_VARIANTS)
    assert decision["passed"] is True
    assert decision["status"] == "ready_for_12_step_probe"
    assert decision["actual_product_benefit_claim_allowed"] is False


def test_phase81_sanity_blocks_slow_or_unstable_probe() -> None:
    attempt = _attempt(4)
    attempt["duration_seconds"] = 4000.0
    metrics = {
        "base_mid_4step_sanity": _metrics(0.58, sessions=7),
        "adapter_mid_4step_sanity": _metrics(0.35, sessions=7, truncated=0.3),
    }
    decision = build_phase81_sanity_decision(metrics=metrics, training_attempt=attempt)

    assert decision["passed"] is False
    assert decision["status"] == "archive_4_step_sanity_failed"
    assert "training_duration_within_3600_seconds" in decision["failed_checks"]
    assert "adapter_target_regression_at_most_0_10" in decision["failed_checks"]


def test_phase81_final_decision_qualifies_only_simulated_adapter_benefit() -> None:
    metrics = {
        "base_mid_length_control": _metrics(0.58),
        "runtime_mid_length_control": _metrics(0.65),
        "adapter_mid_12step_length_control": _metrics(0.65),
    }
    decision = build_phase81_final_decision(
        metrics=metrics,
        training_attempt=_attempt(12),
        model_selection=_selection(),
        isolation_audit={"passed": True},
        public_private_audit={"passed": True},
    )

    assert set(metrics) == set(PHASE81_VARIANTS)
    assert decision["status"] == "qualified_simulated_mid_model_adapter"
    assert decision["recommendation"] == "phase82_full_coverage_mid_model_probe"
    assert decision["simulated_lab_adapter_benefit"] is True
    assert decision["actual_product_benefit_claim_allowed"] is False
    assert decision["auto_promotion_allowed"] is False


def test_phase81_final_decision_archives_adapter_when_only_runtime_helps() -> None:
    metrics = {
        "base_mid_length_control": _metrics(0.58),
        "runtime_mid_length_control": _metrics(0.64),
        "adapter_mid_12step_length_control": _metrics(0.60),
    }
    decision = build_phase81_final_decision(
        metrics=metrics,
        training_attempt=_attempt(12),
        model_selection=_selection(),
        isolation_audit={"passed": True},
        public_private_audit={"passed": True},
    )

    assert decision["status"] == "archive_adapter_no_incremental_benefit"
    assert decision["recommendation"] == "phase82_mid_model_runtime_contract_path"
    assert decision["simulated_lab_adapter_benefit"] is False


def test_phase81_final_decision_treats_length_failure_as_archive_not_missing_evidence() -> None:
    metrics = {
        "base_mid_length_control": _metrics(0.58, truncated=0.2),
        "runtime_mid_length_control": _metrics(0.64),
        "adapter_mid_12step_length_control": _metrics(0.65, truncated=0.2),
    }
    decision = build_phase81_final_decision(
        metrics=metrics,
        training_attempt=_attempt(12),
        model_selection=_selection(),
        isolation_audit={"passed": True},
        public_private_audit={"passed": True},
    )

    assert decision["status"] == "archive_adapter_no_incremental_benefit"
    assert decision["benefit_checks"]["adapter_truncation_at_most_0_10"] is False
    assert decision["failed_checks"] == []


def test_phase81_final_decision_blocks_incomplete_training_evidence() -> None:
    metrics = {name: _metrics(0.6) for name in PHASE81_VARIANTS}
    attempt = _attempt(12)
    attempt["adapter_validation"] = {"valid": False}
    decision = build_phase81_final_decision(
        metrics=metrics,
        training_attempt=attempt,
        model_selection=_selection(),
        isolation_audit={"passed": True},
        public_private_audit={"passed": True},
    )

    assert decision["status"] == "archive_incomplete_mid_model_probe"
    assert "adapter_artifact_valid" in decision["failed_checks"]
    assert decision["promotion_allowed"] is False
