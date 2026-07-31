from __future__ import annotations

from pathlib import Path
import sys


CORE_ROOT = Path(__file__).resolve().parents[1] / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase104_autonomous_loop_decision import (
    PHASE104_ALLOWED_RECOMMENDATIONS,
    build_phase104_final_decision,
)


def _phase(*, passed: bool, status: str, checks: dict | None = None):
    return {
        "passed": passed,
        "status": status,
        "checks": checks or {},
        "product_gate_qualified": False,
    }


def test_phase104_keeps_runtime_primary_when_training_has_no_benefit():
    decision = build_phase104_final_decision(
        phase100=_phase(passed=True, status="phase100_generation_boundary_qualified"),
        phase101=_phase(passed=False, status="archive_phase101", checks={"real_training_completed": True}),
        phase102=_phase(passed=False, status="archive_phase102", checks={"real_dpo_training_completed": True}),
        phase103=_phase(passed=False, status="phase103_no_detectable_adapter_user_benefit"),
        cumulative_model_calls=240,
    )
    assert decision["recommendation"] == "runtime_contract_remains_primary"
    assert decision["training_execution"]["sft_real_training_completed"] is True
    assert decision["training_execution"]["dpo_real_training_completed"] is True
    assert decision["metric_improvement"]["adapter_metric_improvement_proved"] is False
    assert decision["product_benefit"]["simulated_user_benefit_proved"] is False
    assert decision["product_gate_qualified"] is False
    assert decision["automatic_promotion_allowed"] is False


def test_phase104_never_emits_an_unapproved_recommendation():
    decision = build_phase104_final_decision(
        phase100=_phase(passed=False, status="archive_phase100"),
        phase101=_phase(passed=False, status="archive_phase101"),
        phase102=_phase(passed=False, status="archive_phase102"),
        phase103=_phase(passed=False, status="archive_phase103"),
        cumulative_model_calls=0,
    )
    assert decision["recommendation"] == "archive"
    assert decision["recommendation"] in PHASE104_ALLOWED_RECOMMENDATIONS


def test_phase104_budget_check_is_strict():
    decision = build_phase104_final_decision(
        phase100=_phase(passed=True, status="phase100_generation_boundary_qualified"),
        phase101=_phase(passed=False, status="archive_phase101"),
        phase102=_phase(passed=False, status="archive_phase102"),
        phase103=_phase(passed=False, status="phase103_no_benefit"),
        cumulative_model_calls=271,
    )
    assert decision["checks"]["model_call_budget_respected"] is False
