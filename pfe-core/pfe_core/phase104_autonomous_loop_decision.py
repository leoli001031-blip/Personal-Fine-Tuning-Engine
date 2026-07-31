from __future__ import annotations

from typing import Any, Mapping


PHASE104_ALLOWED_RECOMMENDATIONS = {
    "archive",
    "runtime_contract_remains_primary",
    "promote_after_manual_review",
}


def build_phase104_final_decision(
    *,
    phase100: Mapping[str, Any],
    phase101: Mapping[str, Any],
    phase102: Mapping[str, Any],
    phase103: Mapping[str, Any],
    cumulative_model_calls: int,
) -> dict[str, Any]:
    sft_training_completed = bool(
        dict(phase101.get("checks") or {}).get("real_training_completed")
    )
    dpo_training_completed = bool(
        dict(phase102.get("checks") or {}).get("real_dpo_training_completed")
    )
    adapter_metric_improvement = bool(phase101.get("passed") or phase102.get("passed"))
    simulated_user_benefit = phase103.get("passed") is True
    phase100_passed = phase100.get("passed") is True
    budget_ok = 0 <= int(cumulative_model_calls) <= 270
    if simulated_user_benefit and adapter_metric_improvement:
        recommendation = "promote_after_manual_review"
        status = "phase104_adapter_benefit_requires_manual_review"
    elif phase100_passed:
        recommendation = "runtime_contract_remains_primary"
        status = "phase104_runtime_contract_primary_adapters_archived"
    else:
        recommendation = "archive"
        status = "archive_phase104_no_qualified_runtime_or_adapter"
    checks = {
        "phase100_generation_boundary_qualified": phase100_passed,
        "sft_training_completed": sft_training_completed,
        "dpo_training_completed": dpo_training_completed,
        "adapter_metric_improvement_proved": adapter_metric_improvement,
        "simulated_user_benefit_proved": simulated_user_benefit,
        "model_call_budget_respected": budget_ok,
        "recommendation_allowed": recommendation in PHASE104_ALLOWED_RECOMMENDATIONS,
        "all_phase_product_gates_false": all(
            payload.get("product_gate_qualified") is False
            for payload in (phase100, phase101, phase102, phase103)
        ),
    }
    return {
        "kind": "phase104_autonomous_loop_final_decision",
        "status": status,
        "recommendation": recommendation,
        "training_execution": {
            "sft_real_training_completed": sft_training_completed,
            "dpo_real_training_completed": dpo_training_completed,
        },
        "metric_improvement": {
            "adapter_metric_improvement_proved": adapter_metric_improvement,
            "sft_status": phase101.get("status"),
            "dpo_status": phase102.get("status"),
        },
        "product_benefit": {
            "simulated_user_benefit_proved": simulated_user_benefit,
            "phase103_status": phase103.get("status"),
            "runtime_contract_remains_primary": recommendation == "runtime_contract_remains_primary",
        },
        "checks": checks,
        "cumulative_model_call_count": int(cumulative_model_calls),
        "long_run_total_call_budget": 270,
        "remaining_model_call_budget": 270 - int(cumulative_model_calls),
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "deployment_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
