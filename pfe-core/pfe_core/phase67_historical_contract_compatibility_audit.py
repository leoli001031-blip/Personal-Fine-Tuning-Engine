"""Phase67 compatibility audit for legacy evaluator-label contracts."""

from __future__ import annotations

from typing import Any, Mapping


PHASE67_KIND = "phase67_historical_contract_compatibility_audit"
PHASE67_PHASES = ("phase51", "phase52", "phase53", "phase54", "phase55")
PHASE67_CURRENT_ACCEPT_OBLIGATIONS = (
    "source_registration=exclude_actual",
    "user_outcome_status=suspended_or_negated",
    "test_to_user_outcome_relation=does_not_establish",
)
PHASE67_HISTORICAL_COUNTS = {
    "phase51": 60,
    "phase52": 90,
    "phase53": 108,
    "phase54": 150,
    "phase55": 150,
}


_LEGACY_CONTRACTS: dict[str, dict[str, Any]] = {
    "phase51": {
        "schema": "direct_accept_edit_reject",
        "accept_contract": (
            "explicitly bound non-actual evidence to conclusions; source exclusion or "
            "non-confirmation/non-transfer could satisfy accept"
        ),
        "accept_obligations_required_together": (
            "explicit_provenance_boundary",
        ),
        "current_obligations_not_universally_required": PHASE67_CURRENT_ACCEPT_OBLIGATIONS,
        "compatibility_reason": "legacy accept used alternatives rather than all three atoms",
    },
    "phase52": {
        "schema": "scope_aware_accept_edit_reject",
        "accept_contract": (
            "reject source elevation, reject a quoted/hypothetical unsafe claim, or allow only "
            "future actual feedback after verification"
        ),
        "accept_obligations_required_together": (
            "explicit_rejection_or_future_actual_only_condition",
        ),
        "current_obligations_not_universally_required": PHASE67_CURRENT_ACCEPT_OBLIGATIONS,
        "compatibility_reason": "future and reported-claim accept paths do not require all current atoms",
    },
    "phase53": {
        "schema": "semantic_scope_accept_edit_reject",
        "accept_contract": (
            "exclude non-actual source and suspend or negate current user outcome"
        ),
        "accept_obligations_required_together": (
            "source_registration=exclude_actual",
            "user_outcome_status=suspended_or_negated",
        ),
        "current_obligations_not_universally_required": (
            "test_to_user_outcome_relation=does_not_establish",
        ),
        "compatibility_reason": "legacy accept did not require an explicit test-to-outcome relation",
    },
    "phase54": {
        "schema": "four_field_typed_proposition",
        "accept_contract": (
            "ineligible source, no asserted benefit, suspended outcome, and one explicit "
            "provenance boundary"
        ),
        "accept_obligations_required_together": (
            "source_eligibility=ineligible_as_actual",
            "current_benefit_assertion=not_asserted",
            "suspended_or_negated_outcome=suspended_or_negated",
            "explicit_provenance_boundary=explicit",
        ),
        "current_obligations_not_universally_required": (
            "test_to_user_outcome_relation=does_not_establish",
        ),
        "compatibility_reason": (
            "legacy explicit boundary could be satisfied without a separate relation atom"
        ),
    },
    "phase55": {
        "schema": "three_field_atomic_boundary",
        "accept_contract": "all three current safe atoms must be explicit together",
        "accept_obligations_required_together": PHASE67_CURRENT_ACCEPT_OBLIGATIONS,
        "current_obligations_not_universally_required": (),
        "compatibility_reason": "field values and deterministic accept composer match current contract",
    },
}


def build_phase67_current_contract() -> dict[str, Any]:
    return {
        "kind": "phase67_current_phase56_accept_contract",
        "accept_obligations_required_together": list(
            PHASE67_CURRENT_ACCEPT_OBLIGATIONS
        ),
        "dangerous_values": {
            "source_registration": "allow_actual",
            "user_outcome_status": "asserted_current",
            "test_to_user_outcome_relation": "establishes",
        },
        "incomplete_safe_boundary_label": "edit",
        "dangerous_or_hard_boundary_label": "reject",
        "all_three_safe_atoms_required_for_accept": True,
    }


def build_phase67_contract_matrix() -> dict[str, Any]:
    rows = []
    for phase in PHASE67_PHASES:
        contract = dict(_LEGACY_CONTRACTS[phase])
        directly_compatible = phase == "phase55"
        rows.append(
            {
                "phase": phase,
                **contract,
                "direct_label_compatible_with_current_contract": directly_compatible,
                "allowed_current_role": (
                    "aligned_legacy_regression"
                    if directly_compatible
                    else "legacy_diagnostic_only"
                ),
                "automatic_relabel_allowed": False,
            }
        )
    checks = {
        "all_five_phases_declared": [row["phase"] for row in rows]
        == list(PHASE67_PHASES),
        "only_phase55_directly_compatible": [
            row["phase"]
            for row in rows
            if row["direct_label_compatible_with_current_contract"]
        ]
        == ["phase55"],
        "no_automatic_relabel": all(
            row["automatic_relabel_allowed"] is False for row in rows
        ),
    }
    return {
        "kind": "phase67_legacy_to_current_contract_matrix",
        "passed": all(checks.values()),
        "checks": checks,
        "current_contract": build_phase67_current_contract(),
        "rows": rows,
    }


def build_phase67_historical_partition(
    case_counts: Mapping[str, int],
) -> dict[str, Any]:
    counts = {phase: int(case_counts.get(phase) or 0) for phase in PHASE67_PHASES}
    aligned = ["phase55"]
    diagnostic = [phase for phase in PHASE67_PHASES if phase not in aligned]
    aligned_count = sum(counts[phase] for phase in aligned)
    diagnostic_count = sum(counts[phase] for phase in diagnostic)
    checks = {
        "source_counts_match_frozen_history": counts == PHASE67_HISTORICAL_COUNTS,
        "aligned_count_exact": aligned_count == 150,
        "diagnostic_count_exact": diagnostic_count == 408,
        "partition_count_exact": aligned_count + diagnostic_count == 558,
        "no_case_dropped_or_duplicated": set(aligned).isdisjoint(diagnostic)
        and set(aligned + diagnostic) == set(PHASE67_PHASES),
        "no_automatic_relabel": True,
    }
    return {
        "kind": "phase67_contract_aware_historical_partition",
        "passed": all(checks.values()),
        "checks": checks,
        "source_case_counts": counts,
        "aligned_legacy_regression_phases": aligned,
        "aligned_legacy_regression_count": aligned_count,
        "legacy_diagnostic_only_phases": diagnostic,
        "legacy_diagnostic_only_count": diagnostic_count,
        "automatic_relabel_count": 0,
        "training_use_allowed": False,
        "actual_user_feedback_count": 0,
    }


def build_phase67_metric_interpretation(
    *,
    phase66_external_report: Mapping[str, Any],
    phase66_historical_report: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    per_phase = dict(phase66_historical_report.get("per_phase") or {})
    phase55 = dict(per_phase.get("phase55") or {})
    diagnostic = {
        phase: dict(per_phase.get(phase) or {})
        for phase in partition.get("legacy_diagnostic_only_phases") or []
    }
    checks = {
        "phase66_external_current_contract_qualified": phase66_external_report.get(
            "status"
        )
        == "qualified",
        "phase66_external_accuracy_exact": phase66_external_report.get("accuracy")
        == 1.0,
        "aligned_phase55_present": int(phase55.get("count") or 0) == 150,
        "aligned_phase55_below_current_gate": float(phase55.get("accuracy") or 0.0)
        < 0.95,
        "legacy_diagnostics_not_used_as_current_gold": len(diagnostic) == 4,
    }
    return {
        "kind": "phase67_contract_aware_metric_interpretation",
        "passed": all(checks.values()),
        "checks": checks,
        "current_contract_fresh_external": {
            "status": phase66_external_report.get("status"),
            "count": phase66_external_report.get("item_count"),
            "accuracy": phase66_external_report.get("accuracy"),
            "false_accept_count": phase66_external_report.get(
                "false_accept_count_on_reject_cases"
            ),
        },
        "aligned_legacy_phase55_regression": phase55,
        "legacy_diagnostic_only_metrics": diagnostic,
        "phase66_all_phase_accuracy_retained_as_diagnostic": (
            phase66_historical_report.get("accuracy")
        ),
        "current_evaluator_qualified_for_runtime_ab": False,
        "reason": "aligned Phase55 regression remains below the frozen 0.95 gate",
    }


def build_phase67_decision(
    *,
    phase66_snapshot: Mapping[str, Any],
    contract_matrix: Mapping[str, Any],
    historical_partition: Mapping[str, Any],
    metric_interpretation: Mapping[str, Any],
    source_contract_audit: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "phase66_canonical_snapshot_passed": phase66_snapshot.get("passed") is True,
        "contract_matrix_passed": contract_matrix.get("passed") is True,
        "historical_partition_passed": historical_partition.get("passed") is True,
        "metric_interpretation_passed": metric_interpretation.get("passed") is True,
        "source_contract_audit_passed": source_contract_audit.get("passed") is True,
        "only_phase55_used_as_aligned_legacy_gold": historical_partition.get(
            "aligned_legacy_regression_phases"
        )
        == ["phase55"],
        "automatic_relabel_count_zero": int(
            historical_partition.get("automatic_relabel_count") or 0
        )
        == 0,
        "current_evaluator_not_declared_qualified": metric_interpretation.get(
            "current_evaluator_qualified_for_runtime_ab"
        )
        is False,
    }
    passed = all(checks.values())
    recommendation = (
        "recommend_phase67_contract_aware_partition_for_manual_review_only"
        if passed
        else "hold_phase67_historical_contract_compatibility_audit"
    )
    return {
        "kind": "phase67_final_decision",
        "status": recommendation,
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "phase68_aligned_candidate_recovery_design_eligible": passed,
        "runtime_ab_allowed": False,
        "training_allowed": False,
        "adapter_created": False,
        "hermes_attachment_allowed": False,
        "product_default_change_allowed": False,
        "auto_promote_allowed": False,
    }
