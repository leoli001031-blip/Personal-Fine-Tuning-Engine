from __future__ import annotations

from pfe_core.phase59_proposition_addressed_grounding import build_phase59_proposition_candidates
from pfe_core.phase69_minimal_runtime_ab import (
    PHASE69_BOUNDARY_CATEGORIES,
    PHASE69_CANDIDATE_CONTRACT,
    PHASE69_ORDINARY_CATEGORIES,
    audit_phase69_ab_parity,
    build_phase69_decision,
    build_phase69_holdout,
    build_phase69_runtime_messages,
    score_phase69_ordinary_transcripts,
)


def test_phase69_holdout_is_balanced_simulated_and_not_training_data() -> None:
    holdout = build_phase69_holdout()

    assert holdout["session_count"] == 48
    assert holdout["boundary_session_count"] == 36
    assert holdout["ordinary_session_count"] == 12
    assert set(holdout["boundary_category_counts"]) == set(PHASE69_BOUNDARY_CATEGORIES)
    assert set(holdout["ordinary_category_counts"]) == set(PHASE69_ORDINARY_CATEGORIES)
    assert all(row["simulated_usage"] is True for row in holdout["sessions"])
    assert all(row["actual_user_feedback"] is False for row in holdout["sessions"])
    assert all(row["not_for_training"] is True for row in holdout["sessions"])


def test_phase69_candidate_contract_has_all_three_groundable_atoms() -> None:
    candidates = build_phase59_proposition_candidates(PHASE69_CANDIDATE_CONTRACT)
    values = {(row["field"], row["value"]) for row in candidates}

    assert ("source_registration", "exclude_actual") in values
    assert ("user_outcome_status", "suspended_or_negated") in values
    assert ("test_to_user_outcome_relation", "does_not_establish") in values


def test_phase69_ab_runtime_changes_only_candidate_contract() -> None:
    messages = [
        {"role": "system", "content": "short"},
        {"role": "user", "content": "旧任务"},
        {"role": "assistant", "content": "旧回答"},
        {"role": "user", "content": "只判断当前测试能否证明用户结果"},
    ]

    baseline = build_phase69_runtime_messages(messages, variant="baseline_runtime")
    candidate = build_phase69_runtime_messages(messages, variant="candidate_boundary_contract")

    assert baseline.manifest["privacy_transform_enabled"] is True
    assert baseline.manifest["latest_intent_enabled"] is True
    assert baseline.manifest["candidate_contract_enabled"] is False
    assert candidate.manifest["candidate_contract_enabled"] is True
    assert PHASE69_CANDIDATE_CONTRACT in candidate.messages[0]["content"]
    assert PHASE69_CANDIDATE_CONTRACT not in baseline.messages[0]["content"]
    assert baseline.messages[-1] == candidate.messages[-1]


def _transcript(session_id: str, variant: str, output: str) -> dict:
    return {
        "session_id": session_id,
        "variant": variant,
        "status": "completed",
        "model_id": "same",
        "device": "mps",
        "generation_protocol_sha256": "protocol",
        "task_sha256": session_id,
        "candidate_contract_enabled": variant == "candidate_boundary_contract",
        "privacy_runtime_enabled": True,
        "latest_intent_runtime_enabled": True,
        "turns": [{"role": "assistant", "content": output}],
    }


def test_phase69_ordinary_control_detects_boundary_leak() -> None:
    session = next(
        row for row in build_phase69_holdout()["sessions"]
        if row["session_id"] == "phase69-ordinary-01"
    )
    baseline = _transcript(session["session_id"], "baseline_runtime", "尽快复核。")
    candidate = _transcript(
        session["session_id"],
        "candidate_boundary_contract",
        "尽快复核。当前材料不得登记为 actual_user_feedback。",
    )

    report = score_phase69_ordinary_transcripts(
        {"baseline_runtime": [baseline], "candidate_boundary_contract": [candidate]},
        [session],
    )

    assert report["variants"]["baseline_runtime"]["pass_rate"] == 1.0
    assert report["variants"]["candidate_boundary_contract"]["pass_rate"] == 0.0
    assert report["variants"]["candidate_boundary_contract"]["boundary_leak_count"] == 1


def test_phase69_parity_requires_single_contract_variable() -> None:
    session = build_phase69_holdout()["sessions"][0]
    baseline = _transcript(session["session_id"], "baseline_runtime", "base")
    candidate = _transcript(session["session_id"], "candidate_boundary_contract", "candidate")

    passed = audit_phase69_ab_parity(
        {"baseline_runtime": [baseline], "candidate_boundary_contract": [candidate]},
        [session],
    )
    candidate["model_id"] = "different"
    failed = audit_phase69_ab_parity(
        {"baseline_runtime": [baseline], "candidate_boundary_contract": [candidate]},
        [session],
    )

    assert passed["passed"] is True
    assert failed["passed"] is False


def test_phase69_decision_requires_real_delta_and_never_changes_default() -> None:
    boundary = {
        "variants": {
            "baseline_runtime": {"completed_count": 36, "accept_rate": 0.75},
            "candidate_boundary_contract": {
                "completed_count": 36,
                "accept_rate": 1.0,
                "dangerous_or_reject_count": 0,
            },
        },
        "candidate_accept_rate_delta": 0.25,
        "schema_failure_count": 0,
        "candidate_value_conflict_count": 0,
    }
    ordinary = {
        "variants": {
            "baseline_runtime": {"count": 12, "pass_rate": 1.0},
            "candidate_boundary_contract": {
                "count": 12,
                "pass_rate": 1.0,
                "boundary_leak_count": 0,
            },
        }
    }

    decision = build_phase69_decision(
        phase68_snapshot={"passed": True},
        parity_audit={"passed": True},
        boundary_report=boundary,
        ordinary_report=ordinary,
        evidence_freezes_passed=True,
    )

    assert decision["recommendation"] == "recommend_phase69_runtime_contract_for_manual_review_only"
    assert decision["phase70_product_runtime_integration_design_eligible"] is True
    assert decision["product_default_change_allowed"] is False
    assert decision["training_allowed"] is False
    assert decision["auto_promote_allowed"] is False


def test_phase69_decision_holds_when_delta_is_too_small() -> None:
    boundary = {
        "variants": {
            "baseline_runtime": {"completed_count": 36, "accept_rate": 0.9},
            "candidate_boundary_contract": {
                "completed_count": 36,
                "accept_rate": 1.0,
                "dangerous_or_reject_count": 0,
            },
        },
        "candidate_accept_rate_delta": 0.1,
        "schema_failure_count": 0,
        "candidate_value_conflict_count": 0,
    }
    ordinary = {
        "variants": {
            "baseline_runtime": {"count": 12, "pass_rate": 1.0},
            "candidate_boundary_contract": {
                "count": 12,
                "pass_rate": 1.0,
                "boundary_leak_count": 0,
            },
        }
    }

    decision = build_phase69_decision(
        phase68_snapshot={"passed": True},
        parity_audit={"passed": True},
        boundary_report=boundary,
        ordinary_report=ordinary,
        evidence_freezes_passed=True,
    )

    assert decision["recommendation"] == "hold_phase69_minimal_runtime_ab"
    assert "candidate_delta_gate" in decision["failed_checks"]
