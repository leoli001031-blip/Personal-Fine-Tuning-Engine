from __future__ import annotations

import copy

from pfe_core.phase76_conditional_persona_runtime import build_phase76_holdout
from pfe_core.phase77_private_value_guarded_runtime import (
    PHASE77_CONTROL_COUNT,
    PHASE77_PRIVATE_PLACEHOLDER,
    PHASE77_TARGET_COUNT,
    audit_phase77_ordinary_identity,
    audit_phase77_private_guards,
    audit_phase77_public_private_values,
    audit_phase77_routes,
    build_phase77_blind_pairs,
    build_phase77_decision,
    build_phase77_holdout,
    build_phase77_router_calibration,
    classify_phase77_persona_route,
    guard_phase77_messages,
    guard_phase77_output,
    score_phase77_blind_pairs_deterministic,
    summarize_phase77_blind_results,
)


def _transcript(
    session: dict[str, object],
    variant: str,
    answer: str,
    *,
    echo: bool = False,
) -> dict[str, object]:
    turns = []
    for user in (
        session["user_goal"],
        session["user_correction"],
        session["continuation_request"],
    ):
        turns.extend(
            (
                {"role": "user", "content": str(user)},
                {"role": "assistant", "content": answer},
            )
        )
    for value in session["declared_private_values"]:
        turns = [
            {**row, "content": str(row["content"]).replace(str(value), PHASE77_PRIVATE_PLACEHOLDER)}
            for row in turns
        ]
    expected_routes = list(session["expected_routes"])
    return {
        "session_id": session["session_id"],
        "variant": variant,
        "turns": turns,
        "route_manifests": [{"routed": value} for value in expected_routes],
        "private_input_guards": [
            {
                "replacement_count": 1 if session["declared_private_values"] and index == 0 else 0,
                "model_input_contains_declared_private_value": False,
            }
            for index in range(3)
        ],
        "status": "completed",
        "actual_model_call": True,
        "privacy_canary_echo_detected": echo,
    }


def test_phase77_private_guard_removes_values_and_persists_only_hashes() -> None:
    private = "SYNTHETIC_PHASE77_SECRET"
    messages, input_audit = guard_phase77_messages(
        [{"role": "user", "content": f"do not echo {private}"}],
        [private],
    )
    output, output_audit = guard_phase77_output(f"saw {private}", [private])

    assert messages[0]["content"] == f"do not echo {PHASE77_PRIVATE_PLACEHOLDER}"
    assert input_audit["replacement_count"] == 1
    assert input_audit["model_input_contains_declared_private_value"] is False
    assert private not in str(input_audit)
    assert output == f"saw {PHASE77_PRIVATE_PLACEHOLDER}"
    assert output_audit["raw_model_private_echo_detected"] is True
    assert output_audit["returned_output_contains_declared_private_value"] is False
    assert private not in str(output_audit)


def test_phase77_router_calibration_and_fresh_holdout_audit_are_exact() -> None:
    calibration = build_phase77_router_calibration()
    holdout = build_phase77_holdout()
    audit = audit_phase77_routes(holdout["sessions"])

    assert calibration["case_count"] == 64
    assert calibration["accuracy"] == 1.0
    assert calibration["passed"] is True
    assert holdout["session_count"] == PHASE77_TARGET_COUNT + PHASE77_CONTROL_COUNT == 48
    assert holdout["target_count"] == 36
    assert holdout["ordinary_control_count"] == 12
    assert holdout["privacy_target_count"] == 6
    assert audit["decision_count"] == 144
    assert audit["accuracy"] == 1.0
    assert audit["passed"] is True

    old = build_phase76_holdout()["sessions"]
    keys = ("user_goal", "user_correction", "continuation_request")
    old_text = {str(row[key]).strip().lower() for row in old for key in keys}
    new_text = {str(row[key]).strip().lower() for row in holdout["sessions"] for key in keys}
    assert not old_text & new_text
    assert all(row["not_for_training"] is True for row in holdout["sessions"])
    assert all(row["actual_user_feedback"] is False for row in holdout["sessions"])


def test_phase77_router_uses_latest_explicit_action_then_inherits_context() -> None:
    switched = classify_phase77_persona_route(
        [
            {"role": "user", "content": "把 adapter status 翻译成英文。"},
            {"role": "assistant", "content": "adapter status"},
            {"role": "user", "content": "现在检查 git 状态。"},
        ]
    )
    ordinary = classify_phase77_persona_route(
        [
            {"role": "user", "content": "检查 git 状态。"},
            {"role": "assistant", "content": "clean"},
            {"role": "user", "content": "把结果翻译成英文，只给译文。"},
        ]
    )
    negated = classify_phase77_persona_route(
        [
            {"role": "user", "content": "润色‘服务有点慢’。"},
            {"role": "assistant", "content": "服务响应较慢。"},
            {"role": "user", "content": "不要检查服务，只交付文字。"},
        ]
    )

    assert switched["routed"] is True
    assert switched["reason"] == "latest_explicit_workflow_action"
    assert ordinary["routed"] is False
    assert ordinary["reason"] == "latest_explicit_ordinary_action"
    assert negated["routed"] is False
    assert negated["reason"] == "inherited_ordinary_context"


def test_phase77_blind_pair_hides_identity_and_reports_slices() -> None:
    sessions = build_phase77_holdout()["sessions"]
    transcripts = {
        "base_minimal": [_transcript(row, "base_minimal", "已经完成，可以上线。") for row in sessions],
        "guarded_conditional_persona_runtime": [
            _transcript(
                row,
                "guarded_conditional_persona_runtime",
                "结论：blocked\n依据：当前证据不足。\n下一步：继续检查。",
            )
            for row in sessions
        ],
    }
    blind = build_phase77_blind_pairs(transcripts, sessions)
    public = str(blind["public_pairs"])
    assert blind["pair_count"] == 48
    assert "base_minimal" not in public
    assert "guarded_conditional_persona_runtime" not in public
    assert PHASE77_PRIVATE_PLACEHOLDER in public
    assert all(
        private_value not in public
        for session in sessions
        for private_value in session["declared_private_values"]
    )
    public_private_audit = audit_phase77_public_private_values(blind["public_pairs"], sessions)
    assert public_private_audit["passed"] is True
    assert public_private_audit["raw_private_value_pair_count"] == 0
    assert public_private_audit["redaction_marker_pair_count"] == 6

    results = score_phase77_blind_pairs_deterministic(blind, sessions)
    summary = summarize_phase77_blind_results(results, blind["hidden_key"], blind["public_pairs"])
    assert summary["invalid_result_count"] == 0
    assert summary["slices"]["persona_target"]["pair_count"] == 36
    assert summary["slices"]["ordinary_control"]["pair_count"] == 12


def test_phase77_ordinary_identity_and_private_guard_audits() -> None:
    sessions = build_phase77_holdout()["sessions"]
    base = [_transcript(row, "base_minimal", "direct result", echo=True) for row in sessions]
    candidate = [
        _transcript(row, "guarded_conditional_persona_runtime", "direct result", echo=False)
        for row in sessions
    ]
    controls = [row for row in sessions if row["task_type"] == "ordinary_control"]
    for row in base:
        if row["session_id"] in {item["session_id"] for item in controls}:
            row["privacy_canary_echo_detected"] = False
    ordinary = audit_phase77_ordinary_identity(
        {"base_minimal": base, "guarded_conditional_persona_runtime": candidate},
        sessions,
    )
    private = audit_phase77_private_guards(
        {"base_minimal": base, "guarded_conditional_persona_runtime": candidate},
        sessions,
    )
    assert ordinary["passed"] is True
    assert ordinary["full_transcript_identity_rate"] == 1.0
    assert private["passed"] is True
    assert private["candidate_raw_model_echo_rate"] == 0.0

    changed = copy.deepcopy(candidate)
    changed[-1]["turns"][-1]["content"] = "changed"
    failed = audit_phase77_ordinary_identity(
        {"base_minimal": base, "guarded_conditional_persona_runtime": changed},
        sessions,
    )
    assert failed["passed"] is False


def _metrics(score: float) -> dict[str, object]:
    categories = {
        name: {"composite_personalization_score": score, "hard_gate_pass_rate": 1.0}
        for name in (
            "evidence_truthfulness",
            "latest_action_switch",
            "provenance_labeling",
            "autonomous_execution",
            "concise_workstyle",
            "privacy_non_echo",
            "ordinary_direct",
        )
    }
    return {
        "actual_model_calls": True,
        "session_count": 48,
        "category_metrics": categories,
        "privacy_canary_echo_rate": 0.0,
        "unsupported_claim_rate": 0.0,
    }


def test_phase77_decision_requires_guard_and_semantic_gates() -> None:
    exact = {"passed": True, "accuracy": 1.0}
    identity = {"passed": True, "full_transcript_identity_rate": 1.0, "route_off_rate": 1.0}
    guard = {"passed": True}
    blind = {"slices": {"persona_target": {"candidate_win_rate": 0.65}}}
    decision = build_phase77_decision(
        base_metrics=_metrics(0.65),
        candidate_metrics=_metrics(0.78),
        router_calibration=exact,
        route_audit=exact,
        ordinary_identity=identity,
        private_guard_audit=guard,
        public_private_audit={"passed": True},
        deterministic=blind,
        independent={"gemma4:31b": blind, "qwen3.6": blind},
    )
    assert decision["status"] == "qualified_guarded_runtime_reference"
    assert decision["next_gate"] == "phase78_persona_internalization_training_design"
    assert decision["auto_promotion_allowed"] is False

    guard["passed"] = False
    held = build_phase77_decision(
        base_metrics=_metrics(0.65),
        candidate_metrics=_metrics(0.78),
        router_calibration=exact,
        route_audit=exact,
        ordinary_identity=identity,
        private_guard_audit=guard,
        public_private_audit={"passed": True},
        deterministic=blind,
        independent={"gemma4:31b": blind, "qwen3.6": blind},
    )
    assert held["status"] == "hold"
    assert "private_guard_audit_passed" in held["failed_checks"]

    guard["passed"] = True
    leaked_public = build_phase77_decision(
        base_metrics=_metrics(0.65),
        candidate_metrics=_metrics(0.78),
        router_calibration=exact,
        route_audit=exact,
        ordinary_identity=identity,
        private_guard_audit=guard,
        public_private_audit={"passed": False},
        deterministic=blind,
        independent={"gemma4:31b": blind, "qwen3.6": blind},
    )
    assert leaked_public["status"] == "hold"
    assert "public_blind_package_private_zero" in leaked_public["failed_checks"]
