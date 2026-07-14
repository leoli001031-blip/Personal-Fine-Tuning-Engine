from __future__ import annotations

import copy

from pfe_core.phase76_conditional_persona_runtime import (
    PHASE76_CONTROL_COUNT,
    PHASE76_TARGET_COUNT,
    audit_phase76_ordinary_identity,
    audit_phase76_routes,
    build_phase76_blind_pairs,
    build_phase76_decision,
    build_phase76_holdout,
    build_phase76_router_calibration,
    classify_phase76_persona_route,
    score_phase76_blind_pairs_deterministic,
    summarize_phase76_blind_results,
)


def _transcript(session: dict[str, object], variant: str, answer: str) -> dict[str, object]:
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
    return {
        "session_id": session["session_id"],
        "variant": variant,
        "turns": turns,
        "route_manifests": [
            {"routed": session["task_type"] == "persona_target"}
            for _ in range(3)
        ],
        "status": "completed",
        "actual_model_call": True,
        "privacy_canary_echo_detected": False,
    }


def test_phase76_router_calibration_and_multiturn_audit_are_exact() -> None:
    calibration = build_phase76_router_calibration()
    holdout = build_phase76_holdout()
    route_audit = audit_phase76_routes(holdout["sessions"])

    assert calibration["case_count"] == 64
    assert calibration["accuracy"] == 1.0
    assert calibration["passed"] is True
    assert route_audit["decision_count"] == 144
    assert route_audit["accuracy"] == 1.0
    assert route_audit["false_positive_count"] == 0
    assert route_audit["false_negative_count"] == 0
    assert route_audit["passed"] is True


def test_phase76_router_keeps_ordinary_context_but_honors_explicit_switch() -> None:
    ordinary = classify_phase76_persona_route(
        [
            {"role": "user", "content": "把 adapter 状态翻译成英文。"},
            {"role": "assistant", "content": "adapter status"},
            {"role": "user", "content": "不要判断训练结果，只给译文。"},
        ]
    )
    switched = classify_phase76_persona_route(
        [
            {"role": "user", "content": "把 adapter 状态翻译成英文。"},
            {"role": "assistant", "content": "adapter status"},
            {"role": "user", "content": "停止翻译，改为检查 git 状态。"},
        ]
    )

    assert ordinary["routed"] is False
    assert ordinary["reason"] == "ordinary_conversation"
    assert switched["routed"] is True
    assert switched["reason"] == "latest_switch_to_workflow"
    assert ordinary["raw_user_text_persisted"] is False


def test_phase76_holdout_is_fresh_balanced_and_not_training_data() -> None:
    holdout = build_phase76_holdout()
    sessions = holdout["sessions"]

    assert holdout["session_count"] == PHASE76_TARGET_COUNT + PHASE76_CONTROL_COUNT == 48
    assert holdout["target_count"] == PHASE76_TARGET_COUNT == 36
    assert holdout["ordinary_control_count"] == PHASE76_CONTROL_COUNT == 12
    assert len({row["session_id"] for row in sessions}) == 48
    assert all(row["not_for_training"] is True for row in sessions)
    assert all(row["feedback_source"] == "simulated_usage" for row in sessions)
    assert all(row["actual_user_feedback"] is False for row in sessions)


def test_phase76_blind_pair_hides_identity_and_reports_target_slice() -> None:
    sessions = build_phase76_holdout()["sessions"]
    transcripts = {
        "base_minimal": [
            _transcript(row, "base_minimal", "已经完成，可以直接上线。")
            for row in sessions
        ],
        "conditional_persona_runtime": [
            _transcript(row, "conditional_persona_runtime", "结论：blocked\n依据：当前证据不足。\n下一步：继续检查。")
            for row in sessions
        ],
    }
    blind = build_phase76_blind_pairs(transcripts, sessions)
    public = str(blind["public_pairs"])

    assert blind["pair_count"] == 48
    assert blind["identity_hidden_from_judge"] is True
    assert "base_minimal" not in public
    assert "conditional_persona_runtime" not in public

    results = score_phase76_blind_pairs_deterministic(blind, sessions)
    summary = summarize_phase76_blind_results(
        results,
        blind["hidden_key"],
        blind["public_pairs"],
    )
    assert summary["invalid_result_count"] == 0
    assert summary["slices"]["persona_target"]["pair_count"] == 36
    assert summary["slices"]["ordinary_control"]["pair_count"] == 12


def test_phase76_ordinary_identity_requires_bytes_and_routes_off() -> None:
    controls = [
        row
        for row in build_phase76_holdout()["sessions"]
        if row["task_type"] == "ordinary_control"
    ]
    base = [_transcript(row, "base_minimal", "direct result") for row in controls]
    candidate = copy.deepcopy(base)
    for row in candidate:
        row["variant"] = "conditional_persona_runtime"
        row["route_manifests"] = [{"routed": False} for _ in range(3)]

    passed = audit_phase76_ordinary_identity(
        {"base_minimal": base, "conditional_persona_runtime": candidate},
        controls,
    )
    assert passed["passed"] is True
    assert passed["full_transcript_identity_rate"] == 1.0
    assert passed["route_off_rate"] == 1.0

    candidate[0]["turns"][-1]["content"] = "changed"
    failed = audit_phase76_ordinary_identity(
        {"base_minimal": base, "conditional_persona_runtime": candidate},
        controls,
    )
    assert failed["passed"] is False


def _metrics(score: float) -> dict[str, object]:
    categories = {
        name: {"composite_personalization_score": score, "hard_gate_pass_rate": 1.0}
        for name in (
            "evidence_truthfulness",
            "latest_correction",
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


def test_phase76_decision_requires_semantic_wins_and_exact_passthrough() -> None:
    exact = {"passed": True, "accuracy": 1.0}
    identity = {
        "passed": True,
        "full_transcript_identity_rate": 1.0,
        "route_off_rate": 1.0,
    }
    blind = {"slices": {"persona_target": {"candidate_win_rate": 0.65}}}
    decision = build_phase76_decision(
        base_metrics=_metrics(0.65),
        candidate_metrics=_metrics(0.78),
        router_calibration=exact,
        route_audit=exact,
        ordinary_identity=identity,
        deterministic=blind,
        independent={"gemma4:31b": blind, "qwen3.6": blind},
    )
    assert decision["status"] == "qualified_runtime_reference"
    assert decision["recommendation"] == "qualified_for_phase77_persona_internalization_training_design"
    assert decision["new_training_executed"] is False
    assert decision["auto_promotion_allowed"] is False

    identity["passed"] = False
    held = build_phase76_decision(
        base_metrics=_metrics(0.65),
        candidate_metrics=_metrics(0.78),
        router_calibration=exact,
        route_audit=exact,
        ordinary_identity=identity,
        deterministic=blind,
        independent={"gemma4:31b": blind, "qwen3.6": blind},
    )
    assert held["status"] == "hold"
    assert "ordinary_transcripts_byte_identical" in held["failed_checks"]
