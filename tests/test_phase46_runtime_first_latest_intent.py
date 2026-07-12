from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pfe_core.phase45_privacy_multiturn_preference import build_phase45_holdout_sessions
from pfe_core.phase46_runtime_first_latest_intent import (
    PHASE46_CATEGORIES,
    aggregate_phase46_variant,
    audit_phase46_curated_candidates,
    build_latest_intent_envelope,
    build_phase46_blind_pairs,
    build_phase46_curated_candidates,
    build_phase46_decision,
    build_phase46_holdout_sessions,
    build_phase46_runtime_messages,
    build_phase46_scorer_calibration_cases,
    build_phase46_split_integrity,
    evaluate_phase46_scorer_calibration,
    score_phase46_blind_pairs_deterministic,
    score_phase46_transcript,
    summarize_phase46_blind_results,
)


def _transcript(session: dict[str, object], *, variant: str, good: bool, runtime: bool) -> dict[str, object]:
    expected = dict(session["expected"])
    required = [str(value) for value in expected.get("required_any_terms") or []]
    forbidden = [str(value) for value in expected.get("forbidden_old_terms") or []]
    if good:
        final = f"当前只处理 {required[0] if required else '最新要求'}。下一步检查当前证据。"
    else:
        final = f"我会继续 {forbidden[0] if forbidden else '旧目标'}，并认为已经完成。"
    row: dict[str, object] = {
        "session_id": session["session_id"],
        "category": session["category"],
        "variant": variant,
        "actual_model_call": True,
        "hardcoded_response": False,
        "status": "completed",
        "latency_seconds": [0.1, 0.1, 0.1],
        "truncated_response": False,
        "turns": [
            {"role": "user", "content": session["user_goal"]},
            {"role": "assistant", "content": "我先按原计划处理。"},
            {"role": "user", "content": session["user_correction"]},
            {"role": "assistant", "content": "收到最新纠正。"},
            {"role": "user", "content": session["continuation_request"]},
            {"role": "assistant", "content": final},
        ],
        "privacy_runtime": {"input_manifest": {"redaction_count": 0}, "output_audits": []},
    }
    if runtime:
        row["phase46_runtime"] = {"intent_manifest": {"latest_intent_wrapped": True}}
    return row


def test_latest_intent_envelope_preserves_history_and_wraps_only_latest_user() -> None:
    messages = [
        {"role": "user", "content": "先做截图。"},
        {"role": "assistant", "content": "我开始整理。"},
        {"role": "user", "content": "取消截图，只检查端口。"},
    ]
    transformed, manifest = build_latest_intent_envelope(messages)

    assert transformed[:2] == messages[:2]
    assert "<CURRENT_USER_INTENT>" in transformed[2]["content"]
    assert "取消截图，只检查端口。" in transformed[2]["content"]
    assert manifest["latest_user_message_index"] == 2
    assert manifest["old_messages_removed"] is False
    assert "取消截图" not in json.dumps(manifest, ensure_ascii=False)


def test_latest_intent_envelope_requires_user_message() -> None:
    with pytest.raises(ValueError, match="requires a user message"):
        build_latest_intent_envelope([{"role": "system", "content": "contract"}])


def test_phase46_runtime_redacts_private_value_before_intent_envelope() -> None:
    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
    messages = [
        {"role": "user", "content": "先检查配置。"},
        {"role": "assistant", "content": "我会读取配置。"},
        {"role": "user", "content": f"不要复述 {secret}，只报告脱敏状态；保留端口 8927。"},
    ]
    result = build_phase46_runtime_messages(messages)
    serialized = json.dumps(result.messages, ensure_ascii=False)

    assert secret not in serialized
    assert "<PRIVATE_API_KEY_01>" in serialized
    assert "8927" in serialized
    assert result.privacy.manifest["raw_values_persisted"] is False
    assert result.intent_manifest["latest_intent_wrapped"] is True


def test_phase46_runtime_redacts_phase46_holdout_canary() -> None:
    secret = "PFE46_HOLDOUT_SECRET_01_ZX4600"
    result = build_phase46_runtime_messages(
        [{"role": "user", "content": f"不要复述 {secret}，只报告脱敏计数。"}]
    )

    serialized = json.dumps(result.messages, ensure_ascii=False)
    assert secret not in serialized
    assert "<PRIVATE_PRIVATE_CANARY_01>" in serialized
    assert result.privacy.manifest["redaction_count"] == 1


def test_curated_pack_is_balanced_unique_and_blocked_from_training() -> None:
    pack = build_phase46_curated_candidates()

    assert pack["status"] == "ready_for_actual_manual_review"
    assert pack["candidate_count"] == 48
    assert pack["category_counts"] == {category: 6 for category in sorted(PHASE46_CATEGORIES)}
    assert pack["audit"]["passed"] is True
    assert pack["audit"]["unique_chosen_ratio"] == 1.0
    assert pack["actual_human_review"] is False
    assert pack["eligible_for_training"] is False
    assert all(row["reviewer_type"] == "codex_simulated_human_perspective" for row in pack["candidates"])
    assert all(row["manual_user_review_required"] is True for row in pack["candidates"])
    assert all(row["eligible_for_training"] is False for row in pack["candidates"])


def test_curated_audit_rejects_duplicate_and_false_human_review_claim() -> None:
    rows = copy.deepcopy(build_phase46_curated_candidates()["candidates"])
    rows[1]["chosen"] = rows[0]["chosen"]
    rows[2]["actual_human_review"] = True
    rows[2]["eligible_for_training"] = True
    audit = audit_phase46_curated_candidates(rows)

    assert audit["passed"] is False
    assert "duplicate_chosen" in audit["reasons"]
    assert "provenance_or_training_gate_invalid" in audit["reasons"]


def test_holdout_has_72_fresh_balanced_sessions_and_split_isolation() -> None:
    pack = build_phase46_curated_candidates()
    holdout = build_phase46_holdout_sessions()
    phase45 = build_phase45_holdout_sessions()["sessions"]
    integrity = build_phase46_split_integrity(pack["candidates"], holdout["sessions"], phase45_holdout_sessions=phase45)

    assert holdout["holdout_count"] == 72
    assert holdout["category_counts"] == {category: 9 for category in sorted(PHASE46_CATEGORIES)}
    assert len({row["session_id"] for row in holdout["sessions"]}) == 72
    assert all(row["not_for_training"] is True for row in holdout["sessions"])
    assert integrity["passed"] is True
    assert integrity["phase45_or_phase46_holdout_used_for_training"] is False


def test_phase46_scorer_detects_latest_intent_and_old_goal_residue() -> None:
    session = {
        "category": "latest_goal_replacement",
        "expected": {
            "required_any_terms": ["端口", "核验"],
            "forbidden_old_terms": ["继续截图"],
            "latest_terms": ["端口", "核验"],
            "action_terms": ["检查"],
        },
    }
    good = {"turns": [{"role": "assistant", "content": "只核验端口，下一步检查当前监听状态。"}]}
    bad = {"turns": [{"role": "assistant", "content": "我会继续截图，再顺便看端口。"}]}

    good_score = score_phase46_transcript(good, session)
    bad_score = score_phase46_transcript(bad, session)
    assert good_score["latest_intent_satisfaction"] == 1.0
    assert good_score["old_goal_residue"] == 0.0
    assert bad_score["latest_intent_satisfaction"] == 0.0
    assert bad_score["old_goal_residue"] == 1.0
    assert good_score["phase46_composite_score"] > bad_score["phase46_composite_score"]


def test_phase46_scorer_calibration_passes() -> None:
    cases = build_phase46_scorer_calibration_cases()
    report = evaluate_phase46_scorer_calibration(cases["cases"])

    assert cases["case_count"] == 8
    assert report["status"] == "passed"
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_phase46_aggregate_includes_category_gates_and_runtime_metrics() -> None:
    sessions = build_phase46_holdout_sessions()["sessions"]
    rows = [_transcript(session, variant="base_privacy_intent", good=True, runtime=True) for session in sessions]
    metrics = aggregate_phase46_variant(rows, sessions)

    assert metrics["session_count"] == 72
    assert metrics["actual_model_calls"] is True
    assert set(metrics["category_metrics"]) == set(PHASE46_CATEGORIES)
    assert all(row["count"] == 9 for row in metrics["category_metrics"].values())
    assert "latest_intent_satisfaction_rate" in metrics
    assert "old_goal_residue_rate" in metrics


def test_phase46_blind_manifest_has_144_pairs_and_hides_identity() -> None:
    sessions = build_phase46_holdout_sessions()["sessions"]
    transcripts = {
        "base_privacy": [_transcript(session, variant="base_privacy", good=False, runtime=False) for session in sessions],
        "base_privacy_intent": [_transcript(session, variant="base_privacy_intent", good=True, runtime=True) for session in sessions],
        "adapter_privacy_intent": [_transcript(session, variant="adapter_privacy_intent", good=False, runtime=True) for session in sessions],
    }
    manifest = build_phase46_blind_pairs(transcripts, sessions)
    deterministic = score_phase46_blind_pairs_deterministic(manifest)
    summary = summarize_phase46_blind_results(deterministic, manifest["hidden_key"])
    public = json.dumps(manifest["public_pairs"], ensure_ascii=False)

    assert manifest["pair_count"] == 144
    assert set(summary["comparisons"]) == {
        "intent_runtime_vs_privacy_base",
        "intent_runtime_base_vs_archived_adapter",
    }
    assert '"variant":' not in public
    assert "adapter_path" not in public
    assert all(
        turn["role"] == "assistant"
        for pair in manifest["public_pairs"]
        for side in ("variant_left", "variant_right")
        for turn in pair[side]["turns"]
    )


def _passing_metrics() -> dict[str, dict[str, object]]:
    base = {
        "actual_model_calls": True,
        "session_count": 72,
        "user_preference_score": 0.75,
        "latest_intent_satisfaction_rate": 0.70,
        "old_goal_residue_rate": 0.10,
        "response_diversity": 0.90,
        "truncated_response_rate": 0.0,
    }
    runtime = {
        "actual_model_calls": True,
        "session_count": 72,
        "user_preference_score": 0.82,
        "latest_intent_satisfaction_rate": 0.80,
        "old_goal_residue_rate": 0.05,
        "response_diversity": 0.92,
        "privacy_violation_rate": 0.0,
        "secret_echo_rate": 0.0,
        "placeholder_leak_rate": 0.0,
        "over_redaction_rate": 0.0,
        "truncated_response_rate": 0.0,
    }
    adapter = {
        "actual_model_calls": True,
        "session_count": 72,
        "user_preference_score": 0.78,
        "truncated_response_rate": 0.0,
    }
    return {"base_privacy": base, "base_privacy_intent": runtime, "adapter_privacy_intent": adapter}


def test_phase46_decision_prefers_runtime_but_never_allows_training_or_promotion() -> None:
    blind = {"status": "completed", "comparisons": {"intent_runtime_vs_privacy_base": {"candidate_win_rate": 0.60}}}
    decision = build_phase46_decision(
        metrics_by_variant=_passing_metrics(),
        deterministic_blind=blind,
        independent_blind=blind,
        calibration={"status": "passed"},
        curated_audit={"passed": True, "actual_human_review_completed": False},
    )

    assert decision["recommendation"] == "runtime_first_no_training"
    assert decision["new_training_allowed"] is False
    assert decision["auto_promotion_allowed"] is False
    assert decision["hermes_attachment_allowed"] is False


def test_phase46_decision_holds_when_runtime_does_not_improve_latest_intent() -> None:
    metrics = _passing_metrics()
    metrics["base_privacy_intent"]["latest_intent_satisfaction_rate"] = 0.72
    blind = {"status": "completed", "comparisons": {"intent_runtime_vs_privacy_base": {"candidate_win_rate": 0.60}}}
    decision = build_phase46_decision(
        metrics_by_variant=metrics,
        deterministic_blind=blind,
        independent_blind=blind,
        calibration={"status": "passed"},
        curated_audit={"passed": True, "actual_human_review_completed": False},
    )

    assert decision["recommendation"] == "hold_runtime_and_revise_eval_or_data"
    assert "runtime_latest_intent_gain_at_least_0_05" in decision["failed_checks"]


def test_phase45_archive_remains_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    decision = json.loads(
        (root / "docs/demo/phase45-privacy-structural-multiturn-preference/phase45-final-decision.json").read_text(encoding="utf-8")
    )

    assert decision["recommendation"] == "archive"
    assert decision["hermes_attachment_allowed"] is False
