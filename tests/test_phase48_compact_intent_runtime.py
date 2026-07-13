from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
TOOLS = ROOT / "tools"
for path in (CORE, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pfe_core.phase46_runtime_first_latest_intent import (
    build_phase46_scorer_calibration_cases,
    evaluate_phase46_scorer_calibration,
)
from pfe_core.phase48_compact_intent_runtime import (
    PHASE48_CATEGORIES,
    PHASE48_COMPACT_INTENT_CONTRACT,
    build_phase48_blind_pairs,
    build_phase48_compact_runtime_messages,
    build_phase48_decision,
    build_phase48_holdout_sessions,
    build_phase48_split_integrity,
    score_phase48_blind_pairs_deterministic,
    summarize_phase48_blind_results,
)
from phase48_blind_eval import _sanitize_public_value


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _transcript(session: dict, variant: str, *, good: bool) -> dict:
    expected = session["expected"]
    final = "，".join(expected.get("latest_terms") or expected.get("required_any_terms") or ["未确认"])
    if not good:
        final = "已经完成，继续最初的进度卡片和展示文案。"
    mode = {
        "base_privacy": "privacy_base",
        "base_compact_intent": "compact_system_instruction",
        "base_full_intent": "phase46_full_envelope",
    }[variant]
    return {
        "session_id": session["session_id"],
        "category": session["category"],
        "variant": variant,
        "status": "completed",
        "actual_model_call": True,
        "hardcoded_response": False,
        "turns": [{"role": "assistant", "content": final}],
        "generation": [{"latency_seconds": 0.1}],
        "latency_seconds": [0.1],
        "privacy_runtime": {"input_manifest": {"redaction_count": 0}, "output_audits": []},
        "phase46_runtime": {"intent_manifest": {"latest_intent_wrapped": variant == "base_full_intent"}},
        "phase48_runtime": {"mode": mode},
        "truncated_response": False,
    }


def test_compact_runtime_preserves_history_without_xml_wrapper() -> None:
    messages = [
        {"role": "system", "content": "保持简洁。"},
        {"role": "user", "content": "先写报告。"},
        {"role": "assistant", "content": "我会写报告。"},
        {"role": "user", "content": "取消报告，只检查当前分支。"},
    ]
    result = build_phase48_compact_runtime_messages(messages)

    assert len(result.messages) == len(messages)
    assert PHASE48_COMPACT_INTENT_CONTRACT in result.messages[0]["content"]
    assert result.messages[-1]["content"] == messages[-1]["content"]
    assert "<CURRENT_USER_INTENT>" not in json.dumps(result.messages, ensure_ascii=False)
    assert result.manifest["history_preserved"] is True
    assert result.manifest["xml_or_tag_envelope_used"] is False


def test_compact_runtime_requires_user_and_redacts_phase48_canary() -> None:
    with pytest.raises(ValueError, match="requires a user message"):
        build_phase48_compact_runtime_messages([{"role": "system", "content": "contract"}])

    secret = "PFE48_HOLDOUT_SECRET_01_QX4800"
    result = build_phase48_compact_runtime_messages([{"role": "user", "content": f"不要复述 {secret}"}])
    serialized = json.dumps(result.messages, ensure_ascii=False)
    assert secret not in serialized
    assert "<PRIVATE_PRIVATE_CANARY_01>" in serialized
    assert result.privacy.manifest["redaction_count"] == 1


def test_phase48_holdout_is_fresh_balanced_and_not_training_data() -> None:
    holdout = build_phase48_holdout_sessions()
    rows = holdout["sessions"]

    assert holdout["holdout_count"] == 64
    assert holdout["category_counts"] == {category: 8 for category in sorted(PHASE48_CATEGORIES)}
    assert len({row["session_id"] for row in rows}) == 64
    assert all(row["not_for_training"] is True for row in rows)
    assert all(row["fresh_phase48_eval"] is True for row in rows)
    assert all(row["phase46_holdout_reused"] is False for row in rows)


def test_phase48_split_isolated_from_reviewed_candidates_and_phase46_holdout() -> None:
    reviewed = _jsonl(
        ROOT / "docs/demo/phase47-simulated-user-review/evidence-candidates/reviewed_candidates.jsonl"
    )
    phase46 = json.loads(
        (ROOT / "docs/demo/phase46-runtime-first-latest-intent-ablation/evidence-holdout/holdout.json").read_text(
            encoding="utf-8"
        )
    )["sessions"]
    phase48 = build_phase48_holdout_sessions()["sessions"]

    integrity = build_phase48_split_integrity(reviewed, phase48, prior_holdout_sessions=phase46)
    assert integrity["passed"] is True
    assert integrity["candidate_exact_text_overlap"] == []
    assert integrity["prior_holdout_exact_text_overlap"] == []
    assert integrity["phase47_reviewed_candidates_used_for_holdout"] is False


def test_phase46_frozen_scorer_calibration_still_passes() -> None:
    report = evaluate_phase46_scorer_calibration(build_phase46_scorer_calibration_cases()["cases"])
    assert report["status"] == "passed"
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_phase48_blind_pairs_hide_identity_and_use_all_sessions() -> None:
    sessions = build_phase48_holdout_sessions()["sessions"]
    transcripts = {
        variant: [_transcript(row, variant, good=variant == "base_compact_intent") for row in sessions]
        for variant in ("base_privacy", "base_compact_intent", "base_full_intent")
    }
    blind = build_phase48_blind_pairs(transcripts, sessions)

    assert blind["pair_count"] == 128
    assert all(
        all(turn["role"] == "assistant" for turn in pair[side]["turns"])
        for pair in blind["public_pairs"]
        for side in ("variant_left", "variant_right")
    )
    public = json.dumps(blind["public_pairs"], ensure_ascii=False)
    assert "base_compact_intent" not in public
    assert "base_full_intent" not in public


def test_phase48_public_blind_evidence_sanitizes_nested_canaries() -> None:
    secret = "PFE48_HOLDOUT_SECRET_01_QX4800"
    payload = {"user_goal": f"检查 {secret}", "expected": {"privacy_canary": secret, "terms": [secret]}}

    sanitized = _sanitize_public_value(payload)
    serialized = json.dumps(sanitized, ensure_ascii=False)

    assert secret not in serialized
    assert serialized.count("<SYNTHETIC_CANARY>") == 3


def test_phase48_deterministic_summary_reports_non_tie_win_rate() -> None:
    sessions = build_phase48_holdout_sessions()["sessions"]
    transcripts = {
        variant: [_transcript(row, variant, good=variant == "base_compact_intent") for row in sessions]
        for variant in ("base_privacy", "base_compact_intent", "base_full_intent")
    }
    blind = build_phase48_blind_pairs(transcripts, sessions)
    results = score_phase48_blind_pairs_deterministic(blind)
    summary = summarize_phase48_blind_results(results, blind["hidden_key"])

    assert summary["comparisons"]["compact_vs_privacy_base"]["candidate_non_tie_win_rate"] == 1.0
    assert summary["comparisons"]["compact_vs_full_envelope"]["candidate_non_tie_win_rate"] == 1.0


def _metrics(score: float, latest: float, residue: float, repetition: float = 0.10) -> dict:
    return {
        "actual_model_calls": True,
        "session_count": 64,
        "truncated_response_rate": 0.0,
        "privacy_violation_rate": 0.0,
        "secret_echo_rate": 0.0,
        "placeholder_leak_rate": 0.0,
        "over_redaction_rate": 0.0,
        "latest_intent_satisfaction_rate": latest,
        "old_goal_residue_rate": residue,
        "user_preference_score": score,
        "response_diversity": 0.98,
        "repetition_rate": repetition,
    }


def test_phase48_decision_requires_real_blind_and_metric_wins() -> None:
    metrics = {
        "base_privacy": _metrics(0.88, 0.84, 0.05),
        "base_compact_intent": _metrics(0.93, 0.90, 0.01),
        "base_full_intent": _metrics(0.92, 0.89, 0.01),
    }
    comparisons = {
        "compact_vs_privacy_base": {
            "candidate_wins": 32,
            "benchmark_wins": 20,
            "candidate_non_tie_win_rate": 0.6154,
        },
        "compact_vs_full_envelope": {
            "candidate_wins": 28,
            "benchmark_wins": 24,
            "candidate_non_tie_win_rate": 0.5385,
        },
    }
    blind = {"status": "completed", "comparisons": comparisons}
    decision = build_phase48_decision(
        metrics_by_variant=metrics,
        deterministic_blind=blind,
        independent_blind=blind,
        calibration={"status": "passed"},
        split_integrity={"passed": True},
    )

    assert decision["recommendation"] == "recommend_compact_runtime_for_manual_shadow_only"
    assert decision["manual_shadow_trial_allowed"] is True
    assert decision["new_training_allowed"] is False
    assert decision["product_default_change_allowed"] is False


def test_phase48_decision_holds_when_blind_prefers_base() -> None:
    metrics = {
        "base_privacy": _metrics(0.88, 0.84, 0.05),
        "base_compact_intent": _metrics(0.93, 0.90, 0.01),
        "base_full_intent": _metrics(0.92, 0.89, 0.01),
    }
    weak = {
        "status": "completed",
        "comparisons": {
            "compact_vs_privacy_base": {
                "candidate_wins": 18,
                "benchmark_wins": 30,
                "candidate_non_tie_win_rate": 0.375,
            },
            "compact_vs_full_envelope": {
                "candidate_wins": 20,
                "benchmark_wins": 28,
                "candidate_non_tie_win_rate": 0.4167,
            },
        },
    }
    decision = build_phase48_decision(
        metrics_by_variant=metrics,
        deterministic_blind=weak,
        independent_blind=weak,
        calibration={"status": "passed"},
        split_integrity={"passed": True},
    )
    assert decision["recommendation"] == "hold_compact_runtime"
    assert decision["manual_shadow_trial_allowed"] is False


def test_phase47_training_boundary_remains_blocked() -> None:
    decision = json.loads(
        (ROOT / "docs/demo/phase47-simulated-user-review/phase47-final-decision.json").read_text(encoding="utf-8")
    )
    assert decision["training_status"] == "blocked"
    assert decision["actual_human_review_completed"] is False
    assert decision["hermes_attachment_allowed"] is False
