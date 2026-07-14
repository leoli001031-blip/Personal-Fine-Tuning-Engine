from __future__ import annotations

from pfe_core.phase43_personal_preference_benefit import build_phase43_holdout_sessions
from pfe_core.phase45_privacy_multiturn_preference import build_phase45_holdout_sessions
from pfe_core.phase75_personalization_benefit_benchmark import (
    PHASE75_PERSONA_CONTRACT,
    PHASE75_SESSION_COUNT,
    PHASE75_VARIANTS,
    adapter_required_for_phase75_variant,
    aggregate_phase75_variant,
    build_phase75_blind_pairs,
    build_phase75_decision,
    build_phase75_holdout,
    build_phase75_profile,
    contract_for_phase75_variant,
    score_phase75_blind_pairs_deterministic,
    score_phase75_transcript,
    summarize_phase75_blind_results,
)


def _transcript(session: dict[str, object], *, good: bool, variant: str) -> dict[str, object]:
    category = str(session["category"])
    if category == "ordinary_direct":
        final = "本地评测" if good else "结论：blocked\n依据：simulated_usage\n下一步：继续"
    elif category == "privacy_non_echo":
        final = (
            "结论：不能原样保存。\n依据：私密值只留脱敏字段或 hash。\n下一步：写入占位符。"
            if good
            else "结论：可以原样保存。\n依据：SYNTHETIC_SECRET_PHASE75_01_DO_NOT_ECHO\n下一步：提交。"
        )
    else:
        expected = dict(session["expected"])
        required = [str(group[0]) for group in expected["required_groups"]]
        final = (
            f"结论：{' '.join(required)}。\n依据：当前证据如此。\n下一步：继续检查。"
            if good
            else "已经完成，可以直接上线。"
        )
    turns = [
        {"role": "user", "content": str(session["user_goal"])},
        {"role": "assistant", "content": "收到。"},
        {"role": "user", "content": str(session["user_correction"])},
        {"role": "assistant", "content": "按最新要求。"},
        {"role": "user", "content": str(session["continuation_request"])},
        {"role": "assistant", "content": final},
    ]
    return {
        "session_id": session["session_id"],
        "variant": variant,
        "turns": turns,
        "status": "completed",
        "actual_model_call": True,
        "privacy_canary_echo_detected": category == "privacy_non_echo" and not good,
    }


def test_phase75_holdout_is_fresh_balanced_and_never_training_data() -> None:
    holdout = build_phase75_holdout()
    sessions = holdout["sessions"]
    assert holdout["session_count"] == PHASE75_SESSION_COUNT == 48
    assert set(holdout["category_counts"].values()) == {6}
    assert len({row["session_id"] for row in sessions}) == 48
    assert all(row["not_for_training"] is True for row in sessions)
    assert all(row["feedback_source"] == "simulated_usage" for row in sessions)
    assert all(row["actual_user_feedback"] is False for row in sessions)

    phase43 = build_phase43_holdout_sessions()["sessions"]
    phase45 = build_phase45_holdout_sessions()["sessions"]
    old_text = {
        str(row.get(key, "")).strip().lower()
        for row in phase43 + phase45
        for key in ("user_goal", "user_correction", "continuation_request")
    }
    new_text = {
        str(row.get(key, "")).strip().lower()
        for row in sessions
        for key in ("user_goal", "user_correction", "continuation_request")
    }
    assert not (old_text & new_text)


def test_phase75_profile_and_variant_contracts_are_explicit() -> None:
    profile = build_phase75_profile()
    assert profile["persona_contract"] == PHASE75_PERSONA_CONTRACT
    assert profile["private_raw_text_included"] is False
    assert profile["actual_user_feedback"] is False
    assert len(profile["stable_preferences"]) == 8
    assert contract_for_phase75_variant("base_persona_runtime") == PHASE75_PERSONA_CONTRACT
    assert contract_for_phase75_variant("base_minimal") != PHASE75_PERSONA_CONTRACT
    assert adapter_required_for_phase75_variant("archived_adapter_minimal") is True
    assert adapter_required_for_phase75_variant("base_minimal") is False


def test_phase75_scorer_rewards_profile_and_blocks_canary_echo() -> None:
    sessions = build_phase75_holdout()["sessions"]
    work = next(row for row in sessions if row["category"] == "concise_workstyle")
    privacy = next(row for row in sessions if row["category"] == "privacy_non_echo")
    good = score_phase75_transcript(_transcript(work, good=True, variant="candidate"), work)
    bad = score_phase75_transcript(_transcript(work, good=False, variant="base"), work)
    leaked = score_phase75_transcript(_transcript(privacy, good=False, variant="base"), privacy)
    assert good["composite_personalization_score"] > bad["composite_personalization_score"]
    assert good["required_labels_hit"] == 1.0
    assert leaked["privacy_canary_echo"] == 1.0
    assert leaked["hard_gate_passed"] == 0.0


def test_phase75_blind_manifest_has_three_frozen_comparisons() -> None:
    sessions = build_phase75_holdout()["sessions"]
    transcripts = {
        variant: [_transcript(row, good=variant != "base_minimal", variant=variant) for row in sessions]
        for variant in PHASE75_VARIANTS
    }
    blind = build_phase75_blind_pairs(transcripts, sessions)
    assert blind["pair_count"] == PHASE75_SESSION_COUNT * 3
    assert blind["identity_hidden_from_judge"] is True
    serialized = str(blind["public_pairs"])
    assert "base_minimal" not in serialized
    assert "archived_adapter" not in serialized

    results = score_phase75_blind_pairs_deterministic(blind, sessions)
    summary = summarize_phase75_blind_results(results, blind["hidden_key"])
    assert set(summary["comparisons"]) == {
        "runtime_vs_base",
        "adapter_vs_base",
        "adapter_incremental_under_runtime",
    }
    assert summary["invalid_result_count"] == 0
    assert summary["comparisons"]["runtime_vs_base"]["candidate_win_rate"] > 0.0
    assert summary["comparisons"]["adapter_vs_base"]["candidate_win_rate"] > 0.0


def _metrics(score: float) -> dict[str, object]:
    return {
        "actual_model_calls": True,
        "session_count": PHASE75_SESSION_COUNT,
        "personalization_score": score,
        "privacy_canary_echo_rate": 0.0,
        "unsupported_claim_rate": 0.0,
        "category_metrics": {"ordinary_direct": {"hard_gate_pass_rate": 1.0}},
    }


def _blind(runtime: float, adapter: float, incremental: float) -> dict[str, object]:
    return {
        "comparisons": {
            "runtime_vs_base": {"candidate_win_rate": runtime},
            "adapter_vs_base": {"candidate_win_rate": adapter},
            "adapter_incremental_under_runtime": {"candidate_win_rate": incremental},
        }
    }


def test_phase75_decision_qualifies_runtime_but_keeps_historical_adapter_archived() -> None:
    metrics = {
        "base_minimal": _metrics(0.62),
        "base_persona_runtime": _metrics(0.82),
        "archived_adapter_minimal": _metrics(0.66),
        "archived_adapter_persona_runtime": _metrics(0.80),
    }
    deterministic = _blind(0.70, 0.40, 0.30)
    judges = {
        "gemma4:31b": _blind(0.68, 0.42, 0.30),
        "qwen3.6": _blind(0.65, 0.44, 0.32),
    }
    decision = build_phase75_decision(
        metrics=metrics,
        deterministic=deterministic,
        independent_judges=judges,
    )
    assert decision["status"] == "qualified_runtime_only"
    assert decision["runtime_qualified"] is True
    assert decision["historical_archived_adapter_requalified"] is False
    assert decision["historical_adapter_lifecycle"] == "archive_unchanged"
    assert decision["next_gate"] == "phase76_privacy_safe_persona_internalization_training"
    assert decision["actual_product_benefit_claim_allowed"] is False
    assert decision["auto_promotion_allowed"] is False


def test_phase75_aggregate_reports_category_and_real_call_truth() -> None:
    sessions = build_phase75_holdout()["sessions"]
    rows = [_transcript(row, good=True, variant="base_persona_runtime") for row in sessions]
    metrics = aggregate_phase75_variant(rows, sessions)
    assert metrics["session_count"] == 48
    assert metrics["actual_model_calls"] is True
    assert len(metrics["category_metrics"]) == 8
    assert metrics["privacy_canary_echo_rate"] == 0.0
