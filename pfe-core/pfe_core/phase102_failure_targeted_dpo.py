from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping

from .phase99_qwen3_native_generation_boundary import PHASE99_NEAR_DUPLICATE_THRESHOLD


PHASE102_CATEGORIES = ("exact_three_line", "false_block", "provenance")


def select_phase102_dpo_pairs(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = {category: [] for category in PHASE102_CATEGORIES}
    for row in candidates:
        category = str(row.get("category") or "")
        if category in by_category:
            by_category[category].append(dict(row))
    selected = []
    for category in PHASE102_CATEGORIES:
        for row in by_category[category][:8]:
            selected.append({
                "pair_id": str(row.get("sample_id") or "").replace("phase101-sft", "phase102-dpo"),
                "preference_category": category,
                "instruction": row.get("instruction"),
                "chosen": row.get("chosen"),
                "rejected": row.get("rejected"),
                "failure_origin": row.get("failure_origin"),
                "feedback_source": "simulated_usage",
                "simulated_usage": True,
                "actual_user_feedback": False,
                "eligible_for_training": True,
            })
    return selected


def audit_phase102_pairs(
    pairs: Iterable[Mapping[str, Any]],
    holdout: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(row) for row in pairs]
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    counts = {
        category: sum(str(row.get("preference_category")) == category for row in rows)
        for category in PHASE102_CATEGORIES
    }
    train_prompts = {str(row.get("instruction") or "").strip() for row in rows}
    holdout_turns = {
        str(turn).strip()
        for row in sessions
        for turn in row.get("user_turns") or []
        if str(turn).strip()
    }
    near = [
        turn
        for turn in holdout_turns
        if max((SequenceMatcher(None, turn, prompt).ratio() for prompt in train_prompts), default=0.0)
        >= PHASE99_NEAR_DUPLICATE_THRESHOLD
    ]
    checks = {
        "pair_count_24": len(rows) == 24,
        "eight_pairs_per_category": all(value == 8 for value in counts.values()),
        "chosen_rejected_complete": all(row.get("instruction") and row.get("chosen") and row.get("rejected") for row in rows),
        "chosen_rejected_distinct": all(row.get("chosen") != row.get("rejected") for row in rows),
        "all_simulated_not_actual": all(row.get("simulated_usage") is True and row.get("actual_user_feedback") is False for row in rows),
        "holdout_not_for_training": all(row.get("not_for_training") is True for row in sessions),
        "holdout_exact_overlap_zero": not bool(train_prompts & holdout_turns),
        "holdout_near_duplicate_overlap_zero": not near,
    }
    return {
        "kind": "phase102_dpo_pair_holdout_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "category_counts": counts,
        "near_duplicate_count": len(near),
        "near_duplicate_threshold": PHASE99_NEAR_DUPLICATE_THRESHOLD,
    }


def build_phase102_dpo_decision(
    *,
    base_metrics: Mapping[str, Any],
    sft_metrics: Mapping[str, Any],
    runtime_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    training_completed: bool,
) -> dict[str, Any]:
    higher = (
        "exact_three_line_rate",
        "false_block_avoidance_rate",
        "provenance_correct_rate",
        "ordinary_control_rate",
        "complete_content_before_termination_rate",
        "native_termination_rate",
    )
    lower = (
        "unsupported_assertion_rate",
        "think_leak_rate",
        "privacy_echo_rate",
        "repeated_output_rate",
        "extra_text_after_first_answer_rate",
        "forbidden_generation_rate",
    )
    candidate_not_worse_than_runtime = all(
        float(candidate_metrics.get(key) or 0) >= float(runtime_metrics.get(key) or 0)
        for key in higher
    ) and all(
        float(candidate_metrics.get(key) or 0) <= float(runtime_metrics.get(key) or 0)
        for key in lower
    )
    candidate_beats_base = any(
        float(candidate_metrics.get(key) or 0) > float(base_metrics.get(key) or 0)
        for key in higher
    ) or any(
        float(candidate_metrics.get(key) or 0) < float(base_metrics.get(key) or 0)
        for key in lower
    )
    candidate_beats_sft = any(
        float(candidate_metrics.get(key) or 0) > float(sft_metrics.get(key) or 0)
        for key in higher
    ) or any(
        float(candidate_metrics.get(key) or 0) < float(sft_metrics.get(key) or 0)
        for key in lower
    )
    dependency_improved = (
        float(candidate_metrics.get("runtime_control_dependency_rate") or 0)
        < float(runtime_metrics.get("runtime_control_dependency_rate") or 0)
    )
    checks = {
        "real_dpo_training_completed": training_completed,
        "candidate_not_worse_than_runtime_contract": candidate_not_worse_than_runtime,
        "candidate_beats_base_on_core_metric": candidate_beats_base,
        "candidate_beats_archived_sft_on_core_metric": candidate_beats_sft,
        "runtime_control_dependency_improved": dependency_improved,
    }
    passed = all(checks.values())
    return {
        "kind": "phase102_dpo_product_gate",
        "passed": passed,
        "status": "phase102_dpo_candidate_retained" if passed else "archive_phase102_dpo_not_better_than_runtime",
        "checks": checks,
        "next_action": "run_phase103_multiturn_with_best_candidate" if passed else "run_phase103_runtime_vs_base_acceptance",
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
    }
