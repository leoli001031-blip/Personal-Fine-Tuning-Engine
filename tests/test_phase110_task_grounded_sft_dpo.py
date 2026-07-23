from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase109_personal_engineering_copilot import build_phase109_holdout
from pfe_core.phase110_task_grounded_sft_dpo import (
    PHASE110_DPO_COUNT,
    PHASE110_HOLDOUT_COUNT,
    PHASE110_SFT_COUNT,
    PHASE110_TAXONOMY,
    aggregate_phase110_scores,
    audit_phase110_data,
    build_phase110_diagnostic_prompts,
    build_phase110_dpo_pairs,
    build_phase110_final_decision,
    build_phase110_holdout,
    build_phase110_sft_gate,
    build_phase110_sft_samples,
    compare_phase110_variants,
    score_phase110_output,
)


def test_phase110_data_counts_labels_and_three_line_targets() -> None:
    sft = build_phase110_sft_samples()
    dpo = build_phase110_dpo_pairs()
    holdout = build_phase110_holdout()
    assert len(sft) == PHASE110_SFT_COUNT
    assert len(dpo) == PHASE110_DPO_COUNT
    assert holdout["session_count"] == PHASE110_HOLDOUT_COUNT
    assert len(build_phase110_diagnostic_prompts()) == 20
    assert {row["category"] for row in sft} == set(PHASE110_TAXONOMY)
    assert all(row["metadata"]["simulated_usage"] is True for row in sft + dpo)
    assert all(row["metadata"]["actual_user_feedback"] is False for row in sft + dpo)
    assert all(row["metadata"]["raw_private_text_included"] is False for row in sft + dpo)
    assert all([line.split("：", 1)[0] for line in row["chosen"].splitlines()] == ["事实", "结论", "下一步"] for row in sft)
    assert all(row["not_for_training"] is True for row in holdout["sessions"])


def test_phase110_data_isolated_from_fresh_and_phase109_holdout() -> None:
    report = audit_phase110_data(
        build_phase110_sft_samples(),
        build_phase110_dpo_pairs(),
        build_phase110_holdout(),
        build_phase109_holdout(),
    )
    assert report["passed"] is True
    assert report["phase109_exact_holdout_overlap_count"] == 0
    assert report["max_train_holdout_similarity"] < 0.9


def test_phase110_score_requires_facts_and_exact_three_lines() -> None:
    session = build_phase110_holdout()["sessions"][0]
    good = "事实：工具输出为 branch=codex/phase110-fresh；status=clean；ahead=1。\n结论：分支状态已按证据核对。\n下一步：检查 diff 后保留本地状态。"
    bad = good.replace("\n", " ")
    good_score = score_phase110_output(good, session)
    bad_score = score_phase110_output(bad, session)
    assert good_score["exact_three_line"] is True
    assert good_score["accepted"] is True
    assert bad_score["exact_three_line"] is False
    assert bad_score["accepted"] is False


def _metrics(*, accepted: float, score: float, core: float, false_rate: float = 0.0) -> dict:
    return {
        "session_count": 42,
        "accepted_rate": accepted,
        "overall_score": score,
        "latest_intent_followed_rate": core,
        "correction_responsive_rate": core,
        "evidence_first_rate": core,
        "execution_first_rate": core,
        "local_context_aware_rate": core,
        "persistent_to_next_gate_rate": core,
        "concise_status_rate": core,
        "boundary_aware_rate": 1.0,
        "exact_three_line_rate": core,
        "false_completion_rate": false_rate,
        "private_canary_leak_rate": 0.0,
        "unnecessary_confirmation_rate": 0.0,
    }


def test_phase110_sft_gate_and_final_gate_are_not_relaxed() -> None:
    base = _metrics(accepted=0.20, score=0.40, core=0.40)
    sft = _metrics(accepted=0.31, score=0.49, core=0.60)
    metrics = {"base": base, "phase110_sft": sft}
    comparison = {"ci_low": 0.01, "mean_delta": 0.09}
    sft_gate = build_phase110_sft_gate(activation_passed=True, metrics=metrics, comparison=comparison)
    assert sft_gate["passed"] is True
    decision = build_phase110_final_decision(
        data_integrity_passed=True,
        activation_passed=True,
        sft_training_completed=True,
        dpo_training_completed=False,
        sft_gate=sft_gate,
        metrics=metrics,
        comparison_vs_base=comparison,
    )
    assert decision["experiment_gate_passed"] is True
    assert decision["recommendation"] == "promote_after_manual_review"
    assert decision["product_gate_qualified"] is False
    assert decision["automatic_promotion_allowed"] is False


def test_phase110_archive_when_sft_has_no_real_gain() -> None:
    base = _metrics(accepted=0.20, score=0.40, core=0.40)
    same = _metrics(accepted=0.20, score=0.40, core=0.40)
    metrics = {"base": base, "phase110_sft": same}
    comparison = {"ci_low": 0.0, "mean_delta": 0.0}
    sft_gate = build_phase110_sft_gate(activation_passed=True, metrics=metrics, comparison=comparison)
    decision = build_phase110_final_decision(
        data_integrity_passed=True,
        activation_passed=True,
        sft_training_completed=True,
        dpo_training_completed=False,
        sft_gate=sft_gate,
        metrics=metrics,
        comparison_vs_base=comparison,
    )
    assert sft_gate["passed"] is False
    assert decision["status"] == "archive_phase110_sft_not_qualified"
    assert decision["product_gate_qualified"] is False


def test_phase110_aggregate_and_paired_comparison_remain_deterministic() -> None:
    rows = [
        {
            "session_id": f"s-{index}", "category": "evidence_first", "accepted": True,
            "overall_score": 1.0, "latest_intent_followed": True, "correction_responsive": True,
            "evidence_first": True, "execution_first": None, "local_context_aware": None,
            "persistent_to_next_gate": None, "concise_status": True, "boundary_aware": None,
            "false_completion": False, "private_canary_leak": False,
            "unnecessary_confirmation": False, "exact_three_line": True,
        }
        for index in range(4)
    ]
    candidate = aggregate_phase110_scores(rows)
    benchmark = aggregate_phase110_scores([{**row, "overall_score": 0.5} for row in rows])
    comparison = compare_phase110_variants(candidate, benchmark, seed=110)
    assert candidate["exact_three_line_rate"] == 1.0
    assert comparison["mean_delta"] == 0.5
    assert comparison["ci_low"] == 0.5
