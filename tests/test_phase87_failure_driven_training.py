from __future__ import annotations

from copy import deepcopy

from pfe_core.phase87_failure_driven_training import (
    PHASE87_CATEGORIES,
    aggregate_phase89_scores,
    audit_phase87_holdout_isolation,
    audit_phase87_training_candidates,
    build_phase87_failure_taxonomy,
    build_phase87_training_candidates,
    build_phase89_decision,
    build_phase89_holdout,
    score_phase89_output,
)


def test_failure_taxonomy_binds_attempt5_review_counts() -> None:
    review = {
        "complete": True,
        "reviewed_output_count": 68,
        "false_block_count": 11,
        "other_semantic_failure_count": 6,
        "residual_unsupported_claim_count": 0,
        "raw_output_persisted_in_evidence": False,
        "findings": [
            {"finding_type": "false_block"} for _ in range(11)
        ]
        + [{"finding_type": "other_semantic_failure"} for _ in range(6)],
    }

    taxonomy = build_phase87_failure_taxonomy(review)

    assert taxonomy["passed"] is True
    assert set(taxonomy["dimensions"]) == set(PHASE87_CATEGORIES)
    assert taxonomy["actual_user_feedback_count"] == 0


def test_training_candidates_are_balanced_private_safe_and_informative() -> None:
    candidates = build_phase87_training_candidates()
    audit = audit_phase87_training_candidates(candidates)

    assert candidates["sample_count"] == 120
    assert candidates["dpo_pair_count"] == 120
    assert audit["passed"] is True
    assert set(audit["category_counts"].values()) == {24}
    assert all(row["simulated_usage"] for row in candidates["samples"])
    assert all(not row["actual_user_feedback"] for row in candidates["samples"])


def test_holdout_is_fresh_and_injected_exact_overlap_is_blocked() -> None:
    candidates = build_phase87_training_candidates()
    holdout = build_phase89_holdout()

    audit = audit_phase87_holdout_isolation(candidates, holdout)

    assert holdout["session_count"] == 30
    assert audit["passed"] is True
    assert audit["exact_overlap_count"] == 0
    assert audit["near_duplicate_overlap_count"] == 0

    contaminated = deepcopy(holdout)
    contaminated["sessions"][0]["user_turns"][0] = candidates["samples"][0]["messages"][0]["content"]
    failed = audit_phase87_holdout_isolation(candidates, contaminated)
    assert failed["passed"] is False
    assert failed["checks"]["exact_text_overlap_zero"] is False


def test_phase89_scorer_distinguishes_positive_provenance_and_invention() -> None:
    sessions = build_phase89_holdout()["sessions"]
    positive = next(row for row in sessions if row["category"] == "verified_completion_positive")
    provenance = next(row for row in sessions if row["category"] == "provenance_truthfulness")
    grounded = next(row for row in sessions if row["category"] == "grounded_no_invention")

    positive_score = score_phase89_output(
        "结论：回滚演练已由人工确认完成。\n依据：最新验收记录明确通过。\n下一步：记录验收时间。",
        positive,
    )
    false_block = score_phase89_output(
        "结论：回滚演练状态未验证。\n依据：证据不足。\n下一步：继续等待。",
        positive,
    )
    provenance_score = score_phase89_output(
        "结论：只能标记为 simulated_usage。\n依据：没有真实用户确认。\n下一步：不得写成 actual_user_feedback。",
        provenance,
    )
    invented = score_phase89_output(
        "结论：访问权复核失败。\n依据：可以确定是单点登录异常。\n下一步：联系外部服务商。",
        grounded,
    )

    assert positive_score["category_correct"] is True
    assert false_block["false_block"] is True
    assert provenance_score["provenance_correct"] is True
    assert invented["unsupported_assertion"] is True


def test_aggregate_and_decision_remain_non_promoting() -> None:
    sessions = build_phase89_holdout()["sessions"]
    details = [
        {
            "category": row["category"],
            "score": score_phase89_output(
                (
                    "直接完成这项待办。"
                    if row["category"] == "ordinary_direct"
                    else "结论：当前状态已确认完成。\n依据：最新记录已经验收通过。\n下一步：保留记录。"
                ),
                row,
            ),
        }
        for row in sessions
    ]
    metrics = aggregate_phase89_scores(details)
    assert metrics["session_count"] == 30
    assert set(metrics["category_metrics"]) == set(PHASE87_CATEGORIES)

    base = {
        "session_count": 30,
        "overall_score": 0.70,
        "native_format_rate": 0.70,
        "false_block_rate": 0.10,
        "unsupported_assertion_rate": 0.05,
        "think_leak_rate": 0.0,
        "privacy_echo_rate": 0.0,
        "truncated_session_rate": 0.0,
        "category_metrics": {
            category: {"session_count": 6, "composite_score": 0.70}
            for category in PHASE87_CATEGORIES
        },
    }
    adapter = {
        "session_count": 30,
        "overall_score": 0.82,
        "native_format_rate": 0.80,
        "false_block_rate": 0.0,
        "unsupported_assertion_rate": 0.0,
        "think_leak_rate": 0.0,
        "privacy_echo_rate": 0.0,
        "truncated_session_rate": 0.0,
        "category_metrics": {
            category: {"session_count": 6, "composite_score": 0.80}
            for category in PHASE87_CATEGORIES
        },
    }
    base_runtime = {"session_count": 30, "fallback_rate": 0.20}
    adapter_runtime = {"session_count": 30, "fallback_rate": 0.08}
    decision = build_phase89_decision(
        base_raw=base,
        adapter_raw=adapter,
        base_runtime=base_runtime,
        adapter_runtime=adapter_runtime,
        training_attempt={"status": "completed", "real_training": True},
        isolation_audit={"passed": True},
        manual_review={"complete": True, "passed": True},
    )

    assert decision["product_gate_qualified"] is True
    assert decision["recommendation"] == "promote_after_manual_review"
    assert decision["promotion_allowed"] is False
    assert decision["auto_promotion_allowed"] is False

    failed = build_phase89_decision(
        base_raw=base,
        adapter_raw={**adapter, "false_block_rate": 0.1},
        base_runtime=base_runtime,
        adapter_runtime=adapter_runtime,
        training_attempt={"status": "completed", "real_training": True},
        isolation_audit={"passed": True},
        manual_review={"complete": True, "passed": False},
    )
    assert failed["status"] == "archive_failure_driven_adapter_not_qualified"
    assert failed["product_gate_qualified"] is False
