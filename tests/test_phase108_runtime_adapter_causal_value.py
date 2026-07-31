from __future__ import annotations

from copy import deepcopy

from pfe_core.phase108_runtime_adapter_causal_value import (
    PHASE108_CALL_BUDGET,
    PHASE108_DOMAINS,
    aggregate_phase108_runtime_evidence,
    aggregate_phase108_scores,
    audit_phase108_sessions,
    build_phase108_decision,
    build_phase108_sessions,
    build_phase108_stopping_criteria,
    compare_phase108_variants,
    phase108_content_complete,
    phase108_diagnostic_session_ids,
    score_phase108_session,
)


def test_phase108_holdout_is_fresh_balanced_and_budgeted() -> None:
    payload = build_phase108_sessions()
    sessions = payload["sessions"]
    assert payload["session_count"] == 40
    assert payload["domain_counts"] == {domain: 10 for domain in sorted(PHASE108_DOMAINS)}
    assert payload["main_model_calls"] == 240
    assert payload["diagnostic_model_calls"] == 20
    assert payload["confirmation_call_reserve"] == 40
    assert payload["total_call_budget"] == PHASE108_CALL_BUDGET == 300
    assert audit_phase108_sessions(sessions)["passed"] is True
    assert len(phase108_diagnostic_session_ids(sessions)) == 10


def test_phase108_holdout_audit_detects_prior_overlap() -> None:
    sessions = build_phase108_sessions()["sessions"]
    audit = audit_phase108_sessions(
        sessions,
        previous_texts=[sessions[0]["user_goal"]],
    )
    assert audit["passed"] is False
    assert audit["exact_prior_overlap_count"] == 1


def test_phase108_simulated_user_score_uses_goal_correction_and_boundaries() -> None:
    session = build_phase108_sessions()["sessions"][0]
    good = score_phase108_session(
        session=session,
        outputs=[
            "资料 A 记录周二启动。",
            "- 资料 A：周二启动，尚不能证明运行成功。\n- 资料 B：周三收到日志。\n- 下一步：核验启动记录。",
        ],
        native_turns_complete=True,
        system_contract_enabled=True,
    )
    bad = score_phase108_session(
        session=session,
        outputs=["周二运行成功。", "试验已经完成。"],
        native_turns_complete=True,
        system_contract_enabled=True,
    )
    assert good["accepted"] is True
    assert good["task_complete"] is True
    assert good["correction_followed"] is True
    assert bad["accepted"] is False
    assert bad["factual_guard"] is False
    assert good["overall_product_score"] > bad["overall_product_score"]


def test_phase108_raw_privacy_echo_is_not_hidden_by_returned_output_guard() -> None:
    session = build_phase108_sessions()["sessions"][6]
    score = score_phase108_session(
        session=session,
        outputs=["召回率待复测。", "- 召回率待复测\n- 延迟只测过一次"],
        native_turns_complete=True,
        system_contract_enabled=True,
        raw_privacy_echo_detected=True,
    )
    assert score["returned_output_privacy_guard"] is True
    assert score["raw_privacy_echo_detected"] is True
    assert score["privacy_boundary"] is False
    assert score["accepted"] is False


def test_phase108_contract_score_rejects_external_law_invention() -> None:
    session = build_phase108_sessions()["sessions"][30]
    good = "摘要：付款后 30 日交付。\n风险：交付条件需核验。\n引用：contract-a:payment\n人工确认：不输出法律结论。"
    bad = good + "\n民法典第 999 条已经确认该条款合法。"
    good_score = score_phase108_session(
        session=session,
        outputs=[good, good],
        native_turns_complete=True,
        system_contract_enabled=True,
    )
    bad_score = score_phase108_session(
        session=session,
        outputs=[good, bad],
        native_turns_complete=True,
        system_contract_enabled=True,
    )
    assert good_score["factual_guard"] is True
    assert bad_score["factual_guard"] is False
    assert bad_score["external_law_reference"] is True


def test_phase108_paired_bootstrap_is_deterministic_and_session_paired() -> None:
    sessions = build_phase108_sessions()["sessions"][:8]
    benchmark = []
    candidate = []
    for index, session in enumerate(sessions):
        base = {
            "session_id": session["session_id"],
            "domain": session["domain"],
            "overall_product_score": 0.50,
        }
        improved = {**base, "overall_product_score": 0.70 if index < 6 else 0.50}
        benchmark.append(base)
        candidate.append(improved)
    first = compare_phase108_variants(
        candidate_scores=candidate,
        benchmark_scores=benchmark,
        comparison="candidate_vs_base",
    )
    second = compare_phase108_variants(
        candidate_scores=candidate,
        benchmark_scores=benchmark,
        comparison="candidate_vs_base",
    )
    assert first == second
    assert first["pair_count"] == 8
    assert first["candidate_wins"] == 6
    assert first["ties"] == 2
    assert first["bootstrap"]["ci_low"] >= 0.0


def _metrics(value: float) -> dict:
    return {
        "accepted_rate": value,
        "task_complete_rate": value,
        "correction_followed_rate": value,
        "preference_adherence_rate": value,
        "factual_guard_rate": 1.0,
        "privacy_boundary_rate": 1.0,
        "false_block_rate": 0.0,
    }


def test_phase108_decision_never_auto_promotes_and_requires_real_gain() -> None:
    runtime = aggregate_phase108_runtime_evidence(
        [
            {
                "usage_class": "simulated_usage",
                "simulated_usage": True,
                "actual_user_feedback": False,
                "training_eligible": False,
                "source_ids": [],
                "generation_origin": "local_model",
                "contract_version": "pfe.provenance.v1",
            }
        ]
    )
    comparisons = {
        "phase107_dpo_vs_base": {
            "candidate_wins": 28,
            "benchmark_wins": 5,
            "improved_domain_count": 4,
        },
        "phase107_dpo_vs_phase106_sft": {
            "candidate_wins": 24,
            "benchmark_wins": 8,
            "improved_domain_count": 3,
        },
    }
    passed = build_phase108_decision(
        metrics={"base": _metrics(0.50), "phase106_sft": _metrics(0.55), "phase107_dpo": _metrics(0.65)},
        comparisons=comparisons,
        runtime_metrics=runtime,
        phase107_remains_archive=True,
    )
    failed_metrics = deepcopy(_metrics(0.55))
    failed = build_phase108_decision(
        metrics={"base": _metrics(0.50), "phase106_sft": _metrics(0.55), "phase107_dpo": failed_metrics},
        comparisons=comparisons,
        runtime_metrics=runtime,
        phase107_remains_archive=True,
    )
    assert passed["passed"] is True
    assert passed["recommendation"] == "promote_after_manual_review"
    assert passed["product_gate_qualified"] is False
    assert passed["automatic_promotion_allowed"] is False
    assert failed["passed"] is False
    assert failed["recommendation"] == "runtime_contract_primary_archive_adapter"


def test_phase108_aggregate_keeps_product_and_runtime_metrics_separate() -> None:
    session = build_phase108_sessions()["sessions"][0]
    score = score_phase108_session(
        session=session,
        outputs=["资料 A。", "- 资料 A：周二启动。\n- 资料 B：周三收到日志。\n- 下一步：核验。"],
        native_turns_complete=True,
        system_contract_enabled=True,
    )
    metrics = aggregate_phase108_scores([score])
    assert "provenance_envelope_valid_rate" not in metrics
    assert metrics["session_count"] == 1


def test_phase108_content_boundary_matches_each_format_contract() -> None:
    assert phase108_content_complete("标题", format_mode="single_line") is False
    assert phase108_content_complete("标题\n", format_mode="single_line") is True
    assert phase108_content_complete("谨慎结论。", format_mode="single_sentence") is True
    assert phase108_content_complete("状态：运行中。\n下一步：查退出码。", format_mode="two_lines") is True
    assert phase108_content_complete(
        "结论：暂不能合并。\n依据：e2e 未运行。\n下一步：补跑 e2e。",
        format_mode="three_sections",
    ) is True
    assert phase108_content_complete(
        "摘要：付款后交付。\n风险：条件待核验。\n引用：contract-a:payment。\n人工确认：不输出法律结论。",
        format_mode="four_sections",
    ) is True
    assert phase108_content_complete("- 第一项。\n- 第二项。", format_mode="bullets", minimum_lines=3) is False
    assert phase108_content_complete("- 第一项。\n- 第二项。\n- 第三项。", format_mode="bullets", minimum_lines=3) is True
    assert phase108_content_complete("证据有限，仍需复测。", format_mode="short_paragraph") is True


def test_phase108_stopping_criterion_reports_native_boundary() -> None:
    class _Tokenizer:
        def decode(self, generated, skip_special_tokens=True):
            return "".join(chr(int(value)) for value in generated)

    stopping, state = build_phase108_stopping_criteria(
        tokenizer=_Tokenizer(),
        input_length=2,
        format_mode="two_lines",
    )
    criterion = stopping[0]
    prefix = [1, 2]
    incomplete = prefix + [ord(value) for value in "状态：运行中。"]
    complete = prefix + [ord(value) for value in "状态：运行中。\n下一步：查退出码。"]
    assert criterion([incomplete], None) is False
    assert criterion([complete], None) is True
    assert state["triggered"] is True
    assert state["decoded_text"].endswith("。")
