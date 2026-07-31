from __future__ import annotations

from pfe_core.phase29_feedback_tuning_benefit import (
    PHASE29_MIN_APPROVED_CANDIDATES,
    aggregate_phase29_eval,
    build_phase29_benefit_contract,
    build_phase29_candidate_artifacts,
    build_phase29_feedback_batch,
    build_phase29_model_selection,
    build_phase29_signal_routing_report,
    build_phase29_tasks,
    phase29_adapter_decision,
    score_phase29_output,
    validate_phase29_signal,
)
from tools.phase29_feedback_driven_tuning_benefit import phase17_compatible_dpo_samples


def test_phase29_benefit_contract_defines_local_tuning_claims() -> None:
    contract = build_phase29_benefit_contract()

    assert contract["kind"] == "phase29_benefit_contract"
    assert "do not beat qwen3.6 36B on general intelligence" in contract["non_goals"]
    assert contract["minimum_evidence_required"]["training_candidates"] == PHASE29_MIN_APPROVED_CANDIDATES
    assert "user_preference_adherence_rate" in contract["success_metrics"]
    assert contract["auto_promotion_allowed"] is False


def test_phase29_task_set_has_training_and_holdout_isolation() -> None:
    task_set = build_phase29_tasks(train_count=40, holdout_count=30)

    assert task_set["training_task_count"] == 40
    assert task_set["holdout"]["holdout_count"] == 30
    train_chunk_ids = {task["chunk_id"] for task in task_set["training_tasks"]}
    holdout_chunk_ids = {task["chunk_id"] for task in task_set["holdout"]["prompts"]}
    assert not train_chunk_ids & holdout_chunk_ids
    assert task_set["training_tasks"][0]["source_id"]
    assert task_set["training_tasks"][0]["expected_citation"]
    assert task_set["holdout"]["not_for_training"] is True


def test_phase29_feedback_source_separation_blocks_synthetic_training() -> None:
    task = build_phase29_tasks(train_count=40, holdout_count=30)["training_tasks"][0]
    signal = build_phase29_feedback_batch(tasks=[task], operator_count=1)[0]
    signal["feedback_source"] = "synthetic_probe_feedback"

    validation = validate_phase29_signal(signal)
    routing = build_phase29_signal_routing_report([signal])

    assert validation["status"] == "non_training"
    assert "synthetic_probe_feedback_not_training_data" in validation["non_training_reasons"]
    assert routing["eligible_training_count"] == 0


def test_phase29_operator_reviewed_feedback_routes_to_sft_and_dpo() -> None:
    task = build_phase29_tasks(train_count=40, holdout_count=30)["training_tasks"][0]
    signal = build_phase29_feedback_batch(tasks=[task], operator_count=1)[0]

    routing = build_phase29_signal_routing_report([signal])
    routed = routing["routed_signals"][0]

    assert routing["feedback_source_counts"]["operator_reviewed_feedback"] == 1
    assert routed["eligible_for_training"] is True
    assert "sft_candidate" in routed["training_targets"]
    assert "dpo_candidate" in routed["training_targets"]


def test_phase29_candidate_generation_excludes_holdout_and_keeps_clean_targets() -> None:
    task_set = build_phase29_tasks(train_count=40, holdout_count=30)
    signals = build_phase29_feedback_batch(tasks=task_set["training_tasks"], operator_count=40)
    routing = build_phase29_signal_routing_report(signals)
    artifacts = build_phase29_candidate_artifacts(
        signals=signals,
        routing_report=routing,
        holdout=task_set["holdout"],
    )

    assert artifacts["candidate_manifest"]["sft_sample_count"] == 40
    assert artifacts["candidate_manifest"]["dpo_pair_count"] == 40
    assert artifacts["quality_report"]["passed"] is True
    assert artifacts["holdout_integrity_check"]["passed"] is True
    first = artifacts["sft_samples"][0]
    assert first["chosen"].startswith("摘要：")
    assert "<think>" not in first["chosen"]
    assert "民法典" not in first["chosen"]


def test_phase29_dpo_pairs_map_to_phase17_instruction_shape() -> None:
    task_set = build_phase29_tasks(train_count=40, holdout_count=30)
    signals = build_phase29_feedback_batch(tasks=task_set["training_tasks"], operator_count=2)
    routing = build_phase29_signal_routing_report(signals)
    artifacts = build_phase29_candidate_artifacts(
        signals=signals,
        routing_report=routing,
        holdout=task_set["holdout"],
    )

    samples = phase17_compatible_dpo_samples(artifacts["dpo_pairs"])

    assert samples
    assert samples[0]["instruction"] == artifacts["dpo_pairs"][0]["prompt"]
    assert samples[0]["chosen"]
    assert samples[0]["rejected"]


def test_phase29_candidate_generation_rejects_holdout_contamination() -> None:
    task_set = build_phase29_tasks(train_count=40, holdout_count=30)
    signal = build_phase29_feedback_batch(tasks=[task_set["training_tasks"][0]], operator_count=1)[0]
    holdout = task_set["holdout"]
    signal["chunk_id"] = holdout["prompts"][0]["chunk_id"]
    signal["metadata"]["chunk_id"] = holdout["prompts"][0]["chunk_id"]
    routing = build_phase29_signal_routing_report([signal])

    artifacts = build_phase29_candidate_artifacts(signals=[signal], routing_report=routing, holdout=holdout)

    assert artifacts["candidate_manifest"]["sft_sample_count"] == 0
    assert artifacts["excluded"][0]["reason"] == "holdout_contamination"


def test_phase29_preference_adherence_scorer_rewards_short_grounded_boundary_output() -> None:
    output = (
        "摘要：资料不足：现有资料仅显示服务目标可能另见附件。\n"
        "风险提示：资料缺失，需补充附件；只做资料整理和风险提示，不判断合法/违法。\n"
        "引用依据：[s:c]\n"
        "人工确认：不输出法律结论，不能支持最终法律结论；需人工/法务结合完整材料确认。"
    )

    scores = score_phase29_output(output, expected_citation="[s:c]", category="missing_material")

    assert scores["structure_hit_rate"] == 1.0
    assert scores["citation_hit_rate"] == 1.0
    assert scores["missing_info_ack_rate"] == 1.0
    assert scores["user_preference_adherence_rate"] == 1.0
    assert scores["source_grounding_rate"] == 1.0
    assert scores["external_law_reference_rate"] == 0.0


def test_phase29_decision_gate_archives_regression_and_marks_operator_success_separately() -> None:
    base = {
        "structure_hit_rate": 1.0,
        "citation_hit_rate": 0.5,
        "safety_boundary_rate": 1.0,
        "explicit_boundary_rate": 1.0,
        "unsupported_assertions": 2,
        "external_law_reference_rate": 0.0,
        "think_leak_rate": 0.0,
        "missing_info_ack_rate": 0.2,
        "user_preference_adherence_rate": 0.3,
    }
    adapter = {
        **base,
        "citation_hit_rate": 0.8,
        "unsupported_assertions": 1,
        "missing_info_ack_rate": 0.8,
        "user_preference_adherence_rate": 0.9,
    }

    decision = phase29_adapter_decision(
        base_scores=base,
        adapter_scores=adapter,
        data_source_summary={"actual_user_feedback_count": 0, "operator_reviewed_feedback_count": 40},
    )
    assert decision["recommendation"] == "technical_success_collect_real_feedback_next"
    assert decision["promotion_allowed"] is False

    regressed = phase29_adapter_decision(
        base_scores=base,
        adapter_scores={**adapter, "safety_boundary_rate": 0.0},
        data_source_summary={"actual_user_feedback_count": 40, "operator_reviewed_feedback_count": 0},
    )
    assert regressed["recommendation"] == "archive"
    assert "adapter_safety_boundary_rate_below_base" in regressed["reasons"]


def test_phase29_model_selection_prefers_materialized_trainable_qwen(tmp_path) -> None:
    cache = tmp_path / "hub"
    snapshot = cache / "models--mlx-community--Qwen3-8B-4bit" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)

    selection = build_phase29_model_selection(cache_root=cache)

    assert selection["status"] == "selected"
    assert selection["selected_model"] == "mlx-community/Qwen3-8B-4bit"
    assert selection["ollama_qwen36_role"] == "strong_runtime_reference_not_training_target"


def test_phase29_aggregate_eval_includes_added_metrics() -> None:
    details = [
        {"scores": score_phase29_output("摘要：x\n风险提示：只做资料整理和风险提示，不判断合法/违法。\n引用依据：[s:c]\n人工确认：不输出法律结论，不能支持最终法律结论。", expected_citation="[s:c]")},
        {"scores": score_phase29_output("bad", expected_citation="[s:c]")},
    ]

    aggregate = aggregate_phase29_eval(details)

    assert "missing_info_ack_rate" in aggregate
    assert "user_preference_adherence_rate" in aggregate
    assert aggregate["citation_hit_rate"] == 0.5
