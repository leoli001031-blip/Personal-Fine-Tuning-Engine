from __future__ import annotations

import json
from pathlib import Path

import pytest

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase42_reliability_hardening import build_phase42_final_decision
from pfe_core.phase43_personal_preference_benefit import (
    PHASE43_MIN_HOLDOUT_SESSIONS,
    PHASE43_MIN_REVIEWED_PAIRS,
    aggregate_phase43_variant,
    build_holdout_integrity_check,
    build_phase43_blind_pairs,
    build_phase43_decision,
    build_phase43_dpo_job_spec,
    build_phase43_holdout_sessions,
    build_phase43_preference_taxonomy,
    build_phase43_sft_job_spec,
    review_phase41_v2_candidates,
    score_phase43_blind_pairs_deterministic,
    summarize_phase43_blind_results,
)
from pfe_core.trainer.executors import (
    _encode_sft_examples,
    _find_non_finite_trainer_metrics,
    _trainable_parameter_fingerprint,
)


def _source_candidates(count: int = 24) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "pair_id": f"phase41-v2-{index + 1:03d}",
                "scenario_id": f"scenario-{index + 1:03d}",
                "instruction": (
                    f"用户目标：核对项目状态 {index}\n"
                    "用户纠正：别猜，先检查证据\n"
                    f"继续要求：给出下一步动作 {index}\n"
                    f"本轮核验重点：状态证据 {index}"
                ),
                "chosen": (
                    f"我会先核对状态证据 {index}，再给结论。"
                    "只报告真实命令、路径或计数，缺证据就标 blocked；"
                    "该记录是 simulated_usage，不是 actual_user_feedback，也不允许自动 promote。"
                ),
                "rejected": f"已经基本完成 {index}，后续继续优化即可。",
                "feedback_source": "simulated_usage",
                "simulated_usage": True,
                "actual_model_call": False,
                "actual_product_benefit_claim_allowed": False,
            }
        )
    return rows


def _transcript(session: dict[str, object], *, variant: str, good: bool = True) -> dict[str, object]:
    expected = dict(session["expected"])
    latest = "、".join(expected["latest_terms"])
    action = "、".join(expected["action_terms"])
    if good:
        answers = [
            "当前证据不足，我会先检查真实状态。",
            f"收到最新纠正，只围绕 {latest} 核对，不能在没有输出时宣称完成。",
            f"当前判断为 blocked；依据是缺少真实输出。下一步先执行 {action}。",
        ]
    else:
        answers = ["可以。", "整体分析后已经基本完成。", "后续继续优化即可。"]
    return {
        "session_id": session["session_id"],
        "variant": variant,
        "model_id": "hidden-model",
        "adapter_path": "/hidden/adapter",
        "actual_model_call": True,
        "latency_seconds": [0.1, 0.1, 0.1],
        "turns": [
            {"role": "user", "content": session["user_goal"]},
            {"role": "assistant", "content": answers[0]},
            {"role": "user", "content": session["user_correction"]},
            {"role": "assistant", "content": answers[1]},
            {"role": "user", "content": session["continuation_request"]},
            {"role": "assistant", "content": answers[2]},
        ],
    }


def test_preference_taxonomy_is_personal_and_not_legal_four_section() -> None:
    taxonomy = build_phase43_preference_taxonomy()

    assert taxonomy["legal_four_section_format_required"] is False
    assert set(taxonomy["dimension_ids"]) == {
        "evidence_before_claim",
        "latest_intent",
        "useful_next_action",
        "evidence_provenance",
        "no_false_completion",
        "privacy_boundary",
        "concise_specific",
    }
    assert taxonomy["actual_product_benefit_claim_allowed"] is False


def test_candidate_review_repairs_boilerplate_and_keeps_simulated_boundary() -> None:
    holdout = build_phase43_holdout_sessions()["sessions"]
    manifest = review_phase41_v2_candidates(_source_candidates(), holdout_sessions=holdout)

    assert manifest["status"] == "approved_for_manual_training_probe"
    assert manifest["approved_count"] >= PHASE43_MIN_REVIEWED_PAIRS
    assert manifest["original_repair_required_count"] == 24
    assert manifest["selected_chosen_unique_ratio"] == 1.0
    for pair in manifest["selected_preference_pairs"]:
        assert "该记录是 simulated_usage" not in pair["chosen"]
        assert pair["simulated_manual_review"] is True
        assert pair["actual_user_feedback"] is False
        assert pair["actual_product_benefit_claim_allowed"] is False


def test_holdout_has_forty_multiturn_sessions_and_isolated_from_training() -> None:
    holdout = build_phase43_holdout_sessions()
    manifest = review_phase41_v2_candidates(_source_candidates(), holdout_sessions=holdout["sessions"])
    integrity = build_holdout_integrity_check(manifest["selected_preference_pairs"], holdout["sessions"])

    assert holdout["holdout_count"] == PHASE43_MIN_HOLDOUT_SESSIONS
    assert len(holdout["categories"]) == 10
    assert all(row["not_for_training"] is True for row in holdout["sessions"])
    assert integrity["passed"] is True
    assert integrity["exact_text_overlap_count"] == 0


def test_sft_job_and_tokenization_mask_prompt_tokens() -> None:
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            del tokenize
            rendered = "".join(f"<{row['role']}>{row['content']}" for row in messages)
            return rendered + ("<assistant>" if add_generation_prompt else "")

        def __call__(self, text, **kwargs):
            max_length = int(kwargs.get("max_length") or 1024)
            return {"input_ids": [ord(char) % 251 + 3 for char in str(text)[:max_length]]}

    pair = _source_candidates(1)[0]
    pair["chosen"] = "先检查真实状态，再给结论。"
    job = build_phase43_sft_job_spec(
        pairs=[pair],
        base_model="/models/Qwen3-4B",
        output_dir="/tmp/phase43-sft",
        max_steps=12,
    )
    encoded = _encode_sft_examples(
        tokenizer=_Tokenizer(),
        training_examples=job["training_examples"],
        max_length=256,
        vocab_size=300,
    )[0]
    prompt_text = _Tokenizer().apply_chat_template(
        [{"role": "user", "content": pair["instruction"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_length = len(_Tokenizer()(prompt_text, max_length=256)["input_ids"])

    assert job["phase43"]["completion_only_loss_required"] is True
    assert all(label == -100 for label in encoded["labels"][:prompt_length])
    assert any(label != -100 for label in encoded["labels"][prompt_length:])


def test_dpo_job_preserves_distinct_chosen_and_rejected_boundaries() -> None:
    reviewed = review_phase41_v2_candidates(_source_candidates())["selected_preference_pairs"]
    job = build_phase43_dpo_job_spec(
        pairs=reviewed,
        base_model="/models/Qwen3-4B",
        output_dir="/tmp/phase43-dpo",
        max_steps=12,
    )

    assert len(job["training_examples"]) == 24
    assert job["phase43"]["chosen_rejected_boundary_required"] is True
    assert all(row["instruction"] and row["chosen"] and row["rejected"] for row in job["training_examples"])
    assert all(row["chosen"] != row["rejected"] for row in job["training_examples"])


def test_real_parameter_fingerprint_and_safetensors_validation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    module = torch.nn.Linear(4, 4, bias=False)
    before = _trainable_parameter_fingerprint(module)
    with torch.no_grad():
        module.weight.add_(1.0)
    after = _trainable_parameter_fingerprint(module)
    artifact = tmp_path / "adapter_model.safetensors"
    save_file({"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": module.weight}, str(artifact))
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 4}),
        encoding="utf-8",
    )
    validation = validate_adapter_artifact(
        tmp_path,
        {"artifact_name": artifact.name, "artifact_format": "peft_lora"},
    )

    assert before["sha256"] != after["sha256"]
    assert validation["valid"] is True
    assert validation["tensor_count"] == 1


def test_dpo_non_finite_metrics_are_rejected_even_when_loss_is_zero() -> None:
    problems = _find_non_finite_trainer_metrics(
        [
            {"step": 1, "loss": 0.6, "grad_norm": 1.2},
            {"step": 2, "loss": 0.0, "grad_norm": "nan", "entropy": float("nan")},
        ]
    )

    assert {row["metric"] for row in problems} == {"grad_norm", "entropy"}


def test_blind_randomization_hides_identity_and_scores_candidate() -> None:
    sessions = build_phase43_holdout_sessions()["sessions"][:2]
    base = [_transcript(session, variant="base", good=False) for session in sessions]
    sft = [_transcript(session, variant="sft", good=True) for session in sessions]
    blind = build_phase43_blind_pairs({"base": base, "sft": sft}, sessions, seed=43)
    deterministic = score_phase43_blind_pairs_deterministic(blind)
    summary = summarize_phase43_blind_results(deterministic, blind["hidden_key"])

    assert blind["pair_count"] == 2
    public = json.dumps(blind["public_pairs"], ensure_ascii=False)
    assert "hidden-model" not in public
    assert "/hidden/adapter" not in public
    assert '"variant":' not in public
    assert summary["variants"]["sft"]["candidate_win_rate"] == 1.0


def test_eval_metrics_and_decision_gate_require_both_judges() -> None:
    sessions = build_phase43_holdout_sessions()["sessions"]
    sft_rows = [_transcript(session, variant="sft", good=True) for session in sessions]
    metrics = aggregate_phase43_variant(sft_rows, sessions)
    assert metrics["session_count"] == 40
    assert metrics["actual_model_calls"] is True

    base = {
        "user_preference_score": 0.5,
        "correction_responsiveness_rate": 0.5,
        "generic_answer_rate": 0.3,
        "false_completion_rate": 0.0,
        "response_diversity": 0.8,
    }
    candidate = {
        "actual_model_calls": True,
        "session_count": 40,
        "user_preference_score": 0.7,
        "correction_responsiveness_rate": 0.7,
        "generic_answer_rate": 0.1,
        "false_completion_rate": 0.0,
        "privacy_violation_rate": 0.0,
        "training_leakage_rate": 0.0,
        "response_diversity": 0.9,
    }
    blind = {"variants": {"sft": {"candidate_win_rate": 0.7}}}
    independent = {"status": "completed", **blind}
    decision = build_phase43_decision(
        base_metrics=base,
        candidate_metrics={"sft": candidate},
        deterministic_blind=blind,
        independent_blind=independent,
        training_status={"sft": "completed"},
    )

    assert decision["status"] == "ready_for_manual_acceptance_trial"
    assert decision["actual_user_benefit_claim_allowed"] is False
    assert decision["auto_promotion_allowed"] is False

    blocked = build_phase43_decision(
        base_metrics=base,
        candidate_metrics={"sft": candidate},
        deterministic_blind=blind,
        independent_blind={"status": "blocked", "variants": {}},
        training_status={"sft": "completed"},
    )
    assert blocked["status"] == "archive"
    assert "independent_blind_win_rate_at_least_0_60" in blocked["candidate_decisions"]["sft"]["failed_checks"]


def test_phase42_regression_and_phase43_never_overclaim_actual_benefit() -> None:
    phase42 = build_phase42_final_decision(
        adapter_report={"passed": False, "version": "006"},
        lifecycle_decision={"action": "archived", "artifact_retained": True},
        training_attempt={
            "real_training": True,
            "adapter_validation": {"valid": True},
            "execution": {"parameters_updated": True},
        },
        context_smoke={"passed": True},
        hermes_streaming_passed=True,
        security_tests_passed=True,
        full_validation_passed=True,
        phase41_current={"training_candidate_status": "blocked"},
        phase41_v2={"status": "quality_ready_for_future_manual_review", "candidate_quality": {"passed": True}},
    )

    assert phase42["reliability_gate_passed"] is True
    assert phase42["actual_product_benefit_claim_allowed"] is False
