from __future__ import annotations

import copy

from pfe_core.phase77_private_value_guarded_runtime import (
    PHASE77_PRIVATE_PLACEHOLDER,
    build_phase77_holdout,
)
from pfe_core.phase78_persona_internalization_training import (
    PHASE78_PERSONA_CATEGORIES,
    audit_phase78_isolation,
    audit_phase78_public_private_values,
    audit_phase78_training_samples,
    build_phase78_blind_pairs,
    build_phase78_decision,
    build_phase78_holdout,
    build_phase78_sft_job_spec,
    build_phase78_training_samples,
    score_phase78_blind_pairs_deterministic,
    summarize_phase78_blind_results,
)
from pfe_core.trainer.executors import (
    _build_sft_prompt_and_text,
    _encode_sft_examples,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del tokenize
        rendered = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        maximum = int(kwargs.get("max_length") or 1024)
        return {"input_ids": [ord(char) % 251 + 3 for char in str(text)[:maximum]]}


def _transcript(session: dict[str, object], variant: str, answer: str) -> dict[str, object]:
    turns = []
    for prompt in (
        session["user_goal"],
        session["user_correction"],
        session["continuation_request"],
    ):
        turns.extend(
            (
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": answer},
            )
        )
    return {
        "session_id": session["session_id"],
        "variant": variant,
        "status": "completed",
        "actual_model_call": True,
        "privacy_canary_echo_detected": False,
        "turns": turns,
    }


def _metrics(target: float, ordinary: float = 0.9) -> dict[str, object]:
    categories = {
        name: {
            "composite_personalization_score": target,
            "hard_gate_pass_rate": 1.0,
        }
        for name in PHASE78_PERSONA_CATEGORIES
    }
    categories["ordinary_direct"] = {
        "composite_personalization_score": ordinary,
        "hard_gate_pass_rate": 1.0,
    }
    return {
        "actual_model_calls": True,
        "session_count": 48,
        "category_metrics": categories,
        "privacy_canary_echo_rate": 0.0,
        "unsupported_claim_rate": 0.0,
    }


def _judge_summary(base_win: float = 0.7, runtime_win: float = 0.4) -> dict[str, object]:
    return {
        "status": "completed",
        "actual_model_calls": True,
        "completed_pair_count": 96,
        "failure_count": 0,
        "invalid_result_count": 0,
        "comparisons": {
            "adapter_vs_base": {
                "slices": {
                    "persona_target": {
                        "candidate_win_rate": base_win,
                        "tie_rate": 0.1,
                    }
                }
            },
            "adapter_vs_runtime": {
                "slices": {
                    "persona_target": {
                        "candidate_win_rate": runtime_win,
                        "tie_rate": 0.3,
                    }
                }
            },
        }
    }


def _training_attempt() -> dict[str, object]:
    return {
        "status": "completed",
        "real_training": True,
        "requested_steps": 120,
        "adapter_validation": {"valid": True},
        "execution": {"parameters_updated": True},
        "exposure": {"full_coverage": True},
    }


def test_phase78_training_set_is_balanced_private_safe_and_simulated() -> None:
    samples = build_phase78_training_samples()
    audit = audit_phase78_training_samples(samples)

    assert len(samples) == 120
    assert audit["passed"] is True
    assert audit["category_counts"]["ordinary_direct"] == 24
    assert all(audit["category_counts"][name] == 16 for name in PHASE78_PERSONA_CATEGORIES)
    assert all(row["feedback_source"] == "simulated_usage" for row in samples)
    assert all(row["actual_user_feedback"] is False for row in samples)
    assert all("SYNTHETIC_PHASE" not in str(row) for row in samples)
    privacy = [row for row in samples if row["taxonomy_dimension"] == "privacy_non_echo"]
    assert len(privacy) == 16
    assert all(PHASE77_PRIVATE_PLACEHOLDER in str(row["messages"]) for row in privacy)
    assert all(PHASE77_PRIVATE_PLACEHOLDER not in row["chosen"] for row in privacy)


def test_phase78_holdout_is_fresh_and_never_training_data() -> None:
    samples = build_phase78_training_samples()
    holdout = build_phase78_holdout()
    previous = build_phase77_holdout()
    audit = audit_phase78_isolation(samples, holdout["sessions"], previous["sessions"])

    assert holdout["session_count"] == 48
    assert holdout["persona_target_count"] == 36
    assert holdout["ordinary_control_count"] == 12
    assert holdout["privacy_target_count"] == 6
    assert audit["passed"] is True
    assert audit["training_text_overlap"] == []
    assert audit["phase77_text_overlap"] == []
    assert all(row["not_for_training"] is True for row in holdout["sessions"])


def test_phase78_job_spec_masks_every_prompt_turn_and_trains_only_completion() -> None:
    samples = build_phase78_training_samples()
    spec = build_phase78_sft_job_spec(
        samples=samples,
        base_model="models/Qwen3-4B",
        output_dir="trainer_job_outputs/phase78-test",
        max_steps=120,
    )
    source = samples[0]
    tokenizer = _Tokenizer()
    prompt, _ = _build_sft_prompt_and_text(
        tokenizer,
        source["instruction"],
        source["chosen"],
        messages=source["messages"],
    )
    encoded = _encode_sft_examples(
        tokenizer=tokenizer,
        training_examples=[source],
        max_length=512,
        vocab_size=300,
    )[0]
    prompt_length = len(tokenizer(prompt, max_length=512)["input_ids"])

    assert spec["ready"] is True
    assert spec["recipe"]["training"]["train_type"] == "native_multiturn_sft_completion_only"
    assert spec["recipe"]["training"]["sampling_strategy"] == "seeded_stratified"
    assert all(label == -100 for label in encoded["labels"][:prompt_length])
    assert any(label != -100 for label in encoded["labels"][prompt_length:])


def test_phase78_blind_pairs_hide_identity_and_redact_both_comparisons() -> None:
    sessions = build_phase78_holdout()["sessions"]
    transcripts = {
        "base_minimal_guarded": [
            _transcript(row, "base_minimal_guarded", "已经完成。") for row in sessions
        ],
        "adapter_minimal_guarded": [
            _transcript(
                row,
                "adapter_minimal_guarded",
                "结论：尚未验证。\n依据：证据不足。\n下一步：继续检查。",
            )
            for row in sessions
        ],
        "runtime_reference": [
            _transcript(
                row,
                "runtime_reference",
                "结论：尚未验证。\n依据：证据不足。\n下一步：继续检查。",
            )
            for row in sessions
        ],
    }
    blind = build_phase78_blind_pairs(transcripts, sessions)
    public = str(blind["public_pairs"])
    audit = audit_phase78_public_private_values(blind["public_pairs"], sessions)
    results = score_phase78_blind_pairs_deterministic(blind, sessions)
    summary = summarize_phase78_blind_results(
        results,
        blind["hidden_key"],
        blind["public_pairs"],
    )

    assert blind["pair_count"] == 96
    assert "base_minimal_guarded" not in public
    assert "adapter_minimal_guarded" not in public
    assert "runtime_reference" not in public
    assert audit["passed"] is True
    assert audit["raw_private_value_pair_count"] == 0
    assert audit["redaction_marker_pair_count"] == 12
    assert summary["invalid_result_count"] == 0
    assert summary["comparisons"]["adapter_vs_base"]["slices"]["persona_target"]["pair_count"] == 36
    assert summary["comparisons"]["adapter_vs_runtime"]["slices"]["ordinary_control"]["pair_count"] == 12


def test_phase78_decision_requires_real_adapter_benefit_not_training_success() -> None:
    metrics = {
        "base_minimal_guarded": _metrics(0.62),
        "adapter_minimal_guarded": _metrics(0.74),
        "runtime_reference": _metrics(0.76),
    }
    judges = {
        "gemma4:31b": _judge_summary(),
        "qwen3.6": _judge_summary(),
    }
    decision = build_phase78_decision(
        metrics=metrics,
        training_attempt=_training_attempt(),
        quality_audit={"passed": True},
        isolation_audit={"passed": True},
        completion_boundary={"passed": True},
        public_private_audit={"passed": True},
        deterministic=_judge_summary(),
        independent=judges,
    )

    assert decision["status"] == "qualified_simulated_persona_adapter"
    assert decision["recommendation"] == "manual_review_then_actual_usage_pilot"
    assert decision["promotion_allowed"] is False
    assert decision["actual_product_benefit_claim_allowed"] is False

    no_benefit = copy.deepcopy(metrics)
    no_benefit["adapter_minimal_guarded"] = _metrics(0.63)
    archived = build_phase78_decision(
        metrics=no_benefit,
        training_attempt=_training_attempt(),
        quality_audit={"passed": True},
        isolation_audit={"passed": True},
        completion_boundary={"passed": True},
        public_private_audit={"passed": True},
        deterministic=_judge_summary(),
        independent=judges,
    )
    assert archived["status"] == "archive"
    assert "adapter_target_gain_at_least_0_08" in archived["failed_checks"]
    assert archived["simulated_lab_benefit_claim_allowed"] is False
