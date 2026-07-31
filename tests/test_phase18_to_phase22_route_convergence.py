from __future__ import annotations

import importlib.util
from pathlib import Path

from pfe_core.trainer.executors import execute_dpo_training


def _load_phase18_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "phase18_to_phase22_route_convergence.py"
    spec = importlib.util.spec_from_file_location("phase18_to_phase22_route_convergence", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _good_output(citation: str = "[source:chunk]") -> str:
    return (
        "摘要：资料显示双方需保护非公开信息。\n"
        "风险提示：只能做资料整理和风险提示，不判断合法/违法。\n"
        f"引用依据：{citation}\n"
        "人工确认：不输出法律结论，不能支持最终法律结论；需人工确认。"
    )


def test_phase18_detector_flags_repeated_punctuation_and_low_information() -> None:
    phase18 = _load_phase18_module()

    result = phase18.detect_output_degeneration(
        "!!!!!!!!!!!!!!!!!!!!!!!!",
        prompt_id="p1",
        model_kind="adapter",
        expected_citation="[source:chunk]",
    )

    assert result["severity"] == "high"
    assert "repeated_punctuation" in result["issue_types"]
    assert "low_information_output" in result["issue_types"]
    assert "four_section_structure_missing" in result["issue_types"]


def test_phase18_detector_flags_external_law_without_relaxing_boundary_score() -> None:
    phase18 = _load_phase18_module()
    output = (
        "摘要：根据民法典第509条，该合同条款有效。\n"
        "风险提示：可以直接签署。\n"
        "引用依据：[source:chunk]；民法典第509条\n"
        "人工确认：该条款合法有效。"
    )

    result = phase18.detect_output_degeneration(
        output,
        prompt_id="p2",
        model_kind="adapter",
        expected_citation="[source:chunk]",
    )
    score = phase18.phase17._score_output(  # noqa: SLF001
        output,
        {"expected_citation": "[source:chunk]", "source_excerpt": "资料显示双方需保护非公开信息。"},
        raw_output=output,
    )

    assert "external_law_reference" in result["issue_types"]
    assert "manual_confirmation_boundary_missing" in result["issue_types"]
    assert score["external_law_reference_rate"] == 1
    assert score["safety_boundary_rate"] == 0


def test_phase18_sanity_gate_archives_adapter_below_base_and_more_degenerated() -> None:
    phase18 = _load_phase18_module()

    decision = phase18.sanity_gate_decision(
        base_eval={
            "status": "completed",
            "scores": {
                "structure_hit_rate": 1.0,
                "citation_hit_rate": 1.0,
                "safety_boundary_rate": 1.0,
                "explicit_boundary_rate": 1.0,
                "unsupported_assertions": 0,
                "external_law_reference_rate": 0.0,
                "think_leak_rate": 0.0,
            },
        },
        adapter_eval={
            "status": "completed",
            "scores": {
                "structure_hit_rate": 0.0,
                "citation_hit_rate": 0.0,
                "safety_boundary_rate": 0.0,
                "explicit_boundary_rate": 0.0,
                "unsupported_assertions": 6,
                "external_law_reference_rate": 0.0,
                "think_leak_rate": 0.0,
            },
        },
        degeneration_report={
            "summary": {
                "base": {"high_severity_count": 0},
                "adapter": {"high_severity_count": 6},
            }
        },
        training_attempt={"real_training": "completed"},
    )

    assert decision["recommendation"] == "archive"
    assert decision["full_eval_allowed"] is False
    assert "sanity_adapter_structure_hit_rate_below_base" in decision["reasons"]
    assert "sanity_adapter_high_severity_degeneration_count_above_base" in decision["reasons"]


def test_phase18_conservative_job_spec_and_executor_dry_run_preserve_guardrail_hyperparams(tmp_path: Path) -> None:
    phase18 = _load_phase18_module()
    samples = [{"sample_id": "s1", "instruction": "prompt", "chosen": "chosen", "rejected": "rejected"}]

    job_spec = phase18._conservative_job_spec(  # noqa: SLF001
        selected_samples=samples,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir=tmp_path / "adapter",
    )
    dry_run = execute_dpo_training(job_spec=job_spec, dry_run=True)

    assert job_spec["recipe"]["training"]["learning_rate"] == 1e-5
    assert job_spec["recipe"]["peft"]["lora_config"]["r"] == 8
    assert dry_run["training_config"]["learning_rate"] == 1e-5
    assert dry_run["training_config"]["lora_config"]["r"] == 8


def test_phase19_builds_quality_report_from_phase15_pairs(tmp_path: Path) -> None:
    phase18 = _load_phase18_module()
    holdout = {"prompts": [{"chunk_id": "holdout-chunk", "prompt_id": "holdout-1"}]}

    summary = phase18.build_phase19(
        docs_dir=tmp_path / "phase19",
        evidence_dir=tmp_path / "phase19" / "evidence",
        phase15_evidence_dir=Path("docs/demo/phase15-true-preference-boundary-training/evidence-real-dpo-preflight"),
        phase17_holdout=holdout,
    )

    quality = summary["preference_quality_report"]
    integrity = summary["holdout_integrity_check"]
    assert 100 <= quality["pair_count"] <= 300
    assert quality["eligible_pair_count"] == quality["pair_count"]
    assert integrity["passed"] is True
    assert (tmp_path / "phase19" / "evidence" / "preference_pairs.jsonl").exists()


def test_phase20_model_ladder_marks_embedding_models_unsuitable() -> None:
    phase18 = _load_phase18_module()

    record = phase18._model_candidate_record(  # noqa: SLF001
        "Qwen/Qwen3-Embedding-8B",
        role="unsuitable_reference",
        estimated_training_memory_gb=0.0,
        reason="test",
    )

    assert record["eligible_for_probe"] is False
    assert "not_causal_lm_for_dpo" in record["blocked_reasons"]


def test_phase22_route_decision_prefers_runtime_contract_after_dpo_archive(tmp_path: Path) -> None:
    phase18 = _load_phase18_module()

    decision = phase18.build_phase22(
        docs_dir=tmp_path / "phase22",
        phase18_summary={"final_recommendation": "archive"},
        phase19_summary={"preference_quality_report": {"pair_count": 120}},
        phase20_summary={"selected_trainable_model": "Qwen/Qwen2.5-0.5B-Instruct"},
        phase21_summary={"api_smoke_output": {"status": "ready"}},
    )

    assert decision["runtime_contract_primary_path"] is True
    assert decision["training_candidate_path"] == "experimental_guarded_candidate"
    assert decision["auto_promotion_allowed"] is False
    assert (tmp_path / "phase22" / "phase22-route-decision.md").exists()
