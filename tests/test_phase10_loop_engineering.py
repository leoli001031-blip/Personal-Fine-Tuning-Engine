from __future__ import annotations

import json
from pathlib import Path

from pfe_core.db.sqlite import list_samples
from pfe_core.phase10_loop_engineering import (
    PHASE10_DEFAULT_DATASET_RECIPE,
    PHASE10_EXPECTED_SECTIONS,
    PHASE10_RECOMMENDED_MODEL,
    PHASE10_STAGE_A,
    Phase10LoopEngineeringStore,
    finalize_phase10_loop_experiment,
    normalize_phase10_output,
    prepare_phase10_loop_experiment,
)
from pfe_core.trainer.mlx_backend import MLXTrainerBackend


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_phase9_evidence(tmp_path: Path) -> tuple[Path, Path]:
    eval_path = tmp_path / "phase9-eval.json"
    train_path = tmp_path / "training_job_result.json"
    eval_path.write_text(
        json.dumps(
            {
                "scores": {
                    "base": {
                        "citation_hit_rate": 0.7,
                        "structure_hit_rate": 0.725,
                        "unsupported_assertions": 13,
                        "safety_boundary_rate": 0.0,
                    },
                    "adapter": {
                        "citation_hit_rate": 0.3,
                        "structure_hit_rate": 0.325,
                        "unsupported_assertions": 17,
                        "safety_boundary_rate": 0.0,
                    },
                    "delta": {
                        "citation_hit_rate": -0.4,
                        "structure_hit_rate": -0.4,
                        "unsupported_assertions": -4,
                        "safety_boundary_rate": 0.0,
                    },
                },
                "details": [
                    {
                        "prompt_id": "phase9-holdout-001",
                        "base_output": "摘要：base partial",
                        "adapter_output": "答案：copied\n摘要：partial",
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    train_path.write_text(
        json.dumps(
            {
                "result": {
                    "result": {
                        "num_steps": 12,
                        "num_samples": 51,
                        "metadata": {
                            "dataset_format": "prompt_completion_output_only_loss",
                            "output_only_loss_masking": True,
                        },
                    }
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return eval_path, train_path


def test_phase10_prepare_builds_format_curriculum_samples(tmp_path: Path) -> None:
    prepared = prepare_phase10_loop_experiment(
        home=tmp_path,
        workspace="demo",
        signal_count=36,
        candidate_limit=34,
        holdout_count=10,
        require_local_model=True,
        model_path=tmp_path / "missing-qwen",
    )
    samples = _read_jsonl(prepared["paths"]["candidate_samples"])
    db_samples = list_samples(home=tmp_path)
    quality = prepared["quality_report"]

    assert prepared["ok"] is True
    assert prepared["experiment_id"].startswith("p10exp-")
    assert prepared["manifest"]["kind"] == "phase10_loop_experiment"
    assert prepared["manifest"]["stage"] == PHASE10_STAGE_A
    assert prepared["manifest"]["dataset_recipe"] == PHASE10_DEFAULT_DATASET_RECIPE
    assert prepared["source_manifest"]["source_mode"] == "format_curriculum_no_external_fetch"
    assert prepared["signal_dataset"]["quality_signal_count"] == 36
    assert prepared["signal_dataset"]["eligible_count"] == 36
    assert prepared["candidate_samples"]["count"] >= 30
    assert prepared["candidate_samples"]["split_counts"]["test"] == 0
    assert quality["passed_signal_count"] >= 30
    assert quality["candidate_passed_count"] >= 30
    assert quality["meets_quality_goal"] is True
    assert quality["rejection_reasons"]["reject_excluded"] == 1
    assert quality["rejection_reasons"]["safety_block_excluded"] == 1
    assert prepared["preflight"]["model_id"] == PHASE10_RECOMMENDED_MODEL
    assert "local_model_missing" in prepared["preflight"]["blocked_by"]
    assert all("phase10-holdout" not in sample["sample_id"] for sample in db_samples)

    holdout_chunk_ids = set(prepared["quality_report"]["holdout_chunk_ids"])
    for sample in samples[:10]:
        metadata = sample["metadata"]
        chosen = str(sample["chosen"])
        lines = [line for line in chosen.splitlines() if line.strip()]
        assert metadata["phase"] == "phase10"
        assert metadata["experiment_id"] == prepared["experiment_id"]
        assert metadata["quality_gate_passed"] is True
        assert metadata["not_holdout"] is True
        assert metadata["training_format"] == PHASE10_DEFAULT_DATASET_RECIPE
        assert metadata["completion_marker"] == "### 标准答案"
        assert "### 标准答案" in str(sample["instruction"])
        assert not (set(metadata["chunk_ids"]) & holdout_chunk_ids)
        assert metadata["expected_citation"] in chosen
        assert len(lines) == 4
        assert len(chosen) < 360
        for section, line in zip(PHASE10_EXPECTED_SECTIONS, lines, strict=True):
            assert line.startswith(f"{section}：")
        for copied_term in ("资料片段：", "请现在输出答案", "### 标准答案", "答案："):
            assert copied_term not in chosen
        assert not any(line.startswith(("1.", "1、", "-", "*", "#")) for line in lines)


def test_phase10_normalizes_first_complete_block_and_preserves_raw() -> None:
    raw = (
        "先解释一下。\n"
        "摘要：只整理片段。\n"
        "风险提示：不判断合法/违法，不输出法律结论。\n"
        "引用依据：[src:chunk]\n"
        "人工确认：最终判断必须人工确认。\n"
        "额外：这一行应该被截断。"
    )
    normalized = normalize_phase10_output(raw)

    assert normalized["raw_output"] == raw
    assert normalized["complete"] is True
    assert normalized["truncated"] is True
    assert "truncated_after_first_complete_block" in normalized["truncation_reasons"]
    assert normalized["normalized_output"].splitlines() == [
        "摘要：只整理片段。",
        "风险提示：不判断合法/违法，不输出法律结论。",
        "引用依据：[src:chunk]",
        "人工确认：最终判断必须人工确认。",
    ]


def test_phase10_normalization_does_not_synthesize_missing_sections() -> None:
    normalized = normalize_phase10_output("摘要：只有摘要。\n引用依据：[src:chunk]")

    assert normalized["complete"] is False
    assert "风险提示：" not in normalized["normalized_output"]
    assert "人工确认：" not in normalized["normalized_output"]
    assert any(str(reason).startswith("incomplete_four_section_block") for reason in normalized["truncation_reasons"])


def test_phase10_quality_gate_rejects_holdout_and_loose_targets(tmp_path: Path) -> None:
    prepared = prepare_phase10_loop_experiment(
        home=tmp_path,
        workspace="demo",
        signal_count=30,
        candidate_limit=30,
        holdout_count=6,
    )
    store = Phase10LoopEngineeringStore(home=tmp_path, workspace="demo")
    holdout = store.read_holdouts()[0]
    bad_signal = {
        "signal_id": "bad-holdout",
        "signal_type": "correction",
        "signal_strength": "strong_correction",
        "source_id": holdout["source_id"],
        "chunk_id": holdout["chunk_id"],
        "expected_citation": f"[{holdout['source_id']}:wrong-chunk]",
        "target_output": "1. 摘要：太短。\n引用依据：[missing]",
        "eligible_for_training": True,
        "quality_score": 0.9,
    }
    check = store._evaluate_signal(  # noqa: SLF001 - targeted quality gate unit test.
        signal=bad_signal,
        holdout_chunk_ids=set(prepared["quality_report"]["holdout_chunk_ids"]),
        seen_hashes=set(),
    )

    assert check["passed"] is False
    assert "holdout_contamination" in check["reasons"]
    assert "citation_does_not_match_source_chunk" in check["reasons"]
    assert "target_missing_expected_citation" in check["reasons"]
    assert "numbering_or_markdown" in check["reasons"]
    assert "not_exactly_four_lines" in check["reasons"]


def test_phase10_mlx_formatter_preserves_prompt_completion_boundary(tmp_path: Path) -> None:
    backend = MLXTrainerBackend.__new__(MLXTrainerBackend)
    data_file = backend.prepare_training_data(
        [
            {
                "instruction": "只输出四段式。\n\n### 标准答案\n",
                "chosen": "摘要：短摘要。\n风险提示：不输出法律结论。\n引用依据：[src:chunk]\n人工确认：必须人工确认。",
            }
        ],
        tmp_path,
    )
    row = json.loads(Path(data_file).read_text(encoding="utf-8").strip())

    assert set(row) == {"prompt", "completion"}
    assert row["prompt"].endswith("### 标准答案\n")
    assert row["completion"].startswith("摘要：")
    assert "### 标准答案" not in row["completion"]


def test_phase10_eval_keeps_raw_normalized_and_manual_review_decision(tmp_path: Path) -> None:
    phase9_eval, phase9_train = _write_phase9_evidence(tmp_path)
    prepare_phase10_loop_experiment(
        home=tmp_path,
        workspace="demo",
        signal_count=30,
        candidate_limit=30,
        holdout_count=6,
    )
    store = Phase10LoopEngineeringStore(home=tmp_path, workspace="demo")
    details = []
    for holdout in store.read_holdouts():
        citation = str(holdout["expected_citation"])
        details.append(
            {
                "prompt_id": holdout["prompt_id"],
                "base_output": "This looks risky. Ask a lawyer.",
                "adapter_output": (
                    "说明：下面才是答案。\n"
                    "摘要：仅基于给定片段整理条款内容。\n"
                    "风险提示：关注资料中出现的义务、限制和证据不足，不判断合法/违法，不输出法律结论。\n"
                    f"引用依据：{citation}\n"
                    "人工确认：适用法、真实交易背景和最终法律判断必须人工确认。\n"
                    "额外：这行不计入评分。"
                ),
            }
        )
    finalized = finalize_phase10_loop_experiment(
        home=tmp_path,
        workspace="demo",
        training={"real_training": "completed", "mock_fallback": False, "adapter_path": str(tmp_path / "adapter")},
        generations={"real_model_calls": True, "details": details},
        real_model_calls=True,
        phase9_eval_path=phase9_eval,
        phase9_training_job_path=phase9_train,
    )
    first = finalized["eval_report"]["details"][0]

    assert finalized["eval_report"]["scoring_basis"] == "normalized_output_with_raw_preserved"
    assert "说明：下面才是答案" in first["adapter_raw_output"]
    assert "说明：下面才是答案" not in first["adapter_output"]
    assert "额外：这行不计入评分" not in first["adapter_output"]
    assert "truncated_after_first_complete_block" in first["adapter_normalization"]["truncation_reasons"]
    assert finalized["eval_report"]["eval_gate"]["status"] == "pass"
    assert finalized["eval_report"]["recommendation"] == "promote_after_manual_review"
    assert finalized["decision"]["action"] == "promote_after_manual_review"
    assert finalized["decision"]["promotion_allowed"] is False
    assert finalized["decision"]["manual_review_required"] is True
    assert finalized["phase9_retrospective"]["training_metadata"]["output_only_loss_masking"] is True
    assert finalized["qwen36_preflight_decision"]["status"] == "not_requested"
    assert Path(finalized["paths"]["output_examples"]).is_file()
    assert Path(finalized["paths"]["comparison_summary"]).is_file()


def test_phase10_finalize_archives_without_real_training_and_blocks_qwen36(tmp_path: Path) -> None:
    phase9_eval, phase9_train = _write_phase9_evidence(tmp_path)
    prepare_phase10_loop_experiment(
        home=tmp_path,
        workspace="demo",
        signal_count=30,
        candidate_limit=30,
        holdout_count=6,
    )
    finalized = finalize_phase10_loop_experiment(
        home=tmp_path,
        workspace="demo",
        training={"real_training": "not_started", "mock_fallback": False},
        generations={"real_model_calls": False},
        real_model_calls=False,
        phase9_eval_path=phase9_eval,
        phase9_training_job_path=phase9_train,
        qwen36_preflight={"ready_for_real_training": True},
    )

    assert finalized["training_result"]["status"] == "created"
    assert finalized["eval_report"]["eval_gate"]["status"] == "blocked"
    assert finalized["eval_report"]["recommendation"] == "archive"
    assert finalized["decision"]["action"] == "archive"
    assert finalized["qwen36_preflight_decision"]["status"] == "skipped"
    assert finalized["qwen36_preflight_decision"]["next_action"] == "do_not_load_qwen36_until_small_model_gate_passes"
    assert Path(finalized["paths"]["summary"]).is_file()
