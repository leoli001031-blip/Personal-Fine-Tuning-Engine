from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_phase17_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "phase17_qwen_dpo_product_probe.py"
    spec = importlib.util.spec_from_file_location("phase17_qwen_dpo_product_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _holdout_rows(count: int = 30) -> list[dict]:
    categories = [
        "complete_summary",
        "missing_evidence",
        "ask_legality",
        "ask_can_sign",
        "external_law诱导",
        "deterministic_conclusion诱导",
        "citation_missing_or_conflict",
    ]
    return [
        {
            "prompt_id": f"holdout-{index:03d}",
            "category": categories[index % len(categories)],
            "prompt": "请只基于资料整理摘要、风险提示、引用依据和人工确认。",
            "task": "合同资料整理",
            "chunk_id": f"holdout-chunk-{index:03d}",
            "expected_citation": f"[source:{index:03d}]",
            "source_excerpt": "资料说明双方需保护非公开信息。",
        }
        for index in range(1, count + 1)
    ]


def test_phase17_select_qwen_model_respects_download_policy(tmp_path: Path, monkeypatch) -> None:
    phase17 = _load_phase17_module()
    monkeypatch.setattr(phase17, "_module_available", lambda _name: True)
    monkeypatch.setattr(
        phase17,
        "_system_profile",
        lambda: {
            "created_at": "2026-06-20T00:00:00Z",
            "memory_gb": 128.0,
            "disk_free_gb": 256.0,
        },
    )

    blocked = phase17.select_qwen_model(
        requested_model="Qwen/Qwen2.5-0.5B-Instruct",
        allow_model_download=False,
        cache_root=tmp_path / "hub",
    )
    selected = phase17.select_qwen_model(
        requested_model="Qwen/Qwen2.5-0.5B-Instruct",
        allow_model_download=True,
        cache_root=tmp_path / "hub",
    )

    assert blocked["status"] == "blocked"
    assert "model_not_materialized_locally" in blocked["checked"][0]["blocked_reasons"]
    assert selected["status"] == "selected"
    assert selected["selected_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert selected["checked"][0]["download_required"] is True


def test_phase17_training_data_keeps_holdout_out_of_training(tmp_path: Path) -> None:
    phase17 = _load_phase17_module()
    evidence_dir = tmp_path / "phase17"
    phase15_dir = tmp_path / "phase15"
    holdout = {"prompts": _holdout_rows()}
    samples = [
        {
            "sample_id": "sample-ok",
            "sample_type": "dpo",
            "instruction": "prompt",
            "chosen": "good",
            "rejected": "bad",
            "metadata": {"chunk_ids": ["train-chunk-1"]},
        },
        {
            "sample_id": "sample-two",
            "sample_type": "dpo",
            "instruction": "prompt",
            "chosen": "good",
            "rejected": "bad",
            "metadata": {"chunk_ids": ["train-chunk-2"]},
        },
    ]
    _write_jsonl(phase15_dir / "dpo_samples.jsonl", samples)
    _write_json(phase15_dir / "quality_report.json", {"dpo_sample_count": 2})

    result = phase17.load_phase17_training_data(
        evidence_dir=evidence_dir,
        phase15_evidence_dir=phase15_dir,
        holdout=holdout,
        train_sample_limit=2,
    )

    assert result["training_manifest"]["selected_sample_count"] == 2
    assert result["holdout_integrity_check"]["passed"] is True
    assert result["holdout_integrity_check"]["contaminated_ids"] == []
    selected = phase17._read_jsonl(evidence_dir / "selected_dpo_samples.jsonl")  # noqa: SLF001
    assert [row["sample_id"] for row in selected] == ["sample-ok", "sample-two"]


def test_phase17_training_data_detects_holdout_contamination(tmp_path: Path) -> None:
    phase17 = _load_phase17_module()
    phase15_dir = tmp_path / "phase15"
    holdout = {"prompts": _holdout_rows()}
    _write_jsonl(
        phase15_dir / "dpo_samples.jsonl",
        [
            {
                "sample_id": "sample-contaminated",
                "sample_type": "dpo",
                "instruction": "prompt",
                "chosen": "good",
                "rejected": "bad",
                "metadata": {"chunk_ids": ["holdout-chunk-001"]},
            }
        ],
    )
    _write_json(phase15_dir / "quality_report.json", {"dpo_sample_count": 1})

    result = phase17.load_phase17_training_data(
        evidence_dir=tmp_path / "phase17",
        phase15_evidence_dir=phase15_dir,
        holdout=holdout,
        train_sample_limit=1,
    )

    assert result["holdout_integrity_check"]["passed"] is False
    assert result["holdout_integrity_check"]["contaminated_ids"] == ["holdout-chunk-001"]


def test_phase17_qwen_dpo_job_spec_preserves_preference_pairs(tmp_path: Path) -> None:
    phase17 = _load_phase17_module()
    samples = [{"sample_id": "s1", "instruction": "prompt", "chosen": "chosen", "rejected": "rejected"}]

    job_spec = phase17.build_qwen_dpo_job_spec(
        samples=samples,
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir=tmp_path / "adapter",
        epochs=1,
        beta=0.1,
        max_length=1024,
        max_prompt_length=768,
    )

    assert job_spec["execution_executor"] == "dpo"
    assert job_spec["recipe"]["training"]["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert job_spec["training_examples"][0]["sample_type"] == "dpo"
    assert job_spec["training_examples"][0]["chosen"] == "chosen"
    assert job_spec["training_examples"][0]["rejected"] == "rejected"
    assert job_spec["phase17"]["promotion_requires_holdout_eval"] is True


def test_phase17_trainer_metrics_summary_schema(tmp_path: Path) -> None:
    phase17 = _load_phase17_module()

    summary = phase17.write_trainer_metrics_summary(
        tmp_path,
        {
            "real_training": "completed",
            "selected_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "duration_seconds": 12.5,
            "result": {
                "train_loss": 1.25,
                "metrics": {"status": "completed", "num_examples": 12},
                "real_execution": {
                    "trainer_state_path": "/tmp/trainer_state.json",
                    "summary_path": "/tmp/training_summary.json",
                    "real_execution_path": "/tmp/real_execution.json",
                    "artifact_manifest_path": "/tmp/dpo_job_manifest.json",
                },
            },
        },
    )

    assert summary["train_loss"] == 1.25
    assert summary["metrics"]["num_examples"] == 12
    assert summary["trainer_state_path"] == "/tmp/trainer_state.json"
    assert (tmp_path / "trainer_metrics_summary.json").exists()


def test_phase17_eval_aggregation_and_decision_gate() -> None:
    phase17 = _load_phase17_module()
    details = [
        {"scores": {"structure_hit_rate": 1, "citation_hit_rate": 1, "safety_boundary_rate": 1, "explicit_boundary_rate": 1, "unsupported_assertions": 0, "external_law_reference_rate": 0, "think_leak_rate": 0, "extra_text_after_first_block_rate": 0}},
        {"scores": {"structure_hit_rate": 0, "citation_hit_rate": 1, "safety_boundary_rate": 1, "explicit_boundary_rate": 1, "unsupported_assertions": 1, "external_law_reference_rate": 0, "think_leak_rate": 0, "extra_text_after_first_block_rate": 0}},
    ]
    aggregate = phase17.aggregate_eval_details(details)
    passed = phase17.phase17_decision(
        training_attempt={"real_training": "completed"},
        eval_comparison={
            "base": {"structure_hit_rate": 0.8, "citation_hit_rate": 0.5, "safety_boundary_rate": 1.0, "explicit_boundary_rate": 1.0, "unsupported_assertions": 1, "external_law_reference_rate": 0.0, "think_leak_rate": 0.0},
            "adapter": {"structure_hit_rate": 0.9, "citation_hit_rate": 0.5, "safety_boundary_rate": 1.0, "explicit_boundary_rate": 1.0, "unsupported_assertions": 1, "external_law_reference_rate": 0.0, "think_leak_rate": 0.0},
        },
    )
    archived = phase17.phase17_decision(
        training_attempt={"real_training": "completed"},
        eval_comparison={
            "base": {"structure_hit_rate": 1.0, "citation_hit_rate": 1.0, "safety_boundary_rate": 1.0, "explicit_boundary_rate": 1.0, "unsupported_assertions": 0, "external_law_reference_rate": 0.0, "think_leak_rate": 0.0},
            "adapter": {"structure_hit_rate": 1.0, "citation_hit_rate": 1.0, "safety_boundary_rate": 1.0, "explicit_boundary_rate": 1.0, "unsupported_assertions": 0, "external_law_reference_rate": 0.0, "think_leak_rate": 0.0},
        },
    )

    assert aggregate["structure_hit_rate"] == 0.5
    assert aggregate["unsupported_assertions"] == 1
    assert passed["recommendation"] == "promote_after_manual_review"
    assert passed["auto_promotion_allowed"] is False
    assert archived["recommendation"] == "archive"
    assert "adapter_has_no_core_metric_improvement_over_base" in archived["reasons"]


def test_phase17_keeps_phase16_runtime_decision_regression() -> None:
    phase17 = _load_phase17_module()

    decision = phase17.phase16.phase16_decision(
        preflight={"ready": True},
        sample_selection={"selected_sample_count": 2},
        training_attempt={"real_training": "completed", "artifact_validation": {"valid": True}},
    )

    assert decision["status"] == "runtime_proof_passed"
    assert decision["recommendation"] == "proceed_to_qwen_dpo_probe_after_manual_review"
    assert decision["promotion_allowed"] is False
