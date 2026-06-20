from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_phase15_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "phase15_preference_boundary_training.py"
    spec = importlib.util.spec_from_file_location("phase15_preference_boundary_training", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase15_builds_true_dpo_pairs_from_phase14_hard_negatives(tmp_path: Path) -> None:
    phase15 = _load_phase15_module()
    phase14_dir = tmp_path / "phase14-source"
    evidence_dir = tmp_path / "phase15-evidence"

    dataset = phase15.build_phase15_preference_dataset(
        evidence_dir=evidence_dir,
        phase14_evidence_dir=phase14_dir,
        candidate_count=90,
        holdout_count=80,
        pair_limit=90,
    )
    samples = phase15._read_jsonl(evidence_dir / "dpo_samples.jsonl")  # noqa: SLF001
    quality = dataset["quality_report"]

    assert quality["meets_quality_goal"] is True
    assert quality["dpo_sample_count"] >= 80
    assert quality["failed_quality_count"] == 0
    assert samples
    assert all(sample["sample_type"] == "dpo" for sample in samples)
    assert all(sample["chosen"] != sample["rejected"] for sample in samples)
    assert all(sample["metadata"]["training_strategy"] == "true_preference_dpo_boundary_pair" for sample in samples)
    assert any(sample["metadata"]["hard_negative_category"] == "external_law_bait" for sample in samples)


def test_phase15_preference_quality_rejects_non_dpo_sample(tmp_path: Path) -> None:
    phase15 = _load_phase15_module()
    dataset = phase15.build_phase15_preference_dataset(
        evidence_dir=tmp_path / "phase15",
        phase14_evidence_dir=tmp_path / "phase14",
        candidate_count=80,
        holdout_count=80,
        pair_limit=80,
    )
    sample = phase15._read_jsonl(Path(dataset["dpo_samples"]["path"]))[0]  # noqa: SLF001
    sample["sample_type"] = "sft"

    check = phase15.preference_pair_quality_check(sample, holdout_chunk_ids=set(dataset["quality_report"]["holdout_chunk_ids"]))

    assert check["passed"] is False
    assert "sample_type_not_dpo" in check["reasons"]


def test_phase15_dpo_job_spec_dry_run_uses_rejected_pairs(tmp_path: Path) -> None:
    phase15 = _load_phase15_module()
    phase15.build_phase15_preference_dataset(
        evidence_dir=tmp_path / "phase15",
        phase14_evidence_dir=tmp_path / "phase14",
        candidate_count=80,
        holdout_count=80,
        pair_limit=80,
    )
    samples = phase15._read_jsonl(tmp_path / "phase15" / "dpo_samples.jsonl")  # noqa: SLF001
    job_spec = phase15.build_dpo_job_spec(
        samples=samples,
        base_model="Qwen/Qwen3-0.6B",
        output_dir=tmp_path / "trainer-output",
        epochs=1,
        beta=0.1,
        max_length=512,
        max_prompt_length=384,
        sample_limit=16,
    )
    dry_run = phase15.run_dpo_dry_run(evidence_dir=tmp_path / "phase15", job_spec=job_spec)

    assert job_spec["execution_executor"] == "dpo"
    assert len(job_spec["training_examples"]) == 16
    assert all(example["sample_type"] == "dpo" for example in job_spec["training_examples"])
    assert all(example["rejected"] for example in job_spec["training_examples"])
    assert dry_run["dry_run_result"]["backend"] == "dpo"
    assert dry_run["dry_run_result"]["status"] == "prepared"
    assert dry_run["dry_run_result"]["num_examples"] == 16


def test_phase15_decision_archives_until_real_dpo_training_completes() -> None:
    phase15 = _load_phase15_module()

    blocked = phase15.phase15_decision(
        quality_report={"meets_quality_goal": True},
        preflight={"ready": False, "missing_modules": ["trl"]},
        training_attempt={"real_training": "blocked"},
    )
    passed_for_manual_eval = phase15.phase15_decision(
        quality_report={"meets_quality_goal": True},
        preflight={"ready": True},
        training_attempt={"real_training": "completed"},
    )

    assert blocked["recommendation"] == "archive"
    assert "dpo_runtime_dependencies_not_ready" in blocked["reasons"]
    assert "real_dpo_training_not_completed" in blocked["reasons"]
    assert passed_for_manual_eval["recommendation"] == "promote_after_manual_review"
    assert passed_for_manual_eval["auto_promotion_allowed"] is False
