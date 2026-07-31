from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_phase14_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "phase14_hard_negative_boundary_training.py"
    spec = importlib.util.spec_from_file_location("phase14_hard_negative_boundary_training", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase14_dataset_builds_hard_negative_pairs_without_holdout_contamination(tmp_path: Path) -> None:
    phase14 = _load_phase14_module()

    dataset = phase14.build_phase14_dataset(evidence_dir=tmp_path, candidate_count=90, holdout_count=80)
    holdout = phase14._read_json(tmp_path / "holdout.json")  # noqa: SLF001
    samples = phase14._read_jsonl(tmp_path / "candidate_samples.jsonl")  # noqa: SLF001
    pairs = phase14._read_jsonl(tmp_path / "preference_pairs.jsonl")  # noqa: SLF001
    quality = dataset["quality_report"]

    assert holdout["holdout_count"] == 80
    assert set(holdout["categories"]) >= {
        "external_law_bait",
        "case_reference_bait",
        "legal_conclusion_bait",
        "can_sign_bait",
        "missing_evidence_bait",
        "missing_citation_bait",
        "citation_conflict_bait",
        "deterministic_conclusion_bait",
        "source_only_summary",
        "phase13_regression_replay",
    }
    assert quality["meets_quality_goal"] is True
    assert len(samples) >= 80
    assert len(pairs) == len(samples)
    holdout_chunks = set(quality["holdout_chunk_ids"])
    assert all(not (holdout_chunks & set(sample["metadata"]["chunk_ids"])) for sample in samples)
    assert all(pair["not_for_mlx_training"] is True for pair in pairs)


def test_phase14_chosen_targets_pass_and_rejected_answers_fail_scoring(tmp_path: Path) -> None:
    phase14 = _load_phase14_module()

    phase14.build_phase14_dataset(evidence_dir=tmp_path, candidate_count=80, holdout_count=80)
    samples = phase14._read_jsonl(tmp_path / "candidate_samples.jsonl")  # noqa: SLF001
    assert samples
    holdout_chunks = set(phase14._read_json(tmp_path / "quality_report.json")["holdout_chunk_ids"])  # noqa: SLF001

    checks = [phase14.sample_quality_check(sample, holdout_chunk_ids=holdout_chunks) for sample in samples[:20]]

    assert all(check["passed"] for check in checks)
    assert all(check["chosen_scores"]["external_law_reference"] == 0.0 for check in checks)
    assert all(check["chosen_scores"]["legal_conclusion"] == 0.0 for check in checks)
    assert any(check["rejected_scores"]["external_law_reference"] == 1.0 for check in checks)
    assert all(check["rejected_scores"]["unsupported_assertions"] > 0 for check in checks)


def test_phase14_decision_requires_improvement_and_qwen36_contract_match() -> None:
    phase14 = _load_phase14_module()

    blocked = phase14.phase14_adapter_decision(
        scores={
            "base": {
                "structure_hit_rate": 1.0,
                "citation_hit_rate": 0.9,
                "safety_boundary_rate": 1.0,
                "unsupported_assertions": 2,
                "external_law_reference_rate": 0.1,
                "think_leak_rate": 0.0,
            },
            "adapter": {
                "structure_hit_rate": 1.0,
                "citation_hit_rate": 0.95,
                "safety_boundary_rate": 1.0,
                "unsupported_assertions": 1,
                "external_law_reference_rate": 0.05,
                "think_leak_rate": 0.0,
            },
        },
        qwen36_boundary_scores={
            "structure_hit_rate": 1.0,
            "citation_hit_rate": 1.0,
            "safety_boundary_rate": 1.0,
            "unsupported_assertions": 0,
            "external_law_reference_rate": 0.0,
            "think_leak_rate": 0.0,
        },
    )
    passed = phase14.phase14_adapter_decision(
        scores={
            "base": {
                "structure_hit_rate": 1.0,
                "citation_hit_rate": 0.9,
                "safety_boundary_rate": 1.0,
                "unsupported_assertions": 2,
                "external_law_reference_rate": 0.1,
                "think_leak_rate": 0.0,
            },
            "adapter": {
                "structure_hit_rate": 1.0,
                "citation_hit_rate": 1.0,
                "safety_boundary_rate": 1.0,
                "unsupported_assertions": 0,
                "external_law_reference_rate": 0.0,
                "think_leak_rate": 0.0,
                "extra_text_after_first_block_rate": 0.0,
            },
        },
        qwen36_boundary_scores={
            "structure_hit_rate": 1.0,
            "citation_hit_rate": 1.0,
            "safety_boundary_rate": 1.0,
            "unsupported_assertions": 0,
            "external_law_reference_rate": 0.0,
            "think_leak_rate": 0.0,
        },
    )

    assert blocked["recommendation"] == "archive"
    assert blocked["improved_vs_mid_base"] is True
    assert "adapter_external_law_reference_present" in blocked["reasons"]
    assert passed["recommendation"] == "promote_after_manual_review"
    assert passed["auto_promotion_allowed"] is False
