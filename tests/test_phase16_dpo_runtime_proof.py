from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_phase16_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "phase16_dpo_runtime_proof.py"
    spec = importlib.util.spec_from_file_location("phase16_dpo_runtime_proof", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_phase16_selects_phase15_dpo_samples(tmp_path: Path) -> None:
    phase16 = _load_phase16_module()
    phase15_dir = tmp_path / "phase15-source"
    evidence_dir = tmp_path / "phase16-evidence"
    rows = [
        {"sample_id": f"sample-{index}", "sample_type": "dpo", "instruction": "prompt", "chosen": "good", "rejected": "bad"}
        for index in range(3)
    ]
    _write_jsonl(phase15_dir / "dpo_samples.jsonl", rows)
    (phase15_dir / "quality_report.json").write_text('{"dpo_sample_count": 3}\n', encoding="utf-8")

    selection = phase16.load_or_build_phase15_samples(evidence_dir=evidence_dir, phase15_evidence_dir=phase15_dir, pair_limit=2)
    selected = phase16._read_jsonl(evidence_dir / "selected_dpo_samples.jsonl")  # noqa: SLF001

    assert selection["source_sample_count"] == 3
    assert selection["selected_sample_count"] == 2
    assert [row["sample_id"] for row in selected] == ["sample-0", "sample-1"]
    assert all(row["sample_type"] == "dpo" for row in selected)


def test_phase16_tiny_job_spec_and_skip_real_proof_dry_run(tmp_path: Path) -> None:
    phase16 = _load_phase16_module()
    samples = [{"sample_id": "s1", "instruction": "prompt", "chosen": "good", "rejected": "bad"}]
    job_spec = phase16.build_tiny_dpo_job_spec(
        samples=samples,
        base_model="hf-internal-testing/tiny-random-gpt2",
        output_dir=tmp_path / "adapter",
        epochs=1,
        beta=0.1,
        max_length=128,
        max_prompt_length=96,
    )

    attempt = phase16.run_dpo_runtime_proof(
        evidence_dir=tmp_path / "evidence",
        job_spec=job_spec,
        preflight={"ready": True},
        run_real_dpo_proof=False,
    )

    assert job_spec["execution_executor"] == "dpo"
    assert job_spec["training_examples"][0]["sample_type"] == "dpo"
    assert job_spec["training_examples"][0]["rejected"] == "bad"
    assert attempt["real_training"] == "not_started"
    assert attempt["dry_run_result"]["status"] == "prepared"
    assert attempt["dry_run_result"]["backend"] == "dpo"


def test_phase16_validates_dpo_adapter_artifacts(tmp_path: Path) -> None:
    phase16 = _load_phase16_module()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("stub\n", encoding="utf-8")

    valid = phase16.validate_dpo_artifact({"artifact_dir": str(adapter_dir)})
    missing = phase16.validate_dpo_artifact({"artifact_dir": str(tmp_path / "missing")})

    assert valid["valid"] is True
    assert valid["adapter_config_exists"] is True
    assert valid["adapter_model_exists"] is True
    assert missing["valid"] is False


def test_phase16_decision_blocks_or_proceeds_after_runtime_proof() -> None:
    phase16 = _load_phase16_module()

    blocked = phase16.phase16_decision(
        preflight={"ready": False, "missing_modules": ["trl"]},
        sample_selection={"selected_sample_count": 2},
        training_attempt={"real_training": "blocked"},
    )
    passed = phase16.phase16_decision(
        preflight={"ready": True},
        sample_selection={"selected_sample_count": 2},
        training_attempt={"real_training": "completed", "artifact_validation": {"valid": True}},
    )

    assert blocked["recommendation"] == "archive"
    assert "dpo_runtime_dependencies_not_ready" in blocked["reasons"]
    assert "real_dpo_runtime_proof_not_completed" in blocked["reasons"]
    assert passed["status"] == "runtime_proof_passed"
    assert passed["recommendation"] == "proceed_to_qwen_dpo_probe_after_manual_review"
    assert passed["promotion_allowed"] is False
