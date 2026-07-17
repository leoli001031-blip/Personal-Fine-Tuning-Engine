from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "tools/phase107_runtime_provenance_and_token_faithful_dpo.py"


def test_phase107_driver_is_local_bounded_and_uses_phase106_parent():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"' in text
    assert 'PARENT_ADAPTER_ROOT = REPO_ROOT / "trainer_job_outputs/phase106-qwen3-4b-sft-30step/peft_lora"' in text
    assert "MODEL_CALL_BUDGET = 180" in text
    assert '"local_only": True' in text
    assert 'if steps not in (1, 12, 30):' in text
    assert "http://" not in text
    assert "https://" not in text


def test_phase107_driver_keeps_private_outputs_outside_repo():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'PRIVATE_ROOT = Path("/private/tmp/pfe-phase107-simulated-review")' in text
    assert '"private_cache_outside_repo": True' in text
    assert '"private_transcripts_committed": False' in text
    assert "_write_private_jsonl(cache_path, private_rows)" in text
    assert '"raw_output":' not in (REPO_ROOT / "pfe-core/pfe_core/phase107_runtime_provenance_dpo.py").read_text(encoding="utf-8")


def test_phase107_driver_never_auto_promotes_or_qualifies_product_gate():
    text = DRIVER.read_text(encoding="utf-8")
    assert '"product_gate_qualified": False' in text
    assert '"automatic_promotion_allowed": False' in text
    assert "promote_after_manual_review" in text
    assert "push" not in text
    assert "deploy" not in text


def test_phase107_driver_freezes_three_variants_and_strict_provenance_metrics():
    text = DRIVER.read_text(encoding="utf-8")
    assert '{"base", "phase106_sft", "phase107_dpo"}' in text
    assert '"provenance_envelope_integrity_rate"' in text
    assert '"metadata_injection_resistance_rate"' in text
    assert "semantic_provenance_rate" in text
    assert "literal_provenance_rate" in text
