from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "tools/phase100_qwen3_generation_boundary_closure.py"


def test_phase100_driver_is_local_only_and_forbids_post_hoc_truncation():
    text = DRIVER.read_text(encoding="utf-8")
    assert 'MODEL_PATH = REPO_ROOT / "models/Qwen3-4B"' in text
    assert 'os.environ.setdefault("HF_HUB_OFFLINE", "1")' in text
    assert 'os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")' in text
    assert "local_files_only=True" in text
    assert '"post_hoc_truncation_allowed": False' in text
    assert '"post_hoc_truncation_used": False' in text
    assert "http://" not in text
    assert "https://" not in text


def test_phase100_driver_keeps_the_270_call_and_two_diagnostic_caps():
    text = DRIVER.read_text(encoding="utf-8")
    assert '"long_run_total_call_budget": 270' in text
    assert "if attempt not in (1, 2):" in text
    assert '"phase100_model_call_budget_maximum": 48' in text


def test_phase100_driver_requires_diagnostic_before_final_gate():
    text = DRIVER.read_text(encoding="utf-8")
    assert "diagnostic_passed" in text
    assert "diagnostics_exhausted = len(diagnostic_paths) == 2" in text
    assert "requires a passing diagnostic or both frozen diagnostic attempts" in text
    assert "build_phase100_generation_controls" in text
    assert "logits_processor=logits_processor" in text
    assert "stopping_criteria=stopping_criteria" in text
