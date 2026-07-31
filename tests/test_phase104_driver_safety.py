from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "tools/phase104_finalize_autonomous_loop.py"


def test_phase104_finalizer_preserves_archive_and_runtime_labels():
    text = DRIVER.read_text(encoding="utf-8")
    assert '"final_recommendation_runtime_primary"' in text
    assert '"phase101_archived"' in text
    assert '"phase102_archived"' in text
    assert '"product_gate_false"' in text
    assert '"automatic_promotion_false"' in text
    assert '"deployment_false"' in text


def test_phase104_finalizer_does_not_commit_private_transcripts():
    text = DRIVER.read_text(encoding="utf-8")
    assert '"private_transcripts_committed": False' in text
    assert '"private_transcripts_not_committed"' in text
    assert "/private/tmp/pfe-phase103-simulated-review" in text


def test_phase104_runbook_contains_all_local_phases():
    text = DRIVER.read_text(encoding="utf-8")
    for phase in ("phase100", "phase101", "phase102", "phase103", "phase104"):
        assert phase in text
    assert "push" not in text
    assert "deploy any Phase101/102 adapter" in text
