from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _make_targets() -> set[str]:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    return {
        match.group(1)
        for match in re.finditer(r"^([A-Za-z0-9_.-]+):(?:\s|$)", makefile, flags=re.MULTILINE)
    }


def test_release_workflow_references_existing_make_targets() -> None:
    workflow = (ROOT / ".github" / "workflows" / "pfe-release-gates.yml").read_text(
        encoding="utf-8"
    )
    targets = _make_targets()

    referenced_targets = {
        match.group(1)
        for match in re.finditer(r"\bmake\s+([A-Za-z0-9_.-]+)\b", workflow)
    }

    assert {
        "test-unit",
        "test-surface",
        "test-e2e-mock",
        "smoke-beta",
        "smoke-release-strict",
        "benchmark-release",
        "audit-release-evidence-report",
        "bundle-release-evidence",
    }.issubset(referenced_targets)
    assert referenced_targets <= targets


def test_release_workflow_keeps_strict_gate_requirements() -> None:
    workflow = (ROOT / ".github" / "workflows" / "pfe-release-gates.yml").read_text(
        encoding="utf-8"
    )

    assert "release-gate:" in workflow
    assert "if: github.event_name != 'pull_request'" in workflow
    assert '.venv/bin/python -m pip install -e ".[e2e]"' in workflow
    assert ".venv/bin/python -m playwright install --with-deps chromium" in workflow
    assert ".venv/bin/python tools/prepare_tiny_hf_model.py" in workflow
    assert 'export PFE_REAL_LOCAL_MODEL="$HOME/.cache/pfe/release-models/tiny-gpt2-local"' in workflow
    assert "make test-e2e-mock" in workflow
    assert "make smoke-release-strict" in workflow
    assert "make benchmark-release" in workflow
    assert "make audit-release-evidence-report" in workflow
    assert "make bundle-release-evidence" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "pfe-release-evidence" in workflow
    assert "/tmp/pfe-release-perf-report.json" in workflow
    assert "/tmp/pfe-release-evidence-audit.json" in workflow
    assert "/tmp/pfe-release-evidence-bundle.json" in workflow


def test_release_local_evidence_target_sequences_required_gates() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "release-local-evidence:" in makefile
    assert "$(MAKE) test-e2e-mock" in makefile
    assert "$(MAKE) smoke-release-strict" in makefile
    assert "$(MAKE) benchmark-release PERF_REPORT=$(PERF_REPORT)" in makefile
    assert "$(MAKE) audit-release-evidence-report AUDIT_REPORT=$(AUDIT_REPORT)" in makefile
    assert (
        "$(MAKE) bundle-release-evidence PERF_REPORT=$(PERF_REPORT) "
        "AUDIT_REPORT=$(AUDIT_REPORT) REMOTE_EVIDENCE_REPORT=$(REMOTE_EVIDENCE_REPORT) "
        "BUNDLE_REPORT=$(BUNDLE_REPORT)"
    ) in makefile
