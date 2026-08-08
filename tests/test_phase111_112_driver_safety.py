from __future__ import annotations

import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE85_TEST = REPO_ROOT / "tests/test_phase85_driver_safety.py"
WORKFLOW = REPO_ROOT / ".github/workflows/pfe-release-gates.yml"
FROZEN_HASH = "02de4e86e2d4b018b4a100cfc310b47c6782cae33790e70814dcea8ff425139f"


def test_phase85_frozen_driver_test_remains_byte_identical() -> None:
    assert hashlib.sha256(PHASE85_TEST.read_bytes()).hexdigest() == FROZEN_HASH


def test_fast_gate_places_pytest_basetemp_outside_os_temp_allowlist() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'PYTEST_ADDOPTS: "--basetemp=${{ github.workspace }}/.pytest-tmp"' in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "make test-unit" in workflow
    assert "make test-surface" in workflow
    assert "make test-e2e-mock" in workflow
    assert "make smoke-beta" in workflow
