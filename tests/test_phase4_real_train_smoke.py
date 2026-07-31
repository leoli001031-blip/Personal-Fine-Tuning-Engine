from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _pythonpath() -> str:
    parts = [str(ROOT / name) for name in ("pfe-core", "pfe-cli", "pfe-server")]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def test_phase4_real_train_smoke_defaults_to_clear_skip(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    env.pop("PFE_PHASE4_REAL_TRAIN_MODEL", None)
    env.pop("PFE_PHASE4_PREPARE_TINY_MODEL", None)

    completed = subprocess.run(
        [
            sys.executable,
            "tools/phase4_real_train_smoke.py",
            "--workdir",
            str(tmp_path / "phase4-train-smoke"),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["real_training"] == "skipped"
    assert payload["skip_reason"] == "PFE_PHASE4_REAL_TRAIN_MODEL is not set"
    assert payload["mock_fallback"] is True
    assert payload["saved_training_samples"] > 0
    assert payload["split_counts"]["train"] > 0
    assert payload["candidate_adapter_state"] == "pending_eval"
