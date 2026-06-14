from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    module_path = ROOT / "tools" / "release_evidence_bundle.py"
    spec = importlib.util.spec_from_file_location("release_evidence_bundle", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bundle_passes_with_local_reports_and_optional_remote_warning(tmp_path: Path) -> None:
    mod = _load_module()
    perf = tmp_path / "perf.json"
    audit = tmp_path / "audit.json"
    remote = tmp_path / "remote.json"
    _write_json(perf, {"status": "passed"})
    _write_json(audit, {"status": "passed"})

    bundle = mod.build_bundle(
        perf_report=perf,
        audit_report=audit,
        remote_evidence_report=remote,
        require_remote=False,
    )

    assert bundle["status"] == "passed"
    assert bundle["summary"]["present"] == 2
    assert bundle["summary"]["warnings"] == 1
    assert bundle["warnings"] == [f"remote_actions: missing ({remote})"]
    assert all("sha256" in item for item in bundle["reports"] if item["present"])


def test_bundle_requires_remote_success_when_requested(tmp_path: Path) -> None:
    mod = _load_module()
    perf = tmp_path / "perf.json"
    audit = tmp_path / "audit.json"
    remote = tmp_path / "remote.json"
    _write_json(perf, {"status": "passed"})
    _write_json(audit, {"status": "passed"})
    _write_json(remote, {"status": "missing", "release_ready": False})

    bundle = mod.build_bundle(
        perf_report=perf,
        audit_report=audit,
        remote_evidence_report=remote,
        require_remote=True,
    )

    assert bundle["status"] == "blocked"
    assert bundle["blockers"] == ["remote_actions: status=missing release_ready=False"]
