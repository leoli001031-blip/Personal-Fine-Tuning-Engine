from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    module_path = ROOT / "tools" / "render_remote_release_evidence.py"
    spec = importlib.util.spec_from_file_location("render_remote_release_evidence", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_remote_release_evidence_success_with_bundle() -> None:
    mod = _load_module()

    markdown = mod.render_remote_release_evidence(
        remote_evidence={
            "status": "passed",
            "release_ready": True,
            "workflow_name": "PFE release gates",
            "repo": "owner/repo",
            "blockers": [],
            "run": {
                "html_url": "https://github.com/owner/repo/actions/runs/42",
                "status": "completed",
                "conclusion": "success",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "head_sha": "abc123",
                "run_started_at": "2026-06-15T00:01:00Z",
                "updated_at": "2026-06-15T00:10:00Z",
            },
        },
        bundle={
            "status": "passed",
            "summary": {"blockers": 0, "present": 3, "total": 3, "warnings": 0},
            "reports": [
                {"label": "performance", "status": "passed", "sha256": "abc", "size_bytes": 123}
            ],
        },
    )

    assert "## Remote CI evidence" in markdown
    assert "- release_ready: `yes`" in markdown
    assert "- run: https://github.com/owner/repo/actions/runs/42" in markdown
    assert "- performance: `passed` | sha256=abc | bytes=123" in markdown


def test_render_remote_release_evidence_missing_includes_blocker() -> None:
    mod = _load_module()

    markdown = mod.render_remote_release_evidence(
        remote_evidence={
            "status": "missing",
            "release_ready": False,
            "workflow_name": "PFE release gates",
            "repo": "owner/repo",
            "run": None,
            "blockers": ["matching GitHub Actions run not found"],
        }
    )

    assert "- status: `missing`" in markdown
    assert "- release_ready: `no`" in markdown
    assert "- blockers: matching GitHub Actions run not found" in markdown
