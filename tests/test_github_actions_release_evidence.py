from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    module_path = ROOT / "tools" / "github_actions_release_evidence.py"
    spec = importlib.util.spec_from_file_location("github_actions_release_evidence", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_release_run_filters_workflow_branch_and_event() -> None:
    mod = _load_module()
    payload = {
        "workflow_runs": [
            {
                "id": 1,
                "name": "Other workflow",
                "event": "workflow_dispatch",
                "head_branch": "main",
            },
            {
                "id": 2,
                "name": "PFE release gates",
                "event": "schedule",
                "head_branch": "main",
            },
            {
                "id": 3,
                "name": "PFE release gates",
                "event": "workflow_dispatch",
                "head_branch": "release",
            },
        ]
    }

    run = mod.select_release_run(
        payload,
        workflow_name="PFE release gates",
        branch="release",
        event="workflow_dispatch",
    )

    assert run["id"] == 3


def test_build_evidence_marks_successful_completed_run_release_ready() -> None:
    mod = _load_module()
    evidence = mod.build_evidence(
        repo="owner/repo",
        workflow_name="PFE release gates",
        require_success=True,
        run={
            "id": 42,
            "name": "PFE release gates",
            "event": "workflow_dispatch",
            "status": "completed",
            "conclusion": "success",
            "html_url": "https://github.com/owner/repo/actions/runs/42",
            "head_branch": "main",
            "head_sha": "abc123",
            "created_at": "2026-06-15T00:00:00Z",
            "updated_at": "2026-06-15T00:10:00Z",
            "run_started_at": "2026-06-15T00:01:00Z",
        },
    )

    assert evidence["status"] == "passed"
    assert evidence["release_ready"] is True
    assert evidence["blockers"] == []
    assert evidence["run"]["html_url"].endswith("/actions/runs/42")


def test_build_evidence_blocks_missing_or_failed_required_run() -> None:
    mod = _load_module()

    missing = mod.build_evidence(
        repo="owner/repo",
        workflow_name="PFE release gates",
        run=None,
        require_success=True,
    )
    failed = mod.build_evidence(
        repo="owner/repo",
        workflow_name="PFE release gates",
        require_success=True,
        run={
            "id": 43,
            "name": "PFE release gates",
            "status": "completed",
            "conclusion": "failure",
        },
    )

    assert missing["status"] == "missing"
    assert missing["release_ready"] is False
    assert failed["status"] == "blocked"
    assert failed["release_ready"] is False
    assert failed["blockers"] == [
        "latest matching run is not successful: status=completed conclusion=failure"
    ]


def test_main_returns_nonzero_when_required_run_is_missing(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    report_path = tmp_path / "remote-evidence.json"
    monkeypatch.setattr(mod, "_gh_runs", lambda repo, per_page: {"workflow_runs": []})

    exit_code = mod.main(["--require-success", "--output-path", str(report_path)])

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["status"] == "missing"
    assert payload["blockers"] == ["matching GitHub Actions run not found"]
