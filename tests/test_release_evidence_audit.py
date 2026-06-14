from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    module_path = ROOT / "tools" / "release_evidence_audit.py"
    spec = importlib.util.spec_from_file_location("release_evidence_audit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_evidence_audit_passes_local_contract() -> None:
    audit = _load_audit_module()

    items = audit.audit_release_evidence(check_processes=False)

    blockers = [item for item in items if item.status == "blocker"]
    assert blockers == []
    assert {item.code for item in items} >= {
        "make_targets",
        "workflow_strict_gate",
        "dashboard_offline",
        "release_evidence_doc",
        "remote_ci_run_evidence",
    }


def test_release_evidence_audit_require_remote_blocks_without_actions_run(tmp_path: Path) -> None:
    audit = _load_audit_module()
    root = tmp_path

    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs" / "reference").mkdir(parents=True)
    (root / "docs" / "guides").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tests").mkdir()
    (root / "pfe-server" / "pfe_server" / "static").mkdir(parents=True)

    (root / "Makefile").write_text(
        "\n".join(
            [
                "test-unit:",
                "test-surface:",
                "test-e2e-mock:",
                "smoke-beta:",
                "smoke-release-strict:",
                "soak-release:",
                "benchmark-release:",
                "release-local-evidence:",
                "audit-release-evidence:",
                "audit-release-evidence-report:",
                "bundle-release-evidence:",
                "record-remote-release-evidence:",
                "render-remote-release-evidence:",
            ]
        ),
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "pfe-release-gates.yml").write_text(
        """
release-gate:
if: github.event_name != 'pull_request'
.venv/bin/python -m pip install -e ".[e2e]"
.venv/bin/python -m playwright install --with-deps chromium
.venv/bin/python tools/prepare_tiny_hf_model.py
export PFE_REAL_LOCAL_MODEL="$HOME/.cache/pfe/release-models/tiny-gpt2-local"
make test-unit
make test-surface
make test-e2e-mock
make smoke-beta
make smoke-release-strict
make benchmark-release
make audit-release-evidence-report
make bundle-release-evidence
uses: actions/upload-artifact@v4
name: pfe-release-evidence
/tmp/pfe-release-perf-report.json
/tmp/pfe-release-evidence-audit.json
/tmp/pfe-release-evidence-bundle.json
""",
        encoding="utf-8",
    )
    (root / "pfe-server" / "pfe_server" / "static" / "dashboard.html").write_text(
        "window.Chart = OfflineChart;",
        encoding="utf-8",
    )
    (root / "docs" / "reference" / "release-readiness-evidence.md").write_text(
        "\n".join(
            [
                "smoke-beta",
                "make test-e2e-mock",
                "make smoke-release-strict",
                "release_soak_smoke.py --duration-seconds 1800",
                "make benchmark-release",
                "make release-local-evidence",
                "Dashboard offline-first",
                "CI workflow contract",
                "make audit-release-evidence",
                "record-remote-release-evidence",
                "render-remote-release-evidence",
                "workflow_count=0",
            ]
        ),
        encoding="utf-8",
    )
    for path in [
        "docs/reference/release-candidate-checklist.md",
        "docs/reference/release-notes-phase2-rc.md",
        "docs/guides/install-upgrade.md",
        "docs/reference/known-limitations.md",
        "docs/reference/user-acceptance-checklist.md",
        "tools/release_perf_benchmark.py",
        "tools/release_soak_smoke.py",
        "tools/release_evidence_audit.py",
        "tools/release_evidence_bundle.py",
        "tools/github_actions_release_evidence.py",
        "tools/render_remote_release_evidence.py",
        "tools/browser_ui_live_smoke.py",
        "tools/real_local_happy_path_smoke.py",
        "tests/test_release_evidence_audit.py",
        "tests/test_release_evidence_bundle.py",
        "tests/test_github_actions_release_evidence.py",
        "tests/test_render_remote_release_evidence.py",
        "tests/test_release_workflow_contract.py",
    ]:
        (root / path).write_text("", encoding="utf-8")

    items = audit.audit_release_evidence(
        root=root,
        require_remote=True,
        check_processes=False,
        remote_evidence_report=tmp_path / "missing-remote-evidence.json",
    )

    blockers = {item.code for item in items if item.status == "blocker"}
    assert blockers == {"remote_ci_run_evidence", "remote_evidence_report"}


def test_release_evidence_audit_accepts_successful_remote_report(tmp_path: Path) -> None:
    audit = _load_audit_module()
    root = tmp_path

    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs" / "reference").mkdir(parents=True)
    (root / "docs" / "guides").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tests").mkdir()
    (root / "pfe-server" / "pfe_server" / "static").mkdir(parents=True)

    (root / "Makefile").write_text(
        "\n".join(
            [
                "test-unit:",
                "test-surface:",
                "test-e2e-mock:",
                "smoke-beta:",
                "smoke-release-strict:",
                "soak-release:",
                "benchmark-release:",
                "release-local-evidence:",
                "audit-release-evidence:",
                "audit-release-evidence-report:",
                "bundle-release-evidence:",
                "record-remote-release-evidence:",
                "render-remote-release-evidence:",
            ]
        ),
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "pfe-release-gates.yml").write_text(
        """
release-gate:
if: github.event_name != 'pull_request'
.venv/bin/python -m pip install -e ".[e2e]"
.venv/bin/python -m playwright install --with-deps chromium
.venv/bin/python tools/prepare_tiny_hf_model.py
export PFE_REAL_LOCAL_MODEL="$HOME/.cache/pfe/release-models/tiny-gpt2-local"
make test-unit
make test-surface
make test-e2e-mock
make smoke-beta
make smoke-release-strict
make benchmark-release
make audit-release-evidence-report
make bundle-release-evidence
uses: actions/upload-artifact@v4
name: pfe-release-evidence
/tmp/pfe-release-perf-report.json
/tmp/pfe-release-evidence-audit.json
/tmp/pfe-release-evidence-bundle.json
""",
        encoding="utf-8",
    )
    (root / "pfe-server" / "pfe_server" / "static" / "dashboard.html").write_text(
        "window.Chart = OfflineChart;",
        encoding="utf-8",
    )
    (root / "docs" / "reference" / "release-readiness-evidence.md").write_text(
        "\n".join(
            [
                "smoke-beta",
                "make test-e2e-mock",
                "make smoke-release-strict",
                "release_soak_smoke.py --duration-seconds 1800",
                "make benchmark-release",
                "make release-local-evidence",
                "Dashboard offline-first",
                "CI workflow contract",
                "make audit-release-evidence",
                "record-remote-release-evidence",
                "render-remote-release-evidence",
                "https://github.com/owner/repo/actions/runs/42",
            ]
        ),
        encoding="utf-8",
    )
    for path in [
        "docs/reference/release-candidate-checklist.md",
        "docs/reference/release-notes-phase2-rc.md",
        "docs/guides/install-upgrade.md",
        "docs/reference/known-limitations.md",
        "docs/reference/user-acceptance-checklist.md",
        "tools/release_perf_benchmark.py",
        "tools/release_soak_smoke.py",
        "tools/release_evidence_audit.py",
        "tools/release_evidence_bundle.py",
        "tools/github_actions_release_evidence.py",
        "tools/render_remote_release_evidence.py",
        "tools/browser_ui_live_smoke.py",
        "tools/real_local_happy_path_smoke.py",
        "tests/test_release_evidence_audit.py",
        "tests/test_release_evidence_bundle.py",
        "tests/test_github_actions_release_evidence.py",
        "tests/test_render_remote_release_evidence.py",
        "tests/test_release_workflow_contract.py",
    ]:
        (root / path).write_text("", encoding="utf-8")
    remote_report = tmp_path / "remote.json"
    remote_report.write_text(
        json.dumps(
            {
                "status": "passed",
                "release_ready": True,
                "run": {"html_url": "https://github.com/owner/repo/actions/runs/42"},
            }
        ),
        encoding="utf-8",
    )

    items = audit.audit_release_evidence(
        root=root,
        require_remote=True,
        check_processes=False,
        remote_evidence_report=remote_report,
    )

    blockers = [item for item in items if item.status == "blocker"]
    assert blockers == []


def test_release_evidence_audit_writes_json_report(tmp_path: Path) -> None:
    audit = _load_audit_module()
    report_path = tmp_path / "audit.json"
    items = [
        audit.AuditItem("local_gate", "ok", "local gate passed"),
        audit.AuditItem("remote_ci", "warn", "remote CI not recorded"),
    ]

    audit._write_report(
        report_path=report_path,
        items=items,
        require_remote=False,
        check_remote=False,
        check_processes=True,
        repo="owner/repo",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["summary"] == {"blocker": 0, "ok": 1, "total": 2, "warn": 1}
    assert payload["items"][0] == {
        "code": "local_gate",
        "message": "local gate passed",
        "status": "ok",
    }
    assert payload["repo"] == "owner/repo"
