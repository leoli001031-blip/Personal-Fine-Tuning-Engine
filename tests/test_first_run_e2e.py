from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.e2e
def test_first_run_smoke_script_reaches_usable_local_surfaces() -> None:
    script = ROOT / "tools" / "first_run_smoke.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--python",
            sys.executable,
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout
    assert "$ pfe init --workspace first_run --base-model ./models/local-base --home .pfe" in output
    assert "$ pfe doctor --workspace first_run" in output
    assert "local model: available=yes | requested_base_model=./models/local-base" in output
    assert "$ pfe next --workspace first_run" in output
    assert "state: collect_feedback" in output
    assert "$ pfe generate --scenario life-coach --style warm --num 8 --workspace first_run" in output
    assert "Saved 8 distilled sample(s)" in output
    assert "$ pfe trigger configure --workspace first_run --enable --min-new-samples 1 --queue-mode deferred" in output
    assert "[ AUTO TRAIN ACTION ]" in output
    assert "queue mode" in output
    assert "mock_local" in output
    assert "$ pfe collect ingest --workspace first_run --event-id evt-first-run-feedback-1" in output
    assert "Signal ingested" in output
    assert "Event Chain Complete: True" in output
    assert "Auto Train: queued (enqueued)" in output
    assert "$ pfe collect status --workspace first_run" in output
    assert "Total Signals: 1" in output
    assert "$ pfe collect review --workspace first_run --type accept --limit 5" in output
    assert "Signal ID: evt-first-run-feedback-1" in output
    assert "Confidence: 0.90" in output
    assert "$ pfe trigger status --workspace first_run" in output
    assert "[ AUTO TRAIN TRIGGER ]" in output
    assert "$ pfe trigger process-next --workspace first_run" in output
    assert "process_next" in output
    assert "state: queue_ready" in output
    assert "state: candidate_ready" in output
    assert "queue adapter version:" in output
    assert "$ pfe eval --base-model base --adapter " in output
    assert "[ EVALUATION RESULT ]" in output
    assert "recommendation:" in output
    assert "$ pfe adapter promote " in output
    assert "latest:" in output
    assert "$ pfe serve --host 127.0.0.1 --port 8921 --workspace first_run" in output
    assert "[ SERVE PREVIEW ]" in output
    assert "[ LATEST PROMOTED ]" in output
    assert "preview only" in output
    assert "FIRST-RUN SMOKE PASSED" in output
    assert "base_model: ./models/local-base" in output


@pytest.mark.e2e
def test_first_run_smoke_can_stop_after_auto_train_queue() -> None:
    script = ROOT / "tools" / "first_run_smoke.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--python",
            sys.executable,
            "--stop-after",
            "queue",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout
    assert "$ pfe trigger configure --workspace first_run --enable --min-new-samples 1 --queue-mode deferred" in output
    assert "Auto Train: queued (enqueued)" in output
    assert "$ pfe trigger process-next --workspace first_run" in output
    assert "process_next" in output
    assert "state: queue_ready" in output
    assert "state: candidate_ready" in output
    assert "queue adapter version:" in output
    assert "AUTO-TRAIN QUEUE SMOKE PASSED" in output
    assert "manifest:" in output
    assert "queue:" in output
    assert "$ pfe eval --base-model base --adapter " not in output
    assert "$ pfe adapter promote " not in output
    assert "$ pfe serve --host 127.0.0.1 --port 8921 --workspace first_run" not in output


@pytest.mark.e2e
def test_real_local_readiness_smoke_script_reaches_preview_surfaces() -> None:
    script = ROOT / "tools" / "real_local_readiness_smoke.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--python",
            sys.executable,
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout
    assert "REAL-LOCAL READINESS SMOKE PASSED" in output
    assert "workspace:  real_local_ready" in output
    assert "base_model: ./models/local-base" in output
    assert "config:" in output


@pytest.mark.e2e
def test_studio_model_path_smoke_script_reaches_api_handoff() -> None:
    script = ROOT / "tools" / "studio_model_path_smoke.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--python",
            sys.executable,
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout
    assert "STUDIO MODEL PATH SMOKE PASSED" in output
    assert "workspace:           studio-client" in output
    assert "api_url:             http://127.0.0.1:8921/v1/chat/completions" in output
    assert "real_local_enabled:  True" in output
    assert "training_preflight:  True" in output
    assert "training_jobs:       0" in output
    assert "chat_served_by:      mock" in output
    assert "chat_resolved_model:" in output


@pytest.mark.e2e
def test_server_live_smoke_script_probes_http_surfaces() -> None:
    script = ROOT / "tools" / "server_live_smoke.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--python",
            sys.executable,
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout
    assert "SERVER LIVE SMOKE PASSED" in output
    assert "workspace: server_live" in output
    assert "base_url:  http://127.0.0.1:" in output
    assert "manifest:" in output


@pytest.mark.e2e
def test_dashboard_console_live_smoke_script_probes_frontend_contracts() -> None:
    script = ROOT / "tools" / "dashboard_console_live_smoke.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--python",
            sys.executable,
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout
    assert "DASHBOARD CONSOLE LIVE SMOKE PASSED" in output
    assert "workspace:     dashboard_console_live" in output
    assert "base_url:      http://127.0.0.1:" in output
    assert "dashboard_api: ok" in output
    assert "chat_feedback: ok" in output


@pytest.mark.e2e
def test_real_local_happy_path_smoke_skips_without_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PFE_REAL_LOCAL_MODEL", raising=False)
    script = ROOT / "tools" / "real_local_happy_path_smoke.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--python",
            sys.executable,
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout
    assert "REAL-LOCAL HAPPY PATH SMOKE SKIPPED" in output
    assert "set PFE_REAL_LOCAL_MODEL" in output


@pytest.mark.e2e
def test_browser_ui_live_smoke_skips_without_playwright(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    script = ROOT / "tools" / "browser_ui_live_smoke.py"
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("browser_ui_live_smoke_skip_test", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_load_sync_playwright", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--repo-root",
            str(ROOT),
            "--python",
            sys.executable,
        ],
    )

    assert module.main() == 0
    output = capsys.readouterr().out
    assert "BROWSER UI LIVE SMOKE SKIPPED" in output
    assert "Playwright is not installed" in output


@pytest.mark.e2e
def test_real_local_happy_path_smoke_strict_fails_without_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PFE_REAL_LOCAL_MODEL", raising=False)
    script = ROOT / "tools" / "real_local_happy_path_smoke.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--python",
            sys.executable,
            "--strict",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 2
    output = completed.stdout
    assert "REAL-LOCAL HAPPY PATH SMOKE SKIPPED" in output
    assert "set PFE_REAL_LOCAL_MODEL" in output


@pytest.mark.e2e
def test_browser_ui_live_smoke_strict_fails_without_playwright(monkeypatch: pytest.MonkeyPatch) -> None:
    script = ROOT / "tools" / "browser_ui_live_smoke.py"
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("browser_ui_live_smoke_strict_test", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_load_sync_playwright", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--repo-root",
            str(ROOT),
            "--python",
            sys.executable,
            "--strict",
        ],
    )

    assert module.main() == 2
