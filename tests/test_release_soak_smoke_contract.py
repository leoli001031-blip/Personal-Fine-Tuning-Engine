from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_release_soak_module():
    tools_path = str(ROOT / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    module_path = ROOT / "tools" / "release_soak_smoke.py"
    spec = importlib.util.spec_from_file_location("release_soak_smoke", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_soak_iteration_checks_legacy_chat_explicit_route(monkeypatch) -> None:
    soak = _load_release_soak_module()
    text_probes: list[tuple[str, str]] = []

    def fake_json_probe(base_url, path, *, args, stats, method="GET", body=None):
        if path == "/pfe/auto-train/worker-daemon":
            return {"running": True}
        return {}

    def fake_json_payload_probe(base_url, path, *, args, stats, method="GET", body=None):
        return []

    def fake_text_probe(base_url, path, *, args, stats, expected):
        text_probes.append((path, expected))

    monkeypatch.setattr(soak, "_json_probe", fake_json_probe)
    monkeypatch.setattr(soak, "_json_payload_probe", fake_json_payload_probe)
    monkeypatch.setattr(soak, "_text_probe", fake_text_probe)
    monkeypatch.setattr(soak, "_validate_status", lambda status, *, expected_version: None)
    monkeypatch.setattr(soak, "_validate_daemon", lambda daemon: None)

    soak._run_iteration(
        "http://127.0.0.1:9999",
        args=SimpleNamespace(chat_every=0),
        stats=soak.SoakStats(),
        expected_version="test-version",
    )

    assert ("/dashboard", "PFE Observability Dashboard") in text_probes
    assert ("/chat", "PFE Local Chat") in text_probes
    assert ("/", "PFE Local Chat") not in text_probes
