from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.db.schema import SIGNALS_TABLE
from pfe_core.db.sqlite import initialize_database, record_signal, signals_db_path, status_snapshot
from pfe_core.server_services import InferenceServiceAdapter
from pfe_server.app import _load_status_snapshot


def test_current_latest_version_ignores_regular_file_pointer(pfe_home: Path) -> None:
    store = AdapterStore(home=pfe_home)
    latest_path = pfe_home / "adapters" / "user_default" / "latest"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text("not-a-symlink", encoding="utf-8")

    assert store.current_latest_version() is None


def test_signal_summary_counts_only_present_lineage_fields(pfe_home: Path) -> None:
    record_signal(
        {
            "event_id": "evt-complete",
            "source_event_id": "evt-source",
            "request_id": "req-1",
            "session_id": "sess-1",
            "adapter_version": "20260420-001",
            "event_type": "accept",
            "timestamp": "2026-04-20T08:00:00+00:00",
        },
        home=pfe_home,
    )
    conn = initialize_database(signals_db_path(pfe_home))
    try:
        conn.execute(
            f"""
            INSERT INTO {SIGNALS_TABLE} (
                id, source_event_id, request_id, session_id, parent_event_id, source_event_ids,
                event_chain_ids, adapter_version, event_type, timestamp, context, model_output,
                user_input, action_detail, user_action, lineage, metadata, processed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "evt-legacy-empty",
                "",
                "",
                "",
                None,
                json.dumps([]),
                json.dumps([]),
                "20260420-001",
                "accept",
                "2026-04-20T08:05:00+00:00",
                None,
                None,
                None,
                json.dumps({}),
                json.dumps({}),
                json.dumps({}),
                json.dumps({}),
                0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    snapshot = status_snapshot(home=pfe_home)
    signal_summary = snapshot["signal_summary"]

    assert snapshot["signal_count"] == 2
    assert signal_summary["source_event_id_count"] == 1
    assert signal_summary["request_id_count"] == 1
    assert signal_summary["session_id_count"] == 1


def test_load_status_snapshot_degrades_when_latest_manifest_is_missing(pfe_home: Path) -> None:
    store = AdapterStore(home=pfe_home)
    created = store.create_training_version(
        base_model="base-model",
        training_config={"backend": "mock_local", "train_type": "sft"},
    )
    store.mark_pending_eval(created["version"], num_samples=1)
    store.promote(created["version"])

    version_dir = pfe_home / "adapters" / "user_default" / created["version"]
    (version_dir / "adapter_manifest.json").unlink()

    snapshot = _load_status_snapshot("user_default")

    assert snapshot["latest_adapter_version"] == created["version"]
    assert snapshot["latest_adapter"]["version"] == created["version"]
    assert snapshot["latest_adapter"]["path"] == str(version_dir)
    assert snapshot["latest_adapter"]["state"] is None


def test_inference_service_status_uses_snapshot_workspace_for_latest_adapter() -> None:
    class FakePipeline:
        def status(self) -> dict[str, object]:
            return {
                "workspace": "alpha",
                "latest_adapter": {"version": "20260420-001"},
            }

    class FakeStore:
        def load(self, version: str) -> str:
            assert version == "latest"
            return "/tmp/alpha-latest"

    captured: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, config):
            captured["adapter_path"] = config.adapter_path

        def status(self) -> dict[str, object]:
            return {"engine": "ok"}

    service = InferenceServiceAdapter.__new__(InferenceServiceAdapter)
    service.pipeline = FakePipeline()

    def fake_create_adapter_store(workspace=None):
        captured["workspace"] = workspace
        return FakeStore()

    with patch("pfe_core.server_services.create_adapter_store", side_effect=fake_create_adapter_store), patch(
        "pfe_core.server_services.InferenceEngine",
        FakeEngine,
    ):
        payload = asyncio.run(service.status())

    assert captured["workspace"] == "alpha"
    assert captured["adapter_path"] == "/tmp/alpha-latest"
    assert payload["workspace"] == "alpha"
