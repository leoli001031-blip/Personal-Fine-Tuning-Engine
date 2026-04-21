from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.db.sqlite import list_signals, record_signal, status_snapshot
from pfe_core.pipeline import PipelineService


class SignalStorageEventChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.home = Path(self.tempdir.name) / ".pfe"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_record_signal_preserves_event_chain_fields(self) -> None:
        payload = {
            "event_id": "evt-1",
            "source_event_id": "evt-src-1",
            "request_id": "req-1",
            "session_id": "sess-1",
            "parent_event_id": "evt-src-0",
            "source_event_ids": ["evt-src-0", "evt-src-1", "evt-1"],
            "event_chain_ids": ["evt-src-0", "evt-src-1", "evt-1"],
            "adapter_version": "20260325-001",
            "event_type": "accept",
            "timestamp": "2026-03-25T07:00:00+00:00",
            "context": "user context",
            "model_output": "assistant output",
            "user_input": "user input",
            "action_detail": {"edited": False},
            "user_action": {"type": "accept", "final_text": "user input"},
            "scenario": "life-coach",
            "metadata": {"source": "chat", "note": "lineage"},
        }

        record_signal(payload, home=self.home)

        signals = list_signals(home=self.home)
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal["request_id"], "req-1")
        self.assertEqual(signal["session_id"], "sess-1")
        self.assertEqual(signal["parent_event_id"], "evt-src-0")
        self.assertEqual(signal["source_event_ids"], ["evt-src-0", "evt-src-1", "evt-1"])
        self.assertEqual(signal["event_chain_ids"], ["evt-src-0", "evt-src-1", "evt-1"])
        self.assertEqual(signal["lineage"]["request_id"], "req-1")
        self.assertEqual(signal["lineage"]["session_id"], "sess-1")
        self.assertEqual(signal["lineage"]["source_event_ids"], ["evt-src-0", "evt-src-1", "evt-1"])
        self.assertEqual(signal["lineage"]["event_chain_ids"], ["evt-src-0", "evt-src-1", "evt-1"])
        self.assertEqual(signal["metadata"]["note"], "lineage")

        snapshot = status_snapshot(home=self.home)
        self.assertEqual(snapshot["signal_count"], 1)
        self.assertEqual(snapshot["latest_signal"]["request_id"], "req-1")
        self.assertEqual(snapshot["latest_signal"]["source_event_ids"], ["evt-src-0", "evt-src-1", "evt-1"])

    def test_record_signal_backfills_chain_when_missing_explicit_list(self) -> None:
        payload = {
            "event_id": "evt-2",
            "source_event_id": "evt-src-2",
            "request_id": "req-2",
            "session_id": "sess-2",
            "adapter_version": "20260325-001",
            "event_type": "accept",
            "timestamp": "2026-03-25T07:05:00+00:00",
        }

        record_signal(payload, home=self.home)

        signals = list_signals(home=self.home)
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal["source_event_ids"], ["evt-src-2"])
        self.assertEqual(signal["event_chain_ids"], ["evt-src-2"])
        self.assertEqual(signal["lineage"]["source_event_ids"], ["evt-src-2"])
        self.assertEqual(signal["lineage"]["event_chain_ids"], ["evt-src-2"])

    def test_pipeline_signal_preserves_event_chain_ids(self) -> None:
        previous_home = os.environ.get("PFE_HOME")
        os.environ["PFE_HOME"] = str(self.home)
        try:
            pipeline = PipelineService()
            result = pipeline.signal(
                {
                    "event_id": "evt-3",
                    "source_event_id": "evt-src-3",
                    "event_chain_ids": ["evt-src-3", "evt-3"],
                    "request_id": "req-3",
                    "session_id": "sess-3",
                    "adapter_version": "20260325-001",
                    "event_type": "accept",
                    "timestamp": "2026-03-25T07:10:00+00:00",
                    "context": "user context",
                    "model_output": "assistant output",
                    "user_action": {"type": "accept", "final_text": "user context"},
                }
            )
        finally:
            if previous_home is None:
                os.environ.pop("PFE_HOME", None)
            else:
                os.environ["PFE_HOME"] = previous_home

        self.assertGreaterEqual(result["curated_samples"], 1)
        signal = list_signals(home=self.home, limit=1)[0]
        self.assertEqual(signal["source_event_ids"], ["evt-src-3", "evt-3"])
        self.assertEqual(signal["event_chain_ids"], ["evt-src-3", "evt-3"])
        self.assertEqual(signal["lineage"]["event_chain_ids"], ["evt-src-3", "evt-3"])


if __name__ == "__main__":
    unittest.main()
