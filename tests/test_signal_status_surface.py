from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli import main as cli_main
from pfe_cli.main import _format_status
from pfe_core.pipeline import PipelineService
from pfe_server.app import build_serve_plan, smoke_test_request
from tests.matrix_test_compat import strip_ansi


class SignalStatusSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("PFE_HOME")
        self.pfe_home = Path(self.tempdir.name) / ".pfe"
        os.environ["PFE_HOME"] = str(self.pfe_home)

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = self.previous_home
        self.tempdir.cleanup()

    def test_cli_status_surfaces_signal_readiness_and_sample_details(self) -> None:
        service = PipelineService()
        service.signal(
            {
                "event_id": "evt-signal-status-1",
                "request_id": "req-signal-status-1",
                "session_id": "sess-signal-status-1",
                "source_event_id": "evt-source-status-1",
                "source_event_ids": ["evt-source-status-1", "evt-signal-status-1"],
                "event_type": "accept",
                "user_input": "我今天有点焦虑",
                "model_output": "我们先把事情拆小，一步一步来。",
                "user_action": {"type": "accept"},
                "metadata": {"scenario": "life-coach"},
            }
        )

        text = _format_status(service.status(), workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ SIGNAL READINESS ]", clean)
        self.assertIn("state:", clean)
        self.assertIn("READY", clean)
        self.assertIn("event chain ready:", clean)
        self.assertIn("[ ADAPTER LIFECYCLE ]", clean)
        self.assertIn("samples:", clean)
        self.assertIn("train=1", clean)

    def test_server_status_surfaces_signal_readiness_and_counts(self) -> None:
        plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
        app = plan.app
        request_payload = {
            "event_id": "evt-signal-status-2",
            "request_id": "req-signal-status-2",
            "session_id": "sess-signal-status-2",
            "source_event_id": "evt-source-status-2",
            "source_event_ids": ["evt-source-status-2", "evt-signal-status-2"],
            "event_type": "accept",
            "user_input": "我有点紧张",
            "model_output": "先放慢一点，我们把压力拆开看。",
            "user_action": {"type": "accept"},
            "metadata": {"scenario": "life-coach"},
        }

        async def scenario() -> dict[str, object]:
            await smoke_test_request(app, path="/pfe/signal", method="POST", body=request_payload)
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        self.assertIn("signal_summary", body)
        self.assertIn("signal_sample_counts", body)
        self.assertIn("signal_sample_details", body)
        self.assertGreaterEqual(body["signal_count"], 1)
        self.assertGreaterEqual(body["signal_sample_count"], 1)
        self.assertTrue(body["signal_summary"]["event_chain_ready"])
        self.assertGreaterEqual(body["signal_sample_counts"]["train"], 1)
        self.assertGreaterEqual(len(body["signal_sample_details"]), 1)

    def test_status_surfaces_signal_quality_filter_counts_and_reasons(self) -> None:
        service = PipelineService()
        service.signal(
            {
                "event_id": "evt-signal-quality-1",
                "request_id": "req-signal-quality-1",
                "session_id": "sess-signal-quality-1",
                "source_event_id": "evt-source-quality-1",
                "source_event_ids": ["evt-source-quality-1", "evt-signal-quality-1"],
                "event_type": "accept",
                "user_input": "我需要一个明确的计划",
                "model_output": "",
                "user_action": {"type": "accept"},
                "metadata": {"scenario": "work-coach"},
            }
        )

        snapshot = service.status()
        signal_summary = snapshot["signal_summary"]
        quality_summary = snapshot["signal_quality_summary"]
        text = _format_status(snapshot, workspace="user_default")
        self.assertEqual(signal_summary["quality_filter_state"], "blocked")
        self.assertEqual(signal_summary["quality_filtered_count"], 1)
        self.assertEqual(quality_summary["filtered_count"], 1)
        self.assertIn("missing_model_output", quality_summary["filtered_reasons"])
        clean = strip_ansi(text)
        self.assertIn("[ SIGNAL READINESS ]", clean)
        self.assertIn("quality filter:", clean)
        self.assertIn("rejected=1", clean)
        self.assertIn("missing model output", clean)

    def test_collect_stop_disables_signal_ingest_until_reenabled(self) -> None:
        service = PipelineService()
        stopped = service.stop_signal_collection()
        self.assertFalse(stopped["signal_summary"]["collection_enabled"])
        self.assertEqual(stopped["signal_summary"]["state"], "disabled")

        disabled_result = service.signal(
            {
                "event_id": "evt-signal-disabled-1",
                "request_id": "req-signal-disabled-1",
                "session_id": "sess-signal-disabled-1",
                "source_event_id": "evt-source-disabled-1",
                "source_event_ids": ["evt-source-disabled-1", "evt-signal-disabled-1"],
                "event_type": "accept",
                "user_input": "先不要记录",
                "model_output": "这条应该被跳过。",
                "user_action": {"type": "accept"},
            }
        )
        self.assertFalse(disabled_result["recorded"])
        self.assertEqual(disabled_result["curation_state"], "disabled")

        text = _format_status(service.status(), workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ SIGNAL READINESS ]", clean)
        self.assertIn("[ DISABLED ]", clean)

        started = service.start_signal_collection()
        self.assertTrue(started["signal_summary"]["collection_enabled"])
        self.assertEqual(started["signal_summary"]["state"], "idle")

    def test_collect_cli_start_and_stop_toggle_signal_collection(self) -> None:
        runner = CliRunner()

        stopped = runner.invoke(cli_main.app, ["collect", "stop"])
        self.assertEqual(stopped.exit_code, 0, stopped.stdout)
        stopped_clean = strip_ansi(stopped.stdout)
        self.assertIn("[ DISABLED ]", stopped_clean)

        started = runner.invoke(cli_main.app, ["collect", "start"])
        self.assertEqual(started.exit_code, 0, started.stdout)
        started_clean = strip_ansi(started.stdout)
        self.assertIn("[ IDLE ]", started_clean)


if __name__ == "__main__":
    unittest.main()
