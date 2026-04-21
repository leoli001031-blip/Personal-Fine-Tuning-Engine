from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli.main import _format_status
from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService
from pfe_server.app import build_serve_plan, smoke_test_request
from tests.matrix_test_compat import strip_ansi


class CandidateSummarySurfaceTests(unittest.TestCase):
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

    def _build_promoted_and_pending_service(self) -> tuple[PipelineService, str, str]:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        first_result = service.train_result(method="qlora", epochs=1, train_type="sft")
        AdapterStore(home=self.pfe_home).promote(first_result.version)

        service.generate(scenario="work-coach", style="direct", num_samples=8)
        second_result = service.train_result(method="qlora", epochs=1, train_type="sft")
        return service, first_result.version, second_result.version

    def test_cli_status_keeps_latest_promoted_and_recent_candidate_distinct(self) -> None:
        service, promoted_version, pending_version = self._build_promoted_and_pending_service()

        status = service.status()
        candidate_summary = status["candidate_summary"]

        self.assertEqual(candidate_summary["latest_promoted_version"], promoted_version)
        self.assertEqual(candidate_summary["recent_version"], pending_version)
        self.assertEqual(candidate_summary["candidate_version"], pending_version)
        self.assertEqual(candidate_summary["candidate_state"], "pending_eval")
        self.assertTrue(candidate_summary["has_pending_candidate"])
        self.assertTrue(candidate_summary["candidate_needs_promotion"])
        self.assertEqual(candidate_summary["pending_eval_count"], 1)

        text = _format_status(status, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ ADAPTER LIFECYCLE ]", clean)
        self.assertIn(f"latest promoted:         {promoted_version} | state=promoted", clean)
        self.assertIn(f"recent training:         {pending_version} | state=pending_eval", clean)
        self.assertIn("lifecycle counts:        promoted=1 | pending_eval=1", clean)

    def test_server_status_surfaces_candidate_summary_and_snapshot(self) -> None:
        service, promoted_version, pending_version = self._build_promoted_and_pending_service()
        plan = build_serve_plan(workspace="user_default", dry_run=False)
        app = plan.app

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        candidate_summary = body["metadata"]["snapshot"]["candidate_summary"]

        self.assertEqual(body["latest_adapter"]["version"], promoted_version)
        self.assertEqual(candidate_summary["latest_promoted_version"], promoted_version)
        self.assertEqual(candidate_summary["recent_version"], pending_version)
        self.assertEqual(candidate_summary["candidate_state"], "pending_eval")
        self.assertTrue(candidate_summary["has_pending_candidate"])
        self.assertTrue(candidate_summary["candidate_needs_promotion"])
        self.assertEqual(body["metadata"]["snapshot"]["candidate_summary"]["candidate_version"], pending_version)
        self.assertEqual(body["metadata"]["snapshot"]["candidate_summary"]["latest_promoted_version"], promoted_version)
        self.assertEqual(body["metadata"]["candidate_summary"]["candidate_version"], pending_version)


if __name__ == "__main__":
    unittest.main()
