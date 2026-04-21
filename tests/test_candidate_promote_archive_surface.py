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

import pfe_cli.main as cli_main
from pfe_cli.main import _format_status
from pfe_cli.adapter_commands import adapter_app
from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService
from pfe_server.app import build_serve_plan, smoke_test_request
from tests.matrix_test_compat import strip_ansi


class CandidatePromoteArchiveSurfaceTests(unittest.TestCase):
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

    def _build_promoted_and_candidate(self) -> tuple[PipelineService, str, str]:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        first_result = service.train_result(method="qlora", epochs=1, train_type="sft")
        AdapterStore(home=self.pfe_home).promote(first_result.version)

        service.generate(scenario="work-coach", style="direct", num_samples=8)
        second_result = service.train_result(method="qlora", epochs=1, train_type="sft")
        return service, first_result.version, second_result.version

    def test_cli_status_shows_archive_after_promoting_candidate(self) -> None:
        service, promoted_version, candidate_version = self._build_promoted_and_candidate()
        store = AdapterStore(home=self.pfe_home)
        store.promote(candidate_version)

        status = service.status()
        candidate_summary = status["candidate_summary"]
        lifecycle = status["adapter_lifecycle"]

        self.assertEqual(status["latest_adapter_version"], candidate_version)
        self.assertEqual(status["latest_adapter"]["state"], "promoted")
        self.assertEqual(candidate_summary["latest_promoted_version"], candidate_version)
        self.assertEqual(candidate_summary["recent_version"], candidate_version)
        self.assertEqual(candidate_summary["candidate_version"], promoted_version)
        self.assertEqual(candidate_summary["candidate_state"], "archived")
        self.assertFalse(candidate_summary["candidate_needs_promotion"])
        self.assertEqual(lifecycle["counts"]["promoted"], 1)
        self.assertEqual(lifecycle["counts"]["archived"], 1)
        self.assertEqual(lifecycle["promoted_versions"], [candidate_version])
        self.assertIn(promoted_version, lifecycle["archived_versions"])

        text = _format_status(status, workspace="user_default")
        clean = strip_ansi(text)
        self.assertIn("[ ADAPTER LIFECYCLE ]", clean)
        self.assertIn(f"latest promoted:         {candidate_version} | state=promoted", clean)
        self.assertIn("[ CANDIDATE SUMMARY ]", clean)
        self.assertIn(f"candidate version:       {promoted_version}", clean)
        self.assertIn("candidate state:         archived", clean)
        self.assertIn("lifecycle counts:        promoted=1 | archived=1", clean)

    def test_adapter_promote_cli_echoes_candidate_action_and_keeps_archive_distinct(self) -> None:
        service, promoted_version, candidate_version = self._build_promoted_and_candidate()
        runner = CliRunner()

        result = runner.invoke(
            adapter_app,
            ["promote", candidate_version, "--workspace", "user_default"],
        )
        clean = strip_ansi(result.stdout)

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn(f"latest: {candidate_version}", clean)
        self.assertIn("lifecycle: promoted", clean)

        status = service.status()
        candidate_summary = status["candidate_summary"]
        lifecycle = status["adapter_lifecycle"]

        self.assertEqual(status["latest_adapter_version"], candidate_version)
        self.assertEqual(status["latest_adapter"]["state"], "promoted")
        self.assertEqual(candidate_summary["latest_promoted_version"], candidate_version)
        self.assertEqual(candidate_summary["recent_version"], candidate_version)
        self.assertEqual(candidate_summary["candidate_version"], promoted_version)
        self.assertEqual(candidate_summary["candidate_state"], "archived")
        self.assertFalse(candidate_summary["candidate_needs_promotion"])
        self.assertEqual(lifecycle["counts"]["promoted"], 1)
        self.assertEqual(lifecycle["counts"]["archived"], 1)
        self.assertIn(promoted_version, lifecycle["archived_versions"])

    def test_http_status_shows_archive_after_promoting_candidate(self) -> None:
        service, promoted_version, candidate_version = self._build_promoted_and_candidate()
        store = AdapterStore(home=self.pfe_home)
        store.promote(candidate_version)

        plan = build_serve_plan(workspace="user_default", dry_run=False)
        app = plan.app

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        candidate_summary = body["candidate_summary"]
        lifecycle = body["metadata"]["snapshot"]["adapter_lifecycle"]

        self.assertEqual(body["latest_adapter"]["version"], candidate_version)
        self.assertEqual(body["latest_adapter"]["state"], "promoted")
        self.assertEqual(candidate_summary["latest_promoted_version"], candidate_version)
        self.assertEqual(candidate_summary["candidate_version"], promoted_version)
        self.assertEqual(candidate_summary["candidate_state"], "archived")
        self.assertFalse(candidate_summary["candidate_needs_promotion"])
        self.assertEqual(lifecycle["counts"]["promoted"], 1)
        self.assertEqual(lifecycle["counts"]["archived"], 1)
        self.assertEqual(body["metadata"]["candidate_summary"]["candidate_version"], promoted_version)
        self.assertEqual(body["metadata"]["candidate_summary"]["candidate_state"], "archived")

    def test_candidate_promote_cli_surfaces_candidate_action(self) -> None:
        service, promoted_version, candidate_version = self._build_promoted_and_candidate()
        runner = CliRunner()

        result = runner.invoke(
            cli_main.app,
            ["candidate", "promote", "--workspace", "user_default"],
        )
        clean = strip_ansi(result.stdout)

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("[ CANDIDATE ACTION ]", clean)
        self.assertIn("action:                  promote_candidate", clean)
        self.assertIn("status:                  completed", clean)
        self.assertIn(f"candidate version:       {candidate_version}", clean)
        self.assertIn(f"promoted version:        {candidate_version}", clean)

        status = service.status()
        self.assertEqual(status["latest_adapter_version"], candidate_version)
        self.assertEqual(status["candidate_action"]["action"], "promote_candidate")
        self.assertEqual(status["candidate_action"]["status"], "completed")
        self.assertEqual(status["candidate_action"]["promoted_version"], candidate_version)
        self.assertEqual(status["candidate_history"]["last_action"], "promote_candidate")
        self.assertEqual(status["candidate_history"]["last_status"], "completed")
        self.assertGreaterEqual(status["candidate_history"]["count"], 1)
        self.assertEqual(status["candidate_summary"]["candidate_version"], promoted_version)
        self.assertEqual(status["candidate_summary"]["candidate_state"], "archived")

    def test_candidate_archive_http_surfaces_candidate_action(self) -> None:
        service, promoted_version, candidate_version = self._build_promoted_and_candidate()

        plan = build_serve_plan(workspace="user_default", dry_run=False)
        app = plan.app

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/candidate/archive", method="POST")
            return result["body"]

        body = asyncio.run(scenario())
        self.assertEqual(body["action"], "archive_candidate")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["candidate_version"], candidate_version)
        self.assertEqual(body["archived_version"], candidate_version)
        self.assertEqual(body["candidate_summary"]["candidate_version"], candidate_version)
        self.assertEqual(body["candidate_summary"]["candidate_state"], "archived")
        self.assertIn("candidate_action", body["metadata"])
        self.assertIn("candidate_history", body["metadata"])
        self.assertEqual(body["candidate_history"]["last_action"], "archive_candidate")

        status = service.status()
        self.assertEqual(status["latest_adapter_version"], promoted_version)
        self.assertEqual(status["candidate_action"]["action"], "archive_candidate")
        self.assertEqual(status["candidate_action"]["status"], "completed")
        self.assertEqual(status["candidate_history"]["last_action"], "archive_candidate")
        self.assertEqual(status["candidate_summary"]["candidate_version"], candidate_version)
        self.assertEqual(status["candidate_summary"]["candidate_state"], "archived")

    def test_frontend_surface_exposes_candidate_and_queue_controls(self) -> None:
        self._build_promoted_and_candidate()
        plan = build_serve_plan(workspace="user_default", dry_run=False)
        app = plan.app

        async def scenario() -> str:
            result = await smoke_test_request(app, path="/", method="GET")
            return result["text"]

        text = asyncio.run(scenario())
        self.assertIn("Candidate", text)
        self.assertIn("Candidate Action", text)
        self.assertIn("Train Queue", text)
        self.assertIn("Promote Candidate", text)
        self.assertIn("Archive Candidate", text)
        self.assertIn("处理下一条队列任务", text)
        self.assertIn("重试 Auto Train", text)
        self.assertIn("重置 Trigger 状态", text)


if __name__ == "__main__":
    unittest.main()
