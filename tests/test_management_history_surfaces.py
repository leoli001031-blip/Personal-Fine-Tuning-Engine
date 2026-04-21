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
from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class ManagementHistorySurfaceTests(unittest.TestCase):
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

    def _service(self) -> PipelineService:
        return PipelineService()

    def _app(self, service: PipelineService):
        return create_app(
            ServiceBundle(
                inference=InferenceServiceAdapter(service),
                pipeline=PipelineServiceAdapter(service),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
            )
        )

    def _build_candidate_history(self) -> tuple[PipelineService, str]:
        service = self._service()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        first = service.train_result(method="qlora", epochs=1, train_type="sft")
        AdapterStore(home=self.pfe_home).promote(first.version)
        service.generate(scenario="work-coach", style="direct", num_samples=8)
        second = service.train_result(method="qlora", epochs=1, train_type="sft")
        service.promote_candidate(note="ready_for_rollout")
        return service, second.version

    def _build_queue_history(self) -> tuple[PipelineService, str]:
        service = self._service()
        service._append_train_queue_item(
            {
                "job_id": "job-history-1",
                "state": "queued",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": "20260325-801",
                "confirmation_required": True,
                "confirmation_reason": "operator_review_required",
            },
            workspace="user_default",
        )
        service._update_train_queue_item(
            "job-history-1",
            {
                "state": "awaiting_confirmation",
                "confirmation_required": True,
                "confirmation_reason": "operator_review_required",
                "updated_at": "2026-03-25T10:00:00+00:00",
                "history_event": "awaiting_confirmation",
                "history_reason": "operator_review_required",
            },
            workspace="user_default",
        )
        service.approve_next_train_queue(note="safe_to_run")
        service._update_train_queue_item(
            "job-history-1",
            {
                "state": "completed",
                "updated_at": "2026-03-25T10:01:00+00:00",
                "history_event": "completed",
                "history_reason": "training_completed",
            },
            workspace="user_default",
        )
        return service, "job-history-1"

    def test_candidate_history_cli_and_http_surface(self) -> None:
        service, candidate_version = self._build_candidate_history()
        runner = CliRunner()

        result = runner.invoke(cli_main.app, ["candidate", "history", "--workspace", "user_default", "--limit", "5"])
        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("PFE candidate history", result.stdout)
        self.assertIn("last_action=promote_candidate", result.stdout)
        self.assertIn(f"candidate_version={candidate_version}", result.stdout)

        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/candidate/history", method="GET", query_params={"limit": "5"})
            return result["body"]

        body = asyncio.run(scenario())
        self.assertEqual(body["last_action"], "promote_candidate")
        self.assertEqual(body["last_status"], "completed")
        self.assertGreaterEqual(body["count"], 1)
        self.assertEqual(body["items"][-1]["candidate_version"], candidate_version)
        self.assertEqual(body["last_note"], "ready_for_rollout")
        self.assertEqual(body["items"][-1]["operator_note"], "ready_for_rollout")

    def test_queue_history_cli_and_http_surface(self) -> None:
        service, job_id = self._build_queue_history()
        runner = CliRunner()

        result = runner.invoke(
            cli_main.app,
            ["trigger", "queue-history", "--workspace", "user_default", "--job-id", job_id, "--limit", "10"],
        )
        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("PFE train queue history", result.stdout)
        self.assertIn(f"job_id={job_id}", result.stdout)
        self.assertIn("event=approved", result.stdout)
        self.assertIn("event=completed", result.stdout)

        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(
                app,
                path="/pfe/auto-train/queue-history",
                method="GET",
                query_params={"job_id": job_id, "limit": "10"},
            )
            return result["body"]

        body = asyncio.run(scenario())
        self.assertEqual(body["job_id"], job_id)
        self.assertEqual(body["state"], "completed")
        self.assertGreaterEqual(body["history_count"], 4)
        self.assertEqual(body["history"][-1]["event"], "completed")
        self.assertEqual(body["history"][-2]["note"], "safe_to_run")
        self.assertIn("transition_count", body["history_summary"])

    def test_http_actions_accept_note_and_surface_review_summary(self) -> None:
        service, _ = self._build_candidate_history()
        app = self._app(service)

        async def candidate_scenario() -> dict[str, object]:
            result = await smoke_test_request(
                app,
                path="/pfe/candidate/archive",
                method="POST",
                query_params={"note": "archive_after_review"},
            )
            return result["body"]

        candidate_body = asyncio.run(candidate_scenario())
        self.assertEqual(candidate_body["operator_note"], "archive_after_review")
        self.assertEqual(candidate_body["candidate_history"]["last_note"], "archive_after_review")

        queue_service, _job_id = self._build_queue_history()
        queue_app = self._app(queue_service)
        queue_service._append_train_queue_item(
            {
                "job_id": "job-http-note-1",
                "state": "awaiting_confirmation",
                "workspace": "user_default",
                "source": "signal_auto_train",
                "adapter_version": "20260325-901",
                "confirmation_required": True,
                "confirmation_reason": "manual_review_required_by_policy",
            },
            workspace="user_default",
        )

        async def queue_scenario() -> dict[str, object]:
            result = await smoke_test_request(
                queue_app,
                path="/pfe/auto-train/approve-next",
                method="POST",
                query_params={"note": "looks_good"},
            )
            return result["body"]

        queue_body = asyncio.run(queue_scenario())
        self.assertEqual(queue_body["operator_note"], "looks_good")
        self.assertEqual(queue_body["review_summary"]["last_review_note"], "looks_good")

    def test_frontend_status_cards_expose_candidate_and_queue_history_labels(self) -> None:
        app = self._app(self._service())

        async def scenario() -> str:
            result = await smoke_test_request(app, path="/", method="GET")
            return result["text"]

        text = asyncio.run(scenario())
        self.assertIn("Operations", text)
        self.assertIn("Candidate History", text)
        self.assertIn("Queue History", text)
        self.assertIn("Worker Runner Timeline / History", text)


if __name__ == "__main__":
    unittest.main()
