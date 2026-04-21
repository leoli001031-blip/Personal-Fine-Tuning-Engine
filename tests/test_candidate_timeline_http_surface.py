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

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class CandidateTimelineHttpSurfaceTests(unittest.TestCase):
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

    def _build_candidate_timeline(self) -> tuple[PipelineService, str, str]:
        service = self._service()
        service.generate(scenario="life-coach", style="warm", num_samples=8)
        first = service.train_result(method="qlora", epochs=1, train_type="sft")
        AdapterStore(home=self.pfe_home).promote(first.version)
        service.generate(scenario="work-coach", style="direct", num_samples=8)
        second = service.train_result(method="qlora", epochs=1, train_type="sft")
        service.promote_candidate(note="ready_for_rollout")
        service.archive_candidate(note="archive_after_review")
        return service, first.version, second.version

    def test_status_exposes_candidate_timeline_summary(self) -> None:
        service, archived_version, promoted_version = self._build_candidate_timeline()

        status = service.status()
        timeline = status["candidate_timeline"]

        self.assertEqual(timeline["current_stage"], "archived")
        self.assertGreaterEqual(timeline["transition_count"], 2)
        self.assertEqual(timeline["last_candidate_version"], archived_version)
        self.assertEqual(timeline["last_reason"], "candidate_archived")
        self.assertIsNotNone(timeline["latest_timestamp"])
        self.assertEqual(timeline["last_transition"]["action"], "archive_candidate")
        self.assertEqual(timeline["last_transition"]["archived_version"], archived_version)
        self.assertEqual(status["candidate_summary"]["latest_promoted_version"], promoted_version)

    def test_http_candidate_timeline_endpoint_returns_timeline_items(self) -> None:
        service, archived_version, promoted_version = self._build_candidate_timeline()
        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/candidate/timeline", method="GET", query_params={"limit": "10"})
            return result["body"]

        body = asyncio.run(scenario())
        self.assertEqual(body["current_stage"], "archived")
        self.assertGreaterEqual(body["count"], 2)
        self.assertEqual(body["last_candidate_version"], archived_version)
        self.assertEqual(body["items"][-1]["action"], "archive_candidate")
        self.assertEqual(body["items"][-1]["stage"], "archived")
        self.assertEqual(body["items"][-2]["stage"], "promoted")
        self.assertEqual(body["items"][-2]["candidate_version"], promoted_version)

    def test_http_status_embeds_candidate_timeline(self) -> None:
        service, archived_version, _promoted_version = self._build_candidate_timeline()
        app = self._app(service)

        async def scenario() -> dict[str, object]:
            result = await smoke_test_request(app, path="/pfe/status", method="GET")
            return result["body"]

        body = asyncio.run(scenario())
        self.assertEqual(body["candidate_timeline"]["current_stage"], "archived")
        self.assertEqual(body["candidate_timeline"]["last_candidate_version"], archived_version)
        self.assertEqual(body["metadata"]["candidate_timeline"]["current_stage"], "archived")
        self.assertEqual(body["metadata"]["pipeline"]["candidate_timeline"]["current_stage"], "archived")


if __name__ == "__main__":
    unittest.main()
