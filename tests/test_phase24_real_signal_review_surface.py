from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class Phase24RealSignalReviewSurfaceTests(unittest.TestCase):
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

    def _app(self):
        service = PipelineService()
        return create_app(
            ServiceBundle(
                inference=InferenceServiceAdapter(service),
                pipeline=PipelineServiceAdapter(service),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
            )
        )

    def test_phase24_review_queue_surface(self) -> None:
        response = asyncio.run(smoke_test_request(self._app(), "/pfe/phase24/review-queue"))

        self.assertEqual(response["status_code"], 200)
        body = response["body"]
        self.assertEqual(body["kind"], "phase24_review_queue_surface")
        self.assertIn(body["status"], {"ready", "blocked"})
        self.assertIn("queue", body)
        self.assertIn("review_summary", body)
        self.assertFalse(body["auto_promotion_allowed"])

    def test_phase24_training_candidate_value_surface(self) -> None:
        response = asyncio.run(smoke_test_request(self._app(), "/pfe/phase24/training-candidate-value"))

        self.assertEqual(response["status_code"], 200)
        body = response["body"]
        self.assertEqual(body["kind"], "phase24_training_candidate_value")
        self.assertIn(body["status"], {"ready", "blocked"})
        self.assertIn("comparison_summary", body)
        self.assertFalse(body["auto_promotion_allowed"])
