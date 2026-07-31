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


class Phase21TrainingCandidateWorkbenchSurfaceTests(unittest.TestCase):
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

    def test_phase21_workbench_api_surfaces_guarded_candidate_plan_contract(self) -> None:
        service = PipelineService()
        app = create_app(
            ServiceBundle(
                inference=InferenceServiceAdapter(service),
                pipeline=PipelineServiceAdapter(service),
                security=ServerSecurityConfig(),
                provider="core",
                workspace=str(self.pfe_home),
            )
        )

        response = asyncio.run(smoke_test_request(app, "/pfe/phase21/training-candidate-workbench"))

        self.assertEqual(response["status_code"], 200)
        body = response["body"]
        self.assertEqual(body["kind"], "phase21_training_candidate_workbench")
        self.assertIn(body["status"], {"ready", "blocked"})
        self.assertIn("candidate_plan", body)
        plan = body["candidate_plan"]
        self.assertIn("preference_signal_count", plan)
        self.assertIn("trainable_candidate_count", plan)
        self.assertIn("holdout_isolation_status", plan)
        self.assertIn("sanity_gate_result", plan)
        self.assertIn("degeneration_report_summary", plan)
        self.assertIn("final_decision", plan)
        self.assertFalse(plan["auto_promotion_allowed"])
