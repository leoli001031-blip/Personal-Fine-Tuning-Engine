from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from pfe_core.phase28_real_feedback_loop_engineering import build_phase28_task_pack
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


def _valid_body() -> dict:
    task = build_phase28_task_pack(count=1)["tasks"][0]
    return {
        "items": [
            {
                "task_id": task["task_id"],
                "collection_id": task["collection_id"],
                "scenario_id": task["scenario_id"],
                "prompt": task["user_prompt"],
                "messages": task["messages"],
                "runtime_output": task["runtime_output"],
                "response_under_review": task["runtime_output"],
                "metadata": task["source_metadata"],
                "feedback_source": "actual_user_feedback",
                "feedback": {
                    "action": "correction",
                    "edited_text": task["suggested_target_template"],
                    "user_feedback": "真实用户确认：这版可进入人工候选审阅。",
                    "signal_id": "phase28-api-signal-001",
                },
                "attestation": {
                    "operator_id": "human-reviewer-001",
                    "capture_method": "phase28_api",
                    "captured_at": "2026-06-21T10:00:00+08:00",
                    "confirmed_actual_user_feedback": True,
                    "not_scripted_or_curated": True,
                    "consent_for_training_candidate_review": True,
                },
                "request_id": "phase28-api-request",
                "session_id": "phase28-api-session",
            }
        ]
    }


class Phase28RealFeedbackLoopSurfaceTests(unittest.TestCase):
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
                workspace="phase28-surface",
            )
        )

    def test_phase28_task_pack_surface(self) -> None:
        response = asyncio.run(smoke_test_request(self._app(), "/pfe/phase28/task-pack"))

        self.assertEqual(response["status_code"], 200)
        body = response["body"]
        self.assertEqual(body["kind"], "phase28_task_pack_surface")
        self.assertEqual(body["task_pack"]["task_count"], 36)
        self.assertTrue(body["task_pack"]["template_not_training_data"])
        self.assertFalse(body["auto_promotion_allowed"])

    def test_phase28_import_review_loop_and_readiness_surfaces(self) -> None:
        app = self._app()
        import_response = asyncio.run(
            smoke_test_request(
                app,
                "/pfe/phase28/feedback-batch",
                method="POST",
                body=_valid_body(),
            )
        )
        self.assertEqual(import_response["status_code"], 200)
        self.assertEqual(import_response["body"]["batch"]["accepted_pending_review_count"], 1)

        queue_response = asyncio.run(smoke_test_request(app, "/pfe/phase28/review-queue"))
        self.assertEqual(queue_response["status_code"], 200)
        self.assertEqual(queue_response["body"]["review_state"]["pending_review_count"], 1)

        loop_response = asyncio.run(smoke_test_request(app, "/pfe/phase28/loop-state"))
        self.assertEqual(loop_response["status_code"], 200)
        self.assertEqual(loop_response["body"]["loop_state"]["current_state"], "review")

        decision_response = asyncio.run(
            smoke_test_request(
                app,
                "/pfe/phase28/review-decisions",
                method="POST",
                body={
                    "signal_id": "phase28-api-signal-001",
                    "state": "approved_for_candidate",
                    "reason": "passes four-section citation boundary",
                    "reviewer_id": "reviewer-001",
                },
            )
        )
        self.assertEqual(decision_response["status_code"], 200)
        self.assertEqual(
            decision_response["body"]["review_state"]["approved_for_candidate_count"],
            1,
        )

        readiness_response = asyncio.run(smoke_test_request(app, "/pfe/phase28/training-readiness"))
        self.assertEqual(readiness_response["status_code"], 200)
        self.assertEqual(readiness_response["body"]["actual_feedback_count"], 1)
        self.assertEqual(readiness_response["body"]["training_readiness"]["status"], "collect_actual_feedback")
        self.assertEqual(readiness_response["body"]["training_attempt"]["status"], "blocked")
        self.assertIn(
            "insufficient_approved_actual_user_feedback",
            readiness_response["body"]["training_readiness"]["blockers"],
        )
        self.assertFalse(readiness_response["body"]["auto_promotion_allowed"])

    def test_phase28_import_blocks_simulation_payload(self) -> None:
        task = build_phase28_task_pack(count=1)["tasks"][0]
        response = asyncio.run(
            smoke_test_request(
                self._app(),
                "/pfe/phase28/feedback-batch",
                method="POST",
                body={
                    "items": [
                        {
                            "task_id": task["task_id"],
                            "prompt": task["user_prompt"],
                            "metadata": {**task["source_metadata"], "simulation_only": True},
                            "feedback_source": "actual_user_feedback",
                            "feedback": {
                                "action": "correction",
                                "edited_text": task["suggested_target_template"],
                                "signal_id": "phase28-sim",
                                "user_feedback": "SIMULATION ONLY",
                            },
                            "attestation": {
                                "operator_id": "simulation-reviewer",
                                "capture_method": "phase27_workflow_simulation",
                                "captured_at": "2026-06-21T10:00:00+08:00",
                                "confirmed_actual_user_feedback": True,
                                "not_scripted_or_curated": True,
                                "consent_for_training_candidate_review": True,
                                "simulation_only": True,
                            },
                            "simulation_only": True,
                        }
                    ]
                },
            )
        )

        self.assertEqual(response["status_code"], 422)
        self.assertEqual(response["body"]["batch"]["non_training_count"], 1)
        self.assertFalse(response["body"]["auto_promotion_allowed"])
