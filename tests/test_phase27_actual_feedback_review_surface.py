from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from pfe_core.phase27_actual_feedback_review_training_loop import build_phase27_collection_pack
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


def _valid_body() -> dict:
    item = build_phase27_collection_pack()["items"][0]
    return {
        "items": [
            {
                "collection_id": item["collection_id"],
                "prompt": item["prompt"],
                "messages": item["messages"],
                "runtime_output": item["runtime_output"],
                "response_under_review": item["runtime_output"],
                "metadata": item["metadata"],
                "feedback_source": "actual_user_feedback",
                "feedback": {
                    "action": "correction",
                    "edited_text": item["suggested_target_template"],
                    "user_feedback": "真实用户确认：这版可进入人工候选审阅。",
                    "signal_id": "phase27-api-signal-001",
                },
                "attestation": {
                    "operator_id": "human-reviewer-001",
                    "capture_method": "phase27_api",
                    "captured_at": "2026-06-21T10:00:00+08:00",
                    "confirmed_actual_user_feedback": True,
                    "not_scripted_or_curated": True,
                    "consent_for_training_candidate_review": True,
                },
                "request_id": "phase27-api-request",
                "session_id": "phase27-api-session",
            }
        ]
    }


class Phase27ActualFeedbackReviewSurfaceTests(unittest.TestCase):
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
                workspace="phase27-surface",
            )
        )

    def test_phase27_collection_pack_surface(self) -> None:
        response = asyncio.run(smoke_test_request(self._app(), "/pfe/phase27/collection-pack"))

        self.assertEqual(response["status_code"], 200)
        body = response["body"]
        self.assertEqual(body["kind"], "phase27_collection_pack_surface")
        self.assertEqual(body["collection_pack"]["collection_count"], 12)
        self.assertFalse(body["auto_promotion_allowed"])

    def test_phase27_import_review_and_readiness_surfaces(self) -> None:
        app = self._app()
        import_response = asyncio.run(
            smoke_test_request(
                app,
                "/pfe/phase27/actual-feedback-batch",
                method="POST",
                body=_valid_body(),
            )
        )
        self.assertEqual(import_response["status_code"], 200)
        self.assertEqual(import_response["body"]["batch"]["accepted_pending_review_count"], 1)

        queue_response = asyncio.run(smoke_test_request(app, "/pfe/phase27/review-queue"))
        self.assertEqual(queue_response["status_code"], 200)
        self.assertEqual(queue_response["body"]["review_state"]["pending_review_count"], 1)

        decision_response = asyncio.run(
            smoke_test_request(
                app,
                "/pfe/phase27/review-decisions",
                method="POST",
                body={
                    "signal_id": "phase27-api-signal-001",
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

        readiness_response = asyncio.run(smoke_test_request(app, "/pfe/phase27/training-readiness"))
        self.assertEqual(readiness_response["status_code"], 200)
        self.assertEqual(readiness_response["body"]["actual_feedback_count"], 1)
        self.assertEqual(readiness_response["body"]["training_readiness"]["status"], "collect_actual_feedback")
        self.assertIn(
            "insufficient_approved_actual_user_feedback",
            readiness_response["body"]["training_readiness"]["blockers"],
        )
        self.assertFalse(readiness_response["body"]["auto_promotion_allowed"])

    def test_phase27_import_blocks_template_payload(self) -> None:
        item = build_phase27_collection_pack()["items"][0]
        response = asyncio.run(
            smoke_test_request(
                self._app(),
                "/pfe/phase27/actual-feedback-batch",
                method="POST",
                body={
                    "items": [
                        {
                            "collection_id": item["collection_id"],
                            "prompt": item["prompt"],
                            "metadata": {**item["metadata"], "template_not_training_data": True},
                            "feedback_source": "template_feedback",
                            "feedback": {"action": "correction", "edited_text": "", "signal_id": "template"},
                            "attestation": {
                                "operator_id": "",
                                "capture_method": "phase27_template",
                                "captured_at": "",
                                "confirmed_actual_user_feedback": False,
                                "not_scripted_or_curated": False,
                                "consent_for_training_candidate_review": False,
                            },
                        }
                    ]
                },
            )
        )

        self.assertEqual(response["status_code"], 422)
        self.assertEqual(response["body"]["batch"]["non_training_count"], 1)
        self.assertFalse(response["body"]["auto_promotion_allowed"])
