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


def _valid_body() -> dict:
    citation = "[phase26-api-source:phase26-api-chunk]"
    excerpt = "资料说明客户需在发票日后三十日内付款。"
    prompt = (
        "任务：请整理付款义务相关摘要、风险提示、引用依据和人工确认项。\n"
        f"资料引用：{citation}\n"
        f"资料摘录：{excerpt}\n"
        "只基于给定资料回答，不输出法律结论。"
    )
    edited = (
        f"摘要：资料显示：{excerpt}\n"
        "风险提示：需核对资料完整性和附件位置；只做资料整理和风险提示，不判断合法/违法。\n"
        f"引用依据：{citation}\n"
        "人工确认：不输出法律结论，不能支持最终法律结论；需人工/法务结合完整材料确认。"
    )
    return {
        "items": [
            {
                "prompt": prompt,
                "metadata": {
                    "response_contract": "contract_boundary_summary",
                    "expected_citation": citation,
                    "source_excerpt": excerpt,
                },
                "feedback_source": "actual_user_feedback",
                "feedback": {
                    "action": "correction",
                    "edited_text": edited,
                    "user_feedback": "真实用户确认：这版可作为候选。",
                    "signal_id": "phase26-api-signal-001",
                },
                "attestation": {
                    "operator_id": "human-reviewer-001",
                    "capture_method": "phase26_api",
                    "captured_at": "2026-06-21T10:00:00+08:00",
                    "confirmed_actual_user_feedback": True,
                    "not_scripted_or_curated": True,
                    "consent_for_training_candidate_review": True,
                },
                "request_id": "phase26-api-request",
                "session_id": "phase26-api-session",
            }
        ]
    }


class Phase26ActualFeedbackCollectionSurfaceTests(unittest.TestCase):
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

    def test_phase26_feedback_collection_pack_surface(self) -> None:
        response = asyncio.run(smoke_test_request(self._app(), "/pfe/phase26/feedback-collection-pack"))

        self.assertEqual(response["status_code"], 200)
        body = response["body"]
        self.assertEqual(body["kind"], "phase26_feedback_collection_pack_surface")
        self.assertIn(body["status"], {"ready", "blocked"})
        self.assertIn("collection_pack", body)
        self.assertFalse(body["auto_promotion_allowed"])

    def test_phase26_actual_feedback_batch_accepts_attested_items(self) -> None:
        response = asyncio.run(
            smoke_test_request(
                self._app(),
                "/pfe/phase26/actual-feedback-batch",
                method="POST",
                body=_valid_body(),
            )
        )

        self.assertEqual(response["status_code"], 200)
        body = response["body"]
        self.assertEqual(body["kind"], "phase26_actual_feedback_batch_response")
        self.assertEqual(body["status"], "accepted_pending_review")
        self.assertEqual(body["batch"]["accepted_pending_review_count"], 1)
        self.assertEqual(body["batch"]["accepted_signals"][0]["feedback_source"], "actual_user_feedback")
        self.assertFalse(body["auto_promotion_allowed"])

    def test_phase26_training_probe_readiness_surface(self) -> None:
        response = asyncio.run(smoke_test_request(self._app(), "/pfe/phase26/training-probe-readiness"))

        self.assertEqual(response["status_code"], 200)
        body = response["body"]
        self.assertEqual(body["kind"], "phase26_training_probe_readiness_surface")
        self.assertIn(body["status"], {"ready", "blocked"})
        self.assertIn("comparison_summary", body)
        self.assertFalse(body["auto_promotion_allowed"])
