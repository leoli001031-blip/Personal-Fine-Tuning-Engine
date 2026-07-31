from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from pfe_core.inference.contracts import normalize_boundary_contract_output
from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app, smoke_test_request
from pfe_server.auth import ServerSecurityConfig


class Phase23RuntimeContractLoopSurfaceTests(unittest.TestCase):
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

    def test_phase23_get_surfaces_runtime_contract_loop_summary(self) -> None:
        response = asyncio.run(smoke_test_request(self._app(), "/pfe/phase23/runtime-contract-loop"))

        self.assertEqual(response["status_code"], 200)
        body = response["body"]
        self.assertEqual(body["kind"], "phase23_runtime_contract_product_loop")
        self.assertIn(body["status"], {"ready", "blocked"})
        self.assertIn("runtime_contract", body)
        self.assertIn("training_candidate_plan", body)
        self.assertFalse(body["auto_promotion_allowed"])

    def test_phase23_post_runs_contract_output_and_captures_feedback_signal(self) -> None:
        edited = (
            "摘要：资料显示付款期限为发票日后三十日。\n"
            "风险提示：付款节点需核对；只做资料整理和风险提示，不判断合法/违法。\n"
            "引用依据：[phase23-api-source:phase23-api-chunk]\n"
            "人工确认：不输出法律结论，不能支持最终法律结论；需人工/法务结合完整材料确认。"
        )
        response = asyncio.run(
            smoke_test_request(
                self._app(),
                "/pfe/phase23/runtime-contract-loop",
                method="POST",
                body={
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "任务：请整理付款条款。\n"
                                "资料引用：[phase23-api-source:phase23-api-chunk]\n"
                                "资料摘录：资料说明客户需在发票日后三十日内付款。"
                            ),
                        }
                    ],
                    "metadata": {
                        "response_contract": "contract_boundary_summary",
                        "expected_citation": "[phase23-api-source:phase23-api-chunk]",
                        "source_excerpt": "资料说明客户需在发票日后三十日内付款。",
                    },
                    "feedback": {
                        "action": "correction",
                        "edited_text": edited,
                        "user_feedback": "修正后的四段式可作为候选。",
                    },
                    "request_id": "phase23-api-req",
                    "session_id": "phase23-api-session",
                },
            )
        )

        self.assertEqual(response["status_code"], 200)
        body = response["body"]
        runtime = body["runtime_contract"]
        self.assertTrue(normalize_boundary_contract_output(runtime["output"])["complete"])
        self.assertEqual(runtime["scores"]["external_law_reference_rate"], 0.0)
        self.assertEqual(runtime["scores"]["think_leak_rate"], 0.0)
        signal = body["signal"]
        self.assertTrue(signal["phase23_route"]["eligible_for_training"])
        self.assertIn("training_candidate", signal["phase23_route"]["lanes"])
        self.assertFalse(body["auto_promotion_allowed"])
        self.assertIsNotNone(body["persisted_signal"])
