from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from datetime import timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.pipeline import PipelineService
from pfe_core.server_services import InferenceServiceAdapter, PipelineServiceAdapter
from pfe_server.app import ServiceBundle, create_app
from pfe_server.auth import ServerSecurityConfig
from pfe_server.models import ChatCompletionRequest, DistillRunRequest, SignalIngestRequest


class ServerAdapterTests(unittest.TestCase):
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

    def test_server_adapters_share_core_pipeline(self) -> None:
        pipeline = PipelineService()
        bundle = ServiceBundle(
            inference=InferenceServiceAdapter(pipeline),
            pipeline=PipelineServiceAdapter(pipeline),
            security=ServerSecurityConfig(),
        )
        app = create_app(bundle)
        self.assertIsNotNone(app)

        async def scenario() -> None:
            inference = InferenceServiceAdapter(pipeline)
            workflow = PipelineServiceAdapter(pipeline)

            with patch.object(
                pipeline,
                "chat_completion",
                return_value={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "model": "local",
                    "adapter_version": None,
                    "request_id": "req-chat",
                    "session_id": "sess-chat",
                    "served_by": "local",
                    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "先深呼吸，我们一步一步来。"},
                            "finish_reason": "stop",
                        }
                    ],
                    "metadata": {
                        "inference": {
                            "backend": "transformers",
                            "healthy": True,
                            "served_by": "local",
                            "runtime_path": "real_local",
                        }
                    },
                },
            ):
                chat = await inference.generate_chat_completion(
                    ChatCompletionRequest(messages=[{"role": "user", "content": "我今天很焦虑"}], model="local")
                )
            self.assertTrue(chat.choices[0].message.content)
            self.assertEqual(chat.served_by, "local")

            signal = await workflow.ingest_signal(
                SignalIngestRequest(event_id="evt-1", request_id="req-1", session_id="sess-1", event_type="chat")
            )
            self.assertTrue(signal.stored)
            self.assertIn(signal.metadata["curation_state"], {"stored_only", "curated", "filtered"})
            self.assertIn("source_event_ids", signal.metadata)

            distill = await workflow.run_distillation(
                DistillRunRequest(scenario="life-coach", style="温和", num_samples=6)
            )
            self.assertEqual(distill.generated_samples, 6)

            status = await workflow.status()
            self.assertGreaterEqual(status["signal_count"], 1)
            self.assertGreaterEqual(status["sample_counts"]["train"], 1)

        asyncio.run(scenario())

    def test_signal_ingest_request_normalizes_naive_timestamp_to_utc(self) -> None:
        request = SignalIngestRequest(
            event_id="evt-naive",
            request_id="req-naive",
            session_id="sess-naive",
            event_type="chat",
            timestamp="2026-04-20T12:00:00",
        )

        self.assertEqual(request.timestamp.tzinfo, timezone.utc)
        self.assertEqual(request.timestamp.isoformat(), "2026-04-20T12:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
