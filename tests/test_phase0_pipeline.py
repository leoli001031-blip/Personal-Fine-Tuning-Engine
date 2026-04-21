from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService, _eval_generation_kwargs


@pytest.mark.slow
class Phase0PipelineTests(unittest.TestCase):
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

    def test_phase0_generate_train_eval_and_promote(self) -> None:
        service = PipelineService()
        summary = service.generate(scenario="life-coach", style="温和、共情", num_samples=12)
        self.assertIn("teacher_model=local_template_teacher", summary)

        train_result = service.train(method="qlora", epochs=1, train_type="sft")
        self.assertIn("State=pending_eval", train_result)

        report = json.loads(service.evaluate(base_model="base", adapter="latest", num_samples=3))
        self.assertGreaterEqual(report["num_test_samples"], 1)
        self.assertGreaterEqual(report["scores"]["quality_preservation"], 0.8)

        store = AdapterStore(home=self.pfe_home)
        records = store.list_version_records(limit=1)
        self.assertEqual(len(records), 1)
        version = records[0]["version"]
        version_dir = self.pfe_home / "adapters" / "user_default" / version
        self.assertTrue((version_dir / "adapter_manifest.json").exists())
        self.assertTrue((version_dir / "eval_report.json").exists())
        self.assertIsNone(store.current_latest_version())

        promote_result = store.promote(version)
        self.assertIn(version, promote_result)
        self.assertEqual(store.current_latest_version(), version)

    def test_latest_is_only_updated_via_promote(self) -> None:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=8)
        service.train(method="qlora", epochs=1, train_type="sft")

        store = AdapterStore(home=self.pfe_home)
        self.assertEqual(len(store.list_version_records(limit=1)), 1)
        self.assertIsNone(store.current_latest_version())

    def test_status_distinguishes_recent_training_from_latest_promoted(self) -> None:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和、共情", num_samples=12)
        train_result = service.train_result(method="qlora", epochs=1, train_type="sft")

        store = AdapterStore(home=self.pfe_home)
        records = store.list_version_records(limit=1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["version"], train_result.version)
        version_dir = self.pfe_home / "adapters" / "user_default" / train_result.version
        pending_manifest = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(pending_manifest["state"], "pending_eval")
        self.assertIsNone(store.current_latest_version())

        service.evaluate(base_model="base", adapter=train_result.version, num_samples=3)
        promoted = AdapterStore(home=self.pfe_home)
        promoted_result = promoted.promote(train_result.version)
        self.assertIn(train_result.version, promoted_result)

        promoted_manifest = json.loads((version_dir / "adapter_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(promoted_manifest["state"], "promoted")
        self.assertIn("promoted_at", promoted_manifest)
        self.assertEqual(promoted.current_latest_version(), train_result.version)

    def test_dpo_training_requires_explicit_rejected_pairs(self) -> None:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和", num_samples=10)
        result = service.train_result(method="qlora", epochs=1, train_type="dpo")
        self.assertGreaterEqual(result.num_samples, 1)
        self.assertEqual(result.metrics["train_type"], "dpo")

    def test_evaluate_uses_compact_generation_settings(self) -> None:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和、共情", num_samples=12)
        train_result = service.train_result(method="qlora", epochs=1, train_type="sft")

        calls: list[dict[str, object]] = []

        def fake_generate(_messages, **kwargs):
            calls.append(dict(kwargs))
            return "stub-response"

        with patch("pfe_core.pipeline.InferenceEngine.generate", side_effect=fake_generate):
            report = json.loads(service.evaluate(base_model="base", adapter=train_result.version, num_samples=1))

        self.assertEqual(report["num_test_samples"], 1)
        self.assertTrue(calls)
        self.assertTrue(all(call.get("max_tokens") == 32 for call in calls))
        self.assertTrue(all(call.get("temperature") == 0.0 for call in calls))
        self.assertTrue(all((call.get("metadata") or {}).get("source") == "pfe-eval" for call in calls))

    def test_evaluate_reuses_shared_runtime_when_real_local_enabled(self) -> None:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和、共情", num_samples=12)
        train_result = service.train_result(method="qlora", epochs=1, train_type="sft")

        fake_bundle = {
            "tokenizer": object(),
            "model": object(),
            "torch": object(),
            "device": "cpu",
            "resolved_base_model": "local-base",
            "adapter_loaded": False,
            "adapter_path": None,
            "adapter_reason": None,
        }
        calls: list[tuple[str, object]] = []

        def fake_load_bundle(*, resolved_base_model, adapter_path=None):
            calls.append(("load", resolved_base_model))
            return fake_bundle

        def fake_attach(bundle, *, adapter_path):
            calls.append(("attach", adapter_path))
            adapted = dict(bundle)
            adapted["adapter_loaded"] = True
            adapted["adapter_path"] = adapter_path
            return adapted

        def fake_generate_response(_messages, *, runtime_bundle=None, **_kwargs):
            if runtime_bundle and runtime_bundle.get("adapter_loaded"):
                return {"text": "adapted"}
            return {"text": "base"}

        with patch.dict(os.environ, {"PFE_ENABLE_REAL_LOCAL_INFERENCE": "1"}, clear=False):
            with patch("pfe_core.pipeline.InferenceEngine._load_uncached_runtime_bundle", side_effect=fake_load_bundle):
                with patch("pfe_core.pipeline.InferenceEngine._attach_adapter_to_runtime_bundle", side_effect=fake_attach):
                    with patch("pfe_core.pipeline.InferenceEngine._generate_real_response", side_effect=fake_generate_response):
                        report = json.loads(
                            service.evaluate(
                                base_model="base",
                                adapter=train_result.version,
                                num_samples=1,
                            )
                        )

        self.assertEqual(report["num_test_samples"], 1)
        self.assertEqual(calls.count(("load", "base")), 0)
        self.assertEqual(sum(1 for kind, _value in calls if kind == "load"), 1)
        self.assertEqual(sum(1 for kind, _value in calls if kind == "attach"), 1)

    def test_eval_generation_kwargs_allows_single_token_smoke_override(self) -> None:
        with patch.dict(os.environ, {"PFE_EVAL_MAX_TOKENS": "1"}, clear=False):
            payload = _eval_generation_kwargs()
        self.assertEqual(payload["max_tokens"], 1)
        self.assertEqual(payload["temperature"], 0.0)

    def test_distill_with_real_teacher_model_attempts_client(self) -> None:
        service = PipelineService()
        from pfe_core.curator.teacher_client import TeacherInferenceClient

        with patch.object(TeacherInferenceClient, "generate", return_value={"text": "fake-local-teacher-output", "backend": "local", "model": "fake", "usage": {}}):
            result = service.distill(
                teacher_model="fake-local-model",
                scenario="life-coach",
                style="温和、共情",
                num_samples=4,
            )
        self.assertIn("teacher_model=fake-local-model", result)


if __name__ == "__main__":
    unittest.main()
