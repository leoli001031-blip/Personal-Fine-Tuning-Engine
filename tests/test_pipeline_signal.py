from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.pipeline import PipelineService
from pfe_core.storage import list_samples


@pytest.mark.slow
class PipelineSignalTests(unittest.TestCase):
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

    def test_signal_creates_train_split_sft_sample(self) -> None:
        service = PipelineService()
        result = service.signal(
            {
                "event_id": "evt_signal_1",
                "request_id": "req_signal_1",
                "session_id": "sess_signal_1",
                "source_event_id": "evt_source_1",
                "source_event_ids": ["evt_source_1", "evt_signal_1"],
                "event_type": "accept",
                "user_input": "我今天很焦虑",
                "model_output": "先深呼吸，我们把压力拆成一件件小事。",
                "user_action": {"type": "accept"},
                "metadata": {"scenario": "life-coach"},
            }
        )
        self.assertEqual(result["curated_samples"], 1)
        self.assertEqual(result["curation_state"], "curated")
        self.assertEqual(result["curation_reason"], "curated_sft")
        self.assertTrue(result["event_chain_complete"])
        self.assertEqual(result["source_event_ids"], ["evt_source_1", "evt_signal_1"])
        self.assertEqual(len(result["curated_sample_ids"]), 1)

        signal_samples = [sample for sample in list_samples(dataset_split="train") if sample["source"] == "signal"]
        self.assertEqual(len(signal_samples), 1)
        self.assertEqual(signal_samples[0]["sample_type"], "sft")
        self.assertEqual(signal_samples[0]["metadata"]["dataset_split"], "train")
        self.assertEqual(signal_samples[0]["source_event_ids"], ["evt_source_1", "evt_signal_1"])

    def test_signal_reports_stored_only_when_event_chain_is_incomplete(self) -> None:
        service = PipelineService()
        result = service.signal(
            {
                "event_id": "evt_signal_2",
                "request_id": "req_signal_2",
                "session_id": "sess_signal_2",
                "event_type": "accept",
                "user_input": "请帮我整理今天的任务",
                "model_output": "先列出三个最重要的任务。",
                "user_action": {"type": "accept"},
                "metadata": {"scenario": "work-coach"},
            }
        )
        self.assertEqual(result["curated_samples"], 0)
        self.assertEqual(result["curation_state"], "filtered")
        self.assertFalse(result["event_chain_complete"])
        self.assertEqual(result["curation_reason"], "incomplete_event_chain")
        self.assertEqual(result["curation_detail"], "incomplete_event_chain")
        self.assertTrue(result["signal_quality_filtered"])
        self.assertEqual(result["signal_quality_reasons"], ["incomplete_event_chain"])
        self.assertEqual(result["source_event_ids"], ["evt_signal_2"])

    def test_signal_filters_low_quality_samples_before_curation(self) -> None:
        service = PipelineService()
        result = service.signal(
            {
                "event_id": "evt_signal_3",
                "request_id": "req_signal_3",
                "session_id": "sess_signal_3",
                "source_event_id": "evt_source_3",
                "source_event_ids": ["evt_source_3", "evt_signal_3"],
                "event_type": "accept",
                "user_input": "请帮我整理今天的任务",
                "model_output": "",
                "user_action": {"type": "accept"},
                "metadata": {"scenario": "work-coach"},
            }
        )
        self.assertTrue(result["recorded"])
        self.assertTrue(result["signal_quality_filtered"])
        self.assertEqual(result["curation_state"], "filtered")
        self.assertEqual(result["curation_reason"], "missing_model_output")
        self.assertEqual(result["signal_quality_reasons"], ["missing_model_output"])
        self.assertEqual(result["curated_samples"], 0)

    def test_signal_marks_explicit_response_preference_reinforcement_in_curated_sample(self) -> None:
        service = PipelineService()
        result = service.signal(
            {
                "event_id": "evt_signal_pref_1",
                "request_id": "req_signal_pref_1",
                "session_id": "sess_signal_pref_1",
                "source_event_id": "evt_source_pref_1",
                "source_event_ids": ["evt_source_pref_1", "evt_signal_pref_1"],
                "event_type": "accept",
                "user_input": "我希望你以后回答更温和、更鼓励式。",
                "model_output": "我能理解你现在的压力，我们可以先从最小的一步开始。",
                "user_action": {"type": "accept"},
                "metadata": {
                    "scenario": "life-coach",
                    "explicit_user_data_routing": {
                        "processed": True,
                        "candidates": [
                            {
                                "key": "回答风格偏好",
                                "value": "更温和、更鼓励式",
                                "kind": "response_preference",
                                "primary_lane": "profile",
                                "training_target": "preference_only",
                            }
                        ],
                    },
                },
            }
        )

        self.assertEqual(result["curated_samples"], 1)
        signal_samples = [sample for sample in list_samples(dataset_split="train") if sample["source"] == "signal"]
        self.assertEqual(len(signal_samples), 1)
        metadata = signal_samples[0]["metadata"]
        self.assertTrue(metadata["explicit_response_preference_reinforced"])
        self.assertEqual(metadata["training_signal_category"], "preference_reinforced")
        self.assertEqual(
            metadata["training_gate_reason"],
            "explicit_response_preference_reinforced_by_feedback",
        )
        self.assertEqual(metadata["explicit_response_preference_values"], ["更温和、更鼓励式"])

    def test_eval_report_contains_holdout_metadata(self) -> None:
        service = PipelineService()
        service.generate(scenario="life-coach", style="温和、共情", num_samples=12)
        service.train(method="qlora", epochs=1, train_type="sft")

        report = json.loads(service.evaluate(base_model="base", adapter="latest", num_samples=3))
        self.assertIn("metadata", report)
        self.assertEqual(report["metadata"]["eval_scope"], "holdout_only")
        self.assertIn(report["metadata"]["eval_split_source"], {"test", "val", "val+test"})


if __name__ == "__main__":
    unittest.main()
