from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.evaluator.auto import AutoEvaluator
from pfe_core.evaluator.judge import EvalReport, compare_eval_reports, make_recommendation
from pfe_core.adapter_store.store import AdapterStore
from pfe_core.pipeline import PipelineService
from pfe_core.curator.distillation import TrainingSample


class EvaluatorCompareReportsTests(unittest.TestCase):
    def _report(
        self,
        *,
        adapter_version: str,
        base_model: str,
        style_match: float,
        preference_alignment: float,
        quality_preservation: float,
        personality_consistency: float,
        style_preference_hit_rate: float | None = None,
    ) -> EvalReport:
        scores = {
            "style_match": style_match,
            "preference_alignment": preference_alignment,
            "quality_preservation": quality_preservation,
            "personality_consistency": personality_consistency,
        }
        if style_preference_hit_rate is not None:
            scores["style_preference_hit_rate"] = style_preference_hit_rate
        return EvalReport(
            adapter_version=adapter_version,
            base_model=base_model,
            num_test_samples=4,
            scores=scores,
            comparison="neutral",
            recommendation="needs_more_data",
            details=[],
            metadata={"scenario": "life-coach"},
        )

    def test_compare_eval_reports_returns_structured_version_delta(self) -> None:
        left = self._report(
            adapter_version="20260324-001",
            base_model="Qwen/Base-A",
            style_match=0.60,
            preference_alignment=0.62,
            quality_preservation=0.90,
            personality_consistency=0.66,
        )
        right = self._report(
            adapter_version="20260324-002",
            base_model="Qwen/Base-B",
            style_match=0.84,
            preference_alignment=0.80,
            quality_preservation=0.93,
            personality_consistency=0.88,
        )

        report = compare_eval_reports(left, right)

        self.assertEqual(report.left_adapter_version, "20260324-001")
        self.assertEqual(report.right_adapter_version, "20260324-002")
        self.assertEqual(report.winner, "right")
        self.assertEqual(report.comparison, "right_better")
        self.assertEqual(report.recommendation, "keep_right")
        self.assertGreater(report.overall_delta, 0.0)
        self.assertAlmostEqual(report.score_deltas["style_match"], 0.24, places=6)
        self.assertEqual(report.details[0].preferred, "right")
        self.assertEqual(report.dimension_summaries[0].dimension, "personalization")
        self.assertIn("style_match", report.dimension_summaries[0].metrics)
        self.assertIn("personalization=", report.summary_line)
        self.assertIn("quality=", report.summary_line)
        self.assertIn("style_match", report.personalization_summary)
        self.assertIn("quality_preservation", report.quality_summary)
        self.assertGreater(report.metadata["personalization_delta"], 0.0)
        self.assertGreater(report.metadata["quality_delta"], 0.0)
        self.assertIn("base_model_mismatch", report.metadata)
        self.assertEqual(report.metadata["delta_threshold"], 0.05)

    def test_compare_eval_reports_includes_style_preference_hit_rate(self) -> None:
        left = self._report(
            adapter_version="20260324-030",
            base_model="Qwen/Base",
            style_match=0.70,
            preference_alignment=0.68,
            quality_preservation=0.91,
            personality_consistency=0.72,
        )
        right = self._report(
            adapter_version="20260324-031",
            base_model="Qwen/Base",
            style_match=0.71,
            preference_alignment=0.69,
            quality_preservation=0.92,
            personality_consistency=0.73,
        )
        left.scores["style_preference_hit_rate"] = 0.25
        right.scores["style_preference_hit_rate"] = 0.75

        report = compare_eval_reports(left, right)

        self.assertIn("style_preference_hit_rate", report.score_deltas)
        self.assertAlmostEqual(report.score_deltas["style_preference_hit_rate"], 0.5, places=6)
        self.assertIn("style_preference_hit_rate", report.dimension_summaries[0].metrics)
        self.assertGreater(report.metadata["personalization_delta"], 0.0)
        self.assertEqual(report.recommendation, "keep_right")

    def test_auto_evaluator_compare_versions_wraps_compare_eval_reports(self) -> None:
        left = self._report(
            adapter_version="20260324-010",
            base_model="Qwen/Base",
            style_match=0.78,
            preference_alignment=0.74,
            quality_preservation=0.92,
            personality_consistency=0.81,
        )
        right = self._report(
            adapter_version="20260324-011",
            base_model="Qwen/Base",
            style_match=0.74,
            preference_alignment=0.71,
            quality_preservation=0.93,
            personality_consistency=0.80,
        )

        evaluator = AutoEvaluator()
        report = evaluator.compare_versions(left, right)

        self.assertEqual(report.left_adapter_version, "20260324-010")
        self.assertEqual(report.right_adapter_version, "20260324-011")
        self.assertEqual(report.comparison, "neutral")
        self.assertEqual(report.winner, "tie")
        self.assertEqual(report.recommendation, "needs_more_data")
        self.assertAlmostEqual(report.score_deltas["quality_preservation"], 0.01, places=6)

    def test_make_recommendation_accounts_for_personality_consistency(self) -> None:
        previous = self._report(
            adapter_version="20260324-020",
            base_model="Qwen/Base",
            style_match=0.72,
            preference_alignment=0.70,
            quality_preservation=0.91,
            personality_consistency=0.69,
        )
        current = self._report(
            adapter_version="20260324-021",
            base_model="Qwen/Base",
            style_match=0.72,
            preference_alignment=0.70,
            quality_preservation=0.91,
            personality_consistency=0.82,
        )

        self.assertEqual(make_recommendation(current, previous), "deploy")

    def test_make_recommendation_accounts_for_style_preference_hit_rate(self) -> None:
        previous = self._report(
            adapter_version="20260324-022",
            base_model="Qwen/Base",
            style_match=0.70,
            preference_alignment=0.70,
            quality_preservation=0.91,
            personality_consistency=0.70,
        )
        current = self._report(
            adapter_version="20260324-023",
            base_model="Qwen/Base",
            style_match=0.70,
            preference_alignment=0.70,
            quality_preservation=0.91,
            personality_consistency=0.70,
        )
        previous.scores["style_preference_hit_rate"] = 0.25
        current.scores["style_preference_hit_rate"] = 0.75

        self.assertEqual(make_recommendation(current, previous), "deploy")

    def test_auto_evaluator_aggregates_style_preference_hit_rate_from_sample_metadata(self) -> None:
        class FakeJudge:
            def prepare_eval_set(self, samples):
                return list(samples)

            def compare(self, base_output, adapted_output, reference):
                return {
                    "scores": {
                        "style_match": 0.8,
                        "preference_alignment": 0.75,
                        "quality_preservation": 0.92,
                        "personality_consistency": 0.82,
                    },
                    "comparison": "improved",
                    "winner": "adapted",
                    "reason": "ok",
                }

        samples = [
            TrainingSample(
                sample_id="sample_hit",
                sample_type="sft",
                instruction="prompt-1",
                chosen="温和回应",
                rejected=None,
                score=1.0,
                source="manual",
                source_event_ids=["event-1"],
                metadata={"explicit_response_style_preference": "温和、共情"},
            ),
            TrainingSample(
                sample_id="sample_miss",
                sample_type="sft",
                instruction="prompt-2",
                chosen="直接回应",
                rejected=None,
                score=1.0,
                source="manual",
                source_event_ids=["event-2"],
                metadata={"style_preferences": ["简洁"]},
            ),
        ]

        evaluator = AutoEvaluator(judge=FakeJudge())
        report = evaluator.evaluate(
            samples=samples,
            base_responses=["base-1", "base-2"],
            adapted_responses=["我理解你的感受，我们一起慢慢来", "我会详细展开说明"],
            base_model="base",
            adapter_version="adapter",
        )

        self.assertIn("style_preference_hit_rate", report.scores)
        self.assertAlmostEqual(report.scores["style_preference_hit_rate"], 0.5, places=6)
        self.assertAlmostEqual(report.details[0].scores["style_preference_hit_rate"], 1.0, places=6)
        self.assertAlmostEqual(report.details[1].scores["style_preference_hit_rate"], 0.0, places=6)
        self.assertEqual(report.metadata["style_preference_hit_rate_sample_count"], 2)
        self.assertEqual(report.metadata["style_preference_hit_rate_sample_ids"], ["sample_hit", "sample_miss"])

    def test_pipeline_compare_evaluation_persists_gate_summary_into_status(self) -> None:
        previous_home = os.environ.get("PFE_HOME")
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                pfe_home = Path(tempdir) / ".pfe"
                os.environ["PFE_HOME"] = str(pfe_home)
                store = AdapterStore(home=pfe_home, workspace="user_default")

                left_version = store.create_training_version(
                    base_model="Qwen/Base",
                    training_config={"backend": "mock_local", "train_type": "sft"},
                )["version"]
                right_version = store.create_training_version(
                    base_model="Qwen/Base",
                    training_config={"backend": "mock_local", "train_type": "sft"},
                )["version"]

                store.attach_eval_report(
                    left_version,
                    {
                        "adapter_version": left_version,
                        "base_model": "Qwen/Base",
                        "num_test_samples": 4,
                        "scores": {
                            "style_match": 0.60,
                            "style_preference_hit_rate": 0.25,
                            "preference_alignment": 0.62,
                            "quality_preservation": 0.90,
                            "personality_consistency": 0.66,
                        },
                        "comparison": "neutral",
                        "recommendation": "needs_more_data",
                        "details": [],
                    },
                )
                store.attach_eval_report(
                    right_version,
                    {
                        "adapter_version": right_version,
                        "base_model": "Qwen/Base",
                        "num_test_samples": 4,
                        "scores": {
                            "style_match": 0.84,
                            "style_preference_hit_rate": 0.75,
                            "preference_alignment": 0.80,
                            "quality_preservation": 0.93,
                            "personality_consistency": 0.88,
                        },
                        "comparison": "improved",
                        "recommendation": "deploy",
                        "details": [],
                    },
                )

                pipeline = PipelineService()
                report_json = pipeline.compare_evaluations(
                    left_adapter=left_version,
                    right_adapter=right_version,
                    workspace="user_default",
                )
                self.assertIn(left_version, report_json)
                self.assertIn(right_version, report_json)

                status = pipeline.status(workspace="user_default")
                self.assertEqual(status["compare_evaluation"]["recommendation"], "keep_right")
                self.assertEqual(status["candidate_summary"]["promotion_compare_recommendation"], "keep_right")
                self.assertEqual(status["candidate_summary"]["promotion_compare_winner"], "right")
                self.assertAlmostEqual(
                    status["candidate_summary"]["promotion_compare_style_preference_hit_rate_delta"],
                    0.5,
                    places=6,
                )
                self.assertIn(
                    "style_preference_hit_rate",
                    status["candidate_summary"]["promotion_compare_personalization_summary"],
                )
                self.assertEqual(status["candidate_summary"]["promotion_gate_status"], "open")
        finally:
            if previous_home is None:
                os.environ.pop("PFE_HOME", None)
            else:
                os.environ["PFE_HOME"] = previous_home

    def test_compare_gate_blocks_promoting_candidate_when_previous_version_is_preferred(self) -> None:
        previous_home = os.environ.get("PFE_HOME")
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                pfe_home = Path(tempdir) / ".pfe"
                os.environ["PFE_HOME"] = str(pfe_home)
                pipeline = PipelineService()
                pipeline.generate(scenario="life-coach", style="warm", num_samples=8)
                first = pipeline.train_result(method="mock_local", epochs=1, base_model="base", workspace="user_default")
                AdapterStore(home=pfe_home, workspace="user_default").promote(first.version)
                pipeline.generate(scenario="work-coach", style="direct", num_samples=8)
                second = pipeline.train_result(method="mock_local", epochs=1, base_model="base", workspace="user_default")

                pipeline._persist_compare_evaluation_state(
                    {
                        "left_adapter": first.version,
                        "right_adapter": second.version,
                        "comparison": "left_better",
                        "winner": "left",
                        "recommendation": "keep_left",
                        "overall_delta": -0.12,
                        "details": [],
                    },
                    workspace="user_default",
                )

                status = pipeline.status(workspace="user_default")
                self.assertEqual(status["candidate_summary"]["candidate_version"], second.version)
                self.assertEqual(status["candidate_summary"]["promotion_gate_status"], "blocked")
                self.assertEqual(status["candidate_summary"]["promotion_gate_reason"], "compare_prefers_previous")
                self.assertEqual(status["candidate_summary"]["promotion_gate_action"], "inspect_compare_evaluation")

                promote = pipeline.promote_candidate(workspace="user_default")
                self.assertEqual(promote["candidate_action"]["status"], "blocked")
                self.assertEqual(promote["candidate_action"]["reason"], "compare_prefers_previous")
                self.assertEqual(promote["candidate_action"]["required_action"], "inspect_compare_evaluation")
        finally:
            if previous_home is None:
                os.environ.pop("PFE_HOME", None)
            else:
                os.environ["PFE_HOME"] = previous_home


if __name__ == "__main__":
    unittest.main()
