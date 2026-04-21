from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_core.converters import to_dataclass, to_pydantic
from pfe_core.curator import RawSignal, SignalSampleCuratorConfig, curate_signals_to_samples, signal_to_sft_sample
from pfe_core.curator.datasets import (
    SampleFilterConfig,
    attach_signal_quality,
    build_signal_quality,
    build_signal_provenance,
    filter_samples,
    normalize_reply_style,
)
from pfe_core.curator.distillation import TeacherDistiller, curate_signals_to_preference_samples, signal_to_preference_sample
from pfe_core.curator.teacher_client import TeacherClientConfig, TeacherInferenceClient
from pfe_core.config import PrivacyConfig
from pfe_core.models import RawSignal as CoreRawSignal
from pfe_core.models import SignalQuality, TrainingSample as CoreTrainingSample


class SignalCuratorPhase1Tests(unittest.TestCase):
    def _signal(
        self,
        *,
        event_id: str = "evt-signal-1",
        source_event_id: str = "evt-source-1",
        event_chain_ids: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        event_type: str = "accept",
        user_input: str = "我今天有点焦虑",
        model_output: str = "先把压力拆成一件件小事，我们慢慢来。",
    ) -> RawSignal:
        return RawSignal(
            signal_id=event_id,
            source_event_id=source_event_id,
            request_id="req-1",
            session_id="sess-1",
            adapter_version="20260325-001",
            event_type=event_type,
            timestamp=datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc),
            context=user_input,
            model_output=model_output,
            user_action={"type": event_type, "final_text": user_input, "accepted_text": model_output},
            metadata=dict(metadata or {}),
            event_chain_ids=list(event_chain_ids or [source_event_id, "evt-mid-1", event_id]),
        )

    def test_signal_to_sft_sample_normalizes_event_chain_and_provenance(self) -> None:
        signal = self._signal(event_chain_ids=["evt-source-1", "evt-mid-1", "evt-mid-1", "evt-signal-1"])

        sample = signal_to_sft_sample(signal, config=SignalSampleCuratorConfig())

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.sample_type, "sft")
        self.assertEqual(sample.source, "signal")
        self.assertEqual(sample.rejected, None)
        self.assertEqual(sample.metadata["dataset_split"], "train")
        self.assertEqual(sample.metadata["request_id"], "req-1")
        self.assertEqual(sample.metadata["session_id"], "sess-1")
        self.assertEqual(sample.metadata["source_event_id"], "evt-source-1")
        self.assertEqual(sample.metadata["source_event_ids"], ["evt-source-1", "evt-mid-1", "evt-signal-1"])
        self.assertEqual(sample.metadata["event_chain_root_id"], "evt-source-1")
        self.assertEqual(sample.metadata["event_chain_terminal_id"], "evt-signal-1")
        self.assertEqual(sample.metadata["event_chain_length"], 3)
        self.assertTrue(sample.metadata["event_chain_complete"])
        self.assertEqual(sample.preference_kind, "accepted")
        self.assertEqual(sample.preference_source, "signal")
        self.assertEqual(sample.preference_reason, "signal_reply_style=accepted")
        self.assertIn("signal_quality", sample.metadata)
        self.assertEqual(sample.metadata["preference"]["kind"], "accepted")
        self.assertEqual(sample.source_event_ids, ["evt-source-1", "evt-mid-1", "evt-signal-1"])

    def test_curate_signals_to_samples_rejects_test_split_and_keeps_train_samples(self) -> None:
        good = self._signal(event_id="evt-good")
        test_split = self._signal(
            event_id="evt-test",
            metadata={"dataset_split": "test"},
        )

        curated, results = curate_signals_to_samples([good, test_split], config=SignalSampleCuratorConfig())

        self.assertEqual(len(curated), 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].reason, "curated_sft")
        self.assertEqual(results[1].reason, "test_split_not_curated")
        self.assertEqual(curated[0].metadata["dataset_split"], "train")

    def test_signal_to_sft_sample_rejects_incomplete_chain(self) -> None:
        signal = self._signal(event_chain_ids=["evt-source-1"], metadata={"dataset_split": "train"})

        sample = signal_to_sft_sample(signal, config=SignalSampleCuratorConfig())

        self.assertIsNone(sample)

    def test_signal_to_preference_sample_builds_dpo_pair_from_edited_signal(self) -> None:
        signal = self._signal(
            event_id="evt-dpo-edit",
            event_type="edit",
            model_output="这段话太急了，我们可以慢一点。",
            user_input="帮我把话说得更温和一点",
            metadata={"dataset_split": "train"},
        )
        signal.user_action = {
            "type": "edit",
            "final_text": "帮我把话说得更温和一点",
            "edited_text": "帮我把话说得更温和一点，请先承认我的感受。",
        }

        sample = signal_to_preference_sample(signal, config=SignalSampleCuratorConfig())

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.sample_type, "dpo")
        self.assertEqual(sample.instruction, "帮我把话说得更温和一点")
        self.assertEqual(sample.chosen, "帮我把话说得更温和一点，请先承认我的感受。")
        self.assertEqual(sample.rejected, "这段话太急了，我们可以慢一点。")
        self.assertEqual(sample.preference_kind, "edited")
        self.assertEqual(sample.preference_source, "signal")
        self.assertIsNotNone(sample.preference_pair_id)
        self.assertEqual(sample.preference_reason, "signal_reply_style=edited")
        self.assertEqual(sample.metadata["preference"]["source"], "signal")
        self.assertEqual(sample.metadata["signal_quality"]["reply_style"], "edited")

    def test_signal_to_sft_sample_marks_explicit_response_preference_reinforcement(self) -> None:
        signal = self._signal(
            metadata={
                "explicit_user_data_routing": {
                    "processed": True,
                    "candidates": [
                        {
                            "key": "回答风格偏好",
                            "value": "请更温和、更鼓励式",
                            "kind": "response_preference",
                            "primary_lane": "profile",
                            "training_target": "preference_only",
                        }
                    ],
                }
            }
        )

        sample = signal_to_sft_sample(signal, config=SignalSampleCuratorConfig())

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertTrue(sample.metadata["explicit_response_preference_reinforced"])
        self.assertEqual(sample.metadata["training_signal_category"], "preference_reinforced")
        self.assertEqual(
            sample.metadata["training_gate_reason"],
            "explicit_response_preference_reinforced_by_feedback",
        )
        self.assertEqual(sample.metadata["explicit_response_preference_values"], ["请更温和、更鼓励式"])
        self.assertGreater(sample.score, 0.9)

    def test_curate_signals_to_preference_samples_rejects_low_quality_or_missing_pairs(self) -> None:
        good = self._signal(
            event_id="evt-dpo-good",
            event_type="edit",
            model_output="这句话有点硬。",
        )
        good.user_action = {
            "type": "edit",
            "final_text": "请把语气放轻一点",
            "edited_text": "请把语气放轻一点，我能理解你的不安。",
        }
        low_quality = self._signal(
            event_id="evt-dpo-low",
            event_type="reject",
            metadata={"dataset_split": "train"},
            model_output="",
        )
        low_quality.user_action = {
            "type": "reject",
            "final_text": "我不想要这个说法",
            "rejected_text": "",
        }

        curated, results = curate_signals_to_preference_samples([good, low_quality], config=SignalSampleCuratorConfig())

        self.assertEqual(len(curated), 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].reason, "curated_dpo")
        self.assertIn(results[1].reason, {"missing_rejected_text", "low_signal_confidence", "missing_preferred_text", "missing_model_output"})
        self.assertEqual(curated[0].sample_type, "dpo")

    def test_signal_to_preference_sample_marks_explicit_response_preference_reinforcement(self) -> None:
        signal = self._signal(
            event_id="evt-dpo-pref",
            event_type="edit",
            user_input="请把语气再温和一些",
            model_output="你应该直接去做。",
            metadata={
                "explicit_user_data_routing": {
                    "processed": True,
                    "candidates": [
                        {
                            "key": "回答风格偏好",
                            "value": "希望你更像教练一样温和回应",
                            "kind": "response_preference",
                            "primary_lane": "profile",
                            "training_target": "preference_only",
                        }
                    ],
                }
            },
        )
        signal.user_action = {
            "type": "edit",
            "final_text": "请把语气再温和一些",
            "edited_text": "我能理解你现在很难受，我们可以先慢慢整理下一步。",
        }

        sample = signal_to_preference_sample(signal, config=SignalSampleCuratorConfig())

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertTrue(sample.metadata["explicit_response_preference_reinforced"])
        self.assertEqual(sample.metadata["training_signal_category"], "preference_reinforced")
        self.assertEqual(
            sample.metadata["training_gate_reason"],
            "explicit_response_preference_reinforced_by_feedback",
        )
        self.assertEqual(
            sample.metadata["explicit_response_preference_values"],
            ["希望你更像教练一样温和回应"],
        )

    def test_teacher_dpo_pair_requires_explicit_inputs(self) -> None:
        distiller = TeacherDistiller()

        with self.assertRaises(ValueError):
            distiller.build_dpo_pair(prompt=" ", chosen="chosen", rejected="rejected")
        with self.assertRaises(ValueError):
            distiller.build_dpo_pair(prompt="prompt", chosen=" ", rejected="rejected")
        with self.assertRaises(ValueError):
            distiller.build_dpo_pair(prompt="prompt", chosen="chosen", rejected=" ")

    def test_teacher_dpo_pair_carries_preference_metadata(self) -> None:
        distiller = TeacherDistiller()

        sample = distiller.build_dpo_pair(prompt="请给我一个更温和的回应", chosen="我能理解你的感受，我们可以慢慢来。", rejected="别想太多，直接做。")

        self.assertEqual(sample.sample_type, "dpo")
        self.assertEqual(sample.preference_kind, "teacher_dpo")
        self.assertEqual(sample.preference_source, "teacher")
        self.assertEqual(sample.preference_pair_id, sample.sample_id)
        self.assertEqual(sample.preference_reason, "explicit_teacher_pair")
        self.assertEqual(sample.metadata["preference"]["kind"], "teacher_dpo")
        self.assertEqual(sample.metadata["sample_type"], "dpo")

    def test_signal_quality_scaffold_supports_reply_styles_and_provenance(self) -> None:
        accepted_payload = {
            "event_id": "evt-quality-1",
            "source_event_id": "evt-source-quality-1",
            "request_id": "req-quality-1",
            "session_id": "sess-quality-1",
            "adapter_version": "20260325-001",
            "event_type": "accept",
            "context": "我今天压力很大",
            "model_output": "先把事情拆成三步。",
            "user_input": "我今天压力很大",
            "source_event_ids": ["evt-source-quality-1", "evt-quality-1"],
            "event_chain_ids": ["evt-source-quality-1", "evt-quality-1"],
            "user_action": {"type": "accept", "final_text": "我今天压力很大", "accepted_text": "先把事情拆成三步。"},
        }
        rejected_payload = {
            "event_id": "evt-quality-2",
            "source_event_id": "evt-source-quality-2",
            "request_id": "req-quality-2",
            "session_id": "sess-quality-2",
            "adapter_version": "20260325-001",
            "event_type": "reject",
            "context": "我不想这么说",
            "model_output": "你可以先试着忽略它。",
            "user_input": "我不想这么说",
            "source_event_ids": ["evt-source-quality-2", "evt-quality-2"],
            "event_chain_ids": ["evt-source-quality-2", "evt-quality-2"],
            "user_action": {"type": "reject", "rejected_text": "你可以先试着忽略它。"},
        }
        edited_payload = {
            "event_id": "evt-quality-3",
            "source_event_id": "evt-source-quality-3",
            "request_id": "req-quality-3",
            "session_id": "sess-quality-3",
            "adapter_version": "20260325-001",
            "event_type": "edit",
            "context": "帮我润色这段话",
            "model_output": "你的表达已经很好了。",
            "user_input": "帮我润色这段话",
            "source_event_ids": ["evt-source-quality-3", "evt-quality-3"],
            "event_chain_ids": ["evt-source-quality-3", "evt-quality-3"],
            "user_action": {"type": "edit", "final_text": "帮我把这段话改得更清楚一点。", "edited_text": "帮我把这段话改得更清楚一点。"},
        }

        accepted = build_signal_quality(accepted_payload)
        rejected = build_signal_quality(rejected_payload)
        edited = build_signal_quality(edited_payload)
        accepted_provenance = build_signal_provenance(accepted_payload)

        self.assertEqual(accepted.reply_style, "accepted")
        self.assertEqual(rejected.reply_style, "rejected")
        self.assertEqual(edited.reply_style, "edited")
        self.assertFalse(accepted.conflict)
        self.assertFalse(rejected.conflict)
        self.assertFalse(edited.conflict)
        self.assertGreater(accepted.confidence, edited.confidence)
        self.assertGreater(edited.confidence, rejected.confidence)
        self.assertTrue(accepted.provenance["event_chain_complete"])
        self.assertEqual(accepted.provenance["source_event_ids"], ["evt-source-quality-1", "evt-quality-1"])
        self.assertEqual(accepted_provenance["event_chain_root_id"], "evt-source-quality-1")
        self.assertEqual(normalize_reply_style(event_type="copy"), "accepted")
        self.assertEqual(normalize_reply_style(user_action={"type": "edit"}), "edited")

        conflict_case = build_signal_quality(
            {
                "event_id": "evt-quality-4",
                "source_event_id": "evt-source-quality-4",
                "request_id": "req-quality-4",
                "session_id": "sess-quality-4",
                "event_type": "accept",
                "model_output": "",
                "source_event_ids": ["evt-source-quality-4"],
                "event_chain_ids": ["evt-source-quality-4"],
                "user_action": {"type": "accept"},
            }
        )
        self.assertTrue(conflict_case.conflict)
        self.assertEqual(conflict_case.conflict_reason, "missing_model_output")
        self.assertLess(conflict_case.confidence, accepted.confidence)

    def test_signal_quality_round_trips_through_converters(self) -> None:
        signal_quality = SignalQuality(
            reply_style="edited",
            confidence=0.77,
            conflict=False,
            confidence_reason="edited_reply_with_complete_chain",
            provenance={
                "request_id": "req-roundtrip",
                "session_id": "sess-roundtrip",
                "source_event_ids": ["evt-source-roundtrip", "evt-signal-roundtrip"],
                "event_chain_ids": ["evt-source-roundtrip", "evt-signal-roundtrip"],
                "event_chain_complete": True,
            },
            details={"confidence_components": {"reply_style": "edited"}},
        )

        schema = to_pydantic(signal_quality)
        roundtrip = to_dataclass(schema)
        self.assertEqual(roundtrip, signal_quality)

    def test_signal_quality_filter_rejects_low_confidence_signal_samples(self) -> None:
        signal_quality = build_signal_quality(
            {
                "event_id": "evt-model-quality-2",
                "source_event_id": "evt-model-source-2",
                "request_id": "req-model-quality-2",
                "session_id": "sess-model-quality-2",
                "adapter_version": "20260325-001",
                "event_type": "accept",
                "context": "请帮我整理一下",
                "model_output": "",
                "user_input": "请帮我整理一下",
                "source_event_ids": ["evt-model-source-2", "evt-model-quality-2"],
                "event_chain_ids": ["evt-model-source-2", "evt-model-quality-2"],
                "user_action": {"type": "accept", "final_text": "请帮我整理一下"},
            }
        )
        sample = CoreTrainingSample(
            sample_id="sample-model-2",
            instruction="请帮我整理一下",
            chosen="先试着把任务列出来。",
            score=0.9,
            source_event_ids=["evt-model-source-2", "evt-model-quality-2"],
            signal_quality=signal_quality,
            metadata={"dataset_split": "train"},
        )

        filtered = filter_samples([sample], config=SampleFilterConfig())
        self.assertEqual(filtered, [])

    def test_core_signal_models_can_carry_signal_quality(self) -> None:
        signal_quality = build_signal_quality(
            {
                "event_id": "evt-model-quality-1",
                "source_event_id": "evt-model-source-1",
                "request_id": "req-model-quality-1",
                "session_id": "sess-model-quality-1",
                "adapter_version": "20260325-001",
                "event_type": "accept",
                "context": "请帮我整理一下",
                "model_output": "当然，我们可以先分三步。",
                "user_input": "请帮我整理一下",
                "source_event_ids": ["evt-model-source-1", "evt-model-quality-1"],
                "event_chain_ids": ["evt-model-source-1", "evt-model-quality-1"],
                "user_action": {"type": "accept", "final_text": "请帮我整理一下"},
            }
        )

        signal = CoreRawSignal(
            signal_id="signal-model-1",
            source_event_id="evt-model-source-1",
            request_id="req-model-quality-1",
            session_id="sess-model-quality-1",
            adapter_version="20260325-001",
            event_type="accept",
            context="请帮我整理一下",
            model_output="当然，我们可以先分三步。",
            user_action={"type": "accept", "final_text": "请帮我整理一下"},
            source_event_ids=["evt-model-source-1", "evt-model-quality-1"],
            event_chain_ids=["evt-model-source-1", "evt-model-quality-1"],
            signal_quality=signal_quality,
        )

        sample = CoreTrainingSample(
            sample_id="sample-model-1",
            instruction="请帮我整理一下",
            chosen="当然，我们可以先分三步。",
            source_event_ids=["evt-model-source-1", "evt-model-quality-1"],
            signal_quality=signal_quality,
            metadata={"dataset_split": "train"},
        )

        signal_schema = to_pydantic(signal)
        sample_schema = to_pydantic(sample)
        self.assertEqual(signal_schema.signal_quality.reply_style, "accepted")
        self.assertEqual(sample_schema.signal_quality.reply_style, "accepted")

        restored_signal = to_dataclass(signal_schema)
        restored_sample = to_dataclass(sample_schema)
        self.assertEqual(restored_signal.signal_quality.reply_style, "accepted")
        self.assertEqual(restored_sample.signal_quality.confidence, signal_quality.confidence)
        self.assertEqual(attach_signal_quality({"sample_id": "dict-sample"}, signal_quality)["signal_quality"], signal_quality)

    def test_teacher_distiller_uses_injected_client(self) -> None:
        class FakeTeacherClient:
            def __init__(self):
                self.config = TeacherClientConfig(backend="local", model="fake")

            def generate(self, messages, **kwargs):
                return {"text": "[fake-teacher]  customized response", "backend": "fake", "model": "fake", "usage": {}}

        client = FakeTeacherClient()
        privacy = PrivacyConfig(mode="strict_local", allow_teacher_cloud=False)
        distiller = TeacherDistiller(teacher_client=client, privacy_config=privacy)
        response = distiller._teacher_response("测试问题", "温和", 0)
        self.assertEqual(response, "[fake-teacher]  customized response")

    def test_teacher_distiller_respects_privacy_gate(self) -> None:
        class FakeTeacherClient:
            def __init__(self):
                self.config = TeacherClientConfig(backend="cloud", model="cloud-model")

            def generate(self, messages, **kwargs):
                return {"text": "[fake-cloud] should not appear", "backend": "fake", "model": "fake", "usage": {}}

        client = FakeTeacherClient()
        privacy = PrivacyConfig(mode="strict_local", allow_teacher_cloud=False)
        distiller = TeacherDistiller(teacher_client=client, privacy_config=privacy)
        response = distiller._teacher_response("测试问题", "温和", 0)
        # Should fall back to deterministic mock because cloud is not allowed
        self.assertNotEqual(response, "[fake-cloud] should not appear")
        self.assertIn("回应", response)

    def test_prada_filter_skips_high_similarity_samples(self) -> None:
        class FakeLocalEngine:
            def generate(self, messages, **kwargs):
                return "This is almost identical to teacher output."

        teacher_text = "This is almost identical to teacher output."
        privacy = PrivacyConfig(mode="strict_local", allow_teacher_cloud=False)
        distiller = TeacherDistiller(
            privacy_config=privacy,
            local_engine=FakeLocalEngine(),
        )
        distiller.config.prada_enabled = True
        distiller.config.prada_similarity_threshold = 0.85
        use_teacher, similarity, _ = distiller._maybe_prada_filter("测试问题", teacher_text)
        self.assertFalse(use_teacher)
        self.assertGreaterEqual(similarity, 0.85)

    def test_teacher_metadata_includes_backend_and_similarity(self) -> None:
        class FakeTeacherClient:
            def __init__(self):
                self.config = TeacherClientConfig(backend="local", model="fake")

            def generate(self, messages, **kwargs):
                return {"text": "fake-output", "backend": "fake", "model": "fake", "usage": {}}

        client = FakeTeacherClient()
        privacy = PrivacyConfig(mode="strict_local", allow_teacher_cloud=False)
        distiller = TeacherDistiller(teacher_client=client, privacy_config=privacy)
        samples = distiller.distill_from_scenario("life-coach", "温和", 1)
        self.assertEqual(len(samples), 1)
        metadata = samples[0].metadata
        self.assertEqual(metadata["teacher_backend"], "local")
        # When local_engine is None, similarity should not be in metadata
        self.assertNotIn("local_teacher_similarity", metadata)
        self.assertFalse(metadata.get("prada_filtered", False))


if __name__ == "__main__":
    unittest.main()
