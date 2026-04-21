"""Tests for teacher distillation logic including difference-driven gate and provenance."""

from __future__ import annotations

import pytest

from pfe_core.curator.distillation import (
    DistillationConfig,
    TeacherDistiller,
    TrainingSample,
    _build_teacher_provenance,
)
from pfe_core.curator.prada import (
    compute_text_similarity,
    should_use_teacher_output,
    should_use_teacher_output_for_user_preference,
)
from pfe_core.curator.teacher_client import TeacherClientConfig, TeacherInferenceClient


class TestPradaDifferenceDriven:
    def test_similarity_high_means_discard(self):
        local = "This is the local model output."
        teacher = "This is the local model output."
        use_teacher, similarity = should_use_teacher_output(local, teacher, threshold=0.85)
        assert not use_teacher
        assert similarity > 0.85

    def test_similarity_low_means_keep(self):
        local = "Short answer."
        teacher = "A much longer and more detailed answer with extra context."
        use_teacher, similarity = should_use_teacher_output(local, teacher, threshold=0.85)
        assert use_teacher
        assert similarity < 0.85

    def test_user_preference_alignment_accept(self):
        teacher = "The user prefers concise and direct responses."
        user_edit = "The user prefers concise and direct responses."
        use_teacher, similarity = should_use_teacher_output_for_user_preference(
            teacher, user_edit, similarity_threshold=0.7
        )
        assert use_teacher
        assert similarity >= 0.7

    def test_user_preference_alignment_reject(self):
        teacher = "A verbose and overly detailed explanation."
        user_edit = "Keep it short and direct."
        use_teacher, similarity = should_use_teacher_output_for_user_preference(
            teacher, user_edit, similarity_threshold=0.7
        )
        assert not use_teacher
        assert similarity < 0.7


class TestTeacherProvenance:
    def test_build_teacher_provenance_structure(self):
        prov = _build_teacher_provenance(
            teacher_model="gpt-4",
            generation_prompt="Rewrite this",
            confidence=0.82,
            was_accepted=True,
            backend="cloud",
        )
        assert prov["source"] == "teacher_distillation"
        assert prov["teacher_model"] == "gpt-4"
        assert prov["generation_prompt"] == "Rewrite this"
        assert prov["confidence"] == 0.82
        assert prov["was_accepted"] is True
        assert prov["backend"] == "cloud"
        assert "generated_at" in prov


class TestTeacherInferenceClient:
    def test_mock_generate(self):
        client = TeacherInferenceClient(config=TeacherClientConfig(backend="mock"))
        result = client.generate([{"role": "user", "content": "Hello"}])
        assert "text" in result
        assert result["backend"] == "mock"
        assert "mock-teacher" in result["text"]

    def test_generate_explanation_fallback(self):
        client = TeacherInferenceClient(config=TeacherClientConfig(backend="mock"))
        explanation = client.generate_explanation("prompt", "base", "feedback")
        # Mock backend returns non-empty text; fallback text is only hit on empty result.
        # Accept either mock output or fallback output.
        assert explanation
        assert "mock-teacher" in explanation or "feedback" in explanation or "prefers" in explanation

    def test_generate_preference_pair_fallback(self):
        client = TeacherInferenceClient(config=TeacherClientConfig(backend="mock"))
        pair = client.generate_preference_pair("prompt", "rejected text")
        assert pair["prompt"] == "prompt"
        assert pair["rejected"] == "rejected text"
        assert pair["chosen"]
        assert "backend" in pair


class TestTeacherDistillerDifferenceDriven:
    def test_difference_driven_gate_accept(self):
        config = DistillationConfig()
        distiller = TeacherDistiller(config)
        teacher_out = "The user wants a short direct answer."
        user_edit = "The user wants a short direct answer."
        use_teacher, similarity = distiller._difference_driven_gate(
            teacher_out, user_edit, similarity_threshold=0.7
        )
        assert use_teacher
        assert similarity >= 0.7

    def test_difference_driven_gate_reject(self):
        config = DistillationConfig()
        distiller = TeacherDistiller(config)
        teacher_out = "A very long and unrelated explanation."
        user_edit = "Short and direct."
        use_teacher, similarity = distiller._difference_driven_gate(
            teacher_out, user_edit, similarity_threshold=0.7
        )
        assert not use_teacher
        assert similarity < 0.7

    def test_distill_from_scenario_returns_samples(self):
        config = DistillationConfig(max_samples=5)
        distiller = TeacherDistiller(config)
        samples = distiller.distill_from_scenario("life-coach", "温和、共情", 3)
        assert len(samples) == 3
        for s in samples:
            assert isinstance(s, TrainingSample)
            assert s.source == "teacher"
            assert s.metadata.get("teacher_model") == config.teacher_model

    def test_distill_from_scenario_teacher_provenance(self):
        config = DistillationConfig(max_samples=2)
        distiller = TeacherDistiller(config)
        samples = distiller.distill_from_scenario("life-coach", "温和、共情", 2)
        for s in samples:
            prov = s.teacher_provenance
            assert prov.get("source") == "teacher_distillation"
            assert "teacher_model" in prov
            assert "generation_prompt" in prov
            assert "confidence" in prov
            assert "was_accepted" in prov


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
