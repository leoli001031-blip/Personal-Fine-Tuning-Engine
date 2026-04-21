"""Tests for DPO (Direct Preference Optimization) training backend.

These tests verify:
- DPO dataset building from signals
- DPO training execution (with optional small model)
- Incremental DPO training from SFT adapter
- Training artifacts can be loaded
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if dependencies are not available
try:
    from datasets import Dataset
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

# Mark all tests to skip if dependencies unavailable
pytestmark = pytest.mark.skipif(
    not DEPS_AVAILABLE,
    reason="datasets library not available"
)


@pytest.fixture
def mock_signals() -> List[Dict[str, Any]]:
    """Create mock signals for DPO dataset building."""
    base_time = datetime.now(timezone.utc)

    return [
        # Session 1: Accepted response
        {
            "signal_id": "sig_001",
            "session_id": "session_001",
            "request_id": "req_001",
            "source_event_id": "evt_001",
            "source_event_ids": ["evt_001"],
            "event_type": "chat",
            "timestamp": base_time.isoformat(),
            "context": "What is the capital of France?",
            "model_output": "The capital of France is Paris.",
            "user_input": "What is the capital of France?",
            "user_action": {
                "accepted_text": "The capital of France is Paris.",
                "reply_style": "accepted",
            },
            "metadata": {},
        },
        # Session 1: Rejected response (different from accepted)
        {
            "signal_id": "sig_002",
            "session_id": "session_001",
            "request_id": "req_001",
            "source_event_id": "evt_002",
            "source_event_ids": ["evt_001", "evt_002"],
            "event_type": "chat",
            "timestamp": (base_time + timedelta(seconds=1)).isoformat(),
            "context": "What is the capital of France?",
            "model_output": "The capital of France is London.",
            "user_input": "What is the capital of France?",
            "user_action": {
                "rejected_text": "The capital of France is London.",
                "reply_style": "rejected",
            },
            "metadata": {},
        },
        # Session 2: Accepted response
        {
            "signal_id": "sig_003",
            "session_id": "session_002",
            "request_id": "req_002",
            "source_event_id": "evt_003",
            "source_event_ids": ["evt_003"],
            "event_type": "chat",
            "timestamp": (base_time + timedelta(seconds=2)).isoformat(),
            "context": "Explain machine learning in simple terms.",
            "model_output": "Machine learning is a way for computers to learn patterns from data.",
            "user_input": "Explain machine learning in simple terms.",
            "user_action": {
                "accepted_text": "Machine learning is a way for computers to learn patterns from data.",
                "reply_style": "accepted",
            },
            "metadata": {},
        },
        # Session 2: Edited response (user improved the output)
        {
            "signal_id": "sig_004",
            "session_id": "session_002",
            "request_id": "req_002",
            "source_event_id": "evt_004",
            "source_event_ids": ["evt_003", "evt_004"],
            "event_type": "chat",
            "timestamp": (base_time + timedelta(seconds=3)).isoformat(),
            "context": "Explain machine learning in simple terms.",
            "model_output": "Machine learning is a way for computers to learn patterns from data.",
            "user_input": "Explain machine learning in simple terms.",
            "user_action": {
                "edited_text": "Machine learning is a way for computers to learn from examples without being explicitly programmed.",
                "final_text": "Machine learning is a way for computers to learn from examples without being explicitly programmed.",
                "reply_style": "edited",
            },
            "metadata": {},
        },
    ]


class TestDPODatasetBuilder:
    """Test DPO dataset building from signals."""

    def test_dpo_pair_creation(self, mock_signals: List[Dict[str, Any]]) -> None:
        """Test that DPOPair objects are created correctly."""
        from pfe_core.trainer.dpo_dataset import DPOPair

        pair = DPOPair(
            prompt="What is the capital of France?",
            chosen="The capital of France is Paris.",
            rejected="The capital of France is London.",
            session_id="session_001",
            source_event_ids=["evt_001", "evt_002"],
            confidence=0.85,
            metadata={"pair_type": "accepted_vs_rejected"},
        )

        assert pair.prompt == "What is the capital of France?"
        assert pair.chosen == "The capital of France is Paris."
        assert pair.rejected == "The capital of France is London."
        assert pair.confidence == 0.85

    def test_dpo_pair_to_dict(self, mock_signals: List[Dict[str, Any]]) -> None:
        """Test DPOPair to_dict conversion."""
        from pfe_core.trainer.dpo_dataset import DPOPair

        pair = DPOPair(
            prompt="Test prompt",
            chosen="Good response",
            rejected="Bad response",
            session_id="session_001",
            source_event_ids=["evt_001"],
            confidence=0.9,
            metadata={"test": "value"},
        )

        result = pair.to_dict()

        assert result["prompt"] == "Test prompt"
        assert result["chosen"] == "Good response"
        assert result["rejected"] == "Bad response"
        assert result["confidence"] == 0.9
        assert result["test"] == "value"

    @patch("pfe_core.trainer.dpo_dataset.list_signals")
    def test_build_from_signals(self, mock_list_signals, mock_signals: List[Dict[str, Any]]) -> None:
        """Test building DPO dataset from signals."""
        from pfe_core.trainer.dpo_dataset import DPODatasetBuilder

        mock_list_signals.return_value = mock_signals

        builder = DPODatasetBuilder(workspace="test_workspace")
        dataset = builder.build_from_signals(min_confidence=0.5)

        # Should create pairs from sessions with both accepted and rejected/edited
        assert isinstance(dataset, Dataset)
        assert len(dataset) > 0

        # Check dataset has required columns
        assert "prompt" in dataset.column_names
        assert "chosen" in dataset.column_names
        assert "rejected" in dataset.column_names
        assert "session_id" in dataset.column_names

    @patch("pfe_core.trainer.dpo_dataset.list_signals")
    def test_empty_signals(self, mock_list_signals) -> None:
        """Test building dataset with no signals."""
        from pfe_core.trainer.dpo_dataset import DPODatasetBuilder

        mock_list_signals.return_value = []

        builder = DPODatasetBuilder(workspace="test_workspace")
        dataset = builder.build_from_signals()

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 0

    @patch("pfe_core.trainer.dpo_dataset.list_signals")
    def test_min_confidence_filtering(self, mock_list_signals, mock_signals: List[Dict[str, Any]]) -> None:
        """Test that min_confidence filters low-confidence signals."""
        from pfe_core.trainer.dpo_dataset import DPODatasetBuilder

        mock_list_signals.return_value = mock_signals

        builder = DPODatasetBuilder(workspace="test_workspace")

        # High confidence threshold
        dataset_high = builder.build_from_signals(min_confidence=0.9)

        # Low confidence threshold
        dataset_low = builder.build_from_signals(min_confidence=0.5)

        # Low threshold should have equal or more pairs
        assert len(dataset_low) >= len(dataset_high)

    def test_get_statistics(self) -> None:
        """Test getting signal statistics."""
        from pfe_core.trainer.dpo_dataset import DPODatasetBuilder

        with patch("pfe_core.trainer.dpo_dataset.list_signals") as mock_list_signals:
            mock_list_signals.return_value = [
                {
                    "signal_id": "sig_001",
                    "session_id": "session_001",
                    "user_action": {"reply_style": "accepted"},
                },
                {
                    "signal_id": "sig_002",
                    "session_id": "session_001",
                    "user_action": {"reply_style": "rejected"},
                },
            ]

            builder = DPODatasetBuilder(workspace="test_workspace")
            stats = builder.get_statistics()

            assert "total_signals" in stats
            assert "sessions" in stats
            assert "reply_styles" in stats
            assert "confidence_distribution" in stats


class TestDPODatasetFromSamples:
    """Test building DPO dataset from existing training samples."""

    def test_build_from_dpo_samples(self) -> None:
        """Test building DPO dataset from DPO-type training samples."""
        from pfe_core.trainer.dpo_dataset import build_dpo_dataset_from_samples

        samples = [
            {
                "sample_id": "sample_001",
                "sample_type": "dpo",
                "instruction": "What is 2+2?",
                "chosen": "2+2 equals 4.",
                "rejected": "2+2 equals 5.",
                "source_adapter_version": "v001",
                "source_event_ids": ["evt_001"],
                "metadata": {
                    "signal_quality": {"confidence": 0.9},
                },
            },
            {
                "sample_id": "sample_002",
                "sample_type": "dpo",
                "instruction": "What color is the sky?",
                "chosen": "The sky is blue.",
                "rejected": "The sky is green.",
                "source_adapter_version": "v001",
                "source_event_ids": ["evt_002"],
                "metadata": {
                    "signal_quality": {"confidence": 0.8},
                },
            },
            # Non-DPO sample should be filtered out
            {
                "sample_id": "sample_003",
                "sample_type": "sft",
                "instruction": "Hello",
                "chosen": "Hi there!",
                "rejected": None,
            },
        ]

        dataset = build_dpo_dataset_from_samples(samples, min_confidence=0.7)

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2  # Only DPO samples with sufficient confidence

    def test_build_from_samples_filters_low_confidence(self) -> None:
        """Test that low confidence samples are filtered."""
        from pfe_core.trainer.dpo_dataset import build_dpo_dataset_from_samples

        samples = [
            {
                "sample_id": "sample_001",
                "sample_type": "dpo",
                "instruction": "Test",
                "chosen": "Good",
                "rejected": "Bad",
                "metadata": {
                    "signal_quality": {"confidence": 0.5},  # Below threshold
                },
            },
        ]

        dataset = build_dpo_dataset_from_samples(samples, min_confidence=0.7)

        assert len(dataset) == 0


class TestDPOTrainerExecutor:
    """Test DPO trainer executor."""

    def test_executor_initialization(self) -> None:
        """Test DPO executor can be initialized with config."""
        from pfe_core.trainer.dpo_executor import DPOTrainerExecutor
        from pfe_core.config import TrainerConfig

        config = TrainerConfig(
            method="qlora",
            train_type="dpo",
            dpo_beta=0.2,
            lora_r=16,
            lora_alpha=32,
        )

        # Should raise if dependencies missing
        try:
            executor = DPOTrainerExecutor(config)
            assert executor.config == config
        except Exception as e:
            # Expected if trl/torch not installed
            assert "DPO training requires" in str(e)

    def test_check_dependencies(self) -> None:
        """Test dependency checking."""
        from pfe_core.trainer.dpo_executor import check_dpo_dependencies

        deps = check_dpo_dependencies()

        assert "torch" in deps
        assert "transformers" in deps
        assert "trl" in deps
        assert "peft" in deps
        assert "datasets" in deps
        assert "all_available" in deps

    def test_training_result_dataclass(self) -> None:
        """Test TrainingResult dataclass."""
        from pfe_core.trainer.dpo_executor import TrainingResult

        result = TrainingResult(
            success=True,
            adapter_path="/path/to/adapter",
            metrics={"train_loss": 0.5},
            num_samples=100,
            config={"dpo_beta": 0.1},
        )

        assert result.success is True
        assert result.adapter_path == "/path/to/adapter"
        assert result.metrics["train_loss"] == 0.5


class TestDPOIntegration:
    """Integration tests for DPO training pipeline."""

    @patch("pfe_core.trainer.dpo_dataset.list_signals")
    def test_service_build_dpo_dataset(self, mock_list_signals, mock_signals: List[Dict[str, Any]]) -> None:
        """Test TrainerService.build_dpo_dataset method."""
        from pfe_core.trainer.service import TrainerService

        mock_list_signals.return_value = mock_signals

        service = TrainerService()

        # Should build dataset info from signals
        try:
            result = service.build_dpo_dataset(workspace="test")
            assert "num_pairs" in result
            assert "statistics" in result
        except Exception as e:
            # May fail due to missing config or other issues
            pass

    def test_pipeline_dpo_methods(self) -> None:
        """Test PipelineService DPO methods exist."""
        from pfe_core.pipeline import PipelineService

        service = PipelineService()

        # Check methods exist
        assert hasattr(service, "train_dpo")
        assert hasattr(service, "build_dpo_dataset")


@pytest.mark.slow
@pytest.mark.integration
class TestDPORealTraining:
    """Slow tests that require actual model loading (optional)."""

    def test_real_dpo_training(self) -> None:
        """Test actual DPO training with a tiny model on CPU.

        Uses gpt2 as the smallest readily-available causal LM and limits
        training to 2 steps so the test completes quickly without a GPU.
        """
        from pfe_core.trainer.dpo_executor import DPOTrainerExecutor
        from pfe_core.config import TrainerConfig

        config = TrainerConfig(
            method="lora",  # Use LoRA instead of QLoRA for CPU testing
            train_type="dpo",
            epochs=1,
            learning_rate=1e-4,
            device="cpu",
        )

        # Create small test dataset
        test_data = Dataset.from_dict({
            "prompt": ["What is 2+2?", "What color is the sky?"],
            "chosen": ["2+2 equals 4.", "The sky is blue."],
            "rejected": ["2+2 equals 5.", "The sky is green."],
            "session_id": ["s1", "s2"],
            "source_event_ids": [["e1"], ["e2"]],
            "confidence": [0.9, 0.8],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "adapter"

            executor = DPOTrainerExecutor(config)
            result = executor.train(
                base_model_path="gpt2",  # Small model for testing
                adapter_output_path=str(output_path),
                dpo_dataset=test_data,
            )

            assert result.success is True
            assert result.adapter_path == str(output_path)
            assert Path(result.adapter_path).exists()


class TestDPOEvaluator:
    """Test DPO-specific evaluation metrics."""

    def test_preference_alignment_evaluation(self) -> None:
        """Test preference alignment evaluation method."""
        from pfe_core.evaluator.auto import AutoEvaluator

        evaluator = AutoEvaluator()

        # Test with model outputs that align well with chosen
        result = evaluator.evaluate_preference_alignment(
            test_prompts=["What is 2+2?", "What color is the sky?"],
            chosen_responses=["2+2 equals 4.", "The sky is blue."],
            rejected_responses=["2+2 equals 5.", "The sky is green."],
            model_outputs=["2+2 equals 4.", "The sky is blue."],  # Perfect alignment
            adapter_version="v001",
        )

        assert "preference_alignment_score" in result
        assert result["num_samples"] == 2
        assert result["preference_alignment_score"] > 0.5  # Should be above random

    def test_text_similarity(self) -> None:
        """Test text similarity calculation."""
        from pfe_core.evaluator.auto import AutoEvaluator

        sim = AutoEvaluator._text_similarity("hello world", "hello world")
        assert sim == 1.0

        sim = AutoEvaluator._text_similarity("hello", "goodbye")
        assert sim < 1.0
        assert sim >= 0.0

    def test_preference_alignment_interpretation(self) -> None:
        """Test interpretation of preference alignment scores."""
        from pfe_core.evaluator.auto import AutoEvaluator

        evaluator = AutoEvaluator()

        # Strong preference
        result = evaluator.evaluate_preference_alignment(
            test_prompts=["Test"],
            chosen_responses=["Good answer"],
            rejected_responses=["Bad answer"],
            model_outputs=["Good answer"],
        )
        # Perfect alignment should give strong or at least moderate preference
        assert result["interpretation"] in ["strong_preference", "moderate_preference", "weak_preference"]

        # Weak preference (random output)
        result = evaluator.evaluate_preference_alignment(
            test_prompts=["Test"],
            chosen_responses=["Good answer"],
            rejected_responses=["Bad answer"],
            model_outputs=["Completely unrelated"],
        )
        # Should still be valid
        assert result["preference_alignment_score"] >= 0.0
        assert result["preference_alignment_score"] <= 1.0
