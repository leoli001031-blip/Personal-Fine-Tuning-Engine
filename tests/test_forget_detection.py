"""Tests for forget detection functionality.

Tests the ForgetDetector class and its integration with the training service.
"""

import pytest
from unittest.mock import MagicMock, patch

from pfe_core.trainer.forget_detector import (
    ForgetDetector,
    ForgetMetrics,
    ReplaySample,
    create_forget_detector,
    detect_forget_from_training_result,
)
from pfe_core.adapter_store.lifecycle import rollback_to_version
from pfe_core.errors import AdapterError


class TestForgetDetector:
    """Test cases for ForgetDetector class."""

    def test_create_forget_detector_default_threshold(self):
        """Test factory creates detector with default threshold."""
        detector = create_forget_detector()
        assert detector.threshold == ForgetDetector.DEFAULT_THRESHOLD
        assert detector.threshold == 0.2

    def test_create_forget_detector_custom_threshold(self):
        """Test factory creates detector with custom threshold."""
        detector = create_forget_detector(threshold=0.3)
        assert detector.threshold == 0.3

    def test_record_before_training(self):
        """Test recording samples before training."""
        detector = ForgetDetector()
        samples = [
            ReplaySample(sample_id="s1", instruction="test1", output="out1"),
            ReplaySample(sample_id="s2", instruction="test2", output="out2"),
        ]
        detector.record_before_training(samples)

        assert len(detector._replay_samples) == 2
        assert "s1" in detector._before_losses
        assert "s2" in detector._before_losses

    def test_detect_forget_no_forgetting(self):
        """Test detection when no forgetting occurred."""
        detector = ForgetDetector(threshold=0.2)
        samples = [
            ReplaySample(sample_id="s1", instruction="test1", output="out1"),
            ReplaySample(sample_id="s2", instruction="test2", output="out2"),
        ]

        # Simulate before losses
        detector.record_before_training(samples)
        before_losses = detector._before_losses.copy()

        # Simulate after losses (lower = better = no forgetting)
        after_losses = {sid: loss * 0.8 for sid, loss in before_losses.items()}

        metrics = detector.detect_forget(after_losses)

        assert metrics.forget_detected is False
        assert metrics.loss_delta < 0  # Loss decreased
        assert metrics.recommendation == "continue"

    def test_detect_forget_with_forgetting(self):
        """Test detection when forgetting occurred."""
        detector = ForgetDetector(threshold=0.2)
        samples = [
            ReplaySample(sample_id="s1", instruction="test1", output="out1"),
            ReplaySample(sample_id="s2", instruction="test2", output="out2"),
        ]

        detector.record_before_training(samples)
        before_losses = detector._before_losses.copy()

        # Simulate after losses (higher = worse = forgetting)
        after_losses = {sid: loss + 0.3 for sid, loss in before_losses.items()}

        metrics = detector.detect_forget(after_losses)

        assert metrics.forget_detected is True
        assert metrics.loss_delta > 0.2
        assert metrics.recommendation in ["rollback_required", "increase_replay_ratio"]

    def test_detect_forget_high_confidence(self):
        """Test high confidence forget detection."""
        detector = ForgetDetector(threshold=0.2)
        samples = [
            ReplaySample(sample_id="s1", instruction="test1", output="out1"),
        ]

        detector.record_before_training(samples)
        before_loss = detector._before_losses["s1"]

        # Large increase triggers high confidence
        after_losses = {"s1": before_loss + 0.5}

        metrics = detector.detect_forget(after_losses)

        assert metrics.forget_detected is True
        assert metrics.loss_delta > ForgetDetector.HIGH_CONFIDENCE_THRESHOLD
        assert metrics.confidence > 0.7
        assert metrics.recommendation == "rollback_required"

    def test_detect_forget_no_samples(self):
        """Test detection with no replay samples."""
        detector = ForgetDetector()
        metrics = detector.detect_forget()

        assert metrics.forget_detected is False
        assert metrics.recommendation == "no_replay_samples"
        assert metrics.replay_samples_count == 0

    def test_compute_replay_ratio_adjustment_no_forget(self):
        """Test ratio adjustment when no forgetting."""
        detector = ForgetDetector()
        metrics = ForgetMetrics(forget_detected=False, loss_delta=0.05)

        new_ratio, reason = detector.compute_replay_ratio_adjustment(
            metrics, current_ratio=0.3
        )

        assert new_ratio == 0.3  # Unchanged
        assert "no_forget" in reason

    def test_compute_replay_ratio_adjustment_with_forget(self):
        """Test ratio adjustment when forgetting detected."""
        detector = ForgetDetector()
        metrics = ForgetMetrics(forget_detected=True, loss_delta=0.25)

        new_ratio, reason = detector.compute_replay_ratio_adjustment(
            metrics, current_ratio=0.3
        )

        assert new_ratio > 0.3  # Increased
        assert "forget_detected" in reason

    def test_compute_replay_ratio_adjustment_with_profile(self):
        """Test ratio adjustment with user profile."""
        detector = ForgetDetector()
        metrics = ForgetMetrics(forget_detected=True, loss_delta=0.25)
        profile = {"style_consistency": 0.9, "domain_diversity": 0.3}

        new_ratio, reason = detector.compute_replay_ratio_adjustment(
            metrics, current_ratio=0.3, user_profile=profile
        )

        # High style consistency reduces replay need
        assert "profile_adj" in reason

    def test_forget_metrics_to_dict(self):
        """Test ForgetMetrics serialization."""
        metrics = ForgetMetrics(
            loss_delta=0.25,
            relative_increase=0.5,
            forget_detected=True,
            replay_samples_count=10,
            recommendation="rollback_required",
        )

        d = metrics.to_dict()

        assert d["loss_delta"] == 0.25
        assert d["relative_increase"] == 0.5
        assert d["forget_detected"] is True
        assert d["replay_samples_count"] == 10
        assert d["recommendation"] == "rollback_required"


class TestDetectForgetFromTrainingResult:
    """Test convenience function for training result integration."""

    def test_detect_from_training_result(self):
        """Test forget detection from training result dict."""
        training_result = {
            "replay_sample_ids": ["s1", "s2", "s3"],
            "dataset_plan": {"selected_replay_ratio": 0.3},
        }

        metrics = detect_forget_from_training_result(training_result)

        assert isinstance(metrics, ForgetMetrics)
        assert metrics.replay_samples_count == 3

    def test_detect_from_empty_training_result(self):
        """Test forget detection with empty result."""
        training_result = {}

        metrics = detect_forget_from_training_result(training_result)

        assert metrics.replay_samples_count == 0
        assert metrics.recommendation == "no_replay_samples"


class TestRollbackToVersion:
    """Test rollback functionality."""

    def test_rollback_to_version_success(self):
        """Test successful rollback."""
        mock_store = MagicMock()
        mock_store.archive.return_value = "Archived 20240101-001"
        mock_store.promote.return_value = "Promoted 20231231-001"

        result = rollback_to_version(
            store=mock_store,
            current_version="20240101-001",
            fallback_version="20231231-001",
            reason="forget_detected",
        )

        assert result["success"] is True
        assert result["promoted_version"] == "20231231-001"
        assert len(result["actions"]) == 2
        mock_store.archive.assert_called_once_with("20240101-001")
        mock_store.promote.assert_called_once_with("20231231-001")

    def test_rollback_auto_find_fallback(self):
        """Test rollback with automatic fallback discovery."""
        mock_store = MagicMock()
        mock_store.list_version_records.return_value = [
            {"version": "20240101-001", "state": "pending_eval"},
            {"version": "20231231-001", "state": "promoted"},
        ]
        mock_store.archive.return_value = "Archived 20240101-001"
        mock_store.promote.return_value = "Promoted 20231231-001"

        result = rollback_to_version(
            store=mock_store,
            current_version="20240101-001",
            fallback_version=None,
        )

        assert result["success"] is True
        assert result["promoted_version"] == "20231231-001"

    def test_rollback_no_fallback_found(self):
        """Test rollback when no fallback available."""
        mock_store = MagicMock()
        mock_store.list_version_records.return_value = []
        mock_store.archive.return_value = "Archived 20240101-001"

        result = rollback_to_version(
            store=mock_store,
            current_version="20240101-001",
            fallback_version=None,
        )

        assert result["success"] is False
        assert "fallback_error" in result

    def test_rollback_archive_failure_continue(self):
        """Test rollback continues even if archive fails."""
        mock_store = MagicMock()
        mock_store.archive.side_effect = AdapterError("Cannot archive promoted version")
        mock_store.promote.return_value = "Promoted 20231231-001"

        result = rollback_to_version(
            store=mock_store,
            current_version="20240101-001",
            fallback_version="20231231-001",
        )

        # Should still succeed with promote
        assert result["success"] is True
        assert len(result["actions"]) == 2
        assert "error" in result["actions"][0]  # Archive action has error

    def test_rollback_promote_failure(self):
        """Test rollback when promote fails."""
        mock_store = MagicMock()
        mock_store.archive.return_value = "Archived 20240101-001"
        mock_store.promote.side_effect = AdapterError("Cannot promote")

        result = rollback_to_version(
            store=mock_store,
            current_version="20240101-001",
            fallback_version="20231231-001",
        )

        assert result["success"] is False
        assert "promote_error" in result


class TestForgetDetectionIntegration:
    """Integration tests for forget detection with training service."""

    def test_training_result_includes_forget_detection(self):
        """Test that TrainingRunResult includes forget detection fields."""
        from pfe_core.trainer.service import TrainingRunResult

        # Create mock forget metrics
        mock_metrics = ForgetMetrics(
            forget_detected=True,
            loss_delta=0.25,
            recommendation="rollback_required",
        )

        # Create a minimal TrainingRunResult
        result = TrainingRunResult(
            version="20240101-001",
            adapter_path="/tmp/test",
            num_samples=10,
            metrics={},
            runtime={},
            backend_plan={},
            backend_dispatch={},
            executor_spec={},
            execution_backend="mock_local",
            execution_executor="mock_local",
            executor_mode="phase0_mock",
            job_bundle={},
            job_execution={},
            job_execution_summary={},
            real_execution_summary={},
            export_runtime={},
            export_command_plan={},
            export_execution={},
            export_toolchain_summary={},
            export_write={},
            requires_export_step=False,
            training_config={},
            incremental_context=None,
            audit_info={},
            forget_detected=True,
            forget_metrics=mock_metrics.to_dict(),
            replay_ratio_adjusted=False,
        )

        assert result.forget_detected is True
        assert result.forget_metrics is not None
        assert result.forget_metrics["forget_detected"] is True

    def test_replay_ratio_adjusted_based_on_profile(self):
        """Test that replay ratio is adjusted based on user profile."""
        from pfe_core.trainer.service import TrainerService

        service = TrainerService()

        # Mock user profile with style consistency
        profile_adjustments = service._get_profile_adjusted_params("user_123")

        # Should return a dict (may be empty if profile store not available)
        assert isinstance(profile_adjustments, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
