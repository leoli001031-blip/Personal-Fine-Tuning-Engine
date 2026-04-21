"""Forget detection for incremental training.

Implements replay-based forget detection following PockEngine-style monitoring.
Compares model performance on replay samples before and after training to detect
knowledge forgetting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import math


@dataclass
class ForgetMetrics:
    """Metrics for forget detection analysis."""

    loss_delta: float = 0.0
    """Difference in loss on replay samples (after - before). Positive = forgetting."""

    relative_increase: float = 0.0
    """Relative increase in loss (delta / before_loss)."""

    forget_detected: bool = False
    """Whether forgetting was detected based on thresholds."""

    replay_samples_count: int = 0
    """Number of replay samples evaluated."""

    before_loss: float = 0.0
    """Average loss on replay samples before training."""

    after_loss: float = 0.0
    """Average loss on replay samples after training."""

    per_sample_deltas: Dict[str, float] = field(default_factory=dict)
    """Per-sample loss deltas for detailed analysis."""

    threshold_used: float = 0.2
    """Threshold used for forget detection."""

    confidence: float = 0.0
    """Confidence score for forget detection (0-1)."""

    recommendation: str = ""
    """Recommended action based on detection results."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss_delta": self.loss_delta,
            "relative_increase": self.relative_increase,
            "forget_detected": self.forget_detected,
            "replay_samples_count": self.replay_samples_count,
            "before_loss": self.before_loss,
            "after_loss": self.after_loss,
            "per_sample_deltas": self.per_sample_deltas,
            "threshold_used": self.threshold_used,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
        }


@dataclass
class ReplaySample:
    """A replay sample with its metadata."""

    sample_id: str
    instruction: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ForgetDetector:
    """Detects forgetting in incremental training using replay loss comparison.

    Strategy: replay_loss
    - Record loss on replay samples before training
    - Record loss on replay samples after training
    - Compare: if loss increases significantly, forgetting is detected

    Threshold: loss_delta > 0.2 indicates forgetting

    Reference: PockEngine-style progressive training monitoring
    """

    DEFAULT_THRESHOLD: float = 0.2
    """Default threshold for loss delta to trigger forget detection."""

    HIGH_CONFIDENCE_THRESHOLD: float = 0.3
    """Threshold for high confidence forget detection."""

    def __init__(self, threshold: Optional[float] = None):
        """Initialize forget detector.

        Args:
            threshold: Loss delta threshold for forget detection. Default: 0.2
        """
        self.threshold = threshold or self.DEFAULT_THRESHOLD
        self._before_losses: Dict[str, float] = {}
        self._replay_samples: Dict[str, ReplaySample] = {}

    def record_before_training(
        self,
        samples: List[ReplaySample],
        losses: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record replay samples and their losses before training.

        Args:
            samples: List of replay samples to monitor
            losses: Optional pre-computed losses per sample_id
        """
        self._replay_samples = {s.sample_id: s for s in samples}
        if losses:
            self._before_losses = dict(losses)
        else:
            # Simulate loss computation if not provided
            self._before_losses = self._simulate_losses(samples)

    def detect_forget(
        self,
        after_losses: Optional[Dict[str, float]] = None,
        samples: Optional[List[ReplaySample]] = None,
    ) -> ForgetMetrics:
        """Detect forgetting by comparing before/after losses on replay samples.

        Args:
            after_losses: Losses on replay samples after training
            samples: Optional samples if not recorded in record_before_training

        Returns:
            ForgetMetrics with detection results
        """
        if samples:
            self._replay_samples = {s.sample_id: s for s in samples}

        if not self._replay_samples:
            return ForgetMetrics(
                forget_detected=False,
                recommendation="no_replay_samples",
                replay_samples_count=0,
            )

        # Get after-training losses
        if after_losses:
            after_losses_dict = dict(after_losses)
        else:
            after_losses_dict = self._simulate_losses(list(self._replay_samples.values()))

        # Calculate per-sample deltas
        per_sample_deltas: Dict[str, float] = {}
        valid_samples = 0
        total_before_loss = 0.0
        total_after_loss = 0.0

        for sample_id, sample in self._replay_samples.items():
            before_loss = self._before_losses.get(sample_id)
            after_loss = after_losses_dict.get(sample_id)

            if before_loss is None or after_loss is None:
                continue

            valid_samples += 1
            total_before_loss += before_loss
            total_after_loss += after_loss
            per_sample_deltas[sample_id] = after_loss - before_loss

        if valid_samples == 0:
            return ForgetMetrics(
                forget_detected=False,
                recommendation="insufficient_data",
                replay_samples_count=0,
            )

        # Calculate aggregate metrics
        avg_before_loss = total_before_loss / valid_samples
        avg_after_loss = total_after_loss / valid_samples
        loss_delta = avg_after_loss - avg_before_loss

        # Calculate relative increase
        relative_increase = (
            loss_delta / avg_before_loss if avg_before_loss > 0 else 0.0
        )

        # Determine if forgetting occurred
        forget_detected = loss_delta > self.threshold

        # Calculate confidence based on magnitude of change
        if loss_delta > self.HIGH_CONFIDENCE_THRESHOLD:
            confidence = min(1.0, 0.7 + (loss_delta - self.HIGH_CONFIDENCE_THRESHOLD) * 0.5)
        elif loss_delta > self.threshold:
            confidence = 0.5 + (loss_delta - self.threshold) / (self.HIGH_CONFIDENCE_THRESHOLD - self.threshold) * 0.2
        else:
            confidence = max(0.0, 0.3 - abs(loss_delta) * 0.5)

        # Generate recommendation
        if forget_detected:
            if loss_delta > self.HIGH_CONFIDENCE_THRESHOLD:
                recommendation = "rollback_required"
            else:
                recommendation = "increase_replay_ratio"
        else:
            recommendation = "continue"

        return ForgetMetrics(
            loss_delta=loss_delta,
            relative_increase=relative_increase,
            forget_detected=forget_detected,
            replay_samples_count=valid_samples,
            before_loss=avg_before_loss,
            after_loss=avg_after_loss,
            per_sample_deltas=per_sample_deltas,
            threshold_used=self.threshold,
            confidence=confidence,
            recommendation=recommendation,
        )

    def _simulate_losses(self, samples: List[ReplaySample]) -> Dict[str, float]:
        """Simulate losses for samples (for testing/mocking purposes).

        In real implementation, this would run model inference to compute losses.
        """
        losses: Dict[str, float] = {}
        for sample in samples:
            # Simulate base loss based on sample characteristics
            base_loss = 0.5 + (hash(sample.sample_id) % 100) / 1000.0
            losses[sample.sample_id] = base_loss
        return losses

    def compute_replay_ratio_adjustment(
        self,
        metrics: ForgetMetrics,
        current_ratio: float,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        """Compute adjusted replay ratio based on forget detection.

        Args:
            metrics: Forget detection metrics
            current_ratio: Current replay ratio
            user_profile: Optional user profile for personalization

        Returns:
            Tuple of (new_ratio, reason)
        """
        if not metrics.forget_detected:
            return current_ratio, "no_forget_detected"

        # Base adjustment based on severity
        if metrics.loss_delta > self.HIGH_CONFIDENCE_THRESHOLD:
            base_adjustment = 0.2
        else:
            base_adjustment = 0.1

        # Apply user profile adjustment if available
        profile_adjustment = 0.0
        if user_profile:
            # Users with consistent style may need less replay
            style_consistency = user_profile.get("style_consistency", 0.5)
            if style_consistency > 0.8:
                profile_adjustment = -0.05
            # Users with diverse domains may need more replay
            domain_diversity = user_profile.get("domain_diversity", 0.5)
            if domain_diversity > 0.7:
                profile_adjustment = 0.05

        new_ratio = min(1.0, current_ratio + base_adjustment + profile_adjustment)

        reason = f"forget_detected_delta_{metrics.loss_delta:.3f}"
        if profile_adjustment != 0.0:
            reason += f"_profile_adj_{profile_adjustment:+.2f}"

        return new_ratio, reason


def create_forget_detector(threshold: Optional[float] = None) -> ForgetDetector:
    """Factory function to create a forget detector."""
    return ForgetDetector(threshold=threshold)


def detect_forget_from_training_result(
    training_result: Dict[str, Any],
    threshold: Optional[float] = None,
) -> ForgetMetrics:
    """Convenience function to detect forget from training result.

    Args:
        training_result: Training result dictionary with replay samples and losses
        threshold: Optional custom threshold

    Returns:
        ForgetMetrics with detection results
    """
    detector = ForgetDetector(threshold=threshold)

    # Extract replay samples from training result
    replay_sample_ids = training_result.get("replay_sample_ids", [])
    dataset_plan = training_result.get("dataset_plan", {})

    samples = []
    for sample_id in replay_sample_ids:
        samples.append(ReplaySample(
            sample_id=str(sample_id),
            instruction="",
            output="",
            metadata={"from_training": True},
        ))

    # Simulate before/after losses
    detector.record_before_training(samples)
    return detector.detect_forget()
