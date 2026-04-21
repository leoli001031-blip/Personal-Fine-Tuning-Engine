"""Automatic rollback policy for incremental training.

Encapsulates rollback decision logic with configurable thresholds,
enabling automatic or operator-assisted rollback when forgetting is detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..adapter_store.lifecycle import rollback_to_version


@dataclass
class RollbackDecision:
    """Result of an automatic rollback policy evaluation."""

    should_rollback: bool = False
    action: str = "continue"
    reason: str = ""
    confidence: float = 0.0
    fallback_version: str | None = None
    policy_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_rollback": self.should_rollback,
            "action": self.action,
            "reason": self.reason,
            "confidence": self.confidence,
            "fallback_version": self.fallback_version,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RollbackDecision":
        return cls(
            should_rollback=data.get("should_rollback", False),
            action=data.get("action", "continue"),
            reason=data.get("reason", ""),
            confidence=data.get("confidence", 0.0),
            fallback_version=data.get("fallback_version"),
            policy_version=data.get("policy_version", "1.0"),
        )


class AutoRollbackPolicy:
    """Configurable automatic rollback policy for forget detection.

    Evaluates forget detection metrics against configurable thresholds
    to decide whether to trigger rollback, increase replay, warn, or continue.
    """

    ACTIONS = frozenset({"rollback", "increase_replay", "warn", "continue"})

    def __init__(
        self,
        enabled: bool = False,
        threshold: float = 0.2,
        high_confidence_threshold: float = 0.3,
        auto_rollback_on_high_confidence: bool = False,
        min_versions_before_rollback: int = 2,
        require_eval_failure: bool = False,
    ):
        """Initialize the rollback policy.

        Args:
            enabled: Master switch for automatic rollback.
            threshold: Loss delta threshold for forget detection.
            high_confidence_threshold: Threshold for high-confidence forget detection.
            auto_rollback_on_high_confidence: If True, auto-rollback when confidence > 0.7.
            min_versions_before_rollback: Minimum versions needed before rollback allowed.
            require_eval_failure: If True, also require eval failure before rollback.
        """
        self.enabled = enabled
        self.threshold = threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.auto_rollback_on_high_confidence = auto_rollback_on_high_confidence
        self.min_versions_before_rollback = min_versions_before_rollback
        self.require_eval_failure = require_eval_failure

    def should_rollback(
        self,
        forget_metrics: dict[str, Any] | None,
        num_versions: int = 0,
        eval_failed: bool = False,
    ) -> RollbackDecision:
        """Evaluate whether rollback should be triggered.

        Args:
            forget_metrics: Forget detection metrics dict (from ForgetMetrics.to_dict()).
            num_versions: Number of existing adapter versions.
            eval_failed: Whether evaluation also failed.

        Returns:
            RollbackDecision with action recommendation.
        """
        if forget_metrics is None:
            return RollbackDecision(
                should_rollback=False,
                action="continue",
                reason="no_forget_metrics",
                confidence=0.0,
            )

        forget_detected = forget_metrics.get("forget_detected", False)
        loss_delta = forget_metrics.get("loss_delta", 0.0)
        confidence = forget_metrics.get("confidence", 0.0)
        recommendation = forget_metrics.get("recommendation", "continue")

        # Not enough versions to rollback safely
        if num_versions < self.min_versions_before_rollback:
            return RollbackDecision(
                should_rollback=False,
                action="warn" if forget_detected else "continue",
                reason=f"insufficient_versions ({num_versions} < {self.min_versions_before_rollback})",
                confidence=confidence,
            )

        # No forgetting detected
        if not forget_detected:
            return RollbackDecision(
                should_rollback=False,
                action="continue",
                reason="no_forgetting_detected",
                confidence=confidence,
            )

        # Forgetting detected - determine severity and action
        if confidence > 0.7 and self.auto_rollback_on_high_confidence:
            if self.require_eval_failure and not eval_failed:
                return RollbackDecision(
                    should_rollback=False,
                    action="warn",
                    reason="high_confidence_forget_but_eval_passed",
                    confidence=confidence,
                )
            return RollbackDecision(
                should_rollback=True,
                action="rollback",
                reason="high_confidence_forget_detected_auto_rollback",
                confidence=confidence,
            )

        if recommendation == "rollback_required":
            if self.require_eval_failure and not eval_failed:
                return RollbackDecision(
                    should_rollback=False,
                    action="warn",
                    reason="rollback_required_but_eval_passed",
                    confidence=confidence,
                )
            return RollbackDecision(
                should_rollback=self.enabled,
                action="rollback" if self.enabled else "warn",
                reason="rollback_required_by_metrics",
                confidence=confidence,
            )

        if recommendation == "increase_replay_ratio":
            return RollbackDecision(
                should_rollback=False,
                action="increase_replay",
                reason="moderate_forget_increase_replay",
                confidence=confidence,
            )

        # Fallback for any other recommendation
        return RollbackDecision(
            should_rollback=False,
            action="warn",
            reason=f"forget_detected_recommendation_{recommendation}",
            confidence=confidence,
        )

    def select_fallback_version(
        self,
        current_version: str,
        lineage_tracker: Any,
    ) -> str | None:
        """Select the best fallback version using lineage information.

        Args:
            current_version: The version that exhibited forgetting.
            lineage_tracker: AdapterLineageTracker instance.

        Returns:
            Best fallback version string, or None if no suitable fallback found.
        """
        if lineage_tracker is None:
            return None

        # Try the immediate parent first
        node = lineage_tracker.get_node(current_version)
        if node and node.parent_version:
            parent = lineage_tracker.get_node(node.parent_version)
            if parent and not parent.forget_detected:
                return parent.version

        # Find the most recent non-forgetting ancestor
        lineage = lineage_tracker.get_lineage(current_version)
        for ancestor in reversed(lineage[:-1]):  # Exclude current version
            if not ancestor.forget_detected:
                return ancestor.version

        # Fallback: most recent ancestor regardless
        if len(lineage) >= 2:
            return lineage[-2].version

        return None

    def execute_rollback(
        self,
        store: Any,
        current_version: str,
        fallback_version: str | None,
        reason: str = "auto_rollback_policy",
    ) -> dict[str, Any]:
        """Execute rollback with policy-level logging.

        Args:
            store: AdapterStore instance.
            current_version: Version to roll back from.
            fallback_version: Version to roll back to.
            reason: Reason for rollback.

        Returns:
            Rollback operation result dict.
        """
        result = rollback_to_version(
            store=store,
            current_version=current_version,
            fallback_version=fallback_version,
            reason=reason,
        )
        result["policy"] = {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "auto_rollback_on_high_confidence": self.auto_rollback_on_high_confidence,
        }
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "auto_rollback_on_high_confidence": self.auto_rollback_on_high_confidence,
            "min_versions_before_rollback": self.min_versions_before_rollback,
            "require_eval_failure": self.require_eval_failure,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoRollbackPolicy":
        return cls(
            enabled=data.get("enabled", False),
            threshold=data.get("threshold", 0.2),
            high_confidence_threshold=data.get("high_confidence_threshold", 0.3),
            auto_rollback_on_high_confidence=data.get("auto_rollback_on_high_confidence", False),
            min_versions_before_rollback=data.get("min_versions_before_rollback", 2),
            require_eval_failure=data.get("require_eval_failure", False),
        )


# Global policy instance
_auto_rollback_policy: AutoRollbackPolicy | None = None


def get_auto_rollback_policy(
    enabled: bool = False,
    threshold: float = 0.2,
    auto_rollback_on_high_confidence: bool = False,
) -> AutoRollbackPolicy:
    """Get or create the global auto-rollback policy.

    Args:
        enabled: Master switch for automatic rollback.
        threshold: Loss delta threshold.
        auto_rollback_on_high_confidence: Auto-rollback on high confidence.

    Returns:
        AutoRollbackPolicy instance.
    """
    global _auto_rollback_policy
    if _auto_rollback_policy is None:
        _auto_rollback_policy = AutoRollbackPolicy(
            enabled=enabled,
            threshold=threshold,
            auto_rollback_on_high_confidence=auto_rollback_on_high_confidence,
        )
    return _auto_rollback_policy
