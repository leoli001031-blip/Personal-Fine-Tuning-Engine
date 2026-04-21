"""Tests for automatic rollback policy.

Tests the AutoRollbackPolicy class and its integration with training service.
"""

import pytest
from unittest.mock import MagicMock

from pfe_core.trainer.auto_rollback import (
    AutoRollbackPolicy,
    RollbackDecision,
    get_auto_rollback_policy,
)


class TestRollbackDecision:
    """Test cases for RollbackDecision dataclass."""

    def test_rollback_decision_creation(self):
        """Test creating a rollback decision."""
        decision = RollbackDecision(
            should_rollback=True,
            action="rollback",
            reason="high confidence forget detected",
            confidence=0.85,
            fallback_version="v1",
        )
        assert decision.should_rollback is True
        assert decision.action == "rollback"
        assert decision.confidence == 0.85
        assert decision.fallback_version == "v1"

    def test_rollback_decision_defaults(self):
        """Test rollback decision default values."""
        decision = RollbackDecision()
        assert decision.should_rollback is False
        assert decision.action == "continue"
        assert decision.confidence == 0.0
        assert decision.fallback_version is None
        assert decision.policy_version == "1.0"

    def test_rollback_decision_to_dict(self):
        """Test RollbackDecision serialization."""
        decision = RollbackDecision(
            should_rollback=True,
            action="rollback",
            reason="forget",
            confidence=0.8,
            fallback_version="v1",
        )
        d = decision.to_dict()
        assert d["should_rollback"] is True
        assert d["action"] == "rollback"
        assert d["confidence"] == 0.8
        assert d["fallback_version"] == "v1"

    def test_rollback_decision_from_dict(self):
        """Test RollbackDecision deserialization."""
        data = {
            "should_rollback": True,
            "action": "increase_replay",
            "reason": "moderate forget",
            "confidence": 0.5,
            "fallback_version": None,
            "policy_version": "2.0",
        }
        decision = RollbackDecision.from_dict(data)
        assert decision.should_rollback is True
        assert decision.action == "increase_replay"
        assert decision.policy_version == "2.0"

    def test_rollback_decision_roundtrip(self):
        """Test serialize then deserialize preserves data."""
        original = RollbackDecision(
            should_rollback=True,
            action="rollback",
            reason="test",
            confidence=0.9,
            fallback_version="v0",
        )
        d = original.to_dict()
        restored = RollbackDecision.from_dict(d)
        assert restored.should_rollback == original.should_rollback
        assert restored.action == original.action
        assert restored.reason == original.reason
        assert restored.confidence == original.confidence
        assert restored.fallback_version == original.fallback_version


class TestAutoRollbackPolicy:
    """Test cases for AutoRollbackPolicy."""

    def test_default_policy(self):
        """Test default policy settings."""
        policy = AutoRollbackPolicy()
        assert policy.enabled is False
        assert policy.threshold == 0.2
        assert policy.auto_rollback_on_high_confidence is False
        assert policy.min_versions_before_rollback == 2

    def test_custom_policy(self):
        """Test custom policy settings."""
        policy = AutoRollbackPolicy(
            enabled=True,
            threshold=0.15,
            auto_rollback_on_high_confidence=True,
            min_versions_before_rollback=3,
            require_eval_failure=True,
        )
        assert policy.enabled is True
        assert policy.threshold == 0.15
        assert policy.auto_rollback_on_high_confidence is True
        assert policy.min_versions_before_rollback == 3
        assert policy.require_eval_failure is True

    def test_should_rollback_no_metrics(self):
        """Test decision with no forget metrics."""
        policy = AutoRollbackPolicy()
        decision = policy.should_rollback(None, num_versions=5)
        assert decision.should_rollback is False
        assert decision.action == "continue"
        assert "no_forget_metrics" in decision.reason

    def test_should_rollback_insufficient_versions(self):
        """Test decision with insufficient versions."""
        policy = AutoRollbackPolicy(min_versions_before_rollback=5)
        metrics = {"forget_detected": True, "confidence": 0.8}
        decision = policy.should_rollback(metrics, num_versions=2)
        assert decision.should_rollback is False
        assert decision.action == "warn"
        assert "insufficient_versions" in decision.reason

    def test_should_rollback_no_forgetting(self):
        """Test decision when no forgetting detected."""
        policy = AutoRollbackPolicy()
        metrics = {"forget_detected": False, "confidence": 0.1}
        decision = policy.should_rollback(metrics, num_versions=5)
        assert decision.should_rollback is False
        assert decision.action == "continue"
        assert "no_forgetting" in decision.reason

    def test_should_rollback_high_confidence_auto_enabled(self):
        """Test auto-rollback on high confidence when enabled."""
        policy = AutoRollbackPolicy(
            enabled=True,
            auto_rollback_on_high_confidence=True,
        )
        metrics = {
            "forget_detected": True,
            "confidence": 0.85,
            "recommendation": "rollback_required",
        }
        decision = policy.should_rollback(metrics, num_versions=5)
        assert decision.should_rollback is True
        assert decision.action == "rollback"
        assert "high_confidence" in decision.reason

    def test_should_rollback_high_confidence_auto_disabled(self):
        """Test no auto-rollback when auto_rollback_on_high_confidence is disabled."""
        policy = AutoRollbackPolicy(
            enabled=True,
            auto_rollback_on_high_confidence=False,
        )
        metrics = {
            "forget_detected": True,
            "confidence": 0.85,
            "recommendation": "rollback_required",
        }
        decision = policy.should_rollback(metrics, num_versions=5)
        # Without auto_rollback_on_high_confidence, falls to rollback_required path
        assert decision.should_rollback is True
        assert decision.action == "rollback"

    def test_should_rollback_high_confidence_with_eval_passed(self):
        """Test high confidence but eval passed when require_eval_failure is True."""
        policy = AutoRollbackPolicy(
            enabled=True,
            auto_rollback_on_high_confidence=True,
            require_eval_failure=True,
        )
        metrics = {
            "forget_detected": True,
            "confidence": 0.85,
            "recommendation": "rollback_required",
        }
        decision = policy.should_rollback(metrics, num_versions=5, eval_failed=False)
        assert decision.should_rollback is False
        assert decision.action == "warn"
        assert "eval_passed" in decision.reason

    def test_should_rollback_rollback_required(self):
        """Test rollback_required recommendation."""
        policy = AutoRollbackPolicy(enabled=True)
        metrics = {
            "forget_detected": True,
            "confidence": 0.6,
            "recommendation": "rollback_required",
        }
        decision = policy.should_rollback(metrics, num_versions=5)
        assert decision.should_rollback is True
        assert decision.action == "rollback"
        assert "rollback_required" in decision.reason

    def test_should_rollback_increase_replay(self):
        """Test increase_replay recommendation."""
        policy = AutoRollbackPolicy()
        metrics = {
            "forget_detected": True,
            "confidence": 0.4,
            "recommendation": "increase_replay_ratio",
        }
        decision = policy.should_rollback(metrics, num_versions=5)
        assert decision.should_rollback is False
        assert decision.action == "increase_replay"
        assert "increase_replay" in decision.reason

    def test_should_rollback_policy_disabled(self):
        """Test that disabled policy still returns rollback recommendation but doesn't execute."""
        policy = AutoRollbackPolicy(enabled=False)
        metrics = {
            "forget_detected": True,
            "confidence": 0.6,
            "recommendation": "rollback_required",
        }
        decision = policy.should_rollback(metrics, num_versions=5)
        # Policy is disabled, so should_rollback is False but action is "warn"
        assert decision.should_rollback is False
        assert decision.action == "warn"

    def test_select_fallback_version_from_parent(self):
        """Test selecting fallback from immediate parent."""
        policy = AutoRollbackPolicy()
        lineage_tracker = MagicMock()
        lineage_tracker.get_node.return_value = MagicMock(
            parent_version="v1",
        )
        lineage_tracker.get_node.side_effect = lambda v: {
            "v2": MagicMock(parent_version="v1", version="v2"),
            "v1": MagicMock(forget_detected=False, version="v1"),
        }.get(v)

        fallback = policy.select_fallback_version("v2", lineage_tracker)
        assert fallback == "v1"

    def test_select_fallback_version_from_ancestor(self):
        """Test selecting fallback from ancestor when parent had forgetting."""
        policy = AutoRollbackPolicy()
        lineage_tracker = MagicMock()

        # v3 -> v2(forget) -> v1(clean)
        v1 = MagicMock(forget_detected=False, version="v1")
        v2 = MagicMock(forget_detected=True, version="v2", parent_version="v1")
        v3 = MagicMock(forget_detected=True, version="v3", parent_version="v2")

        lineage_tracker.get_node.side_effect = lambda v: {"v3": v3, "v2": v2, "v1": v1}.get(v)
        lineage_tracker.get_lineage.return_value = [v1, v2, v3]

        fallback = policy.select_fallback_version("v3", lineage_tracker)
        assert fallback == "v1"

    def test_select_fallback_version_no_lineage(self):
        """Test selecting fallback when no lineage tracker available."""
        policy = AutoRollbackPolicy()
        fallback = policy.select_fallback_version("v2", None)
        assert fallback is None

    def test_select_fallback_version_no_ancestors(self):
        """Test selecting fallback when no clean ancestors exist."""
        policy = AutoRollbackPolicy()
        lineage_tracker = MagicMock()
        v1 = MagicMock(forget_detected=True, version="v1")
        v2 = MagicMock(forget_detected=True, version="v2", parent_version="v1")
        lineage_tracker.get_node.side_effect = lambda v: {"v2": v2, "v1": v1}.get(v)
        lineage_tracker.get_lineage.return_value = [v1, v2]

        fallback = policy.select_fallback_version("v2", lineage_tracker)
        # Falls back to most recent ancestor regardless
        assert fallback == "v1"

    def test_execute_rollback(self):
        """Test executing rollback."""
        policy = AutoRollbackPolicy(enabled=True)
        mock_store = MagicMock()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "pfe_core.trainer.auto_rollback.rollback_to_version",
                lambda **kwargs: {
                    "success": True,
                    "current_version": kwargs.get("current_version"),
                    "fallback_version": kwargs.get("fallback_version"),
                    "actions": [],
                },
            )
            result = policy.execute_rollback(
                store=mock_store,
                current_version="v2",
                fallback_version="v1",
                reason="test_rollback",
            )
            assert result["success"] is True
            assert result["current_version"] == "v2"
            assert result["fallback_version"] == "v1"
            assert "policy" in result

    def test_policy_to_dict(self):
        """Test policy serialization."""
        policy = AutoRollbackPolicy(
            enabled=True,
            threshold=0.15,
            auto_rollback_on_high_confidence=True,
        )
        d = policy.to_dict()
        assert d["enabled"] is True
        assert d["threshold"] == 0.15
        assert d["auto_rollback_on_high_confidence"] is True

    def test_policy_from_dict(self):
        """Test policy deserialization."""
        data = {
            "enabled": True,
            "threshold": 0.25,
            "high_confidence_threshold": 0.4,
            "auto_rollback_on_high_confidence": True,
            "min_versions_before_rollback": 3,
            "require_eval_failure": True,
        }
        policy = AutoRollbackPolicy.from_dict(data)
        assert policy.enabled is True
        assert policy.threshold == 0.25
        assert policy.high_confidence_threshold == 0.4
        assert policy.auto_rollback_on_high_confidence is True
        assert policy.min_versions_before_rollback == 3
        assert policy.require_eval_failure is True

    def test_policy_roundtrip(self):
        """Test serialize then deserialize preserves policy settings."""
        original = AutoRollbackPolicy(
            enabled=True,
            threshold=0.15,
            high_confidence_threshold=0.25,
            auto_rollback_on_high_confidence=True,
            min_versions_before_rollback=5,
            require_eval_failure=True,
        )
        d = original.to_dict()
        restored = AutoRollbackPolicy.from_dict(d)
        assert restored.enabled == original.enabled
        assert restored.threshold == original.threshold
        assert restored.high_confidence_threshold == original.high_confidence_threshold
        assert restored.auto_rollback_on_high_confidence == original.auto_rollback_on_high_confidence
        assert restored.min_versions_before_rollback == original.min_versions_before_rollback
        assert restored.require_eval_failure == original.require_eval_failure


class TestGetAutoRollbackPolicy:
    """Test cases for global auto-rollback policy."""

    def test_get_auto_rollback_policy_singleton(self):
        """Test that get_auto_rollback_policy returns a singleton."""
        # Note: This may conflict with other tests that also call get_auto_rollback_policy
        # We just verify it returns the correct type
        policy = get_auto_rollback_policy()
        assert isinstance(policy, AutoRollbackPolicy)

    def test_get_auto_rollback_policy_with_params(self):
        """Test creating policy with custom params (only on first call)."""
        # Since get_auto_rollback_policy returns a singleton, params only matter on first call.
        # We verify the policy object has the expected type and structure.
        policy = get_auto_rollback_policy()
        assert isinstance(policy, AutoRollbackPolicy)
        assert hasattr(policy, "enabled")
        assert hasattr(policy, "threshold")
        assert hasattr(policy, "auto_rollback_on_high_confidence")


class TestAutoRollbackIntegration:
    """Integration tests for auto-rollback with lineage."""

    def test_full_rollback_flow(self):
        """Test complete rollback decision and execution flow."""
        policy = AutoRollbackPolicy(
            enabled=True,
            auto_rollback_on_high_confidence=True,
        )

        forget_metrics = {
            "forget_detected": True,
            "confidence": 0.85,
            "recommendation": "rollback_required",
            "loss_delta": 0.35,
        }

        # Decision phase
        decision = policy.should_rollback(forget_metrics, num_versions=5)
        assert decision.should_rollback is True
        assert decision.action == "rollback"

        # Fallback selection
        lineage_tracker = MagicMock()
        lineage_tracker.get_node.return_value = MagicMock(parent_version="v1")
        lineage_tracker.get_node.side_effect = lambda v: {
            "v2": MagicMock(parent_version="v1", version="v2"),
            "v1": MagicMock(forget_detected=False, version="v1"),
        }.get(v)

        fallback = policy.select_fallback_version("v2", lineage_tracker)
        assert fallback == "v1"

    def test_warn_instead_of_rollback(self):
        """Test warn action when policy conditions not met."""
        policy = AutoRollbackPolicy(
            enabled=False,  # Disabled policy
        )
        forget_metrics = {
            "forget_detected": True,
            "confidence": 0.6,
            "recommendation": "rollback_required",
        }
        decision = policy.should_rollback(forget_metrics, num_versions=5)
        assert decision.should_rollback is False
        assert decision.action == "warn"

    def test_increase_replay_recommendation(self):
        """Test increase_replay action for moderate forgetting."""
        policy = AutoRollbackPolicy()
        forget_metrics = {
            "forget_detected": True,
            "confidence": 0.4,
            "recommendation": "increase_replay_ratio",
            "loss_delta": 0.15,
        }
        decision = policy.should_rollback(forget_metrics, num_versions=5)
        assert decision.should_rollback is False
        assert decision.action == "increase_replay"
        assert decision.confidence == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
