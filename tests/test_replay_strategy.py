"""Tests for dynamic replay strategy module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from pfe_core.trainer.replay_strategy import (
    AdaptiveReplayStrategy,
    AggressiveReplayStrategy,
    ConservativeReplayStrategy,
    ReplayPlan,
    ReplayStrategy,
    create_replay_strategy,
    get_strategy_summary,
)
from pfe_core.trainer.forget_detector import ForgetMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample(candidate_id: str, created_at: str | datetime | None = None) -> Dict[str, Any]:
    return {
        "id": candidate_id,
        "created_at": (
            created_at.isoformat() if isinstance(created_at, datetime) else (created_at or "")
        ),
        "metadata": {},
    }


def _profile(
    style_consistency: float = 0.5,
    domain_diversity: float = 0.5,
    preference_volatility: float = 0.5,
) -> Dict[str, Any]:
    return {
        "style_consistency": style_consistency,
        "domain_diversity": domain_diversity,
        "preference_volatility": preference_volatility,
    }


# ---------------------------------------------------------------------------
# 1. ReplayPlan
# ---------------------------------------------------------------------------


def test_replay_plan_defaults() -> None:
    plan = ReplayPlan()
    assert plan.replay_ratio == 0.3
    assert plan.replay_samples == []
    assert plan.strategy_name == ""
    assert plan.confidence == 0.0


def test_replay_plan_to_dict() -> None:
    plan = ReplayPlan(
        replay_ratio=0.45,
        strategy_name="adaptive",
        reason="test",
        confidence=0.8,
        time_decay_applied=True,
        profile_adjustment=0.05,
        smooth_adjustment_delta=0.1,
    )
    d = plan.to_dict()
    assert d["replay_ratio"] == 0.45
    assert d["strategy_name"] == "adaptive"
    assert d["replay_samples_count"] == 0
    assert d["time_decay_applied"] is True
    assert d["profile_adjustment"] == 0.05


# ---------------------------------------------------------------------------
# 2. ConservativeReplayStrategy
# ---------------------------------------------------------------------------


def test_conservative_base_ratio() -> None:
    strategy = ConservativeReplayStrategy(base_ratio=0.5)
    # current_ratio == base_ratio => no smoothing needed
    plan = strategy.compute_plan(current_ratio=0.5)
    assert plan.replay_ratio == 0.5
    assert plan.strategy_name == "conservative"
    assert "conservative_base_0.5" in plan.reason


def test_conservative_forget_boost() -> None:
    strategy = ConservativeReplayStrategy(base_ratio=0.5, max_ratio=0.8)
    metrics = ForgetMetrics(forget_detected=True, confidence=0.8, loss_delta=0.25)
    # Start close to expected target (0.74) so smoothing does not clamp
    plan = strategy.compute_plan(current_ratio=0.65, forget_metrics=metrics)
    # base 0.5 + (0.8-0.5)*0.8 = 0.74
    assert plan.replay_ratio == pytest.approx(0.74, abs=0.01)
    assert plan.confidence == 1.0
    assert "forget_boost" in plan.reason


def test_conservative_profile_adjustment() -> None:
    strategy = ConservativeReplayStrategy(base_ratio=0.5)
    profile = _profile(style_consistency=0.9)  # -0.05
    plan = strategy.compute_plan(current_ratio=0.5, user_profile=profile)
    assert plan.replay_ratio == pytest.approx(0.45, abs=0.01)
    assert plan.profile_adjustment == pytest.approx(-0.05, abs=0.001)


def test_conservative_time_decay() -> None:
    strategy = ConservativeReplayStrategy()
    now = datetime.now(timezone.utc)
    candidates = [
        _sample("old", now - timedelta(days=14)),
        _sample("recent", now - timedelta(days=1)),
    ]
    plan = strategy.compute_plan(current_ratio=0.3, replay_candidates=candidates)
    assert plan.time_decay_applied is True
    weights = [s["metadata"]["decayed_weight"] for s in plan.replay_samples]
    assert weights[0] > weights[1]  # recent should have higher weight


def test_conservative_smooth_clamp() -> None:
    strategy = ConservativeReplayStrategy(base_ratio=0.5)
    # Large jump from 0.1 to 0.5 should be smoothed to 0.25 (max_delta 0.15)
    plan = strategy.compute_plan(current_ratio=0.1)
    assert plan.replay_ratio == pytest.approx(0.25, abs=0.01)


# ---------------------------------------------------------------------------
# 3. AggressiveReplayStrategy
# ---------------------------------------------------------------------------


def test_aggressive_base_ratio() -> None:
    strategy = AggressiveReplayStrategy(base_ratio=0.15)
    plan = strategy.compute_plan(current_ratio=0.3)
    assert plan.replay_ratio == 0.15
    assert plan.strategy_name == "aggressive"


def test_aggressive_forget_mitigate() -> None:
    strategy = AggressiveReplayStrategy(base_ratio=0.15)
    metrics = ForgetMetrics(forget_detected=True, confidence=0.6, loss_delta=0.22)
    plan = strategy.compute_plan(current_ratio=0.15, forget_metrics=metrics)
    # base 0.15 + 0.15*0.6 = 0.24
    assert plan.replay_ratio == pytest.approx(0.24, abs=0.01)
    assert "forget_mitigate" in plan.reason


def test_aggressive_min_ratio_floor() -> None:
    strategy = AggressiveReplayStrategy(base_ratio=0.05, min_ratio=0.05)
    profile = _profile(style_consistency=0.9)  # -0.05 would push below min
    plan = strategy.compute_plan(current_ratio=0.05, user_profile=profile)
    assert plan.replay_ratio == pytest.approx(0.05, abs=0.01)


def test_aggressive_profile_domain_diversity() -> None:
    strategy = AggressiveReplayStrategy(base_ratio=0.15)
    profile = _profile(domain_diversity=0.8)  # +0.05
    plan = strategy.compute_plan(current_ratio=0.15, user_profile=profile)
    assert plan.replay_ratio == pytest.approx(0.20, abs=0.01)


# ---------------------------------------------------------------------------
# 4. AdaptiveReplayStrategy
# ---------------------------------------------------------------------------


def test_adaptive_base_no_signals() -> None:
    strategy = AdaptiveReplayStrategy(base_ratio=0.3)
    plan = strategy.compute_plan(current_ratio=0.3)
    assert plan.replay_ratio == pytest.approx(0.3, abs=0.01)
    assert plan.strategy_name == "adaptive"
    assert "adaptive_base_0.3" in plan.reason


def test_adaptive_forget_detected() -> None:
    strategy = AdaptiveReplayStrategy(base_ratio=0.3, forget_boost_factor=0.25)
    metrics = ForgetMetrics(forget_detected=True, confidence=0.8, loss_delta=0.25)
    # Start close to expected target (0.5) so smoothing does not clamp
    plan = strategy.compute_plan(current_ratio=0.4, forget_metrics=metrics)
    # base 0.3 + 0.25*0.8 = 0.5
    assert plan.replay_ratio == pytest.approx(0.5, abs=0.01)
    assert plan.confidence == pytest.approx(0.9, abs=0.01)


def test_adaptive_rollback_required_extra_boost() -> None:
    strategy = AdaptiveReplayStrategy(base_ratio=0.3)
    metrics = ForgetMetrics(
        forget_detected=True, confidence=0.9, loss_delta=0.35, recommendation="rollback_required"
    )
    # Start close to expected target (0.625) so smoothing does not clamp
    plan = strategy.compute_plan(current_ratio=0.55, forget_metrics=metrics)
    # base 0.3 + 0.25*0.9 + 0.1 = 0.625, clamped to max 0.7
    assert plan.replay_ratio == pytest.approx(0.625, abs=0.01)
    assert "rollback_required_extra_boost" in plan.reason


def test_adaptive_no_forget_drift() -> None:
    strategy = AdaptiveReplayStrategy(base_ratio=0.3)
    metrics = ForgetMetrics(forget_detected=False, confidence=0.1, loss_delta=0.05)
    plan = strategy.compute_plan(current_ratio=0.5, forget_metrics=metrics)
    # drift from 0.5 toward 0.3 by 30%: 0.5 + (0.3-0.5)*0.3 = 0.44
    assert plan.replay_ratio == pytest.approx(0.44, abs=0.01)
    assert "no_forget_drift_to_base" in plan.reason


def test_adaptive_profile_volatility() -> None:
    strategy = AdaptiveReplayStrategy(base_ratio=0.3)
    profile = _profile(preference_volatility=0.7)  # +0.08
    plan = strategy.compute_plan(current_ratio=0.3, user_profile=profile)
    assert plan.replay_ratio == pytest.approx(0.38, abs=0.01)
    assert plan.profile_adjustment == pytest.approx(0.08, abs=0.001)


def test_adaptive_lineage_forget_trend() -> None:
    strategy = AdaptiveReplayStrategy(base_ratio=0.3, lineage_window=3)
    lineage = [
        {"forget_detected": True},
        {"forget_detected": True},
        {"forget_detected": False},
    ]
    plan = strategy.compute_plan(current_ratio=0.3, lineage_history=lineage)
    # 2/3 forget rate >= 0.5 => +0.10 * (2/3) = +0.0667
    assert plan.replay_ratio == pytest.approx(0.3667, abs=0.01)
    assert "lineage_adj" in plan.reason


def test_adaptive_lineage_stable_trend() -> None:
    strategy = AdaptiveReplayStrategy(base_ratio=0.3, lineage_window=3)
    lineage = [
        {"forget_detected": False},
        {"forget_detected": False},
        {"forget_detected": False},
    ]
    plan = strategy.compute_plan(current_ratio=0.3, lineage_history=lineage)
    # 0/3 forget rate == 0 => -0.05
    assert plan.replay_ratio == pytest.approx(0.25, abs=0.01)


def test_adaptive_smooth_clamp_up() -> None:
    strategy = AdaptiveReplayStrategy(base_ratio=0.3)
    # current very low, target high -> smoothed to current + 0.15
    metrics = ForgetMetrics(forget_detected=True, confidence=1.0, loss_delta=0.35)
    plan = strategy.compute_plan(current_ratio=0.1, forget_metrics=metrics)
    # target would be 0.3 + 0.25*1.0 = 0.55, but smoothed from 0.1 => 0.25
    assert plan.replay_ratio == pytest.approx(0.25, abs=0.01)


def test_adaptive_smooth_clamp_down() -> None:
    strategy = AdaptiveReplayStrategy(base_ratio=0.3)
    # current very high, no forget -> drift toward base but clamped
    metrics = ForgetMetrics(forget_detected=False, confidence=0.1, loss_delta=0.02)
    plan = strategy.compute_plan(current_ratio=0.6, forget_metrics=metrics)
    # drift target = 0.6 + (0.3-0.6)*0.3 = 0.51, delta = -0.09, within clamp
    assert plan.replay_ratio == pytest.approx(0.51, abs=0.01)


def test_adaptive_time_decay_sorting() -> None:
    strategy = AdaptiveReplayStrategy(decay_half_life_days=7.0)
    now = datetime.now(timezone.utc)
    candidates = [
        _sample("a", now - timedelta(days=10)),
        _sample("b", now - timedelta(days=3)),
        _sample("c", now - timedelta(days=0)),
    ]
    plan = strategy.compute_plan(current_ratio=0.3, replay_candidates=candidates)
    ids = [s["id"] for s in plan.replay_samples]
    assert ids == ["c", "b", "a"]
    weights = [s["metadata"]["decayed_weight"] for s in plan.replay_samples]
    assert weights[0] > weights[1] > weights[2]


def test_adaptive_bounds_clamping() -> None:
    strategy = AdaptiveReplayStrategy(base_ratio=0.3, min_ratio=0.05, max_ratio=0.7)
    # Extreme forget should not exceed max_ratio
    metrics = ForgetMetrics(
        forget_detected=True, confidence=1.0, loss_delta=0.5, recommendation="rollback_required"
    )
    plan = strategy.compute_plan(current_ratio=0.3, forget_metrics=metrics)
    assert plan.replay_ratio <= 0.7
    assert plan.replay_ratio >= 0.05


# ---------------------------------------------------------------------------
# 5. Factory & Summary
# ---------------------------------------------------------------------------


def test_create_strategy_adaptive() -> None:
    s = create_replay_strategy("adaptive", base_ratio=0.25)
    assert isinstance(s, AdaptiveReplayStrategy)
    assert s.base_ratio == 0.25


def test_create_strategy_conservative() -> None:
    s = create_replay_strategy("conservative", base_ratio=0.6)
    assert isinstance(s, ConservativeReplayStrategy)
    assert s.base_ratio == 0.6


def test_create_strategy_aggressive() -> None:
    s = create_replay_strategy("aggressive", base_ratio=0.1)
    assert isinstance(s, AggressiveReplayStrategy)
    assert s.base_ratio == 0.1


def test_create_strategy_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown replay strategy"):
        create_replay_strategy("unknown")


def test_get_strategy_summary_adaptive() -> None:
    s = AdaptiveReplayStrategy(base_ratio=0.25, max_ratio=0.6)
    summary = get_strategy_summary(s)
    assert summary["name"] == "adaptive"
    assert summary["base_ratio"] == 0.25
    assert summary["max_ratio"] == 0.6


def test_get_strategy_summary_conservative() -> None:
    s = ConservativeReplayStrategy(base_ratio=0.5, max_ratio=0.8)
    summary = get_strategy_summary(s)
    assert summary["name"] == "conservative"
    assert summary["base_ratio"] == 0.5


def test_get_strategy_summary_aggressive() -> None:
    s = AggressiveReplayStrategy(base_ratio=0.15, min_ratio=0.05)
    summary = get_strategy_summary(s)
    assert summary["name"] == "aggressive"
    assert summary["min_ratio"] == 0.05


# ---------------------------------------------------------------------------
# 6. Edge Cases
# ---------------------------------------------------------------------------


def test_empty_candidates() -> None:
    strategy = AdaptiveReplayStrategy()
    plan = strategy.compute_plan(current_ratio=0.3, replay_candidates=[])
    assert plan.replay_samples == []
    assert plan.time_decay_applied is False


def test_none_inputs() -> None:
    strategy = AdaptiveReplayStrategy()
    plan = strategy.compute_plan(
        current_ratio=0.3,
        forget_metrics=None,
        user_profile=None,
        replay_candidates=None,
        lineage_history=None,
    )
    assert plan.replay_ratio == pytest.approx(0.3, abs=0.01)


def test_time_decay_missing_timestamp() -> None:
    strategy = AdaptiveReplayStrategy()
    candidates = [{"id": "x", "metadata": {}}]
    plan = strategy.compute_plan(current_ratio=0.3, replay_candidates=candidates)
    # Default age = one half-life => weight = exp(-1) ~ 0.3679
    weight = plan.replay_samples[0]["metadata"]["decayed_weight"]
    assert weight == pytest.approx(0.3679, abs=0.01)
