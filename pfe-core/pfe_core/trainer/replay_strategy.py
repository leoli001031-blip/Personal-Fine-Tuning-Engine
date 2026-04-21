"""Dynamic replay strategy for incremental training.

Provides pluggable replay ratio strategies that adapt to forget detection
metrics, user profile signals, and training lineage state. The goal is to
balance retention of historical knowledge against absorption of new
preferences.

Strategies
----------
- ConservativeReplayStrategy: high replay ratio, prioritizes stability
- AggressiveReplayStrategy: low replay ratio, prioritizes new knowledge
- AdaptiveReplayStrategy (default): adjusts replay ratio based on
  ForgetMetrics with exponential time decay, user profile integration,
  and smooth adjustment to avoid violent fluctuations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .forget_detector import ForgetMetrics


@dataclass
class ReplayPlan:
    """Structured replay configuration produced by a strategy."""

    replay_ratio: float = 0.3
    """Target ratio of replay samples vs new samples."""

    replay_samples: List[Dict[str, Any]] = field(default_factory=list)
    """Selected replay sample dicts (or IDs) to include."""

    strategy_name: str = ""
    """Name of the strategy that produced this plan."""

    reason: str = ""
    """Human-readable explanation for the chosen replay ratio."""

    confidence: float = 0.0
    """Confidence in this plan (0-1)."""

    time_decay_applied: bool = False
    """Whether exponential time decay was applied to sample weights."""

    profile_adjustment: float = 0.0
    """Net adjustment applied from user profile signals."""

    smooth_adjustment_delta: float = 0.0
    """Delta after smoothing was applied."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replay_ratio": self.replay_ratio,
            "replay_samples_count": len(self.replay_samples),
            "strategy_name": self.strategy_name,
            "reason": self.reason,
            "confidence": self.confidence,
            "time_decay_applied": self.time_decay_applied,
            "profile_adjustment": self.profile_adjustment,
            "smooth_adjustment_delta": self.smooth_adjustment_delta,
        }


class ReplayStrategy(ABC):
    """Abstract base class for replay strategies."""

    name: str = ""

    @abstractmethod
    def compute_plan(
        self,
        *,
        current_ratio: float,
        forget_metrics: Optional[ForgetMetrics] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        replay_candidates: Optional[List[Dict[str, Any]]] = None,
        lineage_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ReplayPlan:
        """Compute a replay plan given current state and optional signals.

        Args:
            current_ratio: The current replay ratio in use.
            forget_metrics: Optional forget detection metrics from the last run.
            user_profile: Optional user profile dict for personalization.
            replay_candidates: Optional list of candidate replay samples.
            lineage_history: Optional list of previous training run summaries.

        Returns:
            ReplayPlan with the computed replay configuration.
        """
        ...

    def _apply_time_decay(
        self,
        candidates: List[Dict[str, Any]],
        decay_half_life_days: float = 7.0,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Apply exponential time decay to candidate samples based on age.

        Args:
            candidates: List of sample dicts; each may contain a
                ``created_at`` or ``timestamp`` ISO string.
            decay_half_life_days: Half-life in days for the decay.
            now: Reference datetime; defaults to UTC now.

        Returns:
            Candidates sorted by decayed weight (highest first), each
            annotated with ``decayed_weight`` in metadata.
        """
        if not candidates:
            return []

        now = now or datetime.now(timezone.utc)
        half_life_seconds = decay_half_life_days * 24 * 3600

        scored: List[tuple[float, Dict[str, Any]]] = []
        for sample in candidates:
            ts_str = (
                sample.get("created_at")
                or sample.get("timestamp")
                or sample.get("metadata", {}).get("created_at")
                or ""
            )
            age_seconds = half_life_seconds  # default: one half-life old
            if ts_str:
                try:
                    ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    age_seconds = max(0.0, (now - ts).total_seconds())
                except Exception:
                    pass

            weight = math.exp(-age_seconds / half_life_seconds) if half_life_seconds > 0 else 1.0
            annotated = dict(sample)
            meta = dict(annotated.get("metadata", {}))
            meta["decayed_weight"] = round(weight, 6)
            annotated["metadata"] = meta
            scored.append((weight, annotated))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]

    def _profile_adjustment(
        self,
        user_profile: Optional[Dict[str, Any]],
    ) -> float:
        """Compute net replay ratio adjustment from user profile.

        Adjustments:
        - style_consistency > 0.8  => -0.05 (stable style needs less replay)
        - domain_diversity > 0.7   => +0.05 (diverse domains need more replay)
        - preference_volatility > 0.6 => +0.08 (volatile preferences need replay)

        Returns:
            Net adjustment in the range [-0.15, +0.15].
        """
        if not user_profile:
            return 0.0

        adjustment = 0.0
        style_consistency = float(user_profile.get("style_consistency", 0.5))
        if style_consistency > 0.8:
            adjustment -= 0.05

        domain_diversity = float(user_profile.get("domain_diversity", 0.5))
        if domain_diversity > 0.7:
            adjustment += 0.05

        preference_volatility = float(user_profile.get("preference_volatility", 0.5))
        if preference_volatility > 0.6:
            adjustment += 0.08

        return max(-0.15, min(0.15, adjustment))

    def _smooth_adjustment(
        self,
        current_ratio: float,
        target_ratio: float,
        max_delta: float = 0.15,
    ) -> float:
        """Smoothly clamp the ratio change to avoid violent fluctuations.

        Args:
            current_ratio: Existing replay ratio.
            target_ratio: Desired replay ratio.
            max_delta: Maximum allowed absolute change per step.

        Returns:
            Smoothed replay ratio.
        """
        delta = target_ratio - current_ratio
        if abs(delta) > max_delta:
            delta = math.copysign(max_delta, delta)
        return max(0.0, min(1.0, current_ratio + delta))


class ConservativeReplayStrategy(ReplayStrategy):
    """High replay ratio strategy that prioritizes historical knowledge retention.

    Suitable for early training stages or when forget detection is
    consistently positive.
    """

    name = "conservative"

    def __init__(self, base_ratio: float = 0.5, max_ratio: float = 0.8) -> None:
        self.base_ratio = base_ratio
        self.max_ratio = max_ratio

    def compute_plan(
        self,
        *,
        current_ratio: float,
        forget_metrics: Optional[ForgetMetrics] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        replay_candidates: Optional[List[Dict[str, Any]]] = None,
        lineage_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ReplayPlan:
        target = self.base_ratio

        # If forgetting detected, push toward max
        if forget_metrics and forget_metrics.forget_detected:
            severity = min(1.0, forget_metrics.confidence)
            target = self.base_ratio + (self.max_ratio - self.base_ratio) * severity

        profile_adj = self._profile_adjustment(user_profile)
        target = max(0.0, min(1.0, target + profile_adj))
        smoothed = self._smooth_adjustment(current_ratio, target)

        candidates = replay_candidates or []
        if candidates:
            candidates = self._apply_time_decay(candidates)

        reason = f"conservative_base_{self.base_ratio}"
        if forget_metrics and forget_metrics.forget_detected:
            reason += f"_forget_boost_{forget_metrics.confidence:.2f}"
        if profile_adj != 0.0:
            reason += f"_profile_adj_{profile_adj:+.2f}"

        return ReplayPlan(
            replay_ratio=round(smoothed, 4),
            replay_samples=candidates,
            strategy_name=self.name,
            reason=reason,
            confidence=1.0,
            time_decay_applied=bool(candidates),
            profile_adjustment=profile_adj,
            smooth_adjustment_delta=round(smoothed - current_ratio, 4),
        )


class AggressiveReplayStrategy(ReplayStrategy):
    """Low replay ratio strategy that prioritizes new knowledge absorption.

    Suitable when the user is providing strong, consistent new signals
    and historical knowledge appears stable.
    """

    name = "aggressive"

    def __init__(self, base_ratio: float = 0.15, min_ratio: float = 0.05) -> None:
        self.base_ratio = base_ratio
        self.min_ratio = min_ratio

    def compute_plan(
        self,
        *,
        current_ratio: float,
        forget_metrics: Optional[ForgetMetrics] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        replay_candidates: Optional[List[Dict[str, Any]]] = None,
        lineage_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ReplayPlan:
        target = self.base_ratio

        # If forgetting detected, temporarily raise ratio but stay low
        if forget_metrics and forget_metrics.forget_detected:
            severity = min(1.0, forget_metrics.confidence)
            target = self.base_ratio + 0.15 * severity

        profile_adj = self._profile_adjustment(user_profile)
        target = max(self.min_ratio, min(1.0, target + profile_adj))
        smoothed = self._smooth_adjustment(current_ratio, target)

        candidates = replay_candidates or []
        if candidates:
            candidates = self._apply_time_decay(candidates)

        reason = f"aggressive_base_{self.base_ratio}"
        if forget_metrics and forget_metrics.forget_detected:
            reason += f"_forget_mitigate_{forget_metrics.confidence:.2f}"
        if profile_adj != 0.0:
            reason += f"_profile_adj_{profile_adj:+.2f}"

        return ReplayPlan(
            replay_ratio=round(smoothed, 4),
            replay_samples=candidates,
            strategy_name=self.name,
            reason=reason,
            confidence=1.0,
            time_decay_applied=bool(candidates),
            profile_adjustment=profile_adj,
            smooth_adjustment_delta=round(smoothed - current_ratio, 4),
        )


class AdaptiveReplayStrategy(ReplayStrategy):
    """Default adaptive replay strategy.

    Dynamically adjusts replay ratio based on:
    - ForgetMetrics (loss_delta, confidence, recommendation)
    - Exponential time decay on candidate replay samples
    - User profile integration (style_consistency, domain_diversity,
      preference_volatility)
    - Smooth adjustment to avoid violent fluctuations
    - Lineage history (recent forget trends)

    The strategy targets a balanced replay ratio and shifts toward
    conservative or aggressive poles based on observed signals.
    """

    name = "adaptive"

    def __init__(
        self,
        base_ratio: float = 0.3,
        min_ratio: float = 0.05,
        max_ratio: float = 0.7,
        forget_boost_factor: float = 0.25,
        lineage_window: int = 3,
        decay_half_life_days: float = 7.0,
    ) -> None:
        self.base_ratio = base_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.forget_boost_factor = forget_boost_factor
        self.lineage_window = lineage_window
        self.decay_half_life_days = decay_half_life_days

    def compute_plan(
        self,
        *,
        current_ratio: float,
        forget_metrics: Optional[ForgetMetrics] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        replay_candidates: Optional[List[Dict[str, Any]]] = None,
        lineage_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ReplayPlan:
        target = self.base_ratio
        confidence = 0.5
        reason_parts: List[str] = [f"adaptive_base_{self.base_ratio}"]

        # 1. Forget metrics adjustment
        if forget_metrics is not None:
            if forget_metrics.forget_detected:
                severity = min(1.0, forget_metrics.confidence)
                boost = self.forget_boost_factor * severity
                target += boost
                confidence = 0.5 + 0.5 * severity
                reason_parts.append(f"forget_boost_{boost:.3f}_conf_{severity:.2f}")

                # Extra boost for rollback_required recommendation
                if forget_metrics.recommendation == "rollback_required":
                    target += 0.1
                    confidence = min(1.0, confidence + 0.2)
                    reason_parts.append("rollback_required_extra_boost")
            else:
                # No forgetting: gently drift toward base
                target = current_ratio + (self.base_ratio - current_ratio) * 0.3
                confidence = 0.7
                reason_parts.append("no_forget_drift_to_base")

        # 2. Lineage history trend adjustment
        lineage_adj = self._lineage_adjustment(lineage_history)
        if lineage_adj != 0.0:
            target += lineage_adj
            reason_parts.append(f"lineage_adj_{lineage_adj:+.3f}")

        # 3. User profile adjustment
        profile_adj = self._profile_adjustment(user_profile)
        if profile_adj != 0.0:
            target += profile_adj
            reason_parts.append(f"profile_adj_{profile_adj:+.2f}")

        # Clamp to bounds
        target = max(self.min_ratio, min(self.max_ratio, target))

        # 4. Smooth adjustment
        smoothed = self._smooth_adjustment(current_ratio, target)
        smooth_delta = round(smoothed - current_ratio, 4)

        # 5. Time decay on candidates
        candidates = replay_candidates or []
        time_decay_applied = False
        if candidates:
            candidates = self._apply_time_decay(
                candidates, decay_half_life_days=self.decay_half_life_days
            )
            time_decay_applied = True

        return ReplayPlan(
            replay_ratio=round(smoothed, 4),
            replay_samples=candidates,
            strategy_name=self.name,
            reason="_".join(reason_parts),
            confidence=round(confidence, 4),
            time_decay_applied=time_decay_applied,
            profile_adjustment=profile_adj,
            smooth_adjustment_delta=smooth_delta,
        )

    def _lineage_adjustment(
        self,
        lineage_history: Optional[List[Dict[str, Any]]],
    ) -> float:
        """Adjust replay ratio based on recent lineage forget trends.

        If recent runs in the lineage show forgetting, increase replay.
        If recent runs are stable, allow slight decrease.

        Returns:
            Adjustment in the range [-0.05, +0.10].
        """
        if not lineage_history:
            return 0.0

        recent = lineage_history[-self.lineage_window :]
        forget_count = sum(1 for run in recent if run.get("forget_detected"))
        total = len(recent)
        if total == 0:
            return 0.0

        forget_rate = forget_count / total
        if forget_rate >= 0.5:
            return 0.10 * forget_rate
        elif forget_rate == 0.0:
            return -0.05
        return 0.0


def create_replay_strategy(
    name: str,
    **kwargs: Any,
) -> ReplayStrategy:
    """Factory to create a named replay strategy.

    Args:
        name: One of "adaptive", "conservative", "aggressive".
        **kwargs: Strategy-specific constructor arguments.

    Returns:
        ReplayStrategy instance.

    Raises:
        ValueError: If the strategy name is unknown.
    """
    mapping: Dict[str, type[ReplayStrategy]] = {
        "adaptive": AdaptiveReplayStrategy,
        "conservative": ConservativeReplayStrategy,
        "aggressive": AggressiveReplayStrategy,
    }
    cls = mapping.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown replay strategy '{name}'. "
            f"Choose from: {', '.join(mapping.keys())}"
        )
    return cls(**kwargs)


def get_strategy_summary(strategy: ReplayStrategy) -> Dict[str, Any]:
    """Return a summary dict for a strategy instance."""
    base: Dict[str, Any] = {"name": strategy.name}
    if isinstance(strategy, AdaptiveReplayStrategy):
        base.update(
            {
                "base_ratio": strategy.base_ratio,
                "min_ratio": strategy.min_ratio,
                "max_ratio": strategy.max_ratio,
                "forget_boost_factor": strategy.forget_boost_factor,
                "lineage_window": strategy.lineage_window,
                "decay_half_life_days": strategy.decay_half_life_days,
            }
        )
    elif isinstance(strategy, ConservativeReplayStrategy):
        base.update(
            {
                "base_ratio": strategy.base_ratio,
                "max_ratio": strategy.max_ratio,
            }
        )
    elif isinstance(strategy, AggressiveReplayStrategy):
        base.update(
            {
                "base_ratio": strategy.base_ratio,
                "min_ratio": strategy.min_ratio,
            }
        )
    return base
