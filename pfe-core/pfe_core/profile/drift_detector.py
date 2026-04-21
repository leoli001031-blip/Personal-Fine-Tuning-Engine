"""Profile drift detection for PFE Phase 2.

Compares historical and current user profiles across multiple dimensions to
detect preference drift, compute confidence scores, and recommend actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DimensionDrift:
    """Drift result for a single profile dimension."""

    dimension: str
    similarity: float
    similarity_delta: float
    confidence: float
    drift_score: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "similarity": round(self.similarity, 4),
            "similarity_delta": round(self.similarity_delta, 4),
            "confidence": round(self.confidence, 4),
            "drift_score": round(self.drift_score, 4),
            "details": self.details,
        }


@dataclass
class DriftReport:
    """Complete drift detection report."""

    drift_detected: bool
    overall_drift_score: float
    threshold: float
    recommendation: Literal["update_profile", "retrain", "monitor"]
    dimension_drifts: list[DimensionDrift] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "overall_drift_score": round(self.overall_drift_score, 4),
            "threshold": self.threshold,
            "recommendation": self.recommendation,
            "dimension_drifts": [d.to_dict() for d in self.dimension_drifts],
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ProfileDriftDetector:
    """Detects preference drift between historical and current user profiles.

    Supports four detection dimensions:
    - style_preference
    - domain_interests
    - communication_tone
    - long_term_goals

    Each dimension computes similarity, similarity change (delta), drift
    confidence, and a per-dimension drift score. An overall 0-1 drift score
    is produced along with a recommendation when drift exceeds threshold.
    """

    DEFAULT_DIMENSIONS: tuple[str, ...] = (
        "style_preference",
        "domain_interests",
        "communication_tone",
        "long_term_goals",
    )

    def __init__(
        self,
        threshold: float = 0.35,
        time_decay_hours: float = 168.0,
        dimensions: tuple[str, ...] | None = None,
    ):
        """Initialize detector.

        Args:
            threshold: Overall drift score above which drift is considered
                detected (0-1).
            time_decay_hours: Half-life for time-decay weighting in hours.
                More recent snapshots receive higher weight.
            dimensions: Dimensions to evaluate. Defaults to all four.
        """
        self.threshold = threshold
        self.time_decay_hours = time_decay_hours
        self.dimensions = dimensions or self.DEFAULT_DIMENSIONS

    def detect(
        self,
        historical_profile: dict[str, Any],
        current_profile: dict[str, Any],
        historical_timestamp: datetime | None = None,
    ) -> DriftReport:
        """Run drift detection between historical and current profiles.

        Args:
            historical_profile: Previously extracted profile dict.
            current_profile: Latest extracted profile dict.
            historical_timestamp: Optional timestamp of the historical profile
                used for time-decay weighting.

        Returns:
            DriftReport with per-dimension results and overall assessment.
        """
        dimension_drifts: list[DimensionDrift] = []
        total_weight = 0.0
        weighted_drift_sum = 0.0

        for dim in self.dimensions:
            dim_drift = self._compute_dimension_drift(
                historical_profile, current_profile, dim
            )
            weight = self._compute_time_weight(historical_timestamp)
            dimension_drifts.append(dim_drift)
            weighted_drift_sum += dim_drift.drift_score * weight
            total_weight += weight

        overall_drift_score = weighted_drift_sum / total_weight if total_weight > 0 else 0.0
        overall_drift_score = min(1.0, max(0.0, overall_drift_score))
        drift_detected = overall_drift_score > self.threshold
        recommendation = self._recommendation(overall_drift_score, drift_detected)

        return DriftReport(
            drift_detected=drift_detected,
            overall_drift_score=overall_drift_score,
            threshold=self.threshold,
            recommendation=recommendation,
            dimension_drifts=dimension_drifts,
            metadata={
                "dimensions_evaluated": list(self.dimensions),
                "time_decay_hours": self.time_decay_hours,
            },
        )

    def _compute_dimension_drift(
        self,
        historical: dict[str, Any],
        current: dict[str, Any],
        dimension: str,
    ) -> DimensionDrift:
        """Compute drift for a single dimension."""
        hist_items = self._extract_items(historical, dimension)
        curr_items = self._extract_items(current, dimension)

        # Handle empty / missing data gracefully
        if not hist_items and not curr_items:
            return DimensionDrift(
                dimension=dimension,
                similarity=1.0,
                similarity_delta=0.0,
                confidence=0.0,
                drift_score=0.0,
                details={"reason": "both_empty"},
            )

        if not hist_items or not curr_items:
            return DimensionDrift(
                dimension=dimension,
                similarity=0.0,
                similarity_delta=-1.0 if hist_items else 1.0,
                confidence=0.5,
                drift_score=0.5,
                details={"reason": "one_side_empty"},
            )

        similarity = self._compute_similarity(hist_items, curr_items, dimension)
        # Delta reflects change from perfect similarity (1.0)
        similarity_delta = similarity - 1.0
        # Confidence based on data richness (more items -> higher confidence)
        confidence = self._compute_confidence(hist_items, curr_items)
        # Drift score: how far from similarity, scaled by confidence
        drift_score = (1.0 - similarity) * confidence
        drift_score = min(1.0, max(0.0, drift_score))

        return DimensionDrift(
            dimension=dimension,
            similarity=similarity,
            similarity_delta=similarity_delta,
            confidence=confidence,
            drift_score=drift_score,
            details={
                "historical_count": len(hist_items),
                "current_count": len(curr_items),
            },
        )

    def _extract_items(
        self, profile: dict[str, Any], dimension: str
    ) -> list[dict[str, Any]]:
        """Extract structured items from profile for a given dimension."""
        if not isinstance(profile, dict):
            return []

        mapping: dict[str, list[str]] = {
            "style_preference": [
                "style_indicators",
                "style_preferences",
                "llm_style_indicators",
            ],
            "domain_interests": [
                "domain_interests",
                "domain_preferences",
                "llm_domain_interests",
            ],
            "communication_tone": [
                "response_preferences",
                "communication_tone",
                "llm_response_preferences",
            ],
            "long_term_goals": [
                "stable_preferences",
                "long_term_goals",
                "llm_stable_preferences",
            ],
        }

        items: list[dict[str, Any]] = []
        for key in mapping.get(dimension, []):
            value = profile.get(key)
            if isinstance(value, list):
                items.extend([v for v in value if isinstance(v, dict)])
            elif isinstance(value, dict):
                items.append(value)
        return items

    def _compute_similarity(
        self,
        hist_items: list[dict[str, Any]],
        curr_items: list[dict[str, Any]],
        dimension: str,
    ) -> float:
        """Compute cosine-like similarity between two sets of items."""
        hist_vec = self._items_to_vector(hist_items, dimension)
        curr_vec = self._items_to_vector(curr_items, dimension)

        if not hist_vec and not curr_vec:
            return 1.0
        if not hist_vec or not curr_vec:
            return 0.0

        all_keys = set(hist_vec.keys()) | set(curr_vec.keys())
        dot = 0.0
        hist_norm = 0.0
        curr_norm = 0.0

        for key in all_keys:
            h = hist_vec.get(key, 0.0)
            c = curr_vec.get(key, 0.0)
            dot += h * c
            hist_norm += h * h
            curr_norm += c * c

        if hist_norm == 0.0 or curr_norm == 0.0:
            return 0.0

        cosine = dot / (math.sqrt(hist_norm) * math.sqrt(curr_norm))
        # Clamp to [0, 1] since negative similarity is treated as 0
        return max(0.0, min(1.0, cosine))

    def _items_to_vector(
        self, items: list[dict[str, Any]], dimension: str
    ) -> dict[str, float]:
        """Convert list of items to a normalized feature vector."""
        vec: dict[str, float] = {}
        for item in items:
            if dimension == "style_preference":
                key = item.get("trait") or item.get("category") or item.get("style")
                val = self._extract_numeric(item, "level", "confidence", "score")
            elif dimension == "domain_interests":
                key = item.get("domain") or item.get("key")
                val = self._extract_numeric(item, "level", "confidence", "score")
            elif dimension == "communication_tone":
                key = item.get("aspect") or item.get("tone") or item.get("trait")
                val = self._extract_numeric(item, "confidence", "level", "score")
            elif dimension == "long_term_goals":
                key = item.get("category") or item.get("goal") or item.get("preference")
                val = self._extract_numeric(item, "confidence", "level", "score")
            else:
                key = None
                val = 0.0

            if key and isinstance(key, str):
                # Normalize key
                norm_key = key.strip().lower()
                if norm_key:
                    vec[norm_key] = max(vec.get(norm_key, 0.0), val)
        return vec

    @staticmethod
    def _extract_numeric(item: dict[str, Any], *fields: str) -> float:
        """Extract first available numeric value from item."""
        for f in fields:
            v = item.get(f)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return 0.5  # default neutral weight

    @staticmethod
    def _compute_confidence(
        hist_items: list[dict[str, Any]], curr_items: list[dict[str, Any]]
    ) -> float:
        """Compute confidence based on data richness."""
        total = len(hist_items) + len(curr_items)
        # Confidence grows with more data, capped at 1.0
        raw = 1.0 - math.exp(-total / 4.0)
        return round(min(1.0, max(0.0, raw)), 4)

    def _compute_time_weight(
        self, historical_timestamp: datetime | None
    ) -> float:
        """Compute time-decay weight: more recent snapshots get higher weight."""
        if historical_timestamp is None:
            return 1.0
        age_hours = (_utc_now() - historical_timestamp).total_seconds() / 3600.0
        # Exponential decay with half-life = time_decay_hours
        decay = (2.0 ** (-age_hours / self.time_decay_hours)) if self.time_decay_hours > 0 else 1.0
        # Weight is inverse of decay: recent = higher weight
        weight = 2.0 - decay
        return round(min(2.0, max(1.0, weight)), 4)

    def _recommendation(
        self, overall_drift_score: float, drift_detected: bool
    ) -> Literal["update_profile", "retrain", "monitor"]:
        """Recommend action based on drift severity."""
        if not drift_detected:
            return "monitor"
        if overall_drift_score > 0.7:
            return "retrain"
        return "update_profile"
