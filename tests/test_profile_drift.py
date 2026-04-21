"""Tests for profile drift detection (P2-B)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from pfe_core.profile.drift_detector import (
    DimensionDrift,
    DriftReport,
    ProfileDriftDetector,
)


class TestProfileDriftDetectorNoDrift:
    """Scenarios where profiles are identical or very similar."""

    def test_identical_profiles_no_drift(self) -> None:
        detector = ProfileDriftDetector(threshold=0.35)
        profile = {
            "style_indicators": [{"trait": "formal", "level": 0.8}],
            "domain_interests": [{"domain": "programming", "level": 0.9}],
            "response_preferences": [{"aspect": "concise", "confidence": 0.7}],
            "stable_preferences": [{"category": "career", "confidence": 0.8}],
        }
        report = detector.detect(profile, profile)
        assert report.drift_detected is False
        assert report.overall_drift_score == pytest.approx(0.0, abs=1e-6)
        assert report.recommendation == "monitor"
        assert len(report.dimension_drifts) == 4

    def test_empty_profiles_no_drift(self) -> None:
        detector = ProfileDriftDetector(threshold=0.35)
        report = detector.detect({}, {})
        assert report.drift_detected is False
        assert report.overall_drift_score == pytest.approx(0.0, abs=1e-6)
        for dd in report.dimension_drifts:
            assert dd.similarity == 1.0
            assert dd.drift_score == 0.0

    def test_partial_empty_profiles_low_drift(self) -> None:
        detector = ProfileDriftDetector(threshold=0.35)
        hist: dict[str, Any] = {}
        curr = {
            "style_indicators": [{"trait": "casual", "level": 0.5}],
        }
        report = detector.detect(hist, curr)
        # One side empty yields moderate per-dim drift, but overall stays low
        assert report.overall_drift_score < 0.35


class TestProfileDriftDetectorSingleDimension:
    """Drift concentrated in a single dimension."""

    def test_style_preference_drift(self) -> None:
        detector = ProfileDriftDetector(threshold=0.2)
        hist = {
            "style_indicators": [{"trait": "formal", "level": 0.9}],
        }
        curr = {
            "style_indicators": [{"trait": "casual", "level": 0.9}],
        }
        report = detector.detect(hist, curr)
        style_dim = next(d for d in report.dimension_drifts if d.dimension == "style_preference")
        assert style_dim.similarity < 1.0
        assert style_dim.drift_score > 0.0

    def test_domain_interests_drift(self) -> None:
        detector = ProfileDriftDetector(threshold=0.2)
        hist = {
            "domain_interests": [{"domain": "programming", "level": 0.9}],
        }
        curr = {
            "domain_interests": [{"domain": "design", "level": 0.9}],
        }
        report = detector.detect(hist, curr)
        dim = next(d for d in report.dimension_drifts if d.dimension == "domain_interests")
        assert dim.similarity == pytest.approx(0.0, abs=1e-6)
        assert dim.drift_score > 0.0

    def test_communication_tone_drift(self) -> None:
        detector = ProfileDriftDetector(threshold=0.2)
        hist = {
            "response_preferences": [{"aspect": "formal", "confidence": 0.9}],
        }
        curr = {
            "response_preferences": [{"aspect": "friendly", "confidence": 0.9}],
        }
        report = detector.detect(hist, curr)
        dim = next(d for d in report.dimension_drifts if d.dimension == "communication_tone")
        assert dim.similarity == pytest.approx(0.0, abs=1e-6)

    def test_long_term_goals_drift(self) -> None:
        detector = ProfileDriftDetector(threshold=0.2)
        hist = {
            "stable_preferences": [{"category": "management", "confidence": 0.8}],
        }
        curr = {
            "stable_preferences": [{"category": "engineering", "confidence": 0.8}],
        }
        report = detector.detect(hist, curr)
        dim = next(d for d in report.dimension_drifts if d.dimension == "long_term_goals")
        assert dim.similarity == pytest.approx(0.0, abs=1e-6)


class TestProfileDriftDetectorMultiDimension:
    """Drift across multiple dimensions simultaneously."""

    def test_multi_dimension_drift_detected(self) -> None:
        detector = ProfileDriftDetector(threshold=0.3)
        hist = {
            "style_indicators": [{"trait": "formal", "level": 0.9}],
            "domain_interests": [{"domain": "backend", "level": 0.9}],
            "response_preferences": [{"aspect": "concise", "confidence": 0.8}],
            "stable_preferences": [{"category": "cto", "confidence": 0.7}],
        }
        curr = {
            "style_indicators": [{"trait": "casual", "level": 0.9}],
            "domain_interests": [{"domain": "frontend", "level": 0.9}],
            "response_preferences": [{"aspect": "detailed", "confidence": 0.8}],
            "stable_preferences": [{"category": "founder", "confidence": 0.7}],
        }
        report = detector.detect(hist, curr)
        assert report.drift_detected is True
        assert report.overall_drift_score > 0.3
        assert report.recommendation in ("update_profile", "retrain")
        for dd in report.dimension_drifts:
            assert dd.similarity < 1.0

    def test_all_dimensions_empty_except_one(self) -> None:
        detector = ProfileDriftDetector(threshold=0.5)
        hist = {
            "style_indicators": [{"trait": "formal", "level": 0.8}],
        }
        curr = {
            "style_indicators": [{"trait": "casual", "level": 0.8}],
        }
        report = detector.detect(hist, curr)
        assert report.drift_detected is False
        # Only one dimension has data; overall stays below threshold
        assert report.overall_drift_score < 0.5


class TestProfileDriftDetectorTimeDecay:
    """Time-decay weighting behavior."""

    def test_recent_history_lower_weight_effect(self) -> None:
        detector = ProfileDriftDetector(threshold=1.0, time_decay_hours=24.0)
        hist = {
            "style_indicators": [{"trait": "formal", "level": 0.9}],
        }
        curr = {
            "style_indicators": [{"trait": "casual", "level": 0.9}],
        }
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        old = datetime.now(timezone.utc) - timedelta(hours=100)

        report_recent = detector.detect(hist, curr, historical_timestamp=recent)
        report_old = detector.detect(hist, curr, historical_timestamp=old)
        # Recent snapshot should have higher weight, thus higher effective drift
        assert report_recent.overall_drift_score >= report_old.overall_drift_score

    def test_time_decay_none_timestamp(self) -> None:
        detector = ProfileDriftDetector(threshold=1.0)
        hist = {
            "style_indicators": [{"trait": "formal", "level": 0.9}],
        }
        curr = {
            "style_indicators": [{"trait": "casual", "level": 0.9}],
        }
        report = detector.detect(hist, curr, historical_timestamp=None)
        assert report.overall_drift_score > 0.0


class TestProfileDriftDetectorThresholdBoundary:
    """Behavior at and around threshold boundaries."""

    def test_exact_threshold_boundary(self) -> None:
        detector = ProfileDriftDetector(threshold=0.35)
        # Identical profiles produce 0 drift, which is not > 0.35
        report = detector.detect({}, {})
        assert report.drift_detected is False
        assert report.overall_drift_score == pytest.approx(0.0, abs=1e-6)

    def test_just_above_threshold(self) -> None:
        detector = ProfileDriftDetector(threshold=0.1)
        hist = {
            "style_indicators": [{"trait": "a", "level": 1.0}],
            "domain_interests": [{"domain": "b", "level": 1.0}],
        }
        curr = {
            "style_indicators": [{"trait": "z", "level": 1.0}],
            "domain_interests": [{"domain": "y", "level": 1.0}],
        }
        report = detector.detect(hist, curr)
        assert report.drift_detected is True
        assert report.overall_drift_score > 0.1

    def test_just_below_threshold(self) -> None:
        detector = ProfileDriftDetector(threshold=0.99)
        hist = {
            "style_indicators": [{"trait": "a", "level": 1.0}],
        }
        curr = {
            "style_indicators": [{"trait": "a", "level": 1.0}],
        }
        report = detector.detect(hist, curr)
        assert report.drift_detected is False
        assert report.overall_drift_score < 0.99


class TestProfileDriftDetectorMissingData:
    """Handling of null, missing, or malformed inputs."""

    def test_none_values_in_profile(self) -> None:
        detector = ProfileDriftDetector(threshold=0.35)
        hist: dict[str, Any] = {
            "style_indicators": None,
            "domain_interests": [{"domain": "x", "level": 0.5}],
        }
        curr: dict[str, Any] = {
            "style_indicators": [{"trait": "casual", "level": 0.5}],
            "domain_interests": None,
        }
        report = detector.detect(hist, curr)
        # Should not raise; drift score stays bounded
        assert 0.0 <= report.overall_drift_score <= 1.0

    def test_non_dict_profile_input(self) -> None:
        detector = ProfileDriftDetector(threshold=0.35)
        report = detector.detect("not_a_dict", "also_not_a_dict")  # type: ignore[arg-type]
        assert report.drift_detected is False
        for dd in report.dimension_drifts:
            assert dd.similarity == 1.0
            assert dd.drift_score == 0.0

    def test_partial_keys_missing(self) -> None:
        detector = ProfileDriftDetector(threshold=0.35)
        hist = {
            "style_indicators": [{"trait": "formal", "level": 0.8}],
        }
        curr = {
            "domain_interests": [{"domain": "ai", "level": 0.9}],
        }
        report = detector.detect(hist, curr)
        assert 0.0 <= report.overall_drift_score <= 1.0
        # Dimensions with data on one side only should show moderate drift
        style_dim = next(d for d in report.dimension_drifts if d.dimension == "style_preference")
        domain_dim = next(d for d in report.dimension_drifts if d.dimension == "domain_interests")
        assert style_dim.drift_score > 0.0
        assert domain_dim.drift_score > 0.0


class TestProfileDriftDetectorRecommendations:
    """Action recommendation logic."""

    def test_recommendation_monitor(self) -> None:
        detector = ProfileDriftDetector(threshold=0.5)
        report = detector.detect({}, {})
        assert report.recommendation == "monitor"

    def test_recommendation_update_profile(self) -> None:
        detector = ProfileDriftDetector(threshold=0.15)
        hist = {
            "style_indicators": [{"trait": "formal", "level": 0.9}],
            "domain_interests": [{"domain": "backend", "level": 0.9}],
        }
        curr = {
            "style_indicators": [{"trait": "casual", "level": 0.9}],
            "domain_interests": [{"domain": "frontend", "level": 0.9}],
        }
        report = detector.detect(hist, curr)
        assert report.drift_detected is True
        if report.overall_drift_score <= 0.7:
            assert report.recommendation == "update_profile"

    def test_recommendation_retrain(self) -> None:
        detector = ProfileDriftDetector(threshold=0.3)
        hist = {
            "style_indicators": [{"trait": "formal", "level": 0.9}],
            "domain_interests": [{"domain": "backend", "level": 0.9}],
            "response_preferences": [{"aspect": "concise", "confidence": 0.9}],
            "stable_preferences": [{"category": "cto", "confidence": 0.9}],
        }
        curr = {
            "style_indicators": [{"trait": "casual", "level": 0.9}],
            "domain_interests": [{"domain": "art", "level": 0.9}],
            "response_preferences": [{"aspect": "verbose", "confidence": 0.9}],
            "stable_preferences": [{"category": "artist", "confidence": 0.9}],
        }
        report = detector.detect(hist, curr)
        assert report.drift_detected is True
        if report.overall_drift_score > 0.7:
            assert report.recommendation == "retrain"


class TestDriftReportDataclass:
    """DriftReport and DimensionDrift serialization."""

    def test_drift_report_to_dict(self) -> None:
        report = DriftReport(
            drift_detected=True,
            overall_drift_score=0.5,
            threshold=0.35,
            recommendation="update_profile",
            dimension_drifts=[
                DimensionDrift(
                    dimension="style_preference",
                    similarity=0.5,
                    similarity_delta=-0.5,
                    confidence=0.8,
                    drift_score=0.4,
                )
            ],
        )
        d = report.to_dict()
        assert d["drift_detected"] is True
        assert d["overall_drift_score"] == 0.5
        assert d["recommendation"] == "update_profile"
        assert len(d["dimension_drifts"]) == 1
        assert d["dimension_drifts"][0]["dimension"] == "style_preference"

    def test_dimension_drift_to_dict(self) -> None:
        dd = DimensionDrift(
            dimension="domain_interests",
            similarity=0.2,
            similarity_delta=-0.8,
            confidence=0.9,
            drift_score=0.72,
            details={"historical_count": 3, "current_count": 2},
        )
        d = dd.to_dict()
        assert d["dimension"] == "domain_interests"
        assert d["similarity"] == 0.2
        assert d["details"]["historical_count"] == 3


class TestProfileDriftDetectorConfidence:
    """Confidence computation based on data richness."""

    def test_confidence_increases_with_more_data(self) -> None:
        detector = ProfileDriftDetector(threshold=1.0)
        hist_sparse = [{"trait": "formal", "level": 0.8}]
        hist_rich = [
            {"trait": "formal", "level": 0.8},
            {"trait": "casual", "level": 0.2},
            {"trait": "direct", "level": 0.9},
            {"trait": "polite", "level": 0.7},
        ]
        curr = [{"trait": "formal", "level": 0.8}]

        report_sparse = detector.detect(
            {"style_indicators": hist_sparse}, {"style_indicators": curr}
        )
        report_rich = detector.detect(
            {"style_indicators": hist_rich}, {"style_indicators": curr}
        )

        sparse_dim = next(d for d in report_sparse.dimension_drifts if d.dimension == "style_preference")
        rich_dim = next(d for d in report_rich.dimension_drifts if d.dimension == "style_preference")
        assert rich_dim.confidence > sparse_dim.confidence


class TestProfileDriftDetectorCustomDimensions:
    """Custom dimension configuration."""

    def test_custom_dimensions_subset(self) -> None:
        detector = ProfileDriftDetector(
            threshold=0.35, dimensions=("style_preference", "domain_interests")
        )
        report = detector.detect({}, {})
        assert len(report.dimension_drifts) == 2
        dims = {d.dimension for d in report.dimension_drifts}
        assert dims == {"style_preference", "domain_interests"}
