"""Tests for the P2-A semantic conflict detection engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from pfe_core.signal.conflict_detector import (
    ConflictReport,
    _context_similarity,
    _extract_context,
    _extract_reply_style,
    _extract_timestamp,
    _ngram_similarity,
    _sequence_similarity,
    apply_conflict_detection,
    detect_conflicts,
)


class TestSimilarityFunctions:
    """Unit tests for the pure similarity helpers."""

    def test_sequence_similarity_identical(self) -> None:
        assert _sequence_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_sequence_similarity_empty(self) -> None:
        assert _sequence_similarity("", "") == pytest.approx(1.0)
        assert _sequence_similarity("a", "") == pytest.approx(0.0)

    def test_ngram_similarity_identical(self) -> None:
        assert _ngram_similarity("hello", "hello") == pytest.approx(1.0)

    def test_ngram_similarity_empty(self) -> None:
        assert _ngram_similarity("", "") == pytest.approx(1.0)
        assert _ngram_similarity("a", "") == pytest.approx(0.0)

    def test_context_similarity_english(self) -> None:
        s1 = "how do I bake sourdough bread"
        s2 = "how do I bake sourdough bread at home"
        sim = _context_similarity(s1, s2)
        assert 0.5 < sim <= 1.0

    def test_context_similarity_chinese(self) -> None:
        s1 = "如何制作酸面包"
        s2 = "如何在家制作酸面包"
        sim = _context_similarity(s1, s2)
        assert 0.5 < sim <= 1.0

    def test_context_similarity_different_topics(self) -> None:
        s1 = "how do I bake sourdough bread"
        s2 = "what is the capital of france"
        sim = _context_similarity(s1, s2)
        assert sim < 0.5

    def test_similarity_around_0_6(self) -> None:
        """A pair that yields similarity around 0.6 should not trigger at 0.75."""
        s1 = "explain quantum mechanics"
        s2 = "explain general relativity"
        sim = _context_similarity(s1, s2)
        assert sim < 0.75


class TestExtractors:
    """Unit tests for signal field extractors."""

    def test_extract_context_primary(self) -> None:
        assert _extract_context({"context": "  Hello World  "}) == "hello world"

    def test_extract_context_fallback(self) -> None:
        assert _extract_context({"instruction": "Foo Bar"}) == "foo bar"
        assert _extract_context({"user_input": "Baz Qux"}) == "baz qux"

    def test_extract_reply_style_from_signal_quality(self) -> None:
        sig: dict[str, Any] = {"signal_quality": {"reply_style": "accepted"}}
        assert _extract_reply_style(sig) == "accepted"

    def test_extract_reply_style_from_event_type(self) -> None:
        assert _extract_reply_style({"event_type": "accept"}) == "accepted"
        assert _extract_reply_style({"event_type": "reject"}) == "rejected"
        assert _extract_reply_style({"event_type": "edit"}) == "edited"

    def test_extract_timestamp_datetime(self) -> None:
        now = datetime.now(timezone.utc)
        assert _extract_timestamp({"timestamp": now}) == now

    def test_extract_timestamp_iso(self) -> None:
        ts = "2024-01-15T10:30:00+00:00"
        assert _extract_timestamp({"timestamp": ts}) == datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)

    def test_extract_timestamp_naive_iso_is_normalized_to_utc(self) -> None:
        ts = "2024-01-15T10:30:00"
        assert _extract_timestamp({"timestamp": ts}) == datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)

    def test_extract_timestamp_missing(self) -> None:
        result = _extract_timestamp({})
        assert result.year == 1970


class TestDetectConflicts:
    """Integration tests for detect_conflicts."""

    def _make_signal(
        self,
        sid: str,
        context: str,
        event_type: str,
        timestamp: datetime,
        user_action: dict[str, Any] | None = None,
        signal_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sq = signal_quality or {}
        if "reply_style" not in sq:
            sq = {**sq, "reply_style": event_type.replace("accept", "accepted").replace("reject", "rejected").replace("edit", "edited")}
        return {
            "signal_id": sid,
            "context": context,
            "event_type": event_type,
            "timestamp": timestamp,
            "user_action": user_action or {},
            "signal_quality": sq,
        }

    def test_empty_signals(self) -> None:
        report = detect_conflicts([])
        assert report.conflict_detected is False
        assert report.conflicting_pairs == []

    def test_no_conflict_different_contexts(self) -> None:
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal("s1", "how to bake bread", "accept", now - timedelta(hours=1)),
            self._make_signal("s2", "capital of france", "reject", now),
        ]
        report = detect_conflicts(signals)
        assert report.conflict_detected is False

    def test_conflict_accept_then_reject(self) -> None:
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal("s1", "how to bake sourdough bread", "accept", now - timedelta(hours=1)),
            self._make_signal("s2", "how to bake sourdough bread", "reject", now),
        ]
        report = detect_conflicts(signals)
        assert report.conflict_detected is True
        assert ("s1", "s2") in report.conflicting_pairs
        assert report.conflict_reasons["s1"] == "accept_then_rejected"
        assert report.conflict_reasons["s2"] == "accept_then_rejected"

    def test_conflict_accept_then_edit(self) -> None:
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal("s1", "how to bake sourdough bread", "accept", now - timedelta(hours=2)),
            self._make_signal("s2", "how to bake sourdough bread", "edit", now - timedelta(hours=1), user_action={"final_text": "use rye flour"}),
        ]
        report = detect_conflicts(signals)
        assert report.conflict_detected is True
        assert ("s1", "s2") in report.conflicting_pairs

    def test_conflict_two_edits_divergence(self) -> None:
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal("s1", "how to bake sourdough bread", "edit", now - timedelta(hours=2), user_action={"final_text": "use wheat flour"}),
            self._make_signal("s2", "how to bake sourdough bread", "edit", now - timedelta(hours=1), user_action={"final_text": "use rye flour"}),
        ]
        report = detect_conflicts(signals)
        assert report.conflict_detected is True
        assert ("s1", "s2") in report.conflicting_pairs
        assert report.conflict_reasons["s1"] == "edit_divergence"

    def test_no_conflict_two_edits_same_direction(self) -> None:
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal("s1", "how to bake sourdough bread", "edit", now - timedelta(hours=2), user_action={"final_text": "use wheat flour"}),
            self._make_signal("s2", "how to bake sourdough bread", "edit", now - timedelta(hours=1), user_action={"final_text": "use wheat flour"}),
        ]
        report = detect_conflicts(signals)
        assert report.conflict_detected is False

    def test_similar_but_not_identical_context_no_conflict(self) -> None:
        """Context similarity ~0.6 should not trigger at default 0.75 threshold."""
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal("s1", "explain quantum mechanics", "accept", now - timedelta(hours=1)),
            self._make_signal("s2", "explain general relativity", "reject", now),
        ]
        report = detect_conflicts(signals)
        assert report.conflict_detected is False

    def test_lookback_window_excludes_old_signals(self) -> None:
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal("s1", "how to bake sourdough bread", "accept", now - timedelta(hours=200)),
            self._make_signal("s2", "how to bake sourdough bread", "reject", now),
        ]
        report = detect_conflicts(signals, lookback_window_hours=168.0)
        assert report.conflict_detected is False

    def test_stable_preference_contradicted(self) -> None:
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal(
                "s1",
                "how to bake sourdough bread",
                "accept",
                now - timedelta(hours=2),
                signal_quality={"reply_style": "accepted", "confidence": 0.9},
            ),
            self._make_signal("s2", "how to bake sourdough bread", "reject", now),
        ]
        report = detect_conflicts(signals)
        assert report.conflict_detected is True
        assert report.conflict_reasons["s1"] == "accept_then_rejected"

    def test_resolution_recommendation_keep_newer(self) -> None:
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal(
                "s1",
                "how to bake sourdough bread",
                "accept",
                now - timedelta(hours=2),
                signal_quality={"reply_style": "accepted", "confidence": 0.6},
            ),
            self._make_signal(
                "s2",
                "how to bake sourdough bread",
                "reject",
                now,
                signal_quality={"reply_style": "rejected", "confidence": 0.9},
            ),
        ]
        report = detect_conflicts(signals)
        assert report.conflict_detected is True
        assert report.resolution_recommendation == "keep_newer"

    def test_resolution_recommendation_quarantine_both(self) -> None:
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal(
                "s1",
                "how to bake sourdough bread",
                "accept",
                now - timedelta(hours=2),
                signal_quality={"reply_style": "accepted", "confidence": 0.9},
            ),
            self._make_signal(
                "s2",
                "how to bake sourdough bread",
                "reject",
                now,
                signal_quality={"reply_style": "rejected", "confidence": 0.5},
            ),
        ]
        report = detect_conflicts(signals)
        assert report.conflict_detected is True
        assert report.resolution_recommendation == "quarantine_both"

    def test_multiple_conflicts(self) -> None:
        now = datetime.now(timezone.utc)
        signals = [
            self._make_signal("s1", "how to bake sourdough bread", "accept", now - timedelta(hours=3)),
            self._make_signal("s2", "how to bake sourdough bread", "edit", now - timedelta(hours=2), user_action={"final_text": "a"}),
            self._make_signal("s3", "how to bake sourdough bread", "edit", now - timedelta(hours=1), user_action={"final_text": "b"}),
        ]
        report = detect_conflicts(signals)
        assert report.conflict_detected is True
        # s1 vs s2 (accept then edit)
        assert ("s1", "s2") in report.conflicting_pairs
        # s2 vs s3 (edit divergence)
        assert ("s2", "s3") in report.conflicting_pairs

    def test_mixed_naive_and_aware_timestamps_do_not_crash(self) -> None:
        signals = [
            {
                "signal_id": "s1",
                "context": "how to bake sourdough bread",
                "event_type": "accept",
                "timestamp": "2026-04-20T12:00:00",
                "signal_quality": {"reply_style": "accepted", "confidence": 0.9},
            },
            self._make_signal(
                "s2",
                "how to bake sourdough bread",
                "reject",
                datetime(2026, 4, 20, 13, 0, tzinfo=timezone.utc),
            ),
        ]

        report = detect_conflicts(signals)

        assert report.conflict_detected is True
        assert ("s1", "s2") in report.conflicting_pairs


class TestApplyConflictDetection:
    """Tests for the integration helper apply_conflict_detection."""

    def test_dict_samples_updated(self) -> None:
        now = datetime.now(timezone.utc)
        samples = [
            {
                "signal_id": "s1",
                "context": "how to bake bread",
                "event_type": "accept",
                "timestamp": now - timedelta(hours=1),
                "signal_quality": {"reply_style": "accepted", "confidence": 0.8, "conflict": False},
            },
            {
                "signal_id": "s2",
                "context": "how to bake bread",
                "event_type": "reject",
                "timestamp": now,
                "signal_quality": {"reply_style": "rejected", "confidence": 0.8, "conflict": False},
            },
        ]
        updated = apply_conflict_detection(samples)
        sq1 = updated[0]["signal_quality"]
        sq2 = updated[1]["signal_quality"]
        assert sq1["conflict"] is True
        assert sq2["conflict"] is True
        assert "accept_then_rejected" in sq1["conflict_reason"]

    def test_no_conflict_samples_unchanged(self) -> None:
        now = datetime.now(timezone.utc)
        samples = [
            {
                "signal_id": "s1",
                "context": "how to bake sourdough bread",
                "event_type": "accept",
                "timestamp": now - timedelta(hours=1),
                "signal_quality": {"reply_style": "accepted", "conflict": False},
            },
            {
                "signal_id": "s2",
                "context": "what is the capital of france",
                "event_type": "reject",
                "timestamp": now,
                "signal_quality": {"reply_style": "rejected", "conflict": False},
            },
        ]
        updated = apply_conflict_detection(samples)
        assert updated[0]["signal_quality"]["conflict"] is False
        assert updated[1]["signal_quality"]["conflict"] is False

    def test_preserves_existing_conflict_reason(self) -> None:
        now = datetime.now(timezone.utc)
        samples = [
            {
                "signal_id": "s1",
                "context": "how to bake bread",
                "event_type": "accept",
                "timestamp": now - timedelta(hours=1),
                "signal_quality": {"reply_style": "accepted", "conflict": False, "conflict_reason": "incomplete_event_chain"},
            },
            {
                "signal_id": "s2",
                "context": "how to bake bread",
                "event_type": "reject",
                "timestamp": now,
                "signal_quality": {"reply_style": "rejected", "conflict": False},
            },
        ]
        updated = apply_conflict_detection(samples)
        reason = updated[0]["signal_quality"]["conflict_reason"]
        assert "incomplete_event_chain" in reason
        assert "accept_then_rejected" in reason

    def test_empty_samples(self) -> None:
        assert apply_conflict_detection([]) == []


class TestConflictReportDataclass:
    """Smoke tests for ConflictReport itself."""

    def test_defaults(self) -> None:
        r = ConflictReport()
        assert r.conflict_detected is False
        assert r.conflicting_pairs == []
        assert r.conflict_reasons == {}
        assert r.resolution_recommendation == "quarantine_both"
