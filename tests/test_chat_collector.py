"""Tests for ChatCollector implicit signal extraction."""

from __future__ import annotations

import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pfe_core.collector import ChatCollector, ChatInteraction, CollectorConfig, ImplicitSignal
from pfe_core.models import SignalQuality
from pfe_core.user_memory import get_user_memory_store


@pytest.fixture(autouse=True)
def _isolated_pfe_home(pfe_home: Path) -> Path:
    return pfe_home


class TestChatInteraction:
    """Tests for ChatInteraction data model."""

    def test_basic_creation(self):
        """Test creating a basic ChatInteraction."""
        interaction = ChatInteraction(
            session_id="session_123",
            request_id="req_456",
            user_message="Hello, how are you?",
            assistant_message="I'm doing well, thank you!",
        )

        assert interaction.session_id == "session_123"
        assert interaction.request_id == "req_456"
        assert interaction.user_message == "Hello, how are you?"
        assert interaction.assistant_message == "I'm doing well, thank you!"
        assert interaction.event_id is not None
        assert interaction.timestamp is not None

    def test_with_optional_fields(self):
        """Test ChatInteraction with optional fields."""
        ts = datetime.now(timezone.utc)
        interaction = ChatInteraction(
            session_id="session_123",
            request_id="req_456",
            user_message="Hello",
            assistant_message="Hi there!",
            timestamp=ts,
            adapter_version="v1.0",
            metadata={"source": "test"},
            response_time_seconds=3.5,
        )

        assert interaction.timestamp == ts
        assert interaction.adapter_version == "v1.0"
        assert interaction.metadata == {"source": "test"}
        assert interaction.response_time_seconds == 3.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        interaction = ChatInteraction(
            session_id="session_123",
            request_id="req_456",
            user_message="Hello",
            assistant_message="Hi!",
        )

        d = interaction.to_dict()
        assert d["session_id"] == "session_123"
        assert d["request_id"] == "req_456"
        assert d["user_message"] == "Hello"
        assert d["assistant_message"] == "Hi!"
        assert "event_id" in d
        assert "timestamp" in d


class TestImplicitSignal:
    """Tests for ImplicitSignal data model."""

    def test_basic_creation(self):
        """Test creating a basic ImplicitSignal."""
        signal = ImplicitSignal(
            signal_type="accept",
            confidence=0.7,
            extraction_rule="accept_normal",
        )

        assert signal.signal_type == "accept"
        assert signal.confidence == 0.7
        assert signal.extraction_rule == "accept_normal"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        signal = ImplicitSignal(
            signal_id="sig_123",
            source_event_id="evt_456",
            request_id="req_789",
            session_id="session_abc",
            event_type="accept",
            context="Hello",
            model_output="Hi!",
            signal_type="accept",
            confidence=0.7,
            extraction_rule="accept_normal",
        )

        d = signal.to_dict()
        assert d["event_id"] == "sig_123"
        assert d["source_event_id"] == "evt_456"
        assert d["request_id"] == "req_789"
        assert d["session_id"] == "session_abc"
        assert d["event_type"] == "accept"
        assert d["context"] == "Hello"
        assert d["model_output"] == "Hi!"
        assert d["metadata"]["signal_type"] == "accept"
        assert d["metadata"]["confidence"] == 0.7
        assert d["signal_quality"]["reply_style"] == "accepted"


class TestCollectorConfig:
    """Tests for CollectorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CollectorConfig()

        assert config.enabled is True
        assert config.accept_confidence_threshold == 0.5
        assert config.edit_confidence_threshold == 0.5
        assert config.reject_confidence_threshold == 0.5
        assert config.regenerate_confidence_threshold == 0.5
        assert config.edit_distance_metric == "levenshtein"
        assert config.time_decay_enabled is True
        assert config.strong_accept_threshold_seconds == 5.0
        assert config.weak_accept_threshold_seconds == 60.0

    def test_validation(self):
        """Test configuration validation."""
        # Invalid confidence threshold
        with pytest.raises(ValueError, match="accept_confidence_threshold must be between 0 and 1"):
            CollectorConfig(accept_confidence_threshold=1.5)

        with pytest.raises(ValueError, match="accept_confidence_threshold must be between 0 and 1"):
            CollectorConfig(accept_confidence_threshold=-0.1)

        # Invalid time thresholds
        with pytest.raises(ValueError, match="strong_accept_threshold_seconds must be less than weak_accept_threshold_seconds"):
            CollectorConfig(
                strong_accept_threshold_seconds=60.0,
                weak_accept_threshold_seconds=5.0
            )

        # Invalid metric
        with pytest.raises(ValueError, match="edit_distance_metric must be 'levenshtein' or 'jaro_winkler'"):
            CollectorConfig(edit_distance_metric="invalid")


class TestChatCollector:
    """Tests for ChatCollector class."""

    @pytest.fixture
    def collector(self):
        """Create a test collector instance."""
        config = CollectorConfig(enabled=True)
        return ChatCollector(workspace="test_workspace", config=config)

    @pytest.fixture
    def basic_interaction(self):
        """Create a basic test interaction."""
        return ChatInteraction(
            session_id="session_123",
            request_id="req_456",
            user_message="What is Python?",
            assistant_message="Python is a programming language.",
        )

    def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector.workspace == "test_workspace"
        assert collector.config.enabled is True
        assert collector._interactions == []
        assert collector._signals == []

    def test_disabled_collector(self, collector, basic_interaction):
        """Test that disabled collector returns no signals."""
        collector.config.enabled = False
        signals = collector.on_interaction(basic_interaction, action="continue")
        assert signals == []

    def test_audit_write_failures_do_not_block_signal_collection(self, collector, basic_interaction, monkeypatch):
        """Audit log write failures should not block signal extraction."""
        assert collector._pii_audit is not None

        def fail_append(_entry):
            raise PermissionError("audit path unavailable")

        monkeypatch.setattr(collector._pii_audit, "_append_to_file", fail_append)

        signals = collector.on_interaction(basic_interaction, action="continue")

        assert len(signals) == 1
        assert collector._pii_audit._entries[-1].success is False

    def test_accept_signal_extraction(self, collector, basic_interaction):
        """Test extracting accept signals."""
        signals = collector.on_interaction(basic_interaction, action="continue")

        assert len(signals) == 1
        assert signals[0].signal_type == "accept"
        assert signals[0].confidence == 0.7
        assert signals[0].extraction_rule == "accept_no_timing"

    def test_strong_accept_signal(self, collector, basic_interaction):
        """Test strong accept signal with quick response."""
        basic_interaction.response_time_seconds = 3.0
        signals = collector.on_interaction(basic_interaction, action="continue")

        assert len(signals) == 1
        assert signals[0].signal_type == "accept"
        assert signals[0].confidence == 0.9  # 0.7 * 1.29 rounded
        assert signals[0].extraction_rule == "accept_quick_response"
        assert signals[0].response_time_seconds == 3.0

    def test_weak_accept_signal(self, collector, basic_interaction):
        """Test weak accept signal with slow response."""
        # Lower threshold to allow weak signals
        collector.config.accept_confidence_threshold = 0.3
        basic_interaction.response_time_seconds = 70.0
        signals = collector.on_interaction(basic_interaction, action="continue")

        assert len(signals) == 1
        assert signals[0].signal_type == "accept"
        assert signals[0].confidence == 0.4  # 0.7 * 0.57 rounded
        assert signals[0].extraction_rule == "accept_slow_response"

    def test_accept_below_threshold(self, collector, basic_interaction):
        """Test that weak signals below threshold are filtered."""
        collector.config.accept_confidence_threshold = 0.5
        basic_interaction.response_time_seconds = 70.0  # Would produce 0.4 confidence
        signals = collector.on_interaction(basic_interaction, action="continue")

        assert len(signals) == 0  # Filtered out

    def test_routes_explicit_user_data_into_memory_and_signal_metadata(self):
        """Explicit user facts/preferences should be routed through shared policy."""
        with tempfile.TemporaryDirectory() as tempdir:
            config = CollectorConfig(enabled=True)
            collector = ChatCollector(workspace="test_workspace", config=config, home=tempdir)
            interaction = ChatInteraction(
                session_id="session_explicit",
                request_id="req_explicit",
                user_message="我叫小王，我是程序员，我希望你以后回答更温和、更鼓励式。",
                assistant_message="好的，我会尽量更温和地回应你。",
            )

            signals = collector.on_interaction(interaction, action="continue")
            store = get_user_memory_store(tempdir)
            profile = store.get_profile("session_explicit")

            assert profile.get_fact("名字") == "小王"
            assert profile.get_fact("职业") == "程序员"
            assert any("温和" in value for value in profile.preferences.values())
            assert len(signals) == 1
            assert "explicit_user_data_routing" in signals[0].metadata
            routing_summary = signals[0].metadata["explicit_user_data_routing"]
            assert routing_summary["candidate_count"] >= 3
            assert routing_summary["stored_preference_count"] >= 1

    def test_explicit_user_data_ingestion_is_idempotent_per_request(self):
        """Repeated processing of the same request should not duplicate preference storage."""
        with tempfile.TemporaryDirectory() as tempdir:
            store = get_user_memory_store(tempdir)

            first = store.ingest_explicit_user_data(
                user_id="session_idempotent",
                user_message="我叫小王，我希望你以后回答更温和一点。",
                request_id="req_same",
            )
            second = store.ingest_explicit_user_data(
                user_id="session_idempotent",
                user_message="我叫小王，我希望你以后回答更温和一点。",
                request_id="req_same",
            )

            profile = store.get_profile("session_idempotent")
            assert first["processed"] is True
            assert second["processed"] is False
            assert second["reason"] == "duplicate_request_message"
            assert len(profile.preferences) == 1

    def test_edit_signal_slight(self, collector, basic_interaction):
        """Test edit signal with slight changes."""
        edited = "Python is a popular programming language."
        signals = collector.on_interaction(basic_interaction, edited_text=edited)

        assert len(signals) == 1
        assert signals[0].signal_type == "edit"
        assert signals[0].confidence == 0.6
        assert signals[0].extraction_rule == "edit_slight"
        assert signals[0].edit_distance is not None
        assert signals[0].edit_distance_ratio is not None
        assert signals[0].edit_distance_ratio <= 0.2  # Allow equality at boundary

    def test_edit_signal_moderate(self, collector, basic_interaction):
        """Test edit signal with moderate changes."""
        # More substantial edit - ensure it's in the moderate range (20-50%)
        # Target: ~30-40% change
        edited = "Python is a high-level programming language with dynamic typing."
        signals = collector.on_interaction(basic_interaction, edited_text=edited)

        assert len(signals) == 1
        assert signals[0].signal_type == "edit"
        # Check that it's in moderate range (0.8 confidence) or strong (0.9)
        assert signals[0].confidence in (0.8, 0.9)
        assert signals[0].extraction_rule in ("edit_moderate", "edit_strong")
        assert signals[0].edit_distance_ratio > 0.2

    def test_edit_signal_strong(self, collector, basic_interaction):
        """Test edit signal with strong changes."""
        # Very different text
        edited = "Java is a completely different programming language with static typing."
        signals = collector.on_interaction(basic_interaction, edited_text=edited)

        assert len(signals) == 1
        assert signals[0].signal_type == "edit"
        assert signals[0].confidence == 0.9
        assert signals[0].extraction_rule == "edit_strong"
        assert signals[0].edit_distance_ratio >= 0.5

    def test_reject_signal(self, collector, basic_interaction):
        """Test reject signal extraction."""
        signals = collector.on_interaction(basic_interaction, action="delete")

        assert len(signals) == 1
        assert signals[0].signal_type == "reject"
        assert signals[0].confidence == 0.95
        assert signals[0].extraction_rule == "reject_delete"

    def test_regenerate_signal(self, collector, basic_interaction):
        """Test regenerate signal extraction."""
        signals = collector.on_interaction(basic_interaction, action="regenerate")

        assert len(signals) == 1
        assert signals[0].signal_type == "regenerate"
        assert signals[0].confidence == 0.85
        assert signals[0].extraction_rule == "regenerate_explicit"

    def test_next_user_message_accept(self, collector, basic_interaction):
        """Test accept signal from next user message."""
        signals = collector.on_interaction(
            basic_interaction,
            next_user_message="Tell me more about Python."
        )

        assert len(signals) == 1
        assert signals[0].signal_type == "accept"

    def test_multiple_signals_not_extracted(self, collector, basic_interaction):
        """Test that only one signal type is extracted per interaction."""
        # Edit takes precedence over accept
        signals = collector.on_interaction(
            basic_interaction,
            edited_text="Modified text",
            next_user_message="Continue"
        )

        assert len(signals) == 1
        assert signals[0].signal_type == "edit"

    def test_get_stats(self, collector, basic_interaction):
        """Test statistics collection."""
        # Initially empty
        stats = collector.get_stats()
        assert stats["total_interactions"] == 0
        assert stats["total_signals"] == 0
        assert stats["signals_by_type"] == {}

        # Add some signals
        collector.on_interaction(basic_interaction, action="continue")
        collector.on_interaction(basic_interaction, action="delete")
        collector.on_interaction(basic_interaction, edited_text="Modified")

        stats = collector.get_stats()
        assert stats["total_interactions"] == 3
        assert stats["total_signals"] == 3
        assert stats["signals_by_type"]["accept"] == 1
        assert stats["signals_by_type"]["reject"] == 1
        assert stats["signals_by_type"]["edit"] == 1

    def test_get_signals_for_review(self, collector, basic_interaction):
        """Test signal review functionality."""
        # Add signals with different confidences (use separate sessions to avoid contradictions)
        collector.on_interaction(basic_interaction, action="continue")  # 0.7

        other = ChatInteraction(
            session_id="session_other",
            request_id="req_other",
            user_message="Hello",
            assistant_message="Hi there!",
        )
        collector.on_interaction(other, action="delete")  # 0.95

        # Get all signals
        all_signals = collector.get_signals_for_review()
        assert len(all_signals) == 2

        # Filter by type
        accept_signals = collector.get_signals_for_review(signal_type="accept")
        assert len(accept_signals) == 1
        assert accept_signals[0].signal_type == "accept"

        # Filter by confidence
        high_confidence = collector.get_signals_for_review(min_confidence=0.9)
        assert len(high_confidence) == 1
        assert high_confidence[0].confidence == 0.95

        # Filter by confidence range
        mid_confidence = collector.get_signals_for_review(min_confidence=0.6, max_confidence=0.8)
        assert len(mid_confidence) == 1
        assert mid_confidence[0].confidence == 0.7


class TestEditDistance:
    """Tests for edit distance calculations."""

    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        config = CollectorConfig(edit_distance_metric="levenshtein")
        collector = ChatCollector(workspace="test", config=config)

        # Same string
        assert collector._levenshtein_distance("hello", "hello") == 0

        # One character different
        assert collector._levenshtein_distance("hello", "hallo") == 1

        # Insertion
        assert collector._levenshtein_distance("hello", "hello!") == 1

        # Deletion
        assert collector._levenshtein_distance("hello", "hell") == 1

        # Multiple changes
        assert collector._levenshtein_distance("kitten", "sitting") == 3

    def test_jaro_winkler_distance(self):
        """Test Jaro-Winkler distance calculation."""
        config = CollectorConfig(edit_distance_metric="jaro_winkler")
        collector = ChatCollector(workspace="test", config=config)

        # Same string
        assert collector._jaro_winkler_distance("hello", "hello") == 0

        # Empty strings
        assert collector._jaro_winkler_distance("", "abc") == 3
        assert collector._jaro_winkler_distance("abc", "") == 3

        # Similar strings (should have low distance)
        dist = collector._jaro_winkler_distance("martha", "marhta")
        assert dist < len("martha")  # Should be less than full length

    def test_edit_distance_metric_selection(self):
        """Test that correct metric is used based on config."""
        # Levenshtein
        config = CollectorConfig(edit_distance_metric="levenshtein")
        collector = ChatCollector(workspace="test", config=config)
        assert collector._calculate_edit_distance("hello", "hallo") == 1

        # Jaro-Winkler
        config = CollectorConfig(edit_distance_metric="jaro_winkler")
        collector = ChatCollector(workspace="test", config=config)
        dist = collector._calculate_edit_distance("hello", "hallo")
        assert dist >= 0  # Just verify it runs without error


class TestSignalRules:
    """Tests for signal extraction rules."""

    def test_default_rules(self):
        """Test default signal rules configuration."""
        config = CollectorConfig()
        rules = config.signal_rules

        # Accept rules
        assert "accept" in rules
        assert rules["accept"]["base_confidence"] == 0.7
        assert rules["accept"]["strong_multiplier"] == 1.29
        assert rules["accept"]["weak_multiplier"] == 0.57

        # Edit rules
        assert "edit" in rules
        assert rules["edit"]["slight_threshold"] == 0.2
        assert rules["edit"]["moderate_threshold"] == 0.5
        assert rules["edit"]["slight_confidence"] == 0.6
        assert rules["edit"]["moderate_confidence"] == 0.8
        assert rules["edit"]["strong_confidence"] == 0.9

        # Reject rules
        assert "reject" in rules
        assert rules["reject"]["base_confidence"] == 0.95

        # Regenerate rules
        assert "regenerate" in rules
        assert rules["regenerate"]["base_confidence"] == 0.85

    def test_custom_rules(self):
        """Test custom signal rules."""
        custom_rules = {
            "accept": {
                "base_confidence": 0.6,
                "strong_multiplier": 1.5,
                "weak_multiplier": 0.5,
            },
            "edit": {
                "slight_threshold": 0.15,
                "moderate_threshold": 0.4,
                "slight_confidence": 0.5,
                "moderate_confidence": 0.7,
                "strong_confidence": 0.95,
            },
            "reject": {"base_confidence": 0.99},
            "regenerate": {"base_confidence": 0.8},
        }

        config = CollectorConfig(signal_rules=custom_rules)
        collector = ChatCollector(workspace="test", config=config)

        # Verify custom rules are loaded
        loaded_rules = collector._load_rules()
        assert loaded_rules["accept"]["base_confidence"] == 0.6
        assert loaded_rules["reject"]["base_confidence"] == 0.99


class TestContradictionDetection:
    """Tests for contradictory signal detection."""

    @pytest.fixture
    def collector(self):
        config = CollectorConfig(
            enabled=True,
            contradiction_detection_enabled=True,
            accept_confidence_threshold=0.3,
        )
        return ChatCollector(workspace="test", config=config)

    @pytest.fixture
    def basic_interaction(self):
        return ChatInteraction(
            session_id="session_123",
            request_id="req_456",
            user_message="What is Python?",
            assistant_message="Python is a programming language.",
        )

    def test_accept_reject_contradiction(self, collector, basic_interaction):
        """Test that accept + reject for same request is marked as conflict."""
        accept_signals = collector.on_interaction(basic_interaction, action="continue")
        assert len(accept_signals) == 1
        assert accept_signals[0].signal_quality.conflict is False

        reject_signals = collector.on_interaction(basic_interaction, action="delete")
        assert len(reject_signals) == 1

        # Both signals should now be marked as conflicting
        assert accept_signals[0].signal_quality.conflict is True
        assert accept_signals[0].signal_quality.conflict_reason == "contradictory_accept_reject"
        assert reject_signals[0].signal_quality.conflict is True
        assert reject_signals[0].signal_quality.conflict_reason == "contradictory_accept_reject"

    def test_accept_regenerate_contradiction(self, collector, basic_interaction):
        """Test that accept + regenerate for same request is marked as conflict."""
        accept_signals = collector.on_interaction(basic_interaction, action="continue")
        regenerate_signals = collector.on_interaction(basic_interaction, action="regenerate")

        assert accept_signals[0].signal_quality.conflict is True
        assert accept_signals[0].signal_quality.conflict_reason == "contradictory_accept_regenerate"
        assert regenerate_signals[0].signal_quality.conflict is True

    def test_accept_edit_contradiction(self, collector, basic_interaction):
        """Test that accept + edit for same request is marked as conflict."""
        accept_signals = collector.on_interaction(basic_interaction, action="continue")
        edit_signals = collector.on_interaction(
            basic_interaction,
            edited_text="Python is a high-level language.",
        )

        assert accept_signals[0].signal_quality.conflict is True
        assert accept_signals[0].signal_quality.conflict_reason == "contradictory_accept_then_edit"
        assert edit_signals[0].signal_quality.conflict is True

    def test_contradiction_reduces_confidence(self, collector, basic_interaction):
        """Test that conflicting signals have reduced confidence."""
        accept_signals = collector.on_interaction(basic_interaction, action="continue")
        original_confidence = accept_signals[0].confidence

        collector.on_interaction(basic_interaction, action="delete")

        reduced = round(original_confidence * 0.7, 2)
        assert accept_signals[0].confidence == reduced
        assert accept_signals[0].signal_quality.confidence == reduced

    def test_no_contradiction_across_sessions(self, collector, basic_interaction):
        """Test that different sessions do not trigger contradictions."""
        collector.on_interaction(basic_interaction, action="continue")

        other = ChatInteraction(
            session_id="session_other",
            request_id="req_456",
            user_message="What is Python?",
            assistant_message="Python is a programming language.",
        )
        signals = collector.on_interaction(other, action="delete")

        assert signals[0].signal_quality.conflict is False

    def test_contradiction_detection_disabled(self, basic_interaction):
        """Test that contradictions are not detected when disabled."""
        config = CollectorConfig(
            enabled=True,
            contradiction_detection_enabled=False,
        )
        collector = ChatCollector(workspace="test", config=config)

        accept_signals = collector.on_interaction(basic_interaction, action="continue")
        collector.on_interaction(basic_interaction, action="delete")

        assert accept_signals[0].signal_quality.conflict is False


class TestReplayAndRollback:
    """Tests for replay buffer and rollback functionality."""

    @pytest.fixture
    def collector(self):
        config = CollectorConfig(
            enabled=True,
            contradiction_detection_enabled=True,
            enable_replay_buffer=True,
            accept_confidence_threshold=0.3,
        )
        return ChatCollector(workspace="test", config=config)

    @pytest.fixture
    def basic_interaction(self):
        return ChatInteraction(
            session_id="session_123",
            request_id="req_456",
            user_message="What is Python?",
            assistant_message="Python is a programming language.",
        )

    def test_replay_candidates_include_conflicted_signals(self, collector, basic_interaction):
        """Test that conflicted signals appear in replay candidates."""
        collector.on_interaction(basic_interaction, action="continue")
        collector.on_interaction(basic_interaction, action="delete")

        candidates = collector.get_replay_candidates()
        assert len(candidates) == 2
        assert all(s.signal_quality.conflict for s in candidates)

    def test_replay_candidates_include_low_confidence(self, collector, basic_interaction):
        """Test that low-confidence signals appear in replay candidates."""
        basic_interaction.response_time_seconds = 70.0  # Weak accept = 0.4 confidence
        collector.config.accept_confidence_threshold = 0.3
        collector.on_interaction(basic_interaction, action="continue")

        candidates = collector.get_replay_candidates()
        assert len(candidates) == 1
        assert candidates[0].confidence == 0.4

    def test_rollback_signal_returns_and_removes(self, collector, basic_interaction):
        """Test that rollback removes signal and marks it rolled_back."""
        signals = collector.on_interaction(basic_interaction, action="continue")
        signal_id = signals[0].signal_id

        assert len(collector._signals) == 1

        rolled = collector.rollback_signal(signal_id)
        assert rolled is not None
        assert rolled.signal_id == signal_id
        assert rolled.signal_quality.rolled_back is True
        assert "rolled_back_for_replay" in (rolled.signal_quality.confidence_reason or "")
        assert len(collector._signals) == 0

    def test_rollback_unknown_signal_returns_none(self, collector):
        """Test that rolling back unknown signal returns None."""
        assert collector.rollback_signal("nonexistent") is None

    def test_contradiction_summary(self, collector, basic_interaction):
        """Test contradiction summary for observability."""
        collector.on_interaction(basic_interaction, action="continue")
        collector.on_interaction(basic_interaction, action="delete")

        summary = collector.get_contradiction_summary()
        assert summary["total_signals"] == 2
        assert summary["conflicted_signals"] == 2
        assert summary["conflict_reasons"]["contradictory_accept_reject"] == 2
        assert summary["replay_candidates"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
