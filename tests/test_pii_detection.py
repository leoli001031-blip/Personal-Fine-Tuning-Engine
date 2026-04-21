"""Tests for PII detection and anonymization."""
import pytest
from pfe_core.pii_detector import PIIDetector, PIIType, PIISeverity
from pfe_core.anonymizer import Anonymizer, AnonymizationConfig, AnonymizationStrategy


class TestPIIDetector:
    """Test PII detection functionality."""

    @pytest.fixture
    def detector(self):
        return PIIDetector(sensitivity="medium")

    def test_detect_email(self, detector):
        text = "Contact me at john@example.com"
        result = detector.detect(text)
        assert result.has_pii
        assert PIIType.EMAIL in result.pii_types_found

    def test_detect_phone(self, detector):
        text = "Call me at 13800138000"
        result = detector.detect(text)
        assert result.has_pii
        assert PIIType.PHONE in result.pii_types_found

    def test_detect_id_card(self, detector):
        text = "My ID is 110105199001011234"
        result = detector.detect(text)
        assert result.has_pii
        assert PIIType.ID_CARD in result.pii_types_found

    def test_no_pii(self, detector):
        text = "The weather is nice today."
        result = detector.detect(text)
        assert not result.has_pii
        assert len(result.findings) == 0

    def test_sensitivity_levels(self):
        text = "IP: 192.168.1.1"

        low = PIIDetector(sensitivity="low")
        medium = PIIDetector(sensitivity="medium")
        high = PIIDetector(sensitivity="high")

        # All should detect IP at some level
        r_low = low.detect(text)
        r_medium = medium.detect(text)
        r_high = high.detect(text)

        assert PIIType.IP_ADDRESS in r_low.pii_types_found
        assert PIIType.IP_ADDRESS in r_medium.pii_types_found
        assert PIIType.IP_ADDRESS in r_high.pii_types_found


class TestAnonymizer:
    """Test anonymization functionality."""

    @pytest.fixture
    def detector(self):
        return PIIDetector(sensitivity="medium")

    def test_replace_strategy(self, detector):
        text = "Email: john@example.com"
        result = detector.detect(text)

        config = AnonymizationConfig(strategy=AnonymizationStrategy.REPLACE)
        anonymizer = Anonymizer(config)
        anonymized = anonymizer.anonymize(text, result)

        assert "<EMAIL>" in anonymized
        assert "john@example.com" not in anonymized

    def test_mask_strategy(self, detector):
        text = "Phone: 13800138000"
        result = detector.detect(text)

        config = AnonymizationConfig(strategy=AnonymizationStrategy.MASK)
        anonymizer = Anonymizer(config)
        anonymized = anonymizer.anonymize(text, result)

        assert "****" in anonymized
        assert "13800138000" not in anonymized

    def test_remove_strategy(self, detector):
        text = "Email: john@example.com"
        result = detector.detect(text)

        config = AnonymizationConfig(strategy=AnonymizationStrategy.REMOVE)
        anonymizer = Anonymizer(config)
        anonymized = anonymizer.anonymize(text, result)

        assert "john@example.com" not in anonymized


class TestCollectorPIIIntegration:
    """Test PII integration with ChatCollector."""

    def test_pii_components_initialized(self):
        from pfe_core.collector.chat_collector import ChatCollector
        from pfe_core.collector.config import CollectorConfig

        config = CollectorConfig(
            pii_detection_enabled=True,
            pii_anonymization_strategy="mask",
        )

        collector = ChatCollector(workspace="test", config=config)
        assert collector._pii_detector is not None
        assert collector._anonymizer is not None

    def test_pii_disabled(self):
        from pfe_core.collector.chat_collector import ChatCollector
        from pfe_core.collector.config import CollectorConfig

        config = CollectorConfig(pii_detection_enabled=False)
        collector = ChatCollector(workspace="test", config=config)

        assert collector._pii_detector is None
        assert collector._anonymizer is None
