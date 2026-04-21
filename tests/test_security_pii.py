"""Tests for pfe_core.security.pii_guard (PIIGuard)."""

from __future__ import annotations

import pytest

from pfe_core.security.pii_guard import PIIGuard, PIIScanReport, PIISampleVerdict


class TestPIIGuardScanSample:
    def test_clean_sample_is_safe(self):
        guard = PIIGuard()
        sample = {"sample_id": "s1", "instruction": "Hello world", "output": "Hi there."}
        verdict = guard.scan_sample(sample)
        assert verdict.is_safe is True
        assert verdict.quarantined is False
        assert verdict.high_risk_types == set()
        assert verdict.low_risk_types == set()

    def test_high_risk_credit_card_quarantined(self):
        guard = PIIGuard()
        sample = {
            "sample_id": "s2",
            "instruction": "My card is 4111-1111-1111-1111",
            "output": "OK.",
        }
        verdict = guard.scan_sample(sample)
        assert verdict.quarantined is True
        assert "credit_card" in verdict.high_risk_types
        assert verdict.is_safe is False

    def test_high_risk_phone_quarantined(self):
        guard = PIIGuard()
        sample = {
            "sample_id": "s3",
            "instruction": "Call me at 13800138000",
            "output": "Sure.",
        }
        verdict = guard.scan_sample(sample)
        assert verdict.quarantined is True
        assert "phone" in verdict.high_risk_types
        assert verdict.is_safe is False

    def test_low_risk_name_tagged(self):
        guard = PIIGuard(tag_low_risk=True, allow_low_risk_in_training=False)
        sample = {
            "sample_id": "s4",
            "instruction": "My name is Alice",
            "output": "Nice to meet you.",
        }
        verdict = guard.scan_sample(sample)
        assert "person_name" in verdict.low_risk_types
        assert verdict.quarantined is False
        assert verdict.is_safe is False

    def test_low_risk_allowed_when_configured(self):
        guard = PIIGuard(tag_low_risk=True, allow_low_risk_in_training=True)
        sample = {
            "sample_id": "s5",
            "instruction": "My name is Bob",
            "output": "Hello Bob.",
        }
        verdict = guard.scan_sample(sample)
        assert "person_name" in verdict.low_risk_types
        assert verdict.is_safe is True

    def test_multiple_high_risk_types(self):
        guard = PIIGuard()
        sample = {
            "sample_id": "s6",
            "instruction": "Email: alice@test.com, Phone: 13800138000",
            "output": "Got it.",
        }
        verdict = guard.scan_sample(sample)
        assert verdict.quarantined is True
        assert {"email", "phone"}.issubset(verdict.high_risk_types)

    def test_messages_field_scanned(self):
        guard = PIIGuard()
        sample = {
            "sample_id": "s7",
            "messages": [
                {"role": "user", "content": "My SSN is 123-45-6789"},
                {"role": "assistant", "content": "I cannot help with that."},
            ],
        }
        verdict = guard.scan_sample(sample)
        # id_card pattern is Chinese; SSN won't match, but we verify messages are scanned
        # by using a phone number inside messages instead
        sample2 = {
            "sample_id": "s7b",
            "messages": [
                {"role": "user", "content": "Call me at 13800138000"},
            ],
        }
        verdict2 = guard.scan_sample(sample2)
        assert verdict2.quarantined is True
        assert "phone" in verdict2.high_risk_types

    def test_empty_sample(self):
        guard = PIIGuard()
        sample = {"sample_id": "s8"}
        verdict = guard.scan_sample(sample)
        assert verdict.is_safe is True
        assert verdict.quarantined is False


class TestPIIGuardScanBatch:
    def test_batch_mixed_safety(self):
        guard = PIIGuard()
        samples = [
            {"sample_id": "s1", "instruction": "Hello", "output": "Hi."},
            {"sample_id": "s2", "instruction": "Card: 4111-1111-1111-1111", "output": "OK."},
            {"sample_id": "s3", "instruction": "My name is Alice", "output": "Hi Alice."},
        ]
        report = guard.scan_batch(samples)
        assert report.total_samples == 3
        assert report.safe_count == 1
        assert report.quarantined_count == 1
        assert report.tagged_count == 1
        assert report.severity == "high"

    def test_batch_critical_severity(self):
        guard = PIIGuard()
        samples = [
            {"sample_id": f"s{i}", "instruction": f"Card: 4111-1111-1111-111{i}"}
            for i in range(10)
        ]
        report = guard.scan_batch(samples)
        assert report.severity == "critical"
        assert report.blocked is True

    def test_batch_empty(self):
        guard = PIIGuard()
        report = guard.scan_batch([])
        assert report.total_samples == 0
        assert report.severity == "low"

    def test_batch_recommendations(self):
        guard = PIIGuard()
        samples = [
            {"sample_id": "s1", "instruction": "Hello"},
        ]
        report = guard.scan_batch(samples)
        assert any("safe" in r.lower() for r in report.recommendations)


class TestPIIGuardSanitize:
    def test_sanitize_redacts_email(self):
        guard = PIIGuard()
        sample = {
            "sample_id": "s1",
            "instruction": "Contact me at alice@example.com please.",
            "output": "Sure.",
        }
        sanitized = guard.sanitize_for_training(sample)
        assert "[REDACTED_email]" in sanitized["instruction"]
        assert "alice@example.com" not in sanitized["instruction"]
        # Original sample unchanged
        assert "alice@example.com" in sample["instruction"]

    def test_sanitize_messages(self):
        guard = PIIGuard()
        sample = {
            "sample_id": "s2",
            "messages": [
                {"role": "user", "content": "My email is bob@test.com"},
            ],
        }
        sanitized = guard.sanitize_for_training(sample)
        assert "[REDACTED_email]" in sanitized["messages"][0]["content"]

    def test_sanitize_no_pii_unchanged(self):
        guard = PIIGuard()
        sample = {"sample_id": "s3", "instruction": "Hello world", "output": "Hi."}
        sanitized = guard.sanitize_for_training(sample)
        assert sanitized["instruction"] == "Hello world"
        assert sanitized["output"] == "Hi."


class TestPIIGuardIsTrainingSafe:
    def test_safe_sample(self):
        guard = PIIGuard()
        sample = {"sample_id": "s1", "instruction": "Hello", "output": "Hi."}
        assert guard.is_training_safe(sample) is True

    def test_unsafe_high_risk(self):
        guard = PIIGuard()
        sample = {"sample_id": "s2", "instruction": "Card: 4111-1111-1111-1111"}
        assert guard.is_training_safe(sample) is False

    def test_unsafe_low_risk_when_not_allowed(self):
        guard = PIIGuard(allow_low_risk_in_training=False)
        sample = {"sample_id": "s3", "instruction": "My name is Alice"}
        assert guard.is_training_safe(sample) is False

    def test_safe_low_risk_when_allowed(self):
        guard = PIIGuard(allow_low_risk_in_training=True)
        sample = {"sample_id": "s4", "instruction": "My name is Alice"}
        assert guard.is_training_safe(sample) is True


class TestPIIScanReport:
    def test_to_dict_structure(self):
        report = PIIScanReport(
            total_samples=2,
            safe_count=1,
            quarantined_count=1,
            high_risk_hits={"phone": 1},
        )
        d = report.to_dict()
        assert d["total_samples"] == 2
        assert d["safe_count"] == 1
        assert d["high_risk_hits"] == {"phone": 1}


class TestPIISampleVerdict:
    def test_to_dict_structure(self):
        verdict = PIISampleVerdict(
            sample_id="s1",
            is_safe=False,
            high_risk_types={"email"},
            low_risk_types=set(),
            quarantined=True,
            reason="high-risk PII detected",
        )
        d = verdict.to_dict()
        assert d["sample_id"] == "s1"
        assert d["is_safe"] is False
        assert d["quarantined"] is True
        assert d["high_risk_types"] == ["email"]
