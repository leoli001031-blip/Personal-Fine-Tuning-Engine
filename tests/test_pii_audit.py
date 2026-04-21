"""Unit tests for PII audit and sanitization in data_policy.py."""

from pfe_core.data_policy import (
    HIGH_RISK_PII_TYPES,
    PIIAuditReport,
    audit_pii_exposure,
    sanitize_for_training,
)


def test_high_risk_pii_types_extended():
    assert "biometric" in HIGH_RISK_PII_TYPES
    assert "health_record" in HIGH_RISK_PII_TYPES
    assert "financial_account" in HIGH_RISK_PII_TYPES


def test_sanitize_for_training_no_pii():
    text = "The quick brown fox jumps over the lazy dog."
    assert sanitize_for_training(text) == text


def test_sanitize_for_training_email():
    text = "Contact me at alice@example.com for details."
    result = sanitize_for_training(text)
    assert "[REDACTED_email]" in result
    assert "alice@example.com" not in result


def test_sanitize_for_training_phone():
    text = "My number is 13800138000."
    result = sanitize_for_training(text)
    assert "[REDACTED_phone]" in result
    assert "13800138000" not in result


def test_sanitize_for_training_id_card():
    text = "ID: 11010519900101123X"
    result = sanitize_for_training(text)
    assert "[REDACTED_id_card]" in result
    assert "11010519900101123X" not in result


def test_sanitize_for_training_biometric():
    text = "We collected the user's fingerprint and facial recognition data."
    result = sanitize_for_training(text)
    assert "[REDACTED_biometric]" in result


def test_sanitize_for_training_health_record():
    text = "Patient diagnosis and prescription records are attached."
    result = sanitize_for_training(text)
    assert "[REDACTED_health_record]" in result


def test_sanitize_for_training_financial_account():
    text = "Bank account and Alipay transfer records included."
    result = sanitize_for_training(text)
    assert "[REDACTED_financial_account]" in result


def test_audit_pii_exposure_empty():
    report = audit_pii_exposure([])
    assert report.total_samples == 0
    assert report.severity == "low"
    assert report.pii_detected_count == 0


def test_audit_pii_exposure_clean():
    samples = [
        {"sample_id": "s1", "instruction": "What is the weather?", "output": "Sunny."},
    ]
    report = audit_pii_exposure(samples)
    assert report.total_samples == 1
    assert report.pii_detected_count == 0
    assert report.severity == "low"


def test_audit_pii_exposure_detected():
    samples = [
        {"sample_id": "s1", "instruction": "Email me at bob@test.com", "output": "Done."},
        {"sample_id": "s2", "instruction": "Call 13800138000", "output": "OK."},
    ]
    report = audit_pii_exposure(samples)
    assert report.pii_detected_count == 2
    assert "email" in report.pii_types_found
    assert "phone" in report.pii_types_found
    assert report.severity in ("medium", "high", "critical")
    assert len(report.recommendations) > 0


def test_audit_pii_exposure_critical_severity():
    samples = []
    for i in range(12):
        samples.append(
            {"sample_id": f"s{i}", "instruction": f"My card is 6222021234567890{i}", "output": "OK."}
        )
    report = audit_pii_exposure(samples)
    assert report.severity == "critical"
    assert report.pii_detected_count == 12


def test_audit_pii_exposure_messages_format():
    samples = [
        {
            "sample_id": "s1",
            "messages": [
                {"role": "user", "content": "Reach me at alice@example.com"},
                {"role": "assistant", "content": "Sure."},
            ],
        }
    ]
    report = audit_pii_exposure(samples)
    assert report.pii_detected_count == 1
    assert "email" in report.pii_types_found
