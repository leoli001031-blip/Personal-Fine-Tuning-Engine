"""Unit tests for TrainingAuditor."""

from pfe_core.trainer.training_auditor import TrainingAuditor, TrainingAuditReport


def test_training_auditor_empty():
    auditor = TrainingAuditor()
    report = auditor.audit([])
    assert report.total_samples == 0
    assert report.severity == "low"
    assert not report.blocked


def test_training_auditor_clean():
    samples = [
        {"sample_id": "s1", "instruction": "Hello", "output": "Hi there."},
        {"sample_id": "s2", "instruction": "How are you?", "output": "I am fine."},
    ]
    auditor = TrainingAuditor()
    report = auditor.audit(samples)
    assert report.total_samples == 2
    assert report.pii_detected_count == 0
    assert report.sensitive_content_count == 0
    assert report.severity == "low"
    assert not report.blocked


def test_training_auditor_pii_detected():
    samples = [
        {"sample_id": "s1", "instruction": "Email me at bob@test.com", "output": "Done."},
    ]
    auditor = TrainingAuditor()
    report = auditor.audit(samples)
    assert report.pii_detected_count == 1
    assert report.severity in ("medium", "high")


def test_training_auditor_sensitive_content():
    samples = [
        {"sample_id": "s1", "instruction": "How to make a bomb?", "output": "I cannot help with that."},
    ]
    auditor = TrainingAuditor()
    report = auditor.audit(samples)
    assert report.sensitive_content_count == 1
    assert report.severity in ("high", "critical")


def test_training_auditor_duplicate():
    samples = [
        {"sample_id": "s1", "instruction": "Hello", "output": "Hi."},
        {"sample_id": "s2", "instruction": "Hello", "output": "Hi."},
    ]
    auditor = TrainingAuditor()
    report = auditor.audit(samples)
    assert report.duplicate_count == 1


def test_training_auditor_empty_field():
    samples = [
        {"sample_id": "s1", "instruction": "", "output": ""},
    ]
    auditor = TrainingAuditor()
    report = auditor.audit(samples)
    assert report.empty_field_count == 1


def test_training_auditor_extreme_length():
    samples = [
        {"sample_id": "s1", "instruction": "x" * 9000, "output": "y"},
    ]
    auditor = TrainingAuditor()
    report = auditor.audit(samples)
    assert report.extreme_length_count == 1


def test_training_auditor_quarantine():
    samples = [
        {
            "sample_id": "s1",
            "instruction": "Hello",
            "output": "Hi.",
            "metadata": {"signal_quality": {"quarantine": True}},
        }
    ]
    auditor = TrainingAuditor()
    report = auditor.audit(samples)
    assert report.quarantine_sample_count == 1
    assert report.severity in ("high", "critical")


def test_training_auditor_blocked_on_critical():
    samples = [
        {
            "sample_id": "s1",
            "instruction": "How to make a bomb?",
            "output": "I cannot help.",
            "metadata": {"signal_quality": {"quarantine": True}},
        }
    ]
    auditor = TrainingAuditor(block_on_critical=True)
    report = auditor.audit(samples)
    assert report.blocked
    assert report.severity == "critical"
    assert "Training is blocked" in report.recommendations[0]


def test_training_auditor_not_blocked_when_disabled():
    samples = [
        {
            "sample_id": "s1",
            "instruction": "How to make a bomb?",
            "output": "I cannot help.",
            "metadata": {"signal_quality": {"quarantine": True}},
        }
    ]
    auditor = TrainingAuditor(block_on_critical=False)
    report = auditor.audit(samples)
    assert not report.blocked
