"""Training auditor for P2-G: pre-training compliance and quality checks.

The TrainingAuditor runs a multi-dimensional audit on training samples before
they are fed into the trainer.  If the resulting severity is ``critical``,
training must be blocked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..data_policy import HIGH_RISK_PII_TYPES, audit_pii_exposure
from ..signal.conflict_detector import detect_conflicts


# Sensitive content keyword lists (heuristic, conservative)
_VIOLENCE_KEYWORDS = {
    "kill", "murder", "assassinate", "bomb", "terrorist", "shoot", "stab",
    "屠杀", "杀人", "爆炸", "恐怖袭击", "枪击", "刺杀",
}

_HATE_KEYWORDS = {
    "hate", "racist", "nazi", "supremacist", "genocide", "ethnic cleansing",
    "种族歧视", "纳粹", "至上主义", "种族灭绝", "仇恨",
}

_ILLEGAL_KEYWORDS = {
    "drug trafficking", "money laundering", "fraud", "hack", "exploit",
    "毒品交易", "洗钱", "诈骗", "黑客", "漏洞利用", "非法入侵",
}

_SENSITIVE_KEYWORDS = _VIOLENCE_KEYWORDS | _HATE_KEYWORDS | _ILLEGAL_KEYWORDS


@dataclass
class TrainingAuditReport:
    """Result of a pre-training audit."""

    total_samples: int = 0
    pii_detected_count: int = 0
    pii_types_found: dict[str, int] = field(default_factory=dict)
    sensitive_content_count: int = 0
    sensitive_keywords_found: dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    empty_field_count: int = 0
    extreme_length_count: int = 0
    conflict_sample_count: int = 0
    quarantine_sample_count: int = 0
    severity: str = "low"  # "low" | "medium" | "high" | "critical"
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "pii_detected_count": self.pii_detected_count,
            "pii_types_found": dict(self.pii_types_found),
            "sensitive_content_count": self.sensitive_content_count,
            "sensitive_keywords_found": dict(self.sensitive_keywords_found),
            "duplicate_count": self.duplicate_count,
            "empty_field_count": self.empty_field_count,
            "extreme_length_count": self.extreme_length_count,
            "conflict_sample_count": self.conflict_sample_count,
            "quarantine_sample_count": self.quarantine_sample_count,
            "severity": self.severity,
            "blocked": self.blocked,
            "reasons": list(self.reasons),
            "recommendations": list(self.recommendations),
        }


class TrainingAuditor:
    """Audit training samples across PII, sensitive content, quality, and conflict dimensions."""

    # Configurable thresholds
    DEFAULT_MIN_TEXT_LENGTH = 1
    DEFAULT_MAX_TEXT_LENGTH = 8192
    DEFAULT_DUPLICATE_SIMILARITY = 0.95

    def __init__(
        self,
        *,
        min_text_length: int = DEFAULT_MIN_TEXT_LENGTH,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        duplicate_similarity: float = DEFAULT_DUPLICATE_SIMILARITY,
        block_on_critical: bool = True,
    ):
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.duplicate_similarity = duplicate_similarity
        self.block_on_critical = block_on_critical

    def audit(self, samples: list[dict[str, Any]]) -> TrainingAuditReport:
        """Run the full audit suite on *samples*.

        Returns a ``TrainingAuditReport``.  If ``severity == "critical"`` and
        ``block_on_critical`` is enabled, ``blocked`` is set to ``True``.
        """
        report = TrainingAuditReport(total_samples=len(samples))
        if not samples:
            report.severity = "low"
            return report

        # 1. PII exposure
        pii_report = audit_pii_exposure(samples)
        report.pii_detected_count = pii_report.pii_detected_count
        report.pii_types_found = dict(pii_report.pii_types_found)

        # 2. Sensitive content
        sensitive_result = self._check_sensitive_content(samples)
        report.sensitive_content_count = sensitive_result["count"]
        report.sensitive_keywords_found = sensitive_result["keywords_found"]

        # 3. Sample quality
        quality_result = self._check_sample_quality(samples)
        report.duplicate_count = quality_result["duplicate_count"]
        report.empty_field_count = quality_result["empty_field_count"]
        report.extreme_length_count = quality_result["extreme_length_count"]

        # 4. Conflict / quarantine
        conflict_result = self._check_conflict_and_quarantine(samples)
        report.conflict_sample_count = conflict_result["conflict_count"]
        report.quarantine_sample_count = conflict_result["quarantine_count"]

        # Severity scoring
        report.severity = self._compute_severity(report)
        report.blocked = self.block_on_critical and report.severity == "critical"

        # Reasons
        if report.pii_detected_count:
            report.reasons.append(f"PII detected in {report.pii_detected_count} sample(s)")
        if report.sensitive_content_count:
            report.reasons.append(f"Sensitive content in {report.sensitive_content_count} sample(s)")
        if report.duplicate_count:
            report.reasons.append(f"{report.duplicate_count} duplicate sample(s)")
        if report.empty_field_count:
            report.reasons.append(f"{report.empty_field_count} sample(s) with empty fields")
        if report.extreme_length_count:
            report.reasons.append(f"{report.extreme_length_count} sample(s) with extreme length")
        if report.conflict_sample_count:
            report.reasons.append(f"{report.conflict_sample_count} conflicted sample(s)")
        if report.quarantine_sample_count:
            report.reasons.append(f"{report.quarantine_sample_count} quarantined sample(s)")

        # Recommendations
        if report.severity == "critical":
            report.recommendations.append("Training is blocked due to critical audit findings.")
        if report.pii_detected_count:
            report.recommendations.append("Apply sanitize_for_training() before training.")
        if report.sensitive_content_count:
            report.recommendations.append("Review and remove samples containing prohibited content.")
        if report.duplicate_count:
            report.recommendations.append("Deduplicate samples to improve training stability.")
        if report.empty_field_count or report.extreme_length_count:
            report.recommendations.append("Clean low-quality samples (empty or extreme length).")
        if report.conflict_sample_count:
            report.recommendations.append("Resolve conflicting signals before training.")
        if report.quarantine_sample_count:
            report.recommendations.append("Review quarantined samples and either fix or remove them.")

        return report

    @staticmethod
    def _extract_texts(sample: dict[str, Any]) -> list[str]:
        """Extract all text strings from a sample dict."""
        texts: list[str] = []
        for field in ("instruction", "input", "output", "chosen", "rejected", "context", "model_output", "conversation"):
            val = sample.get(field)
            if isinstance(val, str) and val:
                texts.append(val)
        messages = sample.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content:
                        texts.append(content)
        return texts

    def _check_sensitive_content(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        count = 0
        keywords_found: dict[str, int] = {}
        for sample in samples:
            texts = self._extract_texts(sample)
            sample_hit = False
            for text in texts:
                lower = text.lower()
                for kw in _SENSITIVE_KEYWORDS:
                    if kw in lower:
                        keywords_found[kw] = keywords_found.get(kw, 0) + 1
                        sample_hit = True
            if sample_hit:
                count += 1
        return {"count": count, "keywords_found": keywords_found}

    def _check_sample_quality(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        empty_count = 0
        extreme_count = 0
        seen_texts: list[str] = []
        duplicate_count = 0

        for sample in samples:
            texts = self._extract_texts(sample)
            if not texts:
                empty_count += 1
                continue

            total_len = sum(len(t) for t in texts)
            if total_len < self.min_text_length or total_len > self.max_text_length:
                extreme_count += 1

            # Simple exact-match dedup on concatenated text
            combined = " ".join(texts)
            is_dup = False
            for seen in seen_texts:
                if seen == combined:
                    is_dup = True
                    break
            if is_dup:
                duplicate_count += 1
            else:
                seen_texts.append(combined)

        return {
            "duplicate_count": duplicate_count,
            "empty_field_count": empty_count,
            "extreme_length_count": extreme_count,
        }

    def _check_conflict_and_quarantine(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Check for semantic conflicts and quarantine flags."""
        conflict_count = 0
        quarantine_count = 0

        # Quarantine check: look at metadata/signal_quality
        for sample in samples:
            metadata = sample.get("metadata") or {}
            signal_quality = metadata.get("signal_quality") or sample.get("signal_quality") or {}
            if isinstance(signal_quality, dict):
                if signal_quality.get("quarantine") or signal_quality.get("conflict"):
                    quarantine_count += 1
            if metadata.get("quarantine") or metadata.get("conflict"):
                quarantine_count += 1

        # Conflict detection via conflict_detector engine
        signals = []
        for sample in samples:
            sig = dict(sample)
            sig.setdefault("event_id", sample.get("sample_id") or sample.get("id") or "")
            sig.setdefault("context", sample.get("instruction") or sample.get("input") or "")
            sig.setdefault("timestamp", sample.get("timestamp") or "")
            sig.setdefault("event_type", sample.get("sample_type") or "")
            signals.append(sig)

        if len(signals) >= 2:
            report = detect_conflicts(signals)
            if report.conflict_detected:
                conflict_count = len(report.conflicting_pairs)

        return {"conflict_count": conflict_count, "quarantine_count": quarantine_count}

    def _compute_severity(self, report: TrainingAuditReport) -> str:
        """Map audit metrics to a severity level."""
        critical_signals = 0
        if report.pii_detected_count > 0:
            high_risk = set(report.pii_types_found.keys()) & HIGH_RISK_PII_TYPES
            if high_risk:
                critical_signals += 1
        if report.sensitive_content_count > 0:
            critical_signals += 1
        if report.quarantine_sample_count > 0:
            critical_signals += 1

        if critical_signals >= 2:
            return "critical"
        if critical_signals == 1:
            return "high"
        if report.pii_detected_count > 0 or report.conflict_sample_count > 0:
            return "medium"
        if report.duplicate_count > 0 or report.empty_field_count > 0 or report.extreme_length_count > 0:
            return "low"
        return "low"
