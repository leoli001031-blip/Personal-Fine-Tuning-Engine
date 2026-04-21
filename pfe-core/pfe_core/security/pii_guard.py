"""PII Guard for pre-pipeline data sanitization and isolation.

Scans incoming data for PII before it enters the training pipeline.
Supports strict isolation for high-risk PII types and tagging for
low-risk PII. Provides training-safe sanitization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..data_policy import HIGH_RISK_PII_TYPES, sanitize_for_training


# Low-risk PII types that can be tagged but do not require strict isolation
_LOW_RISK_PII_TYPES = {
    "person_name",
    "city",
    "country",
    "organization",
    "job_title",
    "nickname",
}

# Compile heuristic patterns for low-risk PII detection
_LOW_RISK_PATTERNS = {
    "person_name": re.compile(
        r"(?:我叫|我的名字是|我是)\s*([^\s,，。.!！?？]{1,16})|"
        r"(?:my name is|i am)\s+([A-Za-z][A-Za-z\s\-']{0,31})",
        re.IGNORECASE,
    ),
    "city": re.compile(
        r"(?:来自|住在|居住在|籍贯)\s*([^\s,，。.!！?？]{1,16})|"
        r"(?:from|live in|based in)\s+([A-Za-z][A-Za-z\s\-]{0,31})",
        re.IGNORECASE,
    ),
    "country": re.compile(
        r"(?:国家|国籍)\s*([^\s,，。.!！?？]{1,16})|"
        r"(?:country|nationality)\s+([A-Za-z][A-Za-z\s\-]{0,31})",
        re.IGNORECASE,
    ),
    "organization": re.compile(
        r"(?:公司|学校|单位|机构)\s*([^\s,，。.!！?？]{1,24})|"
        r"(?:company|school|organization|university)\s+([A-Za-z][A-Za-z\s\-]{0,47})",
        re.IGNORECASE,
    ),
    "job_title": re.compile(
        r"(?:职位|职务|岗位)\s*([^\s,，。.!！?？]{1,16})|"
        r"(?:position|title|role)\s+([A-Za-z][A-Za-z\s\-]{0,31})",
        re.IGNORECASE,
    ),
    "nickname": re.compile(
        r"(?:昵称|网名|外号)\s*([^\s,，。.!！?？]{1,16})|"
        r"(?:nickname|username|alias)\s+([A-Za-z0-9][A-Za-z0-9_\-]{0,31})",
        re.IGNORECASE,
    ),
}


def _compile_high_risk_patterns() -> dict[str, re.Pattern[str]]:
    """Compile regex patterns for high-risk PII detection."""
    return {
        "credit_card": re.compile(
            r"(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2})[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}"
        ),
        "bank_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4,7}\b"),
        "id_card": re.compile(
            r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"
        ),
        "phone": re.compile(r"(?:\+86[\s-]?)?1[3-9]\d{9}"),
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "address": re.compile(
            r"(?:省|市|区|县|镇|乡|村|街道|路|街|号|楼|单元|室|栋|层|小区|花园|公寓)[\u4e00-\u9fa5\d\s]+"
        ),
        "password": re.compile(r"(?:password|密码|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
        "private_key": re.compile(
            r"(?:private[_-]?key|secret[_-]?key|api[_-]?key|token)\s*[:=]\s*[a-zA-Z0-9+/=]+",
            re.IGNORECASE,
        ),
        "secret": re.compile(
            r"(?:secret|密钥|秘钥)\s*[:=]\s*[a-zA-Z0-9+/=\-_]+",
            re.IGNORECASE,
        ),
        "token": re.compile(
            r"\b(?:bearer\s+[a-zA-Z0-9_\-\.]+|"
            r"[a-zA-Z0-9]{32,64}|"
            r"sk-[a-zA-Z0-9]{20,})\b",
            re.IGNORECASE,
        ),
        "api_key": re.compile(
            r"(?:api[_-]?key|app[_-]?key|access[_-]?key)\s*[:=]\s*[a-zA-Z0-9+/=\-_]+",
            re.IGNORECASE,
        ),
        "biometric": re.compile(
            r"(?:指纹|虹膜|面部识别|人脸识别|声纹|掌纹|DNA|基因序列|"
            r"biometric|fingerprint|iris|face\s*recognition|voiceprint|palm\s*print)",
            re.IGNORECASE,
        ),
        "health_record": re.compile(
            r"(?:病历|诊断|处方|医保号|医保卡|社保卡|住院号|门诊号|体检报告|检验单|"
            r"medical\s*record|diagnosis|prescription|health\s*record)",
            re.IGNORECASE,
        ),
        "financial_account": re.compile(
            r"(?:银行卡|账户|账号|开户行|支行|支付宝|微信支付|余额|转账记录|"
            r"bank\s*account|account\s*number|financial\s*account|alipay|wechat\s*pay)",
            re.IGNORECASE,
        ),
    }


_HIGH_RISK_PATTERNS = _compile_high_risk_patterns()


@dataclass
class PIISampleVerdict:
    """Verdict for a single sample after PII scanning."""

    sample_id: str
    is_safe: bool = True
    high_risk_types: set[str] = field(default_factory=set)
    low_risk_types: set[str] = field(default_factory=set)
    quarantined: bool = False
    sanitized_text: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "is_safe": self.is_safe,
            "high_risk_types": sorted(self.high_risk_types),
            "low_risk_types": sorted(self.low_risk_types),
            "quarantined": self.quarantined,
            "sanitized_text": self.sanitized_text,
            "reason": self.reason,
        }


@dataclass
class PIIScanReport:
    """Report for a batch PII scan."""

    total_samples: int = 0
    safe_count: int = 0
    quarantined_count: int = 0
    tagged_count: int = 0
    high_risk_hits: dict[str, int] = field(default_factory=dict)
    low_risk_hits: dict[str, int] = field(default_factory=dict)
    verdicts: list[PIISampleVerdict] = field(default_factory=list)
    severity: str = "low"  # "low" | "medium" | "high" | "critical"
    blocked: bool = False
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "safe_count": self.safe_count,
            "quarantined_count": self.quarantined_count,
            "tagged_count": self.tagged_count,
            "high_risk_hits": dict(self.high_risk_hits),
            "low_risk_hits": dict(self.low_risk_hits),
            "verdicts": [v.to_dict() for v in self.verdicts],
            "severity": self.severity,
            "blocked": self.blocked,
            "recommendations": list(self.recommendations),
        }


class PIIGuard:
    """Guard that scans data for PII before it enters the training pipeline.

    Supports:
    - Strict isolation (quarantine) for high-risk PII types (e.g., SSN, credit card).
    - Tagging for low-risk PII types (e.g., person name, city).
    - Training-safe sanitization via ``sanitize_for_training()``.
    - Safety assessment via ``is_training_safe()``.
    """

    def __init__(
        self,
        *,
        quarantine_high_risk: bool = True,
        tag_low_risk: bool = True,
        allow_low_risk_in_training: bool = False,
    ):
        self.quarantine_high_risk = quarantine_high_risk
        self.tag_low_risk = tag_low_risk
        self.allow_low_risk_in_training = allow_low_risk_in_training

    @staticmethod
    def _extract_texts(sample: dict[str, Any]) -> list[str]:
        """Extract all text strings from a sample dict."""
        texts: list[str] = []
        for field in (
            "instruction",
            "input",
            "output",
            "chosen",
            "rejected",
            "context",
            "model_output",
            "conversation",
        ):
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

    def _scan_text(self, text: str) -> tuple[set[str], set[str]]:
        """Scan a single text for high-risk and low-risk PII.

        Returns (high_risk_types_found, low_risk_types_found).
        """
        high_risk_found: set[str] = set()
        low_risk_found: set[str] = set()

        for pii_type, pattern in _HIGH_RISK_PATTERNS.items():
            if pattern.search(text):
                high_risk_found.add(pii_type)

        if self.tag_low_risk:
            for pii_type, pattern in _LOW_RISK_PATTERNS.items():
                if pattern.search(text):
                    low_risk_found.add(pii_type)

        return high_risk_found, low_risk_found

    def scan_sample(self, sample: dict[str, Any]) -> PIISampleVerdict:
        """Scan a single sample and return a verdict."""
        sample_id = str(sample.get("sample_id") or sample.get("id") or "")
        texts = self._extract_texts(sample)

        all_high_risk: set[str] = set()
        all_low_risk: set[str] = set()

        for text in texts:
            hr, lr = self._scan_text(text)
            all_high_risk.update(hr)
            all_low_risk.update(lr)

        quarantined = bool(all_high_risk) and self.quarantine_high_risk
        is_safe = not quarantined and (self.allow_low_risk_in_training or not all_low_risk)

        reason_parts: list[str] = []
        if all_high_risk:
            reason_parts.append(f"high-risk PII detected: {sorted(all_high_risk)}")
        if all_low_risk:
            reason_parts.append(f"low-risk PII detected: {sorted(all_low_risk)}")
        if not reason_parts:
            reason_parts.append("no PII detected")

        return PIISampleVerdict(
            sample_id=sample_id,
            is_safe=is_safe,
            high_risk_types=all_high_risk,
            low_risk_types=all_low_risk,
            quarantined=quarantined,
            reason="; ".join(reason_parts),
        )

    def scan_batch(self, samples: list[dict[str, Any]]) -> PIIScanReport:
        """Scan a batch of samples and return a comprehensive report."""
        report = PIIScanReport(total_samples=len(samples))
        if not samples:
            report.severity = "low"
            return report

        for sample in samples:
            verdict = self.scan_sample(sample)
            report.verdicts.append(verdict)

            if verdict.quarantined:
                report.quarantined_count += 1
            elif verdict.low_risk_types:
                report.tagged_count += 1
            else:
                report.safe_count += 1

            for hr in verdict.high_risk_types:
                report.high_risk_hits[hr] = report.high_risk_hits.get(hr, 0) + 1
            for lr in verdict.low_risk_types:
                report.low_risk_hits[lr] = report.low_risk_hits.get(lr, 0) + 1

        report.severity = self._compute_severity(report)
        report.blocked = report.severity == "critical"
        report.recommendations = self._build_recommendations(report)
        return report

    def sanitize_for_training(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return a sanitized copy of *sample* with PII redacted.

        Uses the existing ``sanitize_for_training`` from ``data_policy`` on
        all text fields. The original sample is not mutated.
        """
        sanitized = dict(sample)
        for field in (
            "instruction",
            "input",
            "output",
            "chosen",
            "rejected",
            "context",
            "model_output",
            "conversation",
        ):
            val = sanitized.get(field)
            if isinstance(val, str) and val:
                sanitized[field] = sanitize_for_training(val)

        messages = sanitized.get("messages")
        if isinstance(messages, list):
            new_messages: list[dict[str, Any]] = []
            for msg in messages:
                if isinstance(msg, dict):
                    new_msg = dict(msg)
                    content = new_msg.get("content")
                    if isinstance(content, str) and content:
                        new_msg["content"] = sanitize_for_training(content)
                    new_messages.append(new_msg)
                else:
                    new_messages.append(msg)
            sanitized["messages"] = new_messages

        return sanitized

    def is_training_safe(self, sample: dict[str, Any]) -> bool:
        """Return whether *sample* is safe to use for training.

        A sample is considered unsafe if:
        - It contains high-risk PII and ``quarantine_high_risk`` is enabled.
        - It contains low-risk PII and ``allow_low_risk_in_training`` is disabled.
        """
        verdict = self.scan_sample(sample)
        return verdict.is_safe

    @staticmethod
    def _compute_severity(report: PIIScanReport) -> str:
        """Map scan results to a severity level."""
        high_risk_types = set(report.high_risk_hits.keys())
        if report.quarantined_count >= 10 or len(high_risk_types) >= 3:
            return "critical"
        if report.quarantined_count >= 1:
            return "high"
        if report.tagged_count >= 5:
            return "medium"
        if report.tagged_count >= 1:
            return "low"
        return "low"

    @staticmethod
    def _build_recommendations(report: PIIScanReport) -> list[str]:
        """Generate actionable recommendations from a scan report."""
        recs: list[str] = []
        if report.severity == "critical":
            recs.append("Block pipeline ingestion until high-risk PII is removed or redacted.")
        if report.quarantined_count > 0:
            recs.append(
                f"Quarantine {report.quarantined_count} sample(s) containing high-risk PII."
            )
        if report.high_risk_hits:
            recs.append(
                f"Review high-risk PII types: {sorted(report.high_risk_hits.keys())}."
            )
        if report.tagged_count > 0:
            recs.append(
                f"Tag {report.tagged_count} sample(s) with low-risk PII for audit."
            )
        if report.safe_count == report.total_samples:
            recs.append("No PII detected; samples are safe for pipeline ingestion.")
        return recs
