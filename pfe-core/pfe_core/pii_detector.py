"""PII (Personally Identifiable Information) detection engine.

Provides rule-based and pattern-based detection of sensitive information
in text data for privacy compliance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PIISeverity(Enum):
    """Severity levels for PII detection."""
    LOW = "low"          # Generic patterns, higher false positive risk
    MEDIUM = "medium"    # Standard PII patterns
    HIGH = "high"        # High-confidence PII patterns


class PIIType(Enum):
    """Types of PII that can be detected."""
    # Contact Information
    EMAIL = "email"
    PHONE = "phone"

    # Identity Documents
    ID_CARD = "id_card"           # Chinese ID card
    PASSPORT = "passport"
    BANK_CARD = "bank_card"

    # Personal Names
    PERSON_NAME = "person_name"

    # Location
    ADDRESS = "address"

    # Online Identifiers
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"

    # Financial
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"


@dataclass
class PIIFinding:
    """A detected PII instance."""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    severity: PIISeverity
    confidence: float = 0.0
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pii_type": self.pii_type.value,
            "value": self.value[:50] + "..." if len(self.value) > 50 else self.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 2),
            "context": self.context[:100] if self.context else "",
        }


@dataclass
class PIIDetectionResult:
    """Result of PII detection on a text."""
    text_length: int
    findings: list[PIIFinding] = field(default_factory=list)
    has_pii: bool = False
    pii_types_found: set[PIIType] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_length": self.text_length,
            "has_pii": self.has_pii,
            "pii_types_found": [t.value for t in self.pii_types_found],
            "finding_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
        }


class PIIDetector:
    """Detects PII in text using rule-based patterns."""

    # Regex patterns for different PII types
    PATTERNS: dict[PIIType, list[tuple[str, PIISeverity]]] = {
        PIIType.EMAIL: [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', PIISeverity.HIGH),
        ],
        PIIType.PHONE: [
            # Chinese mobile numbers
            (r'1[3-9]\d{9}', PIISeverity.HIGH),
            # International format with country code
            (r'\+86[\s-]?1[3-9]\d{9}', PIISeverity.HIGH),
            # Phone with separators
            (r'1[3-9]\d{1}[\s-]?\d{4}[\s-]?\d{4}', PIISeverity.MEDIUM),
        ],
        PIIType.ID_CARD: [
            # 18-digit Chinese ID card
            (r'[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]', PIISeverity.HIGH),
        ],
        PIIType.CREDIT_CARD: [
            # Visa, Mastercard, Amex patterns
            (r'4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}', PIISeverity.HIGH),
            (r'5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}', PIISeverity.HIGH),
            (r'3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}', PIISeverity.HIGH),
        ],
        PIIType.BANK_CARD: [
            # 16-19 digit bank card numbers
            (r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4,7}', PIISeverity.HIGH),
        ],
        PIIType.IP_ADDRESS: [
            # IPv4
            (r'\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', PIISeverity.LOW),
        ],
        PIIType.MAC_ADDRESS: [
            (r'(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})', PIISeverity.LOW),
        ],
    }

    # Common Chinese surnames for name detection
    COMMON_SURNAMES = {
        '王', '李', '张', '刘', '陈', '杨', '黄', '赵', '周', '吴',
        '徐', '孙', '朱', '马', '胡', '郭', '林', '何', '高', '罗',
        '郑', '梁', '谢', '宋', '唐', '许', '韩', '冯', '邓', '曹',
        '彭', '曾', '肖', '田', '董', '袁', '潘', '于', '蒋', '蔡',
        '余', '杜', '叶', '程', '苏', '魏', '吕', '丁', '任', '沈',
        '姚', '卢', '姜', '崔', '钟', '谭', '陆', '汪', '范', '金',
        '石', '廖', '贾', '夏', '韦', '付', '方', '白', '邹', '孟',
        '熊', '秦', '邱', '江', '尹', '薛', '闫', '段', '雷', '侯',
        '龙', '史', '陶', '黎', '贺', '顾', '毛', '郝', '龚', '邵',
        '万', '钱', '严', '覃', '武', '戴', '莫', '孔', '向', '汤',
    }

    # Address keywords for Chinese addresses
    ADDRESS_KEYWORDS = [
        '省', '市', '区', '县', '镇', '乡', '村', '街道', '路', '街',
        '号', '楼', '单元', '室', '栋', '层', '小区', '花园', '公寓',
        '省', '市', '自治區', '特別行政區', '縣', '鎮', '鄉',
    ]

    def __init__(self, sensitivity: str = "medium"):
        """Initialize detector with sensitivity level.

        Args:
            sensitivity: "low", "medium", or "high"
        """
        self.sensitivity = sensitivity
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for performance."""
        self._compiled: dict[PIIType, list[tuple[Any, PIISeverity]]] = {}
        for pii_type, patterns in self.PATTERNS.items():
            self._compiled[pii_type] = [
                (re.compile(pattern, re.IGNORECASE), severity)
                for pattern, severity in patterns
            ]

    def detect(
        self,
        text: str,
        pii_types: Optional[list[PIIType]] = None,
        min_confidence: float = 0.5
    ) -> PIIDetectionResult:
        """Detect PII in the given text.

        Args:
            text: Text to analyze
            pii_types: Specific PII types to check, or None for all
            min_confidence: Minimum confidence threshold

        Returns:
            PIIDetectionResult with findings
        """
        result = PIIDetectionResult(text_length=len(text))

        if not text:
            return result

        types_to_check = pii_types or list(PIIType)

        for pii_type in types_to_check:
            if pii_type in self._compiled:
                for pattern, severity in self._compiled[pii_type]:
                    for match in pattern.finditer(text):
                        confidence = self._calculate_confidence(
                            pii_type, match.group(), severity
                        )
                        if confidence >= min_confidence:
                            finding = PIIFinding(
                                pii_type=pii_type,
                                value=match.group(),
                                start_pos=match.start(),
                                end_pos=match.end(),
                                severity=severity,
                                confidence=confidence,
                                context=self._get_context(text, match.start(), match.end())
                            )
                            result.findings.append(finding)
                            result.pii_types_found.add(pii_type)

        # Check for person names and addresses with special handling
        if PIIType.PERSON_NAME in types_to_check:
            result.findings.extend(self._detect_names(text))
            if any(f.pii_type == PIIType.PERSON_NAME for f in result.findings):
                result.pii_types_found.add(PIIType.PERSON_NAME)

        if PIIType.ADDRESS in types_to_check:
            result.findings.extend(self._detect_addresses(text))
            if any(f.pii_type == PIIType.ADDRESS for f in result.findings):
                result.pii_types_found.add(PIIType.ADDRESS)

        result.has_pii = len(result.findings) > 0
        return result

    def _calculate_confidence(
        self,
        pii_type: PIIType,
        value: str,
        base_severity: PIISeverity
    ) -> float:
        """Calculate confidence score for a match."""
        base_scores = {
            PIISeverity.LOW: 0.6,
            PIISeverity.MEDIUM: 0.75,
            PIISeverity.HIGH: 0.9
        }
        confidence = base_scores.get(base_severity, 0.5)

        # Adjust based on value characteristics
        if pii_type == PIIType.EMAIL:
            # Check for common false positives
            if '@example.' in value.lower():
                confidence -= 0.3
            if value.count('@') == 1 and len(value) > 5:
                confidence += 0.05

        elif pii_type == PIIType.PHONE:
            # Validate Chinese mobile prefix
            if re.match(r'1[3-9]', value):
                confidence += 0.05

        elif pii_type == PIIType.ID_CARD:
            # Check length (18 digits for Chinese ID)
            digits = re.sub(r'[^\dXx]', '', value)
            if len(digits) == 18:
                confidence += 0.05

        return min(confidence, 1.0)

    def _get_context(
        self,
        text: str,
        start: int,
        end: int,
        context_chars: int = 20
    ) -> str:
        """Get surrounding context for a match."""
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        return text[context_start:context_end]

    def _detect_names(self, text: str) -> list[PIIFinding]:
        """Detect potential person names in text."""
        findings = []

        # Pattern: surname followed by 1-2 given name characters
        # This is a heuristic and has higher false positive rate
        for surname in self.COMMON_SURNAMES:
            # Match surname + 1-2 Chinese characters
            pattern = re.compile(f'{surname}[\u4e00-\u9fa5]{{1,2}}')
            for match in pattern.finditer(text):
                # Check context to reduce false positives
                context_before = text[max(0, match.start()-10):match.start()]
                context_after = text[match.end():min(len(text), match.end()+10)]

                # Skip if looks like a place name, company, etc.
                skip_contexts = ['公司', '大学', '中学', '小学', '医院', '银行', '酒店', '餐厅', '机场', '车站']
                if any(sc in context_after for sc in skip_contexts):
                    continue

                # Check for name-indicating context
                name_indicators = ['我叫', '姓名', '联系人', '先生', '女士', '经理', '医生', '老师']
                confidence = 0.5
                for indicator in name_indicators:
                    if indicator in context_before or indicator in context_after:
                        confidence = 0.7
                        break

                if self.sensitivity == "high" or confidence >= 0.6:
                    finding = PIIFinding(
                        pii_type=PIIType.PERSON_NAME,
                        value=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        severity=PIISeverity.MEDIUM,
                        confidence=confidence,
                        context=self._get_context(text, match.start(), match.end())
                    )
                    findings.append(finding)

        return findings

    def _detect_addresses(self, text: str) -> list[PIIFinding]:
        """Detect potential addresses in text."""
        findings = []

        # Look for address patterns with keywords
        # This is a simplified heuristic
        for keyword in self.ADDRESS_KEYWORDS:
            # Find keyword and extract surrounding context
            for match in re.finditer(keyword, text):
                # Extract a window around the keyword
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                address_candidate = text[start:end]

                # Check for multiple address keywords to increase confidence
                keyword_count = sum(1 for k in self.ADDRESS_KEYWORDS if k in address_candidate)

                if keyword_count >= 2:
                    confidence = 0.5 + (keyword_count * 0.1)
                    if self.sensitivity == "high" or confidence >= 0.6:
                        finding = PIIFinding(
                            pii_type=PIIType.ADDRESS,
                            value=address_candidate,
                            start_pos=start,
                            end_pos=end,
                            severity=PIISeverity.MEDIUM,
                            confidence=min(confidence, 0.85),
                            context=self._get_context(text, start, end)
                        )
                        findings.append(finding)

        return findings

    def scan_data_sample(
        self,
        sample: dict[str, Any],
        text_fields: list[str] | None = None
    ) -> dict[str, PIIDetectionResult]:
        """Scan a data sample (e.g., training sample) for PII.

        Args:
            sample: Dictionary containing text data
            text_fields: Fields to scan, defaults to ['instruction', 'input', 'output']

        Returns:
            Dictionary mapping field names to detection results
        """
        fields = text_fields or ['instruction', 'input', 'output', 'conversation', 'messages']
        results = {}

        for field in fields:
            if field in sample and isinstance(sample[field], str):
                results[field] = self.detect(sample[field])
            elif field == 'messages' and field in sample:
                # Handle conversation format
                messages = sample[field]
                if isinstance(messages, list):
                    full_text = '\n'.join(
                        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                        for m in messages
                    )
                    results[field] = self.detect(full_text)

        return results


def batch_detect(
    detector: PIIDetector,
    texts: list[str],
    pii_types: list[PIIType] | None = None,
    min_confidence: float = 0.5
) -> list[PIIDetectionResult]:
    """Detect PII in multiple texts efficiently.

    Args:
        detector: The PIIDetector instance to use
        texts: List of texts to analyze
        pii_types: Specific PII types to check, or None for all
        min_confidence: Minimum confidence threshold

    Returns:
        List of PIIDetectionResult, one per input text
    """
    return [
        detector.detect(text, pii_types=pii_types, min_confidence=min_confidence)
        for text in texts
    ]


__all__ = [
    "PIISeverity",
    "PIIType",
    "PIIFinding",
    "PIIDetectionResult",
    "PIIDetector",
    "batch_detect",
]
