"""PII anonymization strategies for data privacy protection.

Provides multiple anonymization strategies:
- Replace: Replace PII with placeholder tags
- Hash: Irreversible hashing of sensitive data
- Mask: Partial masking while preserving format
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from .pii_detector import PIIDetectionResult, PIIFinding, PIIType


class AnonymizationStrategy(Enum):
    """Available anonymization strategies."""
    REPLACE = "replace"      # <EMAIL>, <PHONE>
    HASH = "hash"            # Irreversible hash
    MASK = "mask"            # Partial masking: ***@example.com, 138****8888
    REMOVE = "remove"        # Complete removal


@dataclass
class AnonymizationConfig:
    """Configuration for anonymization behavior."""
    strategy: AnonymizationStrategy = AnonymizationStrategy.REPLACE
    preserve_length: bool = False  # Keep placeholder same length as original
    custom_placeholders: dict[PIIType, str] | None = None
    salt: str | None = None  # For hashing


class Anonymizer:
    """Anonymizes detected PII in text."""

    DEFAULT_PLACEHOLDERS: dict[PIIType, str] = {
        PIIType.EMAIL: "<EMAIL>",
        PIIType.PHONE: "<PHONE>",
        PIIType.ID_CARD: "<ID_CARD>",
        PIIType.PASSPORT: "<PASSPORT>",
        PIIType.BANK_CARD: "<BANK_CARD>",
        PIIType.CREDIT_CARD: "<CREDIT_CARD>",
        PIIType.PERSON_NAME: "<NAME>",
        PIIType.ADDRESS: "<ADDRESS>",
        PIIType.IP_ADDRESS: "<IP>",
        PIIType.MAC_ADDRESS: "<MAC>",
        PIIType.BANK_ACCOUNT: "<ACCOUNT>",
    }

    def __init__(self, config: AnonymizationConfig | None = None):
        """Initialize anonymizer with configuration."""
        self.config = config or AnonymizationConfig()
        self._placeholders = self.config.custom_placeholders or self.DEFAULT_PLACEHOLDERS

    def anonymize(
        self,
        text: str,
        detection_result: PIIDetectionResult
    ) -> str:
        """Anonymize text based on detection results.

        Args:
            text: Original text
            detection_result: PII detection results

        Returns:
            Anonymized text
        """
        if not detection_result.findings:
            return text

        # Sort findings by position in reverse order to replace from end
        sorted_findings = sorted(
            detection_result.findings,
            key=lambda f: f.start_pos,
            reverse=True
        )

        result = text
        for finding in sorted_findings:
            replacement = self._get_replacement(finding)
            result = (
                result[:finding.start_pos] +
                replacement +
                result[finding.end_pos:]
            )

        return result

    def _get_replacement(self, finding: PIIFinding) -> str:
        """Get replacement string for a finding based on strategy."""
        strategy = self.config.strategy

        if strategy == AnonymizationStrategy.REMOVE:
            return ""

        if strategy == AnonymizationStrategy.REPLACE:
            placeholder = self._placeholders.get(
                finding.pii_type,
                f"<{finding.pii_type.value.upper()}>"
            )
            if self.config.preserve_length:
                return self._adjust_length(placeholder, len(finding.value))
            return placeholder

        if strategy == AnonymizationStrategy.HASH:
            return self._hash_value(finding.value)

        if strategy == AnonymizationStrategy.MASK:
            return self._mask_value(finding)

        return finding.value  # Fallback

    def _adjust_length(self, text: str, target_length: int) -> str:
        """Adjust placeholder to match target length."""
        if len(text) >= target_length:
            return text[:target_length]
        return text + '*' * (target_length - len(text))

    def _hash_value(self, value: str) -> str:
        """Create irreversible hash of value."""
        salt = self.config.salt or ""
        hasher = hashlib.sha256()
        hasher.update(f"{value}{salt}".encode('utf-8'))
        return f"[HASH:{hasher.hexdigest()[:16]}]"

    def _mask_value(self, finding: PIIFinding) -> str:
        """Mask value while preserving some format."""
        value = finding.value
        length = len(value)

        if finding.pii_type == PIIType.EMAIL:
            # Show first 2 chars and domain
            if '@' in value:
                local, domain = value.split('@', 1)
                masked_local = local[:min(2, len(local))] + '***'
                return f"{masked_local}@{domain}"

        elif finding.pii_type == PIIType.PHONE:
            # Show first 3 and last 2 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 11:
                return f"{digits[:3]}****{digits[-4:]}"
            return value[:2] + '****' + value[-2:] if length > 4 else "****"

        elif finding.pii_type == PIIType.ID_CARD:
            # Show first 6 and last 4 digits
            digits = re.sub(r'[^\dXx]', '', value)
            if len(digits) == 18:
                return f"{digits[:6]}********{digits[-4:]}"

        elif finding.pii_type == PIIType.CREDIT_CARD:
            # Show last 4 digits only
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 13:
                return f"****-****-****-{digits[-4:]}"

        elif finding.pii_type == PIIType.PERSON_NAME:
            # Replace given name with *
            if length >= 2:
                return value[0] + '*' * (length - 1)
            return "*"

        # Default: mask middle portion
        if length <= 2:
            return '*' * length
        if length <= 4:
            return value[0] + '*' * (length - 1)

        visible = max(1, length // 4)
        return value[:visible] + '*' * (length - 2 * visible) + value[-visible:]


class PresidioAnonymizerWrapper:
    """Wrapper for Microsoft's Presidio anonymizer if available."""

    def __init__(self):
        self._available = False
        self._anonymizer = None
        try:
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
            self._anonymizer = AnonymizerEngine()
            self._operator_config = OperatorConfig
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        """Check if Presidio is available."""
        return self._available

    def anonymize(
        self,
        text: str,
        analyzer_results: list[Any],
        operators: dict[str, Any] | None = None
    ) -> str:
        """Anonymize using Presidio engine."""
        if not self._available or not self._anonymizer:
            return text

        default_operators = {
            "DEFAULT": self._operator_config("replace", {"new_value": "<PII>"})
        }

        try:
            result = self._anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators or default_operators
            )
            return result.text
        except Exception:
            return text


def create_anonymizer(
    strategy: str = "replace",
    **kwargs
) -> Anonymizer:
    """Factory function to create an anonymizer.

    Args:
        strategy: "replace", "hash", "mask", or "remove"
        **kwargs: Additional config options (salt, preserve_length, etc.)

    Returns:
        Configured Anonymizer instance
    """
    strategy_enum = AnonymizationStrategy(strategy.lower())
    config = AnonymizationConfig(strategy=strategy_enum, **kwargs)
    return Anonymizer(config)


def anonymize_text(
    text: str,
    detector: Any,  # PIIDetector
    strategy: str = "replace"
) -> tuple[str, PIIDetectionResult]:
    """Convenience function to detect and anonymize in one step.

    Args:
        text: Text to anonymize
        detector: PIIDetector instance
        strategy: Anonymization strategy

    Returns:
        Tuple of (anonymized_text, detection_result)
    """
    from .pii_detector import PIIDetector

    detection_result = detector.detect(text)
    anonymizer = create_anonymizer(strategy)
    anonymized = anonymizer.anonymize(text, detection_result)
    return anonymized, detection_result


__all__ = [
    "AnonymizationStrategy",
    "AnonymizationConfig",
    "Anonymizer",
    "PresidioAnonymizerWrapper",
    "create_anonymizer",
    "anonymize_text",
]
