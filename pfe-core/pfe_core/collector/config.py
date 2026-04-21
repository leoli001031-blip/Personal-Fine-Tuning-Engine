"""Configuration for the ChatCollector signal extraction system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CollectorConfig:
    """Configuration for signal collection and extraction.

    Controls how implicit signals are extracted from user interactions,
    including confidence thresholds and time-based heuristics.
    """

    enabled: bool = True
    """Whether signal collection is enabled."""

    accept_confidence_threshold: float = 0.5
    """Minimum confidence for accept signals to be recorded."""

    edit_confidence_threshold: float = 0.5
    """Minimum confidence for edit signals to be recorded."""

    reject_confidence_threshold: float = 0.5
    """Minimum confidence for reject signals to be recorded."""

    regenerate_confidence_threshold: float = 0.5
    """Minimum confidence for regenerate signals to be recorded."""

    edit_distance_metric: str = "levenshtein"
    """Metric for calculating edit distance (levenshtein or jaro_winkler)."""

    time_decay_enabled: bool = True
    """Whether to apply time-based confidence decay for accept signals."""

    strong_accept_threshold_seconds: float = 5.0
    """Response time below which is considered strong accept."""

    weak_accept_threshold_seconds: float = 60.0
    """Response time above which is considered weak accept."""

    store_interactions: bool = True
    """Whether to store raw ChatInteraction records."""

    max_interaction_history: int = 1000
    """Maximum number of interactions to keep in memory."""

    signal_rules: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "accept": {
            "base_confidence": 0.7,
            "strong_multiplier": 1.29,  # 0.7 * 1.29 ≈ 0.9
            "weak_multiplier": 0.57,    # 0.7 * 0.57 ≈ 0.4
        },
        "edit": {
            "slight_threshold": 0.2,
            "moderate_threshold": 0.5,
            "slight_confidence": 0.6,
            "moderate_confidence": 0.8,
            "strong_confidence": 0.9,
        },
        "reject": {
            "base_confidence": 0.95,
        },
        "regenerate": {
            "base_confidence": 0.85,
        },
    })
    """Fine-grained signal extraction rules with configurable thresholds."""

    # Contradiction Detection Configuration
    contradiction_detection_enabled: bool = True
    """Whether to detect contradictory signals within a session."""

    contradiction_window_seconds: float = 300.0
    """Time window for detecting contradictions within a session."""

    # Replay Buffer Configuration
    enable_replay_buffer: bool = True
    """Whether to maintain a replay buffer for low-quality signals."""

    replay_minimum_confidence: float = 0.3
    """Minimum confidence for a signal to be eligible for replay."""

    replay_confidence_threshold: float = 0.5
    """Confidence below which a stored signal is considered a replay candidate."""

    # PII Detection Configuration
    pii_detection_enabled: bool = True
    """Whether to enable PII detection on signals."""

    pii_sensitivity: str = "medium"
    """PII detection sensitivity: low, medium, or high."""

    pii_anonymization_strategy: str = "mask"
    """Anonymization strategy: replace, hash, mask, or remove."""

    pii_action_on_detect: str = "anonymize"
    """Action when PII detected: anonymize, block, or flag."""

    pii_min_confidence: float = 0.7
    """Minimum confidence threshold for PII detection."""

    pii_audit_enabled: bool = True
    """Whether to enable PII audit logging."""

    explicit_user_data_routing_enabled: bool = True
    """Whether to route explicit user facts/preferences into memory/profile."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0 <= self.accept_confidence_threshold <= 1:
            raise ValueError("accept_confidence_threshold must be between 0 and 1")
        if not 0 <= self.edit_confidence_threshold <= 1:
            raise ValueError("edit_confidence_threshold must be between 0 and 1")
        if not 0 <= self.reject_confidence_threshold <= 1:
            raise ValueError("reject_confidence_threshold must be between 0 and 1")
        if not 0 <= self.regenerate_confidence_threshold <= 1:
            raise ValueError("regenerate_confidence_threshold must be between 0 and 1")
        if self.strong_accept_threshold_seconds >= self.weak_accept_threshold_seconds:
            raise ValueError("strong_accept_threshold_seconds must be less than weak_accept_threshold_seconds")
        if self.edit_distance_metric not in ("levenshtein", "jaro_winkler"):
            raise ValueError("edit_distance_metric must be 'levenshtein' or 'jaro_winkler'")
        if self.pii_sensitivity not in ("low", "medium", "high"):
            raise ValueError("pii_sensitivity must be 'low', 'medium', or 'high'")
        if self.pii_anonymization_strategy not in ("replace", "hash", "mask", "remove"):
            raise ValueError("pii_anonymization_strategy must be 'replace', 'hash', 'mask', or 'remove'")
        if self.pii_action_on_detect not in ("anonymize", "block", "flag"):
            raise ValueError("pii_action_on_detect must be 'anonymize', 'block', or 'flag'")
