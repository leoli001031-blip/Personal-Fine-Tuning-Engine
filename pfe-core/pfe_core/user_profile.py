"""User Profile system for PFE Phase 2.

This module provides comprehensive user profiling capabilities,
including style preferences, domain preferences, and interaction patterns
derived from implicit signals and conversation history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .storage import resolve_home


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PreferenceScore:
    """A scored preference with confidence and history."""
    value: str
    score: float = 0.5
    confidence: float = 0.5
    frequency: int = 1
    first_seen: datetime = field(default_factory=_utc_now)
    last_updated: datetime = field(default_factory=_utc_now)
    source_signals: list[str] = field(default_factory=list)
    # Historical score window for drift detection
    score_history: list[float] = field(default_factory=list)
    history_window_size: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "frequency": self.frequency,
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "source_signals": self.source_signals,
            "score_history": self.score_history,
            "history_window_size": self.history_window_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreferenceScore":
        return cls(
            value=data["value"],
            score=data.get("score", 0.5),
            confidence=data.get("confidence", 0.5),
            frequency=data.get("frequency", 1),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            source_signals=data.get("source_signals", []),
            score_history=data.get("score_history", []),
            history_window_size=data.get("history_window_size", 20),
        )

    def update(self, new_score: float, signal_id: str | None = None) -> None:
        """Update preference with new observation."""
        # Exponential moving average with decay
        alpha = 0.3  # Learning rate
        self.score = (1 - alpha) * self.score + alpha * new_score
        self.frequency += 1
        self.confidence = min(1.0, self.confidence + 0.05)
        self.last_updated = _utc_now()
        if signal_id and signal_id not in self.source_signals:
            self.source_signals.append(signal_id)
        # Maintain rolling history window
        self.score_history.append(new_score)
        if len(self.score_history) > self.history_window_size:
            self.score_history.pop(0)

    def detect_drift(self, threshold: float = 0.3) -> dict[str, Any] | None:
        """Detect preference drift by comparing recent vs historical averages.

        Returns a drift alert dict if drift exceeds threshold, otherwise None.
        """
        if len(self.score_history) < 4:
            return None
        # Split history: older half vs recent half
        mid = len(self.score_history) // 2
        old_scores = self.score_history[:mid]
        new_scores = self.score_history[mid:]
        old_avg = sum(old_scores) / len(old_scores)
        new_avg = sum(new_scores) / len(new_scores)
        diff = new_avg - old_avg
        if abs(diff) <= threshold:
            return None
        direction = "increase" if diff > 0 else "decrease"
        severity = "high" if abs(diff) > threshold * 2 else "medium"
        return {
            "preference_key": self.value,
            "old_avg": round(old_avg, 4),
            "new_avg": round(new_avg, 4),
            "drift_direction": direction,
            "severity": severity,
        }


@dataclass
class UserProfile:
    """Comprehensive user profile for personalization.

    Contains:
    - Style preferences (formal/casual, concise/detailed, etc.)
    - Domain preferences (programming, writing, learning, etc.)
    - Interaction patterns (likes examples, direct answers, etc.)
    - Update history and metadata
    """
    user_id: str

    # Style preferences: formal vs casual, concise vs detailed, etc.
    style_preferences: dict[str, PreferenceScore] = field(default_factory=dict)

    # Domain preferences: programming, writing, learning, etc.
    domain_preferences: dict[str, PreferenceScore] = field(default_factory=dict)

    # Interaction patterns: likes examples, prefers direct answers, etc.
    interaction_patterns: dict[str, PreferenceScore] = field(default_factory=dict)

    # Temporal dynamics
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    analysis_count: int = 0
    last_analysis_at: Optional[datetime] = None

    # Derived insights
    profile_summary: str = ""
    dominant_style: str = ""
    dominant_domains: list[str] = field(default_factory=list)

    # LLM extraction metadata
    llm_extracted_at: Optional[datetime] = None
    llm_extraction_confidence: float = 0.0
    extracted_by: str = "rule_based"  # "llm" | "rule_based"

    # LLM-extracted structured fields
    llm_identity: dict[str, str] = field(default_factory=dict)
    llm_stable_preferences: list[dict[str, Any]] = field(default_factory=list)
    llm_response_preferences: list[dict[str, Any]] = field(default_factory=list)
    llm_style_indicators: list[dict[str, Any]] = field(default_factory=list)
    llm_domain_interests: list[dict[str, Any]] = field(default_factory=list)
    llm_temporal_notes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "style_preferences": {k: v.to_dict() for k, v in self.style_preferences.items()},
            "domain_preferences": {k: v.to_dict() for k, v in self.domain_preferences.items()},
            "interaction_patterns": {k: v.to_dict() for k, v in self.interaction_patterns.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "analysis_count": self.analysis_count,
            "last_analysis_at": self.last_analysis_at.isoformat() if self.last_analysis_at else None,
            "profile_summary": self.profile_summary,
            "dominant_style": self.dominant_style,
            "dominant_domains": self.dominant_domains,
            "llm_extracted_at": self.llm_extracted_at.isoformat() if self.llm_extracted_at else None,
            "llm_extraction_confidence": self.llm_extraction_confidence,
            "extracted_by": self.extracted_by,
            "llm_identity": self.llm_identity,
            "llm_stable_preferences": self.llm_stable_preferences,
            "llm_response_preferences": self.llm_response_preferences,
            "llm_style_indicators": self.llm_style_indicators,
            "llm_domain_interests": self.llm_domain_interests,
            "llm_temporal_notes": self.llm_temporal_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        profile = cls(
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            analysis_count=data.get("analysis_count", 0),
            last_analysis_at=datetime.fromisoformat(data["last_analysis_at"]) if data.get("last_analysis_at") else None,
            profile_summary=data.get("profile_summary", ""),
            dominant_style=data.get("dominant_style", ""),
            dominant_domains=data.get("dominant_domains", []),
            llm_extracted_at=datetime.fromisoformat(data["llm_extracted_at"]) if data.get("llm_extracted_at") else None,
            llm_extraction_confidence=data.get("llm_extraction_confidence", 0.0),
            extracted_by=data.get("extracted_by", "rule_based"),
            llm_identity=data.get("llm_identity", {}),
            llm_stable_preferences=data.get("llm_stable_preferences", []),
            llm_response_preferences=data.get("llm_response_preferences", []),
            llm_style_indicators=data.get("llm_style_indicators", []),
            llm_domain_interests=data.get("llm_domain_interests", []),
            llm_temporal_notes=data.get("llm_temporal_notes", []),
        )
        profile.style_preferences = {
            k: PreferenceScore.from_dict(v) for k, v in data.get("style_preferences", {}).items()
        }
        profile.domain_preferences = {
            k: PreferenceScore.from_dict(v) for k, v in data.get("domain_preferences", {}).items()
        }
        profile.interaction_patterns = {
            k: PreferenceScore.from_dict(v) for k, v in data.get("interaction_patterns", {}).items()
        }
        return profile

    def update_style_preference(self, key: str, score: float, signal_id: str | None = None) -> None:
        """Update a style preference."""
        if key in self.style_preferences:
            self.style_preferences[key].update(score, signal_id)
        else:
            self.style_preferences[key] = PreferenceScore(
                value=key,
                score=score,
                confidence=0.5,
                source_signals=[signal_id] if signal_id else [],
            )
        self.updated_at = _utc_now()

    def update_domain_preference(self, key: str, score: float, signal_id: str | None = None) -> None:
        """Update a domain preference."""
        if key in self.domain_preferences:
            self.domain_preferences[key].update(score, signal_id)
        else:
            self.domain_preferences[key] = PreferenceScore(
                value=key,
                score=score,
                confidence=0.5,
                source_signals=[signal_id] if signal_id else [],
            )
        self.updated_at = _utc_now()

    def update_interaction_pattern(self, key: str, score: float, signal_id: str | None = None) -> None:
        """Update an interaction pattern."""
        if key in self.interaction_patterns:
            self.interaction_patterns[key].update(score, signal_id)
        else:
            self.interaction_patterns[key] = PreferenceScore(
                value=key,
                score=score,
                confidence=0.5,
                source_signals=[signal_id] if signal_id else [],
            )
        self.updated_at = _utc_now()

    def get_top_style_preferences(self, n: int = 3) -> list[tuple[str, float]]:
        """Get top N style preferences by score."""
        sorted_prefs = sorted(
            self.style_preferences.items(),
            key=lambda x: x[1].score * x[1].confidence,
            reverse=True
        )
        return [(k, v.score) for k, v in sorted_prefs[:n]]

    def get_top_domains(self, n: int = 3) -> list[tuple[str, float]]:
        """Get top N domain preferences by score."""
        sorted_domains = sorted(
            self.domain_preferences.items(),
            key=lambda x: x[1].score * x[1].confidence,
            reverse=True
        )
        return [(k, v.score) for k, v in sorted_domains[:n]]

    def get_top_interaction_patterns(self, n: int = 3) -> list[tuple[str, float]]:
        """Get top N interaction patterns by score."""
        sorted_patterns = sorted(
            self.interaction_patterns.items(),
            key=lambda x: x[1].score * x[1].confidence,
            reverse=True
        )
        return [(k, v.score) for k, v in sorted_patterns[:n]]

    def compute_dominant_traits(self) -> None:
        """Compute dominant style and domains from preferences."""
        # Compute dominant style
        if self.style_preferences:
            top_style = max(self.style_preferences.items(), key=lambda x: x[1].score)
            self.dominant_style = top_style[0]

        # Compute dominant domains
        if self.domain_preferences:
            sorted_domains = sorted(
                self.domain_preferences.items(),
                key=lambda x: x[1].score * x[1].confidence,
                reverse=True
            )
            self.dominant_domains = [k for k, v in sorted_domains[:3]]

    def format_for_prompt(self) -> str:
        """Format profile as a string to include in system prompt."""
        lines = ["\n用户画像："]

        if self.dominant_style:
            lines.append(f"- 偏好风格: {self.dominant_style}")

        if self.dominant_domains:
            lines.append(f"- 常用领域: {', '.join(self.dominant_domains)}")

        top_patterns = self.get_top_interaction_patterns(2)
        if top_patterns:
            patterns_str = ', '.join([f"{k}({v:.2f})" for k, v in top_patterns])
            lines.append(f"- 交互偏好: {patterns_str}")

        if self.profile_summary:
            lines.append(f"- 画像摘要: {self.profile_summary}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def get_preference_vector(self) -> dict[str, float]:
        """Get flattened preference vector for similarity comparison."""
        vector = {}
        for category, prefs in [
            ("style", self.style_preferences),
            ("domain", self.domain_preferences),
            ("pattern", self.interaction_patterns),
        ]:
            for key, pref in prefs.items():
                vector[f"{category}.{key}"] = pref.score * pref.confidence
        return vector

    def detect_preference_drift(self, threshold: float = 0.3) -> list[dict[str, Any]]:
        """Detect drift across all preference categories.

        Compares recent score averages with historical averages for each
        tracked preference. Returns a list of drift alert dicts.
        """
        alerts: list[dict[str, Any]] = []
        all_prefs: dict[str, PreferenceScore] = {}
        all_prefs.update(self.style_preferences)
        all_prefs.update(self.domain_preferences)
        all_prefs.update(self.interaction_patterns)
        for key, pref in all_prefs.items():
            alert = pref.detect_drift(threshold)
            if alert:
                alert["preference_key"] = key
                alerts.append(alert)
        return alerts


class UserProfileStore:
    """Storage for user profiles."""

    def __init__(self, home: str | Path | None = None):
        self.home = Path(resolve_home(home))
        self.profiles_dir = self.home / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, UserProfile] = {}

    def _profile_path(self, user_id: str) -> Path:
        return self.profiles_dir / f"{user_id}_profile.json"

    def get_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        if user_id not in self._cache:
            profile_path = self._profile_path(user_id)
            if profile_path.exists():
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache[user_id] = UserProfile.from_dict(data)
            else:
                self._cache[user_id] = UserProfile(user_id=user_id)
        return self._cache[user_id]

    def save_profile(self, user_id: str) -> None:
        """Save user profile to disk."""
        if user_id in self._cache:
            profile_path = self._profile_path(user_id)
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(self._cache[user_id].to_dict(), f, ensure_ascii=False, indent=2)

    def list_profiles(self) -> list[str]:
        """List all user IDs with profiles."""
        profiles = []
        for path in self.profiles_dir.glob("*_profile.json"):
            user_id = path.stem.replace("_profile", "")
            profiles.append(user_id)
        return profiles

    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile."""
        profile_path = self._profile_path(user_id)
        if profile_path.exists():
            profile_path.unlink()
            if user_id in self._cache:
                del self._cache[user_id]
            return True
        return False

    def export_profile(self, user_id: str) -> dict[str, Any] | None:
        """Export profile as dictionary."""
        profile = self.get_profile(user_id)
        return profile.to_dict()

    def import_profile(self, user_id: str, data: dict[str, Any]) -> UserProfile:
        """Import profile from dictionary."""
        profile = UserProfile.from_dict(data)
        profile.user_id = user_id  # Ensure user_id matches
        profile.updated_at = _utc_now()
        self._cache[user_id] = profile
        self.save_profile(user_id)
        return profile


# Global store instance
_user_profile_store: UserProfileStore | None = None


def get_user_profile_store(home: str | Path | None = None) -> UserProfileStore:
    """Get or create global user profile store."""
    global _user_profile_store
    if _user_profile_store is None:
        _user_profile_store = UserProfileStore(home)
    return _user_profile_store
