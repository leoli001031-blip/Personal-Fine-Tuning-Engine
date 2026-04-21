"""User memory/profile system for PFE.

This module provides user profiling and memory capabilities,
allowing the model to remember user preferences, facts, and context
across conversations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .data_policy import extract_user_data_candidates, route_user_datum
from .storage import resolve_home


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class UserFact:
    """A single fact about the user."""
    key: str
    value: str
    confidence: float = 1.0
    source: str = "extraction"  # extraction, explicit, inference
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserFact":
        return cls(
            key=data["key"],
            value=data["value"],
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "extraction"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
        )


@dataclass
class UserProfile:
    """User profile containing facts and preferences."""
    user_id: str
    facts: dict[str, UserFact] = field(default_factory=dict)
    preferences: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    total_interactions: int = 0

    def add_fact(self, key: str, value: str, confidence: float = 1.0, source: str = "extraction") -> None:
        """Add or update a fact about the user."""
        if key in self.facts:
            self.facts[key].value = value
            self.facts[key].confidence = confidence
            self.facts[key].updated_at = _utc_now()
        else:
            self.facts[key] = UserFact(key=key, value=value, confidence=confidence, source=source)
        self.updated_at = _utc_now()

    def add_preference(self, key: str, value: str) -> None:
        """Add or update a user preference string."""
        self.preferences[key] = value
        self.updated_at = _utc_now()

    def get_fact(self, key: str) -> Optional[str]:
        """Get a fact value and update access stats."""
        if key in self.facts:
            fact = self.facts[key]
            fact.access_count += 1
            fact.last_accessed = _utc_now()
            return fact.value
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "facts": {k: v.to_dict() for k, v in self.facts.items()},
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_interactions": self.total_interactions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        profile = cls(
            user_id=data["user_id"],
            preferences=data.get("preferences", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            total_interactions=data.get("total_interactions", 0),
        )
        profile.facts = {
            k: UserFact.from_dict(v) for k, v in data.get("facts", {}).items()
        }
        return profile

    def format_for_prompt(self, max_facts: int = 5, max_preferences: int = 3) -> str:
        """Format profile as a string to include in system prompt."""
        if not self.facts and not self.preferences:
            return ""

        sorted_facts = sorted(
            self.facts.values(),
            key=lambda f: (f.confidence, f.updated_at.timestamp()),
            reverse=True,
        )[:max_facts]

        lines = ["\n关于用户的信息："]
        for fact in sorted_facts:
            lines.append(f"- {fact.key}: {fact.value}")

        if self.preferences:
            lines.append("- 长期偏好:")
            for value in list(self.preferences.values())[:max_preferences]:
                lines.append(f"  - {value}")

        return "\n".join(lines)


class UserMemoryStore:
    """Storage for user profiles and memories."""

    def __init__(self, home: str | Path | None = None):
        self.home = Path(resolve_home(home))
        self.profiles_dir = self.home / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, UserProfile] = {}
        self._processed_message_keys: set[str] = set()

    def _profile_path(self, user_id: str) -> Path:
        return self.profiles_dir / f"{user_id}.json"

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

    def _processed_message_key(self, user_id: str, user_message: str, request_id: str | None = None) -> str:
        digest = hashlib.sha256(user_message.strip().encode("utf-8")).hexdigest()[:16]
        return f"{user_id}:{request_id or 'no-request'}:{digest}"

    def ingest_explicit_user_data(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str = "",
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest explicit user data through the shared routing policy.

        This method is intentionally idempotent per `(user_id, request_id, user_message)`
        within a process so the same request can be observed by both the server layer
        and the collector layer without double-counting.
        """
        del assistant_message
        profile = self.get_profile(user_id)
        new_facts: list[UserFact] = []
        stored_preferences: list[str] = []
        blocked_candidates: list[dict[str, Any]] = []
        candidates_payload: list[dict[str, Any]] = []

        message = (user_message or "").strip()
        if not message:
            return {
                "processed": False,
                "reason": "empty_user_message",
                "new_facts": [],
                "stored_preferences": [],
                "blocked_candidates": [],
                "candidates": [],
            }

        dedupe_key = self._processed_message_key(user_id, message, request_id=request_id)
        if dedupe_key in self._processed_message_keys:
            return {
                "processed": False,
                "reason": "duplicate_request_message",
                "new_facts": [],
                "stored_preferences": [],
                "blocked_candidates": [],
                "candidates": [],
            }

        candidates = extract_user_data_candidates(message)
        for datum in candidates:
            decision = route_user_datum(datum)
            candidates_payload.append(
                {
                    "key": datum.key,
                    "value": datum.value,
                    "kind": datum.kind,
                    "primary_lane": decision.primary_lane,
                    "training_target": decision.training_target,
                    "reason": decision.reason,
                }
            )

            if decision.pii_blocked or decision.primary_lane == "discard":
                blocked_candidates.append(candidates_payload[-1])
                continue

            if decision.should_persist_memory:
                profile.add_fact(datum.key, datum.value, confidence=datum.confidence, source=datum.source)
                new_facts.append(profile.facts[datum.key])

            if decision.should_update_profile:
                preference_key = self._preference_key(datum.kind, datum.value)
                profile.add_preference(preference_key, datum.value)
                stored_preferences.append(datum.value)

        if candidates_payload:
            profile.total_interactions += 1
            self.save_profile(user_id)
            self._processed_message_keys.add(dedupe_key)

        return {
            "processed": bool(candidates_payload),
            "reason": "ok" if candidates_payload else "no_explicit_user_data_candidates",
            "new_facts": new_facts,
            "stored_preferences": stored_preferences,
            "blocked_candidates": blocked_candidates,
            "candidates": candidates_payload,
        }

    def _preference_key(self, kind: str, value: str) -> str:
        normalized = "".join(ch.lower() if ch.isascii() else ch for ch in value.strip())
        normalized = "_".join(part for part in normalized.replace("，", " ").replace(",", " ").split() if part)
        normalized = normalized[:48] or "preference"
        return f"{kind}:{normalized}"

    def extract_facts_from_conversation(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        request_id: str | None = None,
    ) -> list[UserFact]:
        """Extract and route explicit user data from a conversation turn."""
        result = self.ingest_explicit_user_data(
            user_id=user_id,
            user_message=user_message,
            assistant_message=assistant_message,
            request_id=request_id,
        )
        return list(result.get("new_facts", []))

    def get_profile_for_prompt(self, user_id: str) -> str:
        """Get formatted profile for inclusion in system prompt."""
        profile = self.get_profile(user_id)
        return profile.format_for_prompt()


# Global store instance
_user_memory_store: UserMemoryStore | None = None


def get_user_memory_store(home: str | Path | None = None) -> UserMemoryStore:
    """Get or create global user memory store."""
    global _user_memory_store
    requested_home = Path(resolve_home(home)) if home is not None else None
    if (
        _user_memory_store is None
        or (requested_home is not None and _user_memory_store.home != requested_home)
    ):
        _user_memory_store = UserMemoryStore(home)
    return _user_memory_store
