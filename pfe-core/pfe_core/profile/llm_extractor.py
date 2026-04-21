"""LLM-based structured profile extraction.

Uses an OpenAI-compatible endpoint (via requests) to extract structured user
profile fields from conversation history. Falls back to rule-based extraction
when the LLM call fails or returns invalid JSON.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


LLM_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "identity": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "location": {"type": "string"},
            },
        },
        "stable_preferences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "preference": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        },
        "response_preferences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "preference": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        },
        "style_indicators": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "trait": {"type": "string"},
                    "evidence": {"type": "string"},
                },
            },
        },
        "domain_interests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "level": {"type": "number"},
                },
            },
        },
        "temporal_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "date_hint": {"type": "string"},
                },
            },
        },
    },
}

EXTRACTION_PROMPT_TEMPLATE = """You are a structured user-profile extraction engine.

Analyze the following conversation history and produce a JSON object that captures the user's profile.

Rules:
- Only include fields when there is clear evidence in the conversation.
- If a field has no evidence, return its default empty value (empty string / empty list).
- Do not hallucinate. Confidence should reflect how explicit the evidence is.
- Output ONLY valid JSON. No markdown, no explanation.

Conversation history:
%(conversation_text)s

Required JSON structure:
{
  "identity": {"name": "", "role": "", "location": ""},
  "stable_preferences": [{"category": "", "preference": "", "confidence": 0.0}],
  "response_preferences": [{"aspect": "", "preference": "", "confidence": 0.0}],
  "style_indicators": [{"trait": "", "evidence": ""}],
  "domain_interests": [{"domain": "", "level": 0.0}],
  "temporal_notes": [{"note": "", "date_hint": ""}]
}
"""


@dataclass
class DriftAlert:
    """Alert emitted when a preference score drifts beyond a threshold."""
    preference_key: str
    old_avg: float
    new_avg: float
    drift_direction: str  # "increase" | "decrease"
    severity: str  # "low" | "medium" | "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "preference_key": self.preference_key,
            "old_avg": round(self.old_avg, 4),
            "new_avg": round(self.new_avg, 4),
            "drift_direction": self.drift_direction,
            "severity": self.severity,
        }


class LLMProfileExtractor:
    """Extract structured profile fields from conversation messages using an LLM."""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-3.5-turbo",
        timeout: float = 30.0,
        temperature: float = 0.2,
    ):
        self.api_base = (api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("PFE_LLM_EXTRACTOR_MODEL", "gpt-3.5-turbo")
        self.timeout = timeout
        self.temperature = temperature

    def _build_prompt(self, messages: list[dict]) -> str:
        lines: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        conversation_text = "\n".join(lines)
        return EXTRACTION_PROMPT_TEMPLATE % {"conversation_text": conversation_text}

    def _call_llm(self, prompt: str) -> str:
        import requests

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("LLM response contains no choices")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise ValueError("LLM response content is empty")
        return str(content).strip()

    def _parse_json(self, text: str) -> dict[str, Any]:
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            snippet = text[start:end].strip()
            return json.loads(snippet)
        if "```" in text:
            start = text.index("```") + len("```")
            end = text.index("```", start)
            snippet = text[start:end].strip()
            return json.loads(snippet)

        # Try first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])

        raise ValueError("Unable to extract valid JSON from LLM response")

    def _validate_output(self, data: dict[str, Any]) -> dict[str, Any]:
        """Ensure all required top-level keys exist with correct types."""
        defaults: dict[str, Any] = {
            "identity": {"name": "", "role": "", "location": ""},
            "stable_preferences": [],
            "response_preferences": [],
            "style_indicators": [],
            "domain_interests": [],
            "temporal_notes": [],
        }
        result: dict[str, Any] = {}
        for key, default in defaults.items():
            value = data.get(key)
            if isinstance(value, dict):
                result[key] = {k: str(v) if v is not None else "" for k, v in value.items()}
            elif isinstance(value, list):
                result[key] = [v for v in value if isinstance(v, dict)]
            else:
                result[key] = default
        return result

    def extract_from_conversation(self, messages: list[dict]) -> dict[str, Any]:
        """Extract structured profile from conversation messages.

        Returns a dict with keys: identity, stable_preferences, response_preferences,
        style_indicators, domain_interests, temporal_notes.

        If the LLM call fails or returns invalid data, falls back to an empty
        structure so callers can decide whether to use rule-based fallback.
        """
        if not messages:
            validated = self._validate_output({})
            validated["_extraction_meta"] = {
                "success": True,
                "method": "llm",
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }
            return validated

        prompt = self._build_prompt(messages)
        try:
            raw_text = self._call_llm(prompt)
            parsed = self._parse_json(raw_text)
            validated = self._validate_output(parsed)
            validated["_extraction_meta"] = {
                "success": True,
                "method": "llm",
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }
            return validated
        except Exception as exc:
            logger.warning("LLM profile extraction failed: %s", exc)
            fallback = self._validate_output({})
            fallback["_extraction_meta"] = {
                "success": False,
                "method": "llm_failed",
                "error": str(exc),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }
            return fallback
