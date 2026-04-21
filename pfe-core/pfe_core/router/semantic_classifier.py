"""Lightweight semantic classifier for scenario routing.

Uses TF-IDF + cosine similarity when sklearn is available,
falling back to difflib.SequenceMatcher for a pure-Python implementation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..scenarios import ScenarioConfig


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization with lowercasing."""
    return re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())


def _fallback_similarity(text_a: str, text_b: str) -> float:
    """Pure-Python text similarity using SequenceMatcher."""
    if not text_a or not text_b:
        return 0.0
    return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


def _aggregate_scores(scores: list[float]) -> float:
    """Aggregate multiple similarity scores into a single score.

    Uses max for strong signal, plus a small mean bonus for breadth.
    """
    if not scores:
        return 0.0
    return max(scores) * 0.8 + (sum(scores) / len(scores)) * 0.2


@dataclass
class SemanticClassificationResult:
    """Result of semantic classification."""

    scenario_id: str
    confidence: float  # 0-1
    scores: dict[str, float] = field(default_factory=dict)
    method: str = ""  # "tfidf" or "fallback"


class SemanticClassifier:
    """Lightweight semantic classifier for routing.

    When sklearn is available, uses TF-IDF + cosine similarity against
    scenario example phrases. Falls back to difflib.SequenceMatcher.
    """

    def __init__(
        self,
        scenarios: dict[str, ScenarioConfig] | None = None,
    ) -> None:
        from ..scenarios import BUILTIN_SCENARIOS

        self.scenarios = scenarios or BUILTIN_SCENARIOS
        self._method = "tfidf" if _SKLEARN_AVAILABLE else "fallback"
        self._vectorizer: "TfidfVectorizer | None" = None
        self._scenario_ids: list[str] = []
        self._example_matrix: "Any | None" = None
        self._all_examples: list[str] = []
        self._example_to_scenario: list[str] = []
        self._build_index()

    def _build_index(self) -> None:
        """Build the semantic index from scenario example_phrases."""
        self._scenario_ids = []
        self._all_examples = []
        self._example_to_scenario = []

        for scenario_id, scenario in self.scenarios.items():
            self._scenario_ids.append(scenario_id)
            phrases = getattr(scenario, "example_phrases", None) or []
            for phrase in phrases:
                self._all_examples.append(phrase)
                self._example_to_scenario.append(scenario_id)

        if not self._all_examples:
            return

        if self._method == "tfidf":
            try:
                self._vectorizer = TfidfVectorizer(
                    tokenizer=_tokenize,
                    token_pattern=None,
                    lowercase=False,
                )
                self._example_matrix = self._vectorizer.fit_transform(self._all_examples)
            except Exception:
                # TF-IDF build failed; degrade to fallback
                self._method = "fallback"
                self._vectorizer = None
                self._example_matrix = None

    def classify(self, text: str) -> SemanticClassificationResult:
        """Classify text against scenario examples.

        Args:
            text: User input text to classify.

        Returns:
            SemanticClassificationResult with best-matching scenario and scores.
        """
        if not text or not text.strip():
            return SemanticClassificationResult(
                scenario_id="chat",
                confidence=0.0,
                scores={"chat": 0.0},
                method=self._method,
            )

        if not self._all_examples:
            return SemanticClassificationResult(
                scenario_id="chat",
                confidence=0.0,
                scores={"chat": 0.0},
                method=self._method,
            )

        raw_scores: dict[str, list[float]] = {sid: [] for sid in self._scenario_ids}

        if self._method == "tfidf" and self._vectorizer is not None and self._example_matrix is not None:
            try:
                text_vec = self._vectorizer.transform([text])
                sims = cosine_similarity(text_vec, self._example_matrix)[0]
                for idx, sim in enumerate(sims):
                    sid = self._example_to_scenario[idx]
                    raw_scores[sid].append(float(sim))
            except Exception:
                # Degrade to fallback on any error
                return self._fallback_classify(text)
        else:
            return self._fallback_classify(text)

        aggregated = {
            sid: _aggregate_scores(scores) if scores else 0.0
            for sid, scores in raw_scores.items()
        }

        # Normalize to 0-1
        max_score = max(aggregated.values()) if aggregated else 0.0
        if max_score > 0:
            normalized = {sid: min(1.0, score / max_score) for sid, score in aggregated.items()}
        else:
            normalized = {sid: 0.0 for sid in aggregated}

        best_sid = max(normalized, key=normalized.get) if normalized else "chat"
        best_confidence = normalized.get(best_sid, 0.0)

        return SemanticClassificationResult(
            scenario_id=best_sid,
            confidence=best_confidence,
            scores=normalized,
            method=self._method,
        )

    def _fallback_classify(self, text: str) -> SemanticClassificationResult:
        """Pure-Python classification using SequenceMatcher."""
        raw_scores: dict[str, list[float]] = {sid: [] for sid in self._scenario_ids}

        for idx, example in enumerate(self._all_examples):
            sid = self._example_to_scenario[idx]
            sim = _fallback_similarity(text, example)
            raw_scores[sid].append(sim)

        aggregated = {
            sid: _aggregate_scores(scores) if scores else 0.0
            for sid, scores in raw_scores.items()
        }

        max_score = max(aggregated.values()) if aggregated else 0.0
        if max_score > 0:
            normalized = {sid: min(1.0, score / max_score) for sid, score in aggregated.items()}
        else:
            normalized = {sid: 0.0 for sid in aggregated}

        best_sid = max(normalized, key=normalized.get) if normalized else "chat"
        best_confidence = normalized.get(best_sid, 0.0)

        return SemanticClassificationResult(
            scenario_id=best_sid,
            confidence=best_confidence,
            scores=normalized,
            method="fallback",
        )

    def refresh_index(self) -> None:
        """Rebuild the semantic index (call after scenarios change)."""
        self._build_index()
