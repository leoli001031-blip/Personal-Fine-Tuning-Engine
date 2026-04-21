"""Multi-scenario Router system for PFE Phase 2.

This module implements the routing logic for selecting the most appropriate
adapter based on user input intent classification.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import defaultdict

from ..scenarios import ScenarioConfig, BUILTIN_SCENARIOS, get_builtin_scenario
from ..config import PFEConfig
from ..user_profile import UserProfile
from .semantic_classifier import SemanticClassifier, SemanticClassificationResult

_DEFAULT_SCENARIO_CONFIG_PATH = "~/.pfe/scenarios.json"


@dataclass
class IntentClassification:
    """Result of intent classification.

    Attributes:
        primary_intent: The primary detected intent/scenario_id
        confidence: Confidence score (0-1)
        all_scores: Dictionary of all scenario scores
        matched_keywords: Keywords that triggered the classification
        reasoning: Human-readable reasoning for the classification
    """

    primary_intent: str
    confidence: float
    all_scores: dict[str, float] = field(default_factory=dict)
    matched_keywords: dict[str, list[str]] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "primary_intent": self.primary_intent,
            "confidence": self.confidence,
            "all_scores": self.all_scores,
            "matched_keywords": self.matched_keywords,
            "reasoning": self.reasoning,
        }


@dataclass
class RoutingResult:
    """Result of routing decision.

    Attributes:
        scenario_id: Selected scenario ID
        adapter_version: Adapter version to use
        confidence: Routing confidence (0-1)
        fallback: Whether this is a fallback selection
        reasoning: Explanation of the routing decision
    """

    scenario_id: str
    adapter_version: str
    confidence: float
    fallback: bool = False
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "scenario_id": self.scenario_id,
            "adapter_version": self.adapter_version,
            "confidence": self.confidence,
            "fallback": self.fallback,
            "reasoning": self.reasoning,
        }


class IntentClassifier:
    """Rule-based intent classifier using keyword matching.

    This classifier uses a combination of:
    1. Exact keyword matching
    2. Partial word matching
    3. Example similarity (basic)
    4. Scenario priority weighting
    """

    def __init__(self, scenarios: dict[str, ScenarioConfig] | None = None):
        self.scenarios = scenarios or BUILTIN_SCENARIOS
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficient matching."""
        self._patterns: dict[str, list[tuple[str, re.Pattern]]] = {}
        for scenario_id, scenario in self.scenarios.items():
            patterns = []
            for keyword in scenario.trigger_keywords:
                # Create word boundary pattern for multi-word keywords
                if len(keyword.split()) > 1:
                    pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
                else:
                    pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
                patterns.append((keyword, pattern))
            self._patterns[scenario_id] = patterns

    def classify(self, text: str) -> IntentClassification:
        """Classify the intent of the input text.

        Args:
            text: User input text to classify

        Returns:
            IntentClassification with primary intent and confidence scores
        """
        if not text or not text.strip():
            return IntentClassification(
                primary_intent="chat",
                confidence=0.0,
                all_scores={"chat": 0.0},
                matched_keywords={},
                reasoning="Empty input, defaulting to chat",
            )

        text_lower = text.lower().strip()
        scores: dict[str, float] = defaultdict(float)
        matched_keywords: dict[str, list[str]] = defaultdict(list)

        # Score each scenario based on keyword matches
        for scenario_id, scenario in self.scenarios.items():
            score = 0.0
            matches = []

            # Check keyword matches
            patterns = self._patterns.get(scenario_id, [])
            for keyword, pattern in patterns:
                matches_count = len(pattern.findall(text_lower))
                if matches_count > 0:
                    # Weight by keyword specificity (longer keywords = more specific)
                    keyword_weight = 1.0 + (len(keyword) / 20.0)
                    score += matches_count * keyword_weight
                    matches.append(keyword)

            # Check example similarity (simple containment check)
            for example in scenario.examples:
                example_lower = example.lower()
                # Calculate word overlap
                text_words = set(text_lower.split())
                example_words = set(example_lower.split())
                overlap = text_words & example_words
                if overlap:
                    overlap_score = len(overlap) / max(len(example_words), 1)
                    score += overlap_score * 0.5  # Lower weight for example matches

            # Apply priority boost
            priority_boost = scenario.priority / 100.0
            score *= (1.0 + priority_boost)

            # Apply scenario-specific confidence boost
            score += scenario.confidence_boost

            if score > 0:
                scores[scenario_id] = score
                matched_keywords[scenario_id] = matches

        # Normalize scores to 0-1 range
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                normalized_scores = {
                    k: min(1.0, v / max_score) for k, v in scores.items()
                }
            else:
                normalized_scores = {k: 0.0 for k in scores}
        else:
            normalized_scores = {"chat": 0.0}

        # Select primary intent
        if normalized_scores:
            primary_intent = max(normalized_scores, key=normalized_scores.get)
            confidence = normalized_scores[primary_intent]
        else:
            primary_intent = "chat"
            confidence = 0.0

        # Build reasoning
        reasoning_parts = []
        if primary_intent in matched_keywords and matched_keywords[primary_intent]:
            keywords_str = ", ".join(matched_keywords[primary_intent][:5])
            reasoning_parts.append(f"Matched keywords: {keywords_str}")
        reasoning_parts.append(f"Primary intent: {primary_intent} (confidence: {confidence:.2f})")

        return IntentClassification(
            primary_intent=primary_intent,
            confidence=confidence,
            all_scores=dict(normalized_scores),
            matched_keywords=dict(matched_keywords),
            reasoning="; ".join(reasoning_parts),
        )


class ConfidenceScorer:
    """Calculate and adjust confidence scores for routing decisions.

    This scorer considers:
    1. Base classification confidence
    2. Input length (longer inputs = more confident)
    3. Keyword specificity
    4. Historical accuracy (placeholder for future ML)
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.3,
        high_confidence_threshold: float = 0.7,
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold

    def score(
        self,
        classification: IntentClassification,
        text: str,
    ) -> float:
        """Calculate final confidence score.

        Args:
            classification: The intent classification result
            text: Original input text

        Returns:
            Adjusted confidence score (0-1)
        """
        base_confidence = classification.confidence

        # Adjust based on input length (longer inputs tend to be more classifiable)
        text_length = len(text.strip())
        if text_length < 10:
            length_factor = 0.8  # Very short inputs are less certain
        elif text_length < 50:
            length_factor = 0.9
        elif text_length < 200:
            length_factor = 1.0
        else:
            length_factor = 1.05  # Longer inputs may have more context

        # Adjust based on keyword diversity
        primary_matches = classification.matched_keywords.get(classification.primary_intent, [])
        unique_keywords = len(set(primary_matches))
        if unique_keywords >= 3:
            diversity_factor = 1.1
        elif unique_keywords >= 2:
            diversity_factor = 1.05
        else:
            diversity_factor = 1.0

        # Calculate final score
        final_score = base_confidence * length_factor * diversity_factor
        return min(1.0, max(0.0, final_score))

    def is_confident_enough(self, score: float) -> bool:
        """Check if confidence score meets the minimum threshold."""
        return score >= self.min_confidence_threshold

    def is_high_confidence(self, score: float) -> bool:
        """Check if confidence score is high."""
        return score >= self.high_confidence_threshold


class ScenarioRouter:
    """Main router for selecting scenarios and adapters.

    This router coordinates:
    1. Intent classification
    2. Confidence scoring
    3. Scenario-to-adapter mapping
    4. Fallback handling
    5. Scenario configuration persistence
    """

    def __init__(
        self,
        config: PFEConfig | None = None,
        scenarios: dict[str, ScenarioConfig] | None = None,
        scenario_adapter_map: dict[str, str] | None = None,
    ):
        self.config = config or PFEConfig.load()
        self._scenarios_config_path = self._get_scenarios_config_path()
        self.scenarios = scenarios or BUILTIN_SCENARIOS.copy()
        self.classifier = IntentClassifier(self.scenarios)
        self.semantic_classifier = SemanticClassifier(self.scenarios)
        self.scorer = ConfidenceScorer(
            min_confidence_threshold=getattr(self.config.router, 'min_routing_confidence', 0.3),
            high_confidence_threshold=getattr(self.config.router, 'confidence_threshold', 0.7),
        )
        self.scenario_adapter_map = scenario_adapter_map or {}
        self._routing_cache: dict[str, RoutingResult] = {}
        self._cache_enabled = True
        # Load persisted scenario bindings
        self._load_scenario_bindings()

    def _get_scenarios_config_path(self) -> Path:
        """Get the path for scenario configuration storage."""
        config_path = str(getattr(self.config.router, "scenario_config_path", _DEFAULT_SCENARIO_CONFIG_PATH) or _DEFAULT_SCENARIO_CONFIG_PATH).strip()
        if not config_path or config_path == _DEFAULT_SCENARIO_CONFIG_PATH:
            return self.config.home / "scenarios.json"
        if config_path.startswith("~/.pfe/"):
            return self.config.home / config_path[len("~/.pfe/") :]
        return Path(config_path).expanduser()

    def _load_scenario_bindings(self) -> None:
        """Load persisted scenario-to-adapter bindings."""
        if not self._scenarios_config_path.exists():
            return
        try:
            import json
            data = json.loads(self._scenarios_config_path.read_text(encoding='utf-8'))
            bindings = data.get('scenario_bindings', {})
            for scenario_id, adapter_version in bindings.items():
                if scenario_id in self.scenarios:
                    self.scenarios[scenario_id].adapter_version = adapter_version
                    self.scenario_adapter_map[scenario_id] = adapter_version
        except Exception:
            pass  # Silently ignore load errors

    def _save_scenario_bindings(self) -> None:
        """Persist scenario-to-adapter bindings."""
        try:
            import json
            self._scenarios_config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'scenario_bindings': self.scenario_adapter_map,
                'updated_at': datetime.now(timezone.utc).isoformat(),
            }
            self._scenarios_config_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception:
            pass  # Silently ignore save errors

    def add_scenario(self, scenario: ScenarioConfig) -> None:
        """Add a custom scenario to the router."""
        self.scenarios[scenario.scenario_id] = scenario
        self.classifier._compile_patterns()
        self.semantic_classifier.refresh_index()

    def remove_scenario(self, scenario_id: str) -> bool:
        """Remove a scenario from the router."""
        if scenario_id in self.scenarios:
            del self.scenarios[scenario_id]
            self.classifier._compile_patterns()
            self.semantic_classifier.refresh_index()
            return True
        return False

    def bind_scenario_to_adapter(self, scenario_id: str, adapter_version: str) -> bool:
        """Bind a scenario to a specific adapter version."""
        if scenario_id not in self.scenarios:
            return False
        self.scenarios[scenario_id].adapter_version = adapter_version
        self.scenario_adapter_map[scenario_id] = adapter_version
        self._save_scenario_bindings()
        return True

    def route(
        self,
        text: str,
        use_cache: bool = True,
        user_profile: UserProfile | None = None,
    ) -> RoutingResult:
        """Route input text to the appropriate scenario and adapter.

        Args:
            text: User input text
            use_cache: Whether to use routing cache
            user_profile: Optional user profile for preference-aware routing

        Returns:
            RoutingResult with selected scenario and adapter
        """
        # Check cache
        cache_key = text.strip().lower()
        if use_cache and self._cache_enabled and cache_key in self._routing_cache:
            return self._routing_cache[cache_key]

        strategy = getattr(self.config.router, "strategy", "keyword")

        # Classify intent
        if strategy == "hybrid":
            classification = self._hybrid_classify(text)
        else:
            classification = self.classifier.classify(text)

        # Apply profile boost if available
        if user_profile is not None:
            classification = self._apply_profile_boost(classification, user_profile)

        # Calculate confidence
        confidence = self.scorer.score(classification, text)

        # Determine if we should use fallback
        if not self.scorer.is_confident_enough(confidence):
            result = RoutingResult(
                scenario_id="chat",
                adapter_version=self._get_adapter_for_scenario("chat"),
                confidence=confidence,
                fallback=True,
                reasoning=f"Low confidence ({confidence:.2f}) for '{classification.primary_intent}', falling back to default",
            )
        else:
            scenario_id = classification.primary_intent
            adapter_version = self._get_adapter_for_scenario(scenario_id)
            result = RoutingResult(
                scenario_id=scenario_id,
                adapter_version=adapter_version,
                confidence=confidence,
                fallback=False,
                reasoning=classification.reasoning,
            )

        # Cache result
        if self._cache_enabled:
            self._routing_cache[cache_key] = result

        return result

    def _hybrid_classify(self, text: str) -> IntentClassification:
        """Combine keyword and semantic scores with weighted fusion.

        Default weights: keyword 0.6, semantic 0.4.
        """
        keyword_result = self.classifier.classify(text)
        semantic_result = self.semantic_classifier.classify(text)

        keyword_weight = 0.6
        semantic_weight = 0.4

        all_scenarios = set(keyword_result.all_scores.keys()) | set(semantic_result.scores.keys())
        fused_scores: dict[str, float] = {}
        matched_keywords: dict[str, list[str]] = dict(keyword_result.matched_keywords)

        for sid in all_scenarios:
            kw_score = keyword_result.all_scores.get(sid, 0.0)
            sem_score = semantic_result.scores.get(sid, 0.0)
            fused_scores[sid] = kw_score * keyword_weight + sem_score * semantic_weight

        if fused_scores:
            primary_intent = max(fused_scores, key=fused_scores.get)
            confidence = fused_scores[primary_intent]
        else:
            primary_intent = "chat"
            confidence = 0.0

        reasoning_parts = []
        if primary_intent in matched_keywords and matched_keywords[primary_intent]:
            keywords_str = ", ".join(matched_keywords[primary_intent][:5])
            reasoning_parts.append(f"Matched keywords: {keywords_str}")
        reasoning_parts.append(
            f"Primary intent: {primary_intent} (confidence: {confidence:.2f}, method=hybrid)"
        )

        return IntentClassification(
            primary_intent=primary_intent,
            confidence=confidence,
            all_scores=fused_scores,
            matched_keywords=matched_keywords,
            reasoning="; ".join(reasoning_parts),
        )

    def _apply_profile_boost(
        self,
        classification: IntentClassification,
        user_profile: UserProfile,
    ) -> IntentClassification:
        """Boost scenario scores based on user domain preferences.

        If a domain preference score > 0.7, the corresponding scenario gets +0.15.
        """
        boosted_scores = dict(classification.all_scores)
        for domain_key, pref in user_profile.domain_preferences.items():
            if pref.score > 0.7:
                # Map domain key to scenario_id heuristically
                scenario_id = domain_key.lower().replace(" ", "_")
                if scenario_id not in boosted_scores:
                    # Try fuzzy match against known scenario ids
                    for sid in self.scenarios:
                        if sid in scenario_id or scenario_id in sid:
                            scenario_id = sid
                            break
                if scenario_id in boosted_scores:
                    boosted_scores[scenario_id] = min(1.0, boosted_scores[scenario_id] + 0.15)

        # Recompute primary intent after boost
        if boosted_scores:
            primary_intent = max(boosted_scores, key=boosted_scores.get)
            confidence = boosted_scores[primary_intent]
        else:
            primary_intent = classification.primary_intent
            confidence = classification.confidence

        return IntentClassification(
            primary_intent=primary_intent,
            confidence=confidence,
            all_scores=boosted_scores,
            matched_keywords=classification.matched_keywords,
            reasoning=classification.reasoning + "; profile_boost applied",
        )

    def _get_adapter_for_scenario(self, scenario_id: str) -> str:
        """Get the adapter version for a scenario."""
        # Check explicit mapping first
        if scenario_id in self.scenario_adapter_map:
            return self.scenario_adapter_map[scenario_id]

        # Check scenario config
        if scenario_id in self.scenarios:
            return self.scenarios[scenario_id].adapter_version

        # Default to latest
        return "latest"

    def test_route(
        self,
        text: str,
        user_profile: UserProfile | None = None,
    ) -> dict[str, Any]:
        """Test routing for a given input without caching.

        Returns detailed information about the routing decision.
        """
        strategy = getattr(self.config.router, "strategy", "keyword")
        if strategy == "hybrid":
            classification = self._hybrid_classify(text)
        else:
            classification = self.classifier.classify(text)

        if user_profile is not None:
            classification = self._apply_profile_boost(classification, user_profile)

        confidence = self.scorer.score(classification, text)

        # Get all possible routes sorted by score
        sorted_scenarios = sorted(
            classification.all_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Primary route
        primary_route = self.route(text, use_cache=False, user_profile=user_profile)

        return {
            "input": text,
            "classification": classification.to_dict(),
            "confidence": confidence,
            "primary_route": primary_route.to_dict(),
            "all_routes": [
                {
                    "scenario_id": scenario_id,
                    "score": score,
                    "adapter_version": self._get_adapter_for_scenario(scenario_id),
                }
                for scenario_id, score in sorted_scenarios
            ],
            "thresholds": {
                "min_confidence": self.scorer.min_confidence_threshold,
                "high_confidence": self.scorer.high_confidence_threshold,
            },
            "strategy": strategy,
        }

    def list_scenarios(self) -> list[dict[str, Any]]:
        """List all available scenarios with their configurations."""
        return [
            {
                "scenario_id": s.scenario_id,
                "name": s.name,
                "description": s.description,
                "adapter_version": s.adapter_version,
                "keyword_count": len(s.trigger_keywords),
                "example_count": len(s.examples),
                "priority": s.priority,
            }
            for s in self.scenarios.values()
        ]

    def clear_cache(self) -> None:
        """Clear the routing cache."""
        self._routing_cache.clear()

    def enable_cache(self) -> None:
        """Enable routing cache."""
        self._cache_enabled = True

    def disable_cache(self) -> None:
        """Disable routing cache."""
        self._cache_enabled = False


def create_router(
    config: PFEConfig | None = None,
    custom_scenarios: list[ScenarioConfig] | None = None,
) -> ScenarioRouter:
    """Factory function to create a configured router.

    Args:
        config: PFE configuration
        custom_scenarios: List of custom scenarios to add

    Returns:
        Configured ScenarioRouter instance
    """
    router = ScenarioRouter(config=config)

    if custom_scenarios:
        for scenario in custom_scenarios:
            router.add_scenario(scenario)

    return router
