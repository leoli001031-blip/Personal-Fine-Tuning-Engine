"""ChatCollector for extracting implicit signals from conversations.

This module provides the ChatCollector class which processes user interactions
with an AI assistant to extract implicit feedback signals. These signals can be
used for training and fine-tuning models based on user behavior rather than
explicit feedback.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from ..models import ChatInteraction, ImplicitSignal, SignalQuality
from .config import CollectorConfig
from ..pii_detector import PIIDetector
from ..anonymizer import Anonymizer, AnonymizationConfig
from ..pii_audit import PIIAuditLog
from ..storage import resolve_home
from ..user_memory import get_user_memory_store


def _new_id() -> str:
    """Generate a new unique identifier."""
    return uuid4().hex


def _utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class ChatCollector:
    """Collects and extracts implicit signals from chat interactions.

    The collector processes ChatInteraction objects and extracts implicit
    feedback signals based on user behavior patterns such as:
    - Accept: User continues conversation without changes
    - Edit: User modifies the assistant's response
    - Reject: User deletes or dismisses the response
    - Regenerate: User requests a new generation
    """

    def __init__(self, workspace: str, config: CollectorConfig | None = None, home: str | None = None):
        """Initialize the ChatCollector.

        Args:
            workspace: The workspace identifier for storage
            config: Configuration for signal extraction (uses defaults if None)
            home: Optional PFE home directory path
        """
        self.workspace = workspace
        self.config = config or CollectorConfig()
        self._interactions: list[ChatInteraction] = []
        self._signals: list[ImplicitSignal] = []
        self.home = home
        self._user_memory = (
            get_user_memory_store(home)
            if getattr(self.config, "explicit_user_data_routing_enabled", True)
            else None
        )

        # Initialize PII detector and anonymizer
        self._pii_detector: PIIDetector | None = None
        self._anonymizer: Anonymizer | None = None
        self._pii_audit: PIIAuditLog | None = None
        if self.config.pii_detection_enabled:
            self._pii_detector = PIIDetector(sensitivity=self.config.pii_sensitivity)
            anon_config = AnonymizationConfig(
                strategy=self.config.pii_anonymization_strategy,
            )
            self._anonymizer = Anonymizer(anon_config)
            if self.config.pii_audit_enabled:
                self._pii_audit = PIIAuditLog(log_dir=resolve_home(self.home) / "audit")

    def _detect_and_anonymize_pii(
        self,
        text: str,
        source_type: str,
        source_id: str,
    ) -> tuple[str, bool]:
        """Detect and anonymize PII in text.

        Args:
            text: Text to check
            source_type: Type of source for audit log
            source_id: Source identifier for audit log

        Returns:
            Tuple of (processed_text, had_pii)
        """
        if not self._pii_detector or not self.config.pii_detection_enabled:
            return text, False

        detection_result = self._pii_detector.detect(
            text,
            min_confidence=self.config.pii_min_confidence,
        )

        if not detection_result.has_pii:
            # Log negative detection for audit trail
            if self._pii_audit:
                self._pii_audit.log_detection(
                    source_type=source_type,
                    source_id=source_id,
                    detection_result=detection_result,
                    action_taken="none",
                )
            return text, False

        # PII detected - take configured action
        action = self.config.pii_action_on_detect

        if action == "block":
            # Block signal with PII
            if self._pii_audit:
                self._pii_audit.log_detection(
                    source_type=source_type,
                    source_id=source_id,
                    detection_result=detection_result,
                    action_taken="blocked",
                )
            return text, True  # Signal caller to block

        elif action == "flag":
            # Just log, don't modify
            if self._pii_audit:
                self._pii_audit.log_detection(
                    source_type=source_type,
                    source_id=source_id,
                    detection_result=detection_result,
                    action_taken="flagged",
                )
            return text, True

        else:  # anonymize
            if self._anonymizer:
                anonymized = self._anonymizer.anonymize(text, detection_result)
                if self._pii_audit:
                    self._pii_audit.log_detection(
                        source_type=source_type,
                        source_id=source_id,
                        detection_result=detection_result,
                        action_taken="anonymized",
                    )
                return anonymized, True

        return text, True

    def _load_rules(self) -> dict[str, Any]:
        """Load signal extraction rules from configuration."""
        return self.config.signal_rules

    def on_interaction(
        self,
        interaction: ChatInteraction,
        next_user_message: str | None = None,
        edited_text: str | None = None,
        action: Literal["continue", "edit", "delete", "regenerate"] | None = None,
    ) -> list[ImplicitSignal]:
        """Process a chat interaction and extract implicit signals.

        This is the main entry point for signal extraction. It analyzes the
        interaction and any follow-up user action to determine what signals
        can be extracted.

        Args:
            interaction: The chat interaction to analyze
            next_user_message: The next message from the user (if available)
            edited_text: The edited version of assistant's message (if edited)
            action: Explicit action type if known

        Returns:
            List of extracted implicit signals
        """
        if not self.config.enabled:
            return []

        signals: list[ImplicitSignal] = []

        # Store interaction if configured
        if self.config.store_interactions:
            self._store_interaction(interaction)

        explicit_user_data_summary = self._route_explicit_user_data(interaction)

        # Determine signal type based on available information
        if edited_text is not None:
            # User edited the assistant's response
            signal = self._extract_edit_signal(interaction, edited_text)
            if signal and signal.confidence >= self.config.edit_confidence_threshold:
                self._attach_explicit_user_data_summary(signal, explicit_user_data_summary)
                signals.append(signal)

        elif action == "delete":
            # User deleted the response
            signal = self._extract_reject_signal(interaction, "delete")
            if signal and signal.confidence >= self.config.reject_confidence_threshold:
                self._attach_explicit_user_data_summary(signal, explicit_user_data_summary)
                signals.append(signal)

        elif action == "regenerate":
            # User requested regeneration
            signal = self._extract_regenerate_signal(interaction)
            if signal and signal.confidence >= self.config.regenerate_confidence_threshold:
                self._attach_explicit_user_data_summary(signal, explicit_user_data_summary)
                signals.append(signal)

        elif next_user_message is not None or action == "continue":
            # User continued the conversation - accept signal
            signal = self._extract_accept_signal(interaction, next_user_message)
            if signal and signal.confidence >= self.config.accept_confidence_threshold:
                self._attach_explicit_user_data_summary(signal, explicit_user_data_summary)
                signals.append(signal)

        # Store extracted signals (with PII handling)
        for signal in signals:
            processed_signal = self._apply_pii_handling_to_signal(signal)
            if processed_signal:  # None if blocked
                self._store_signal(processed_signal)

        return signals

    def _route_explicit_user_data(self, interaction: ChatInteraction) -> dict[str, Any]:
        """Route explicit user facts/preferences into memory/profile stores."""
        if not self._user_memory or not interaction.user_message.strip():
            return {}
        try:
            return self._user_memory.ingest_explicit_user_data(
                user_id=interaction.session_id or interaction.request_id or "default_user",
                user_message=interaction.user_message,
                assistant_message=interaction.assistant_message,
                request_id=interaction.request_id,
            )
        except Exception:
            return {}

    def _attach_explicit_user_data_summary(
        self,
        signal: ImplicitSignal,
        summary: dict[str, Any],
    ) -> None:
        """Attach explicit user data routing summary to signal metadata."""
        if not summary or not summary.get("candidates"):
            return
        signal.metadata = dict(signal.metadata or {})
        signal.metadata["explicit_user_data_routing"] = {
            "processed": bool(summary.get("processed", False)),
            "reason": summary.get("reason"),
            "candidate_count": len(summary.get("candidates", [])),
            "stored_preference_count": len(summary.get("stored_preferences", [])),
            "new_fact_count": len(summary.get("new_facts", [])),
            "blocked_count": len(summary.get("blocked_candidates", [])),
            "candidates": list(summary.get("candidates", [])),
        }

    def _store_interaction(self, interaction: ChatInteraction) -> None:
        """Store interaction in memory history."""
        self._interactions.append(interaction)
        # Trim if exceeds max
        if len(self._interactions) > self.config.max_interaction_history:
            self._interactions = self._interactions[-self.config.max_interaction_history :]

    def _apply_pii_handling_to_signal(
        self,
        signal: ImplicitSignal,
    ) -> ImplicitSignal | None:
        """Apply PII detection and anonymization to a signal.

        Args:
            signal: The signal to process

        Returns:
            Processed signal, or None if blocked due to PII
        """
        if not self.config.pii_detection_enabled:
            return signal

        # Check context (user message)
        if signal.context:
            processed_context, had_pii_context = self._detect_and_anonymize_pii(
                signal.context,
                source_type="signal",
                source_id=signal.signal_id,
            )
            if had_pii_context and self.config.pii_action_on_detect == "block":
                return None  # Block this signal
            signal.context = processed_context

        # Check model output (assistant message)
        if signal.model_output:
            processed_output, had_pii_output = self._detect_and_anonymize_pii(
                signal.model_output,
                source_type="signal",
                source_id=signal.signal_id,
            )
            if had_pii_output and self.config.pii_action_on_detect == "block":
                return None  # Block this signal
            signal.model_output = processed_output

        # Check edited text if present
        if signal.user_action and "edited_text" in signal.user_action:
            edited_text = signal.user_action["edited_text"]
            if isinstance(edited_text, str):
                processed_edit, had_pii_edit = self._detect_and_anonymize_pii(
                    edited_text,
                    source_type="signal",
                    source_id=signal.signal_id,
                )
                if had_pii_edit and self.config.pii_action_on_detect == "block":
                    return None
                signal.user_action["edited_text"] = processed_edit

        return signal

    def _store_signal(self, signal: ImplicitSignal) -> None:
        """Store signal in memory and database."""
        self._signals.append(signal)
        # Run contradiction detection after storing
        if self.config.contradiction_detection_enabled:
            self._detect_and_mark_contradictions(signal)
        # Store to database using storage module
        try:
            from ..storage import record_signal
            record_signal(signal.to_dict(), home=self.home)
        except Exception:
            # Log but don't fail - signal is still in memory
            pass

    def _get_session_signals(
        self,
        session_id: str,
        within_seconds: float | None = None,
    ) -> list[ImplicitSignal]:
        """Get signals for a session, optionally within a time window."""
        results = []
        cutoff = None
        if within_seconds is not None:
            cutoff = _utc_now().timestamp() - within_seconds
        for s in self._signals:
            if s.session_id != session_id:
                continue
            if cutoff is not None:
                ts = s.timestamp.timestamp() if isinstance(s.timestamp, datetime) else 0.0
                if ts < cutoff:
                    continue
            results.append(s)
        return results

    def _detect_and_mark_contradictions(self, new_signal: ImplicitSignal) -> None:
        """Detect contradictory signals in the same session and mark them.

        Contradictions include:
        - Accept vs Reject for the same request_id
        - Accept vs Regenerate for the same request_id
        - Accept followed by Edit for the same request_id (accept is downgraded)
        """
        session_signals = self._get_session_signals(
            new_signal.session_id,
            within_seconds=self.config.contradiction_window_seconds,
        )

        for other in session_signals:
            if other.signal_id == new_signal.signal_id:
                continue

            # Only check signals for the same request
            if other.request_id != new_signal.request_id:
                continue

            pair = {new_signal.signal_type, other.signal_type}
            conflict = False
            conflict_reason: str | None = None

            # Accept + Reject is a direct contradiction
            if pair == {"accept", "reject"}:
                conflict = True
                conflict_reason = "contradictory_accept_reject"
            # Accept + Regenerate is a contradiction
            elif pair == {"accept", "regenerate"}:
                conflict = True
                conflict_reason = "contradictory_accept_regenerate"
            # Accept + Edit indicates the accept was premature
            elif pair == {"accept", "edit"}:
                conflict = True
                conflict_reason = "contradictory_accept_then_edit"
            # Reject + Edit is contradictory about what the user wanted
            elif pair == {"reject", "edit"}:
                conflict = True
                conflict_reason = "contradictory_reject_then_edit"

            if conflict:
                for target in (new_signal, other):
                    if target.signal_quality is None:
                        target.signal_quality = SignalQuality()
                    target.signal_quality.conflict = True
                    target.signal_quality.conflict_reason = conflict_reason
                    # Reduce confidence for conflicted signals
                    target.confidence = round(target.confidence * 0.7, 2)
                    target.signal_quality.confidence = target.confidence

    def get_replay_candidates(
        self,
        limit: int = 100,
    ) -> list[ImplicitSignal]:
        """Get signals eligible for replay/rollback due to low quality or conflict.

        Returns signals that:
        - Have a quality conflict, OR
        - Have confidence below the replay threshold but above the minimum
          (meaning they were stored but are marginal quality)
        """
        candidates = []
        for signal in self._signals:
            if signal.signal_quality and signal.signal_quality.conflict:
                candidates.append(signal)
                continue
            if signal.confidence < self.config.replay_minimum_confidence:
                continue
            if signal.confidence < self.config.replay_confidence_threshold:
                candidates.append(signal)
                continue
            if len(candidates) >= limit:
                break
        return candidates

    def rollback_signal(self, signal_id: str) -> ImplicitSignal | None:
        """Remove a signal by ID and return it for replay.

        This allows low-quality or conflicted signals to be removed from the
        active pool and potentially re-evaluated or corrected.
        """
        for i, signal in enumerate(self._signals):
            if signal.signal_id == signal_id:
                removed = self._signals.pop(i)
                removed.signal_quality = removed.signal_quality or SignalQuality()
                removed.signal_quality.rolled_back = True
                removed.signal_quality.confidence_reason = (
                    removed.signal_quality.confidence_reason or ""
                ) + " | rolled_back_for_replay"
                return removed
        return None

    def get_contradiction_summary(self) -> dict[str, Any]:
        """Get summary of detected contradictions for observability."""
        conflicted = [s for s in self._signals if s.signal_quality and s.signal_quality.conflict]
        reasons: dict[str, int] = {}
        for s in conflicted:
            reason = str(s.signal_quality.conflict_reason or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "total_signals": len(self._signals),
            "conflicted_signals": len(conflicted),
            "conflict_reasons": reasons,
            "replay_candidates": len(self.get_replay_candidates()),
        }

    def _extract_accept_signal(
        self,
        interaction: ChatInteraction,
        next_message: str | None = None,
    ) -> ImplicitSignal | None:
        """Extract an accept signal from the interaction.

        Accept signals indicate the user was satisfied with the response.
        Confidence varies based on response time:
        - Quick response (< 5s): Strong accept (0.9)
        - Normal response: Standard accept (0.7)
        - Slow response (> 60s): Weak accept (0.4)
        """
        rules = self._load_rules().get("accept", {})
        base_confidence = rules.get("base_confidence", 0.7)

        response_time = interaction.response_time_seconds
        if response_time is None:
            # No timing info - use base confidence
            confidence = base_confidence
            rule_name = "accept_no_timing"
        elif response_time < self.config.strong_accept_threshold_seconds:
            # Quick response - strong accept
            multiplier = rules.get("strong_multiplier", 1.29)
            confidence = min(1.0, base_confidence * multiplier)
            rule_name = "accept_quick_response"
        elif response_time > self.config.weak_accept_threshold_seconds:
            # Slow response - weak accept
            multiplier = rules.get("weak_multiplier", 0.57)
            confidence = base_confidence * multiplier
            rule_name = "accept_slow_response"
        else:
            # Normal response - standard accept
            confidence = base_confidence
            rule_name = "accept_normal"

        return ImplicitSignal(
            signal_id=_new_id(),
            source_event_id=interaction.event_id,
            request_id=interaction.request_id,
            session_id=interaction.session_id,
            adapter_version=interaction.adapter_version,
            event_type="accept",
            timestamp=_utc_now(),
            context=interaction.user_message,
            model_output=interaction.assistant_message,
            user_action={"action": "accept", "next_message": next_message},
            signal_type="accept",
            confidence=round(confidence, 2),
            extraction_rule=rule_name,
            response_time_seconds=response_time,
            signal_quality=SignalQuality(
                reply_style="accepted",
                confidence=round(confidence, 2),
                confidence_reason=f"Accept signal via {rule_name}",
            ),
        )

    def _extract_edit_signal(
        self,
        interaction: ChatInteraction,
        edited_text: str,
    ) -> ImplicitSignal | None:
        """Extract an edit signal from the interaction.

        Edit signals indicate the user was partially satisfied but made
        corrections. The edit distance determines the confidence level:
        - Distance < 20%: Slight negative (0.6)
        - Distance 20-50%: Moderate negative (0.8)
        - Distance > 50%: Strong negative (0.9)
        """
        original = interaction.assistant_message
        edit_distance = self._calculate_edit_distance(original, edited_text)
        max_len = max(len(original), len(edited_text))
        ratio = edit_distance / max_len if max_len > 0 else 0.0

        rules = self._load_rules().get("edit", {})

        # Determine confidence based on edit ratio
        if ratio < rules.get("slight_threshold", 0.2):
            confidence = rules.get("slight_confidence", 0.6)
            rule_name = "edit_slight"
        elif ratio < rules.get("moderate_threshold", 0.5):
            confidence = rules.get("moderate_confidence", 0.8)
            rule_name = "edit_moderate"
        else:
            confidence = rules.get("strong_confidence", 0.9)
            rule_name = "edit_strong"

        return ImplicitSignal(
            signal_id=_new_id(),
            source_event_id=interaction.event_id,
            request_id=interaction.request_id,
            session_id=interaction.session_id,
            adapter_version=interaction.adapter_version,
            event_type="edit",
            timestamp=_utc_now(),
            context=interaction.user_message,
            model_output=original,
            user_action={"action": "edit", "edited_text": edited_text},
            signal_type="edit",
            confidence=round(confidence, 2),
            edit_distance=edit_distance,
            edit_distance_ratio=round(ratio, 2),
            extraction_rule=rule_name,
            signal_quality=SignalQuality(
                reply_style="edited",
                confidence=round(confidence, 2),
                confidence_reason=f"Edit signal with {ratio:.0%} change via {rule_name}",
            ),
        )

    def _extract_reject_signal(
        self,
        interaction: ChatInteraction,
        reject_type: Literal["delete", "dismiss"],
    ) -> ImplicitSignal | None:
        """Extract a reject signal from the interaction.

        Reject signals indicate strong dissatisfaction with the response.
        These have high confidence (0.95) as the action is unambiguous.
        """
        rules = self._load_rules().get("reject", {})
        confidence = rules.get("base_confidence", 0.95)

        return ImplicitSignal(
            signal_id=_new_id(),
            source_event_id=interaction.event_id,
            request_id=interaction.request_id,
            session_id=interaction.session_id,
            adapter_version=interaction.adapter_version,
            event_type="reject",
            timestamp=_utc_now(),
            context=interaction.user_message,
            model_output=interaction.assistant_message,
            user_action={"action": reject_type},
            signal_type="reject",
            confidence=round(confidence, 2),
            extraction_rule=f"reject_{reject_type}",
            signal_quality=SignalQuality(
                reply_style="rejected",
                confidence=round(confidence, 2),
                confidence_reason=f"Reject signal via {reject_type}",
            ),
        )

    def _extract_regenerate_signal(
        self,
        interaction: ChatInteraction,
    ) -> ImplicitSignal | None:
        """Extract a regenerate signal from the interaction.

        Regenerate signals indicate the user wants a different response
        but didn't want to edit manually. Confidence is high (0.85).
        """
        rules = self._load_rules().get("regenerate", {})
        confidence = rules.get("base_confidence", 0.85)

        return ImplicitSignal(
            signal_id=_new_id(),
            source_event_id=interaction.event_id,
            request_id=interaction.request_id,
            session_id=interaction.session_id,
            adapter_version=interaction.adapter_version,
            event_type="regenerate",
            timestamp=_utc_now(),
            context=interaction.user_message,
            model_output=interaction.assistant_message,
            user_action={"action": "regenerate"},
            signal_type="regenerate",
            confidence=round(confidence, 2),
            extraction_rule="regenerate_explicit",
            signal_quality=SignalQuality(
                reply_style="rejected",
                confidence=round(confidence, 2),
                confidence_reason="Regenerate signal - user requested new generation",
            ),
        )

    def _calculate_edit_distance(self, original: str, edited: str) -> int:
        """Calculate the edit distance between two strings.

        Uses Levenshtein distance by default, but can be configured to use
        Jaro-Winkler distance if specified in config.
        """
        if self.config.edit_distance_metric == "jaro_winkler":
            return self._jaro_winkler_distance(original, edited)
        return self._levenshtein_distance(original, edited)

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings.

        This is the minimum number of single-character edits (insertions,
        deletions, or substitutions) required to change one string into another.
        """
        if len(s1) < len(s2):
            return ChatCollector._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        # Use two rows instead of full matrix for memory efficiency
        previous_row = list(range(len(s2) + 1))
        current_row = [0] * (len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row[0] = i + 1

            for j, c2 in enumerate(s2):
                # Cost is 0 if characters match, 1 if different
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (0 if c1 == c2 else 1)
                current_row[j + 1] = min(insertions, deletions, substitutions)

            # Swap rows
            previous_row, current_row = current_row, previous_row

        return previous_row[len(s2)]

    @staticmethod
    def _jaro_winkler_distance(s1: str, s2: str) -> int:
        """Calculate Jaro-Winkler distance and convert to an integer edit distance.

        Jaro-Winkler is better for short strings and transpositions.
        We convert the similarity score to an approximate edit distance.
        """
        # Jaro similarity
        len1, len2 = len(s1), len(s2)
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1

        # Match window
        match_distance = max(len1, len2) // 2 - 1

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        # Find matches
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j]:
                    continue
                if s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return max(len1, len2)

        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        # Jaro similarity
        jaro = (
            matches / len1 +
            matches / len2 +
            (matches - transpositions / 2) / matches
        ) / 3

        # Convert similarity to approximate edit distance
        max_dist = max(len1, len2)
        return int((1 - jaro) * max_dist)

    def get_stats(self) -> dict[str, Any]:
        """Get collector statistics."""
        return {
            "total_interactions": len(self._interactions),
            "total_signals": len(self._signals),
            "signals_by_type": self._count_signals_by_type(),
            "config": {
                "enabled": self.config.enabled,
                "accept_threshold": self.config.accept_confidence_threshold,
                "edit_threshold": self.config.edit_confidence_threshold,
                "pii_detection": {
                    "enabled": self.config.pii_detection_enabled,
                    "sensitivity": self.config.pii_sensitivity,
                    "strategy": self.config.pii_anonymization_strategy,
                    "action": self.config.pii_action_on_detect,
                },
            },
        }

    def _count_signals_by_type(self) -> dict[str, int]:
        """Count signals by type."""
        counts: dict[str, int] = {}
        for signal in self._signals:
            counts[signal.signal_type] = counts.get(signal.signal_type, 0) + 1
        return counts

    def get_signals_for_review(
        self,
        signal_type: str | None = None,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        limit: int = 100,
    ) -> list[ImplicitSignal]:
        """Get signals for manual review.

        Args:
            signal_type: Filter by signal type
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            limit: Maximum number of signals to return

        Returns:
            List of signals matching the criteria
        """
        results = []
        for signal in self._signals:
            if signal_type and signal.signal_type != signal_type:
                continue
            if not (min_confidence <= signal.confidence <= max_confidence):
                continue
            results.append(signal)
            if len(results) >= limit:
                break
        return results
