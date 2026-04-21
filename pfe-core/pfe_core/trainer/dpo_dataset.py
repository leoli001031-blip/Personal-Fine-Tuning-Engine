"""DPO Dataset Builder for PFE.

Builds preference pairs from signals for Direct Preference Optimization training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

from ..db.sqlite import list_signals, signals_db_path, initialize_database
from ..curator.datasets import build_signal_quality, normalize_reply_style


@dataclass
class DPOPair:
    """DPO preference pair for training.

    Attributes:
        prompt: The input prompt/context
        chosen: The preferred/rejected response (user accepted)
        rejected: The non-preferred response (user rejected or edited)
        session_id: Session identifier for grouping
        source_event_ids: List of event IDs that form the chain
        confidence: Signal quality confidence score
        metadata: Additional metadata about the pair
    """

    prompt: str
    chosen: str
    rejected: str
    session_id: str
    source_event_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for HuggingFace Dataset."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "session_id": self.session_id,
            "source_event_ids": self.source_event_ids,
            "confidence": self.confidence,
            **self.metadata,
        }


class DPODatasetBuilder:
    """Build DPO datasets from PFE signals.

    Queries signals database for accepted/rejected/edited events and creates
    preference pairs suitable for DPO training with trl.DPOTrainer.
    """

    def __init__(self, workspace: Optional[str] = None, home: Optional[str] = None):
        """Initialize the DPO dataset builder.

        Args:
            workspace: Workspace name for adapter organization
            home: PFE home directory path
        """
        self.workspace = workspace or "user_default"
        self.home = home
        self._signals: Optional[List[Dict[str, Any]]] = None

    def build_from_signals(
        self,
        since: Optional[datetime] = None,
        min_confidence: float = 0.7,
        limit: Optional[int] = None,
    ) -> "Dataset":
        """Build DPO dataset from signals.

        Args:
            since: Only include signals after this timestamp
            min_confidence: Minimum signal quality confidence
            limit: Maximum number of pairs to generate

        Returns:
            HuggingFace Dataset with prompt, chosen, rejected columns
        """
        from datasets import Dataset

        signals = self._fetch_signals(since)
        pairs = self._create_pairs(signals, min_confidence)

        if limit is not None:
            pairs = pairs[:limit]

        if not pairs:
            # Return empty dataset with correct schema
            return Dataset.from_dict({
                "prompt": [],
                "chosen": [],
                "rejected": [],
                "session_id": [],
                "source_event_ids": [],
                "confidence": [],
            })

        data = {
            "prompt": [p.prompt for p in pairs],
            "chosen": [p.chosen for p in pairs],
            "rejected": [p.rejected for p in pairs],
            "session_id": [p.session_id for p in pairs],
            "source_event_ids": [p.source_event_ids for p in pairs],
            "confidence": [p.confidence for p in pairs],
        }

        return Dataset.from_dict(data)

    def _fetch_signals(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Fetch signals from database.

        Args:
            since: Filter signals after this timestamp

        Returns:
            List of signal dictionaries
        """
        signals = list_signals(self.home)

        if since is not None:
            since_str = since.isoformat()
            signals = [s for s in signals if s.get("timestamp", "") >= since_str]

        return signals

    def _create_pairs(
        self, signals: List[Dict[str, Any]], min_confidence: float
    ) -> List[DPOPair]:
        """Create DPO preference pairs from signals.

        Pairs are created by:
        1. Grouping signals by session_id
        2. Finding accepted vs rejected pairs
        3. Finding accepted vs edited pairs (edited becomes rejected)

        Args:
            signals: List of signal dictionaries
            min_confidence: Minimum confidence threshold

        Returns:
            List of DPOPair objects
        """
        pairs: List[DPOPair] = []

        # Group signals by session
        sessions: Dict[str, List[Dict[str, Any]]] = {}
        for signal in signals:
            session_id = signal.get("session_id", "")
            if not session_id:
                continue
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(signal)

        # Create pairs from each session
        for session_id, session_signals in sessions.items():
            session_pairs = self._create_pairs_from_session(
                session_id, session_signals, min_confidence
            )
            pairs.extend(session_pairs)

        # Sort by confidence descending
        pairs.sort(key=lambda p: p.confidence, reverse=True)

        return pairs

    def _create_pairs_from_session(
        self,
        session_id: str,
        signals: List[Dict[str, Any]],
        min_confidence: float,
    ) -> List[DPOPair]:
        """Create pairs from signals within a single session.

        Args:
            session_id: The session identifier
            signals: List of signals in this session
            min_confidence: Minimum confidence threshold

        Returns:
            List of DPOPair objects
        """
        pairs: List[DPOPair] = []

        # Categorize signals by reply style
        accepted_signals: List[Dict[str, Any]] = []
        rejected_signals: List[Dict[str, Any]] = []
        edited_signals: List[Dict[str, Any]] = []

        for signal in signals:
            # Build signal quality to get reply style and confidence
            quality = build_signal_quality(signal)

            if quality.confidence < min_confidence:
                continue

            reply_style = quality.reply_style

            if reply_style == "accepted":
                accepted_signals.append(signal)
            elif reply_style == "rejected":
                rejected_signals.append(signal)
            elif reply_style == "edited":
                edited_signals.append(signal)

        # Create pairs: accepted vs rejected
        for accepted in accepted_signals:
            for rejected in rejected_signals:
                pair = self._create_pair_from_signals(
                    session_id, accepted, rejected, "accepted_vs_rejected"
                )
                if pair:
                    pairs.append(pair)

        # Create pairs: accepted vs edited (edited becomes rejected)
        for accepted in accepted_signals:
            for edited in edited_signals:
                pair = self._create_pair_from_signals(
                    session_id, accepted, edited, "accepted_vs_edited"
                )
                if pair:
                    pairs.append(pair)

        return pairs

    def _create_pair_from_signals(
        self,
        session_id: str,
        chosen_signal: Dict[str, Any],
        rejected_signal: Dict[str, Any],
        pair_type: str,
    ) -> Optional[DPOPair]:
        """Create a single DPO pair from two signals.

        Args:
            session_id: The session identifier
            chosen_signal: The preferred signal (accepted)
            rejected_signal: The non-preferred signal (rejected or edited)
            pair_type: Type of pair being created

        Returns:
            DPOPair or None if signals are incompatible
        """
        # Extract prompt (context)
        prompt = chosen_signal.get("context") or chosen_signal.get("user_input", "")
        if not prompt:
            return None

        # Extract chosen text (accepted response)
        chosen_text = self._extract_response_text(chosen_signal, "accepted")
        if not chosen_text:
            return None

        # Extract rejected text
        rejected_text = self._extract_response_text(rejected_signal, "rejected")
        if not rejected_text:
            # For edited signals, use the edited text as rejected
            rejected_text = self._extract_response_text(rejected_signal, "edited")

        if not rejected_text:
            return None

        # Skip if texts are identical
        if chosen_text.strip() == rejected_text.strip():
            return None

        # Build source event IDs
        source_event_ids = list(set(
            (chosen_signal.get("source_event_ids") or []) +
            (rejected_signal.get("source_event_ids") or [])
        ))

        # Calculate confidence as average of both signals
        chosen_quality = build_signal_quality(chosen_signal)
        rejected_quality = build_signal_quality(rejected_signal)
        confidence = (chosen_quality.confidence + rejected_quality.confidence) / 2

        return DPOPair(
            prompt=prompt,
            chosen=chosen_text,
            rejected=rejected_text,
            session_id=session_id,
            source_event_ids=source_event_ids,
            confidence=confidence,
            metadata={
                "pair_type": pair_type,
                "chosen_event_id": chosen_signal.get("signal_id"),
                "rejected_event_id": rejected_signal.get("signal_id"),
                "chosen_reply_style": chosen_quality.reply_style,
                "rejected_reply_style": rejected_quality.reply_style,
            },
        )

    def _extract_response_text(
        self, signal: Dict[str, Any], style: str
    ) -> Optional[str]:
        """Extract response text from signal based on reply style.

        Args:
            signal: Signal dictionary
            style: Expected reply style

        Returns:
            Response text or None
        """
        user_action = signal.get("user_action") or {}

        if style == "accepted":
            # For accepted: use accepted_text or model_output
            text = user_action.get("accepted_text") or signal.get("model_output")
        elif style == "rejected":
            # For rejected: use rejected_text
            text = user_action.get("rejected_text") or user_action.get("disliked_text")
        elif style == "edited":
            # For edited: use final_text or edited_text
            text = user_action.get("final_text") or user_action.get("edited_text")
        else:
            text = None

        return text.strip() if text else None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about available signals for DPO.

        Returns:
            Dictionary with signal counts and pair statistics
        """
        signals = self._fetch_signals()

        stats = {
            "total_signals": len(signals),
            "sessions": len(set(s.get("session_id", "") for s in signals)),
            "reply_styles": {},
            "confidence_distribution": {
                "high": 0,  # >= 0.8
                "medium": 0,  # 0.6 - 0.8
                "low": 0,  # < 0.6
            },
        }

        for signal in signals:
            quality = build_signal_quality(signal)
            style = quality.reply_style

            stats["reply_styles"][style] = stats["reply_styles"].get(style, 0) + 1

            if quality.confidence >= 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif quality.confidence >= 0.6:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1

        # Estimate potential pairs
        accepted = stats["reply_styles"].get("accepted", 0)
        rejected = stats["reply_styles"].get("rejected", 0)
        edited = stats["reply_styles"].get("edited", 0)

        # Rough estimate: each accepted can pair with each rejected/edited in same session
        stats["estimated_pairs"] = accepted * (rejected + edited)

        return stats


def build_dpo_dataset_from_samples(
    samples: List[Dict[str, Any]],
    min_confidence: float = 0.7,
) -> "Dataset":
    """Build DPO dataset from existing training samples.

    This is useful when samples have already been curated and stored.

    Args:
        samples: List of training sample dictionaries
        min_confidence: Minimum confidence threshold

    Returns:
        HuggingFace Dataset with DPO format
    """
    from datasets import Dataset

    pairs: List[Dict[str, Any]] = []

    for sample in samples:
        # Only use DPO-type samples
        if sample.get("sample_type") != "dpo":
            continue

        # Check confidence if available
        metadata = sample.get("metadata", {})
        signal_quality = metadata.get("signal_quality", {})
        confidence = signal_quality.get("confidence", 1.0)

        if confidence < min_confidence:
            continue

        instruction = sample.get("instruction", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected")

        if not instruction or not chosen or not rejected:
            continue

        pairs.append({
            "prompt": instruction,
            "chosen": chosen,
            "rejected": rejected,
            "session_id": sample.get("source_adapter_version", "unknown"),
            "source_event_ids": sample.get("source_event_ids", []),
            "confidence": confidence,
            "sample_id": sample.get("sample_id"),
        })

    if not pairs:
        return Dataset.from_dict({
            "prompt": [],
            "chosen": [],
            "rejected": [],
            "session_id": [],
            "source_event_ids": [],
            "confidence": [],
        })

    return Dataset.from_list(pairs)
