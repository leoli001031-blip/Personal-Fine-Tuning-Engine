"""Semantic conflict detection engine for PFE signal quality.

Implements P2-A signal quality deepening: detects contradictory preferences
on semantically similar questions so they are not mixed directly into training.

The engine uses pure-Python text similarity (difflib + n-gram overlap) to
avoid external embedding dependencies. It works for both Chinese and English.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any

from ..models import normalize_utc_datetime, parse_utc_datetime


def _normalize_text(value: Any) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    if not value:
        return ""
    text = str(value).strip().lower()
    # Collapse all whitespace (works for CJK and Latin)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_context(signal: dict[str, Any]) -> str:
    """Extract the 'context' (user question / instruction) from a signal dict.

    Falls back to common field names used across the codebase.
    """
    for key in ("context", "instruction", "user_input", "prompt", "query"):
        val = signal.get(key)
        if val is not None:
            return _normalize_text(val)
    return ""


def _extract_reply_style(signal: dict[str, Any]) -> str:
    """Extract normalized reply style from a signal dict."""
    # Prefer signal_quality.reply_style if present
    sq = signal.get("signal_quality") or {}
    if isinstance(sq, dict):
        style = sq.get("reply_style")
        if style:
            return str(style).strip().lower()

    # Fall back to event_type or user_action
    event_type = str(signal.get("event_type") or "").strip().lower()
    user_action = signal.get("user_action") or {}
    if isinstance(user_action, dict):
        action_style = str(user_action.get("reply_style") or user_action.get("type") or "").strip().lower()
        if action_style:
            return action_style

    mapping = {
        "accept": "accepted",
        "accepted": "accepted",
        "copy": "accepted",
        "reject": "rejected",
        "rejected": "rejected",
        "edit": "edited",
        "edited": "edited",
        "update": "edited",
    }
    return mapping.get(event_type, "other")


def _extract_final_text(signal: dict[str, Any]) -> str:
    """Extract the final text after an edit action."""
    user_action = signal.get("user_action") or {}
    if isinstance(user_action, dict):
        for key in ("final_text", "edited_text", "updated_text"):
            val = user_action.get(key)
            if val is not None:
                return _normalize_text(val)
    return _normalize_text(signal.get("model_output", ""))


def _extract_timestamp(signal: dict[str, Any]) -> datetime:
    """Extract timestamp from a signal dict, defaulting to epoch."""
    ts = signal.get("timestamp")
    if isinstance(ts, datetime):
        return normalize_utc_datetime(ts)
    if isinstance(ts, str):
        # Try ISO format first
        try:
            return parse_utc_datetime(ts)
        except Exception:
            pass
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _extract_signal_id(signal: dict[str, Any]) -> str:
    """Extract signal_id, falling back to event_id or a placeholder."""
    for key in ("signal_id", "event_id", "id"):
        val = signal.get(key)
        if val is not None:
            return str(val)
    return ""


def _extract_confidence(signal: dict[str, Any]) -> float:
    """Extract confidence from signal_quality or default to 0.5."""
    sq = signal.get("signal_quality") or {}
    if isinstance(sq, dict):
        try:
            return float(sq.get("confidence", 0.5))
        except Exception:
            pass
    return 0.5


def _ngram_set(text: str, n: int = 2) -> set[str]:
    """Build a character n-gram set from text.

    For CJK text this gives meaningful substrings; for Latin it captures
    local character patterns. Falls back to unigrams for very short text.
    """
    if len(text) <= n:
        return set(text) if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _ngram_similarity(left: str, right: str, n: int = 2) -> float:
    """Compute Jaccard-like n-gram overlap similarity in [0, 1]."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    left_set = _ngram_set(left, n)
    right_set = _ngram_set(right, n)
    if not left_set or not right_set:
        return 0.0
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection / union if union else 0.0


def _sequence_similarity(left: str, right: str) -> float:
    """Compute difflib SequenceMatcher ratio."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _context_similarity(left: str, right: str) -> float:
    """Hybrid similarity: average of SequenceMatcher and n-gram overlap.

    SequenceMatcher captures long contiguous matches (good for English),
    while n-gram overlap captures local lexical overlap (good for Chinese
    and short phrases). The average gives robust behaviour across scripts.
    """
    seq_sim = _sequence_similarity(left, right)
    ngram_sim = _ngram_similarity(left, right, n=2)
    return (seq_sim + ngram_sim) / 2.0


def _is_accept(style: str) -> bool:
    return style == "accepted"


def _is_reject(style: str) -> bool:
    return style == "rejected"


def _is_edit(style: str) -> bool:
    return style == "edited"


def _is_contradictory(style: str) -> bool:
    """A style is 'contradictory' if it is reject or edit (non-accept)."""
    return _is_reject(style) or _is_edit(style)


def _signals_conflict(
    older: dict[str, Any],
    newer: dict[str, Any],
    similarity_threshold: float,
) -> tuple[bool, str]:
    """Determine whether two signals on a similar question conflict.

    Returns (conflict_detected, reason).
    """
    older_style = _extract_reply_style(older)
    newer_style = _extract_reply_style(newer)

    # Rule 1: accept + reject/edit on same question
    if _is_accept(older_style) and _is_contradictory(newer_style):
        return True, f"accept_then_{newer_style}"

    # Rule 2: two edits going in different directions
    if _is_edit(older_style) and _is_edit(newer_style):
        older_text = _extract_final_text(older)
        newer_text = _extract_final_text(newer)
        if older_text != newer_text:
            return True, "edit_divergence"

    # Rule 3: stable_preference vs later contradictory signal
    # We interpret "stable_preference" as an accepted signal with high confidence.
    older_conf = _extract_confidence(older)
    if _is_accept(older_style) and older_conf >= 0.85 and _is_contradictory(newer_style):
        return True, "stable_preference_contradicted"

    return False, ""


@dataclass
class ConflictReport:
    """Result of conflict detection over a batch of signals."""

    conflict_detected: bool = False
    conflicting_pairs: list[tuple[str, str]] = field(default_factory=list)
    conflict_reasons: dict[str, str] = field(default_factory=dict)
    resolution_recommendation: str = "quarantine_both"


def detect_conflicts(
    signals: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.75,
    lookback_window_hours: float = 168.0,
) -> ConflictReport:
    """Detect semantic conflicts among a list of signals.

    Two signals are considered "on the same question" when their context
    similarity exceeds *similarity_threshold*. For each pair of signals
    on the same question, a set of conflict rules is evaluated (older vs
    newer, time-ordered). If a conflict is found, both signal IDs are
    marked and a resolution recommendation is produced.

    Args:
        signals: List of signal dictionaries. Each dict should contain at
            least ``context`` (or ``instruction`` / ``user_input``),
            ``timestamp``, and ``signal_id`` (or ``event_id``).
        similarity_threshold: Minimum context similarity [0, 1] to treat
            two signals as addressing the same question.
        lookback_window_hours: Only compare a newer signal against older
            signals whose timestamps fall within this window. Default is
            7 days (168 hours).

    Returns:
        A ``ConflictReport`` describing all detected conflicts.
    """
    if not signals:
        return ConflictReport()

    # Sort by timestamp ascending; stable sort preserves input order for ties
    indexed = [
        {
            "signal": s,
            "ts": _extract_timestamp(s),
            "sid": _extract_signal_id(s) or f"__idx_{i}",
            "ctx": _extract_context(s),
        }
        for i, s in enumerate(signals)
    ]
    indexed.sort(key=lambda x: x["ts"])

    conflicting_pairs: list[tuple[str, str]] = []
    conflict_reasons: dict[str, str] = {}
    window = timedelta(hours=lookback_window_hours)

    for i in range(len(indexed)):
        newer = indexed[i]
        # Scan backwards within the lookback window
        j = i - 1
        while j >= 0:
            older = indexed[j]
            if newer["ts"] - older["ts"] > window:
                break
            # Compute similarity
            sim = _context_similarity(newer["ctx"], older["ctx"])
            if sim >= similarity_threshold:
                conflict, reason = _signals_conflict(
                    older["signal"], newer["signal"], similarity_threshold
                )
                if conflict:
                    pair = (older["sid"], newer["sid"])
                    conflicting_pairs.append(pair)
                    # Mark both signals; if already marked, keep the first reason
                    if older["sid"] not in conflict_reasons:
                        conflict_reasons[older["sid"]] = reason
                    if newer["sid"] not in conflict_reasons:
                        conflict_reasons[newer["sid"]] = reason
            j -= 1

    if not conflicting_pairs:
        return ConflictReport()

    # Determine resolution recommendation
    # Default: quarantine both. If all conflicts involve a later signal
    # overriding an earlier one with lower confidence, recommend keep_newer.
    recommend = "quarantine_both"
    all_newer_higher = True
    for older_sid, newer_sid in conflicting_pairs:
        older_conf = _extract_confidence(indexed[[x["sid"] for x in indexed].index(older_sid)]["signal"])
        newer_conf = _extract_confidence(indexed[[x["sid"] for x in indexed].index(newer_sid)]["signal"])
        if newer_conf <= older_conf:
            all_newer_higher = False
            break
    if all_newer_higher:
        recommend = "keep_newer"

    return ConflictReport(
        conflict_detected=True,
        conflicting_pairs=conflicting_pairs,
        conflict_reasons=conflict_reasons,
        resolution_recommendation=recommend,
    )


def apply_conflict_detection(
    samples: list[Any],
    *,
    similarity_threshold: float = 0.75,
    lookback_window_hours: float = 168.0,
) -> list[Any]:
    """Apply conflict detection to a list of samples and mutate their signal_quality.

    This is the integration point for ``build_signal_quality`` /
    ``summarize_signal_quality_filters``. It does not change the
    signature of existing functions; instead it is called after signal
    quality has been built, to augment samples with semantic conflict
    flags.

    For each sample that carries a ``signal_quality`` dict, if the
    sample's signal is found in a conflicting pair, ``conflict`` is set
    to ``True`` and ``conflict_reason`` is updated with the semantic
    conflict reason.

    Args:
        samples: List of sample objects (dicts or dataclasses). Each
            sample is expected to expose a ``signal_quality`` field or
            metadata entry, and the underlying signal fields
            (``context``, ``timestamp``, ``event_id``, etc.).
        similarity_threshold: Context similarity threshold.
        lookback_window_hours: Lookback window for pairwise comparison.

    Returns:
        The same list of samples, with ``signal_quality`` updated in-place
        for dict samples and replaced for dataclass samples.
    """
    from dataclasses import replace

    if not samples:
        return samples

    # Convert each sample to a signal dict for the engine
    def _sample_to_signal(sample: Any) -> dict[str, Any]:
        if isinstance(sample, dict):
            return dict(sample)
        # Try common attributes
        result: dict[str, Any] = {}
        for attr in (
            "signal_id",
            "event_id",
            "context",
            "instruction",
            "user_input",
            "timestamp",
            "event_type",
            "reply_style",
            "user_action",
            "model_output",
            "signal_quality",
            "metadata",
        ):
            if hasattr(sample, attr):
                result[attr] = getattr(sample, attr)
        return result

    signals = [_sample_to_signal(s) for s in samples]
    report = detect_conflicts(
        signals,
        similarity_threshold=similarity_threshold,
        lookback_window_hours=lookback_window_hours,
    )

    if not report.conflict_detected:
        return samples

    conflict_ids = set(report.conflict_reasons.keys())

    updated_samples: list[Any] = []
    for sample in samples:
        sig = _sample_to_signal(sample)
        sid = _extract_signal_id(sig)
        if sid not in conflict_ids:
            updated_samples.append(sample)
            continue

        reason = report.conflict_reasons[sid]

        # Update signal_quality
        if isinstance(sample, dict):
            updated = dict(sample)
            sq = dict(updated.get("signal_quality", {})) if updated.get("signal_quality") else {}
            sq["conflict"] = True
            # Preserve existing conflict_reason if already set; otherwise use new reason
            existing_reason = sq.get("conflict_reason")
            if not existing_reason:
                sq["conflict_reason"] = reason
            else:
                # Append if different
                if reason not in existing_reason:
                    sq["conflict_reason"] = f"{existing_reason};{reason}"
            updated["signal_quality"] = sq
            updated_samples.append(updated)
        else:
            # Dataclass path
            sq = getattr(sample, "signal_quality", None)
            sq_dict: dict[str, Any] = dict(sq) if isinstance(sq, dict) else {}
            sq_dict["conflict"] = True
            existing_reason = sq_dict.get("conflict_reason")
            if not existing_reason:
                sq_dict["conflict_reason"] = reason
            else:
                if reason not in existing_reason:
                    sq_dict["conflict_reason"] = f"{existing_reason};{reason}"
            try:
                updated = replace(sample, signal_quality=sq_dict)
            except Exception:
                updated = sample
            updated_samples.append(updated)

    return updated_samples
