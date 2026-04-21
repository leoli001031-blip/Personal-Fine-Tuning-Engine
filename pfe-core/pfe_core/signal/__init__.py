"""Signal processing utilities for PFE."""

from __future__ import annotations

from .conflict_detector import (
    ConflictReport,
    detect_conflicts,
    apply_conflict_detection,
)

__all__ = [
    "ConflictReport",
    "detect_conflicts",
    "apply_conflict_detection",
]
