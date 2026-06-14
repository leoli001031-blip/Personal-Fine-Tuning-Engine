"""Candidate, trigger, and queue status sections for Matrix status output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .matrix_status_auto_train import append_auto_train_status_sections
from .matrix_status_candidates import append_candidate_status_sections
from .matrix_status_train_queue import append_train_queue_status_section


def append_training_control_status_sections(lines: list[str], mapping: Mapping[str, Any]) -> None:
    """Append candidate, auto-train, and train queue status boxes."""
    append_candidate_status_sections(lines, mapping)
    append_auto_train_status_sections(lines, mapping)
    append_train_queue_status_section(lines, mapping)


__all__ = ["append_training_control_status_sections"]
