"""Shared helpers for Matrix CLI output test compatibility."""
from __future__ import annotations

import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences for testing."""
    return _ANSI_RE.sub("", text)


def contains_matrix(text: str, *keywords: str) -> bool:
    """Check that all keywords appear in Matrix output (after stripping ANSI)."""
    clean = strip_ansi(text).lower()
    return all(kw.lower() in clean for kw in keywords)
