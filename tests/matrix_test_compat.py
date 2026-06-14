"""Shared helpers for Matrix CLI output test compatibility."""
from __future__ import annotations

from contextlib import contextmanager
import os
import re
import tempfile
from pathlib import Path
from typing import Iterator

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences for testing."""
    return _ANSI_RE.sub("", text)


def contains_matrix(text: str, *keywords: str) -> bool:
    """Check that all keywords appear in Matrix output (after stripping ANSI)."""
    clean = strip_ansi(text).lower()
    return all(kw.lower() in clean for kw in keywords)


@contextmanager
def isolated_cwd() -> Iterator[Path]:
    """Run a CLI test from a temporary working directory."""
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        os.chdir(tmp_path)
        try:
            yield tmp_path
        finally:
            os.chdir(previous)
