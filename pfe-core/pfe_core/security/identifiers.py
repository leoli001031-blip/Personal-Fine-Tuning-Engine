"""Safe stable identifiers for filesystem-backed user state."""

from __future__ import annotations

import hashlib
import re


_SAFE_STORAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def safe_user_storage_id(value: str | None, *, default: str = "default_user") -> str:
    raw = str(value or default).strip() or default
    if _SAFE_STORAGE_ID.fullmatch(raw) and ".." not in raw:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"user-{digest}"
