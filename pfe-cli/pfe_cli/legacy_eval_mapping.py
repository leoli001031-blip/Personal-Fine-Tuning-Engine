"""Input coercion for legacy eval result formatting."""

from __future__ import annotations

import json
from typing import Any

from .legacy_result_deps import LegacyResultFormattingDeps


def coerce_eval_result_mapping(result: Any, *, deps: LegacyResultFormattingDeps) -> dict[str, Any] | None:
    mapping = deps.coerce_mapping(result)
    if mapping is not None:
        return mapping
    if not isinstance(result, str):
        return None
    try:
        loaded = json.loads(result)
    except Exception:
        return None
    return deps.coerce_mapping(loaded)


__all__ = ["coerce_eval_result_mapping"]
