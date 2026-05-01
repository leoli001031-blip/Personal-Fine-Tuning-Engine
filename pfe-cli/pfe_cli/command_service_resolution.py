"""Service and optional helper resolution for CLI commands."""

from __future__ import annotations

import importlib
from typing import Any


def load_service(*module_names: str) -> Any | None:
    """Resolve a future high-level service object if the mainline has provided it."""

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for attr_name in (
            "service",
            "pipeline",
            "trainer",
            "evaluator",
            "server",
            "inference",
            "generate",
            "distill",
            "train",
            "evaluate",
            "serve",
            "status",
        ):
            candidate = getattr(module, attr_name, None)
            if candidate is not None:
                return candidate
    return None


def optional_module_call(module_name: str, attr_name: str, *args: Any, **kwargs: Any) -> Any | None:
    """Call a helper from an optional module without hard-failing CLI formatting."""

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    candidate = getattr(module, attr_name, None)
    if not callable(candidate):
        return None
    try:
        return candidate(*args, **kwargs)
    except Exception:
        return None


def resolve_handler(service: Any, *names: str) -> Any | None:
    for name in names:
        candidate = getattr(service, name, None)
        if candidate is not None:
            return candidate
    if callable(service):
        return service
    return None


__all__ = ["load_service", "optional_module_call", "resolve_handler"]
