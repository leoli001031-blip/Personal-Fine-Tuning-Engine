"""Latest adapter manifest loading for backend plan snapshots."""

from __future__ import annotations

import importlib
from typing import Any


def load_latest_adapter_manifest(workspace: str | None) -> dict[str, Any] | None:
    """Read the latest adapter manifest for a workspace when the store is available."""

    try:
        module = importlib.import_module("pfe_core.adapter_store.store")
    except Exception:
        return None

    store_cls = getattr(module, "AdapterStore", None)
    if store_cls is None:
        return None
    try:
        store = store_cls(workspace=workspace or "user_default")
        latest_version = store.current_latest_version()
        if not latest_version:
            return None
        read_manifest = getattr(store, "_read_manifest", None)
        if not callable(read_manifest):
            return None
        manifest = read_manifest(latest_version)
        return manifest if isinstance(manifest, dict) else None
    except Exception:
        return None


__all__ = ["load_latest_adapter_manifest"]
