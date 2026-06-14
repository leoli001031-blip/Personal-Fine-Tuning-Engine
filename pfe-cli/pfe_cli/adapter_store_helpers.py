"""Adapter store resolution and dispatch helpers."""

from __future__ import annotations

import importlib
from typing import Any

import typer


def _load_adapter_store() -> Any | None:
    """Resolve the future high-level adapter store service if it exists."""

    for module_name in (
        "pfe_core.adapter_store.store",
        "pfe_core.services.adapter_store",
        "pfe_core.pipeline.adapter_store",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for attr_name in ("AdapterStore", "get_adapter_store", "create_adapter_store"):
            candidate = getattr(module, attr_name, None)
            if candidate is not None:
                return candidate
    return None


def _call_store(method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Dispatch to the future adapter store interface when it is available."""

    store_factory = _load_adapter_store()
    if store_factory is None:
        typer.echo(
            f"[pfe] adapter {method_name}: CLI skeleton is wired, but the adapter store backend is not connected yet."
        )
        return None

    store = store_factory() if callable(store_factory) else store_factory
    method = getattr(store, method_name, None)
    if method is None:
        raise typer.BadParameter(
            f"Connected adapter store does not provide '{method_name}'. "
            "Expected a future high-level store interface from pfe_core."
        )
    try:
        return method(*args, **kwargs)
    except typer.Exit:
        raise
    except Exception as exc:
        name = exc.__class__.__name__.lower()
        if "adaptererror" in name:
            typer.secho(f"Adapter error: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        if "trainingerror" in name:
            typer.secho(f"Training error: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        raise


__all__ = ["_call_store", "_load_adapter_store"]
