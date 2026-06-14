"""Legacy plain-text serve target status formatting."""

from __future__ import annotations

from typing import Any


def append_legacy_serve_lines(lines: list[str], mapping: dict[str, Any], *, deps: Any) -> None:
    """Append serve target resolution lines."""
    _coerce_mapping = deps.coerce_mapping
    _format_scalar = deps.format_scalar

    serve_state = _coerce_mapping(mapping.pop("serve", None))
    if serve_state is None:
        return

    adapter_resolution_state = serve_state.get("adapter_resolution_state")
    using_promoted_adapter = serve_state.get("using_promoted_adapter")
    if not adapter_resolution_state and using_promoted_adapter is None:
        return

    serve_parts = []
    if using_promoted_adapter is not None:
        serve_parts.append(f"using_promoted_adapter={_format_scalar(using_promoted_adapter)}")
    if adapter_resolution_state is not None:
        serve_parts.append(f"adapter_resolution_state={_format_scalar(adapter_resolution_state)}")
    fallback_reason = serve_state.get("fallback_reason")
    if fallback_reason:
        serve_parts.append(f"reason={_format_scalar(fallback_reason)}")
    if serve_parts:
        lines.append("serve target: " + " | ".join(serve_parts))


__all__ = ["append_legacy_serve_lines"]
