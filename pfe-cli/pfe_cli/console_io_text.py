"""Console chat, transcript, and snapshot helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .console_io_deps import ConsoleIODeps


def console_chat_text(result: Any, *, deps: ConsoleIODeps) -> str:
    mapping = deps.coerce_mapping(result)
    if not mapping:
        return deps.format_scalar(result)
    choices = list(mapping.get("choices") or [])
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if content:
                return str(content)
    return ""


def append_console_line(lines: list[dict[str, str]], *, role: str, content: str, limit: int = 10) -> None:
    text = str(content or "").strip()
    if not text:
        return
    lines.append({"role": role, "content": text})
    if len(lines) > limit:
        del lines[:-limit]


def console_snapshot_payload(
    handler: Callable[..., Any],
    *,
    workspace: str | None,
    deps: ConsoleIODeps,
) -> dict[str, Any]:
    result = handler(workspace=workspace)
    mapping = deps.coerce_mapping(result)
    return mapping if mapping is not None else {"status_result": str(result)}


__all__ = ["append_console_line", "console_chat_text", "console_snapshot_payload"]
