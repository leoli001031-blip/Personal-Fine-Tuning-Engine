"""Background chat worker startup for the interactive runtime console."""

from __future__ import annotations

import threading
from typing import Any


def start_chat_worker(
    *,
    chat_handler: Any,
    chat_messages: list[dict[str, str]],
    model: str,
    adapter: str,
    temperature: float,
    effective_max_tokens: int,
    real_local: bool,
    session_id: str,
    workspace: str | None,
) -> tuple[threading.Thread, dict[str, Any]]:
    response_holder: dict[str, Any] = {}

    def _run_chat() -> None:
        try:
            response_holder["result"] = chat_handler(
                messages=chat_messages,
                model=model,
                adapter_version=adapter,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                metadata={"enable_real_local": True} if real_local else {},
                session_id=session_id,
                workspace=workspace,
            )
        except Exception as exc:  # pragma: no cover - surfaced in main thread
            response_holder["error"] = exc

    worker = threading.Thread(target=_run_chat, daemon=True)
    worker.start()
    return worker, response_holder


__all__ = ["start_chat_worker"]
