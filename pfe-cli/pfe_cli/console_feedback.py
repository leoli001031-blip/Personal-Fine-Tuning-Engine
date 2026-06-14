"""Console feedback submission helper."""

from __future__ import annotations

from typing import Any


def console_submit_feedback(
    workspace: str,
    session_id: str,
    request_id: str,
    user_message: str,
    assistant_message: str,
    response_time_seconds: float,
    adapter_version: str,
    action: str,
    edited_text: str | None = None,
) -> list[dict[str, Any]]:
    """Submit feedback via ChatCollector for console chat interactions."""

    try:
        from pfe_core.collector import ChatCollector, CollectorConfig
        from pfe_core.config import PFEConfig
        from pfe_core.models import ChatInteraction

        config = PFEConfig.load()
        collector_config = config.collector if hasattr(config, "collector") else CollectorConfig()
        home = str(config.home) if hasattr(config, "home") else None

        collector = ChatCollector(
            workspace=workspace,
            config=collector_config,
            home=home,
        )

        interaction = ChatInteraction(
            session_id=session_id,
            request_id=request_id,
            user_message=user_message,
            assistant_message=assistant_message,
            adapter_version=adapter_version,
            response_time_seconds=response_time_seconds,
        )

        next_message = None
        if action == "continue":
            next_message = ""

        signals = collector.on_interaction(
            interaction=interaction,
            next_user_message=next_message,
            edited_text=edited_text,
            action=action,  # type: ignore[arg-type]
        )
        return [signal.to_dict() for signal in signals]
    except Exception:
        # Feedback collection is best-effort; don't fail the console.
        return []
