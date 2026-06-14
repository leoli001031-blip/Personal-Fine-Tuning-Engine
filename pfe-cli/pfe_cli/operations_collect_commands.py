"""Signal collection command registration."""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

import typer

from .operations_command_deps import OperationsCommandDeps, run_simple_status_command


def _collector_for_workspace(workspace: str | None) -> Any:
    from pfe_core.collector import ChatCollector, CollectorConfig
    from pfe_core.config import PFEConfig

    config = PFEConfig.load()
    collector_config = config.collector if hasattr(config, "collector") else CollectorConfig()
    home = str(config.home) if hasattr(config, "home") else None

    return ChatCollector(
        workspace=workspace or "user_default",
        config=collector_config,
        home=home,
    )


def _config_home() -> str | None:
    from pfe_core.config import PFEConfig

    config = PFEConfig.load()
    return str(config.home) if hasattr(config, "home") else None


def _signal_type(record: dict[str, Any]) -> str:
    metadata = dict(record.get("metadata") or {})
    user_action = record.get("user_action")
    if isinstance(user_action, dict) and user_action.get("type"):
        return str(user_action["type"])
    return str(metadata.get("signal_type") or record.get("event_type") or "unknown")


def _signal_confidence(record: dict[str, Any]) -> float:
    metadata = dict(record.get("metadata") or {})
    try:
        return float(metadata.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _signals_by_type(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        signal_type = _signal_type(record)
        counts[signal_type] = counts.get(signal_type, 0) + 1
    return counts


def _filtered_persisted_signals(
    *,
    home: str | None,
    signal_type: str | None,
    min_confidence: float,
    max_confidence: float,
    limit: int,
) -> list[dict[str, Any]]:
    from pfe_core.storage import list_signals

    results: list[dict[str, Any]] = []
    for record in list_signals(home=home):
        current_type = _signal_type(record)
        confidence = _signal_confidence(record)
        if signal_type and current_type != signal_type:
            continue
        if not (min_confidence <= confidence <= max_confidence):
            continue
        results.append(record)
        if len(results) >= limit:
            break
    return results


def _cli_generated_id(prefix: str) -> str:
    return f"{prefix}-cli-{uuid4().hex[:10]}"


def _action_payload(
    *,
    action: str,
    model_output: str,
    edited_text: str | None,
    rejected_text: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": action}
    if action == "accept":
        payload["accepted_text"] = model_output
    elif action == "edit":
        final_text = edited_text or model_output
        payload["edited_text"] = final_text
        payload["final_text"] = final_text
        payload["accepted_text"] = final_text
        payload["rejected_text"] = model_output
    elif action in {"reject", "regenerate"}:
        payload["rejected_text"] = rejected_text or model_output
    return payload


def _default_signal_confidence(action: str) -> float:
    return {
        "accept": 0.9,
        "edit": 0.85,
        "reject": 0.72,
        "regenerate": 0.72,
    }.get(action, 0.7)


def register_collect_commands(*, collect_app: typer.Typer, deps: OperationsCommandDeps) -> None:
    """Attach signal collection commands to the collect sub-app."""

    @collect_app.command("start")
    def collect_start(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Enable signal collection for the current workspace."""

        run_simple_status_command(
            deps,
            command_name="collect start",
            handler_name="start_signal_collection",
            workspace=workspace,
        )

    @collect_app.command("stop")
    def collect_stop(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Disable signal collection for the current workspace."""

        run_simple_status_command(
            deps,
            command_name="collect stop",
            handler_name="stop_signal_collection",
            workspace=workspace,
        )

    @collect_app.command("status")
    def collect_status(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show signal collection statistics."""

        home = _config_home()
        collector = _collector_for_workspace(workspace)
        stats = collector.get_stats()
        from pfe_core.storage import list_signals, status_snapshot

        persisted_signals = list_signals(home=home)
        persisted_counts = _signals_by_type(persisted_signals)
        snapshot = status_snapshot(home=home, workspace=workspace or "user_default")
        signal_summary = dict(snapshot.get("signal_summary") or {})
        sample_counts = dict(snapshot.get("signal_sample_counts") or {})

        typer.echo("Signal Collection Status")
        typer.echo("=" * 40)
        typer.echo(f"Enabled: {stats['config']['enabled']}")
        typer.echo(f"Total Interactions: {stats['total_interactions']}")
        typer.echo(f"Total Signals: {len(persisted_signals)}")
        typer.echo(f"In-Memory Signals: {stats['total_signals']}")
        typer.echo(f"Curated Samples: {snapshot.get('signal_sample_count', 0)}")
        typer.echo(
            "Dataset Splits: "
            f"train={sample_counts.get('train', 0)} "
            f"val={sample_counts.get('val', 0)} "
            f"test={sample_counts.get('test', 0)}"
        )
        typer.echo(f"Event Chain Ready: {signal_summary.get('event_chain_ready', False)}")
        latest_signal_id = signal_summary.get("latest_signal_id")
        if latest_signal_id:
            typer.echo(f"Latest Signal: {latest_signal_id}")
        typer.echo("\nSignals by Type:")
        for signal_name, count in persisted_counts.items():
            typer.echo(f"  {signal_name}: {count}")
        if not persisted_counts:
            typer.echo("  none: 0")
        typer.echo("\nThresholds:")
        typer.echo(f"  Accept: {stats['config']['accept_threshold']}")
        typer.echo(f"  Edit: {stats['config']['edit_threshold']}")

    @collect_app.command("ingest")
    def collect_ingest(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        user_input: str = typer.Option(..., "--user-input", help="Original user input."),
        model_output: str = typer.Option(..., "--model-output", help="Assistant/model output being judged."),
        action: str = typer.Option("accept", "--action", help="Feedback action: accept, edit, reject, regenerate."),
        edited_text: Optional[str] = typer.Option(None, "--edited-text", help="Replacement text for edit feedback."),
        rejected_text: Optional[str] = typer.Option(None, "--rejected-text", help="Rejected text for reject/regenerate feedback."),
        confidence: Optional[float] = typer.Option(None, "--confidence", min=0.0, max=1.0, help="Review confidence stored with the signal."),
        scenario: Optional[str] = typer.Option(None, "--scenario", help="Optional scenario label."),
        event_id: Optional[str] = typer.Option(None, "--event-id", help="Explicit signal event id."),
        request_id: Optional[str] = typer.Option(None, "--request-id", help="Explicit request id."),
        session_id: Optional[str] = typer.Option(None, "--session-id", help="Explicit session id."),
        source_event_id: Optional[str] = typer.Option(None, "--source-event-id", help="Original chat event id."),
    ) -> None:
        """Ingest one local feedback signal without starting the HTTP server."""

        normalized_action = action.strip().lower()
        valid_actions = {"accept", "edit", "reject", "regenerate"}
        if normalized_action not in valid_actions:
            raise typer.BadParameter(
                "action must be one of: accept, edit, reject, regenerate",
                param_hint="--action",
            )

        from pfe_core.pipeline import PipelineService

        signal_id = event_id or _cli_generated_id("evt")
        source_id = source_event_id or f"{signal_id}-source"
        payload = {
            "event_id": signal_id,
            "request_id": request_id or _cli_generated_id("req"),
            "session_id": session_id or _cli_generated_id("sess"),
            "source_event_id": source_id,
            "source_event_ids": [source_id, signal_id],
            "event_type": normalized_action,
            "user_input": user_input,
            "model_output": model_output,
            "user_action": _action_payload(
                action=normalized_action,
                model_output=model_output,
                edited_text=edited_text,
                rejected_text=rejected_text,
            ),
            "metadata": {
                "workspace": workspace or "user_default",
                "scenario": scenario,
                "source": "pfe_collect_ingest",
                "signal_type": normalized_action,
                "confidence": confidence if confidence is not None else _default_signal_confidence(normalized_action),
            },
        }
        result = PipelineService().signal(payload)
        auto_train = dict(result.get("auto_train") or {})

        typer.echo("Signal ingested")
        typer.echo(f"Signal ID: {result.get('event_id')}")
        typer.echo(f"Recorded: {result.get('recorded')}")
        typer.echo(f"Event Chain Complete: {result.get('event_chain_complete')}")
        typer.echo(f"Curation: {result.get('curation_state')} ({result.get('curation_reason')})")
        typer.echo(f"Curated Samples: {result.get('curated_samples', 0)}")
        curated_ids = list(result.get("curated_sample_ids") or [])
        if curated_ids:
            typer.echo("Curated Sample IDs: " + ", ".join(str(item) for item in curated_ids))
        typer.echo(f"Auto Train: {auto_train.get('state', 'unknown')} ({auto_train.get('reason', 'n/a')})")
        typer.echo(f"Review: pfe collect review --type {normalized_action} --limit 5")

    @collect_app.command("review")
    def collect_review(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        signal_type: Optional[str] = typer.Option(None, "--type", help="Filter by signal type (accept, edit, reject, regenerate)."),
        min_confidence: float = typer.Option(0.0, "--min-confidence", help="Minimum confidence threshold."),
        max_confidence: float = typer.Option(1.0, "--max-confidence", help="Maximum confidence threshold."),
        limit: int = typer.Option(20, "--limit", help="Maximum number of signals to display."),
    ) -> None:
        """Review collected signals for manual verification."""

        signals = _filtered_persisted_signals(
            home=_config_home(),
            signal_type=signal_type,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            limit=limit,
        )

        if not signals:
            typer.echo("No signals found matching the criteria.")
            return

        typer.echo(f"Collected Signals (showing {len(signals)})")
        typer.echo("=" * 60)

        for i, signal in enumerate(signals, 1):
            metadata = dict(signal.get("metadata") or {})
            typer.echo(f"\n[{i}] Signal ID: {signal.get('signal_id')}")
            typer.echo(f"    Type: {_signal_type(signal)}")
            typer.echo(f"    Confidence: {_signal_confidence(signal):.2f}")
            if metadata.get("extraction_rule"):
                typer.echo(f"    Rule: {metadata['extraction_rule']}")
            typer.echo(f"    Session: {signal.get('session_id')}")
            if metadata.get("edit_distance") is not None:
                typer.echo(f"    Edit Distance: {metadata['edit_distance']}")
            if metadata.get("response_time_seconds") is not None:
                typer.echo(f"    Response Time: {float(metadata['response_time_seconds']):.1f}s")
            context = str(signal.get("context") or signal.get("user_input") or "")
            typer.echo(f"    Context: {context[:100]}..." if len(context) > 100 else f"    Context: {context}")
