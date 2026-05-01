"""Signal collection command registration."""

from __future__ import annotations

from typing import Any, Optional

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

        collector = _collector_for_workspace(workspace)
        stats = collector.get_stats()

        typer.echo("Signal Collection Status")
        typer.echo("=" * 40)
        typer.echo(f"Enabled: {stats['config']['enabled']}")
        typer.echo(f"Total Interactions: {stats['total_interactions']}")
        typer.echo(f"Total Signals: {stats['total_signals']}")
        typer.echo("\nSignals by Type:")
        for signal_type, count in stats["signals_by_type"].items():
            typer.echo(f"  {signal_type}: {count}")
        typer.echo("\nThresholds:")
        typer.echo(f"  Accept: {stats['config']['accept_threshold']}")
        typer.echo(f"  Edit: {stats['config']['edit_threshold']}")

    @collect_app.command("review")
    def collect_review(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        signal_type: Optional[str] = typer.Option(None, "--type", help="Filter by signal type (accept, edit, reject, regenerate)."),
        min_confidence: float = typer.Option(0.0, "--min-confidence", help="Minimum confidence threshold."),
        max_confidence: float = typer.Option(1.0, "--max-confidence", help="Maximum confidence threshold."),
        limit: int = typer.Option(20, "--limit", help="Maximum number of signals to display."),
    ) -> None:
        """Review collected signals for manual verification."""

        collector = _collector_for_workspace(workspace)
        signals = collector.get_signals_for_review(
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
            typer.echo(f"\n[{i}] Signal ID: {signal.signal_id}")
            typer.echo(f"    Type: {signal.signal_type}")
            typer.echo(f"    Confidence: {signal.confidence:.2f}")
            typer.echo(f"    Rule: {signal.extraction_rule}")
            typer.echo(f"    Session: {signal.session_id}")
            if signal.edit_distance is not None:
                typer.echo(f"    Edit Distance: {signal.edit_distance}")
            if signal.response_time_seconds is not None:
                typer.echo(f"    Response Time: {signal.response_time_seconds:.1f}s")
            context = signal.context
            typer.echo(f"    Context: {context[:100]}..." if len(context) > 100 else f"    Context: {context}")
