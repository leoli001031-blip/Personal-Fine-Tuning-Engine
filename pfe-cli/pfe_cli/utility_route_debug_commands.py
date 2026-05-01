"""Route debug utility commands."""

from __future__ import annotations

import json

import typer


def register_route_debug_commands(route_app: typer.Typer) -> None:
    @route_app.command("test")
    def route_test(
        text: str = typer.Argument(..., help="Input text to test routing."),
        strategy: str = typer.Option("keyword", "--strategy", help="Routing strategy: keyword or hybrid."),
        show_scores: bool = typer.Option(False, "--show-scores", help="Show detailed scores for all scenarios."),
        json_output: bool = typer.Option(False, "--json", help="Emit detailed JSON output."),
    ) -> None:
        """Test scenario routing for a given input text. Supports keyword and hybrid strategies."""

        from pfe_core.config import PFEConfig
        from pfe_core.router import create_router

        config = PFEConfig.load()
        original_strategy = config.router.strategy
        config.router.strategy = strategy  # type: ignore[misc]
        router = create_router(config=config)
        result = router.test_route(text)
        config.router.strategy = original_strategy  # type: ignore[misc]

        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
            return

        primary = result["primary_route"]
        classification = result["classification"]

        typer.echo(f"Input: {text}")
        typer.echo(f"Strategy: {result.get('strategy', strategy)}")
        typer.echo(f"Primary Intent: {classification['primary_intent']} (confidence: {classification['confidence']:.2f})")
        typer.echo(f"Selected Scenario: {primary['scenario_id']}")
        typer.echo(f"Adapter Version: {primary['adapter_version']}")
        typer.echo(f"Routing Confidence: {primary['confidence']:.2f}")
        if primary["fallback"]:
            typer.echo("Note: Using fallback routing (low confidence)")
        typer.echo(f"Reasoning: {primary['reasoning']}")

        if show_scores and result["all_routes"]:
            typer.echo("\nAll scenario scores:")
            for route in result["all_routes"]:
                typer.echo(f"  {route['scenario_id']}: {route['score']:.3f}")
        elif result["all_routes"]:
            typer.echo("\nTop scenario scores:")
            for route in result["all_routes"][:5]:
                typer.echo(f"  {route['scenario_id']}: {route['score']:.3f}")


__all__ = ["register_route_debug_commands"]
