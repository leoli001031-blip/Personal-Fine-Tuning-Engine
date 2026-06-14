"""Scenario management utility commands."""

from __future__ import annotations

import json

import typer


def register_scenario_commands(scenario_app: typer.Typer) -> None:
    @scenario_app.command("list")
    def scenario_list(
        json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    ) -> None:
        """List all available scenarios."""

        from pfe_core.config import PFEConfig
        from pfe_core.router import create_router

        config = PFEConfig.load()
        router = create_router(config=config)
        scenarios = router.list_scenarios()

        if json_output:
            typer.echo(json.dumps(scenarios, ensure_ascii=False, indent=2))
            return

        if not scenarios:
            typer.echo("No scenarios configured.")
            return

        lines = ["Available scenarios:", ""]
        for scenario in scenarios:
            lines.append(f"  {scenario['scenario_id']}: {scenario['name']}")
            lines.append(f"    Description: {scenario['description']}")
            lines.append(f"    Adapter: {scenario['adapter_version']}")
            lines.append(
                f"    Keywords: {scenario['keyword_count']} | Examples: {scenario['example_count']} "
                f"| Priority: {scenario['priority']}"
            )
            lines.append("")
        typer.echo("\n".join(lines))

    @scenario_app.command("create")
    def scenario_create(
        name: str = typer.Argument(..., help="Scenario ID (e.g., 'coding', 'writing')."),
        adapter: str = typer.Option("latest", "--adapter", help="Adapter version to bind to this scenario."),
        description: str = typer.Option("", "--description", help="Scenario description."),
        keywords: str = typer.Option("", "--keywords", help="Comma-separated trigger keywords."),
        priority: int = typer.Option(0, "--priority", help="Scenario priority (higher = preferred)."),
        json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    ) -> None:
        """Create a new custom scenario."""

        from pfe_core.config import PFEConfig
        from pfe_core.router import create_router
        from pfe_core.scenarios import create_custom_scenario

        config = PFEConfig.load()
        router = create_router(config=config)

        keyword_list = [keyword.strip() for keyword in keywords.split(",") if keyword.strip()]
        scenario = create_custom_scenario(
            scenario_id=name,
            name=name.replace("_", " ").title(),
            description=description or f"Custom scenario: {name}",
            adapter_version=adapter,
            trigger_keywords=keyword_list,
            priority=priority,
        )
        router.add_scenario(scenario)

        result = {
            "created": True,
            "scenario": scenario.to_dict(),
        }

        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            typer.echo(f"Created scenario '{name}' with adapter '{adapter}'.")
            if keyword_list:
                typer.echo(f"  Keywords: {', '.join(keyword_list)}")

    @scenario_app.command("bind")
    def scenario_bind(
        scenario: str = typer.Argument(..., help="Scenario ID to bind."),
        adapter: str = typer.Option(..., "--adapter", help="Adapter version to bind."),
        json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    ) -> None:
        """Bind a scenario to a specific adapter version."""

        from pfe_core.config import PFEConfig
        from pfe_core.router import create_router

        config = PFEConfig.load()
        router = create_router(config=config)

        success = router.bind_scenario_to_adapter(scenario, adapter)
        result = {
            "bound": success,
            "scenario_id": scenario,
            "adapter_version": adapter,
        }

        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        elif success:
            typer.echo(f"Bound scenario '{scenario}' to adapter '{adapter}'.")
        else:
            typer.echo(f"Failed to bind scenario '{scenario}'. Scenario not found.", err=True)
            raise typer.Exit(code=1)


__all__ = ["register_scenario_commands"]
