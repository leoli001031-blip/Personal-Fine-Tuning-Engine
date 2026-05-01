"""Console command wiring for runtime CLI surfaces."""

from __future__ import annotations

import time
from typing import Optional

import typer

from . import formatters_matrix
from .runtime_command_deps import RuntimeCommandDeps
from .runtime_console_interactive import run_interactive_console


def register_console_command(app: typer.Typer, deps: RuntimeCommandDeps) -> None:
    @app.command("console")
    def console(
        workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Workspace label."),
        interactive: bool = typer.Option(False, "--interactive", help="Open a simple prompt loop on top of the console snapshot."),
        model: str = typer.Option("local", "--model", help="Chat model id or special local alias for interactive mode."),
        adapter: str = typer.Option("latest", "--adapter", help="Adapter version used for interactive chat mode."),
        temperature: float = typer.Option(0.7, "--temperature", min=0.0, max=2.0, help="Chat temperature for interactive mode."),
        max_tokens: Optional[int] = typer.Option(None, "--max-tokens", min=1, help="Optional max tokens for interactive chat mode."),
        real_local: bool = typer.Option(False, "--real-local", help="Allow real local inference during interactive chat."),
        watch: bool = typer.Option(False, "--watch", help="Refresh the console snapshot repeatedly."),
        refresh_seconds: float = typer.Option(2.0, "--refresh-seconds", min=0.1, help="Refresh interval when --watch is enabled."),
        cycles: int = typer.Option(1, "--cycles", min=1, help="Number of render cycles. Use 1 for a single snapshot."),
    ) -> None:
        """Render a Rich-based PFE operations console with optional prompt mode."""

        from .console_app import render_console_snapshot
        from .pixel_logo import render_boot_banner

        typer.echo(render_boot_banner())
        typer.echo(f"{formatters_matrix.MatrixColors.GREEN}  [■] Initializing console interface...{formatters_matrix.MatrixColors.RESET}")
        typer.echo(f"{formatters_matrix.MatrixColors.GREEN}  [■] Loading Rich console components...{formatters_matrix.MatrixColors.RESET}")
        typer.echo(f"{formatters_matrix.MatrixColors.GREEN}  [■] Establishing service connections...{formatters_matrix.MatrixColors.RESET}")
        typer.echo("")
        typer.echo(
            f"{formatters_matrix.MatrixColors.GREEN_BRIGHT}{formatters_matrix.MatrixColors.BOLD}  "
            f">> ENTERING MATRIX CONSOLE MODE <<{formatters_matrix.MatrixColors.RESET}"
        )
        typer.echo("")

        service = deps.load_service("pfe_core.pipeline", "pfe_core.status", "pfe_server.app", "pfe_core.services.pipeline")
        if service is None:
            deps.run_placeholder("console")
            return

        handler = deps.resolve_handler(service, "status", "get_status")
        if handler is None:
            deps.run_placeholder("console")
            return

        if interactive:
            run_interactive_console(
                deps=deps,
                service=service,
                handler=handler,
                workspace=workspace,
                model=model,
                adapter=adapter,
                temperature=temperature,
                max_tokens=max_tokens,
                real_local=real_local,
                refresh_seconds=refresh_seconds,
            )
            return

        run_cycles = cycles if watch else 1
        for index in range(run_cycles):
            try:
                result = handler(workspace=workspace)
            except typer.Exit:
                raise
            except Exception as exc:
                friendly = deps.friendly_exception_message(exc)
                if friendly is not None:
                    typer.secho(friendly, err=True, fg=typer.colors.RED)
                    raise typer.Exit(code=1)
                raise
            mapping = deps.coerce_mapping(result)
            payload = mapping if mapping is not None else {"status_result": str(result)}
            render_console_snapshot(payload, workspace=workspace, clear=index > 0)
            if watch and index < run_cycles - 1:
                time.sleep(refresh_seconds)
