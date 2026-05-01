"""Serve command wiring for runtime CLI surfaces."""

from __future__ import annotations

import os
from typing import Optional

import typer

from .runtime_command_deps import RuntimeCommandDeps


def register_serve_command(app: typer.Typer, deps: RuntimeCommandDeps) -> None:
    @app.command("serve")
    def serve(
        port: int = typer.Option(8921, "--port", min=1, max=65535, help="Port to bind."),
        host: str = typer.Option("127.0.0.1", "--host", help="Bind host, default strict_local loopback."),
        adapter: str = typer.Option("latest", "--adapter", help="Adapter version to load at startup."),
        api_key: Optional[str] = typer.Option(None, "--api-key", help="Optional API key for remote access."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        live: bool = typer.Option(False, "--live", help="Actually launch the local server instead of previewing the serve plan."),
        real_local: bool = typer.Option(False, "--real-local", help="Explicitly allow real local model loading for chat inference."),
    ) -> None:
        """Start the OpenAI-compatible inference server. This does not create the personalized loop by itself."""

        service = deps.load_service("pfe_server.app", "pfe_server", "pfe_core.inference", "pfe_core.pipeline")
        if service is None:
            deps.run_placeholder("serve")
            return

        previous_real_local = os.environ.get("PFE_ENABLE_REAL_LOCAL_INFERENCE")
        if real_local:
            os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"
        typer.echo(
            deps.format_serve_preview(
                port=port,
                host=host,
                adapter=adapter,
                workspace=workspace,
                api_key=api_key,
                real_local=real_local,
            )
        )
        handler = deps.resolve_handler(service, "serve", "run", "start")
        if handler is None:
            deps.run_placeholder("serve")
            return

        try:
            deps.run_handler(
                "serve",
                handler,
                formatter=deps.format_serve,
                port=port,
                host=host,
                adapter=adapter,
                api_key=api_key,
                workspace=workspace,
                dry_run=not live,
            )
        finally:
            if previous_real_local is None:
                os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
            else:
                os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = previous_real_local
