"""Basic adapter lifecycle CLI commands."""

from __future__ import annotations

from typing import Optional

import typer

from .adapter_lifecycle_formatting import _echo_result
from .adapter_store_helpers import _call_store


def _echo_store_result(method_name: str, *args: object, **kwargs: object) -> None:
    result = _call_store(method_name, *args, **kwargs)
    if result is not None:
        _echo_result(result)


def register_adapter_basic_commands(adapter_app: typer.Typer) -> None:
    @adapter_app.command("list")
    def list_versions(
        limit: int = typer.Option(20, "--limit", min=1, help="Maximum versions to display."),
        workspace: Optional[str] = typer.Option(
            None,
            "--workspace",
            help="Optional workspace or tenant label for future multi-workspace support.",
        ),
    ) -> None:
        """List adapter versions and their lifecycle state."""

        _echo_store_result("list_versions", limit=limit, workspace=workspace)

    @adapter_app.command("promote")
    def promote(
        version: str = typer.Argument(..., help="Adapter version to promote to latest."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Promote an adapter version. This is the only supported latest update path."""

        _echo_store_result("promote", version, workspace=workspace)

    @adapter_app.command("rollback")
    def rollback(
        version: str = typer.Argument(..., help="Adapter version or relative index to roll back to."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Rollback latest to a prior adapter version."""

        _echo_store_result("rollback", version, workspace=workspace)


__all__ = ["register_adapter_basic_commands"]
