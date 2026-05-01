"""Adapter command result output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import typer

from . import formatters_matrix
from .adapter_lifecycle_summary import _format_lifecycle_summary
from .adapter_value_formatting import _format_value


def _echo_result(result: Any) -> None:
    mapping = result if isinstance(result, dict) else None
    if mapping is not None and "versions" in mapping and isinstance(mapping["versions"], Sequence):
        versions = [item if isinstance(item, dict) else {} for item in mapping["versions"]]
        typer.echo(formatters_matrix.format_adapter_list_matrix(versions))
        return

    lifecycle_lines = _format_lifecycle_summary(result)
    if lifecycle_lines is not None:
        for line in lifecycle_lines:
            typer.echo(line)
        return

    if mapping is not None:
        for key in sorted(mapping):
            typer.echo(f"{key.replace('_', ' ')}: {_format_value(mapping[key])}")
        return

    typer.echo(_format_value(result))


__all__ = ["_echo_result"]
