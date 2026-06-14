"""Train and evaluation result formatting helpers."""

from __future__ import annotations

from typing import Any

from . import formatters_matrix


def format_train_result(result: Any, *, workspace: str | None = None) -> str:
    return formatters_matrix.format_train_result_matrix(result, workspace=workspace)


def format_eval_result(result: Any, *, workspace: str | None = None) -> str:
    return formatters_matrix.format_eval_result_matrix(result, workspace=workspace)
