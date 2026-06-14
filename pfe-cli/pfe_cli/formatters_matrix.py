"""Matrix Green Terminal formatters for PFE CLI output."""

from __future__ import annotations

from .matrix_formatting_common import (
    _coerce_mapping,
    _coerce_sequence_of_mappings,
    _coerce_sequence_of_scalars,
    _format_scalar,
    _ordered_eval_scores,
)
from .matrix_result_formatting import (
    format_adapter_list_matrix,
    format_eval_result_matrix,
    format_serve_matrix,
    format_serve_preview_matrix,
    format_train_result_matrix,
)
from .matrix_status_formatting import format_status_matrix
from .terminal_theme import MatrixColors

__all__ = [
    "MatrixColors",
    "_coerce_mapping",
    "_coerce_sequence_of_mappings",
    "_coerce_sequence_of_scalars",
    "_format_scalar",
    "_ordered_eval_scores",
    "format_status_matrix",
    "format_train_result_matrix",
    "format_serve_preview_matrix",
    "format_serve_matrix",
    "format_eval_result_matrix",
    "format_adapter_list_matrix",
]
