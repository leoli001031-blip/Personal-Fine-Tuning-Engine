"""Doctor formatting symbols for the main compatibility namespace."""

from __future__ import annotations

from typing import Any

from .doctor_formatting import format_doctor
from .main_deps import make_doctor_formatting_deps


def make_result_doctor_symbols(symbols: dict[str, Any]) -> dict[str, Any]:
    def _format_doctor(*, workspace: str | None = None, base_model: str | None = None) -> str:
        return format_doctor(
            workspace=workspace,
            base_model=base_model,
            deps=make_doctor_formatting_deps(symbols),
        )

    return {"_format_doctor": _format_doctor}


__all__ = ["make_result_doctor_symbols"]
