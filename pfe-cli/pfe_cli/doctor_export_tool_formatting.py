"""llama.cpp export tool formatting for doctor output."""

from __future__ import annotations

from .doctor_formatting_deps import DoctorFormattingDeps


def _format_doctor_export_tool(deps: DoctorFormattingDeps) -> str | None:
    resolution = deps.optional_module_call("pfe_core.inference.export_runtime", "resolve_llama_cpp_export_tool_path")
    validation = None
    if resolution is not None:
        validation = deps.optional_module_call(
            "pfe_core.inference.export_runtime",
            "validate_llama_cpp_export_toolchain",
            resolution,
        )

    mapping = deps.coerce_mapping(validation) or deps.coerce_mapping(resolution)
    if mapping is None:
        return "llama.cpp export tool: status=n/a | allowed=n/a | reason=probe unavailable"

    parts: list[str] = []
    if "status" in mapping:
        parts.append(f"status={deps.format_scalar(mapping.get('status'))}")
    if "allowed" in mapping:
        parts.append(f"allowed={deps.format_scalar(mapping.get('allowed'))}")
    if "resolved_path" in mapping:
        parts.append(f"resolved_path={deps.format_scalar(mapping.get('resolved_path'))}")
    if "reason" in mapping:
        parts.append(f"reason={deps.format_scalar(mapping.get('reason'))}")
    if not parts:
        return None
    return "llama.cpp export tool: " + " | ".join(parts)


__all__ = ["_format_doctor_export_tool"]
