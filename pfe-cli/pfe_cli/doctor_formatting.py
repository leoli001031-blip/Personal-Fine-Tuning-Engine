"""Doctor readiness formatting helpers."""

from __future__ import annotations

from .doctor_audit_formatting import (
    _format_doctor_pii_compliance,
    _format_doctor_signal_chain_integrity,
    _format_doctor_training_audit,
)
from .doctor_formatting_deps import DoctorFormattingDeps
from .doctor_model_formatting import (
    _format_doctor_adapter_home,
    _format_doctor_export_tool,
    _format_doctor_local_model,
    _format_doctor_snapshot_summary,
)
from .doctor_package_formatting import (
    _format_doctor_package_mapping,
    _format_doctor_trainer_deps,
)
from .doctor_readiness_formatting import (
    _format_doctor_blocked_capabilities,
    _format_doctor_next_steps,
)


def format_doctor(
    *,
    workspace: str | None = None,
    base_model: str | None = None,
    deps: DoctorFormattingDeps,
) -> str:
    """Return the PFE doctor readiness summary."""

    lines = ["PFE doctor"]

    runtime = deps.optional_module_call("pfe_core.trainer.runtime", "detect_trainer_runtime")
    trainer_line = _format_doctor_trainer_deps(runtime, deps)
    if trainer_line is not None:
        lines.append(trainer_line)

    local_model_line = _format_doctor_local_model(workspace, base_model, deps)
    if local_model_line is not None:
        lines.append(local_model_line)

    export_tool_line = _format_doctor_export_tool(deps)
    if export_tool_line is not None:
        lines.append(export_tool_line)

    latest_snapshot = deps.lookup_adapter_snapshot("latest", workspace=workspace)
    recent_snapshot = deps.lookup_recent_adapter_snapshot(workspace=workspace)
    lines.append(
        _format_doctor_blocked_capabilities(
            trainer_line=trainer_line,
            local_model_line=local_model_line,
            export_tool_line=export_tool_line,
            latest_snapshot=latest_snapshot,
            recent_snapshot=recent_snapshot,
            deps=deps,
        )
    )
    lines.append(
        _format_doctor_next_steps(
            workspace=workspace,
            trainer_line=trainer_line,
            local_model_line=local_model_line,
            export_tool_line=export_tool_line,
            latest_snapshot=latest_snapshot,
            recent_snapshot=recent_snapshot,
            deps=deps,
        )
    )
    lines.append(
        "capability boundaries: "
        "train/core | eval/core | serve/core | generate/heuristic | distill/heuristic | profile/heuristic | route/heuristic"
    )
    lines.append("user modeling: runtime=user_memory | analysis=user_profile")
    lines.append(_format_doctor_adapter_home(workspace, deps))

    pii_line = _format_doctor_pii_compliance(workspace=workspace)
    if pii_line is not None:
        lines.append(pii_line)

    training_audit_line = _format_doctor_training_audit(workspace=workspace)
    if training_audit_line is not None:
        lines.append(training_audit_line)

    signal_chain_line = _format_doctor_signal_chain_integrity(workspace=workspace)
    if signal_chain_line is not None:
        lines.append(signal_chain_line)

    return "\n".join(lines)


__all__ = [
    "DoctorFormattingDeps",
    "_format_doctor_adapter_home",
    "_format_doctor_blocked_capabilities",
    "_format_doctor_export_tool",
    "_format_doctor_local_model",
    "_format_doctor_next_steps",
    "_format_doctor_package_mapping",
    "_format_doctor_pii_compliance",
    "_format_doctor_signal_chain_integrity",
    "_format_doctor_snapshot_summary",
    "_format_doctor_trainer_deps",
    "_format_doctor_training_audit",
    "format_doctor",
]
