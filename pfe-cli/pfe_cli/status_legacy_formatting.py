"""Legacy status output formatting."""

from __future__ import annotations

from typing import Any

from .status_legacy_adapters import append_legacy_adapter_lines
from .status_legacy_auto_train import append_legacy_auto_train_lines
from .status_legacy_candidates import append_legacy_candidate_lines
from .status_legacy_deps import StatusLegacyFormattingDeps
from .status_legacy_formatting_helpers import append_ops_attention_line, append_status_headlines
from .status_legacy_operations import append_legacy_operations_surface_lines
from .status_legacy_queue import append_legacy_train_queue_lines
from .status_legacy_serve import append_legacy_serve_lines
from .status_legacy_sections import extract_legacy_status_sections
from .status_legacy_signal import append_legacy_sample_and_signal_lines
from .status_legacy_training import append_legacy_trainer_and_plan_lines


def format_status_legacy(
    result: Any,
    *,
    workspace: str | None = None,
    deps: StatusLegacyFormattingDeps,
) -> str:
    """Legacy plain text formatter kept for compatibility checks."""

    _coerce_mapping = deps.coerce_mapping
    _format_scalar = deps.format_scalar

    mapping = _coerce_mapping(result)
    if mapping is None:
        return _format_scalar(result)

    lines: list[str] = ["PFE status"]
    sections = extract_legacy_status_sections(mapping, deps=deps)

    append_status_headlines(lines, mapping, deps=deps)

    append_legacy_adapter_lines(
        lines,
        latest_adapter_version=sections.latest_adapter_version,
        latest_adapter_state=sections.latest_adapter_state,
        latest_adapter_map=sections.latest_adapter_map,
        recent_adapter_version=sections.recent_adapter_version,
        recent_adapter_state=sections.recent_adapter_state,
        recent_adapter_map=sections.recent_adapter_map,
        lifecycle=sections.lifecycle,
        deps=deps,
    )
    append_legacy_candidate_lines(
        lines,
        mapping,
        candidate_summary=sections.candidate_summary,
        compare_evaluation=sections.compare_evaluation,
        deps=deps,
    )
    append_legacy_operations_surface_lines(
        lines,
        operations_overview=sections.operations_overview,
        operations_alerts=sections.operations_alerts,
        operations_health=sections.operations_health,
        operations_recovery=sections.operations_recovery,
        operations_next_actions=sections.operations_next_actions,
        operations_dashboard=sections.operations_dashboard,
        operations_alert_policy=sections.operations_alert_policy,
        operations_console=sections.operations_console,
        operations_event_stream=sections.operations_event_stream,
        operations_timeline=sections.operations_timeline,
        candidate_summary=sections.candidate_summary,
        candidate_history=sections.candidate_history,
        candidate_timeline=sections.candidate_timeline,
        daemon_timeline=sections.daemon_timeline,
        runner_timeline=sections.runner_timeline,
        train_queue=sections.train_queue,
        deps=deps,
    )
    append_legacy_train_queue_lines(lines, mapping, train_queue=sections.train_queue, workspace=workspace, deps=deps)
    append_ops_attention_line(lines, sections=sections, deps=deps)
    append_legacy_serve_lines(lines, mapping, deps=deps)

    append_legacy_sample_and_signal_lines(lines, mapping, deps=deps)

    append_legacy_auto_train_lines(lines, mapping, deps=deps)

    append_legacy_trainer_and_plan_lines(
        lines,
        mapping,
        workspace=workspace,
        recent_adapter_version=sections.recent_adapter_version,
        recent_adapter_state=sections.recent_adapter_state,
        recent_adapter_map=sections.recent_adapter_map,
        deps=deps,
    )

    return "\n".join(lines)
