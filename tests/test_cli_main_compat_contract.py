from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in os.sys.path:
        os.sys.path.insert(0, package_path)

from pfe_cli import main as cli_main  # noqa: E402


LEGACY_PRIVATE_HELPERS = (
    "_adapter_snapshot_deps",
    "_append_console_line",
    "_build_operations_alert_surface",
    "_build_operations_console_digest",
    "_build_plan_snapshots",
    "_candidate_timeline_stage",
    "_cli_state_deps",
    "_cli_state_path",
    "_coerce_mapping",
    "_console_actions_deps",
    "_console_apply_edit",
    "_console_apply_history",
    "_console_candidate_summary_text",
    "_console_chat_text",
    "_console_command_output",
    "_console_dashboard_focus",
    "_console_daemon_summary_text",
    "_console_focus_actions",
    "_console_gate_summary_text",
    "_console_help_text",
    "_console_io_deps",
    "_console_queue_summary_text",
    "_console_read_input",
    "_console_routing_deps",
    "_console_runner_summary_text",
    "_console_runtime_summary_text",
    "_console_settings_text",
    "_console_shortcut_hint",
    "_console_snapshot_payload",
    "_console_status_compact_text",
    "_console_submit_feedback",
    "_console_surface_deps",
    "_console_trigger_summary_text",
    "_daemon_formatting_deps",
    "_daemon_recovery_payload",
    "_extract_launch_mode",
    "_format_adapter_export_artifact_line",
    "_format_adapter_snapshot_line",
    "_format_backend_dispatch",
    "_format_bytes_compact",
    "_format_candidate_history",
    "_format_candidate_timeline",
    "_format_candidate_timeline_item",
    "_format_compare_evaluation",
    "_format_compact_plan_line",
    "_format_daemon_alerts",
    "_format_daemon_health_status",
    "_format_daemon_heartbeat_status",
    "_format_daemon_lease_status",
    "_format_daemon_stale_check",
    "_format_daemon_timeline_summary",
    "_format_doctor",
    "_format_eval_result",
    "_format_eval_result_legacy",
    "_format_export_execution_summary",
    "_format_export_toolchain_summary",
    "_format_export_write",
    "_format_incremental_context",
    "_format_job_execution_summary",
    "_format_lifecycle_summary",
    "_format_operations_alert_policy",
    "_format_operations_alert_surface",
    "_format_operations_console_digest",
    "_format_operations_dashboard",
    "_format_operations_event_stream",
    "_format_operations_timeline",
    "_format_ops_attention",
    "_format_plan_block",
    "_format_plan_snapshot_lines",
    "_format_real_execution_summary",
    "_format_recent_training_snapshot",
    "_format_runner_timeline_summary",
    "_format_scalar",
    "_format_serve",
    "_format_serve_legacy",
    "_format_serve_preview",
    "_format_serve_preview_legacy",
    "_format_status",
    "_format_status_legacy",
    "_format_train_preview",
    "_format_train_queue_daemon_history",
    "_format_train_queue_daemon_status",
    "_format_train_queue_history",
    "_format_train_result",
    "_format_train_result_legacy",
    "_format_trainer_block",
    "_format_trainer_summary",
    "_format_worker_runner_history",
    "_format_worker_runner_status",
    "_friendly_exception_message",
    "_history_latest_timestamp",
    "_legacy_result_deps",
    "_load_latest_adapter_manifest",
    "_load_service",
    "_lookup_adapter_snapshot",
    "_lookup_recent_adapter_snapshot",
    "_operations_formatting_deps",
    "_operations_history_formatting_deps",
    "_optional_module_call",
    "_pick_first",
    "_plan_snapshot_deps",
    "_plan_summary",
    "_prefer_inspection_summary_for_generic_monitor",
    "_pfe_home",
    "_read_cli_state",
    "_read_train_queue_daemon_state",
    "_record_train_cli_state",
    "_record_train_queue_daemon_history",
    "_resolve_handler",
    "_run_handler",
    "_run_handler_json",
    "_run_placeholder",
    "_serve_formatting_deps",
    "_serve_preview_launch_mode",
    "_serve_preview_runtime_mapping",
    "_status_formatting_deps",
    "_status_legacy_formatting_deps",
    "_train_queue_daemon_state_path",
    "_training_preview_deps",
    "_update_train_queue_daemon_state",
    "_write_cli_state",
    "_write_train_queue_daemon_state",
    "_yes_no",
)


def test_main_exports_legacy_private_helper_contract() -> None:
    missing = [name for name in LEGACY_PRIVATE_HELPERS if not callable(getattr(cli_main, name, None))]

    assert missing == []


def test_console_routing_deps_resolve_handler_stays_patchable() -> None:
    original_resolve_handler = cli_main._resolve_handler

    class Service:
        patched = object()

    try:
        cli_main._resolve_handler = lambda service, *names: service.patched

        deps = cli_main._console_routing_deps()

        assert deps.resolve_handler(Service(), "status") is Service.patched
    finally:
        cli_main._resolve_handler = original_resolve_handler
