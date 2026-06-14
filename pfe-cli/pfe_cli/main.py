"""Typer entrypoint for the PFE CLI."""

from __future__ import annotations

import typer

from .adapter_commands import adapter_app, _format_lifecycle_summary as adapter_format_lifecycle_summary
from .command_execution import (
    friendly_exception_message,
    load_service,
    optional_module_call,
)
from .legacy_result_formatting import format_bytes_compact
from .main_compat import install_command_compat, install_console_compat, install_state_compat
from .main_format_compat import install_format_compat
from .main_registration import register_main_commands
from .plan_snapshot_helpers import load_latest_adapter_manifest
from .result_formatting import format_eval_result, format_train_result
from .serve_formatting import format_serve
from .shared_formatting import (
    GENERIC_MONITOR_FOCUSES,
    coerce_mapping,
    coerce_sequence_of_mappings,
    coerce_sequence_of_scalars,
    format_backend_dispatch,
    format_compact_plan_line,
    format_export_write,
    format_plan_block,
    format_scalar,
    format_trainer_block,
    format_trainer_summary,
    ordered_eval_scores,
    pick_first,
    plan_summary,
    prefer_inspection_summary_for_generic_monitor,
    yes_no,
)


app = typer.Typer(
    help=(
        "PFE command line interface. Default mode is strict_local. "
        "OpenAI compatibility covers inference only; personalized loops need Signal SDK or /pfe/signal. "
        "Current capability boundary: train/eval/serve are the core loop, while generate/distill/profile/route are still rule-based or bootstrap-oriented surfaces."
    ),
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(adapter_app, name="adapter")
trigger_app = typer.Typer(help="Manage auto-train trigger state and manual recovery.")
app.add_typer(trigger_app, name="trigger")
daemon_app = typer.Typer(help="Manage the background train queue daemon lifecycle.")
app.add_typer(daemon_app, name="daemon")
candidate_app = typer.Typer(help="Manage the current candidate adapter lifecycle.")

eval_trigger_app = typer.Typer(help="Manage auto-evaluation trigger after training.")
app.add_typer(eval_trigger_app, name="eval-trigger")
app.add_typer(candidate_app, name="candidate")
collect_app = typer.Typer(help="Manage signal collection state.")
app.add_typer(collect_app, name="collect")


_load_service = load_service
_friendly_exception_message = friendly_exception_message
_optional_module_call = optional_module_call
_coerce_mapping = coerce_mapping
_coerce_sequence_of_mappings = coerce_sequence_of_mappings
_coerce_sequence_of_scalars = coerce_sequence_of_scalars
_ordered_eval_scores = ordered_eval_scores
_GENERIC_MONITOR_FOCUSES = GENERIC_MONITOR_FOCUSES
_prefer_inspection_summary_for_generic_monitor = prefer_inspection_summary_for_generic_monitor
install_state_compat(globals())


_format_scalar = format_scalar
_yes_no = yes_no
_plan_summary = plan_summary
_format_plan_block = format_plan_block
_format_trainer_block = format_trainer_block
_pick_first = pick_first
_format_compact_plan_line = format_compact_plan_line
_format_trainer_summary = format_trainer_summary
_format_backend_dispatch = format_backend_dispatch
_format_export_write = format_export_write
_format_train_result = format_train_result
_format_lifecycle_summary = adapter_format_lifecycle_summary
_load_latest_adapter_manifest = load_latest_adapter_manifest
_format_serve = format_serve
_format_bytes_compact = format_bytes_compact
_format_eval_result = format_eval_result
install_format_compat(globals())
install_command_compat(globals())
install_console_compat(globals())


register_main_commands(
    app=app,
    trigger_app=trigger_app,
    daemon_app=daemon_app,
    candidate_app=candidate_app,
    eval_trigger_app=eval_trigger_app,
    collect_app=collect_app,
    symbols=globals(),
)


def main() -> None:
    """Console script entrypoint."""

    app()


if __name__ == "__main__":
    main()
