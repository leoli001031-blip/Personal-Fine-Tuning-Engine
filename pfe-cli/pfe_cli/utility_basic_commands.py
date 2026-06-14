"""Root-level utility commands."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Mapping
from typing import Any, Optional

import typer

from . import formatters_matrix
from .utility_command_deps import UtilityCommandDeps


DEFAULT_WORKSPACE = "user_default"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _workspace_name(workspace: str | None) -> str:
    return workspace or DEFAULT_WORKSPACE


def _build_next_plan(workspace: str | None = None) -> dict[str, Any]:
    from pfe_core.config import PFEConfig
    from pfe_core.pipeline import PipelineService

    workspace_name = _workspace_name(workspace)
    home = PFEConfig.resolve_home().expanduser()
    config_path = PFEConfig.default_path()
    plan: dict[str, Any] = {
        "workspace": workspace_name,
        "home": str(home),
        "config_path": str(config_path),
        "state": "unknown",
        "next": "inspect current status",
        "why": "unable to classify local state",
        "commands": [f"pfe doctor --workspace {workspace_name}", f"pfe status --workspace {workspace_name}"],
    }

    if not config_path.exists():
        plan.update(
            {
                "state": "init_required",
                "next": "initialize a local PFE workspace",
                "why": f"config file was not found at {config_path}",
                "commands": [
                    f"pfe init --workspace {workspace_name} --base-model <path-or-model-id>",
                    f"pfe doctor --workspace {workspace_name}",
                ],
            }
        )
        return plan

    config = PFEConfig.load()
    base_model = str(getattr(config.model, "base_model", "") or "").strip()
    plan["base_model"] = base_model or None
    if not base_model:
        plan.update(
            {
                "state": "base_model_required",
                "next": "set the base model used for local readiness checks",
                "why": "config exists but model.base_model is empty",
                "commands": [
                    f"pfe init --workspace {workspace_name} --base-model <path-or-model-id>",
                    f"pfe doctor --workspace {workspace_name} --base-model <path-or-model-id>",
                ],
            }
        )
        return plan

    try:
        status = PipelineService().status(workspace=workspace_name)
    except Exception as exc:
        plan.update(
            {
                "state": "doctor_required",
                "next": "run doctor and inspect the local runtime",
                "why": f"status snapshot failed: {exc.__class__.__name__}: {exc}",
                "commands": [f"pfe doctor --workspace {workspace_name}", f"pfe status --workspace {workspace_name} --json"],
            }
        )
        return plan

    signal_summary = _mapping(status.get("signal_summary"))
    sample_counts = _mapping(status.get("sample_counts"))
    train_queue = _mapping(status.get("train_queue"))
    queue_counts = _mapping(train_queue.get("counts"))
    candidate_summary = _mapping(status.get("candidate_summary"))
    serve = _mapping(status.get("serve"))
    auto_train_trigger = _mapping(status.get("auto_train_trigger"))

    signal_count = _int_value(status.get("signal_count") or signal_summary.get("total"))
    train_samples = _int_value(sample_counts.get("train"))
    queued_count = _int_value(queue_counts.get("queued"))
    confirmation_count = _int_value(queue_counts.get("awaiting_confirmation"))
    completed_count = _int_value(queue_counts.get("completed"))
    candidate_version = _first_non_empty(
        candidate_summary.get("candidate_version"),
        candidate_summary.get("recent_version"),
        status.get("recent_adapter_version"),
    )
    latest_promoted = _first_non_empty(
        candidate_summary.get("latest_promoted_version"),
        serve.get("target_adapter_version"),
        status.get("latest_adapter_version"),
    )

    plan["observed"] = {
        "signal_count": signal_count,
        "train_samples": train_samples,
        "queued": queued_count,
        "awaiting_confirmation": confirmation_count,
        "completed": completed_count,
        "candidate_version": candidate_version,
        "latest_promoted_version": latest_promoted,
        "auto_train_state": auto_train_trigger.get("state"),
        "auto_train_reason": auto_train_trigger.get("reason"),
    }

    if confirmation_count:
        plan.update(
            {
                "state": "queue_confirmation_required",
                "next": "review and approve or reject the next queued training job",
                "why": f"{confirmation_count} queue item(s) are awaiting confirmation",
                "commands": [
                    f"pfe trigger status --workspace {workspace_name}",
                    f"pfe trigger approve-next --workspace {workspace_name}",
                    f"pfe trigger reject-next --workspace {workspace_name}",
                ],
            }
        )
    elif queued_count:
        plan.update(
            {
                "state": "queue_ready",
                "next": "process the next deferred auto-train queue item",
                "why": f"{queued_count} queue item(s) are ready to run",
                "commands": [
                    f"pfe trigger process-next --workspace {workspace_name}",
                    f"pfe trigger status --workspace {workspace_name}",
                ],
            }
        )
    elif bool(candidate_summary.get("candidate_needs_promotion")) and candidate_version:
        plan.update(
            {
                "state": "candidate_ready",
                "next": "evaluate or promote the current candidate adapter",
                "why": f"candidate {candidate_version} needs a lifecycle decision",
                "commands": [
                    f"pfe eval --base-model base --adapter {candidate_version} --workspace {workspace_name}",
                    f"pfe adapter promote {candidate_version} --workspace {workspace_name}",
                    f"pfe serve --port 8921 --workspace {workspace_name} --live",
                ],
            }
        )
    elif bool(serve.get("using_promoted_adapter")) and latest_promoted:
        plan.update(
            {
                "state": "serve_ready",
                "next": "start local serving or open the dashboard",
                "why": f"promoted adapter {latest_promoted} is available",
                "commands": [
                    f"pfe serve --port 8921 --workspace {workspace_name} --live",
                    "pfe dashboard",
                ],
            }
        )
    elif signal_count <= 0:
        plan.update(
            {
                "state": "collect_feedback",
                "next": "create the first local feedback signal and queue a mock training run",
                "why": "no persisted feedback signal is available yet",
                "commands": [
                    f"pfe generate --scenario life-coach --style warm --num 8 --workspace {workspace_name}",
                    (
                        f"pfe trigger configure --workspace {workspace_name} --enable --min-new-samples 1 "
                        "--queue-mode deferred --max-interval-days 0 --no-require-confirmation --epochs 1 --backend mock_local"
                    ),
                    f"pfe collect ingest --workspace {workspace_name} --help",
                    f"pfe trigger process-next --workspace {workspace_name}",
                ],
            }
        )
    elif completed_count and candidate_version:
        plan.update(
            {
                "state": "evaluate_candidate",
                "next": "evaluate the adapter produced by the local queue",
                "why": f"queue has completed work and recent adapter {candidate_version} is available",
                "commands": [
                    f"pfe eval --base-model base --adapter {candidate_version} --num-samples 3 --workspace {workspace_name}",
                    f"pfe adapter promote {candidate_version} --workspace {workspace_name}",
                ],
            }
        )
    else:
        plan.update(
            {
                "state": "inspect_status",
                "next": "inspect the compact trigger and collection surfaces",
                "why": "state is initialized but does not match a single happy-path checkpoint",
                "commands": [
                    f"pfe collect status --workspace {workspace_name}",
                    f"pfe trigger status --workspace {workspace_name}",
                    f"pfe status --workspace {workspace_name}",
                ],
            }
        )
    return plan


def _format_next_plan(plan: Mapping[str, Any]) -> str:
    lines = [
        "PFE next",
        f"workspace: {plan.get('workspace')}",
        f"home: {plan.get('home')}",
        f"state: {plan.get('state')}",
        f"next: {plan.get('next')}",
        f"why: {plan.get('why')}",
    ]
    observed = _mapping(plan.get("observed"))
    if observed:
        compact = " | ".join(f"{key}={value}" for key, value in observed.items() if value not in (None, ""))
        if compact:
            lines.append(f"observed: {compact}")
    commands = [str(item) for item in plan.get("commands") or []]
    if commands:
        lines.append("commands:")
        lines.extend(f"  {command}" for command in commands)
    return "\n".join(lines)


def _dashboard_health_check(host: str, port: int) -> tuple[bool, str]:
    checks = (
        "/healthz",
        "/dashboard",
        "/pfe/dashboard/metrics",
    )
    ok_parts: list[str] = []
    for path in checks:
        url = f"http://{host}:{port}{path}"
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=1.0) as response:
                status = int(getattr(response, "status", 0) or 0)
        except urllib.error.HTTPError as exc:
            return False, f"unhealthy | GET {url} -> HTTP {exc.code}"
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            return False, f"unavailable | GET {url} failed: {reason}"
        except OSError as exc:
            return False, f"unavailable | GET {url} failed: {exc}"
        if not 200 <= status < 300:
            return False, f"unhealthy | GET {url} -> HTTP {status}"
        ok_parts.append(f"GET {url} -> HTTP {status}")
    return True, "ok | " + " | ".join(ok_parts)


def register_basic_utility_commands(app: typer.Typer, deps: UtilityCommandDeps) -> None:
    @app.command("doctor")
    def doctor(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        base_model: Optional[str] = typer.Option(
            None,
            "--base-model",
            help="Override base model path or model id for local model checks.",
        ),
    ) -> None:
        """Show strict_local readiness signals for trainer, model, export, and adapter state."""

        typer.echo(deps.format_doctor(workspace=workspace, base_model=base_model))

    @app.command("next")
    def next_step(
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
        json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON instead of text."),
    ) -> None:
        """Show the next recommended command for the current local PFE state."""

        plan = _build_next_plan(workspace=workspace)
        if json_output:
            typer.echo(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            return
        typer.echo(_format_next_plan(plan))

    @app.command("dashboard")
    def dashboard(
        port: int = typer.Option(8921, "--port", min=1, max=65535, help="Server port to connect to."),
        host: str = typer.Option("127.0.0.1", "--host", help="Server host."),
        open_browser: bool = typer.Option(True, "--open/--no-open", help="Open dashboard in browser."),
    ) -> None:
        """Launch the PFE observability dashboard in a web browser."""

        dashboard_url = f"http://{host}:{port}/dashboard"
        healthy, health_summary = _dashboard_health_check(host, port)

        typer.echo("PFE Observability Dashboard")
        typer.echo(f"URL: {dashboard_url}")
        typer.echo(f"Health check: {health_summary}")

        if open_browser and healthy:
            typer.echo("Opening browser...")
            webbrowser.open(dashboard_url)
        elif open_browser:
            typer.echo("Server is not reachable; browser was not opened.")
            typer.echo(f"Start server: pfe serve --port {port} --live")
        else:
            typer.echo("Use --open to launch browser automatically after the server is healthy.")
            if not healthy:
                typer.echo(f"Start server: pfe serve --port {port} --live")

    @app.command("boot")
    def boot() -> None:
        """Display PFE boot sequence with ZC logo."""

        from .pixel_logo import render_boot_banner, render_commands_matrix, render_loading_sequence

        typer.echo(render_boot_banner(version="2.0.0"))

        steps = [
            "Loading adapter store...",
            "Initializing trainer service...",
            "Mounting signal collector...",
            "Establishing daemon connection...",
            "Calibrating neural weights...",
        ]

        for index, step in enumerate(steps, 1):
            typer.echo(
                f"{formatters_matrix.MatrixColors.GREEN}  "
                f"{render_loading_sequence(index, len(steps))}{formatters_matrix.MatrixColors.RESET} {step}"
            )
            time.sleep(0.15)

        typer.echo("")
        typer.echo(
            f"{formatters_matrix.MatrixColors.GREEN_BRIGHT}{formatters_matrix.MatrixColors.BOLD}  "
            f">> ALL SYSTEMS OPERATIONAL <<{formatters_matrix.MatrixColors.RESET}"
        )
        typer.echo("")
        typer.echo(render_commands_matrix())


__all__ = ["register_basic_utility_commands"]
