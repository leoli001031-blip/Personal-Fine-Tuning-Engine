"""Background runner for the Phase 1 train queue daemon.

Enhanced with reliability mechanisms:
- Heartbeat tracking for stale runner detection
- Lease management for task execution
- Restart policy with exponential backoff
- Recovery checkpoint support
- Alert generation for operations monitoring
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

from .pipeline import PipelineService
from .reliability import (
    AlertLevel,
    ReliabilityService,
    RestartPolicy,
    RunnerHeartbeat,
    TaskExecutionMetadata,
    TaskLease,
)

def run_worker_daemon(
    *,
    workspace: str | None = None,
    runner_max_seconds: float = 30.0,
    idle_sleep_seconds: float = 1.0,
    heartbeat_interval_seconds: float = 2.0,
    lease_timeout_seconds: float = 15.0,
    max_restart_attempts: int = 3,
    restart_backoff_seconds: float = 15.0,
    enable_reliability: bool = True,
) -> int:
    service = PipelineService()
    workspace_name = workspace or "user_default"
    stop_flag = {"requested": False, "reason": "daemon_stopped"}
    daemon_runner_id = f"daemon-{uuid4().hex[:12]}"

    # Initialize reliability service
    reliability: Optional[ReliabilityService] = None
    if enable_reliability:
        restart_policy = RestartPolicy(
            max_restart_attempts=max_restart_attempts,
            base_backoff_seconds=restart_backoff_seconds,
        )
        reliability = ReliabilityService(
            workspace=workspace_name,
            lease_timeout_seconds=lease_timeout_seconds,
            restart_policy=restart_policy,
        )

    def _request_stop(signum: int, _frame: Any) -> None:
        stop_flag["requested"] = True
        stop_flag["reason"] = f"signal_{signum}"

    def _get_memory_metrics() -> dict[str, Any]:
        if psutil is None:
            return {}
        try:
            proc = psutil.Process(os.getpid())
            mem_info = proc.memory_info()
            return {
                "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
                "vms_mb": round(mem_info.vms / (1024 * 1024), 2),
                "percent": round(proc.memory_percent(), 2),
            }
        except Exception:
            return {}

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _request_stop)
        except Exception:
            continue

    def _touch_daemon_state(*, status: str = "running", observed_state: str = "running") -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = service._load_train_queue_daemon_state(workspace=workspace)
        payload.update(
            {
                "workspace": workspace_name,
                "desired_state": "running",
                "observed_state": observed_state,
                "command_status": status,
                "active": True,
                "pid": os.getpid(),
                "started_at": payload.get("started_at") or now,
                "last_heartbeat_at": now,
                "lease_renewed_at": now,
                "stop_requested": False,
                "heartbeat_interval_seconds": heartbeat_interval_seconds,
                "lease_timeout_seconds": lease_timeout_seconds,
                "heartbeat_timeout_seconds": max(lease_timeout_seconds, heartbeat_interval_seconds * 3.0),
                "runner_max_seconds": runner_max_seconds,
                "idle_sleep_seconds": idle_sleep_seconds,
                "reliability_enabled": enable_reliability,
                "memory": _get_memory_metrics(),
            }
        )
        # Include restart policy state if reliability is enabled
        if reliability is not None:
            payload["restart_attempts"] = reliability.restart_policy.current_attempt
            payload["max_restart_attempts"] = reliability.restart_policy.max_restart_attempts
            payload["restart_backoff_seconds"] = reliability.restart_policy.calculate_backoff()
        service._persist_train_queue_daemon_state(payload, workspace=workspace)

    def _send_runner_heartbeat(
        job_id: Optional[str] = None,
        progress_percent: float = 0.0,
        current_step: str = "",
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send heartbeat from runner to daemon."""
        if reliability is None:
            return
        heartbeat = RunnerHeartbeat(
            runner_id=daemon_runner_id,
            job_id=job_id,
            workspace=workspace_name,
            pid=os.getpid(),
            progress_percent=progress_percent,
            current_step=current_step,
            metrics=metrics or {},
        )
        reliability.process_heartbeat(heartbeat)

    _touch_daemon_state()
    service._append_train_queue_daemon_history(
        workspace=workspace,
        event="started",
        reason="daemon_started",
        metadata={
            "pid": os.getpid(),
            "runner_id": daemon_runner_id,
            "runner_max_seconds": runner_max_seconds,
            "heartbeat_interval_seconds": heartbeat_interval_seconds,
            "lease_timeout_seconds": lease_timeout_seconds,
            "reliability_enabled": enable_reliability,
            "max_restart_attempts": max_restart_attempts,
            "restart_backoff_seconds": restart_backoff_seconds,
        },
    )

    processed_total = 0
    failed_total = 0
    loop_cycles = 0
    exit_reason = "stop_requested"
    runner_slice_seconds = max(
        0.1,
        min(
            runner_max_seconds,
            max(0.1, heartbeat_interval_seconds),
            max(0.1, lease_timeout_seconds / 3.0),
        ),
    )
    try:
        while not stop_flag["requested"]:
            current = service._load_train_queue_daemon_state(workspace=workspace)
            if str(current.get("desired_state") or "running") != "running" or bool(current.get("stop_requested")):
                exit_reason = "stop_requested"
                break
            loop_cycles += 1
            _touch_daemon_state(status="running", observed_state="running")

            # Send runner heartbeat if reliability is enabled
            _send_runner_heartbeat(
                job_id=None,  # Daemon-level heartbeat
                progress_percent=0.0,
                current_step="daemon_loop",
                metrics={
                    "loop_cycles": loop_cycles,
                    "processed_total": processed_total,
                    "failed_total": failed_total,
                    "memory": _get_memory_metrics(),
                },
            )

            current = service._load_train_queue_daemon_state(workspace=workspace)
            current.update(
                {
                    "loop_cycles": loop_cycles,
                    "processed_count": processed_total,
                    "failed_count": failed_total,
                }
            )
            service._persist_train_queue_daemon_state(current, workspace=workspace)

            # Check for stalled jobs before processing
            if reliability is not None:
                stalled_jobs = reliability.heartbeat.detect_stalled_jobs()
                for stalled in stalled_jobs:
                    service._append_train_queue_daemon_history(
                        workspace=workspace,
                        event="stalled_job_detected",
                        reason=f"runner_{stalled['runner_id']}_stale",
                        metadata=stalled,
                    )

            # Circuit breaker: check restart policy before running slice
            if reliability is not None:
                can_restart, backoff_delay = reliability.should_restart_daemon()
                if not can_restart and backoff_delay is not None and backoff_delay > 0:
                    _touch_daemon_state(status="backoff", observed_state="running")
                    service._append_train_queue_daemon_history(
                        workspace=workspace,
                        event="circuit_breaker_backoff",
                        reason="daemon_backoff_active",
                        metadata={"backoff_delay_seconds": backoff_delay},
                    )
                    # Sleep backoff in small chunks to remain signal-responsive
                    remaining = backoff_delay
                    while remaining > 0 and not stop_flag["requested"]:
                        sleep_step = min(remaining, max(0.1, heartbeat_interval_seconds))
                        time.sleep(sleep_step)
                        remaining = max(0.0, remaining - sleep_step)
                        _touch_daemon_state(status="running", observed_state="running")
                    continue
                if not can_restart and backoff_delay is None:
                    exit_reason = "max_restarts_exceeded"
                    service._append_train_queue_daemon_history(
                        workspace=workspace,
                        event="circuit_breaker_tripped",
                        reason="max_restart_attempts_exceeded",
                        metadata={},
                    )
                    break

            snapshot: dict[str, Any] = {}
            runner_failed = False
            runner_exception_msg = ""
            try:
                snapshot = service.run_train_queue_worker_runner(
                    workspace=workspace,
                    max_seconds=runner_slice_seconds,
                    idle_sleep_seconds=0.0,
                )
            except Exception as exc:
                runner_failed = True
                runner_exception_msg = str(exc)
                if reliability is not None:
                    backoff = reliability.record_daemon_failure()
                    service._append_train_queue_daemon_history(
                        workspace=workspace,
                        event="runner_failure",
                        reason="run_train_queue_worker_runner_exception",
                        metadata={
                            "error": runner_exception_msg,
                            "backoff_seconds": backoff,
                            "restart_attempt": reliability.restart_policy.current_attempt,
                        },
                    )
                    # Failure cooldown sleep
                    cooldown = min(backoff, max(1.0, idle_sleep_seconds))
                    remaining = cooldown
                    while remaining > 0 and not stop_flag["requested"]:
                        sleep_step = min(remaining, max(0.1, heartbeat_interval_seconds))
                        time.sleep(sleep_step)
                        remaining = max(0.0, remaining - sleep_step)
                        _touch_daemon_state(status="running", observed_state="running")
                else:
                    service._append_train_queue_daemon_history(
                        workspace=workspace,
                        event="runner_failure",
                        reason="run_train_queue_worker_runner_exception",
                        metadata={"error": runner_exception_msg},
                    )
                continue

            action = dict(snapshot.get("auto_train_trigger_action") or {})
            processed_total += int(action.get("processed_count", 0) or 0)
            failed_total += int(action.get("failed_count", 0) or 0)

            # Record success for restart policy
            if reliability is not None and not runner_failed:
                reliability.record_daemon_success()

            _touch_daemon_state(status="running", observed_state="running")
            current = service._load_train_queue_daemon_state(workspace=workspace)
            current.update(
                {
                    "loop_cycles": loop_cycles,
                    "processed_count": processed_total,
                    "failed_count": failed_total,
                }
            )
            service._persist_train_queue_daemon_state(current, workspace=workspace)
            if stop_flag["requested"]:
                exit_reason = stop_flag["reason"]
                break
            if idle_sleep_seconds > 0:
                remaining_sleep = float(idle_sleep_seconds)
                while remaining_sleep > 0 and not stop_flag["requested"]:
                    sleep_step = min(remaining_sleep, max(0.1, heartbeat_interval_seconds))
                    time.sleep(sleep_step)
                    remaining_sleep = max(0.0, remaining_sleep - sleep_step)
                    _touch_daemon_state(status="running", observed_state="running")
        else:
            exit_reason = "loop_exit"
    finally:
        current = service._load_train_queue_daemon_state(workspace=workspace)
        current.update(
            {
                "workspace": workspace_name,
                "active": False,
                "observed_state": "stopped",
                "command_status": "completed" if exit_reason == "stop_requested" else "stopped",
                "stop_requested": True,
                "last_completed_at": datetime.now(timezone.utc).isoformat(),
                "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "lease_renewed_at": datetime.now(timezone.utc).isoformat(),
                "heartbeat_interval_seconds": heartbeat_interval_seconds,
                "lease_timeout_seconds": lease_timeout_seconds,
                "heartbeat_timeout_seconds": max(lease_timeout_seconds, heartbeat_interval_seconds * 3.0),
                "loop_cycles": loop_cycles,
                "processed_count": processed_total,
                "failed_count": failed_total,
            }
        )
        # Include final reliability state
        if reliability is not None:
            current["restart_attempts"] = reliability.restart_policy.current_attempt
            health_summary = reliability.get_health_summary()
            current["reliability_health"] = health_summary

        service._persist_train_queue_daemon_state(current, workspace=workspace)
        service._append_train_queue_daemon_history(
            workspace=workspace,
            event="completed",
            reason=exit_reason,
            metadata={
                "processed_count": processed_total,
                "failed_count": failed_total,
                "loop_cycles": loop_cycles,
                "runner_id": daemon_runner_id,
                "reliability_enabled": enable_reliability,
            },
        )

        # Cleanup reliability resources
        if reliability is not None:
            reliability.lease.cleanup_expired_leases()
            reliability.heartbeat.cleanup_stale_runners()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PFE background train queue daemon.")
    parser.add_argument("--workspace", default="user_default")
    parser.add_argument("--runner-max-seconds", type=float, default=30.0)
    parser.add_argument("--idle-sleep-seconds", type=float, default=1.0)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=2.0)
    parser.add_argument("--lease-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--max-restart-attempts", type=int, default=3)
    parser.add_argument("--restart-backoff-seconds", type=float, default=15.0)
    parser.add_argument("--disable-reliability", action="store_true", help="Disable reliability mechanisms")
    args = parser.parse_args(argv)
    return run_worker_daemon(
        workspace=args.workspace,
        runner_max_seconds=max(0.1, float(args.runner_max_seconds)),
        idle_sleep_seconds=max(0.0, float(args.idle_sleep_seconds)),
        heartbeat_interval_seconds=max(0.1, float(args.heartbeat_interval_seconds)),
        lease_timeout_seconds=max(5.0, float(args.lease_timeout_seconds)),
        max_restart_attempts=max(1, int(args.max_restart_attempts)),
        restart_backoff_seconds=max(0.0, float(args.restart_backoff_seconds)),
        enable_reliability=not args.disable_reliability,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
