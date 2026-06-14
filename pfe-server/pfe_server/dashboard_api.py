"""Dashboard API endpoints for Phase 2.5 observability.

This module provides REST API endpoints for the observability dashboard,
including training metrics, signal quality, and adapter performance comparison.
"""

from __future__ import annotations

import copy
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Any

from .dashboard import DashboardService


_DASHBOARD_CACHE: dict[str, dict[str, Any]] = {}
_DASHBOARD_INFLIGHT: dict[str, Future[dict[str, Any]]] = {}
_DASHBOARD_CACHE_LOCK = threading.Lock()
_DASHBOARD_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pfe-dashboard")


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except Exception:
        return default


def _with_cache_metadata(
    payload: dict[str, Any],
    *,
    cached: bool,
    stale: bool,
    refresh_in_progress: bool,
    age_seconds: float,
    error: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result["dashboard_cache"] = {
        "cached": cached,
        "stale": stale,
        "refresh_in_progress": refresh_in_progress,
        "age_seconds": round(max(0.0, age_seconds), 3),
    }
    if error:
        result["dashboard_cache"]["refresh_error"] = error
    return result


class DashboardAPI:
    """Dashboard API handler."""

    def __init__(self, workspace: str = "user_default"):
        self.service = DashboardService(workspace=workspace)

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get complete dashboard data.

        Returns:
            Dictionary containing all dashboard metrics.
        """
        workspace = self.service.workspace
        ttl_seconds = _float_env("PFE_DASHBOARD_METRICS_CACHE_SECONDS", 10.0)
        stale_timeout = _float_env("PFE_DASHBOARD_METRICS_STALE_TIMEOUT_SECONDS", 0.25)
        cold_timeout = _float_env("PFE_DASHBOARD_METRICS_COLD_TIMEOUT_SECONDS", 4.0)
        now = time.monotonic()

        with _DASHBOARD_CACHE_LOCK:
            cached = _DASHBOARD_CACHE.get(workspace)
            future = _DASHBOARD_INFLIGHT.get(workspace)
            if future is not None and future.done():
                try:
                    payload = future.result()
                except Exception:
                    _DASHBOARD_INFLIGHT.pop(workspace, None)
                    future = None
                else:
                    cached = {"payload": copy.deepcopy(payload), "updated_at": time.monotonic()}
                    _DASHBOARD_CACHE[workspace] = cached
                    _DASHBOARD_INFLIGHT.pop(workspace, None)
                    future = None

            if cached is not None:
                age_seconds = now - float(cached["updated_at"])
                if age_seconds <= ttl_seconds:
                    return _with_cache_metadata(
                        cached["payload"],
                        cached=True,
                        stale=False,
                        refresh_in_progress=future is not None,
                        age_seconds=age_seconds,
                    )

            if future is None:
                future = _DASHBOARD_EXECUTOR.submit(self.service.get_metrics)
                _DASHBOARD_INFLIGHT[workspace] = future
            cached_snapshot = copy.deepcopy(cached) if cached is not None else None

        timeout = stale_timeout if cached_snapshot is not None else cold_timeout
        try:
            payload = future.result(timeout=timeout)
        except TimeoutError:
            if cached_snapshot is not None:
                return _with_cache_metadata(
                    cached_snapshot["payload"],
                    cached=True,
                    stale=True,
                    refresh_in_progress=True,
                    age_seconds=now - float(cached_snapshot["updated_at"]),
                )
            raise
        except Exception as exc:
            if cached_snapshot is not None:
                return _with_cache_metadata(
                    cached_snapshot["payload"],
                    cached=True,
                    stale=True,
                    refresh_in_progress=False,
                    age_seconds=now - float(cached_snapshot["updated_at"]),
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            raise

        with _DASHBOARD_CACHE_LOCK:
            _DASHBOARD_CACHE[workspace] = {"payload": copy.deepcopy(payload), "updated_at": time.monotonic()}
            if _DASHBOARD_INFLIGHT.get(workspace) is future:
                _DASHBOARD_INFLIGHT.pop(workspace, None)
        return _with_cache_metadata(
            payload,
            cached=False,
            stale=False,
            refresh_in_progress=False,
            age_seconds=0.0,
        )

    def get_training_metrics(self, version: str | None = None) -> dict[str, Any]:
        """Get training metrics.

        Args:
            version: Optional adapter version. If not provided, returns latest.

        Returns:
            Training metrics dictionary.
        """
        return self.service.get_training_metrics(version)

    def get_training_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get training history.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of training metrics.
        """
        return self.service.get_training_history(limit)

    def get_signal_quality(self) -> dict[str, Any]:
        """Get signal quality metrics.

        Returns:
            Signal quality metrics dictionary.
        """
        return self.service.get_signal_quality()

    def get_adapter_comparison(self) -> list[dict[str, Any]]:
        """Get adapter performance comparison.

        Returns:
            List of adapter comparison data.
        """
        return self.service.get_adapter_comparison()

    def get_system_health(self) -> dict[str, Any]:
        """Get system health status.

        Returns:
            System health metrics dictionary.
        """
        return self.service.get_system_health()

    def get_realtime_updates(self, since: str | None = None) -> dict[str, Any]:
        """Get realtime updates for dashboard polling.

        Args:
            since: ISO timestamp for incremental updates.

        Returns:
            Updated metrics since the given timestamp.
        """
        return self.service.get_realtime_updates(since)


# Convenience functions for direct use
def get_dashboard_api(workspace: str = "user_default") -> DashboardAPI:
    """Get dashboard API instance.

    Args:
        workspace: Workspace name.

    Returns:
        DashboardAPI instance.
    """
    return DashboardAPI(workspace=workspace)
