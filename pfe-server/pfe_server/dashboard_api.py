"""Dashboard API endpoints for Phase 2.5 observability.

This module provides REST API endpoints for the observability dashboard,
including training metrics, signal quality, and adapter performance comparison.
"""

from __future__ import annotations

from typing import Any

from .dashboard import DashboardService


class DashboardAPI:
    """Dashboard API handler."""

    def __init__(self, workspace: str = "user_default"):
        self.service = DashboardService(workspace=workspace)

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get complete dashboard data.

        Returns:
            Dictionary containing all dashboard metrics.
        """
        return self.service.get_metrics()

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
