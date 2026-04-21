"""Dashboard service providing API for observability dashboard."""

from __future__ import annotations

from typing import Any

from .metrics import AdapterMetricsCollector, DashboardMetrics


class DashboardService:
    """Service for dashboard data aggregation and metrics retrieval."""

    def __init__(self, workspace: str = "user_default"):
        self.workspace = workspace
        self._collector = AdapterMetricsCollector(workspace=workspace)

    def get_metrics(self) -> dict[str, Any]:
        """Get complete dashboard metrics."""
        metrics = self._collector.collect_all_metrics()
        return metrics.to_dict()

    def get_training_metrics(self, version: str | None = None) -> dict[str, Any]:
        """Get training metrics for a specific version or the latest.

        Args:
            version: Adapter version. If None, returns latest training metrics.

        Returns:
            Training metrics dictionary.
        """
        if version:
            metrics = self._collector.collect_training_metrics(version)
            return metrics.to_dict()

        # Get latest training metrics
        all_metrics = self._collector.collect_all_metrics()
        if all_metrics.current_training:
            return all_metrics.current_training.to_dict()
        return {}

    def get_training_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get training history for multiple adapter versions.

        Args:
            limit: Maximum number of history entries to return.

        Returns:
            List of training metrics dictionaries.
        """
        all_metrics = self._collector.collect_all_metrics()
        return [m.to_dict() for m in all_metrics.training_history[:limit]]

    def get_signal_quality(self) -> dict[str, Any]:
        """Get signal quality metrics.

        Returns:
            Signal quality metrics dictionary.
        """
        metrics = self._collector.collect_signal_quality_metrics()
        return metrics.to_dict()

    def get_adapter_comparison(self) -> list[dict[str, Any]]:
        """Get adapter performance comparison data.

        Returns:
            List of adapter comparison metrics.
        """
        comparisons = self._collector.collect_adapter_comparisons()
        return [c.to_dict() for c in comparisons]

    def get_system_health(self) -> dict[str, Any]:
        """Get system health metrics.

        Returns:
            System health metrics dictionary.
        """
        health = self._collector.collect_system_health()
        return health.to_dict()

    def get_realtime_updates(self, since: str | None = None) -> dict[str, Any]:
        """Get realtime updates for dashboard polling.

        Args:
            since: ISO timestamp for incremental updates.

        Returns:
            Dictionary with updated metrics since the given timestamp.
        """
        # For now, return full metrics
        # In production, this could filter by timestamp
        return {
            "timestamp": self._collector.collect_all_metrics().timestamp.isoformat(),
            "training": self.get_training_metrics(),
            "signal_quality": self.get_signal_quality(),
            "system_health": self.get_system_health(),
        }
