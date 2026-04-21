"""PFE Observability Dashboard module.

Provides data aggregation and metrics collection for the Phase 2.5
observability dashboard, including training metrics, signal quality,
and adapter performance comparison.
"""

from .metrics import (
    AdapterComparisonMetrics,
    AdapterMetricsCollector,
    DashboardMetrics,
    SignalQualityMetrics,
    SystemHealthMetrics,
    TrainingMetrics,
)
from .service import DashboardService

__all__ = [
    "AdapterComparisonMetrics",
    "AdapterMetricsCollector",
    "DashboardMetrics",
    "DashboardService",
    "SignalQualityMetrics",
    "SystemHealthMetrics",
    "TrainingMetrics",
]
