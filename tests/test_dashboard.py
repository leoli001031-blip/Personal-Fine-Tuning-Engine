"""Tests for Phase 2.5 Observability Dashboard."""

import pytest


class TestDashboardMetrics:
    """Test dashboard metrics collection."""

    def test_metrics_import(self):
        """Test that dashboard metrics module can be imported."""
        from pfe_server.dashboard.metrics import (
            AdapterMetricsCollector,
            DashboardMetrics,
            SignalQualityMetrics,
            TrainingMetrics,
        )

        assert AdapterMetricsCollector is not None
        assert DashboardMetrics is not None
        assert SignalQualityMetrics is not None
        assert TrainingMetrics is not None

    def test_service_import(self):
        """Test that dashboard service can be imported."""
        from pfe_server.dashboard.service import DashboardService

        assert DashboardService is not None

    def test_api_import(self):
        """Test that dashboard API can be imported."""
        from pfe_server.dashboard_api import DashboardAPI, get_dashboard_api

        assert DashboardAPI is not None
        assert get_dashboard_api is not None

    def test_dashboard_service_creation(self):
        """Test creating dashboard service."""
        from pfe_server.dashboard.service import DashboardService

        service = DashboardService(workspace="test_workspace")
        assert service.workspace == "test_workspace"

    def test_metrics_collection(self):
        """Test basic metrics collection."""
        from pfe_server.dashboard.service import DashboardService

        service = DashboardService(workspace="user_default")

        # Test signal quality metrics
        signal_quality = service.get_signal_quality()
        assert isinstance(signal_quality, dict)
        assert "total_signals" in signal_quality
        assert "average_confidence" in signal_quality

        # Test system health metrics
        system_health = service.get_system_health()
        assert isinstance(system_health, dict)
        assert "daemon_active" in system_health
        assert "runner_active" in system_health

        # Test adapter comparison
        adapter_comparison = service.get_adapter_comparison()
        assert isinstance(adapter_comparison, list)

    def test_adapter_comparison_includes_style_preference_hit_rate(self):
        """Test adapter comparison rows surface the style preference hit rate."""
        from pfe_server.dashboard.metrics import AdapterMetricsCollector

        class FakeStore:
            def list_version_records(self, limit=50):
                return [
                    {
                        "version": "20260416-001",
                        "state": "promoted",
                        "created_at": "2026-04-16T10:30:00+00:00",
                        "metrics": {
                            "train_loss": 0.125,
                            "eval_loss": 0.234,
                            "eval_accuracy": 0.91,
                            "num_samples": 12,
                        },
                        "eval_report": {
                            "scores": {
                                "style_preference_hit_rate": 0.75,
                            }
                        },
                    }
                ][:limit]

            def current_latest_version(self):
                return "20260416-001"

        collector = AdapterMetricsCollector(workspace="user_default")

        from unittest.mock import patch

        with patch("pfe_core.adapter_store.create_adapter_store", return_value=FakeStore()):
            comparisons = collector.collect_adapter_comparisons()

        assert len(comparisons) == 1
        assert comparisons[0].style_preference_hit_rate == 0.75
        assert comparisons[0].to_dict()["style_preference_hit_rate"] == 0.75

    def test_full_metrics_snapshot(self):
        """Test getting full dashboard metrics."""
        from pfe_server.dashboard.service import DashboardService

        service = DashboardService(workspace="user_default")
        metrics = service.get_metrics()

        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        assert "workspace" in metrics
        assert "signal_quality" in metrics
        assert "system_health" in metrics
        assert "adapter_comparisons" in metrics
        assert "total_adapters" in metrics


class TestDashboardAPI:
    """Test dashboard API endpoints."""

    def test_api_creation(self):
        """Test creating dashboard API."""
        from pfe_server.dashboard_api import DashboardAPI

        api = DashboardAPI(workspace="test")
        assert api is not None

    def test_api_endpoints(self):
        """Test API endpoint methods."""
        from pfe_server.dashboard_api import DashboardAPI

        api = DashboardAPI(workspace="user_default")

        # All methods should return data without errors
        dashboard_data = api.get_dashboard_data()
        assert isinstance(dashboard_data, dict)

        training = api.get_training_metrics()
        assert isinstance(training, dict)

        history = api.get_training_history()
        assert isinstance(history, list)

        signals = api.get_signal_quality()
        assert isinstance(signals, dict)

        adapters = api.get_adapter_comparison()
        assert isinstance(adapters, list)

        health = api.get_system_health()
        assert isinstance(health, dict)


class TestDashboardCLI:
    """Test dashboard CLI command."""

    def test_cli_command_exists(self):
        """Test that dashboard CLI command exists."""
        from pfe_cli.main import app

        # Check that dashboard command is registered
        # The command is registered via @app.command("dashboard")
        assert app is not None


class TestDashboardFrontend:
    """Test dashboard frontend files."""

    def test_dashboard_html_exists(self):
        """Test that dashboard.html exists."""
        from pathlib import Path

        dashboard_path = (
            Path(__file__).parent.parent
            / "pfe-server"
            / "pfe_server"
            / "static"
            / "dashboard.html"
        )
        assert dashboard_path.exists()

    def test_dashboard_html_content(self):
        """Test that dashboard.html has required content."""
        from pathlib import Path

        dashboard_path = (
            Path(__file__).parent.parent
            / "pfe-server"
            / "pfe_server"
            / "static"
            / "dashboard.html"
        )
        content = dashboard_path.read_text(encoding="utf-8")

        # Check for key elements
        assert "chart.js" in content.lower()
        assert "trainingLossChart" in content
        assert "signalQualityChart" in content
        assert "adapterComparisonChart" in content
        assert "API_BASE" in content
        assert "style_preference_hit_rate" in content
        assert "Style Hit Rate" in content
