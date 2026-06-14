"""Tests for Phase 2.5 Observability Dashboard."""

from typer.testing import CliRunner
import time



class _FakeHTTPResponse:
    def __init__(self, status: int):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


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

    def test_system_health_uses_lightweight_queue_counts(self, monkeypatch):
        """System health should not call the full pipeline status aggregator."""
        from pfe_core import pipeline as pipeline_module
        from pfe_server.dashboard.metrics import AdapterMetricsCollector

        class FakePipeline:
            def train_queue_daemon_status(self):
                return {"active": True, "observed_state": "healthy"}

            def train_queue_worker_runner_status(self):
                return {"active": False}

            def _load_train_queue_state(self, *, workspace=None):
                assert workspace == "user_default"
                return {
                    "items": [
                        {"job_id": "queued-1", "state": "queued"},
                        {"job_id": "awaiting-1", "state": "awaiting_confirmation"},
                        {"job_id": "running-1", "state": "running"},
                        {"job_id": "completed-1", "state": "completed"},
                        {"job_id": "failed-1", "state": "failed"},
                    ]
                }

            def status(self, workspace=None):
                raise AssertionError("dashboard health should not call full status()")

        monkeypatch.setattr(pipeline_module, "PipelineService", FakePipeline)

        metrics = AdapterMetricsCollector(workspace="user_default").collect_system_health()

        assert metrics.daemon_active is True
        assert metrics.daemon_state == "healthy"
        assert metrics.runner_active is False
        assert metrics.queue_pending_jobs == 2
        assert metrics.queue_processing_jobs == 1
        assert metrics.queue_completed_jobs == 1
        assert metrics.queue_failed_jobs == 1


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

    def test_dashboard_data_returns_stale_snapshot_when_refresh_times_out(self, monkeypatch):
        """Dashboard metrics should stay responsive if a refresh is slow."""
        from pfe_server import dashboard_api as dashboard_api_module

        dashboard_api_module._DASHBOARD_CACHE.clear()
        dashboard_api_module._DASHBOARD_INFLIGHT.clear()
        monkeypatch.setenv("PFE_DASHBOARD_METRICS_CACHE_SECONDS", "0")
        monkeypatch.setenv("PFE_DASHBOARD_METRICS_STALE_TIMEOUT_SECONDS", "0.01")
        monkeypatch.setenv("PFE_DASHBOARD_METRICS_COLD_TIMEOUT_SECONDS", "0.5")

        class FakeService:
            def __init__(self, workspace):
                self.workspace = workspace
                self.calls = 0

            def get_metrics(self):
                self.calls += 1
                if self.calls == 1:
                    return {"workspace": self.workspace, "value": 1}
                time.sleep(0.1)
                return {"workspace": self.workspace, "value": 2}

        fake_service = FakeService("cache_test")
        monkeypatch.setattr(dashboard_api_module, "DashboardService", lambda workspace: fake_service)

        api = dashboard_api_module.DashboardAPI(workspace="cache_test")
        first = api.get_dashboard_data()
        time.sleep(0.02)
        second = api.get_dashboard_data()

        assert first["value"] == 1
        assert first["dashboard_cache"]["cached"] is False
        assert second["value"] == 1
        assert second["dashboard_cache"]["cached"] is True
        assert second["dashboard_cache"]["stale"] is True
        assert second["dashboard_cache"]["refresh_in_progress"] is True

        time.sleep(0.15)
        dashboard_api_module._DASHBOARD_CACHE.clear()
        dashboard_api_module._DASHBOARD_INFLIGHT.clear()


class TestDashboardCLI:
    """Test dashboard CLI command."""

    def test_cli_command_exists(self):
        """Test that dashboard CLI command exists."""
        from pfe_cli.main import app

        # Check that dashboard command is registered
        # The command is registered via @app.command("dashboard")
        assert app is not None

    def test_dashboard_reports_health_failure_and_start_command(self, monkeypatch):
        from pfe_cli import main as cli_main
        from pfe_cli import utility_basic_commands

        opened = []
        monkeypatch.setattr(
            utility_basic_commands,
            "_dashboard_health_check",
            lambda host, port: (False, "unavailable | refused"),
        )
        monkeypatch.setattr(utility_basic_commands.webbrowser, "open", lambda url: opened.append(url))

        result = CliRunner().invoke(cli_main.app, ["dashboard", "--port", "8921"])

        assert result.exit_code == 0
        assert "Health check: unavailable | refused" in result.stdout
        assert "Server is not reachable; browser was not opened." in result.stdout
        assert "Start server: pfe serve --port 8921 --live" in result.stdout
        assert opened == []

    def test_dashboard_opens_browser_when_health_check_passes(self, monkeypatch):
        from pfe_cli import main as cli_main
        from pfe_cli import utility_basic_commands

        opened = []
        monkeypatch.setattr(
            utility_basic_commands,
            "_dashboard_health_check",
            lambda host, port: (True, "ok | HTTP 200"),
        )
        monkeypatch.setattr(utility_basic_commands.webbrowser, "open", lambda url: opened.append(url))

        result = CliRunner().invoke(cli_main.app, ["dashboard", "--port", "8921"])

        assert result.exit_code == 0
        assert "Health check: ok | HTTP 200" in result.stdout
        assert "Opening browser..." in result.stdout
        assert opened == ["http://127.0.0.1:8921/dashboard"]

    def test_dashboard_no_open_keeps_browser_closed_but_reports_health(self, monkeypatch):
        from pfe_cli import main as cli_main
        from pfe_cli import utility_basic_commands

        opened = []
        monkeypatch.setattr(
            utility_basic_commands,
            "_dashboard_health_check",
            lambda host, port: (False, "unavailable | refused"),
        )
        monkeypatch.setattr(utility_basic_commands.webbrowser, "open", lambda url: opened.append(url))

        result = CliRunner().invoke(cli_main.app, ["dashboard", "--no-open", "--port", "8921"])

        assert result.exit_code == 0
        assert "Health check: unavailable | refused" in result.stdout
        assert "Use --open to launch browser automatically after the server is healthy." in result.stdout
        assert "Start server: pfe serve --port 8921 --live" in result.stdout
        assert opened == []

    def test_dashboard_health_check_requires_dashboard_and_metrics(self, monkeypatch):
        from pfe_cli import utility_basic_commands

        seen_paths = []

        def fake_urlopen(request, timeout=1.0):
            seen_paths.append(request.full_url)
            return _FakeHTTPResponse(200)

        monkeypatch.setattr(utility_basic_commands.urllib.request, "urlopen", fake_urlopen)

        healthy, summary = utility_basic_commands._dashboard_health_check("127.0.0.1", 8921)

        assert healthy is True
        assert seen_paths == [
            "http://127.0.0.1:8921/healthz",
            "http://127.0.0.1:8921/dashboard",
            "http://127.0.0.1:8921/pfe/dashboard/metrics",
        ]
        assert "/pfe/dashboard/metrics" in summary

    def test_dashboard_health_check_fails_when_metrics_endpoint_fails(self, monkeypatch):
        from pfe_cli import utility_basic_commands

        def fake_urlopen(request, timeout=1.0):
            if request.full_url.endswith("/pfe/dashboard/metrics"):
                return _FakeHTTPResponse(500)
            return _FakeHTTPResponse(200)

        monkeypatch.setattr(utility_basic_commands.urllib.request, "urlopen", fake_urlopen)

        healthy, summary = utility_basic_commands._dashboard_health_check("127.0.0.1", 8921)

        assert healthy is False
        assert "unhealthy" in summary
        assert "/pfe/dashboard/metrics" in summary
        assert "HTTP 500" in summary


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
        assert "class OfflineChart" in content
        assert "window.Chart = OfflineChart" in content
        assert "https://" not in content
        assert "trainingLossChart" in content
        assert "signalQualityChart" in content
        assert "adapterComparisonChart" in content
        assert "API_BASE" in content
        assert "style_preference_hit_rate" in content
        assert "Style Hit Rate" in content
