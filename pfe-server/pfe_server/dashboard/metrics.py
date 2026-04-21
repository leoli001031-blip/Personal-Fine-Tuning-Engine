"""Metrics collection and aggregation for the observability dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TrainingMetrics:
    """Training metrics for a specific adapter version."""

    version: str
    status: str  # training, completed, failed, pending_eval
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    epochs: int = 0
    current_epoch: int = 0
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    learning_rate: float = 0.0
    steps_per_second: float = 0.0
    samples_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    gradient_norm: Optional[float] = None
    # Time series data for charts
    loss_history: list[dict[str, Any]] = field(default_factory=list)
    lr_history: list[dict[str, Any]] = field(default_factory=list)
    # Metadata
    base_model: str = ""
    training_backend: str = ""
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_minutes": self.duration_minutes,
            "epochs": self.epochs,
            "current_epoch": self.current_epoch,
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "eval_accuracy": self.eval_accuracy,
            "learning_rate": self.learning_rate,
            "steps_per_second": self.steps_per_second,
            "samples_per_second": self.samples_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "gradient_norm": self.gradient_norm,
            "loss_history": self.loss_history,
            "lr_history": self.lr_history,
            "base_model": self.base_model,
            "training_backend": self.training_backend,
            "num_samples": self.num_samples,
        }

    @property
    def duration_minutes(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 60.0
        return 0.0


@dataclass
class SignalQualityMetrics:
    """Signal quality metrics aggregated over time."""

    total_signals: int = 0
    processed_signals: int = 0
    curated_samples: int = 0
    average_confidence: float = 0.0
    # Quality distribution
    high_quality_count: int = 0  # confidence >= 0.8
    medium_quality_count: int = 0  # confidence 0.5-0.8
    low_quality_count: int = 0  # confidence < 0.5
    # Signal types distribution
    signal_type_distribution: dict[str, int] = field(default_factory=dict)
    # Time series for chart
    daily_signal_counts: list[dict[str, Any]] = field(default_factory=list)
    # Source breakdown
    source_distribution: dict[str, int] = field(default_factory=dict)
    # Quality trends
    quality_trend: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_signals": self.total_signals,
            "processed_signals": self.processed_signals,
            "curated_samples": self.curated_samples,
            "average_confidence": round(self.average_confidence, 3),
            "high_quality_count": self.high_quality_count,
            "medium_quality_count": self.medium_quality_count,
            "low_quality_count": self.low_quality_count,
            "quality_distribution": {
                "high": self.high_quality_count,
                "medium": self.medium_quality_count,
                "low": self.low_quality_count,
            },
            "signal_type_distribution": self.signal_type_distribution,
            "daily_signal_counts": self.daily_signal_counts,
            "source_distribution": self.source_distribution,
            "quality_trend": self.quality_trend,
        }


@dataclass
class AdapterComparisonMetrics:
    """Metrics for comparing adapter versions."""

    version: str
    state: str
    created_at: Optional[datetime] = None
    # Training metrics summary
    final_train_loss: Optional[float] = None
    final_eval_loss: Optional[float] = None
    final_eval_accuracy: Optional[float] = None
    style_preference_hit_rate: Optional[float] = None
    training_duration_minutes: float = 0.0
    num_training_samples: int = 0
    # Performance indicators
    is_promoted: bool = False
    is_candidate: bool = False
    # Comparison scores
    improvement_over_baseline: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "state": self.state,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "final_train_loss": self.final_train_loss,
            "final_eval_loss": self.final_eval_loss,
            "final_eval_accuracy": self.final_eval_accuracy,
            "style_preference_hit_rate": self.style_preference_hit_rate,
            "training_duration_minutes": self.training_duration_minutes,
            "num_training_samples": self.num_training_samples,
            "is_promoted": self.is_promoted,
            "is_candidate": self.is_candidate,
            "improvement_over_baseline": self.improvement_over_baseline,
        }


@dataclass
class SystemHealthMetrics:
    """System health and operational metrics."""

    # Daemon status
    daemon_active: bool = False
    daemon_state: str = "unknown"
    last_daemon_heartbeat: Optional[datetime] = None
    # Runner status
    runner_active: bool = False
    runner_state: str = "unknown"
    # Queue status
    queue_pending_jobs: int = 0
    queue_processing_jobs: int = 0
    queue_completed_jobs: int = 0
    queue_failed_jobs: int = 0
    # Recent activity
    last_training_job: Optional[str] = None
    last_training_time: Optional[datetime] = None
    # Alerts
    active_alerts: list[dict[str, Any]] = field(default_factory=list)
    alert_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "daemon_active": self.daemon_active,
            "daemon_state": self.daemon_state,
            "last_daemon_heartbeat": self.last_daemon_heartbeat.isoformat() if self.last_daemon_heartbeat else None,
            "runner_active": self.runner_active,
            "runner_state": self.runner_state,
            "queue_pending_jobs": self.queue_pending_jobs,
            "queue_processing_jobs": self.queue_processing_jobs,
            "queue_completed_jobs": self.queue_completed_jobs,
            "queue_failed_jobs": self.queue_failed_jobs,
            "last_training_job": self.last_training_job,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "active_alerts": self.active_alerts,
            "alert_count": self.alert_count,
        }


@dataclass
class DashboardMetrics:
    """Complete dashboard metrics snapshot."""

    timestamp: datetime = field(default_factory=_utc_now)
    workspace: str = "user_default"
    # Training metrics for current/latest training
    current_training: Optional[TrainingMetrics] = None
    training_history: list[TrainingMetrics] = field(default_factory=list)
    # Signal quality metrics
    signal_quality: SignalQualityMetrics = field(default_factory=SignalQualityMetrics)
    # Adapter comparison
    adapter_comparisons: list[AdapterComparisonMetrics] = field(default_factory=list)
    # System health
    system_health: SystemHealthMetrics = field(default_factory=SystemHealthMetrics)
    # Summary stats
    total_adapters: int = 0
    latest_adapter_version: Optional[str] = None
    promoted_adapter_version: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "workspace": self.workspace,
            "current_training": self.current_training.to_dict() if self.current_training else None,
            "training_history": [t.to_dict() for t in self.training_history],
            "signal_quality": self.signal_quality.to_dict(),
            "adapter_comparisons": [a.to_dict() for a in self.adapter_comparisons],
            "system_health": self.system_health.to_dict(),
            "total_adapters": self.total_adapters,
            "latest_adapter_version": self.latest_adapter_version,
            "promoted_adapter_version": self.promoted_adapter_version,
        }


class AdapterMetricsCollector:
    """Collects metrics from adapter store and training artifacts."""

    def __init__(self, home: Path | None = None, workspace: str = "user_default"):
        self.home = home or self._default_home()
        self.workspace = workspace

    def _default_home(self) -> Path:
        """Get default PFE home directory."""
        try:
            from pfe_core.storage import resolve_home

            return resolve_home()
        except Exception:
            from pfe_core.config import PFEConfig

            config = PFEConfig.load()
            return Path(str(config.home)) if hasattr(config, "home") else Path.home() / ".pfe"

    def collect_training_metrics(self, version: str) -> TrainingMetrics:
        """Collect training metrics for a specific adapter version."""
        metrics = TrainingMetrics(version=version, status="unknown")

        # Load adapter manifest
        adapter_path = self._get_adapter_path(version)
        if adapter_path and adapter_path.exists():
            manifest_path = adapter_path / "adapter_manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    metrics.status = manifest.get("state", "unknown")
                    metrics.base_model = manifest.get("base_model", "")
                    metrics.training_backend = manifest.get("training_backend", "")

                    # Parse timestamps
                    created_at = manifest.get("created_at")
                    if created_at:
                        try:
                            metrics.start_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        except (ValueError, AttributeError):
                            pass

                    # Load training metrics from manifest
                    training_metrics = manifest.get("training_metrics", {})
                    if training_metrics:
                        metrics.epochs = training_metrics.get("num_train_epochs", 0)
                        metrics.train_loss = training_metrics.get("train_loss", 0.0)
                        metrics.eval_loss = training_metrics.get("eval_loss")
                        metrics.learning_rate = training_metrics.get("learning_rate", 0.0)
                        metrics.num_samples = training_metrics.get("num_samples", 0)

                    # Load training output if available
                    training_output_path = adapter_path / "training_output.json"
                    if training_output_path.exists():
                        try:
                            output = json.loads(training_output_path.read_text(encoding="utf-8"))
                            # Extract loss history
                            log_history = output.get("log_history", [])
                            metrics.loss_history = [
                                {
                                    "step": entry.get("step", i),
                                    "epoch": entry.get("epoch", 0),
                                    "loss": entry.get("loss", 0),
                                    "learning_rate": entry.get("learning_rate", 0),
                                    "eval_loss": entry.get("eval_loss"),
                                }
                                for i, entry in enumerate(log_history)
                                if "loss" in entry or "eval_loss" in entry
                            ]

                            # Get final metrics
                            if log_history:
                                final_entry = log_history[-1]
                                metrics.train_loss = final_entry.get("loss", metrics.train_loss)
                                metrics.eval_loss = final_entry.get("eval_loss", metrics.eval_loss)
                                metrics.current_epoch = int(final_entry.get("epoch", metrics.epochs))

                        except (json.JSONDecodeError, IOError):
                            pass

                except (json.JSONDecodeError, IOError):
                    pass

        return metrics

    def collect_signal_quality_metrics(self) -> SignalQualityMetrics:
        """Collect signal quality metrics from the database."""
        metrics = SignalQualityMetrics()

        try:
            from pfe_core.db import list_signals
            from pfe_core.storage import list_samples

            # Get all signals
            signals = list_signals(home=self.home, limit=10000)
            metrics.total_signals = len(signals)

            if not signals:
                return metrics

            # Process signals
            confidences = []
            signal_types: dict[str, int] = {}
            sources: dict[str, int] = {}
            daily_counts: dict[str, int] = {}

            for signal in signals:
                # Confidence
                confidence = signal.get("confidence", 0.5)
                confidences.append(confidence)

                if confidence >= 0.8:
                    metrics.high_quality_count += 1
                elif confidence >= 0.5:
                    metrics.medium_quality_count += 1
                else:
                    metrics.low_quality_count += 1

                # Signal type
                sig_type = signal.get("signal_type", "unknown")
                signal_types[sig_type] = signal_types.get(sig_type, 0) + 1

                # Source
                source = signal.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1

                # Daily counts
                timestamp = signal.get("timestamp", "")
                if timestamp:
                    try:
                        day = timestamp[:10]  # YYYY-MM-DD
                        daily_counts[day] = daily_counts.get(day, 0) + 1
                    except (ValueError, TypeError):
                        pass

            # Calculate average confidence
            if confidences:
                metrics.average_confidence = sum(confidences) / len(confidences)

            metrics.signal_type_distribution = signal_types
            metrics.source_distribution = sources

            # Convert daily counts to sorted list
            metrics.daily_signal_counts = [
                {"date": date, "count": count}
                for date, count in sorted(daily_counts.items())
            ][-30:]  # Last 30 days

            # Get curated samples count
            samples = list_samples(home=self.home)
            signal_samples = [s for s in samples if s.get("source") == "signal"]
            metrics.curated_samples = len(signal_samples)
            metrics.processed_signals = sum(1 for s in signals if s.get("processed", False))

            # Quality trend (last 7 days)
            recent_signals = signals[-100:] if len(signals) > 100 else signals
            if recent_signals:
                recent_confidences = [s.get("confidence", 0.5) for s in recent_signals]
                metrics.quality_trend = [
                    {
                        "period": "recent",
                        "avg_confidence": sum(recent_confidences) / len(recent_confidences),
                        "count": len(recent_confidences),
                    }
                ]

        except Exception:
            # Return empty metrics on error
            pass

        return metrics

    def collect_adapter_comparisons(self) -> list[AdapterComparisonMetrics]:
        """Collect comparison metrics for all adapters."""
        comparisons = []

        try:
            from pfe_core.adapter_store import create_adapter_store

            store = create_adapter_store(workspace=self.workspace, home=str(self.home))
            versions = store.list_version_records(limit=50)

            latest_version = store.current_latest_version()

            for row in versions:
                version = row.get("version", "")
                if not version:
                    continue

                comparison = AdapterComparisonMetrics(
                    version=version,
                    state=row.get("state", "unknown"),
                    is_promoted=(version == latest_version),
                )

                # Parse created_at
                created_at = row.get("created_at")
                if created_at:
                    try:
                        comparison.created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        pass

                # Load training metrics
                metrics = row.get("metrics", {})
                if metrics:
                    comparison.final_train_loss = metrics.get("train_loss")
                    comparison.final_eval_loss = metrics.get("eval_loss")
                    comparison.final_eval_accuracy = metrics.get("eval_accuracy")
                    comparison.num_training_samples = metrics.get("num_samples", 0)

                style_preference_hit_rate = self._extract_style_preference_hit_rate(row)
                if style_preference_hit_rate is not None:
                    comparison.style_preference_hit_rate = style_preference_hit_rate

                comparisons.append(comparison)

        except Exception:
            pass

        return comparisons

    @staticmethod
    def _extract_style_preference_hit_rate(row: dict[str, Any]) -> Optional[float]:
        """Extract a stable style preference hit rate from adapter records.

        The score may appear in eval reports, manifest metadata, or the stored row
        metrics depending on which pipeline stage attached it last.
        """
        candidates: list[Any] = []

        eval_report = row.get("eval_report")
        if isinstance(eval_report, dict):
            candidates.append((eval_report.get("scores") or {}).get("style_preference_hit_rate"))
            candidates.append((eval_report.get("metadata") or {}).get("style_preference_hit_rate"))
            candidates.append((eval_report.get("eval_summary") or {}).get("style_preference_hit_rate"))

        metrics = row.get("metrics")
        if isinstance(metrics, dict):
            candidates.append(metrics.get("style_preference_hit_rate"))

        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            eval_summary = metadata.get("eval_summary")
            if isinstance(eval_summary, dict):
                candidates.append((eval_summary.get("scores") or {}).get("style_preference_hit_rate"))
                candidates.append((eval_summary.get("metadata") or {}).get("style_preference_hit_rate"))
            candidates.append(metadata.get("style_preference_hit_rate"))

        candidates.append(row.get("style_preference_hit_rate"))

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue

        return None

    def collect_system_health(self) -> SystemHealthMetrics:
        """Collect system health metrics."""
        metrics = SystemHealthMetrics()

        try:
            from pfe_core.pipeline import PipelineService

            pipeline = PipelineService()

            # Get daemon status
            daemon_status = pipeline.train_queue_daemon_status()
            metrics.daemon_active = daemon_status.get("active", False)
            metrics.daemon_state = daemon_status.get("observed_state", "unknown")

            # Get runner status
            runner_status = pipeline.train_queue_worker_runner_status()
            metrics.runner_active = runner_status.get("active", False)
            metrics.runner_state = "running" if metrics.runner_active else "idle"

            # Get queue status from pipeline status
            status = pipeline.status()
            train_queue = status.get("train_queue", {})
            metrics.queue_pending_jobs = len(train_queue.get("pending_jobs", []))
            metrics.queue_processing_jobs = len(train_queue.get("processing_jobs", []))
            metrics.queue_completed_jobs = len(train_queue.get("completed_jobs", []))
            metrics.queue_failed_jobs = len(train_queue.get("failed_jobs", []))

        except Exception:
            pass

        return metrics

    def _get_adapter_path(self, version: str) -> Path | None:
        """Get the filesystem path for an adapter version."""
        try:
            from pfe_core.adapter_store import create_adapter_store

            store = create_adapter_store(workspace=self.workspace, home=str(self.home))
            path = store.load(version)
            return Path(path) if path else None
        except Exception:
            return None

    def collect_all_metrics(self) -> DashboardMetrics:
        """Collect all dashboard metrics."""
        dashboard = DashboardMetrics(workspace=self.workspace)

        # Collect adapter comparisons
        dashboard.adapter_comparisons = self.collect_adapter_comparisons()
        dashboard.total_adapters = len(dashboard.adapter_comparisons)

        # Find latest and promoted versions
        for comp in dashboard.adapter_comparisons:
            if comp.is_promoted:
                dashboard.promoted_adapter_version = comp.version
            if not dashboard.latest_adapter_version and comp.created_at:
                dashboard.latest_adapter_version = comp.version

        # Collect current training metrics (most recent non-completed)
        for comp in reversed(dashboard.adapter_comparisons):
            if comp.state in ("training", "pending_eval"):
                dashboard.current_training = self.collect_training_metrics(comp.version)
                break

        # If no active training, get the most recent completed
        if not dashboard.current_training and dashboard.adapter_comparisons:
            latest = dashboard.adapter_comparisons[0]
            dashboard.current_training = self.collect_training_metrics(latest.version)

        # Collect training history
        dashboard.training_history = [
            self.collect_training_metrics(comp.version)
            for comp in dashboard.adapter_comparisons[:10]
        ]

        # Collect signal quality
        dashboard.signal_quality = self.collect_signal_quality_metrics()

        # Collect system health
        dashboard.system_health = self.collect_system_health()

        return dashboard
