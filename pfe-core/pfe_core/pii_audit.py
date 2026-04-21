"""PII auditing and compliance reporting.

Tracks PII detection and anonymization for compliance auditing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .pii_detector import PIIDetectionResult, PIIType
from .storage import resolve_home


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass
class PIIAuditEntry:
    """Single audit entry for PII detection/anonymization."""
    timestamp: datetime
    source_type: str  # "signal", "sample", "training_data"
    source_id: str
    has_pii: bool
    pii_types: list[str] = field(default_factory=list)
    finding_count: int = 0
    action_taken: str = "detected"  # detected, anonymized, blocked
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "source_type": self.source_type,
            "source_id": self.source_id,
            "has_pii": self.has_pii,
            "pii_types": self.pii_types,
            "finding_count": self.finding_count,
            "action_taken": self.action_taken,
            "success": self.success,
        }


@dataclass
class PIIComplianceReport:
    """Compliance report for PII processing."""
    period_start: datetime
    period_end: datetime
    total_scanned: int = 0
    pii_detected_count: int = 0
    pii_anonymized_count: int = 0
    pii_blocked_count: int = 0
    pii_type_distribution: dict[str, int] = field(default_factory=dict)
    source_type_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_scanned": self.total_scanned,
            "pii_detected_count": self.pii_detected_count,
            "pii_anonymized_count": self.pii_anonymized_count,
            "pii_blocked_count": self.pii_blocked_count,
            "pii_type_distribution": self.pii_type_distribution,
            "source_type_distribution": self.source_type_distribution,
        }


class PIIAuditLog:
    """Audit log for PII operations."""

    def __init__(self, log_dir: Path | str | None = None):
        """Initialize audit log.

        Args:
            log_dir: Directory for audit logs, defaults to $PFE_HOME/audit
        """
        if log_dir is None:
            log_dir = resolve_home() / "audit"
        elif isinstance(log_dir, str):
            log_dir = Path(log_dir).expanduser()
        else:
            log_dir = log_dir.expanduser()
        self.log_dir = log_dir
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._entries: list[PIIAuditEntry] = []
        self._current_log_file = self._get_log_file()

    def _get_log_file(self) -> Path:
        """Get current log file path based on date."""
        today = _utc_now().strftime("%Y-%m")
        return self.log_dir / f"pii_audit_{today}.jsonl"

    def log_detection(
        self,
        source_type: str,
        source_id: str,
        detection_result: PIIDetectionResult,
        action_taken: str = "detected",
    ) -> None:
        """Log a PII detection event.

        Args:
            source_type: Type of source (signal, sample, etc.)
            source_id: Unique identifier for the source
            detection_result: Detection result
            action_taken: What action was taken
        """
        entry = PIIAuditEntry(
            timestamp=_utc_now(),
            source_type=source_type,
            source_id=source_id,
            has_pii=detection_result.has_pii,
            pii_types=[t.value for t in detection_result.pii_types_found],
            finding_count=len(detection_result.findings),
            action_taken=action_taken,
        )
        try:
            entry.success = self._append_to_file(entry)
        except OSError:
            entry.success = False
        self._entries.append(entry)

    def _append_to_file(self, entry: PIIAuditEntry) -> bool:
        """Append entry to log file."""
        # Check if we need to rotate to a new month
        current_file = self._get_log_file()
        try:
            current_file.parent.mkdir(parents=True, exist_ok=True)
            with open(current_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            return False
        return True

    def get_report(
        self,
        days: int = 30,
        source_type: str | None = None,
    ) -> PIIComplianceReport:
        """Generate compliance report for the specified period.

        Args:
            days: Number of days to include
            source_type: Filter by source type, or None for all

        Returns:
            Compliance report
        """
        end_date = _utc_now()
        start_date = end_date - timedelta(days=days)

        report = PIIComplianceReport(
            period_start=start_date,
            period_end=end_date,
        )

        # Load entries from log files
        entries = self._load_entries(start_date, end_date, source_type)

        for entry in entries:
            report.total_scanned += 1

            if entry.has_pii:
                report.pii_detected_count += 1

            if entry.action_taken == "anonymized":
                report.pii_anonymized_count += 1
            elif entry.action_taken == "blocked":
                report.pii_blocked_count += 1

            # Count PII types
            for pii_type in entry.pii_types:
                report.pii_type_distribution[pii_type] = (
                    report.pii_type_distribution.get(pii_type, 0) + 1
                )

            # Count source types
            report.source_type_distribution[entry.source_type] = (
                report.source_type_distribution.get(entry.source_type, 0) + 1
            )

        return report

    def _load_entries(
        self,
        start_date: datetime,
        end_date: datetime,
        source_type: str | None = None,
    ) -> list[PIIAuditEntry]:
        """Load entries from log files within date range."""
        entries = []

        # Get relevant log files
        current = start_date.replace(day=1)
        while current <= end_date:
            log_file = self.log_dir / f"pii_audit_{current.strftime('%Y-%m')}.jsonl"
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            entry_date = _normalize_utc(datetime.fromisoformat(data["timestamp"]))

                            # Check date range
                            if not (start_date <= entry_date <= end_date):
                                continue

                            # Check source type filter
                            if source_type and data["source_type"] != source_type:
                                continue

                            entry = PIIAuditEntry(
                                timestamp=entry_date,
                                source_type=data["source_type"],
                                source_id=data["source_id"],
                                has_pii=data["has_pii"],
                                pii_types=data.get("pii_types", []),
                                finding_count=data.get("finding_count", 0),
                                action_taken=data["action_taken"],
                                success=data.get("success", True),
                            )
                            entries.append(entry)
                        except (json.JSONDecodeError, KeyError):
                            continue

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return entries

    def export_report(
        self,
        output_path: Path,
        days: int = 30,
    ) -> None:
        """Export compliance report to file.

        Args:
            output_path: Path to write report
            days: Number of days to include
        """
        report = self.get_report(days=days)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)


class PIIWhitelist:
    """Whitelist for allowed PII patterns (e.g., test data)."""

    def __init__(self, whitelist_path: Path | None = None):
        """Initialize whitelist.

        Args:
            whitelist_path: Path to whitelist file
        """
        if whitelist_path is None:
            whitelist_path = resolve_home() / "pii_whitelist.json"
        else:
            whitelist_path = whitelist_path.expanduser()
        self.whitelist_path = whitelist_path
        self._patterns: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load whitelist from file."""
        if self.whitelist_path.exists():
            try:
                with open(self.whitelist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._patterns = set(data.get("patterns", []))
            except (json.JSONDecodeError, IOError):
                self._patterns = set()

    def save(self) -> None:
        """Save whitelist to file."""
        self.whitelist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.whitelist_path, "w", encoding="utf-8") as f:
            json.dump(
                {"patterns": sorted(self._patterns)},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def add(self, pattern: str) -> None:
        """Add a pattern to whitelist."""
        self._patterns.add(pattern)
        self.save()

    def remove(self, pattern: str) -> None:
        """Remove a pattern from whitelist."""
        self._patterns.discard(pattern)
        self.save()

    def is_whitelisted(self, value: str) -> bool:
        """Check if a value is whitelisted."""
        return value in self._patterns

    def list_patterns(self) -> list[str]:
        """List all whitelisted patterns."""
        return sorted(self._patterns)
