"""Full-chain audit trail for signal, sample, train, eval, and promote stages.

Provides lineage tracking and diagnostic queries such as:
- "Why was a particular preference learned?"
- "Why did a training run degenerate?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AuditEntry:
    """A single audit entry in the full-chain trace."""

    timestamp: str
    actor: str
    action: str
    stage: str  # signal | sample | train | eval | promote
    input_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    decision_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "stage": self.stage,
            "input_refs": list(self.input_refs),
            "output_refs": list(self.output_refs),
            "decision_reason": self.decision_reason,
            "metadata": dict(self.metadata),
        }


@dataclass
class LineageReport:
    """Report tracing the full lifecycle of a signal or training run."""

    target_id: str
    target_type: str  # "signal" | "training_run"
    entries: list[AuditEntry] = field(default_factory=list)
    summary: str = ""
    degenerated: bool = False
    degeneration_reason: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_type": self.target_type,
            "entries": [e.to_dict() for e in self.entries],
            "summary": self.summary,
            "degenerated": self.degenerated,
            "degeneration_reason": self.degeneration_reason,
            "recommendations": list(self.recommendations),
        }


class AuditTrail:
    """Full-chain audit trail tracker.

    Tracks the lifecycle across stages:
    - signal   : raw feedback or user utterance
    - sample   : curated training sample
    - train    : training execution
    - eval     : evaluation run
    - promote  : adapter promotion decision

    Each entry records timestamp, actor, action, input/output references,
    and the reason for the decision.
    """

    STAGES = frozenset({"signal", "sample", "train", "eval", "promote"})

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        # Indexes for fast lookup
        self._by_input_ref: dict[str, list[int]] = {}
        self._by_output_ref: dict[str, list[int]] = {}
        self._by_stage: dict[str, list[int]] = {}
        self._by_actor: dict[str, list[int]] = {}

    def record(
        self,
        *,
        actor: str,
        action: str,
        stage: str,
        input_refs: list[str] | None = None,
        output_refs: list[str] | None = None,
        decision_reason: str = "",
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> AuditEntry:
        """Record a new audit entry.

        Args:
            actor: Who performed the action (e.g., "system", "operator", "curator").
            action: What was done (e.g., "collect", "sanitize", "train", "evaluate").
            stage: Which lifecycle stage (signal, sample, train, eval, promote).
            input_refs: IDs of inputs to this action.
            output_refs: IDs of outputs from this action.
            decision_reason: Human-readable reason for the decision.
            metadata: Additional structured metadata.
            timestamp: ISO-format timestamp; defaults to now.

        Returns:
            The created AuditEntry.
        """
        if stage not in self.STAGES:
            stage = "signal"

        entry = AuditEntry(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            actor=actor,
            action=action,
            stage=stage,
            input_refs=list(input_refs) if input_refs else [],
            output_refs=list(output_refs) if output_refs else [],
            decision_reason=decision_reason,
            metadata=dict(metadata) if metadata else {},
        )

        idx = len(self._entries)
        self._entries.append(entry)

        for ref in entry.input_refs:
            self._by_input_ref.setdefault(ref, []).append(idx)
        for ref in entry.output_refs:
            self._by_output_ref.setdefault(ref, []).append(idx)
        self._by_stage.setdefault(stage, []).append(idx)
        self._by_actor.setdefault(actor, []).append(idx)

        return entry

    def get_entries(
        self,
        *,
        stage: str | None = None,
        actor: str | None = None,
        input_ref: str | None = None,
        output_ref: str | None = None,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters."""
        indexes: set[int] | None = None

        if stage is not None:
            indexes = set(self._by_stage.get(stage, []))
        if actor is not None:
            actor_set = set(self._by_actor.get(actor, []))
            indexes = actor_set if indexes is None else indexes & actor_set
        if input_ref is not None:
            input_set = set(self._by_input_ref.get(input_ref, []))
            indexes = input_set if indexes is None else indexes & input_set
        if output_ref is not None:
            output_set = set(self._by_output_ref.get(output_ref, []))
            indexes = output_set if indexes is None else indexes & output_set

        if indexes is None:
            return list(self._entries)
        return [self._entries[i] for i in sorted(indexes)]

    def get_preference_lineage(self, signal_id: str) -> LineageReport:
        """Trace the full lifecycle of a signal to explain why a preference was learned.

        Walks forward from the signal through sample, train, eval, and promote
        stages using reference linkage.
        """
        report = LineageReport(target_id=signal_id, target_type="signal")
        visited: set[int] = set()
        queue: list[str] = [signal_id]
        visited_refs: set[str] = {signal_id}

        # BFS over reference graph
        while queue:
            current_ref = queue.pop(0)
            # Find entries where this ref is an input or output
            idxs = set(self._by_input_ref.get(current_ref, [])) | set(
                self._by_output_ref.get(current_ref, [])
            )
            for idx in sorted(idxs):
                if idx in visited:
                    continue
                visited.add(idx)
                entry = self._entries[idx]
                report.entries.append(entry)
                for ref in (*entry.input_refs, *entry.output_refs):
                    if ref not in visited_refs:
                        visited_refs.add(ref)
                        queue.append(ref)

        report.entries.sort(key=lambda e: e.timestamp)
        report.summary = self._summarize_signal_lineage(report.entries, signal_id)
        report.recommendations = self._recommendations_for_signal(report.entries)
        return report

    def get_training_degeneration_report(self, training_run_id: str) -> LineageReport:
        """Analyze why a training run degenerated.

        Looks backward from the training run to find:
        - Low-quality or conflicted samples
        - Eval failures
        - Forget detection alerts
        - Rollback decisions
        """
        report = LineageReport(target_id=training_run_id, target_type="training_run")
        visited: set[int] = set()
        queue: list[str] = [training_run_id]
        visited_refs: set[str] = {training_run_id}

        while queue:
            current_ref = queue.pop(0)
            idxs = set(self._by_input_ref.get(current_ref, [])) | set(
                self._by_output_ref.get(current_ref, [])
            )
            for idx in sorted(idxs):
                if idx in visited:
                    continue
                visited.add(idx)
                entry = self._entries[idx]
                report.entries.append(entry)
                for ref in (*entry.input_refs, *entry.output_refs):
                    if ref not in visited_refs:
                        visited_refs.add(ref)
                        queue.append(ref)

        report.entries.sort(key=lambda e: e.timestamp)
        report.degenerated, report.degeneration_reason = self._analyze_degeneration(
            report.entries, training_run_id
        )
        report.summary = self._summarize_training_lineage(report.entries, training_run_id)
        report.recommendations = self._recommendations_for_training(
            report.entries, report.degenerated
        )
        return report

    @staticmethod
    def _summarize_signal_lineage(entries: list[AuditEntry], signal_id: str) -> str:
        """Generate a human-readable summary for a signal's lineage."""
        if not entries:
            return f"No audit trail found for signal {signal_id}."

        stages_seen = sorted({e.stage for e in entries})
        train_entries = [e for e in entries if e.stage == "train"]
        promote_entries = [e for e in entries if e.stage == "promote"]

        if promote_entries:
            return (
                f"Signal {signal_id} progressed through stages: {stages_seen}. "
                f"It was promoted after training ({len(train_entries)} train record(s))."
            )
        if train_entries:
            return (
                f"Signal {signal_id} reached training ({len(train_entries)} record(s)) "
                f"but was not promoted. Stages: {stages_seen}."
            )
        return (
            f"Signal {signal_id} was processed through stages: {stages_seen}. "
            f"No training or promotion record found."
        )

    @staticmethod
    def _recommendations_for_signal(entries: list[AuditEntry]) -> list[str]:
        """Generate recommendations based on a signal's lineage."""
        recs: list[str] = []
        has_conflict = any(
            e.metadata.get("conflict_detected") for e in entries
        )
        has_quarantine = any(
            e.metadata.get("quarantine") for e in entries
        )
        if has_conflict:
            recs.append("Review conflicting signals before allowing promotion.")
        if has_quarantine:
            recs.append("Investigate quarantined samples linked to this signal.")
        if not any(e.stage == "eval" for e in entries):
            recs.append("No evaluation record found; consider running eval before promotion.")
        if not recs:
            recs.append("Signal lineage looks healthy.")
        return recs

    @staticmethod
    def _analyze_degeneration(
        entries: list[AuditEntry], training_run_id: str
    ) -> tuple[bool, str]:
        """Determine whether a training run degenerated and why."""
        if not entries:
            return False, f"No audit trail found for training run {training_run_id}."

        # Look for explicit degeneration markers
        for e in entries:
            if e.metadata.get("forget_detected"):
                return True, f"Forget detected at stage={e.stage}: {e.decision_reason}"
            if e.metadata.get("eval_failed"):
                return True, f"Eval failed at stage={e.stage}: {e.decision_reason}"
            if e.action == "rollback":
                return True, f"Auto-rollback triggered: {e.decision_reason}"
            if e.metadata.get("degeneration"):
                return True, f"Degeneration flagged at stage={e.stage}: {e.decision_reason}"

        # Look for training blockers or severe quality issues
        quality_issues = [
            e for e in entries
            if e.metadata.get("severity") in {"high", "critical"}
            or e.metadata.get("blocked")
        ]
        if quality_issues:
            return True, f"Quality issues detected in {len(quality_issues)} stage(s)."

        return False, "No degeneration markers found in audit trail."

    @staticmethod
    def _summarize_training_lineage(entries: list[AuditEntry], training_run_id: str) -> str:
        """Generate a human-readable summary for a training run's lineage."""
        if not entries:
            return f"No audit trail found for training run {training_run_id}."

        stages_seen = sorted({e.stage for e in entries})
        sample_count = len([e for e in entries if e.stage == "sample"])
        eval_count = len([e for e in entries if e.stage == "eval"])
        promote_count = len([e for e in entries if e.stage == "promote"])

        return (
            f"Training run {training_run_id} touched stages: {stages_seen}. "
            f"Samples: {sample_count}, Evals: {eval_count}, Promotions: {promote_count}."
        )

    @staticmethod
    def _recommendations_for_training(
        entries: list[AuditEntry], degenerated: bool
    ) -> list[str]:
        """Generate recommendations for a training run."""
        recs: list[str] = []
        if degenerated:
            recs.append("Investigate root cause of degeneration before next training run.")
        has_replay = any(e.metadata.get("increased_replay") for e in entries)
        if degenerated and not has_replay:
            recs.append("Consider increasing replay ratio to mitigate forgetting.")
        conflict_samples = [e for e in entries if e.metadata.get("conflict_detected")]
        if conflict_samples:
            recs.append(f"Review {len(conflict_samples)} sample(s) with detected conflicts.")
        if not any(e.stage == "eval" for e in entries):
            recs.append("Add evaluation stage to catch degeneration early.")
        if not recs:
            recs.append("Training lineage looks healthy.")
        return recs

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self._entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditTrail":
        trail = cls()
        for entry_data in data.get("entries", []):
            trail.record(
                actor=entry_data["actor"],
                action=entry_data["action"],
                stage=entry_data["stage"],
                input_refs=entry_data.get("input_refs", []),
                output_refs=entry_data.get("output_refs", []),
                decision_reason=entry_data.get("decision_reason", ""),
                metadata=entry_data.get("metadata", {}),
                timestamp=entry_data.get("timestamp"),
            )
        return trail
