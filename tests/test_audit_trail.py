"""Tests for pfe_core.security.audit_trail (AuditTrail)."""

from __future__ import annotations

import pytest

from pfe_core.security.audit_trail import AuditTrail, AuditEntry, LineageReport


class TestAuditTrailRecord:
    def test_record_basic_entry(self):
        trail = AuditTrail()
        entry = trail.record(
            actor="system",
            action="collect",
            stage="signal",
            input_refs=["user_utterance_1"],
            output_refs=["signal_1"],
            decision_reason="explicit feedback",
        )
        assert entry.actor == "system"
        assert entry.action == "collect"
        assert entry.stage == "signal"
        assert entry.input_refs == ["user_utterance_1"]
        assert entry.output_refs == ["signal_1"]
        assert entry.decision_reason == "explicit feedback"
        assert entry.timestamp

    def test_record_invalid_stage_defaults_to_signal(self):
        trail = AuditTrail()
        entry = trail.record(
            actor="system",
            action="test",
            stage="invalid_stage",
        )
        assert entry.stage == "signal"

    def test_record_with_metadata(self):
        trail = AuditTrail()
        entry = trail.record(
            actor="curator",
            action="sanitize",
            stage="sample",
            metadata={"pii_detected": True, "severity": "high"},
        )
        assert entry.metadata["pii_detected"] is True
        assert entry.metadata["severity"] == "high"

    def test_record_custom_timestamp(self):
        trail = AuditTrail()
        ts = "2026-04-18T12:00:00+00:00"
        entry = trail.record(
            actor="system",
            action="train",
            stage="train",
            timestamp=ts,
        )
        assert entry.timestamp == ts


class TestAuditTrailQuery:
    def test_get_entries_by_stage(self):
        trail = AuditTrail()
        trail.record(actor="a", action="x", stage="signal")
        trail.record(actor="a", action="y", stage="train")
        trail.record(actor="a", action="z", stage="eval")
        results = trail.get_entries(stage="train")
        assert len(results) == 1
        assert results[0].action == "y"

    def test_get_entries_by_actor(self):
        trail = AuditTrail()
        trail.record(actor="system", action="x", stage="signal")
        trail.record(actor="operator", action="y", stage="train")
        results = trail.get_entries(actor="operator")
        assert len(results) == 1
        assert results[0].action == "y"

    def test_get_entries_by_input_ref(self):
        trail = AuditTrail()
        trail.record(actor="a", action="x", stage="signal", input_refs=["ref1"])
        trail.record(actor="a", action="y", stage="sample", input_refs=["ref2"])
        results = trail.get_entries(input_ref="ref1")
        assert len(results) == 1
        assert results[0].action == "x"

    def test_get_entries_by_output_ref(self):
        trail = AuditTrail()
        trail.record(actor="a", action="x", stage="signal", output_refs=["out1"])
        trail.record(actor="a", action="y", stage="sample", output_refs=["out2"])
        results = trail.get_entries(output_ref="out2")
        assert len(results) == 1
        assert results[0].action == "y"

    def test_get_entries_combined_filters(self):
        trail = AuditTrail()
        trail.record(actor="system", action="train", stage="train", input_refs=["s1"])
        trail.record(actor="system", action="eval", stage="eval", input_refs=["s1"])
        trail.record(actor="operator", action="train", stage="train", input_refs=["s1"])
        results = trail.get_entries(stage="train", actor="system")
        assert len(results) == 1
        assert results[0].action == "train"

    def test_get_entries_no_filter_returns_all(self):
        trail = AuditTrail()
        trail.record(actor="a", action="x", stage="signal")
        trail.record(actor="b", action="y", stage="train")
        results = trail.get_entries()
        assert len(results) == 2

    def test_get_entries_no_match(self):
        trail = AuditTrail()
        trail.record(actor="a", action="x", stage="signal")
        results = trail.get_entries(stage="promote")
        assert results == []


class TestAuditTrailPreferenceLineage:
    def test_preference_lineage_no_entries(self):
        trail = AuditTrail()
        report = trail.get_preference_lineage("sig_1")
        assert report.target_id == "sig_1"
        assert report.target_type == "signal"
        assert report.entries == []
        assert "No audit trail found" in report.summary

    def test_preference_lineage_single_signal(self):
        trail = AuditTrail()
        trail.record(
            actor="system",
            action="collect",
            stage="signal",
            output_refs=["sig_1"],
            decision_reason="user explicit feedback",
        )
        report = trail.get_preference_lineage("sig_1")
        assert len(report.entries) == 1
        assert "signal" in report.summary

    def test_preference_lineage_through_train(self):
        trail = AuditTrail()
        trail.record(
            actor="system",
            action="collect",
            stage="signal",
            output_refs=["sig_1"],
        )
        trail.record(
            actor="curator",
            action="build_sample",
            stage="sample",
            input_refs=["sig_1"],
            output_refs=["sample_1"],
        )
        trail.record(
            actor="trainer",
            action="train",
            stage="train",
            input_refs=["sample_1"],
            output_refs=["run_1"],
        )
        report = trail.get_preference_lineage("sig_1")
        stages = [e.stage for e in report.entries]
        assert "signal" in stages
        assert "sample" in stages
        assert "train" in stages
        assert "promoted" in report.summary or "training" in report.summary

    def test_preference_lineage_with_conflict_recommendation(self):
        trail = AuditTrail()
        trail.record(
            actor="system",
            action="collect",
            stage="signal",
            output_refs=["sig_1"],
            metadata={"conflict_detected": True},
        )
        report = trail.get_preference_lineage("sig_1")
        assert any("conflict" in r.lower() for r in report.recommendations)

    def test_preference_lineage_no_eval_recommendation(self):
        trail = AuditTrail()
        trail.record(
            actor="system",
            action="collect",
            stage="signal",
            output_refs=["sig_1"],
        )
        report = trail.get_preference_lineage("sig_1")
        assert any("evaluation" in r.lower() for r in report.recommendations)

    def test_preference_lineage_promoted_summary(self):
        trail = AuditTrail()
        trail.record(
            actor="system", action="collect", stage="signal", output_refs=["sig_1"]
        )
        trail.record(
            actor="curator",
            action="build_sample",
            stage="sample",
            input_refs=["sig_1"],
            output_refs=["sample_1"],
        )
        trail.record(
            actor="trainer",
            action="train",
            stage="train",
            input_refs=["sample_1"],
            output_refs=["run_1"],
        )
        trail.record(
            actor="promoter",
            action="promote",
            stage="promote",
            input_refs=["run_1"],
            output_refs=["adapter_v2"],
        )
        report = trail.get_preference_lineage("sig_1")
        assert "promoted" in report.summary


class TestAuditTrailTrainingDegeneration:
    def test_degeneration_no_entries(self):
        trail = AuditTrail()
        report = trail.get_training_degeneration_report("run_1")
        assert report.degenerated is False
        assert "No audit trail found" in report.degeneration_reason

    def test_degeneration_forget_detected(self):
        trail = AuditTrail()
        trail.record(
            actor="trainer",
            action="train",
            stage="train",
            output_refs=["run_1"],
            metadata={"forget_detected": True},
            decision_reason="replay loss increased by 0.35",
        )
        report = trail.get_training_degeneration_report("run_1")
        assert report.degenerated is True
        assert "Forget detected" in report.degeneration_reason

    def test_degeneration_eval_failed(self):
        trail = AuditTrail()
        trail.record(
            actor="evaluator",
            action="evaluate",
            stage="eval",
            output_refs=["run_1"],
            metadata={"eval_failed": True},
            decision_reason="eval score below threshold",
        )
        report = trail.get_training_degeneration_report("run_1")
        assert report.degenerated is True
        assert "Eval failed" in report.degeneration_reason

    def test_degeneration_rollback_triggered(self):
        trail = AuditTrail()
        trail.record(
            actor="policy",
            action="rollback",
            stage="promote",
            output_refs=["run_1"],
            decision_reason="high confidence forget detected auto rollback",
        )
        report = trail.get_training_degeneration_report("run_1")
        assert report.degenerated is True
        assert "rollback" in report.degeneration_reason.lower()

    def test_degeneration_quality_issues(self):
        trail = AuditTrail()
        trail.record(
            actor="auditor",
            action="audit",
            stage="sample",
            output_refs=["run_1"],
            metadata={"severity": "critical", "blocked": True},
            decision_reason="critical PII exposure",
        )
        report = trail.get_training_degeneration_report("run_1")
        assert report.degenerated is True
        assert "Quality issues" in report.degeneration_reason

    def test_no_degeneration_healthy(self):
        trail = AuditTrail()
        trail.record(
            actor="trainer",
            action="train",
            stage="train",
            output_refs=["run_1"],
            metadata={"train_loss": 0.1},
        )
        trail.record(
            actor="evaluator",
            action="evaluate",
            stage="eval",
            input_refs=["run_1"],
            metadata={"eval_score": 0.92},
        )
        report = trail.get_training_degeneration_report("run_1")
        assert report.degenerated is False
        assert "No degeneration markers" in report.degeneration_reason

    def test_degeneration_recommendations(self):
        trail = AuditTrail()
        trail.record(
            actor="trainer",
            action="train",
            stage="train",
            output_refs=["run_1"],
            metadata={"forget_detected": True},
        )
        report = trail.get_training_degeneration_report("run_1")
        assert any("investigate" in r.lower() for r in report.recommendations)
        assert any("replay" in r.lower() for r in report.recommendations)


class TestAuditTrailSerialization:
    def test_to_dict_and_from_dict_roundtrip(self):
        trail = AuditTrail()
        trail.record(
            actor="system",
            action="collect",
            stage="signal",
            input_refs=["u1"],
            output_refs=["sig_1"],
            decision_reason="user feedback",
            metadata={"confidence": 0.9},
        )
        trail.record(
            actor="trainer",
            action="train",
            stage="train",
            input_refs=["sample_1"],
            output_refs=["run_1"],
        )
        data = trail.to_dict()
        restored = AuditTrail.from_dict(data)
        assert len(restored.get_entries()) == 2
        entries = restored.get_entries()
        assert entries[0].actor == "system"
        assert entries[0].metadata["confidence"] == 0.9
        assert entries[1].actor == "trainer"

    def test_from_dict_empty(self):
        trail = AuditTrail.from_dict({})
        assert trail.get_entries() == []


class TestAuditEntry:
    def test_to_dict_structure(self):
        entry = AuditEntry(
            timestamp="2026-04-18T12:00:00+00:00",
            actor="system",
            action="collect",
            stage="signal",
            input_refs=["ref1"],
            output_refs=["ref2"],
            decision_reason="test",
            metadata={"key": "value"},
        )
        d = entry.to_dict()
        assert d["actor"] == "system"
        assert d["input_refs"] == ["ref1"]
        assert d["metadata"] == {"key": "value"}


class TestLineageReport:
    def test_to_dict_structure(self):
        report = LineageReport(
            target_id="sig_1",
            target_type="signal",
            degenerated=True,
            degeneration_reason="forget detected",
            recommendations=["increase replay"],
        )
        d = report.to_dict()
        assert d["target_id"] == "sig_1"
        assert d["degenerated"] is True
        assert d["recommendations"] == ["increase replay"]
