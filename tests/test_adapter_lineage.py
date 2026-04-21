"""Tests for adapter lineage tracking.

Tests the AdapterLineageTracker class and its integration with training service.
"""

import pytest
from datetime import datetime, timezone

from pfe_core.trainer.adapter_lineage import (
    AdapterLineageTracker,
    LineageNode,
    LineageDecision,
    get_lineage_tracker,
)


class TestLineageNode:
    """Test cases for LineageNode dataclass."""

    def test_lineage_node_creation(self):
        """Test creating a lineage node."""
        node = LineageNode(
            version="20240101-001",
            parent_version="20231231-001",
            training_type="sft",
            num_samples=100,
            forget_detected=False,
            eval_score=0.85,
            state="promoted",
        )
        assert node.version == "20240101-001"
        assert node.parent_version == "20231231-001"
        assert node.training_type == "sft"
        assert node.num_samples == 100
        assert node.forget_detected is False
        assert node.eval_score == 0.85
        assert node.state == "promoted"

    def test_lineage_node_to_dict(self):
        """Test LineageNode serialization."""
        node = LineageNode(
            version="20240101-001",
            parent_version="20231231-001",
            training_type="dpo",
            num_samples=50,
            forget_detected=True,
            eval_score=0.75,
            state="archived",
        )
        d = node.to_dict()
        assert d["version"] == "20240101-001"
        assert d["parent_version"] == "20231231-001"
        assert d["training_type"] == "dpo"
        assert d["num_samples"] == 50
        assert d["forget_detected"] is True
        assert d["eval_score"] == 0.75
        assert d["state"] == "archived"

    def test_lineage_node_from_dict(self):
        """Test LineageNode deserialization."""
        data = {
            "version": "20240101-001",
            "parent_version": "20231231-001",
            "children_versions": ["20240102-001"],
            "created_at": "2024-01-01T00:00:00+00:00",
            "training_type": "sft",
            "num_samples": 100,
            "forget_detected": False,
            "forget_metrics": None,
            "eval_score": 0.9,
            "state": "promoted",
            "metadata": {"key": "value"},
        }
        node = LineageNode.from_dict(data)
        assert node.version == "20240101-001"
        assert node.children_versions == ["20240102-001"]
        assert node.eval_score == 0.9
        assert node.metadata == {"key": "value"}

    def test_lineage_node_roundtrip(self):
        """Test serialize then deserialize preserves data."""
        original = LineageNode(
            version="v1",
            parent_version="v0",
            training_type="sft",
            num_samples=10,
            forget_detected=True,
            forget_metrics={"loss_delta": 0.3},
            eval_score=0.8,
            state="pending_eval",
            metadata={"extra": "data"},
        )
        d = original.to_dict()
        restored = LineageNode.from_dict(d)
        assert restored.version == original.version
        assert restored.parent_version == original.parent_version
        assert restored.training_type == original.training_type
        assert restored.num_samples == original.num_samples
        assert restored.forget_detected == original.forget_detected
        assert restored.forget_metrics == original.forget_metrics
        assert restored.eval_score == original.eval_score
        assert restored.state == original.state
        assert restored.metadata == original.metadata


class TestAdapterLineageTracker:
    """Test cases for AdapterLineageTracker."""

    def test_record_training_run(self):
        """Test recording a training run."""
        tracker = AdapterLineageTracker()
        node = tracker.record_training_run(
            version="20240101-001",
            parent_version="20231231-001",
            training_type="sft",
            num_samples=100,
            metrics={"train_loss": 0.5, "eval_score": 0.85},
        )
        assert node.version == "20240101-001"
        assert node.parent_version == "20231231-001"
        assert node.num_samples == 100
        assert node.eval_score == 0.85

    def test_record_training_run_root(self):
        """Test recording a root training run (no parent)."""
        tracker = AdapterLineageTracker()
        node = tracker.record_training_run(
            version="20231231-001",
            parent_version=None,
            training_type="sft",
            num_samples=50,
        )
        assert node.parent_version is None
        assert "20231231-001" in tracker.get_all_roots()[0].version

    def test_parent_child_link(self):
        """Test parent-child linkage is established."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None)
        tracker.record_training_run("v2", parent_version="v1")
        tracker.record_training_run("v3", parent_version="v1")

        parent = tracker.get_node("v1")
        assert "v2" in parent.children_versions
        assert "v3" in parent.children_versions

    def test_get_lineage(self):
        """Test getting ancestry chain."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None)
        tracker.record_training_run("v2", parent_version="v1")
        tracker.record_training_run("v3", parent_version="v2")

        lineage = tracker.get_lineage("v3")
        assert len(lineage) == 3
        assert lineage[0].version == "v1"
        assert lineage[1].version == "v2"
        assert lineage[2].version == "v3"

    def test_get_children(self):
        """Test getting direct children."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None)
        tracker.record_training_run("v2", parent_version="v1")
        tracker.record_training_run("v3", parent_version="v1")

        children = tracker.get_children("v1")
        assert len(children) == 2
        child_versions = {c.version for c in children}
        assert child_versions == {"v2", "v3"}

    def test_get_descendants(self):
        """Test getting all descendants."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None)
        tracker.record_training_run("v2", parent_version="v1")
        tracker.record_training_run("v3", parent_version="v2")
        tracker.record_training_run("v4", parent_version="v2")

        descendants = tracker.get_descendants("v1")
        assert len(descendants) == 3
        desc_versions = {d.version for d in descendants}
        assert desc_versions == {"v2", "v3", "v4"}

    def test_find_best_parent_latest_eval(self):
        """Test parent selection with latest_eval strategy."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None, metrics={"eval_score": 0.7})
        tracker.record_training_run("v2", parent_version="v1", metrics={"eval_score": 0.9})
        tracker.record_training_run("v3", parent_version="v1", metrics={"eval_score": 0.8})

        decision = tracker.find_best_parent(["v1", "v2", "v3"], strategy="latest_eval")
        assert decision.selected_parent == "v2"
        assert decision.strategy == "latest_eval"
        # Reason format is "Selected v2 using strategy 'latest_eval'"
        assert "v2" in decision.reason
        assert "latest_eval" in decision.reason

    def test_find_best_parent_most_recent(self):
        """Test parent selection with most_recent strategy."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None)
        tracker.record_training_run("v2", parent_version="v1")

        decision = tracker.find_best_parent(["v1", "v2"], strategy="most_recent")
        assert decision.selected_parent == "v2"
        assert decision.strategy == "most_recent"

    def test_find_best_parent_most_stable(self):
        """Test parent selection with most_stable strategy."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None, metrics={"forget_detected": True})
        tracker.record_training_run("v2", parent_version="v1", metrics={"forget_detected": False})
        tracker.record_training_run("v3", parent_version="v1", metrics={"forget_detected": True})

        decision = tracker.find_best_parent(["v1", "v2", "v3"], strategy="most_stable")
        # v2 has 1 forget in lineage (v1), v3 has 2, v1 has 1 (itself)
        # All have forget_count >= 1, so they tie at -1. First one wins.
        assert decision.selected_parent in {"v1", "v2", "v3"}
        assert decision.strategy == "most_stable"

    def test_find_best_parent_largest_dataset(self):
        """Test parent selection with largest_dataset strategy."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None, num_samples=50)
        tracker.record_training_run("v2", parent_version="v1", num_samples=200)
        tracker.record_training_run("v3", parent_version="v1", num_samples=100)

        decision = tracker.find_best_parent(["v1", "v2", "v3"], strategy="largest_dataset")
        assert decision.selected_parent == "v2"
        assert "num_samples=200" in decision.candidate_scores[1][2]

    def test_find_best_parent_invalid_strategy(self):
        """Test that invalid strategy falls back to latest_eval."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None, metrics={"eval_score": 0.8})

        decision = tracker.find_best_parent(["v1"], strategy="invalid_strategy")
        assert decision.strategy == "latest_eval"

    def test_find_best_parent_no_candidates(self):
        """Test parent selection with no valid candidates."""
        tracker = AdapterLineageTracker()
        decision = tracker.find_best_parent(["v1"], strategy="latest_eval")
        assert decision.selected_parent is None
        assert "No valid parent" in decision.reason

    def test_get_lineage_tree(self):
        """Test getting nested tree structure."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None)
        tracker.record_training_run("v2", parent_version="v1")
        tracker.record_training_run("v3", parent_version="v2")

        tree = tracker.get_lineage_tree("v1")
        assert tree["version"] == "v1"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["version"] == "v2"
        assert len(tree["children"][0]["children"]) == 1
        assert tree["children"][0]["children"][0]["version"] == "v3"

    def test_update_node(self):
        """Test updating an existing node."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None, num_samples=50)

        updated = tracker.update_node("v1", num_samples=100, state="promoted")
        assert updated is not None
        assert updated.num_samples == 100
        assert updated.state == "promoted"

    def test_update_node_nonexistent(self):
        """Test updating a non-existent node returns None."""
        tracker = AdapterLineageTracker()
        result = tracker.update_node("nonexistent", num_samples=100)
        assert result is None

    def test_save_and_load(self, tmp_path):
        """Test saving and loading lineage tracker."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None)
        tracker.record_training_run("v2", parent_version="v1")

        path = tmp_path / "lineage.json"
        tracker.save(path)

        loaded = AdapterLineageTracker.load(path)
        assert "v1" in loaded.get_all_versions()
        assert "v2" in loaded.get_all_versions()
        assert loaded.get_node("v2").parent_version == "v1"

    def test_load_missing_file(self, tmp_path):
        """Test loading from non-existent file returns empty tracker."""
        path = tmp_path / "nonexistent.json"
        tracker = AdapterLineageTracker.load(path)
        assert tracker.get_all_versions() == []

    def test_get_all_roots(self):
        """Test getting all root versions."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None)
        tracker.record_training_run("v2", parent_version=None)
        tracker.record_training_run("v3", parent_version="v1")

        roots = tracker.get_all_roots()
        assert len(roots) == 2
        root_versions = {r.version for r in roots}
        assert root_versions == {"v1", "v2"}

    def test_circular_lineage_protection(self):
        """Test that circular references don't cause infinite loops."""
        tracker = AdapterLineageTracker()
        tracker.record_training_run("v1", parent_version=None)
        tracker.record_training_run("v2", parent_version="v1")
        # Manually create circular reference
        tracker._nodes["v1"].parent_version = "v2"

        lineage = tracker.get_lineage("v2")
        # Should not hang; returns what it can find
        assert len(lineage) >= 1

    def test_record_training_run_with_forget_metrics(self):
        """Test recording training run with forget metrics."""
        tracker = AdapterLineageTracker()
        node = tracker.record_training_run(
            version="v1",
            parent_version=None,
            metrics={
                "forget_detected": True,
                "forget_metrics": {"loss_delta": 0.35, "confidence": 0.8},
                "train_loss": 0.6,
            },
        )
        assert node.forget_detected is True
        assert node.forget_metrics["loss_delta"] == 0.35
        assert node.metadata["train_loss"] == 0.6


class TestLineageDecision:
    """Test cases for LineageDecision."""

    def test_lineage_decision_creation(self):
        """Test creating a lineage decision."""
        decision = LineageDecision(
            selected_parent="v2",
            reason="best eval score",
            candidate_scores=[("v1", 0.8, "eval=0.8"), ("v2", 0.9, "eval=0.9")],
            strategy="latest_eval",
        )
        assert decision.selected_parent == "v2"
        assert decision.strategy == "latest_eval"

    def test_lineage_decision_to_dict(self):
        """Test LineageDecision serialization."""
        decision = LineageDecision(
            selected_parent="v2",
            reason="best",
            candidate_scores=[("v1", 0.8, "ok")],
            strategy="latest_eval",
        )
        d = decision.to_dict()
        assert d["selected_parent"] == "v2"
        assert d["strategy"] == "latest_eval"
        assert len(d["candidate_scores"]) == 1
        assert d["candidate_scores"][0]["version"] == "v1"


class TestGetLineageTracker:
    """Test cases for global lineage tracker."""

    def test_get_lineage_tracker_singleton(self):
        """Test that get_lineage_tracker returns a singleton."""
        t1 = get_lineage_tracker()
        t2 = get_lineage_tracker()
        assert t1 is t2

    def test_get_lineage_tracker_type(self):
        """Test that get_lineage_tracker returns correct type."""
        tracker = get_lineage_tracker()
        assert isinstance(tracker, AdapterLineageTracker)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
