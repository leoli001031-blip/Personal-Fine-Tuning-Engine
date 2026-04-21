"""Adapter lineage tracking for incremental training.

Tracks parent-child relationships between adapter versions to enable
smart parent selection, rollback decisions, and training lineage visualization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class LineageNode:
    """A node in the adapter version lineage tree."""

    version: str
    parent_version: str | None = None
    children_versions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    training_type: str = "sft"
    num_samples: int = 0
    forget_detected: bool = False
    forget_metrics: dict[str, Any] | None = None
    eval_score: float | None = None
    state: str = "training"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "children_versions": list(self.children_versions),
            "created_at": self.created_at.isoformat(),
            "training_type": self.training_type,
            "num_samples": self.num_samples,
            "forget_detected": self.forget_detected,
            "forget_metrics": self.forget_metrics,
            "eval_score": self.eval_score,
            "state": self.state,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LineageNode":
        return cls(
            version=data["version"],
            parent_version=data.get("parent_version"),
            children_versions=list(data.get("children_versions", [])),
            created_at=datetime.fromisoformat(data["created_at"]),
            training_type=data.get("training_type", "sft"),
            num_samples=data.get("num_samples", 0),
            forget_detected=data.get("forget_detected", False),
            forget_metrics=data.get("forget_metrics"),
            eval_score=data.get("eval_score"),
            state=data.get("state", "training"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class LineageDecision:
    """Result of a parent adapter selection decision."""

    selected_parent: str | None
    reason: str
    candidate_scores: list[tuple[str, float, str]] = field(default_factory=list)
    strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_parent": self.selected_parent,
            "reason": self.reason,
            "candidate_scores": [
                {"version": v, "score": s, "rationale": r}
                for v, s, r in self.candidate_scores
            ],
            "strategy": self.strategy,
        }


class AdapterLineageTracker:
    """Tracks adapter version lineage for incremental training.

    Maintains a graph of parent-child relationships between adapter versions,
    enabling smart parent selection, rollback decisions, and lineage visualization.
    """

    STRATEGIES = frozenset({"latest_eval", "most_recent", "most_stable", "largest_dataset"})

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}
        self._root_versions: set[str] = set()

    def record_training_run(
        self,
        version: str,
        parent_version: str | None,
        training_type: str = "sft",
        num_samples: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> LineageNode:
        """Record a new training run in the lineage.

        Args:
            version: The new adapter version identifier.
            parent_version: The parent version used for incremental training, or None.
            training_type: Type of training (sft, dpo, etc.).
            num_samples: Number of training samples used.
            metrics: Optional training metrics dict.
        """
        metrics = metrics or {}
        node = LineageNode(
            version=version,
            parent_version=parent_version,
            training_type=training_type,
            num_samples=num_samples,
            forget_detected=metrics.get("forget_detected", False),
            forget_metrics=metrics.get("forget_metrics"),
            eval_score=metrics.get("eval_score"),
            state=metrics.get("state", "training"),
            metadata={
                "train_loss": metrics.get("train_loss"),
                "eval_loss": metrics.get("eval_loss"),
                **{k: v for k, v in metrics.items() if k not in {"forget_detected", "forget_metrics", "eval_score", "state"}},
            },
        )
        self._nodes[version] = node

        if parent_version:
            parent = self._nodes.get(parent_version)
            if parent:
                if version not in parent.children_versions:
                    parent.children_versions.append(version)
            else:
                # Parent not yet tracked; create a placeholder
                self._nodes[parent_version] = LineageNode(
                    version=parent_version,
                    children_versions=[version],
                )
        else:
            self._root_versions.add(version)

        return node

    def update_node(self, version: str, **kwargs: Any) -> LineageNode | None:
        """Update fields on an existing lineage node."""
        node = self._nodes.get(version)
        if node is None:
            return None
        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)
        return node

    def get_node(self, version: str) -> LineageNode | None:
        """Get a lineage node by version."""
        return self._nodes.get(version)

    def get_lineage(self, version: str) -> list[LineageNode]:
        """Return the full ancestry chain from root to the given version.

        The returned list is ordered from oldest ancestor to the target version.
        """
        chain: list[LineageNode] = []
        visited: set[str] = set()
        current = version
        while current:
            if current in visited:
                break
            visited.add(current)
            node = self._nodes.get(current)
            if node is None:
                break
            chain.append(node)
            current = node.parent_version
        chain.reverse()
        return chain

    def get_children(self, version: str) -> list[LineageNode]:
        """Return all direct children of a version."""
        node = self._nodes.get(version)
        if node is None:
            return []
        return [
            self._nodes[cv] for cv in node.children_versions
            if cv in self._nodes
        ]

    def get_descendants(self, version: str) -> list[LineageNode]:
        """Return all descendants of a version (BFS traversal)."""
        descendants: list[LineageNode] = []
        visited: set[str] = set()
        queue = [version]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            node = self._nodes.get(current)
            if node is None:
                continue
            if current != version:
                descendants.append(node)
            for child in node.children_versions:
                if child not in visited:
                    queue.append(child)
        return descendants

    def find_best_parent(
        self,
        candidates: list[str],
        strategy: str = "latest_eval",
    ) -> LineageDecision:
        """Select the best parent adapter from candidates using the given strategy.

        Strategies:
        - "latest_eval": highest eval_score (falls back to most_recent if no evals).
        - "most_recent": most recent created_at.
        - "most_stable": fewest forget_detected incidents in lineage.
        - "largest_dataset": largest num_samples.

        Args:
            candidates: List of version strings to consider.
            strategy: Selection strategy name.

        Returns:
            LineageDecision with selected parent and scoring rationale.
        """
        strategy = strategy if strategy in self.STRATEGIES else "latest_eval"
        candidate_scores: list[tuple[str, float, str]] = []
        selected_parent: str | None = None
        best_score: float = -float("inf")

        for version in candidates:
            node = self._nodes.get(version)
            if node is None:
                candidate_scores.append((version, -float("inf"), "not_in_lineage"))
                continue

            if strategy == "latest_eval":
                score = node.eval_score if node.eval_score is not None else 0.0
                rationale = f"eval_score={score:.4f}" if node.eval_score is not None else "no_eval_score"
                # Tie-break with recency
                if node.eval_score is None:
                    score = node.created_at.timestamp()
                    rationale = f"fallback_recency_ts={score:.0f}"

            elif strategy == "most_recent":
                score = node.created_at.timestamp()
                rationale = f"created_at_ts={score:.0f}"

            elif strategy == "most_stable":
                lineage = self.get_lineage(version)
                forget_count = sum(1 for n in lineage if n.forget_detected)
                score = -forget_count
                rationale = f"forget_count={forget_count}"

            elif strategy == "largest_dataset":
                score = node.num_samples
                rationale = f"num_samples={node.num_samples}"

            else:
                score = 0.0
                rationale = "unknown_strategy"

            candidate_scores.append((version, score, rationale))
            if score > best_score:
                best_score = score
                selected_parent = version

        reason = (
            f"Selected {selected_parent} using strategy '{strategy}'"
            if selected_parent
            else f"No valid parent found using strategy '{strategy}'"
        )
        return LineageDecision(
            selected_parent=selected_parent,
            reason=reason,
            candidate_scores=candidate_scores,
            strategy=strategy,
        )

    def get_lineage_tree(self, version: str) -> dict[str, Any]:
        """Return a nested tree structure for visualization.

        Returns a dict with keys: version, node, children (list of child trees).
        """
        node = self._nodes.get(version)
        if node is None:
            return {"version": version, "node": None, "children": []}

        children_trees = [
            self.get_lineage_tree(child_version)
            for child_version in node.children_versions
            if child_version in self._nodes
        ]
        return {
            "version": version,
            "node": node.to_dict(),
            "children": children_trees,
        }

    def get_all_roots(self) -> list[LineageNode]:
        """Return all root versions (versions with no parent)."""
        return [self._nodes[v] for v in self._root_versions if v in self._nodes]

    def get_all_versions(self) -> list[str]:
        """Return all tracked version strings."""
        return list(self._nodes.keys())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire lineage tracker to a dictionary."""
        return {
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "root_versions": list(self._root_versions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterLineageTracker":
        """Deserialize a lineage tracker from a dictionary."""
        tracker = cls()
        for version, node_data in data.get("nodes", {}).items():
            tracker._nodes[version] = LineageNode.from_dict(node_data)
        tracker._root_versions = set(data.get("root_versions", []))
        return tracker

    def save(self, path: str | Path) -> None:
        """Save lineage tracker to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "AdapterLineageTracker":
        """Load lineage tracker from a JSON file."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# Global lineage tracker instance
_lineage_tracker: AdapterLineageTracker | None = None


def get_lineage_tracker() -> AdapterLineageTracker:
    """Get or create the global adapter lineage tracker."""
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = AdapterLineageTracker()
    return _lineage_tracker
