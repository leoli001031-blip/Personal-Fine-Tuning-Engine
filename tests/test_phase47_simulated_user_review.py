from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "pfe-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from pfe_core.phase47_simulated_user_review import (
    audit_phase47_review,
    build_phase47_decision,
    build_phase47_reviewed_candidates,
    build_phase47_simulated_review,
    review_phase47_candidate,
)


SOURCE_PATH = (
    ROOT
    / "docs/demo/phase46-runtime-first-latest-intent-ablation/evidence-curated-candidates/simulated_review_candidates.jsonl"
)


def _candidates() -> list[dict]:
    return [json.loads(line) for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_phase47_reviews_all_48_candidates_with_real_edits() -> None:
    batch = build_phase47_simulated_review(_candidates())

    assert batch["reviewed_count"] == 48
    assert batch["decision_counts"] == {"accept": 35, "edit": 13}
    assert batch["actual_human_review_count"] == 0
    assert batch["eligible_for_production_training"] is False
    assert all(row["reviewer_mode"] == "codex_simulated_real_user_perspective" for row in batch["decisions"])
    assert all(row["actual_human_review"] is False for row in batch["decisions"])


def test_phase47_edit_corrects_scope_and_false_completion() -> None:
    decisions = {row["pair_id"]: row for row in build_phase47_simulated_review(_candidates())["decisions"]}

    assert decisions["phase46-curated-002"]["decision"] == "edit"
    assert "校验结果" in decisions["phase46-curated-002"]["original_chosen"]
    assert "校验结果" not in decisions["phase46-curated-002"]["reviewed_chosen"]
    assert decisions["phase46-curated-011"]["decision"] == "edit"
    assert "完成" not in decisions["phase46-curated-011"]["reviewed_chosen"]
    assert decisions["phase46-curated-025"]["reviewed_chosen"].endswith("尚未 push。")


def test_phase47_rejects_incomplete_candidate() -> None:
    broken = dict(_candidates()[0])
    broken["messages"] = broken["messages"][:2]

    decision = review_phase47_candidate(broken)
    assert decision["decision"] == "reject"
    assert decision["reviewed_chosen"] is None
    assert decision["eligible_for_simulated_lab_candidate"] is False


def test_phase47_reviewed_pack_is_simulated_only_and_auditable() -> None:
    candidates = _candidates()
    batch = build_phase47_simulated_review(candidates)
    reviewed = build_phase47_reviewed_candidates(candidates, batch["decisions"])
    audit = audit_phase47_review(
        source_candidates=candidates,
        decisions=batch["decisions"],
        reviewed_candidates=reviewed,
    )

    assert len(reviewed) == 48
    assert audit["passed"] is True
    assert audit["edited_candidate_count"] == 13
    assert audit["eligible_for_production_training_count"] == 0
    assert all(row["eligible_for_training"] is False for row in reviewed)
    assert all(row["manual_user_review_required"] is True for row in reviewed)
    assert all(row["training_blocker"] == "pending_actual_human_confirmation" for row in reviewed)


def test_phase47_audit_rejects_false_actual_human_label() -> None:
    candidates = _candidates()
    batch = build_phase47_simulated_review(candidates)
    reviewed = build_phase47_reviewed_candidates(candidates, batch["decisions"])
    reviewed[0]["actual_human_review"] = True

    audit = audit_phase47_review(
        source_candidates=candidates,
        decisions=batch["decisions"],
        reviewed_candidates=reviewed,
    )
    assert audit["passed"] is False
    assert "simulated_actual_boundary_failed" in audit["reasons"]


def test_phase47_decision_allows_runtime_experiment_but_blocks_training() -> None:
    candidates = _candidates()
    batch = build_phase47_simulated_review(candidates)
    reviewed = build_phase47_reviewed_candidates(candidates, batch["decisions"])
    audit = audit_phase47_review(
        source_candidates=candidates,
        decisions=batch["decisions"],
        reviewed_candidates=reviewed,
    )

    decision = build_phase47_decision(audit=audit)
    assert decision["status"] == "ready_for_simulated_runtime_experiment"
    assert decision["runtime_experiment_allowed"] is True
    assert decision["training_status"] == "blocked"
    assert decision["new_training_allowed"] is False
    assert decision["hermes_attachment_allowed"] is False


def test_phase46_archive_decision_remains_unchanged() -> None:
    decision = json.loads(
        (ROOT / "docs/demo/phase46-runtime-first-latest-intent-ablation/phase46-final-decision.json").read_text(
            encoding="utf-8"
        )
    )

    assert decision["recommendation"] == "hold_runtime_and_revise_eval_or_data"
    assert decision["phase45_archived_adapter_status"] == "archive_unchanged"
    assert decision["hermes_attachment_allowed"] is False
