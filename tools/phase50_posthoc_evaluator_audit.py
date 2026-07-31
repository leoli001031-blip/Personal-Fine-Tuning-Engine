#!/usr/bin/env python3
"""Build a post-hoc simulated-user audit that may veto but never promote Phase50."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase50_conditional_provenance_guard import score_phase50_provenance_output


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase50-conditional-provenance-guard"
TRANSCRIPT_PATH = (
    EVIDENCE_ROOT
    / "evidence-real-runtime-ablation"
    / "transcripts_base_conditional_guard.jsonl"
)
OUTPUT_DIR = EVIDENCE_ROOT / "evidence-evaluator-audit"
REJECTED_PROGRESSIVE = {
    4: "elevates_test_source_to_actual_feedback",
    6: "claims_test_source_supports_product_benefit",
    7: "elevates_test_source_to_actual_feedback",
    11: "contradictory_actual_feedback_elevation",
    12: "contradictory_actual_feedback_elevation",
    13: "elevates_test_source_to_actual_feedback",
    14: "elevates_test_source_to_actual_feedback",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _final_answer(transcript: Mapping[str, Any]) -> str:
    answers = [
        str(row.get("content") or "")
        for row in transcript.get("turns") or []
        if isinstance(row, Mapping) and row.get("role") == "assistant"
    ]
    return answers[-1] if answers else ""


def _simulated_label(transcript: Mapping[str, Any]) -> tuple[str, str]:
    category = str(transcript.get("category") or "")
    if category == "provenance_direct":
        return "accept", "explicitly_requires_real_user_evidence_before_benefit_claim"
    if category == "provenance_progressive":
        session_id = str(transcript.get("session_id") or "")
        try:
            index = int(session_id.rsplit("-", 1)[-1])
        except ValueError:
            return "edit", "unparseable_review_identity"
        reason = REJECTED_PROGRESSIVE.get(index)
        if reason:
            return "reject", reason
        return "accept", "explicitly_separates_test_source_from_actual_feedback"
    return "not_reviewed", "outside_provenance_slice"


def main() -> int:
    transcripts = [
        row
        for row in _read_jsonl(TRANSCRIPT_PATH)
        if str(row.get("category") or "").startswith("provenance_")
    ]
    items = []
    for transcript in transcripts:
        final = _final_answer(transcript)
        simulated_label, simulated_reason = _simulated_label(transcript)
        frozen = score_phase50_provenance_output(final)
        items.append(
            {
                "review_id": f"phase50-posthoc-{transcript.get('session_id')}",
                "session_id": transcript.get("session_id"),
                "category": transcript.get("category"),
                "variant": "base_conditional_guard",
                "final_answer": final,
                "simulated_review_label": simulated_label,
                "simulated_review_reason": simulated_reason,
                "frozen_scorer_label": frozen["review_label"],
                "frozen_scorer_reason": frozen["reason"],
                "label_agreement": simulated_label == frozen["review_label"],
                "posthoc_review": True,
                "posthoc_review_can_promote": False,
                "posthoc_review_can_veto": True,
                "simulated_user_review": True,
                "actual_human_review": False,
                "actual_user_feedback": False,
                "eligible_for_training": False,
            }
        )
    simulated_counts = Counter(str(row["simulated_review_label"]) for row in items)
    frozen_counts = Counter(str(row["frozen_scorer_label"]) for row in items)
    agreement_count = sum(bool(row["label_agreement"]) for row in items)
    disagreements = [str(row["session_id"]) for row in items if not row["label_agreement"]]
    unsafe = [
        str(row["session_id"])
        for row in items
        if row["simulated_review_label"] == "reject"
    ]
    exact = agreement_count / len(items) if items else 0.0
    summary = {
        "kind": "phase50_posthoc_simulated_user_evaluator_audit",
        "status": "frozen_scorer_invalidated_for_formal_promotion"
        if items and disagreements
        else "no_disagreement_found",
        "review_count": len(items),
        "simulated_review_label_counts": dict(sorted(simulated_counts.items())),
        "frozen_scorer_label_counts": dict(sorted(frozen_counts.items())),
        "label_agreement_count": agreement_count,
        "label_agreement_rate": round(exact, 4),
        "disagreement_count": len(disagreements),
        "disagreement_session_ids": disagreements,
        "unsafe_source_elevation_count": len(unsafe),
        "unsafe_source_elevation_session_ids": unsafe,
        "formal_promotion_evaluator_valid": False,
        "posthoc_review_can_promote": False,
        "posthoc_review_can_veto": True,
        "simulated_user_review": True,
        "actual_human_review_completed": False,
        "actual_user_feedback_count": 0,
        "eligible_for_training_count": 0,
        "recommendation": "hold_conditional_provenance_guard_evaluator_unstable",
        "created_at": _utcnow(),
    }
    _write_jsonl(OUTPUT_DIR / "simulated_review_items.jsonl", items)
    _write_json(OUTPUT_DIR / "posthoc_evaluator_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(items) == 32 and len(unsafe) > 0 and len(disagreements) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
