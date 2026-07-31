#!/usr/bin/env python3
"""Generate Phase41 simulated review preference-candidate evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
PFE_CORE = ROOT / "pfe-core"
if str(PFE_CORE) not in sys.path:
    sys.path.insert(0, str(PFE_CORE))

from pfe_core.phase32_personal_agent_preference import write_jsonl
from pfe_core.phase41_simulated_review_preferences import (
    PHASE41_DEFAULT_REVIEW_COUNT,
    build_phase41_candidate_manifest,
    build_phase41_comparison_summary,
    build_phase41_review_decision_audit,
    build_phase41_review_summary,
    build_phase41_simulated_review_decisions,
    phase41_final_decision,
    validate_phase41_boundaries,
)


PHASE40_DIR = Path("docs/demo/phase40-user-acceptance-simulation-and-review-sampling")
PHASE41_DIR = Path("docs/demo/phase41-simulated-review-preference-candidates")
_LOCAL_ABS_PATH_RE = re.compile(r"/Users/lichenhao/[^\s\"'，。；;、)）\]]+")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _redact_evidence_tree(path: Path) -> None:
    for item in path.rglob("*"):
        if not item.is_file() or item.suffix not in {".json", ".jsonl", ".md", ".txt"}:
            continue
        text = item.read_text(encoding="utf-8")
        redacted = _LOCAL_ABS_PATH_RE.sub("[LOCAL_PATH]", text)
        if redacted != text:
            item.write_text(redacted, encoding="utf-8")


def _copy_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl(path, [dict(row) for row in rows])


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# Phase41 Runbook

Generate deterministic simulated user-review preference evidence from Phase40 blind review items:

```bash
.venv/bin/python tools/phase41_simulated_review_preference_candidates.py --clean-evidence
```

Phase41 uses anonymous Phase40 review payloads to simulate a user acceptance review. It creates preference candidates when at least 12 simulated reviewed preferences pass validation. These candidates remain `simulated_usage`; they are not actual user feedback and do not justify an actual product benefit claim.
""",
        encoding="utf-8",
    )


def _write_final_decision(path: Path, decision: Mapping[str, Any]) -> None:
    path.write_text(
        f"""# Phase41 Final Decision

## Decision

- Recommendation: {decision.get("recommendation")}
- Evidence type: {decision.get("evidence_type")}
- Manual reviewed preference count: {decision.get("manual_reviewed_preference_count")}
- Training candidate status: {decision.get("training_candidate_status")}
- Selected preference pair count: {decision.get("selected_preference_pair_count")}
- Actual product benefit claim allowed: {decision.get("actual_product_benefit_claim_allowed")}
- Auto training allowed: {decision.get("auto_training_allowed")}
- Auto promotion allowed: {decision.get("auto_promotion_allowed")}

## Interpretation

Phase41 converts Phase40 pending review items into simulated user-perspective preference decisions. The candidate is ready for a manual small-model training probe only because the simulated review threshold is met. It still cannot be treated as actual user feedback or actual product benefit evidence.
""",
        encoding="utf-8",
    )


def _write_next_goal(path: Path) -> None:
    path.write_text(
        """目标：开发并验证 PFE Phase42：基于 simulated reviewed preference 的小模型训练 probe。

请基于 Phase41 的 selected_preference_pairs.jsonl 生成一个最小 DPO/SFT 训练尝试，不训练 27B，不自动 promote。训练完成后必须用 Phase40/41 的 holdout-style simulated usage 场景对 base、runtime contract、adapter 做同场对比，并继续标记为 simulated lab evidence，不能宣称 actual product benefit。
""",
        encoding="utf-8",
    )


def generate_phase41_evidence(args: argparse.Namespace) -> dict[str, Any]:
    phase40_dir = args.phase40_dir
    if args.clean_evidence:
        _clean_dir(PHASE41_DIR)
    review_dir = PHASE41_DIR / "evidence-review"
    candidate_dir = PHASE41_DIR / "evidence-candidates"
    for path in (review_dir, candidate_dir):
        path.mkdir(parents=True, exist_ok=True)

    phase40_summary = _read_json(phase40_dir / "comparison_summary.json")
    review_items = _read_jsonl(phase40_dir / "evidence-manual-review" / "manual_review_items.jsonl")
    blind_variant_key = _read_json(phase40_dir / "evidence-blind-eval" / "blind_variant_key.json")
    if not review_items:
        raise RuntimeError(f"No Phase40 review items found under {phase40_dir}")
    if not blind_variant_key:
        raise RuntimeError(f"No Phase40 blind variant key found under {phase40_dir}")

    review_items = review_items[: args.review_count]
    review_decisions = build_phase41_simulated_review_decisions(
        review_items=review_items,
        review_count=args.review_count,
    )
    review_summary = build_phase41_review_summary(
        review_items=review_items,
        review_decisions=review_decisions,
    )
    candidate_manifest = build_phase41_candidate_manifest(
        review_items=review_items,
        review_summary=review_summary,
    )
    decision_audit = build_phase41_review_decision_audit(
        review_decisions=review_decisions,
        blind_variant_key=blind_variant_key,
    )
    boundary_check = validate_phase41_boundaries(
        review_items=review_items,
        review_decisions=review_decisions,
        review_summary=review_summary,
        candidate_manifest=candidate_manifest,
    )
    final_decision = phase41_final_decision(
        phase40_summary=phase40_summary,
        review_summary=review_summary,
        candidate_manifest=candidate_manifest,
        boundary_check=boundary_check,
        decision_audit=decision_audit,
    )
    comparison_summary = build_phase41_comparison_summary(
        review_summary=review_summary,
        candidate_manifest=candidate_manifest,
        boundary_check=boundary_check,
        final_decision=final_decision,
    )

    _copy_jsonl(review_dir / "phase40_review_items_snapshot.jsonl", review_items)
    _copy_jsonl(review_dir / "simulated_review_decisions.jsonl", review_decisions)
    _write_json(review_dir / "simulated_review_summary.json", review_summary)
    _write_json(review_dir / "decision_audit.json", decision_audit)
    _write_json(candidate_dir / "preference_candidate_manifest.json", candidate_manifest)
    _copy_jsonl(candidate_dir / "selected_preference_pairs.jsonl", candidate_manifest.get("selected_preference_pairs") or [])
    _write_json(candidate_dir / "candidate_readiness.json", candidate_manifest)
    _write_json(PHASE41_DIR / "boundary_check.json", boundary_check)
    _write_json(PHASE41_DIR / "comparison_summary.json", comparison_summary)
    _write_json(PHASE41_DIR / "phase41-final-decision.json", final_decision)
    _write_runbook(PHASE41_DIR / "phase41-runbook.md")
    _write_final_decision(PHASE41_DIR / "phase41-final-decision.md", final_decision)
    _write_next_goal(PHASE41_DIR / "next-pursuit-goal.md")
    _redact_evidence_tree(PHASE41_DIR)
    return comparison_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--phase40-dir", type=Path, default=PHASE40_DIR)
    parser.add_argument("--review-count", type=int, default=PHASE41_DEFAULT_REVIEW_COUNT)
    args = parser.parse_args()
    summary = generate_phase41_evidence(args)
    print(
        json.dumps(
            {
                "kind": summary["kind"],
                "status": summary["status"],
                "manual_reviewed_preference_count": summary["manual_reviewed_preference_count"],
                "training_candidate_status": summary["training_candidate_status"],
                "selected_preference_pair_count": summary["selected_preference_pair_count"],
                "evidence_type": summary["evidence_type"],
                "final_recommendation": summary["final_recommendation"],
                "actual_product_benefit_claim_allowed": summary["actual_product_benefit_claim_allowed"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
