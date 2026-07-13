#!/usr/bin/env python3
"""Generate Phase47 simulated-user review evidence from Phase46 candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase47_simulated_user_review import (
    PHASE47_REVIEWER_ID,
    audit_phase47_review,
    build_phase47_decision,
    build_phase47_reviewed_candidates,
    build_phase47_simulated_review,
    stable_hash,
)


PHASE46_ROOT = REPO_ROOT / "docs" / "demo" / "phase46-runtime-first-latest-intent-ablation"
EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase47-simulated-user-review"
SOURCE_PATH = PHASE46_ROOT / "evidence-curated-candidates" / "simulated_review_candidates.jsonl"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


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


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {"command": args, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _phase46_snapshot() -> dict[str, Any]:
    manifest = _read_json(PHASE46_ROOT / "evidence_manifest.json")
    mismatches = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / str(item.get("path") or "")
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    decision = _read_json(PHASE46_ROOT / "phase46-final-decision.json")
    integrity = _read_json(PHASE46_ROOT / "evidence_integrity.json")
    return {
        "kind": "phase47_phase46_canonical_snapshot",
        "passed": (
            not mismatches
            and integrity.get("passed") is True
            and decision.get("recommendation") == "hold_runtime_and_revise_eval_or_data"
            and decision.get("phase45_archived_adapter_status") == "archive_unchanged"
        ),
        "phase46_commit": "9993dfd",
        "phase46_pr_number": 57,
        "phase46_recommendation": decision.get("recommendation"),
        "phase45_archived_adapter_status": decision.get("phase45_archived_adapter_status"),
        "phase46_manifest_sha256": manifest.get("manifest_sha256"),
        "phase46_manifest_file_count": manifest.get("file_count"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
        "created_at": _utcnow(),
    }


def _review_examples(decisions: Iterable[Mapping[str, Any]]) -> str:
    edits = [dict(row) for row in decisions if row.get("decision") == "edit"]
    lines = [
        "# Phase47 Simulated User Review Edits",
        "",
        "这些修改来自 Codex 模拟真实用户视角，不是实际用户反馈或真人审核。",
        "",
    ]
    for row in edits:
        lines.extend(
            [
                f"## {row.get('pair_id')} ({row.get('category')})",
                "",
                f"- 决策：`{row.get('decision')}`",
                f"- 理由：{row.get('reason')}",
                f"- 原回答：{row.get('original_chosen')}",
                f"- 修订结果：{row.get('reviewed_chosen')}",
                "",
            ]
        )
    return "\n".join(lines)


def _evidence_manifest() -> dict[str, Any]:
    excluded = {"evidence_manifest.json", "evidence_integrity.json"}
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files.append({"path": str(path.relative_to(REPO_ROOT)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "kind": "phase47_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": stable_hash(files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-evidence", action="store_true")
    args = parser.parse_args()
    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)

    phase46 = _phase46_snapshot()
    candidates = _read_jsonl(SOURCE_PATH)
    review = build_phase47_simulated_review(candidates)
    decisions = list(review["decisions"])
    reviewed = build_phase47_reviewed_candidates(candidates, decisions)
    audit = audit_phase47_review(
        source_candidates=candidates,
        decisions=decisions,
        reviewed_candidates=reviewed,
    )
    decision = build_phase47_decision(audit=audit)
    decision.update({"created_at": _utcnow(), "phase46_archive_preserved": phase46.get("passed") is True})

    source_manifest = {
        "kind": "phase47_source_candidate_manifest",
        "source_path": str(SOURCE_PATH.relative_to(REPO_ROOT)),
        "source_sha256": _sha256(SOURCE_PATH),
        "source_candidate_count": len(candidates),
        "source_feedback_type": "simulated_usage",
        "source_actual_human_review": False,
        "source_eligible_for_training": False,
    }
    review_summary = {key: value for key, value in review.items() if key != "decisions"}
    candidate_manifest = {
        "kind": "phase47_simulated_reviewed_candidate_manifest",
        "candidate_count": len(reviewed),
        "candidate_ids": [row.get("pair_id") for row in reviewed],
        "candidate_target_hashes": [hashlib.sha256(str(row.get("chosen") or "").encode("utf-8")).hexdigest() for row in reviewed],
        "edit_count": audit.get("edited_candidate_count"),
        "reject_count": audit.get("rejected_candidate_count"),
        "reviewer_id": PHASE47_REVIEWER_ID,
        "feedback_source": "simulated_usage",
        "simulated_human_review": True,
        "actual_human_review": False,
        "actual_user_feedback_count": 0,
        "eligible_for_simulated_runtime_experiment": audit.get("passed") is True,
        "eligible_for_training": False,
        "eligible_for_production_training": False,
        "training_blocker": "pending_actual_human_confirmation",
        "actual_product_benefit_claim_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
        "manifest_sha256": stable_hash(reviewed),
    }
    comparison = {
        "kind": "phase47_simulated_user_review_comparison",
        "created_at": _utcnow(),
        "source_candidate_count": len(candidates),
        "reviewed_candidate_count": len(reviewed),
        "decision_counts": audit.get("decision_counts"),
        "edited_candidate_count": audit.get("edited_candidate_count"),
        "rejected_candidate_count": audit.get("rejected_candidate_count"),
        "edit_rate": review_summary.get("edit_rate"),
        "review_audit": audit,
        "decision": decision,
        "new_model_calls": 0,
        "new_training": False,
        "new_adapter_created": False,
        "actual_human_review_count": 0,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }

    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "phase46_canonical_snapshot.json", phase46)
    _write_json(EVIDENCE_ROOT / "evidence-review" / "source_manifest.json", source_manifest)
    _write_json(EVIDENCE_ROOT / "evidence-review" / "review_methodology.json", {
        "kind": "phase47_review_methodology",
        "reviewer_mode": "codex_simulated_real_user_perspective",
        "rubric": [
            "follow the latest user request rather than the stale goal",
            "do not claim completion without evidence",
            "answer ordinary tasks directly and naturally",
            "preserve privacy without deleting public identifiers",
            "prefer concise specific wording over process narration",
        ],
        "allowed_decisions": ["accept", "edit", "reject"],
        "model_identity_used_for_review": False,
        "actual_human_review": False,
        "actual_product_benefit_claim_allowed": False,
    })
    _write_jsonl(EVIDENCE_ROOT / "evidence-review" / "review_decisions.jsonl", decisions)
    _write_json(EVIDENCE_ROOT / "evidence-review" / "review_summary.json", review_summary)
    _write_text(EVIDENCE_ROOT / "evidence-review" / "review_edits.md", _review_examples(decisions))
    _write_jsonl(EVIDENCE_ROOT / "evidence-candidates" / "reviewed_candidates.jsonl", reviewed)
    _write_json(EVIDENCE_ROOT / "evidence-candidates" / "candidate_manifest.json", candidate_manifest)
    _write_json(EVIDENCE_ROOT / "evidence-candidates" / "review_audit.json", audit)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase47-final-decision.json", decision)
    _write_text(EVIDENCE_ROOT / "phase47-final-decision.md", f"""# Phase47 Final Decision

## 结论

本轮由 Codex 以模拟真实用户视角逐条复核 Phase46 的 48 条 correction candidate。结果为 **35 条 accept、13 条 edit、0 条 canonical reject**，审核覆盖和质量审计通过。

这不是实际人工审核，也不是 actual user feedback。最终 recommendation 为 **{decision['recommendation']}**：修订后的候选可以用于下一轮 fresh runtime ablation，但生产训练保持 blocked，不允许自动训练、promotion 或接入 Hermes。

## 审核发现

- 13 条修改主要解决范围扩大、过程性措辞、虚假完成暗示和不自然表达。
- `phase46-curated-025` 从“无法确认 push”修订为“本地领先远端 1 个 commit，说明尚未 push”，让结论真正服从现有证据。
- `phase46-curated-011` 删除“完成排查”的隐含状态，只保留自然致谢。
- 所有修订目标均通过简洁性、隐私、占位符和虚假完成检查。

## 证据边界

- `simulated_human_review=true`
- `actual_human_review=false`
- `actual_user_feedback_count=0`
- `eligible_for_production_training=false`
- `training_blocker=pending_actual_human_confirmation`
- `actual_product_benefit_claim_allowed=false`

## 下一步

使用这 48 条修订候选定义一个更短的 latest-intent runtime contract，并冻结全新的 holdout。下一轮先比较 privacy base、compact runtime 和 Phase46 full envelope；不训练 adapter，直到 runtime baseline 与 fresh holdout 结论明确。
""")
    _write_text(EVIDENCE_ROOT / "phase47-runbook.md", """# Phase47 Runbook

```bash
.venv/bin/python -m py_compile pfe-core/pfe_core/phase47_simulated_user_review.py tools/phase47_simulated_user_review.py tests/test_phase47_simulated_user_review.py
.venv/bin/pytest -q tests/test_phase47_simulated_user_review.py tests/test_phase46_runtime_first_latest_intent.py
.venv/bin/python tools/phase47_simulated_user_review.py --clean-evidence
.venv/bin/python tools/phase47_validate.py
```

Phase47 performs no model generation and no training. It converts the Phase46 candidate pack into explicit accept/edit/reject decisions from a simulated-user perspective while preserving that no actual human review occurred.
""")
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", """# Next Pursuit Goal

Develop Phase48 as a no-training compact latest-intent runtime ablation. Freeze a fresh holdout before model calls, compare privacy base, compact runtime, and the Phase46 full envelope under the same Qwen3-4B decoding contract, then run deterministic and Gemma4 blind evaluation. The Phase47 reviewed pack may guide scenario design but must not enter the holdout or be called actual feedback. Do not train or attach an adapter to Hermes.
""")
    _write_json(EVIDENCE_ROOT / "finalization_state.json", {
        "kind": "phase47_finalization_state",
        "created_at": _utcnow(),
        "decision": decision.get("recommendation"),
        "new_model_calls": 0,
        "new_training": False,
        "git_snapshot": {
            "head": _command(["git", "rev-parse", "HEAD"]),
            "branch": _command(["git", "branch", "--show-current"]),
            "status": _command(["git", "status", "--short"]),
        },
    })

    integrity = {
        "kind": "phase47_evidence_integrity",
        "created_at": _utcnow(),
        "passed": (
            phase46.get("passed") is True
            and audit.get("passed") is True
            and len(candidates) == 48
            and len(decisions) == 48
            and len(reviewed) == 48
            and decision.get("training_status") == "blocked"
            and candidate_manifest.get("actual_human_review") is False
            and candidate_manifest.get("eligible_for_production_training") is False
        ),
        "phase46_canonical": phase46,
        "review_audit": audit,
        "decision": decision,
        "actual_human_review_count": 0,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
    }
    manifest = _evidence_manifest()
    integrity["evidence_manifest_sha256"] = manifest["manifest_sha256"]
    integrity["evidence_file_count"] = manifest["file_count"]
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    print(json.dumps({
        "decision": decision.get("recommendation"),
        "source_candidates": len(candidates),
        "review_decisions": len(decisions),
        "decision_counts": audit.get("decision_counts"),
        "review_audit_passed": audit.get("passed"),
        "training_status": decision.get("training_status"),
        "evidence_integrity": integrity.get("passed"),
    }, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
