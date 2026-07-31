#!/usr/bin/env python3
"""Generate Phase31 Obsidian/Agent conversation signal-mining evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PFE_CORE = ROOT / "pfe-core"
if str(PFE_CORE) not in sys.path:
    sys.path.insert(0, str(PFE_CORE))

from pfe_core.phase31_obsidian_agent_signal_mining import (
    build_phase31_candidate_artifacts,
    build_phase31_routing_report,
    discover_phase31_sources,
    extract_phase31_signals,
    phase31_final_decision,
    write_jsonl,
)


PHASE31_DIR = Path("docs/demo/phase31-obsidian-agent-conversation-signal-mining")
PHASE30_DIR = Path("docs/demo/phase30-simulated-human-feedback-quality-loop")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_phase31_phase30_review() -> dict[str, Any]:
    decision = _read_json(PHASE30_DIR / "evidence-eval" / "decision.json")
    summary = _read_json(PHASE30_DIR / "comparison_summary.json")
    scores = _dict(_dict(summary.get("training_probe")).get("scores"))
    return {
        "kind": "phase31_phase30_review",
        "reviewed_paths": [
            str(PHASE30_DIR / "phase30-final-decision.md"),
            str(PHASE30_DIR / "comparison_summary.json"),
            str(PHASE30_DIR / "evidence-eval" / "eval_report.json"),
        ],
        "phase30_decision": decision,
        "phase30_training_scores": scores,
        "phase30_conclusion": [
            "simulated feedback quality passed",
            "Qwen2.5-0.5B DPO probe trained successfully",
            "adapter did not learn stable four-section behavior",
            "next step should use richer historical/real user signals instead of more legal-contract simulation",
        ],
        "phase31_response": [
            "mine AgentMemory/Obsidian conversations as historical user-agent collaboration signals",
            "label source as historical_user_agent_conversation, not actual_user_feedback",
            "extract user preferences, corrections, verification habits, and workflow expectations",
            "redact local paths and quarantine secret-risk conversations before candidates",
        ],
        "created_at": _utcnow_iso(),
    }


def _write_phase30_review_markdown(path: Path, review: Mapping[str, Any]) -> None:
    lines = ["# Phase31 Review Of Phase30", ""]
    lines.append("Phase30 proved simulated feedback sample quality, but not product lift.")
    lines.append("")
    lines.append("## Phase30 Conclusion")
    for item in review.get("phase30_conclusion") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Phase31 Response")
    for item in review.get("phase31_response") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Historical AgentMemory conversations are reviewable signals, not realtime actual feedback.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# Phase31 Runbook

Phase31 mines the user's Obsidian/AgentMemory conversation archive for historical collaboration signals.

It does not train by default, does not modify the vault, and does not label historical conversations as realtime `actual_user_feedback`.

## Default Smoke

```bash
.venv/bin/python tools/phase31_obsidian_agent_signal_mining.py --clean-evidence
```

## Alternate Vault

```bash
.venv/bin/python tools/phase31_obsidian_agent_signal_mining.py \\
  --vault-path [AGENT_MEMORY_VAULT] \\
  --max-conversations 80 \\
  --clean-evidence
```
""",
        encoding="utf-8",
    )


def _write_final_decision(path: Path, summary: Mapping[str, Any]) -> None:
    decision = _dict(summary.get("decision"))
    manifest = _dict(summary.get("candidate_manifest"))
    quality = _dict(summary.get("candidate_quality_report"))
    aggregate = _dict(quality.get("aggregate"))
    path.write_text(
        f"""# Phase31 Final Decision

## Decision

- Recommendation: {decision.get("recommendation")}
- Status: {decision.get("status")}
- Promotion allowed: false
- Product benefit claim allowed: false
- Actual user feedback collected: false
- Historical user-agent conversations used: true
- Training launch allowed: false

## Evidence

- Vault path: {summary.get("vault_path")}
- Discovered conversations: {_dict(summary.get("source_inventory")).get("conversation_count")}
- Selected sources: {_dict(summary.get("source_inventory")).get("selected_source_count")}
- Holdout conversations: {_dict(summary.get("holdout")).get("holdout_count")}
- Historical candidate signals: {manifest.get("historical_conversation_signal_count")}
- Approved candidate signals: {manifest.get("approved_candidate_signal_count")}
- Profile candidates: {manifest.get("profile_candidate_count")}
- Memory candidates: {manifest.get("memory_candidate_count")}
- SFT samples: {manifest.get("sft_sample_count")}
- DPO pairs: {manifest.get("dpo_pair_count")}
- Hard negatives: {manifest.get("hard_negative_pair_count")}
- Excluded signals: {manifest.get("excluded_signal_count")}

## Quality Scores

| Metric | Score |
| --- | ---: |
| source_boundary_rate | {aggregate.get("source_boundary_rate")} |
| no_secret_rate | {aggregate.get("no_secret_rate")} |
| redaction_applied_rate | {aggregate.get("redaction_applied_rate")} |
| user_preference_specificity_rate | {aggregate.get("user_preference_specificity_rate")} |
| chosen_rejected_contrast_rate | {aggregate.get("chosen_rejected_contrast_rate")} |
| profile_memory_routing_rate | {aggregate.get("profile_memory_routing_rate")} |
| not_actual_feedback_rate | {aggregate.get("not_actual_feedback_rate")} |
| holdout_isolation_rate | {aggregate.get("holdout_isolation_rate")} |
| concise_target_rate | {aggregate.get("concise_target_rate")} |

## Boundary

Phase31 turns historical Obsidian/Agent conversations into reviewable profile, memory, SFT, and DPO candidates. These records still require human review before training and cannot prove production product benefit by themselves.
""",
        encoding="utf-8",
    )


def generate_phase31_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean_evidence:
        _clean_dir(PHASE31_DIR)
    for subdir in (
        "evidence",
        "evidence-sources",
        "evidence-signals",
        "evidence-candidates",
        "evidence-eval",
    ):
        (PHASE31_DIR / subdir).mkdir(parents=True, exist_ok=True)

    evidence_dir = PHASE31_DIR / "evidence"
    source_dir = PHASE31_DIR / "evidence-sources"
    signal_dir = PHASE31_DIR / "evidence-signals"
    candidate_dir = PHASE31_DIR / "evidence-candidates"
    eval_dir = PHASE31_DIR / "evidence-eval"

    vault_path = args.vault_path.expanduser().resolve()
    vault_path_label = "[AGENT_MEMORY_VAULT]"
    phase30_review = build_phase31_phase30_review()
    inventory = discover_phase31_sources(vault_path=vault_path, max_conversations=args.max_conversations)
    extracted = extract_phase31_signals(vault_path=vault_path, source_inventory=inventory, holdout_count=args.holdout_count)
    signals = list(extracted["signals"])
    holdout = dict(extracted["holdout"])
    routing = build_phase31_routing_report(signals)
    candidates = build_phase31_candidate_artifacts(signals=signals, routing_report=routing, holdout=holdout)
    decision = phase31_final_decision(
        quality_report=candidates["candidate_quality_report"],
        candidate_manifest=candidates["candidate_manifest"],
    )

    _write_json(evidence_dir / "phase31_phase30_review.json", phase30_review)
    _write_phase30_review_markdown(evidence_dir / "phase31_phase30_review.md", phase30_review)
    _write_json(source_dir / "source_inventory.json", inventory)
    _write_json(source_dir / "source_manifest.json", {"kind": "phase31_source_manifest", "items": inventory["sources"], "created_at": _utcnow_iso()})
    write_jsonl(source_dir / "source_manifest.jsonl", inventory["sources"])
    _write_json(signal_dir / "historical_signal_batch.json", {"kind": "phase31_historical_signal_batch", "items": signals})
    write_jsonl(signal_dir / "historical_signal_batch.jsonl", signals)
    _write_json(signal_dir / "signal_routing_report.json", routing)
    _write_json(evidence_dir / "holdout.json", holdout)
    write_jsonl(candidate_dir / "selected_sft_samples.jsonl", candidates["sft_samples"])
    write_jsonl(candidate_dir / "selected_dpo_pairs.jsonl", candidates["dpo_pairs"])
    write_jsonl(candidate_dir / "selected_hard_negative_pairs.jsonl", candidates["hard_negative_pairs"])
    write_jsonl(candidate_dir / "profile_candidates.jsonl", candidates["profile_candidates"])
    write_jsonl(candidate_dir / "memory_candidates.jsonl", candidates["memory_candidates"])
    _write_json(candidate_dir / "candidate_manifest.json", candidates["candidate_manifest"])
    _write_json(candidate_dir / "candidate_quality_report.json", candidates["candidate_quality_report"])
    _write_json(candidate_dir / "holdout_integrity_check.json", candidates["holdout_integrity_check"])
    _write_json(
        eval_dir / "training_attempt.json",
        {
            "kind": "phase31_training_attempt",
            "status": "not_started",
            "training_launch_allowed": False,
            "skip_reason": "historical_candidates_require_human_review_before_training",
            "created_at": _utcnow_iso(),
        },
    )
    _write_json(eval_dir / "decision.json", decision)

    summary = {
        "kind": "phase31_obsidian_agent_conversation_signal_mining_summary",
        "status": "completed",
        "vault_path": vault_path_label,
        "phase30_review": phase30_review,
        "source_inventory": {
            "conversation_count": inventory["conversation_count"],
            "eligible_source_count": inventory["eligible_source_count"],
            "selected_source_count": inventory["selected_source_count"],
        },
        "holdout": {"holdout_count": holdout["holdout_count"]},
        "routing_report": routing,
        "candidate_manifest": candidates["candidate_manifest"],
        "candidate_quality_report": candidates["candidate_quality_report"],
        "holdout_integrity_check": candidates["holdout_integrity_check"],
        "decision": decision,
        "final_recommendation": decision["recommendation"],
        "created_at": _utcnow_iso(),
    }
    _write_json(PHASE31_DIR / "comparison_summary.json", summary)
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_runbook(PHASE31_DIR / "phase31-runbook.md")
    _write_final_decision(PHASE31_DIR / "phase31-final-decision.md", summary)
    (PHASE31_DIR / "next-pursuit-goal.md").write_text(
        "目标：对 Phase31 历史对话候选进行人工审核，批准一批真实个性化偏好训练样本，再跑小模型训练收益验证。\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Phase31 Obsidian/Agent conversation signal-mining evidence.")
    parser.add_argument("--vault-path", type=Path, default=Path("/Users/lichenhao/AgentMemory"))
    parser.add_argument("--max-conversations", type=int, default=80)
    parser.add_argument("--holdout-count", type=int, default=12)
    parser.add_argument("--clean-evidence", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = generate_phase31_evidence(args)
    compact = {
        "kind": summary.get("kind"),
        "status": summary.get("status"),
        "source_inventory": summary.get("source_inventory"),
        "candidate_manifest": summary.get("candidate_manifest"),
        "quality": _dict(summary.get("candidate_quality_report")).get("aggregate"),
        "decision": summary.get("decision"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
