#!/usr/bin/env python3
"""Generate Phase40 user-acceptance simulation and review-sampling evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PFE_CORE = ROOT / "pfe-core"
if str(PFE_CORE) not in sys.path:
    sys.path.insert(0, str(PFE_CORE))

from pfe_core.phase40_user_acceptance_simulation import (
    PHASE40_MODEL_VARIANTS,
    build_phase40_blind_eval_pairs,
    build_phase40_comparison_summary,
    build_phase40_manual_review_items,
    build_phase40_manual_review_summary,
    build_phase40_phase39_recap,
    build_phase40_preference_candidate_manifest,
    build_phase40_scenario_bank,
    build_phase40_transcripts,
    build_phase40_user_acceptance_scores,
    phase40_final_decision,
    validate_phase40_boundaries,
    validate_phase40_scenario_bank,
    validate_phase40_transcript_structure,
    write_jsonl,
)


PHASE36_39_DIR = Path("docs/demo/phase36-39-local-feedback-product-benefit-loop")
PHASE40_DIR = Path("docs/demo/phase40-user-acceptance-simulation-and-review-sampling")
_LOCAL_ABS_PATH_RE = re.compile(r"/Users/lichenhao/[^\s\"'，。；;、)）\]]+")


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, Mapping):
            rows.append(dict(item))
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


def _load_review_decisions(path: Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload = _read_json(path)
    if isinstance(payload.get("items"), list):
        return [dict(item) for item in payload["items"] if isinstance(item, Mapping)]
    rows = _read_jsonl(path)
    if rows:
        return rows
    if isinstance(payload, Mapping) and payload:
        return [dict(payload)]
    return []


def _write_output_examples(path: Path, transcripts_by_variant: Mapping[str, list[Mapping[str, Any]]]) -> None:
    lines = ["# Phase40 Output Examples", ""]
    for variant in PHASE40_MODEL_VARIANTS:
        transcript = (transcripts_by_variant.get(variant) or [])[0]
        lines.append(f"## {variant}")
        for turn in transcript.get("turns") or []:
            item = _dict(turn)
            if item.get("role") == "assistant":
                lines.append(str(item.get("content") or ""))
                break
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# Phase40 Runbook

Generate deterministic Phase40 user-acceptance simulation and review-sampling evidence:

```bash
.venv/bin/python tools/phase40_user_acceptance_simulation_and_review_sampling.py --clean-evidence
```

Default output contains simulated usage scenarios and pending manual review items only. It does not connect Hermes, train 27B, auto-promote, or claim actual product benefit.

To test reviewed-preference readiness, pass a JSON or JSONL file with explicit human review decisions:

```bash
.venv/bin/python tools/phase40_user_acceptance_simulation_and_review_sampling.py --review-decisions-json path/to/decisions.jsonl
```
""",
        encoding="utf-8",
    )


def _write_final_decision(path: Path, decision: Mapping[str, Any]) -> None:
    path.write_text(
        f"""# Phase40 Final Decision

## Decision

- Recommendation: {decision.get("recommendation")}
- Evidence type: {decision.get("evidence_type")}
- Actual product benefit claim allowed: {decision.get("actual_product_benefit_claim_allowed")}
- Auto promotion allowed: {decision.get("auto_promotion_allowed")}
- Manual reviewed preference count: {decision.get("manual_reviewed_preference_count")}
- Training candidate status: {decision.get("training_candidate_status")}

## Product Signal

- Adapter over base: {decision.get("adapter_over_base")}
- Adapter over runtime contract: {decision.get("adapter_over_runtime_contract")}
- Adapter + runtime contract over runtime contract: {decision.get("adapter_runtime_contract_over_runtime_contract")}

## Interpretation

Phase40 makes the simulated user-acceptance lab more realistic and creates a pending human review entry point. The default evidence remains simulated lab evidence because no human-reviewed preferences are present yet. It must not be used to claim actual user product benefit.
""",
        encoding="utf-8",
    )


def _write_next_goal(path: Path) -> None:
    path.write_text(
        """目标：开发并验证 PFE Phase41：人工抽样偏好审核达标 + 小批量训练候选。

请基于 Phase40 的 pending review items 收集至少 12 条 consent 完整的人工 reviewed preference，生成 preference candidate manifest，并只在 manual reviewed preference 达标后启动下一轮小模型训练 probe。仍然不要把 simulated acceptance review 冒充 actual production feedback。
""",
        encoding="utf-8",
    )


def generate_phase40_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean_evidence:
        _clean_dir(PHASE40_DIR)
    scenario_dir = PHASE40_DIR / "evidence-scenarios"
    transcript_dir = PHASE40_DIR / "evidence-transcripts"
    blind_eval_dir = PHASE40_DIR / "evidence-blind-eval"
    manual_review_dir = PHASE40_DIR / "evidence-manual-review"
    candidates_dir = PHASE40_DIR / "evidence-candidates"
    for path in (scenario_dir, transcript_dir, blind_eval_dir, manual_review_dir, candidates_dir):
        path.mkdir(parents=True, exist_ok=True)

    phase39_summary = _read_json(PHASE36_39_DIR / "comparison_summary.json")
    phase39_recap = build_phase40_phase39_recap(phase39_summary)
    _write_json(scenario_dir / "phase36_39_recap.json", phase39_recap)

    scenarios = build_phase40_scenario_bank(count=args.scenario_count, phase39_recap=phase39_recap)
    scenario_validation = validate_phase40_scenario_bank(scenarios)
    write_jsonl(scenario_dir / "scenario_bank.jsonl", scenarios)
    _write_json(scenario_dir / "scenario_bank_summary.json", scenario_validation)

    transcripts_by_variant = {
        variant: build_phase40_transcripts(scenarios=scenarios, model_variant=variant)
        for variant in PHASE40_MODEL_VARIANTS
    }
    transcript_structure_rows = []
    for variant, rows in transcripts_by_variant.items():
        write_jsonl(transcript_dir / f"{variant}_transcripts.jsonl", rows)
        transcript_structure_rows.extend(validate_phase40_transcript_structure(row) for row in rows)
    _write_json(
        transcript_dir / "transcript_structure_report.json",
        {
            "kind": "phase40_transcript_structure_report",
            "passed": all(row.get("passed") for row in transcript_structure_rows),
            "transcript_count": len(transcript_structure_rows),
            "failures": [row for row in transcript_structure_rows if not row.get("passed")],
            "created_at": _utcnow_iso(),
        },
    )

    blind_pairs, blind_variant_key = build_phase40_blind_eval_pairs(
        scenarios=scenarios,
        transcripts_by_variant=transcripts_by_variant,
    )
    acceptance_scores = build_phase40_user_acceptance_scores(
        blind_pairs=blind_pairs,
        blind_variant_key=blind_variant_key,
    )
    write_jsonl(blind_eval_dir / "blind_eval_pairs.jsonl", blind_pairs)
    _write_json(blind_eval_dir / "blind_variant_key.json", blind_variant_key)
    _write_json(blind_eval_dir / "user_acceptance_scores.json", acceptance_scores)
    _write_output_examples(blind_eval_dir / "output_examples.md", transcripts_by_variant)

    review_items = build_phase40_manual_review_items(blind_pairs=blind_pairs, sample_count=args.review_sample_count)
    review_decisions = _load_review_decisions(args.review_decisions_json)
    manual_review_summary = build_phase40_manual_review_summary(
        review_items=review_items,
        review_decisions=review_decisions,
    )
    write_jsonl(manual_review_dir / "manual_review_items.jsonl", review_items)
    _write_json(manual_review_dir / "review_decisions.json", {"kind": "phase40_manual_review_decisions", "items": review_decisions})
    _write_json(manual_review_dir / "manual_review_summary.json", manual_review_summary)

    candidate_manifest = build_phase40_preference_candidate_manifest(
        review_items=review_items,
        manual_review_summary=manual_review_summary,
    )
    _write_json(candidates_dir / "preference_candidate_manifest.json", candidate_manifest)
    write_jsonl(candidates_dir / "selected_preference_pairs.jsonl", candidate_manifest.get("selected_preference_pairs") or [])
    _write_json(candidates_dir / "candidate_readiness.json", candidate_manifest)

    boundary_check = validate_phase40_boundaries(
        scenarios=scenarios,
        transcripts=[item for rows in transcripts_by_variant.values() for item in rows],
        blind_pairs=blind_pairs,
        review_items=review_items,
        candidate_manifest=candidate_manifest,
    )
    _write_json(PHASE40_DIR / "boundary_check.json", boundary_check)

    final_decision = phase40_final_decision(
        phase39_recap=phase39_recap,
        acceptance_scores=acceptance_scores,
        manual_review_summary=manual_review_summary,
        candidate_manifest=candidate_manifest,
        boundary_check=boundary_check,
    )
    comparison_summary = build_phase40_comparison_summary(
        scenario_validation=scenario_validation,
        acceptance_scores=acceptance_scores,
        manual_review_summary=manual_review_summary,
        candidate_manifest=candidate_manifest,
        final_decision=final_decision,
    )
    _write_json(PHASE40_DIR / "comparison_summary.json", comparison_summary)
    _write_json(blind_eval_dir / "comparison_summary.json", comparison_summary)
    _write_json(PHASE40_DIR / "phase40-final-decision.json", final_decision)
    _write_runbook(PHASE40_DIR / "phase40-runbook.md")
    _write_final_decision(PHASE40_DIR / "phase40-final-decision.md", final_decision)
    _write_next_goal(PHASE40_DIR / "next-pursuit-goal.md")
    _redact_evidence_tree(PHASE40_DIR)
    return comparison_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--scenario-count", type=int, default=120)
    parser.add_argument("--review-sample-count", type=int, default=24)
    parser.add_argument("--review-decisions-json", type=Path)
    args = parser.parse_args()
    summary = generate_phase40_evidence(args)
    print(
        json.dumps(
            {
                "kind": summary["kind"],
                "status": summary["status"],
                "scenario_count": summary["scenario_count"],
                "manual_reviewed_preference_count": summary["manual_reviewed_preference_count"],
                "training_candidate_status": summary["training_candidate_status"],
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
