#!/usr/bin/env python3
"""Generate Phase34 simulated real-user acceptance judge evidence."""

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

from pfe_core.phase34_simulated_user_acceptance_judge import (
    PHASE34_FEEDBACK_SOURCE,
    blind_pair_public_view,
    build_phase34_default_inputs,
    build_phase34_phase33_review,
    validate_phase34_blind_pair,
    write_jsonl,
)


PHASE33_DIR = Path("docs/demo/phase33-simulated-usage-replay-eval")
PHASE34_DIR = Path("docs/demo/phase34-simulated-real-user-acceptance-judge")
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


def _write_judge_sample(path: Path, judgements: Iterable[Mapping[str, Any]]) -> None:
    lines = ["# Phase34 Simulated User Judge Samples", ""]
    for item in list(judgements)[:8]:
        lines.extend(
            [
                f"## {item.get('pair_id')}",
                "",
                f"- Preferred: {item.get('preferred_variant')} -> {item.get('preferred_model_after_unblind')}",
                f"- Decision: {item.get('acceptance_decision')}",
                f"- Would continue using: {item.get('would_continue_using')}",
                f"- Effort reduction: {item.get('user_effort_reduction_score')}",
                f"- Frustration: {item.get('frustration_score')}",
                f"- Reason: {item.get('acceptance_reason')}",
                "",
            ]
        )
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# Phase34 Runbook

Phase34 adds a simulated real-user acceptance judge on top of Phase33 replay transcripts.

It does not train a model, does not collect actual user feedback, and does not auto-promote. The judge simulates whether the user would accept, edit, reject, or block a response based on reduced effort, trust, correction recovery, evidence, privacy boundary, and false-completion risk.

## Default Evidence

```bash
.venv/bin/python tools/phase34_simulated_user_acceptance_judge.py --clean-evidence --scenario-count 100
```

## Boundaries

- Every scenario, blind pair, and judgement is `simulated_user_judgement`.
- `actual_user_feedback_count` must remain 0.
- The public blind-pair payload must not expose which variant is base or adapter.
- The best possible recommendation is `promote_after_manual_review`.
""",
        encoding="utf-8",
    )


def _write_final_decision(path: Path, summary: Mapping[str, Any]) -> None:
    decision = _dict(summary.get("decision"))
    scores = _dict(summary.get("acceptance_scores"))
    path.write_text(
        f"""# Phase34 Final Decision

## Decision

- Recommendation: {decision.get("recommendation")}
- Status: {decision.get("status")}
- Promotion allowed: {decision.get("promotion_allowed")}
- Auto promotion allowed: false
- Product benefit claim allowed: false
- Actual user feedback collected: false
- Simulated user judgement only: true

## User-Value Scores

- Adapter win rate: {scores.get("adapter_win_rate")}
- Base win rate: {scores.get("base_win_rate")}
- Preferred counts: `{json.dumps(scores.get("preferred_counts") or {}, ensure_ascii=False, sort_keys=True)}`
- Base: `{json.dumps(scores.get("base") or {}, ensure_ascii=False, sort_keys=True)}`
- Adapter: `{json.dumps(scores.get("adapter") or {}, ensure_ascii=False, sort_keys=True)}`
- Delta: `{json.dumps(scores.get("score_delta") or {}, ensure_ascii=False, sort_keys=True)}`

## Interpretation

The simulated real-user judge says the adapter is more useful when the user wants quick execution, correction recovery, privacy boundaries, and evidence-backed progress. This is still simulated proof, not actual online user feedback or a product-benefit claim.

## Reasons

{chr(10).join(f"- {reason}" for reason in decision.get("reasons") or ["no blocking reasons"])}
""",
        encoding="utf-8",
    )


def _write_next_goal(path: Path) -> None:
    path.write_text(
        """目标：开发并验证 PFE Phase35：真实在线 Agent 使用采集 + Phase34 模拟验收校准闭环。

请在 /Users/lichenhao/Desktop/PFE 中完成：

1. 接入 Hermes/PFE 在线交互采集，记录用户目标、Agent 回复、纠正、继续推进和最终验收。
2. 真实记录必须有授权、脱敏、operator、timestamp、source_id，并进入 review queue。
3. 将 Phase34 simulated_user_judgement 与真实在线验收反馈做差异分析，校准模拟验收官。
4. simulated_user_judgement 不得进入 actual_user_feedback 训练候选。
5. 只有 approved actual_user_feedback 达标后，才生成训练候选或 DPO pairs。
6. 不自动 promote；最高 recommendation 仍是 promote_after_manual_review。
""",
        encoding="utf-8",
    )


def generate_phase34_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean_evidence:
        _clean_dir(PHASE34_DIR)
    for subdir in ("evidence", "evidence-scenarios", "evidence-blind-eval", "evidence-judge", "evidence-summary"):
        (PHASE34_DIR / subdir).mkdir(parents=True, exist_ok=True)
    evidence_dir = PHASE34_DIR / "evidence"
    scenario_dir = PHASE34_DIR / "evidence-scenarios"
    blind_dir = PHASE34_DIR / "evidence-blind-eval"
    judge_dir = PHASE34_DIR / "evidence-judge"
    summary_dir = PHASE34_DIR / "evidence-summary"

    phase33_summary = _read_json(PHASE33_DIR / "comparison_summary.json")
    phase33_decision = (PHASE33_DIR / "phase33-final-decision.md").read_text(encoding="utf-8") if (PHASE33_DIR / "phase33-final-decision.md").exists() else ""
    phase33_review = build_phase34_phase33_review(
        phase33_summary=phase33_summary,
        phase33_decision_text=phase33_decision,
    )
    generated = build_phase34_default_inputs(scenario_count=args.scenario_count, phase33_summary=phase33_summary)
    scenarios = list(generated["scenario_batch"]["scenarios"])
    pairs = list(generated["blind_eval_pairs"])
    public_pairs = [blind_pair_public_view(pair) for pair in pairs]
    blind_validations = [validate_phase34_blind_pair(pair) for pair in pairs]
    blind_validation_summary = {
        "kind": "phase34_blind_validation_summary",
        "pair_count": len(blind_validations),
        "passed": all(item.get("passed") for item in blind_validations),
        "identity_leak_count": sum(1 for item in blind_validations if item.get("identity_leaked_to_judge")),
        "created_at": _utcnow_iso(),
    }

    _write_json(evidence_dir / "phase33_review.json", phase33_review)
    _write_json(scenario_dir / "acceptance_scenarios.json", generated["scenario_batch"])
    write_jsonl(scenario_dir / "acceptance_scenarios.jsonl", scenarios)
    write_jsonl(blind_dir / "blind_eval_pairs_public.jsonl", public_pairs)
    write_jsonl(blind_dir / "blind_eval_pairs_with_unblind_map.jsonl", pairs)
    _write_json(blind_dir / "blind_validation_summary.json", blind_validation_summary)
    write_jsonl(judge_dir / "simulated_user_judgements.jsonl", generated["judgements"])
    _write_json(judge_dir / "simulated_user_judgements.json", {"kind": "phase34_simulated_user_judgements", "items": generated["judgements"]})
    _write_judge_sample(judge_dir / "judge_output_examples.md", generated["judgements"])
    _write_json(summary_dir / "acceptance_scores.json", generated["acceptance_scores"])
    _write_json(summary_dir / "simulation_boundary_check.json", generated["boundary_check"])
    _write_json(summary_dir / "decision.json", generated["decision"])

    summary = {
        "kind": "phase34_simulated_real_user_acceptance_judge_summary",
        "status": "completed",
        "source": PHASE34_FEEDBACK_SOURCE,
        "simulated_user_judgement": True,
        "actual_user_feedback_count": 0,
        "training_run": False,
        "phase33_review": phase33_review,
        "scenario_batch": {
            "scenario_count": generated["scenario_batch"]["scenario_count"],
            "categories": generated["scenario_batch"]["categories"],
            "scenario_count_within_required_range": generated["scenario_batch"]["scenario_count_within_required_range"],
        },
        "blind_validation_summary": blind_validation_summary,
        "boundary_check": generated["boundary_check"],
        "acceptance_scores": generated["acceptance_scores"],
        "decision": generated["decision"],
        "final_recommendation": generated["decision"]["recommendation"],
        "created_at": _utcnow_iso(),
    }
    _write_json(PHASE34_DIR / "comparison_summary.json", summary)
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_runbook(PHASE34_DIR / "phase34-runbook.md")
    _write_final_decision(PHASE34_DIR / "phase34-final-decision.md", summary)
    _write_next_goal(PHASE34_DIR / "next-pursuit-goal.md")
    _redact_evidence_tree(PHASE34_DIR)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--scenario-count", type=int, default=100)
    args = parser.parse_args()

    summary = generate_phase34_evidence(args)
    print(
        json.dumps(
            {
                "kind": summary["kind"],
                "status": summary["status"],
                "scenario_count": summary["scenario_batch"]["scenario_count"],
                "actual_user_feedback_count": summary["actual_user_feedback_count"],
                "blind_validation": summary["blind_validation_summary"],
                "boundary_check": summary["boundary_check"],
                "acceptance_scores": summary["acceptance_scores"],
                "decision": summary["decision"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
