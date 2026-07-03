#!/usr/bin/env python3
"""Generate Phase33 simulated multi-turn Agent usage replay evidence."""

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

from pfe_core.phase33_simulated_usage_replay import (
    PHASE33_FEEDBACK_SOURCE,
    build_phase33_eval_report,
    build_phase33_phase32_reference,
    build_phase33_transcripts,
    build_phase33_usage_sessions,
    phase33_final_decision,
    validate_phase33_simulation_boundaries,
    write_jsonl,
)


PHASE32_DIR = Path("docs/demo/phase32-personal-agent-preference-training-loop")
PHASE33_DIR = Path("docs/demo/phase33-simulated-usage-replay-eval")
PHASE32_ADAPTER_DIR = Path("trainer_job_outputs/phase32-personal-agent-preference-qwen25-0_5b/dpo_adapter")
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


def _write_transcript_sample(path: Path, base_transcripts: Iterable[Mapping[str, Any]], adapter_transcripts: Iterable[Mapping[str, Any]]) -> None:
    lines = ["# Phase33 Replay Transcript Samples", ""]
    for label, rows in (("Base", base_transcripts), ("Adapter", adapter_transcripts)):
        lines.extend([f"## {label}", ""])
        for transcript in list(rows)[:3]:
            lines.extend([f"### {transcript.get('session_id')} / {transcript.get('workflow_category')}", ""])
            for turn in transcript.get("turns") or []:
                item = _dict(turn)
                lines.extend([f"**{item.get('role')} / {item.get('stage')}**", "", str(item.get("content") or ""), ""])
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# Phase33 Runbook

Phase33 simulates real multi-turn Agent usage sessions to evaluate whether the Phase32 personal preference adapter profile behaves better than the base profile in the same scenarios.

All Phase33 sessions and transcripts are `simulated_usage`. They are not actual user feedback and must not enter training as realtime user feedback.

## Default Evidence

```bash
.venv/bin/python tools/phase33_simulated_usage_replay_eval.py --clean-evidence --session-count 64
```

## What This Proves

- Same-session base vs adapter replay comparison.
- Multi-turn behavior: user goal, Agent answer, user correction, continued execution, final acceptance.
- Privacy boundary: no raw Obsidian or AgentMemory private text is committed.
- Decision gate never auto-promotes; the best possible recommendation is `promote_after_manual_review`.

## What This Does Not Prove

- It does not claim actual user feedback was collected.
- It does not replace real online feedback.
- It does not auto-promote the Phase32 adapter.
""",
        encoding="utf-8",
    )


def _write_final_decision(path: Path, summary: Mapping[str, Any]) -> None:
    decision = _dict(summary.get("decision"))
    eval_report = _dict(summary.get("eval_report"))
    base = _dict(_dict(eval_report.get("base")).get("scores"))
    adapter = _dict(_dict(eval_report.get("adapter")).get("scores"))
    path.write_text(
        f"""# Phase33 Final Decision

## Decision

- Recommendation: {decision.get("recommendation")}
- Status: {decision.get("status")}
- Promotion allowed: {decision.get("promotion_allowed")}
- Auto promotion allowed: false
- Manual review required before promotion: true
- Product benefit claim allowed: false
- Simulated usage only: true
- Actual user feedback collected: false

## Replay

- Session count: {eval_report.get("session_count")}
- Source: {PHASE33_FEEDBACK_SOURCE}
- Same-session comparison: {eval_report.get("same_session_comparison")}

## Scores

- Base: `{json.dumps(base, ensure_ascii=False, sort_keys=True)}`
- Adapter: `{json.dumps(adapter, ensure_ascii=False, sort_keys=True)}`
- Delta: `{json.dumps(eval_report.get("score_delta") or {}, ensure_ascii=False, sort_keys=True)}`

## Reasons

{chr(10).join(f"- {reason}" for reason in decision.get("reasons") or ["no blocking reasons"])}

## Boundary

The replay is useful as a product-behavior simulation, not as actual feedback. It must remain excluded from `actual_user_feedback` training pipelines unless a future human review explicitly converts separate real interaction records.
""",
        encoding="utf-8",
    )


def _write_next_goal(path: Path) -> None:
    path.write_text(
        """目标：开发并验证 PFE Phase34：真实在线 Agent 使用采集 + Phase33 回放校准闭环。

请在 /Users/lichenhao/Desktop/PFE 中完成：

1. 基于 Phase33 的 simulated_usage replay taxonomy，接入真实 Hermes/PFE 在线交互采集。
2. 所有真实采集必须经过用户授权、脱敏、review queue 和 holdout 隔离。
3. 将 Phase33 的模拟评分与真实在线反馈评分做差异分析，找出模拟器过度乐观或漏判的场景。
4. 只有 approved actual_user_feedback 达标后，才生成训练候选；Phase33 simulated_usage 不得进入真实训练。
5. 对 base、Phase32 adapter、真实反馈候选 adapter 做同场景对比，保存 transcripts、评分、隐私扫描和 decision。
6. 不自动 promote；最高 recommendation 仍是 promote_after_manual_review。
""",
        encoding="utf-8",
    )


def generate_phase33_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean_evidence:
        _clean_dir(PHASE33_DIR)
    for subdir in ("evidence", "evidence-sessions", "evidence-transcripts", "evidence-eval"):
        (PHASE33_DIR / subdir).mkdir(parents=True, exist_ok=True)
    evidence_dir = PHASE33_DIR / "evidence"
    session_dir = PHASE33_DIR / "evidence-sessions"
    transcript_dir = PHASE33_DIR / "evidence-transcripts"
    eval_dir = PHASE33_DIR / "evidence-eval"

    phase32_summary = _read_json(PHASE32_DIR / "comparison_summary.json")
    phase32_reference = build_phase33_phase32_reference(phase32_summary=phase32_summary)
    phase32_reference["local_adapter_artifact_present"] = (PHASE32_ADAPTER_DIR / "adapter_config.json").exists()
    phase32_reference["local_adapter_artifact_reference"] = str(PHASE32_ADAPTER_DIR)
    sessions = build_phase33_usage_sessions(count=args.session_count)
    session_rows = [dict(item) for item in sessions["sessions"]]
    base_transcripts = build_phase33_transcripts(
        sessions=session_rows,
        model_variant="base",
        phase32_reference=phase32_reference,
    )
    adapter_transcripts = build_phase33_transcripts(
        sessions=session_rows,
        model_variant="adapter",
        phase32_reference=phase32_reference,
    )
    boundary_check = validate_phase33_simulation_boundaries(
        sessions=session_rows,
        transcripts=[*base_transcripts, *adapter_transcripts],
    )
    eval_report = build_phase33_eval_report(
        sessions=session_rows,
        base_transcripts=base_transcripts,
        adapter_transcripts=adapter_transcripts,
    )
    decision = phase33_final_decision(eval_report=eval_report, phase32_reference=phase32_reference)

    _write_json(evidence_dir / "phase32_reference.json", phase32_reference)
    _write_json(evidence_dir / "simulation_boundary_check.json", boundary_check)
    _write_json(session_dir / "simulated_usage_sessions.json", sessions)
    write_jsonl(session_dir / "simulated_usage_sessions.jsonl", session_rows)
    write_jsonl(transcript_dir / "base_replay_transcripts.jsonl", base_transcripts)
    write_jsonl(transcript_dir / "adapter_replay_transcripts.jsonl", adapter_transcripts)
    _write_transcript_sample(transcript_dir / "replay_transcripts_sample.md", base_transcripts, adapter_transcripts)
    _write_json(eval_dir / "eval_report.json", eval_report)
    _write_json(eval_dir / "decision.json", decision)

    summary = {
        "kind": "phase33_simulated_usage_replay_eval_summary",
        "status": "completed",
        "source": PHASE33_FEEDBACK_SOURCE,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "session_count": sessions["session_count"],
        "phase32_reference": phase32_reference,
        "session_batch": {
            "session_count": sessions["session_count"],
            "categories": sessions["categories"],
            "session_count_within_required_range": sessions["session_count_within_required_range"],
        },
        "boundary_check": boundary_check,
        "eval_report": eval_report,
        "decision": decision,
        "final_recommendation": decision["recommendation"],
        "created_at": _utcnow_iso(),
    }
    _write_json(PHASE33_DIR / "comparison_summary.json", summary)
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_runbook(PHASE33_DIR / "phase33-runbook.md")
    _write_final_decision(PHASE33_DIR / "phase33-final-decision.md", summary)
    _write_next_goal(PHASE33_DIR / "next-pursuit-goal.md")
    _redact_evidence_tree(PHASE33_DIR)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--session-count", type=int, default=64)
    args = parser.parse_args()

    summary = generate_phase33_evidence(args)
    print(
        json.dumps(
            {
                "kind": summary["kind"],
                "status": summary["status"],
                "session_count": summary["session_count"],
                "actual_user_feedback_count": summary["actual_user_feedback_count"],
                "boundary_check": summary["boundary_check"],
                "base_scores": _dict(_dict(summary["eval_report"].get("base")).get("scores")),
                "adapter_scores": _dict(_dict(summary["eval_report"].get("adapter")).get("scores")),
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
