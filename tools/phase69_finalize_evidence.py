#!/usr/bin/env python3
"""Finalize Phase69 comparison, decision, examples, and evidence integrity."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase69_minimal_runtime_ab import (
    PHASE69_VARIANTS,
    build_phase69_decision,
    final_assistant_text,
)


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase69-minimal-runtime-ab"
DYNAMIC_FILES = {
    "evidence_manifest.json",
    "finalization_state.json",
    "validation_gate.txt",
    "validation_summary.json",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
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


def _manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        files.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "phase69_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": digest,
    }


def _paired_examples(
    transcripts: Mapping[str, list[dict[str, Any]]], sessions: list[dict[str, Any]]
) -> str:
    def markdown_output(transcript: Mapping[str, Any]) -> str:
        return "\n".join(
            line.rstrip() for line in final_assistant_text(transcript).splitlines()
        )

    by_variant = {
        variant: {str(row.get("session_id")): row for row in transcripts[variant]}
        for variant in PHASE69_VARIANTS
    }
    selected = []
    seen = set()
    for session in sessions:
        key = (session.get("task_type"), session.get("category"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(session)
    lines = [
        "# Phase69 Paired Output Examples",
        "",
        "这些是 simulated_usage 的真实 Qwen3-4B 输出，不是 actual_user_feedback。A/B 使用相同任务与解码，只有候选来源边界契约不同。",
    ]
    for session in selected:
        session_id = str(session.get("session_id"))
        lines.extend(
            [
                "",
                f"## {session_id} ({session.get('category')})",
                "",
                f"**最终用户请求**：{session.get('continuation_request')} {session.get('acceptance_request')}",
                "",
                "**A baseline_runtime**",
                "",
                markdown_output(by_variant["baseline_runtime"][session_id]),
                "",
                "**B candidate_boundary_contract**",
                "",
                markdown_output(by_variant["candidate_boundary_contract"][session_id]),
            ]
        )
    return "\n".join(lines)


def main() -> int:
    eval_dir = EVIDENCE_ROOT / "evidence-qualified-evaluator"
    phase68 = _read_json(
        EVIDENCE_ROOT / "evidence-baseline/phase68_qualified_evaluator_snapshot.json"
    )
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    parity = _read_json(EVIDENCE_ROOT / "ab_parity_audit.json")
    ordinary = _read_json(EVIDENCE_ROOT / "ordinary_control_report.json")
    boundary = _read_json(eval_dir / "boundary_evaluator_report.json")
    transcripts = {
        variant: _read_jsonl(
            EVIDENCE_ROOT / f"evidence-real-generation/transcripts_{variant}.jsonl"
        )
        for variant in PHASE69_VARIANTS
    }
    metrics = {
        variant: _read_json(
            EVIDENCE_ROOT / f"evidence-real-generation/metrics_{variant}.json"
        )
        for variant in PHASE69_VARIANTS
    }
    generation_freezes = {
        variant: _read_json(
            EVIDENCE_ROOT / f"evidence-real-generation/freeze_check_{variant}.json"
        )
        for variant in PHASE69_VARIANTS
    }
    evidence_freezes_passed = (
        all(row.get("passed") is True for row in generation_freezes.values())
        and _read_json(eval_dir / "freeze_check.json").get("passed") is True
        and _read_json(EVIDENCE_ROOT / "preparation_decision.json").get("status")
        == "ready_for_real_generation"
        and _read_json(eval_dir / "eval_preparation_decision.json").get("status")
        == "ready_for_qualified_evaluator"
    )
    decision = build_phase69_decision(
        phase68_snapshot=phase68,
        parity_audit=parity,
        boundary_report=boundary,
        ordinary_report=ordinary,
        evidence_freezes_passed=evidence_freezes_passed,
    )
    boundary_variants = dict(boundary.get("variants") or {})
    ordinary_variants = dict(ordinary.get("variants") or {})
    comparison = {
        "kind": "phase69_minimal_runtime_ab_comparison",
        "model": metrics["baseline_runtime"].get("model_id"),
        "device": metrics["baseline_runtime"].get("device"),
        "only_ab_variable": "candidate_provenance_boundary_contract",
        "boundary": {
            "baseline_runtime": boundary_variants.get("baseline_runtime"),
            "candidate_boundary_contract": boundary_variants.get(
                "candidate_boundary_contract"
            ),
            "candidate_accept_rate_delta": boundary.get(
                "candidate_accept_rate_delta"
            ),
        },
        "ordinary_controls": ordinary_variants,
        "generation": metrics,
        "actual_generation_session_count": sum(
            int(row.get("actual_model_session_count") or 0) for row in metrics.values()
        ),
        "actual_generation_call_count": sum(
            int(row.get("actual_generation_call_count") or 0) for row in metrics.values()
        ),
        "actual_judge_output_count": boundary.get("successful_model_output_count"),
        "expected_judge_output_count": boundary.get("expected_model_output_count"),
        "exhausted_judge_item_count": boundary.get("failure_count"),
        "actual_model_output_count_total": sum(
            int(row.get("actual_generation_call_count") or 0) for row in metrics.values()
        )
        + int(boundary.get("successful_model_output_count") or 0),
        "recommendation": decision["recommendation"],
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }
    integrity_checks = {
        "phase68_snapshot_passed": phase68.get("passed") is True,
        "all_freezes_passed": evidence_freezes_passed,
        "parity_passed": parity.get("passed") is True,
        "both_generation_arms_complete": all(
            row.get("completed_count") == 48 and row.get("failed_count") == 0
            for row in metrics.values()
        ),
        "all_288_generation_calls_real": sum(
            int(row.get("actual_generation_call_count") or 0) for row in metrics.values()
        )
        == 288,
        "all_judge_outcomes_accounted_for": int(
            boundary.get("successful_model_output_count") or 0
        )
        + int(boundary.get("failure_count") or 0)
        == int(boundary.get("expected_model_output_count") or 0)
        == 144,
        "exhausted_judge_failures_preserved": int(boundary.get("failure_count") or 0)
        == len(boundary.get("failures") or [])
        and boundary.get("raw_failures_preserved") is True,
        "no_training_adapter_hermes_or_default_change": decision.get("training_allowed")
        is False
        and decision.get("adapter_created") is False
        and decision.get("hermes_attachment_allowed") is False
        and decision.get("product_default_change_allowed") is False,
        "actual_user_feedback_remains_zero": comparison["actual_user_feedback_count"] == 0,
    }
    integrity = {
        "kind": "phase69_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }

    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase69-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    prefreeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    frozen_finalizer_hash = str(
        dict(prefreeze.get("source_sha256") or {}).get("phase69_finalize") or ""
    )
    current_finalizer_hash = _sha256(Path(__file__).resolve())
    _write_json(
        EVIDENCE_ROOT / "post_freeze_bookkeeping_change.json",
        {
            "kind": "phase69_post_freeze_bookkeeping_change",
            "changed_source": "phase69_finalize",
            "frozen_sha256": frozen_finalizer_hash,
            "current_sha256": current_finalizer_hash,
            "changed_after_model_calls": frozen_finalizer_hash != current_finalizer_hash,
            "reason": (
                "Accept a fully accounted failed judge attempt as valid hold evidence; "
                "do not require fabricated successful judge outputs."
            ),
            "tasks_contract_decoding_evaluator_and_model_outputs_changed": False,
            "decision_gate_changed": False,
            "failed_outputs_preserved": True,
        },
    )
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase69_training_attempt",
            "status": "not_run_by_design",
            "reason": "Phase69 isolates one runtime-contract variable; training would confound the A/B.",
            "adapter_created": False,
            "training_allowed": False,
        },
    )
    _write_text(EVIDENCE_ROOT / "output_examples.md", _paired_examples(transcripts, sessions))
    passed = decision["recommendation"] == (
        "recommend_phase69_runtime_contract_for_manual_review_only"
    )
    _write_text(
        EVIDENCE_ROOT / "phase69-final-decision.md",
        f"""# Phase69 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。A 组边界 accept rate 为 `{decision.get('boundary_accept_rate_baseline')}`，B 组为 `{decision.get('boundary_accept_rate_candidate')}`，真实增量为 `{decision.get('boundary_accept_rate_delta')}`。

## 实验事实

- 模型：本地 Qwen3-4B，A/B 均未加载 adapter。
- 任务：48 个全新三轮 simulated_usage 会话；36 个来源边界任务，12 个普通对照任务。
- 真实生成：288 次；真实双 judge 输出：{boundary.get('successful_model_output_count')} 次，另有 {boundary.get('failure_count')} 个 exhausted item 的失败原文完整保存。
- A/B 唯一变量：B 组额外注入 provenance boundary contract。
- 普通任务通过率：A `{ordinary_variants.get('baseline_runtime', {}).get('pass_rate')}`，B `{ordinary_variants.get('candidate_boundary_contract', {}).get('pass_rate')}`；B 边界话术泄漏数 `{ordinary_variants.get('candidate_boundary_contract', {}).get('boundary_leak_count')}`。

## 边界

本阶段证明的是模拟 holdout 上的 runtime contract 效果，不是 actual_user_feedback，也不是实际产品收益。Phase69 不训练、不创建 adapter、不接 Hermes、不改产品默认路径、不自动 promote。即使通过，也只允许 manual review。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase69-runbook.md",
        """# Phase69 Runbook

```bash
.venv/bin/python tools/phase69_prepare.py --clean-evidence
.venv/bin/python tools/phase69_generate.py --variant baseline_runtime --clean
.venv/bin/python tools/phase69_generate.py --variant candidate_boundary_contract --clean
.venv/bin/python tools/phase69_prepare_eval.py
.venv/bin/python tools/phase69_execute_eval.py --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase69_finalize_evidence.py
.venv/bin/python tools/phase69_validate.py
```

Do not edit tasks, the candidate contract, decoding, evaluator sources, or gates after `phase69_prepare.py`. Do not edit blinded outputs after `phase69_prepare_eval.py`. Resume interrupted judge calls only with `--resume`.
""",
    )
    next_goal = (
        "Build Phase70 as a narrow product-runtime integration canary. Wire the qualified candidate contract behind an explicit non-default mode, verify API streaming/non-streaming parity and ordinary-task routing, then repeat a smaller independent holdout before any default change. Keep manual review and rollback mandatory; do not train in the same phase."
        if passed
        else
        "Build Phase70 as a contract-revision loop using only Phase69 failure categories. Keep the same model, evaluator, and ordinary controls; change one contract clause, freeze a new independent holdout, and rerun. Do not integrate the candidate or start training while Phase69 remains held."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}")
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase69_finalization_state",
        "status": "completed" if integrity["passed"] else "blocked",
        "recommendation": decision["recommendation"],
        "evidence_integrity_passed": integrity["passed"],
        "manifest_file_count": manifest["file_count"],
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "finalization_state.json", finalization)
    print(json.dumps(finalization, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
