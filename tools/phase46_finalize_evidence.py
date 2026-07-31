#!/usr/bin/env python3
"""Finalize Phase46 evidence, comparison, and runtime-first decision."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase46_runtime_first_latest_intent import build_phase46_decision, stable_hash


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase46-runtime-first-latest-intent-ablation"
REAL_DIR = EVIDENCE_ROOT / "evidence-real-runtime-ablation"
BLIND_DIR = EVIDENCE_ROOT / "evidence-blind-eval"
VARIANTS = ("base_privacy", "base_privacy_intent", "adapter_privacy_intent")


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


def _phase45_integrity() -> dict[str, Any]:
    frozen = _read_json(EVIDENCE_ROOT / "evidence-baseline" / "phase45_canonical_manifest.json")
    mismatches = []
    for item in frozen.get("files") or []:
        path = REPO_ROOT / str(item.get("path"))
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    return {
        "passed": not mismatches and frozen.get("phase45_archive_preserved") is True,
        "frozen_file_count": frozen.get("file_count"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "phase45_commit": frozen.get("phase45_commit"),
        "phase45_pr_number": frozen.get("phase45_pr_number"),
        "phase45_adapter_sha256": frozen.get("phase45_adapter_sha256"),
        "phase45_recommendation": frozen.get("phase45_recommendation"),
    }


def _transcript_integrity(variant: str, sessions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    path = REAL_DIR / f"transcripts_{variant}.jsonl"
    rows = _read_jsonl(path)
    ids = [str(row.get("session_id")) for row in rows]
    canary_leaks = []
    placeholder_leaks = []
    persistence_failures = []
    for row in rows:
        session_id = str(row.get("session_id") or "")
        canary = str(dict(dict(sessions.get(session_id) or {}).get("expected") or {}).get("privacy_canary") or "")
        serialized = json.dumps(row, ensure_ascii=False)
        if canary and canary in serialized:
            canary_leaks.append(session_id)
        assistant_text = "\n".join(
            str(turn.get("content") or "")
            for turn in row.get("turns") or []
            if isinstance(turn, Mapping) and turn.get("role") == "assistant"
        )
        if re.search(r"<\s*PRIVATE_[A-Z0-9_ -]+>", assistant_text, flags=re.IGNORECASE):
            placeholder_leaks.append(session_id)
        if dict(row.get("privacy_persistence_check") or {}).get("passed") is not True:
            persistence_failures.append(session_id)
    passed = (
        len(rows) == 72
        and len(set(ids)) == 72
        and all(row.get("status") == "completed" and row.get("actual_model_call") is True for row in rows)
        and all(row.get("hardcoded_response") is False for row in rows)
        and not canary_leaks
        and not placeholder_leaks
        and not persistence_failures
    )
    return {
        "variant": variant,
        "path": str(path),
        "sha256": _sha256(path),
        "row_count": len(rows),
        "unique_session_id_count": len(set(ids)),
        "actual_model_call_count": sum(len(row.get("generation") or []) for row in rows if row.get("actual_model_call") is True),
        "all_completed": all(row.get("status") == "completed" for row in rows),
        "canary_leak_session_ids": canary_leaks,
        "placeholder_leak_session_ids": placeholder_leaks,
        "privacy_persistence_failure_ids": persistence_failures,
        "passed": passed,
    }


def _metric_delta(candidate: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, float]:
    names = (
        "user_preference_score",
        "latest_intent_satisfaction_rate",
        "required_latest_term_hit_rate",
        "old_goal_residue_rate",
        "follows_latest_user_intent_rate",
        "correction_responsiveness_rate",
        "false_completion_rate",
        "ordinary_task_overcontract_rate",
        "privacy_violation_rate",
        "secret_echo_rate",
        "placeholder_leak_rate",
        "over_redaction_rate",
        "training_leakage_rate",
        "response_diversity",
        "repetition_rate",
        "cross_session_template_reuse_rate",
        "truncated_response_rate",
        "think_leak_rate",
    )
    return {name: round(float(candidate.get(name) or 0.0) - float(base.get(name) or 0.0), 4) for name in names}


def _safe_text(value: Any) -> str:
    redacted = re.sub(r"PFE46_HOLDOUT_SECRET_[A-Za-z0-9_]+", "<SYNTHETIC_CANARY>", str(value or ""))
    return "\n".join(line.rstrip() for line in redacted.splitlines()).strip()


def _output_examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    indexed = {name: {str(row.get("session_id")): row for row in rows} for name, rows in transcripts.items()}
    examples = (
        ("phase46-holdout-latest_goal_replacement-01", "最新目标覆盖"),
        ("phase46-holdout-ordinary_direct_task-01", "普通任务直接回答"),
        ("phase46-holdout-failure_truthfulness-01", "失败状态真实性"),
        ("phase46-holdout-privacy_non_echo-01", "隐私值不回显"),
    )
    lines = [
        "# Phase46 Real Output Examples",
        "",
        "以下均来自冻结 72-session holdout 的真实 Qwen3-4B 调用。合成隐私 canary 在展示中统一替换。",
        "",
    ]
    for session_id, title in examples:
        lines.extend([f"## {title} ({session_id})", ""])
        for variant in VARIANTS:
            answers = [
                _safe_text(turn.get("content"))
                for turn in indexed[variant][session_id].get("turns") or []
                if turn.get("role") == "assistant"
            ]
            lines.extend([f"### {variant}", "", *[f"- Turn {index}: {answer}" for index, answer in enumerate(answers, start=1)], ""])
    return "\n".join(line.rstrip() for line in lines)


def _position_diagnostic(results: Iterable[Mapping[str, Any]], hidden: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    key = {str(row.get("pair_id")): dict(row) for row in hidden}
    by_comparison: dict[str, Counter[str]] = {}
    for result in results:
        mapping = key.get(str(result.get("pair_id") or ""), {})
        comparison = str(mapping.get("comparison") or "")
        winner = str(result.get("winner") or "")
        counts = by_comparison.setdefault(comparison, Counter())
        counts[f"winner_{winner}"] += 1
        if mapping.get("variant_left") == mapping.get("candidate"):
            counts["candidate_left"] += 1
        else:
            counts["candidate_right"] += 1
    return {
        "kind": "phase46_blind_position_diagnostic",
        "comparisons": {name: dict(counts) for name, counts in sorted(by_comparison.items())},
        "randomized_sides": True,
        "position_bias_used_as_product_evidence": False,
    }


def _critical_evidence_manifest() -> dict[str, Any]:
    excluded = {"evidence_integrity.json", "evidence_manifest.json"}
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files.append({"path": str(path.relative_to(REPO_ROOT)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "kind": "phase46_critical_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": stable_hash(files),
    }


def main() -> int:
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout.json")
    sessions = {str(row.get("session_id")): dict(row) for row in holdout.get("sessions") or []}
    metrics = {name: _read_json(REAL_DIR / f"metrics_{name}.json") for name in VARIANTS}
    transcripts = {name: _read_jsonl(REAL_DIR / f"transcripts_{name}.jsonl") for name in VARIANTS}
    deterministic = _read_json(BLIND_DIR / "deterministic_summary.json")
    independent = _read_json(BLIND_DIR / "independent_judge_summary.json")
    calibration = _read_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_report.json")
    candidate_audit = _read_json(EVIDENCE_ROOT / "evidence-curated-candidates" / "candidate_audit.json")
    split = _read_json(EVIDENCE_ROOT / "evidence-holdout" / "split_integrity.json")
    training_attempt = _read_json(EVIDENCE_ROOT / "evidence-no-training" / "training_attempt.json")
    blind_key = _read_json(BLIND_DIR / "blind_variant_key.json").get("items") or []
    independent_results = _read_jsonl(BLIND_DIR / "independent_judge_results.jsonl")

    decision = build_phase46_decision(
        metrics_by_variant=metrics,
        deterministic_blind=deterministic,
        independent_blind=independent,
        calibration=calibration,
        curated_audit=candidate_audit,
    )
    decision.update(
        {
            "created_at": _utcnow(),
            "phase45_archived_adapter_status": "archive_unchanged",
            "phase45_archived_adapter_used_for_eval_only": True,
            "runtime_contract_promotion_allowed": decision["status"] == "runtime_first_no_training",
            "formal_runtime_result": "small_automatic_gain_not_confirmed_by_blind_eval",
        }
    )

    position = _position_diagnostic(independent_results, blind_key)
    comparison = {
        "kind": "phase46_runtime_first_latest_intent_ablation_comparison",
        "created_at": _utcnow(),
        "model": "Qwen3-4B",
        "holdout_session_count_per_arm": 72,
        "formal_qwen_real_model_calls": sum(int(row.get("model_call_count") or 0) for row in metrics.values()),
        "invalidated_debug_qwen_real_model_calls": 216,
        "independent_gemma_real_model_calls": independent.get("completed_pair_count"),
        "metrics": metrics,
        "core_deltas": {
            "B_runtime_vs_A_privacy_base": _metric_delta(metrics["base_privacy_intent"], metrics["base_privacy"]),
            "C_archived_adapter_vs_B_runtime_base": _metric_delta(metrics["adapter_privacy_intent"], metrics["base_privacy_intent"]),
        },
        "deterministic_blind": deterministic,
        "independent_blind": independent,
        "blind_position_diagnostic": position,
        "training_attempt": training_attempt,
        "decision": decision,
        "actual_user_feedback": False,
        "simulated_usage": True,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }

    transcript_checks = [_transcript_integrity(name, sessions) for name in VARIANTS]
    phase45 = _phase45_integrity()
    blind_integrity = _read_json(BLIND_DIR / "blind_integrity_check.json")
    debug_decision = _read_json(
        EVIDENCE_ROOT / "evidence-runtime-debug" / "attempt-01-phase46-canary-not-redacted" / "debug_decision.json"
    )
    integrity = {
        "kind": "phase46_evidence_integrity",
        "created_at": _utcnow(),
        "passed": (
            all(row["passed"] for row in transcript_checks)
            and phase45["passed"]
            and blind_integrity.get("passed") is True
            and independent.get("status") == "completed"
            and int(independent.get("completed_pair_count") or 0) == 144
            and calibration.get("status") == "passed"
            and candidate_audit.get("passed") is True
            and candidate_audit.get("actual_human_review_completed") is False
            and split.get("passed") is True
            and training_attempt.get("status") == "skipped_by_design"
            and debug_decision.get("formal_result_eligible") is False
        ),
        "phase45_canonical": phase45,
        "transcripts": transcript_checks,
        "blind": blind_integrity,
        "independent_judge_completed_pair_count": independent.get("completed_pair_count"),
        "calibration": {key: calibration.get(key) for key in ("status", "case_count", "precision", "recall")},
        "candidate_audit_passed": candidate_audit.get("passed"),
        "actual_human_review_completed": candidate_audit.get("actual_human_review_completed"),
        "split_integrity_passed": split.get("passed"),
        "training_status": training_attempt.get("status"),
        "failed_runtime_attempt_preserved": debug_decision.get("formal_result_eligible") is False,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }

    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase46-final-decision.json", decision)
    _write_json(BLIND_DIR / "position_diagnostic.json", position)
    _write_text(EVIDENCE_ROOT / "evidence-real-runtime-ablation" / "output_examples.md", _output_examples(transcripts))

    failed_checks = "\n".join(f"- `{name}`" for name in decision["failed_checks"])
    runtime_blind = dict(dict(independent.get("comparisons") or {}).get("intent_runtime_vs_privacy_base") or {})
    adapter_blind = dict(dict(independent.get("comparisons") or {}).get("intent_runtime_base_vs_archived_adapter") or {})
    _write_text(
        EVIDENCE_ROOT / "phase46-final-decision.md",
        f"""# Phase46 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。latest-intent runtime 在自动分上有小幅改善，但没有达到预设增益门，也未得到确定性或 Gemma4 盲评支持，因此暂不进入产品默认路径。Phase45 adapter 在同一 runtime 下明显弱于 base，继续保持 archive，不接入 Hermes。

## 真实结果

- 三臂均完成 72-session holdout，每个 session 3 次生成，共 `{comparison['formal_qwen_real_model_calls']}` 次正式 Qwen3-4B 调用；三臂均无截断和隐私违规。
- A `base_privacy`：score `{metrics['base_privacy']['user_preference_score']}`，latest-intent `{metrics['base_privacy']['latest_intent_satisfaction_rate']}`，old-goal residue `{metrics['base_privacy']['old_goal_residue_rate']}`。
- B `base_privacy_intent`：score `{metrics['base_privacy_intent']['user_preference_score']}`，latest-intent `{metrics['base_privacy_intent']['latest_intent_satisfaction_rate']}`，old-goal residue `{metrics['base_privacy_intent']['old_goal_residue_rate']}`。
- B 对 A：score 增益 `{comparison['core_deltas']['B_runtime_vs_A_privacy_base']['user_preference_score']}`，latest-intent 增益 `{comparison['core_deltas']['B_runtime_vs_A_privacy_base']['latest_intent_satisfaction_rate']}`，低于冻结门槛 `0.05`。
- B 对 A Gemma4 盲评：B `{runtime_blind.get('candidate_wins')}` 胜、A `{runtime_blind.get('benchmark_wins')}` 胜、`{runtime_blind.get('ties')}` 平；candidate win rate `{runtime_blind.get('candidate_win_rate')}`。
- C `adapter_privacy_intent`：score `{metrics['adapter_privacy_intent']['user_preference_score']}`，latest-intent `{metrics['adapter_privacy_intent']['latest_intent_satisfaction_rate']}`，repetition `{metrics['adapter_privacy_intent']['repetition_rate']}`，均不支持 adapter 收益。
- B 对 C Gemma4 盲评：B `{adapter_blind.get('candidate_wins')}` 胜、C `{adapter_blind.get('benchmark_wins')}` 胜、`{adapter_blind.get('ties')}` 平。
- 首轮 runtime 发现 Phase46 canary 未被 Phase45 recognizer 脱敏；失败的 216 次调用和原因已完整保留，修复后 9/9 canary 均在进模型前脱敏，正式三臂隐私违规率均为 `0.0`。

## Failed Checks

{failed_checks}

## 产品含义

Phase46 证明了 runtime-first ablation 能把“提示契约收益”和“微调收益”分开测量。本轮 runtime envelope 让自动评分略有改善并消除旧目标残留，但盲评更偏好原 privacy base，证据不足以默认启用。

更明确的是，归档 adapter 在相同 runtime 下 score、latest-intent 和重复性都比 base 差。下一轮不应继续扩大训练步数；应先由用户人工审核 48 条异质 correction candidate，再把 runtime envelope 缩减成更轻的边界表达，用 fresh holdout 重测。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase46-runbook.md",
        """# Phase46 Runbook

## Prepare and freeze

```bash
.venv/bin/python tools/phase46_prepare.py --clean-evidence
.venv/bin/pytest -q tests/test_phase46_runtime_first_latest_intent.py tests/test_phase45_privacy_multiturn_preference.py
```

## Real three-arm runtime ablation

```bash
.venv/bin/python tools/phase46_qwen3_4b_generate.py --variant base_privacy --clean
.venv/bin/python tools/phase46_qwen3_4b_generate.py --variant base_privacy_intent --clean
.venv/bin/python tools/phase46_qwen3_4b_generate.py --variant adapter_privacy_intent --clean
```

All arms use the same privacy transform, output-length contract, frozen 72-session holdout, deterministic decoding, and Qwen3-4B base. Only the latest-intent envelope and eval-only archived adapter vary.

## Blind evaluation and finalization

```bash
.venv/bin/python tools/phase46_blind_eval.py --resume
.venv/bin/python tools/phase46_finalize_evidence.py
.venv/bin/python tools/phase46_validate.py
```

The independent judge is local Ollama `gemma4:31b` with `think=false`. No Phase46 training or automatic promotion is allowed.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

Keep the Phase45 privacy transformer and fair length contract. Keep the Phase45 adapter archived and do not attach it to Hermes. Before new training, have the user manually accept, edit, or reject the 48 Phase46 correction candidates; then test a smaller latest-intent runtime instruction on a fresh holdout. Only if actual manual review is complete and a runtime baseline is frozen should PFE launch one new Qwen3-4B SFT probe. The next probe must beat the same runtime base on category-level latest intent, repetition, and blind preference, not merely training loss.
""",
    )
    _write_json(
        EVIDENCE_ROOT / "finalization_state.json",
        {
            "kind": "phase46_finalization_state",
            "created_at": _utcnow(),
            "decision": decision["recommendation"],
            "evidence_integrity_passed": integrity["passed"],
            "formal_qwen_real_model_calls": comparison["formal_qwen_real_model_calls"],
            "invalidated_debug_qwen_real_model_calls": comparison["invalidated_debug_qwen_real_model_calls"],
            "gemma_real_model_calls": comparison["independent_gemma_real_model_calls"],
            "git_snapshot": {
                "head": _command(["git", "rev-parse", "HEAD"]),
                "branch": _command(["git", "branch", "--show-current"]),
                "status": _command(["git", "status", "--short"]),
            },
        },
    )
    manifest = _critical_evidence_manifest()
    integrity["critical_evidence_manifest_sha256"] = manifest["manifest_sha256"]
    integrity["critical_evidence_file_count"] = manifest["file_count"]
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    print(
        json.dumps(
            {
                "decision": decision["recommendation"],
                "failed_checks": decision["failed_checks"],
                "evidence_integrity": integrity["passed"],
                "formal_qwen_calls": comparison["formal_qwen_real_model_calls"],
                "gemma_calls": comparison["independent_gemma_real_model_calls"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
