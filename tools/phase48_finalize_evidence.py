#!/usr/bin/env python3
"""Finalize Phase48 compact-runtime evidence and the no-training decision."""

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

from pfe_core.phase46_runtime_first_latest_intent import stable_hash
from pfe_core.phase48_compact_intent_runtime import build_phase48_decision


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase48-compact-intent-runtime-ablation"
REAL_DIR = EVIDENCE_ROOT / "evidence-real-runtime-ablation"
BLIND_DIR = EVIDENCE_ROOT / "evidence-blind-eval"
PHASE46_ROOT = REPO_ROOT / "docs" / "demo" / "phase46-runtime-first-latest-intent-ablation"
PHASE47_ROOT = REPO_ROOT / "docs" / "demo" / "phase47-simulated-user-review"
VARIANTS = ("base_privacy", "base_compact_intent", "base_full_intent")


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


def _canonical_integrity(phase: str, root: Path) -> dict[str, Any]:
    snapshot = _read_json(EVIDENCE_ROOT / "evidence-baseline" / f"{phase}_canonical_snapshot.json")
    manifest = _read_json(root / "evidence_manifest.json")
    mismatches = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / str(item.get("path") or "")
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    manifest_matches = manifest.get("manifest_sha256") == snapshot.get("manifest_sha256")
    return {
        "phase": phase,
        "passed": snapshot.get("passed") is True and manifest_matches and not mismatches,
        "manifest_matches_snapshot": manifest_matches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "recommendation": snapshot.get("recommendation"),
    }


def _transcript_integrity(variant: str, sessions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    path = REAL_DIR / f"transcripts_{variant}.jsonl"
    rows = _read_jsonl(path)
    ids = [str(row.get("session_id") or "") for row in rows]
    canary_leaks = []
    placeholder_leaks = []
    persistence_failures = []
    for row in rows:
        session_id = str(row.get("session_id") or "")
        expected = dict(dict(sessions.get(session_id) or {}).get("expected") or {})
        canary = str(expected.get("privacy_canary") or "")
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
    expected_mode = {
        "base_privacy": "privacy_base",
        "base_compact_intent": "compact_system_instruction",
        "base_full_intent": "phase46_full_envelope",
    }[variant]
    passed = (
        len(rows) == 64
        and len(set(ids)) == 64
        and all(row.get("status") == "completed" and row.get("actual_model_call") is True for row in rows)
        and all(row.get("hardcoded_response") is False and row.get("adapter_loaded") is False for row in rows)
        and all(row.get("runtime_mode") == expected_mode for row in rows)
        and all(len(row.get("generation") or []) == 3 for row in rows)
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
        "actual_model_call_count": sum(len(row.get("generation") or []) for row in rows),
        "runtime_mode": expected_mode,
        "canary_leak_session_ids": canary_leaks,
        "placeholder_leak_session_ids": placeholder_leaks,
        "privacy_persistence_failure_ids": persistence_failures,
        "passed": passed,
    }


def _metric_delta(candidate: Mapping[str, Any], benchmark: Mapping[str, Any]) -> dict[str, float]:
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
    return {
        name: round(float(candidate.get(name) or 0.0) - float(benchmark.get(name) or 0.0), 4)
        for name in names
    }


def _category_analysis(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    base = dict(metrics["base_privacy"].get("category_metrics") or {})
    compact = dict(metrics["base_compact_intent"].get("category_metrics") or {})
    full = dict(metrics["base_full_intent"].get("category_metrics") or {})
    rows = []
    for category in sorted(base):
        rows.append(
            {
                "category": category,
                "base_latest_intent": base[category].get("latest_intent_satisfaction"),
                "compact_latest_intent": compact[category].get("latest_intent_satisfaction"),
                "full_latest_intent": full[category].get("latest_intent_satisfaction"),
                "compact_vs_base_latest_delta": round(
                    float(compact[category].get("latest_intent_satisfaction") or 0.0)
                    - float(base[category].get("latest_intent_satisfaction") or 0.0),
                    4,
                ),
                "compact_vs_full_latest_delta": round(
                    float(compact[category].get("latest_intent_satisfaction") or 0.0)
                    - float(full[category].get("latest_intent_satisfaction") or 0.0),
                    4,
                ),
                "compact_vs_base_score_delta": round(
                    float(compact[category].get("phase46_composite_score") or 0.0)
                    - float(base[category].get("phase46_composite_score") or 0.0),
                    4,
                ),
                "compact_vs_full_score_delta": round(
                    float(compact[category].get("phase46_composite_score") or 0.0)
                    - float(full[category].get("phase46_composite_score") or 0.0),
                    4,
                ),
            }
        )
    return {
        "kind": "phase48_category_analysis",
        "rows": rows,
        "largest_compact_gain_vs_base": max(rows, key=lambda row: row["compact_vs_base_score_delta"]),
        "largest_compact_regression_vs_base": min(rows, key=lambda row: row["compact_vs_base_score_delta"]),
        "interpretation": (
            "The compact instruction improved evidence-status and ordinary direct-task behavior, "
            "but provenance-boundary latest-intent behavior regressed."
        ),
    }


def _safe_text(value: Any) -> str:
    redacted = re.sub(r"PFE\d+_(?:HOLDOUT_)?SECRET_[A-Za-z0-9_]+", "<SYNTHETIC_CANARY>", str(value or ""))
    return "\n".join(line.rstrip() for line in redacted.splitlines()).strip()


def _output_examples(
    transcripts: Mapping[str, list[dict[str, Any]]],
    sessions: Mapping[str, Mapping[str, Any]],
) -> str:
    indexed = {name: {str(row.get("session_id")): row for row in rows} for name, rows in transcripts.items()}
    examples = (
        ("phase48-holdout-latest_goal_replacement-01", "最新目标覆盖"),
        ("phase48-holdout-evidence_status-01", "证据状态"),
        ("phase48-holdout-ordinary_direct_task-01", "普通任务直接回答"),
        ("phase48-holdout-provenance_boundary-01", "来源边界"),
        ("phase48-holdout-privacy_non_echo-01", "隐私值不回显"),
    )
    lines = [
        "# Phase48 Real Output Examples",
        "",
        "以下均来自冻结 64-session holdout 的真实 Qwen3-4B 调用；所有场景均为 simulated_usage，不是实际用户反馈。",
        "",
    ]
    for session_id, title in examples:
        session = sessions[session_id]
        lines.extend(
            [
                f"## {title} ({session_id})",
                "",
                f"- 初始目标：{_safe_text(session.get('user_goal'))}",
                f"- 用户纠正：{_safe_text(session.get('user_correction'))}",
                f"- 最终要求：{_safe_text(session.get('continuation_request'))}",
                "",
            ]
        )
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
        counts = by_comparison.setdefault(comparison, Counter())
        winner = str(result.get("winner") or "")
        counts[f"winner_{winner}"] += 1
        counts["candidate_left" if mapping.get("variant_left") == mapping.get("candidate") else "candidate_right"] += 1
    return {
        "kind": "phase48_blind_position_diagnostic",
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
        "kind": "phase48_critical_evidence_manifest",
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
    split = _read_json(EVIDENCE_ROOT / "evidence-holdout" / "split_integrity.json")
    training_attempt = _read_json(EVIDENCE_ROOT / "evidence-no-training" / "training_attempt.json")
    preparation = _read_json(EVIDENCE_ROOT / "preparation_decision.json")
    blind_key = _read_json(BLIND_DIR / "blind_variant_key.json").get("items") or []
    independent_results = _read_jsonl(BLIND_DIR / "independent_judge_results.jsonl")

    decision = build_phase48_decision(
        metrics_by_variant=metrics,
        deterministic_blind=deterministic,
        independent_blind=independent,
        calibration=calibration,
        split_integrity=split,
    )
    decision.update(
        {
            "created_at": _utcnow(),
            "formal_runtime_result": "blind_preference_but_objective_gate_failed",
            "phase46_runtime_status": "hold_unchanged",
            "phase47_review_pack_used_for_taxonomy_only": True,
        }
    )
    category = _category_analysis(metrics)
    position = _position_diagnostic(independent_results, blind_key)
    comparison = {
        "kind": "phase48_compact_intent_runtime_ablation_comparison",
        "created_at": _utcnow(),
        "model": "Qwen3-4B",
        "holdout_session_count_per_arm": 64,
        "formal_qwen_real_model_calls": sum(int(row.get("model_call_count") or 0) for row in metrics.values()),
        "independent_gemma_real_model_calls": independent.get("completed_pair_count"),
        "metrics": metrics,
        "core_deltas": {
            "compact_vs_privacy_base": _metric_delta(metrics["base_compact_intent"], metrics["base_privacy"]),
            "compact_vs_full_envelope": _metric_delta(metrics["base_compact_intent"], metrics["base_full_intent"]),
        },
        "category_analysis": category,
        "deterministic_blind": deterministic,
        "independent_blind": independent,
        "blind_position_diagnostic": position,
        "training_attempt": training_attempt,
        "decision": decision,
        "actual_user_feedback_count": 0,
        "simulated_usage": True,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }

    transcript_checks = [_transcript_integrity(name, sessions) for name in VARIANTS]
    phase46 = _canonical_integrity("phase46", PHASE46_ROOT)
    phase47 = _canonical_integrity("phase47", PHASE47_ROOT)
    blind_integrity = _read_json(BLIND_DIR / "blind_integrity_check.json")
    integrity = {
        "kind": "phase48_evidence_integrity",
        "created_at": _utcnow(),
        "passed": (
            all(row["passed"] for row in transcript_checks)
            and phase46["passed"]
            and phase47["passed"]
            and blind_integrity.get("passed") is True
            and independent.get("status") == "completed"
            and int(independent.get("completed_pair_count") or 0) == 128
            and calibration.get("status") == "passed"
            and split.get("passed") is True
            and training_attempt.get("status") == "skipped_by_design"
            and preparation.get("status") == "ready_for_real_runtime_ablation"
            and decision.get("recommendation") == "hold_compact_runtime"
            and decision.get("new_training_allowed") is False
            and decision.get("product_default_change_allowed") is False
        ),
        "phase46_canonical": phase46,
        "phase47_canonical": phase47,
        "transcripts": transcript_checks,
        "blind": blind_integrity,
        "independent_judge_completed_pair_count": independent.get("completed_pair_count"),
        "calibration": {key: calibration.get(key) for key in ("status", "case_count", "precision", "recall")},
        "split_integrity_passed": split.get("passed"),
        "training_status": training_attempt.get("status"),
        "decision": decision.get("recommendation"),
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "auto_training_allowed": False,
        "auto_promotion_allowed": False,
    }

    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase48-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence-failure-analysis" / "category_analysis.json", category)
    _write_json(BLIND_DIR / "position_diagnostic.json", position)
    _write_text(REAL_DIR / "output_examples.md", _output_examples(transcripts, sessions))

    failed_checks = "\n".join(f"- `{name}`" for name in decision["failed_checks"])
    independent_base = dict(dict(independent.get("comparisons") or {}).get("compact_vs_privacy_base") or {})
    independent_full = dict(dict(independent.get("comparisons") or {}).get("compact_vs_full_envelope") or {})
    _write_text(
        EVIDENCE_ROOT / "phase48-final-decision.md",
        f"""# Phase48 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。compact runtime 在总分和两组独立盲评上表现更好，但没有达到冻结的 latest-intent 增益门，并且低于 full envelope 的 latest-intent 下限，因此不进入产品默认路径，也不进入 manual shadow。

## 真实结果

- 三组均完成 64 个多轮 session，每个 session 3 次生成，共 `{comparison['formal_qwen_real_model_calls']}` 次真实 Qwen3-4B 调用；隐私违规、截断和 think 泄漏均为 `0`。
- `base_privacy`：score `{metrics['base_privacy']['user_preference_score']}`，latest-intent `{metrics['base_privacy']['latest_intent_satisfaction_rate']}`，repetition `{metrics['base_privacy']['repetition_rate']}`。
- `base_compact_intent`：score `{metrics['base_compact_intent']['user_preference_score']}`，latest-intent `{metrics['base_compact_intent']['latest_intent_satisfaction_rate']}`，repetition `{metrics['base_compact_intent']['repetition_rate']}`。
- `base_full_intent`：score `{metrics['base_full_intent']['user_preference_score']}`，latest-intent `{metrics['base_full_intent']['latest_intent_satisfaction_rate']}`，old-goal residue `{metrics['base_full_intent']['old_goal_residue_rate']}`，repetition `{metrics['base_full_intent']['repetition_rate']}`。
- compact 对 privacy base：score 增益 `{comparison['core_deltas']['compact_vs_privacy_base']['user_preference_score']}`，latest-intent 增益 `{comparison['core_deltas']['compact_vs_privacy_base']['latest_intent_satisfaction_rate']}`，低于冻结门槛 `0.03`。
- Gemma4 盲评 compact 对 base：`{independent_base.get('candidate_wins')}` 胜、`{independent_base.get('benchmark_wins')}` 负、`{independent_base.get('ties')}` 平，非平局胜率 `{independent_base.get('candidate_non_tie_win_rate')}`。
- Gemma4 盲评 compact 对 full：`{independent_full.get('candidate_wins')}` 胜、`{independent_full.get('benchmark_wins')}` 负、`{independent_full.get('ties')}` 平，非平局胜率 `{independent_full.get('candidate_non_tie_win_rate')}`。
- category 级证据显示 compact 改善 `evidence_status` 和 `ordinary_direct_task`，但 `provenance_boundary` 的 latest-intent 相比 base 下降 `0.125`、相比 full 下降 `0.25`。

## Failed Checks

{failed_checks}

## 产品含义

短契约比 full envelope 更自然、重复更少，盲评偏好明确；但它没有稳定守住来源边界，不能仅凭平均分和 judge 偏好默认启用。Phase48 没有训练、没有新 adapter、没有接入 Hermes，`actual_user_feedback_count=0`，所有场景均为 simulated_usage。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase48-runbook.md",
        """# Phase48 Runbook

## Prepare and freeze

```bash
.venv/bin/python tools/phase48_prepare.py --clean-evidence
.venv/bin/pytest -q tests/test_phase48_compact_intent_runtime.py tests/test_phase47_simulated_user_review.py tests/test_phase46_runtime_first_latest_intent.py tests/test_phase45_privacy_multiturn_preference.py
```

## Real three-arm runtime ablation

```bash
.venv/bin/python tools/phase48_qwen3_4b_generate.py --variant base_privacy --clean
.venv/bin/python tools/phase48_qwen3_4b_generate.py --variant base_compact_intent --clean
.venv/bin/python tools/phase48_qwen3_4b_generate.py --variant base_full_intent --clean
```

All arms use the same Qwen3-4B base, privacy transform, length contract, 64-session fresh holdout, deterministic decoding, and no adapter. Only the latest-intent runtime expression varies.

## Blind evaluation and finalization

```bash
.venv/bin/python tools/phase48_blind_eval.py --resume
.venv/bin/python tools/phase48_finalize_evidence.py
.venv/bin/python tools/phase48_validate.py
```

The independent judge is local Ollama `gemma4:31b` with `think=false`. Phase48 permits neither training nor automatic promotion.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

Do not train or attach an adapter yet. Preserve the Phase45 privacy transformer and Phase48 compact contract as held candidates. In Phase49, isolate the provenance-boundary regression: simulate a focused reviewer pass over the failed evidence/provenance sessions, derive one minimal evidence-boundary clause, and compare compact-v1 against compact-v2 on another fresh holdout. Keep privacy, length, model, and decoding frozen. Require category-level provenance recovery without losing the compact runtime's blind preference or ordinary-task directness. Even a passing result may only enter a manual shadow recommendation; simulated usage remains separate from actual user feedback.
""",
    )
    _write_json(
        EVIDENCE_ROOT / "finalization_state.json",
        {
            "kind": "phase48_finalization_state",
            "created_at": _utcnow(),
            "decision": decision["recommendation"],
            "evidence_integrity_passed": integrity["passed"],
            "formal_qwen_real_model_calls": comparison["formal_qwen_real_model_calls"],
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
