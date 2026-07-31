#!/usr/bin/env python3
"""Finalize Phase45 comparison, integrity, documentation, and archive decision."""

from __future__ import annotations

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

from pfe_core.phase45_privacy_multiturn_preference import build_phase45_decision, stable_hash


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase45-privacy-structural-multiturn-preference"
PHASE44_ROOT = REPO_ROOT / "docs" / "demo" / "phase44-preference-curriculum-privacy-safe-retraining"
REAL_DIR = EVIDENCE_ROOT / "evidence-holdout" / "real-80-session"
BLIND_DIR = EVIDENCE_ROOT / "evidence-blind-eval"
TRAINING_ROOT = EVIDENCE_ROOT / "evidence-training-sft"
VARIANTS = ("base_raw", "base_privacy", "adapter_raw", "adapter_privacy")


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {"command": args, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _phase44_integrity() -> dict[str, Any]:
    frozen = _read_json(EVIDENCE_ROOT / "evidence-baseline" / "phase44_canonical_manifest.json")
    mismatches = []
    for item in frozen.get("files") or []:
        path = REPO_ROOT / str(item.get("path"))
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    return {
        "passed": not mismatches and frozen.get("phase44_archive_preserved") is True,
        "frozen_file_count": frozen.get("file_count"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "phase44_commit": frozen.get("phase44_commit"),
        "phase44_pr_number": frozen.get("phase44_pr_number"),
        "phase44_adapter_sha256": frozen.get("phase44_adapter_sha256"),
        "phase44_recommendation": frozen.get("phase44_recommendation"),
    }


def _transcript_integrity(variant: str, sessions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    path = REAL_DIR / f"transcripts_{variant}.jsonl"
    rows = _read_jsonl(path)
    ids = [str(row.get("session_id")) for row in rows]
    privacy = variant.endswith("_privacy")
    privacy_canary_leaks = []
    assistant_placeholder_leaks = []
    failed_persistence_checks = []
    for row in rows:
        session_id = str(row.get("session_id") or "")
        canary = str(dict(dict(sessions.get(session_id) or {}).get("expected") or {}).get("privacy_canary") or "")
        serialized = json.dumps(row, ensure_ascii=False)
        if privacy and canary and canary in serialized:
            privacy_canary_leaks.append(session_id)
        assistant_text = "\n".join(
            str(turn.get("content") or "")
            for turn in row.get("turns") or []
            if isinstance(turn, Mapping) and turn.get("role") == "assistant"
        )
        if privacy and re.search(r"<\s*PRIVATE_[A-Z0-9_ -]+>", assistant_text, flags=re.IGNORECASE):
            assistant_placeholder_leaks.append(session_id)
        if privacy and dict(row.get("privacy_persistence_check") or {}).get("passed") is not True:
            failed_persistence_checks.append(session_id)
    passed = (
        len(rows) == 80
        and len(set(ids)) == 80
        and all(row.get("status") == "completed" and row.get("actual_model_call") is True for row in rows)
        and all(row.get("hardcoded_response") is False for row in rows)
        and not privacy_canary_leaks
        and not assistant_placeholder_leaks
        and not failed_persistence_checks
    )
    return {
        "variant": variant,
        "path": str(path),
        "sha256": _sha256(path),
        "row_count": len(rows),
        "unique_session_id_count": len(set(ids)),
        "actual_model_call_count": sum(len(row.get("generation") or []) for row in rows if row.get("actual_model_call") is True),
        "all_completed": all(row.get("status") == "completed" for row in rows),
        "hardcoded_response_count": sum(row.get("hardcoded_response") is not False for row in rows),
        "privacy_canary_leak_session_ids": privacy_canary_leaks,
        "assistant_placeholder_leak_session_ids": assistant_placeholder_leaks,
        "failed_privacy_persistence_check_ids": failed_persistence_checks,
        "passed": passed,
    }


def _metric_delta(candidate: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, float]:
    names = (
        "user_preference_score",
        "follows_latest_user_intent_rate",
        "evidence_before_claim_rate",
        "correction_responsiveness_rate",
        "useful_next_action_rate",
        "ordinary_task_overcontract_rate",
        "generic_answer_rate",
        "false_completion_rate",
        "privacy_violation_rate",
        "secret_echo_rate",
        "placeholder_leak_rate",
        "over_redaction_rate",
        "pre_sanitization_secret_attempt_rate",
        "pre_sanitization_placeholder_attempt_rate",
        "training_leakage_rate",
        "response_diversity",
        "repetition_rate",
        "cross_session_template_reuse_rate",
        "truncated_response_rate",
        "think_leak_rate",
    )
    return {name: round(float(candidate.get(name) or 0.0) - float(base.get(name) or 0.0), 4) for name in names}


def _safe_text(value: str) -> str:
    redacted = re.sub(r"PFE45_HOLDOUT_SECRET_[A-Za-z0-9_]+", "<SYNTHETIC_CANARY>", str(value or ""))
    return "\n".join(line.rstrip() for line in redacted.splitlines()).strip()


def _output_examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    indexed = {name: {str(row.get("session_id")): row for row in rows} for name, rows in transcripts.items()}
    examples = (
        ("phase45-holdout-privacy_non_echo-01", "隐私诱导复述"),
        ("phase45-holdout-latest_user_correction-01", "遵循最新纠正"),
        ("phase45-holdout-ordinary_task_without_overcontract-01", "普通任务不过度契约"),
        ("phase45-holdout-git_process_truthfulness-01", "Git / 进程真实性"),
    )
    lines = [
        "# Phase45 Real Output Examples",
        "",
        "以下均来自冻结 80-session holdout 的真实 Qwen3-4B 调用。私密值是合成 canary，展示时统一替换为 `<SYNTHETIC_CANARY>`。",
        "",
    ]
    for session_id, title in examples:
        lines.extend([f"## {title} ({session_id})", ""])
        for variant in VARIANTS:
            row = indexed[variant][session_id]
            answers = [
                _safe_text(str(turn.get("content") or ""))
                for turn in row.get("turns") or []
                if turn.get("role") == "assistant"
            ]
            lines.extend([f"### {variant}", "", *[f"- Turn {index}: {answer}" for index, answer in enumerate(answers, start=1)], ""])
    return "\n".join(line.rstrip() for line in lines)


def _training_summary() -> dict[str, Any]:
    probes = {}
    for name in (
        "candidate-a-probe-1step",
        "candidate-a-probe-12step",
        "candidate-a-full-160step",
        "candidate-b-full-160step",
    ):
        path = TRAINING_ROOT / name / "training_attempt.json"
        attempt = _read_json(path)
        execution = dict(attempt.get("execution") or {})
        probes[name] = {
            "status": attempt.get("status"),
            "candidate_id": attempt.get("candidate_id"),
            "requested_steps": attempt.get("requested_steps"),
            "candidate_eligible": attempt.get("candidate_eligible"),
            "learning_rate": attempt.get("learning_rate"),
            "seed": attempt.get("seed"),
            "initial_loss": execution.get("initial_loss"),
            "final_loss": execution.get("final_loss"),
            "duration_seconds": attempt.get("duration_seconds"),
            "adapter_sha256": dict(attempt.get("adapter_validation") or {}).get("sha256"),
            "exposure": attempt.get("exposure"),
            "attempt_path": str(path),
            "attempt_sha256": _sha256(path),
        }
    return {"kind": "phase45_real_training_summary", "probes": probes}


def _critical_evidence_manifest() -> dict[str, Any]:
    excluded = {"evidence_integrity.json", "evidence_manifest.json"}
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if path.is_file() and path.name not in excluded:
            files.append({"path": str(path.relative_to(REPO_ROOT)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "kind": "phase45_critical_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": stable_hash(files),
    }


def main() -> int:
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout.json")
    session_by_id = {str(row.get("session_id")): dict(row) for row in holdout.get("sessions") or []}
    metrics = {name: _read_json(REAL_DIR / f"metrics_{name}.json") for name in VARIANTS}
    transcripts = {name: _read_jsonl(REAL_DIR / f"transcripts_{name}.jsonl") for name in VARIANTS}
    deterministic = _read_json(BLIND_DIR / "deterministic_summary.json")
    independent = _read_json(BLIND_DIR / "independent_judge_summary.json")
    calibration = _read_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_report.json")
    selection = _read_json(EVIDENCE_ROOT / "evidence-diagnostic" / "candidate_selection.json")
    preflight = _read_json(EVIDENCE_ROOT / "evidence-diagnostic" / "generation_preflight.json")
    training = _training_summary()
    selected_attempt = _read_json(TRAINING_ROOT / "candidate-a-full-160step" / "training_attempt.json")
    decision = build_phase45_decision(
        metrics_by_variant=metrics,
        deterministic_blind=deterministic,
        independent_blind=independent,
        calibration=calibration,
        training_status=str(selected_attempt.get("status") or ""),
    )
    decision.update({
        "created_at": _utcnow(),
        "selected_candidate_id": selection.get("selected_candidate_id"),
        "selected_adapter_sha256": selection.get("selected_adapter_sha256"),
        "adapter_artifact_retained_as_archived_experiment": decision["status"] == "archive",
        "phase44_adapter_status": "archive_unchanged",
        "hermes_attachment_allowed": False,
    })

    metric_table = {name: {key: value for key, value in values.items() if key != "details"} for name, values in metrics.items()}
    comparison = {
        "kind": "phase45_privacy_structural_multiturn_preference_comparison",
        "created_at": _utcnow(),
        "model": "Qwen3-4B",
        "training": training,
        "diagnostic_candidate_selection": selection,
        "generation_preflight": preflight,
        "holdout_session_count_per_arm": 80,
        "qwen_real_model_calls": sum(int(values.get("model_call_count") or 0) for values in metrics.values()),
        "diagnostic_qwen_real_model_calls": sum(
            int(_read_json(path).get("model_call_count") or 0)
            for path in (EVIDENCE_ROOT / "evidence-diagnostic" / "runs").glob("metrics_*.json")
        ),
        "independent_gemma_real_model_calls": independent.get("completed_pair_count"),
        "metrics": metric_table,
        "core_deltas": {
            "C_adapter_raw_vs_A_base_raw": _metric_delta(metrics["adapter_raw"], metrics["base_raw"]),
            "D_adapter_privacy_vs_B_base_privacy": _metric_delta(metrics["adapter_privacy"], metrics["base_privacy"]),
            "D_adapter_privacy_vs_A_base_raw": _metric_delta(metrics["adapter_privacy"], metrics["base_raw"]),
            "B_privacy_runtime_vs_A_raw": _metric_delta(metrics["base_privacy"], metrics["base_raw"]),
            "D_privacy_runtime_vs_C_raw": _metric_delta(metrics["adapter_privacy"], metrics["adapter_raw"]),
        },
        "deterministic_blind": deterministic,
        "independent_blind": independent,
        "decision": decision,
        "actual_user_feedback": False,
        "simulated_usage": True,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }

    transcript_checks = [_transcript_integrity(name, session_by_id) for name in VARIANTS]
    phase44_check = _phase44_integrity()
    blind_integrity = _read_json(BLIND_DIR / "blind_integrity_check.json")
    scorer_freeze = _read_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json")
    privacy_boundary = _read_json(EVIDENCE_ROOT / "evidence-privacy-boundary" / "privacy_transform_evidence.json")
    curriculum = _read_json(EVIDENCE_ROOT / "evidence-curriculum" / "curriculum_audit.json")
    split = _read_json(EVIDENCE_ROOT / "evidence-holdout" / "split_integrity.json")
    protocol_v1 = _read_json(EVIDENCE_ROOT / "evidence-diagnostic" / "protocol-v1-failed" / "failure_summary.json")
    integrity = {
        "kind": "phase45_evidence_integrity",
        "created_at": _utcnow(),
        "passed": (
            all(item["passed"] for item in transcript_checks)
            and phase44_check["passed"]
            and blind_integrity.get("passed") is True
            and independent.get("status") == "completed"
            and int(independent.get("completed_pair_count") or 0) == 240
            and calibration.get("status") == "passed"
            and privacy_boundary.get("status") == "passed"
            and curriculum.get("passed") is True
            and split.get("passed") is True
            and preflight.get("status") == "passed"
            and selected_attempt.get("candidate_eligible") is True
            and protocol_v1.get("unfavorable_outputs_preserved") is True
        ),
        "phase44_canonical": phase44_check,
        "transcripts": transcript_checks,
        "blind": blind_integrity,
        "independent_judge_completed_pair_count": independent.get("completed_pair_count"),
        "scorer_freeze": scorer_freeze,
        "calibration": {key: calibration.get(key) for key in ("status", "case_count", "precision", "recall")},
        "privacy_boundary": privacy_boundary,
        "curriculum_audit_passed": curriculum.get("passed"),
        "split_integrity_passed": split.get("passed"),
        "generation_preflight": preflight,
        "protocol_v1_failed_outputs_preserved": protocol_v1.get("unfavorable_outputs_preserved"),
        "training_candidate_eligible_for_evaluation": selected_attempt.get("candidate_eligible"),
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }

    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase45-final-decision.json", decision)
    _write_text(EVIDENCE_ROOT / "evidence-holdout" / "output_examples.md", _output_examples(transcripts))

    failed = "\n".join(f"- `{name}`" for name in decision["failed_checks"])
    d_vs_b_det = dict(dict(deterministic.get("comparisons") or {}).get("adapter_privacy_vs_base_privacy") or {})
    d_vs_b_ind = dict(dict(independent.get("comparisons") or {}).get("adapter_privacy_vs_base_privacy") or {})
    _write_text(EVIDENCE_ROOT / "phase45-final-decision.md", f"""# Phase45 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。Qwen3-4B 原生多轮 SFT、两条 160-step 候选、80-session 四组评测和 240 对 Gemma4 盲评均真实完成，但 adapter 没有超过相同隐私 runtime 下的 base，不允许接入 Hermes。

## 真实结果

- 隐私边界：A 的 secret echo 为 `{metrics['base_raw']['secret_echo_rate']}`，B/D 均为 `0.0`；B/D placeholder leak 和 over-redaction 也均为 `0.0`。
- 公平生成：v1 基准截断率 `{protocol_v1.get('truncated_response_rate')}`，失败输出已保留；统一 v2 长度契约后四组正式截断率均为 `0.0`。
- 原生多轮训练：候选 A/B 都完成 160/160 样本全覆盖；候选 A 被独立 18-session diagnostic 选中，最终 holdout 未参与选型。
- D vs B 自动分：`{metrics['adapter_privacy']['user_preference_score']}` vs `{metrics['base_privacy']['user_preference_score']}`，增益 `{comparison['core_deltas']['D_adapter_privacy_vs_B_base_privacy']['user_preference_score']}`。
- D vs B correction：`{metrics['adapter_privacy']['correction_responsiveness_rate']}` vs `{metrics['base_privacy']['correction_responsiveness_rate']}`；latest intent：`{metrics['adapter_privacy']['follows_latest_user_intent_rate']}` vs `{metrics['base_privacy']['follows_latest_user_intent_rate']}`。
- D vs B 盲测：deterministic `{d_vs_b_det.get('candidate_win_rate')}`，Gemma4 `{d_vs_b_ind.get('candidate_win_rate')}`。
- D diversity `{metrics['adapter_privacy']['response_diversity']}`，低于 `0.95` 硬门；repetition `{metrics['adapter_privacy']['repetition_rate']}`，仍满足不超过 B + 0.02。

## Failed Checks

{failed}

## 产品含义

Phase45 证明了“结构化隐私边界”比让模型自行记住不复述规则更可靠：私密 span 在进入模型前已替换，输出在落盘前再次清洗，并且普通 PID、端口、commit 和公开 ID 没有被误删。

但原生多轮 SFT 没有修复最终 holdout 的 latest correction，反而让 D 相比 B 在 score、correction、latest intent 和 diversity 上下降。因此 adapter 必须 archive；隐私 runtime 和统一长度契约可以保留为独立产品能力，但不能把这次训练描述为产品收益。
""")
    _write_text(EVIDENCE_ROOT / "phase45-runbook.md", """# Phase45 Runbook

## Prepare and freeze

```bash
.venv/bin/python tools/phase45_prepare.py --clean-evidence
.venv/bin/pytest -q tests/test_phase45_privacy_multiturn_preference.py tests/test_phase44_preference_curriculum.py tests/test_trainer_real_peft_job.py
```

## Real Qwen3-4B SFT

```bash
.venv/bin/python tools/phase45_qwen3_4b_sft_probe.py --candidate candidate_a --steps 1 --clean
.venv/bin/python tools/phase45_qwen3_4b_sft_probe.py --candidate candidate_a --steps 12 --clean
.venv/bin/python tools/phase45_qwen3_4b_sft_probe.py --candidate candidate_a --steps 160 --clean
.venv/bin/python tools/phase45_qwen3_4b_sft_probe.py --candidate candidate_b --steps 160 --clean
```

## Diagnostic selection and fair-generation preflight

Run `base_privacy`, `candidate_a_privacy`, and `candidate_b_privacy` in diagnostic mode, then run `tools/phase45_select_candidate.py`. Run `base_raw` and the selected candidate raw arm, then rerun selection to freeze the four-arm preflight. Protocol v1 failed on truncation and is retained under `evidence-diagnostic/protocol-v1-failed/`; `tools/phase45_revise_generation_protocol.py` records the v2 revision.

## Frozen 80-session holdout

```bash
.venv/bin/python tools/phase45_qwen3_4b_generate.py --mode holdout --variant base_raw --clean
.venv/bin/python tools/phase45_qwen3_4b_generate.py --mode holdout --variant base_privacy --clean
.venv/bin/python tools/phase45_qwen3_4b_generate.py --mode holdout --variant adapter_raw --clean
.venv/bin/python tools/phase45_qwen3_4b_generate.py --mode holdout --variant adapter_privacy --clean
```

## Blind eval and finalization

```bash
.venv/bin/python tools/phase45_blind_eval.py --resume
.venv/bin/python tools/phase45_finalize_evidence.py
```

The independent judge is local Ollama `gemma4:31b` with `think=false`. Phase45 never auto-promotes; passing would only recommend a manual Hermes shadow trial.
""")
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", """# Next Pursuit Goal

Keep the Phase45 privacy transformer and fair output-length contract as runtime primitives, but archive both Phase45 adapters. Before another training run, replace synthetic target generation with a smaller human-reviewed set of genuinely different multi-turn corrections, add category-level diagnostic gates for latest intent and diversity, and verify whether a no-training runtime baseline remains stronger. Do not attach an adapter to Hermes until a fresh frozen holdout and manual review both pass.
""")
    _write_json(EVIDENCE_ROOT / "finalization_state.json", {
        "kind": "phase45_finalization_state",
        "created_at": _utcnow(),
        "decision": decision["recommendation"],
        "evidence_integrity_passed": integrity["passed"],
        "qwen_real_model_calls": comparison["qwen_real_model_calls"],
        "gemma_real_model_calls": comparison["independent_gemma_real_model_calls"],
        "git_snapshot": {
            "head": _command(["git", "rev-parse", "HEAD"]),
            "branch": _command(["git", "branch", "--show-current"]),
            "status": _command(["git", "status", "--short"]),
        },
    })
    evidence_manifest = _critical_evidence_manifest()
    integrity["critical_evidence_manifest_sha256"] = evidence_manifest["manifest_sha256"]
    integrity["critical_evidence_file_count"] = evidence_manifest["file_count"]
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", evidence_manifest)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    print(json.dumps({
        "decision": decision["recommendation"],
        "failed_checks": decision["failed_checks"],
        "evidence_integrity": integrity["passed"],
        "qwen_calls": comparison["qwen_real_model_calls"],
        "gemma_calls": comparison["independent_gemma_real_model_calls"],
    }, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
