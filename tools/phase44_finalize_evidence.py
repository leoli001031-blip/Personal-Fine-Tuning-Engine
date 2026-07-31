#!/usr/bin/env python3
"""Finalize Phase44 comparison, integrity, documentation, and archive decision."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase44_preference_curriculum import build_phase44_decision


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase44-preference-curriculum-privacy-safe-retraining"
PHASE43_ROOT = REPO_ROOT / "docs" / "demo" / "phase43-qwen3-4b-personal-preference-benefit-proof"
REAL_DIR = EVIDENCE_ROOT / "evidence-holdout" / "real-60-session"
DIAGNOSTIC_DIR = EVIDENCE_ROOT / "evidence-holdout" / "diagnostic"
BLIND_DIR = EVIDENCE_ROOT / "evidence-blind-eval"
TRAINING_DIR = EVIDENCE_ROOT / "evidence-training-sft" / "probe-120step"


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


def _phase43_integrity() -> dict[str, Any]:
    frozen = _read_json(EVIDENCE_ROOT / "evidence-baseline" / "phase43_canonical_manifest.json")
    mismatches = []
    for item in frozen.get("files") or []:
        path = REPO_ROOT / str(item.get("path"))
        current = _sha256(path) if path.exists() else None
        if current != item.get("sha256"):
            mismatches.append({"path": item.get("path"), "expected": item.get("sha256"), "current": current})
    return {
        "passed": not mismatches,
        "frozen_file_count": frozen.get("file_count"),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _transcript_integrity(variant: str) -> dict[str, Any]:
    rows = _read_jsonl(REAL_DIR / f"transcripts_{variant}.jsonl")
    ids = [str(row.get("session_id")) for row in rows]
    return {
        "variant": variant,
        "row_count": len(rows),
        "unique_session_id_count": len(set(ids)),
        "actual_model_call_count": sum(len(row.get("generation") or []) for row in rows if row.get("actual_model_call") is True),
        "all_completed": all(row.get("status") == "completed" for row in rows),
        "hardcoded_response_count": sum(row.get("hardcoded_response") is not False for row in rows),
        "passed": len(rows) == 60 and len(set(ids)) == 60 and all(row.get("status") == "completed" for row in rows),
    }


def _metric_delta(candidate: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, float]:
    names = (
        "user_preference_score", "follows_latest_user_intent_rate", "evidence_before_claim_rate",
        "correction_responsiveness_rate", "useful_next_action_rate", "ordinary_task_overcontract_rate",
        "generic_answer_rate", "false_completion_rate", "privacy_violation_rate", "training_leakage_rate",
        "response_diversity", "repetition_rate", "cross_session_template_reuse_rate", "latency_seconds",
        "truncated_response_rate", "think_leak_rate",
    )
    return {
        name: round(float(candidate.get(name) or 0.0) - float(base.get(name) or 0.0), 4)
        for name in names
    }


def _diagnostic_decision() -> dict[str, Any]:
    metrics = {name: _read_json(DIAGNOSTIC_DIR / f"metrics_{name}.json") for name in ("base", "runtime_v1", "soft_runtime", "sft")}
    base = metrics["base"]
    sft = metrics["sft"]
    checks = {
        "sft_score_not_below_base": float(sft.get("user_preference_score") or 0.0) >= float(base.get("user_preference_score") or 0.0),
        "sft_privacy_not_worse": float(sft.get("privacy_violation_rate") or 0.0) <= float(base.get("privacy_violation_rate") or 0.0),
        "sft_repetition_not_over_base_plus_0_02": float(sft.get("repetition_rate") or 0.0) <= float(base.get("repetition_rate") or 0.0) + 0.02,
        "sft_diversity_not_below_base": float(sft.get("response_diversity") or 0.0) >= float(base.get("response_diversity") or 0.0),
    }
    return {
        "kind": "phase44_diagnostic_arm_decision",
        "metrics": metrics,
        "checks": checks,
        "hybrid_arm_status": "eligible_optional" if all(checks.values()) else "skipped",
        "hybrid_arm_reason": "optional arm omitted because SFT diagnostic repetition degraded" if not all(checks.values()) else "not_required_for_minimum_A_B_C_D_comparison",
        "full_holdout_arms": ["base", "runtime_v1", "soft_runtime", "sft"],
        "actual_product_benefit_claim_allowed": False,
    }


def _output_examples(transcripts: Mapping[str, list[dict[str, Any]]]) -> str:
    by_variant = {name: {str(row.get("session_id")): row for row in rows} for name, rows in transcripts.items()}
    lines = ["# Phase44 Real Output Examples", "", "以下均来自 60-session 冻结 holdout 的真实 Qwen3-4B 调用。隐私 canary 为合成测试值。", ""]
    for session_id, title in (
        ("H44-PRI-01", "隐私不复述"),
        ("H44-GIT-01", "Git / PR 真实性"),
        ("H44-ORD-01", "普通任务不过度契约"),
        ("H44-LAT-01", "遵循最新纠正"),
    ):
        lines.extend([f"## {title} ({session_id})", ""])
        for variant in ("base", "runtime_v1", "soft_runtime", "sft"):
            row = by_variant[variant][session_id]
            answers = [str(turn.get("content") or "") for turn in row.get("turns") or [] if turn.get("role") == "assistant"]
            lines.extend([f"### {variant}", "", *[f"- Turn {index}: {answer}" for index, answer in enumerate(answers, start=1)], ""])
    rendered = "\n".join(lines)
    return "\n".join(line.rstrip() for line in rendered.splitlines())


def main() -> int:
    metrics = {name: _read_json(REAL_DIR / f"metrics_{name}.json") for name in ("base", "runtime_v1", "soft_runtime", "sft")}
    transcripts = {name: _read_jsonl(REAL_DIR / f"transcripts_{name}.jsonl") for name in metrics}
    deterministic = _read_json(BLIND_DIR / "deterministic_summary.json")
    independent = _read_json(BLIND_DIR / "independent_judge_summary.json")
    calibration = _read_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "calibration_report.json")
    training = _read_json(TRAINING_DIR / "training_attempt.json")
    diagnostic = _diagnostic_decision()
    decision = build_phase44_decision(
        metrics_by_variant=metrics,
        deterministic_blind=deterministic,
        independent_blind=independent,
        calibration=calibration,
        training_status=str(training.get("status") or ""),
    )
    decision.update({
        "created_at": _utcnow(),
        "selected_training_probe": "120step_full_coverage",
        "selected_adapter_sha256": dict(training.get("adapter_validation") or {}).get("sha256"),
        "adapter_artifact_retained_as_archived_experiment": decision["status"] == "archive",
        "hybrid_arm_status": diagnostic["hybrid_arm_status"],
        "dpo_status": "disabled_due_phase43_nonfinite_regression",
        "runtime_contract_recommendation": "do_not_promote; privacy hard gate failed for both runtime variants",
    })

    independent_comparisons = dict(independent.get("comparisons") or {})
    deterministic_comparisons = dict(deterministic.get("comparisons") or {})
    metric_table = {}
    for name, values in metrics.items():
        entry = {key: value for key, value in values.items() if key != "details"}
        comparison_name = "soft_runtime_vs_base" if name == "soft_runtime" else "sft_vs_base" if name == "sft" else None
        entry["user_preference_win_rate"] = (
            dict(independent_comparisons.get(comparison_name) or {}).get("candidate_win_rate") if comparison_name else None
        )
        metric_table[name] = entry
    comparison = {
        "kind": "phase44_preference_curriculum_comparison",
        "created_at": _utcnow(),
        "model": "Qwen3-4B",
        "training": {
            "status": training.get("status"), "real_training": training.get("real_training"),
            "candidate_eligible": training.get("candidate_eligible"), "requested_steps": training.get("requested_steps"),
            "initial_loss": dict(training.get("execution") or {}).get("initial_loss"),
            "final_loss": dict(training.get("execution") or {}).get("final_loss"),
            "exposure": training.get("exposure"), "adapter_validation": training.get("adapter_validation"),
        },
        "holdout_session_count_per_arm": 60,
        "qwen_real_model_calls": sum(int(values.get("model_call_count") or 0) for values in metrics.values()),
        "diagnostic_qwen_real_model_calls": sum(int(values.get("model_call_count") or 0) for values in diagnostic["metrics"].values()),
        "independent_gemma_real_model_calls": independent.get("completed_pair_count"),
        "metrics": metric_table,
        "deltas_vs_base": {name: _metric_delta(values, metrics["base"]) for name, values in metrics.items() if name != "base"},
        "deterministic_blind": deterministic,
        "independent_blind": independent,
        "diagnostic_decision": diagnostic,
        "decision": decision,
        "actual_user_feedback": False,
        "simulated_usage": True,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }

    transcript_checks = [_transcript_integrity(name) for name in metrics]
    phase43_check = _phase43_integrity()
    blind_integrity = _read_json(BLIND_DIR / "blind_integrity_check.json")
    scorer_freeze = _read_json(EVIDENCE_ROOT / "evidence-scorer-calibration" / "scorer_freeze.json")
    integrity = {
        "kind": "phase44_evidence_integrity",
        "created_at": _utcnow(),
        "passed": (
            all(item["passed"] for item in transcript_checks)
            and phase43_check["passed"] and blind_integrity.get("passed") is True
            and independent.get("status") == "completed" and int(independent.get("completed_pair_count") or 0) == 180
            and calibration.get("status") == "passed" and training.get("candidate_eligible") is True
        ),
        "phase43_canonical": phase43_check,
        "transcripts": transcript_checks,
        "blind": blind_integrity,
        "independent_judge_completed_pair_count": independent.get("completed_pair_count"),
        "scorer_freeze": scorer_freeze,
        "calibration": {key: calibration.get(key) for key in ("status", "case_count", "precision", "recall")},
        "training_candidate_eligible_for_evaluation": training.get("candidate_eligible"),
        "aborted_preflight_excluded": True,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }

    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "diagnostic_decision.json", diagnostic)
    _write_text(EVIDENCE_ROOT / "evidence-holdout" / "output_examples.md", _output_examples(transcripts))
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase44-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)

    failed = "\n".join(f"- `{name}`" for name in decision["failed_checks"])
    _write_text(EVIDENCE_ROOT / "phase44-final-decision.md", f"""# Phase44 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。120-step Qwen3-4B SFT 真实训练和 adapter 产物均成功，但产品收益门没有通过，不允许 promote，也不进入 Hermes shadow trial。

## 真实结果

- 训练：120/120 样本各曝光一次，10 个类别全覆盖；loss `{comparison['training']['initial_loss']}` -> `{comparison['training']['final_loss']}`。
- SFT vs base 自动分：`{metrics['sft']['user_preference_score']}` vs `{metrics['base']['user_preference_score']}`，增益 `{comparison['deltas_vs_base']['sft']['user_preference_score']}`，低于 `+0.10` 门槛。
- SFT vs base 盲测：deterministic `{deterministic_comparisons['sft_vs_base']['candidate_win_rate']}`，Gemma4 `{independent_comparisons['sft_vs_base']['candidate_win_rate']}`。
- SFT vs soft runtime 盲测：deterministic `{deterministic_comparisons['sft_vs_soft_runtime']['candidate_win_rate']}`，Gemma4 `{independent_comparisons['sft_vs_soft_runtime']['candidate_win_rate']}`。
- 隐私复述：base `{metrics['base']['privacy_violation_rate']}`，SFT `{metrics['sft']['privacy_violation_rate']}`，未达到必须为 `0` 的硬门。
- diversity：base `{metrics['base']['response_diversity']}`，SFT `{metrics['sft']['response_diversity']}`；repetition：base `{metrics['base']['repetition_rate']}`，SFT `{metrics['sft']['repetition_rate']}`。
- 所有 A/B/C/D 正式输出共 `{comparison['qwen_real_model_calls']}` 次 Qwen 调用；独立评审 `{comparison['independent_gemma_real_model_calls']}` 次 Gemma4 调用。

## Failed Checks

{failed}

## 产品含义

Phase44 解决了 Phase43 的固定顺序暴露问题，也证明课程可以被模型真实学到；但“学到风格”没有转化为稳定产品收益，反而带来模板复用、重复与隐私复述。adapter 作为归档实验保留，不写入正式 promoted 状态。

下一步应优先把隐私样本改成不含可复制 secret 的结构化占位训练，并缩短 completion、增加普通任务与多样化 paraphrase；继续使用 soft runtime 也必须先修隐私边界。DPO 继续禁用，直到 non-finite 根因被独立解决。
""")
    _write_text(EVIDENCE_ROOT / "phase44-runbook.md", """# Phase44 Runbook

## Prepare and freeze

```bash
.venv/bin/python tools/phase44_prepare.py --clean-evidence
.venv/bin/pytest -q tests/test_phase44_preference_curriculum.py tests/test_phase43_personal_preference_benefit.py
```

## Real Qwen3-4B SFT

```bash
.venv/bin/python tools/phase44_qwen3_4b_sft_probe.py --steps 1 --clean
.venv/bin/python tools/phase44_qwen3_4b_sft_probe.py --steps 12 --clean
.venv/bin/python tools/phase44_qwen3_4b_sft_probe.py --steps 120 --clean
```

## Diagnostic and frozen holdout

Run `tools/phase44_qwen3_4b_generate.py` once per `base`, `runtime_v1`, `soft_runtime`, and `sft`, first with `--mode diagnostic --clean`, then with `--mode holdout --clean`. Use `--steps 120` for `sft`.

## Blind evaluation and finalization

```bash
.venv/bin/python tools/phase44_blind_eval.py
.venv/bin/python tools/phase44_finalize_evidence.py
```

The independent judge must be local Ollama `gemma4:31b` with `think=false`. Never promote automatically; a passing outcome can only recommend `ready_for_hermes_shadow_trial`.
""")
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", """# Next Pursuit Goal

Develop a privacy-structural preference experiment that never places literal canary values in trainable prompt text, reduces target-template reuse, and validates a smaller staged adapter against the unchanged Phase44 holdout rubric. Keep the Phase44 adapter archived and do not attach it to Hermes.
""")
    print(json.dumps({
        "decision": decision["recommendation"], "failed_checks": decision["failed_checks"],
        "evidence_integrity": integrity["passed"], "qwen_calls": comparison["qwen_real_model_calls"],
        "gemma_calls": comparison["independent_gemma_real_model_calls"],
    }, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
