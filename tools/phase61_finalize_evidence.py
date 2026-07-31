#!/usr/bin/env python3
"""Finalize Phase61 compact-wire evidence for every gated outcome."""

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

from pfe_core.phase61_compact_candidate_wire_protocol import build_phase61_decision


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase61-compact-candidate-wire-protocol"
ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
DYNAMIC_FILES = {"evidence_manifest.json", "finalization_state.json", "validation_gate.txt", "validation_summary.json"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _stage_counts(directory: Path) -> dict[str, int]:
    successes = sum(len(_read_jsonl(directory / f"judge_wire_results_{alias}.jsonl")) for alias in ALIASES)
    failure_attempts = sum(len(_read_jsonl(directory / f"wire_failure_attempts_{alias}.jsonl")) for alias in ALIASES)
    return {"successful_model_output_count": successes, "raw_failure_attempt_count": failure_attempts}


def _failure_rows(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for alias in ALIASES:
        rows.extend(_read_jsonl(directory / f"wire_failure_attempts_{alias}.jsonl"))
    return rows


def _raw_failures_preserved(rows: list[dict[str, Any]]) -> bool:
    return all(
        "raw_response" in row
        and row.get("raw_response_sha256")
        == hashlib.sha256(str(row.get("raw_response") or "").encode("utf-8")).hexdigest()
        for row in rows
    )


def _failure_analysis(report: Mapping[str, Any]) -> dict[str, Any]:
    failures = [dict(row) for row in report.get("details") or [] if row.get("passed") is False]
    transitions: dict[str, int] = {}
    wrong_selections = []
    for row in failures:
        transition = f"{row.get('expected_label')}->{row.get('actual_label')}"
        transitions[transition] = transitions.get(transition, 0) + 1
        expected = dict(row.get("expected_candidate_ids") or {})
        for selection in row.get("grounded_selections") or []:
            for field in ("source_registration", "user_outcome_status", "test_to_user_outcome_relation"):
                actual = selection.get(f"{field}_candidate_id")
                if actual == expected.get(field):
                    continue
                wrong_selections.append(
                    {
                        "item_id": row.get("item_id"),
                        "case_id": row.get("case_id"),
                        "category": row.get("category"),
                        "judge_alias": selection.get("judge_alias"),
                        "field": field,
                        "expected_candidate_id": expected.get(field),
                        "actual_candidate_id": actual,
                    }
                )
    return {
        "kind": "phase61_compact_wire_failure_analysis",
        "label_failure_count": len(failures),
        "wrong_candidate_selection_count": len(wrong_selections),
        "label_transition_counts": dict(sorted(transitions.items())),
        "post_model_tuning_performed": False,
        "label_failures": failures,
        "wrong_candidate_selections": wrong_selections,
    }


def _manifest() -> dict[str, Any]:
    files = []
    for path in sorted(EVIDENCE_ROOT.rglob("*")):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        files.append(
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"kind": "phase61_evidence_manifest", "file_count": len(files), "files": files, "manifest_sha256": digest}


def main() -> int:
    preflight_dir = EVIDENCE_ROOT / "evidence-wire-preflight"
    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    baseline = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase60_canonical_snapshot.json")
    preflight = _read_json(preflight_dir / "preflight_report.json")
    calibration = _read_json(calibration_dir / "candidate_evaluator_report.json")
    holdout = _read_json(holdout_dir / "candidate_evaluator_report.json")
    split = _read_json(holdout_dir / "split_integrity.json")
    calibration_audit = _read_json(calibration_dir / "fixture_semantic_audit.json")
    holdout_audit = _read_json(holdout_dir / "fixture_semantic_audit.json")
    hard_calibration = _read_json(calibration_dir / "hard_rule_compatibility.json")
    hard_holdout = _read_json(holdout_dir / "hard_rule_compatibility.json")
    runtime = _read_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json")
    training = _read_json(EVIDENCE_ROOT / "evidence-no-training/training_attempt.json")
    protocol = _read_json(EVIDENCE_ROOT / "evaluator_protocol.json")
    decision = build_phase61_decision(
        phase60_snapshot=baseline,
        preflight_report=preflight,
        calibration_report=calibration,
        holdout_report=holdout,
        calibration_audit=calibration_audit,
        holdout_audit=holdout_audit,
        hard_calibration=hard_calibration,
        hard_holdout=hard_holdout,
        split_integrity=split,
    )
    preflight_counts = _stage_counts(preflight_dir)
    calibration_counts = _stage_counts(calibration_dir)
    holdout_counts = _stage_counts(holdout_dir)
    preflight_failed_items = int(preflight.get("failed_judge_item_count") or 0)
    calibration_failed_items = int(calibration.get("failure_count") or 0)
    holdout_failed_items = int(holdout.get("failure_count") or 0)
    preflight_outcomes = preflight_counts["successful_model_output_count"] + preflight_failed_items
    calibration_outcomes = calibration_counts["successful_model_output_count"] + calibration_failed_items
    holdout_outcomes = holdout_counts["successful_model_output_count"] + holdout_failed_items
    preflight_passed = preflight.get("status") == "passed"
    calibration_qualified = calibration.get("status") == "qualified"
    all_raw_failures = _failure_rows(preflight_dir) + _failure_rows(calibration_dir) + _failure_rows(holdout_dir)
    raw_failures_preserved = _raw_failures_preserved(all_raw_failures)
    call_evidence_complete = (
        preflight_outcomes == 12
        and ((preflight_passed and calibration_outcomes == 60) or (not preflight_passed and calibration_outcomes == 0))
        and (
            (preflight_passed and calibration_qualified and holdout_outcomes == 120)
            or (not (preflight_passed and calibration_qualified) and holdout_outcomes == 0)
        )
    )
    comparison = {
        "kind": "phase61_compact_wire_comparison",
        "phase60": {
            "recommendation": baseline.get("phase60_recommendation"),
            "preflight_status": baseline.get("phase60_preflight_status"),
            "successful_model_output_count": baseline.get("phase60_successful_model_output_count"),
            "failed_judge_item_count": baseline.get("phase60_failed_judge_item_count"),
            "raw_failure_attempt_count": baseline.get("phase60_raw_failure_attempt_count"),
            "raw_failures_preserved": baseline.get("phase60_raw_failures_preserved"),
        },
        "phase61": {
            "wire_preflight_status": preflight.get("status") or "not_run",
            "wire_preflight_successful_model_output_count": preflight_counts["successful_model_output_count"],
            "wire_preflight_failed_judge_item_count": preflight_failed_items,
            "wire_preflight_raw_failure_attempt_count": preflight_counts["raw_failure_attempt_count"],
            "calibration_status": calibration.get("status") or "not_run_after_preflight_failure",
            "calibration_accuracy": calibration.get("accuracy"),
            "calibration_typed_exact_match_rate": calibration.get("typed_exact_match_rate"),
            "calibration_candidate_selection_exact_match_rate": calibration.get("candidate_selection_exact_match_rate"),
            "calibration_successful_model_output_count": calibration_counts["successful_model_output_count"],
            "calibration_failed_judge_item_count": calibration_failed_items,
            "calibration_raw_failure_attempt_count": calibration_counts["raw_failure_attempt_count"],
            "holdout_status": holdout.get("status") or "not_run_after_prior_gate_failure",
            "holdout_accuracy": holdout.get("accuracy"),
            "holdout_typed_exact_match_rate": holdout.get("typed_exact_match_rate"),
            "holdout_candidate_selection_exact_match_rate": holdout.get("candidate_selection_exact_match_rate"),
            "holdout_successful_model_output_count": holdout_counts["successful_model_output_count"],
            "holdout_failed_judge_item_count": holdout_failed_items,
            "holdout_raw_failure_attempt_count": holdout_counts["raw_failure_attempt_count"],
            "raw_wire_failures_preserved": raw_failures_preserved,
        },
        "recommendation": decision["recommendation"],
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "runtime_replay_executed": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }
    integrity_checks = {
        "phase60_canonical_snapshot_passed": baseline.get("passed") is True,
        "split_integrity_passed": split.get("passed") is True,
        "fixture_semantic_audits_passed": (
            calibration_audit.get("status") == "passed" and holdout_audit.get("status") == "passed"
        ),
        "preflight_freeze_check_passed": _read_json(preflight_dir / "freeze_check.json").get("passed") is True,
        "call_evidence_complete_for_gated_path": call_evidence_complete,
        "raw_failure_attempts_preserved_and_hashed": raw_failures_preserved,
        "calibration_freeze_check_consistent": (
            _read_json(calibration_dir / "freeze_check.json").get("passed") is True
            if preflight_passed else calibration_outcomes == 0
        ),
        "holdout_freeze_check_consistent": (
            _read_json(holdout_dir / "freeze_check.json").get("passed") is True
            if preflight_passed and calibration_qualified else holdout_outcomes == 0
        ),
        "no_post_model_call_tuning": protocol.get("post_model_call_tuning_allowed") is False,
        "runtime_not_run": int(runtime.get("runtime_replay_model_call_count") or 0) == 0,
        "training_not_run": training.get("training_executed") is False and training.get("adapter_created") is False,
        "no_training_adapter_hermes_or_default_change": (
            decision["new_training_allowed"] is False
            and decision["new_adapter_created"] is False
            and decision["hermes_attachment_allowed"] is False
            and decision["product_default_change_allowed"] is False
        ),
    }
    integrity = {
        "kind": "phase61_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    if calibration:
        _write_json(calibration_dir / "failure_analysis.json", _failure_analysis(calibration))
    if holdout:
        _write_json(holdout_dir / "failure_analysis.json", _failure_analysis(holdout))
    if all_raw_failures:
        _write_json(
            EVIDENCE_ROOT / "wire_protocol_failure_evidence.json",
            {
                "kind": "phase61_wire_protocol_failure_evidence",
                "status": "preserved",
                "raw_failure_attempt_count": len(all_raw_failures),
                "raw_failures_preserved_and_hashed": raw_failures_preserved,
                "failure_classes": sorted({str(row.get("failure_class") or "") for row in all_raw_failures}),
                "failed_judge_aliases": sorted({str(row.get("judge_alias") or "") for row in all_raw_failures}),
                "post_failure_protocol_change_performed": False,
            },
        )
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase61-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_text(
        EVIDENCE_ROOT / "phase61-final-decision.md",
        f"""# Phase61 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。Compact-wire preflight 为 `{preflight.get('status')}`，calibration accuracy 为 `{calibration.get('accuracy')}`，holdout accuracy 为 `{holdout.get('accuracy')}`。

## 冻结边界

- Phase61 只把 Phase60 的输出传输改成 `PFE1|source|outcome|relation`；候选生成、typed semantics、hard detector、composer 与门槛均未修改。
- 6 条双模型 wire preflight 先于 30 条 calibration；holdout 仅在 calibration qualified 后运行。
- Wire parser 只接受固定顺序、固定数量的 candidate ID；不接受 JSON、YAML、解释或候选含义复制。
- 每个失败调用最多重试两次，原始无效响应与 SHA-256 必须保留。
- 所有输入都是 simulated evaluator fixtures，不是 actual user feedback，也不用于训练。
- Phase61 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase61-runbook.md",
        """# Phase61 Runbook

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

In a second terminal:

```bash
.venv/bin/python tools/phase61_prepare.py --clean-evidence
.venv/bin/python tools/phase61_execute.py --stage preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase61_execute.py --stage calibration --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase61_execute.py --stage holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase61_finalize_evidence.py
.venv/bin/python tools/phase61_validate.py
```

Do not change candidate generation, fixtures, prompts, wire parser, retry count, or gates after prepare. Stop after the first failed gate and finalize that path.
""",
    )
    if decision["phase62_external_replay_design_eligible"]:
        next_goal = (
            "Build Phase62 as a frozen historical replay of the manually reviewable Phase61 evaluator. Preserve the compact wire "
            "protocol, candidate generation, composer, and gates. Require per-phase accuracy, zero false accepts, zero unsupported "
            "candidates, and no hard-rule conflict before considering a minimal runtime A/B. Do not train, attach Hermes, change "
            "defaults, auto-promote, or claim actual user benefit."
        )
    else:
        next_goal = (
            "Hold Phase61. Analyze only aggregate frozen wire or candidate-selection failures, choose one structural correction, and "
            "freeze fresh fixtures before more model calls. Do not run runtime A/B, train, attach Hermes, change defaults, or relax gates."
        )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}\n")
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase61_finalization_state",
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
