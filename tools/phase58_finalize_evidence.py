#!/usr/bin/env python3
"""Finalize Phase58 evidence without runtime or training changes."""

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

from pfe_core.phase58_clause_addressed_grounding import build_phase58_decision


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase58-clause-addressed-grounding"
PHASE57_ROOT = REPO_ROOT / "docs/demo/phase57-span-evaluator-historical-replay"
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


def _call_count(directory: Path) -> int:
    return sum(len(_read_jsonl(directory / f"judge_clause_results_{alias}.jsonl")) for alias in ALIASES)


def _failure_analysis(report: Mapping[str, Any]) -> dict[str, Any]:
    label_failures = [dict(row) for row in report.get("details") or [] if row.get("passed") is False]
    invalid_atoms = []
    transitions: dict[str, int] = {}
    for row in report.get("details") or []:
        if row.get("passed") is False:
            key = f"{row.get('expected_label')}->{row.get('actual_label')}"
            transitions[key] = transitions.get(key, 0) + 1
        for extraction in row.get("grounded_extractions") or []:
            for field in ("source_registration", "user_outcome_status", "test_to_user_outcome_relation"):
                if extraction.get(f"{field}_grounded") is True:
                    continue
                invalid_atoms.append(
                    {
                        "item_id": row.get("item_id"),
                        "case_id": row.get("case_id"),
                        "category": row.get("category"),
                        "judge_alias": extraction.get("judge_alias"),
                        "field": field,
                        "raw_value": extraction.get(f"raw_{field}"),
                        "effective_value": extraction.get(field),
                        "clause_id": extraction.get(f"{field}_clause_id"),
                        "grounding_reason": extraction.get(f"{field}_grounding_reason"),
                        "conservative_reject": extraction.get("conservative_reject") is True,
                    }
                )
    return {
        "kind": "phase58_clause_addressed_failure_analysis",
        "label_failure_count": len(label_failures),
        "invalid_atom_count": len(invalid_atoms),
        "invalid_dangerous_atom_count": int(report.get("invalid_dangerous_atom_count") or 0),
        "label_transition_counts": dict(sorted(transitions.items())),
        "post_model_tuning_performed": False,
        "label_failures": label_failures,
        "invalid_atoms": invalid_atoms,
    }


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
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"kind": "phase58_evidence_manifest", "file_count": len(files), "files": files, "manifest_sha256": digest}


def main() -> int:
    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    calibration = _read_json(calibration_dir / "clause_evaluator_report.json")
    holdout = _read_json(holdout_dir / "clause_evaluator_report.json")
    baseline = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase57_canonical_snapshot.json")
    split = _read_json(holdout_dir / "split_integrity.json")
    hard_calibration = _read_json(calibration_dir / "hard_reject_report.json")
    hard_holdout = _read_json(holdout_dir / "hard_reject_report.json")
    runtime = _read_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json")
    decision = build_phase58_decision(
        phase57_snapshot=baseline,
        calibration_report=calibration,
        holdout_report=holdout,
        hard_calibration=hard_calibration,
        hard_holdout=hard_holdout,
        split_integrity=split,
    )
    calibration_calls = _call_count(calibration_dir)
    holdout_calls = _call_count(holdout_dir)
    calibration_qualified = calibration.get("status") == "qualified"
    call_evidence_complete = calibration_calls == 60 and (
        (calibration_qualified and holdout_calls == 120) or (not calibration_qualified and holdout_calls == 0)
    )
    comparison = {
        "kind": "phase58_clause_addressed_comparison",
        "phase57": {
            "recommendation": baseline.get("phase57_recommendation"),
            "historical_accuracy": baseline.get("phase57_accuracy"),
            "grounding_validity_rate": baseline.get("phase57_grounding_validity_rate"),
            "invalid_atom_count": baseline.get("phase57_invalid_atom_count"),
            "invalid_dangerous_atom_count": baseline.get("phase57_invalid_dangerous_atom_count"),
        },
        "phase58": {
            "calibration_status": calibration.get("status"),
            "calibration_accuracy": calibration.get("accuracy"),
            "calibration_grounding_validity_rate": calibration.get("grounding_validity_rate"),
            "holdout_status": holdout.get("status") or "not_run_after_calibration_failure",
            "holdout_accuracy": holdout.get("accuracy"),
            "holdout_typed_exact_match_rate": holdout.get("typed_exact_match_rate"),
            "holdout_grounding_validity_rate": holdout.get("grounding_validity_rate"),
            "holdout_invalid_atom_count": holdout.get("invalid_atom_count"),
            "holdout_invalid_dangerous_atom_count": holdout.get("invalid_dangerous_atom_count"),
            "holdout_false_accept_count": holdout.get("false_accept_count_on_reject_cases"),
            "calibration_real_model_call_count": calibration_calls,
            "holdout_real_model_call_count": holdout_calls,
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
        "phase57_canonical_snapshot_passed": baseline.get("passed") is True,
        "split_integrity_passed": split.get("passed") is True,
        "calibration_freeze_check_passed": _read_json(calibration_dir / "freeze_check.json").get("passed") is True,
        "call_evidence_complete_for_frozen_gate": call_evidence_complete,
        "holdout_freeze_check_consistent": (
            _read_json(holdout_dir / "freeze_check.json").get("passed") is True
            if calibration_qualified else holdout_calls == 0
        ),
        "no_post_model_tuning": _read_json(EVIDENCE_ROOT / "evaluator_protocol.json").get("post_calibration_tuning_allowed") is False,
        "runtime_not_run": int(runtime.get("runtime_replay_model_call_count") or 0) == 0,
        "no_training_adapter_hermes_or_default_change": (
            decision["new_training_allowed"] is False
            and decision["new_adapter_created"] is False
            and decision["hermes_attachment_allowed"] is False
            and decision["product_default_change_allowed"] is False
        ),
    }
    integrity = {
        "kind": "phase58_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    _write_json(calibration_dir / "failure_analysis.json", _failure_analysis(calibration))
    if holdout:
        _write_json(holdout_dir / "failure_analysis.json", _failure_analysis(holdout))
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase58-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_text(
        EVIDENCE_ROOT / "phase58-final-decision.md",
        f"""# Phase58 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。Calibration accuracy 为 `{calibration.get('accuracy')}`，holdout accuracy 为 `{holdout.get('accuracy')}`，holdout grounding validity 为 `{holdout.get('grounding_validity_rate')}`。

## 冻结边界

- Phase58 只把自由文本 evidence span 改为不可变 clause ID；Phase53 hard detector 与 Phase56 composer 未修改。
- 全新 30 条 calibration 和 60 条 holdout 在任何模型调用前冻结，holdout 仅在 calibration qualified 后运行。
- 两位 judge 看不到 gold label、typed atoms、category 或 gold clause IDs，也不直接返回 label。
- 所有输入均为 simulated evaluator fixtures，不是 actual user feedback，不用于训练。
- Phase58 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径，也不声称实际产品收益。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase58-runbook.md",
        """# Phase58 Runbook

Start an isolated four-slot Ollama service in terminal A:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

Run the frozen gate in terminal B:

```bash
.venv/bin/python tools/phase58_prepare.py --clean-evidence
.venv/bin/python tools/phase58_clause_evaluator.py --split calibration --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase58_clause_evaluator.py --split holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase58_finalize_evidence.py
.venv/bin/python tools/phase58_validate.py
```

Do not modify the rubric, schema, fixture, grounding code, or composer after any calibration call. A failed calibration seals Phase58 without holdout.
""",
    )
    next_goal = (
        "Build Phase59 as one minimal shadow-only runtime A/B using the manually reviewable Phase58 evaluator. Freeze a small "
        "simulated-usage set before calls, compare the current runtime contract with exactly one candidate, and require no safety, "
        "citation, provenance, or boundary regression plus at least one measurable improvement. Do not train, attach Hermes, change "
        "the product default, auto-promote, or claim actual user benefit."
        if decision["phase59_minimal_runtime_ab_design_eligible"]
        else
        "Hold the Phase58 evaluator. Analyze only aggregate frozen failure classes, choose at most one structural correction for a "
        "new phase, and freeze new calibration and holdout before further model calls. Do not run runtime A/B, train, attach Hermes, "
        "change the product default, or relax the gates."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}\n")
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase58_finalization_state",
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
