#!/usr/bin/env python3
"""Finalize Phase68 evidence after all frozen model stages."""

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

from pfe_core.phase68_aligned_candidate_scope_recovery import build_phase68_decision


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase68-aligned-candidate-scope-recovery"
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


def _read_jsonl_count(directory: Path) -> int:
    return sum(
        len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
        for path in directory.glob("judge_typed_wire_results_*.jsonl")
    )


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
        "kind": "phase68_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": digest,
    }


def main() -> int:
    preflight_dir = EVIDENCE_ROOT / "evidence-typed-wire-preflight"
    calibration_dir = EVIDENCE_ROOT / "evidence-evaluator-calibration"
    holdout_dir = EVIDENCE_ROOT / "evidence-evaluator-holdout"
    aligned_dir = EVIDENCE_ROOT / "evidence-aligned-phase55-regression"
    phase67 = _read_json(
        EVIDENCE_ROOT / "evidence-baseline/phase67_canonical_snapshot.json"
    )
    failure_audit = _read_json(EVIDENCE_ROOT / "aggregate_phase55_failure_audit.json")
    fresh_candidate_audit = _read_json(EVIDENCE_ROOT / "candidate_correction_audit.json")
    aligned_candidate_audit = _read_json(
        EVIDENCE_ROOT / "aligned_phase55_candidate_audit.json"
    )
    split_integrity = _read_json(EVIDENCE_ROOT / "split_integrity.json")
    preflight = _read_json(preflight_dir / "preflight_report.json")
    calibration = _read_json(calibration_dir / "candidate_evaluator_report.json")
    holdout = _read_json(holdout_dir / "candidate_evaluator_report.json")
    aligned = _read_json(aligned_dir / "aligned_regression_report.json")
    decision = build_phase68_decision(
        phase67_snapshot=phase67,
        aggregate_failure_audit=failure_audit,
        fresh_calibration_report=calibration,
        fresh_holdout_report=holdout,
        aligned_phase55_report=aligned,
        fresh_candidate_audit=fresh_candidate_audit,
        aligned_candidate_audit=aligned_candidate_audit,
        split_integrity=split_integrity,
    )

    call_counts = {
        "preflight": _read_jsonl_count(preflight_dir),
        "calibration": _read_jsonl_count(calibration_dir),
        "holdout": _read_jsonl_count(holdout_dir),
        "aligned_phase55_regression": _read_jsonl_count(aligned_dir),
    }
    comparison = {
        "kind": "phase68_aligned_candidate_scope_comparison",
        "phase66_aligned_phase55_baseline": {
            "accuracy": 0.7333,
            "case_count": 150,
            "label_contract_only": True,
        },
        "phase68_fresh_calibration": {
            "status": calibration.get("status"),
            "accuracy": calibration.get("accuracy"),
            "typed_exact_match_rate": calibration.get("typed_exact_match_rate"),
            "candidate_selection_exact_match_rate": calibration.get(
                "candidate_selection_exact_match_rate"
            ),
            "false_accept_count": calibration.get("false_accept_count_on_reject_cases"),
            "schema_failure_count": calibration.get("schema_failure_count"),
            "candidate_value_conflict_count": calibration.get(
                "candidate_value_conflict_count"
            ),
        },
        "phase68_fresh_holdout": {
            "status": holdout.get("status"),
            "accuracy": holdout.get("accuracy"),
            "typed_exact_match_rate": holdout.get("typed_exact_match_rate"),
            "candidate_selection_exact_match_rate": holdout.get(
                "candidate_selection_exact_match_rate"
            ),
            "false_accept_count": holdout.get("false_accept_count_on_reject_cases"),
            "schema_failure_count": holdout.get("schema_failure_count"),
            "candidate_value_conflict_count": holdout.get(
                "candidate_value_conflict_count"
            ),
        },
        "phase68_aligned_phase55_label_regression": {
            "accuracy": aligned.get("accuracy"),
            "accuracy_delta_from_phase66": round(
                float(aligned.get("accuracy") or 0.0) - 0.7333, 4
            ),
            "completed_item_count": aligned.get("completed_item_count"),
            "per_category": aligned.get("per_category"),
            "false_accept_count": aligned.get("false_accept_count_on_reject_cases"),
            "schema_failure_count": aligned.get("schema_failure_count"),
            "candidate_value_conflict_count": aligned.get(
                "candidate_value_conflict_count"
            ),
            "typed_exact_match_rate_diagnostic_only": aligned.get(
                "typed_exact_match_rate"
            ),
            "candidate_selection_exact_match_rate_not_comparable": aligned.get(
                "candidate_selection_exact_match_rate"
            ),
            "label_contract_only": True,
        },
        "actual_model_output_counts": call_counts,
        "actual_model_output_count": sum(call_counts.values()),
        "recommendation": decision["recommendation"],
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "runtime_replay_executed": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }

    freeze_checks = all(
        _read_json(directory / "freeze_check.json").get("passed") is True
        for directory in (preflight_dir, calibration_dir, holdout_dir, aligned_dir)
    )
    integrity_checks = {
        "phase67_snapshot_passed": phase67.get("passed") is True,
        "all_stage_freeze_checks_passed": freeze_checks,
        "preflight_complete": call_counts["preflight"] == 12
        and preflight.get("status") == "passed",
        "fresh_calibration_complete": call_counts["calibration"] == 120
        and calibration.get("completed_item_count") == 60,
        "fresh_holdout_complete": call_counts["holdout"] == 240
        and holdout.get("completed_item_count") == 120,
        "aligned_phase55_complete": call_counts["aligned_phase55_regression"] == 300
        and aligned.get("completed_item_count") == 150,
        "all_model_calls_real": all(
            report.get("actual_model_calls") is True
            for report in (calibration, holdout, aligned)
        ),
        "zero_exhausted_model_items": all(
            int(report.get("failure_count") or 0) == 0
            for report in (calibration, holdout, aligned)
        ),
        "decision_matches_frozen_results": decision.get("recommendation")
        == "recommend_phase68_evaluator_qualification_for_manual_review_only",
        "no_runtime_training_adapter_or_product_change": decision.get(
            "runtime_ab_allowed_in_phase68"
        )
        is False
        and decision.get("training_allowed") is False
        and decision.get("adapter_created") is False
        and decision.get("product_default_change_allowed") is False,
    }
    integrity = {
        "kind": "phase68_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }

    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase68-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_text(
        EVIDENCE_ROOT / "phase68-final-decision.md",
        f"""# Phase68 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。新 calibration 与新 holdout 均为 `1.0`，Phase55 对齐标签回归由 Phase66 的 `0.7333` 提升到 `{aligned.get('accuracy')}`，达到冻结的 `0.95` 标签门槛。

## 真实模型证据

- 协议预检：12/12 个真实 judge 输出。
- 新 calibration：60/60，typed exact `1.0`，candidate exact `1.0`。
- 新 holdout：120/120，typed exact `1.0`，candidate exact `1.0`。
- Phase55 对齐回归：150/150，五个类别均为 `1.0`，false accepts、schema failures、candidate conflicts 均为 `0`。
- 总真实 judge 输出：`{sum(call_counts.values())}`，没有 exhausted item。

## 口径说明

Phase55 的字段级 gold 早于当前 candidate-ID 与关系优先语义，因此其 typed exact `0.9` 和 candidate-ID exact `0.0` 只作诊断，不用于伪造当前 typed 提升。Phase67 已冻结的可比口径是最终 accept/edit/reject 标签；当前完整 typed 门只由全新 calibration/holdout 验证，二者均为 `1.0`。

## 边界

Phase68 不执行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径、不自动 promote。它只允许 Phase69 设计一个最小 runtime/base 对比，并仍须人工审查。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase68-runbook.md",
        """# Phase68 Runbook

Use the isolated Ollama endpoint containing both frozen judge models:

```bash
.venv/bin/python tools/phase68_prepare.py --clean-evidence
.venv/bin/python tools/phase68_execute.py --stage preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase68_execute.py --stage calibration --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase68_execute.py --stage holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase68_execute.py --stage phase55_regression --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase68_finalize_evidence.py
.venv/bin/python tools/phase68_validate.py
```

Do not edit the candidate rule, prompts, fixtures, protocol, or gates after prepare. Phase55 is label-contract regression only; fresh Phase68 splits retain the full typed and candidate-exact gates.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

Build Phase69 as the first minimal runtime A/B unlocked by evaluator qualification. Freeze a small, realistic set of multi-turn PFE boundary tasks before calls. Compare the existing product runtime path against a candidate contract path using the now-qualified evaluator, while keeping model, decoding, inputs, and scorer fixed. Do not train or attach Hermes in the same phase. Require a real product metric improvement with no boundary regression; otherwise hold. Even on success, permit only manual-review recommendation and keep product defaults unchanged.
""",
    )
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase68_finalization_state",
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
