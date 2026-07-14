#!/usr/bin/env python3
"""Finalize Phase57 historical replay evidence without runtime changes."""

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

from pfe_core.phase57_span_evaluator_historical_replay import build_phase57_decision


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase57-span-evaluator-historical-replay"
PHASE56_ROOT = REPO_ROOT / "docs/demo/phase56-evidence-span-grounded-atomic"
ALIASES = ("semantic_judge_alpha", "semantic_judge_beta")
DYNAMIC_FILES = {
    "evidence_manifest.json",
    "finalization_state.json",
    "validation_gate.txt",
    "validation_summary.json",
}


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
    return sum(len(_read_jsonl(directory / f"judge_span_results_{alias}.jsonl")) for alias in ALIASES)


def _failure_analysis(report: Mapping[str, Any]) -> dict[str, Any]:
    label_failures = [dict(row) for row in report.get("details") or [] if row.get("passed") is False]
    invalid_groundings = []
    transition_counts: dict[str, int] = {}
    for row in report.get("details") or []:
        if row.get("passed") is False:
            transition = f"{row.get('phase')}:{row.get('expected_label')}->{row.get('actual_label')}"
            transition_counts[transition] = transition_counts.get(transition, 0) + 1
        for extraction in row.get("grounded_extractions") or []:
            for field in ("source_registration", "user_outcome_status", "test_to_user_outcome_relation"):
                if extraction.get(f"{field}_grounded") is True:
                    continue
                invalid_groundings.append(
                    {
                        "item_id": row.get("item_id"),
                        "phase": row.get("phase"),
                        "case_id": row.get("case_id"),
                        "category": row.get("category"),
                        "expected_label": row.get("expected_label"),
                        "actual_label": row.get("actual_label"),
                        "judge_alias": extraction.get("judge_alias"),
                        "field": field,
                        "raw_value": extraction.get(f"raw_{field}"),
                        "effective_value": extraction.get(field),
                        "raw_span": extraction.get(f"{field}_span"),
                        "grounding_reason": extraction.get(f"{field}_grounding_reason"),
                        "conservative_reject": extraction.get("conservative_reject") is True,
                    }
                )
    return {
        "kind": "phase57_historical_replay_failure_analysis",
        "status": "sealed_no_post_replay_tuning",
        "label_failure_count": len(label_failures),
        "invalid_grounding_count": len(invalid_groundings),
        "invalid_dangerous_atom_count": int(report.get("invalid_dangerous_atom_count") or 0),
        "label_transition_counts": dict(sorted(transition_counts.items())),
        "rubric_schema_grounding_composer_or_fixture_modified_after_replay": False,
        "label_failures": label_failures,
        "invalid_groundings": invalid_groundings,
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
    return {
        "kind": "phase57_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": digest,
    }


def main() -> int:
    replay_dir = EVIDENCE_ROOT / "evidence-historical-replay"
    report = _read_json(replay_dir / "historical_replay_report.json")
    baseline = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase56_canonical_snapshot.json")
    replay_integrity = _read_json(replay_dir / "replay_integrity.json")
    runtime = _read_json(EVIDENCE_ROOT / "evidence-no-runtime/runtime_status.json")
    decision = build_phase57_decision(
        phase56_snapshot=baseline,
        replay_integrity=replay_integrity,
        replay_report=report,
        runtime_replay_model_call_count=int(runtime.get("runtime_replay_model_call_count") or 0),
    )
    calls = _call_count(replay_dir)
    comparison = {
        "kind": "phase57_historical_replay_comparison",
        "phase56": {
            "recommendation": _read_json(PHASE56_ROOT / "phase56-final-decision.json").get("recommendation"),
            "holdout_accuracy": _read_json(
                PHASE56_ROOT / "evidence-evaluator-holdout/span_evaluator_report.json"
            ).get("accuracy"),
            "holdout_typed_exact_match_rate": _read_json(
                PHASE56_ROOT / "evidence-evaluator-holdout/span_evaluator_report.json"
            ).get("typed_exact_match_rate"),
            "holdout_grounding_validity_rate": _read_json(
                PHASE56_ROOT / "evidence-evaluator-holdout/span_evaluator_report.json"
            ).get("raw_grounding_validity_rate"),
        },
        "phase57": {
            "status": report.get("status"),
            "historical_replay_accuracy": report.get("accuracy"),
            "per_phase": report.get("per_phase"),
            "per_category": report.get("per_category"),
            "grounding_validity_rate": report.get("raw_grounding_validity_rate"),
            "invalid_atom_count": report.get("invalid_atom_count"),
            "invalid_dangerous_atom_count": report.get("invalid_dangerous_atom_count"),
            "false_accept_count": report.get("false_accept_count_on_reject_cases"),
            "hard_vs_two_safe_accept_conflict_count": report.get(
                "hard_reject_vs_two_safe_accept_conflict_count"
            ),
            "real_model_call_count": calls,
            "phase56_evaluator_unchanged": True,
            "ollama_endpoint": report.get("ollama_endpoint"),
            "parallel_worker_count": report.get("parallel_worker_count"),
            "request_timeout_seconds": report.get("request_timeout_seconds"),
        },
        "recommendation": decision["recommendation"],
        "runtime_replay": {
            "status": runtime.get("runtime_replay_status"),
            "real_model_call_count": runtime.get("runtime_replay_model_call_count"),
        },
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "runtime_replay_executed": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }
    expected_recommendation = (
        "recommend_phase57_external_replay_for_manual_review_only"
        if all(decision.get("checks", {}).values())
        else "hold_phase57_span_evaluator_historical_replay"
    )
    integrity_checks = {
        "phase56_canonical_snapshot_passed": baseline.get("passed") is True,
        "historical_replay_integrity_passed": replay_integrity.get("passed") is True,
        "freeze_check_passed": _read_json(replay_dir / "freeze_check.json").get("passed") is True,
        "replay_complete": (
            calls == 1116
            and int(report.get("item_count") or 0) == 558
            and int(report.get("completed_item_count") or 0) == 558
            and int(report.get("failure_count") or 0) == 0
        ),
        "decision_matches_frozen_gates": decision["recommendation"] == expected_recommendation,
        "phase56_evaluator_unchanged": report.get("phase56_evaluator_unchanged") is True,
        "no_post_replay_tuning": _read_json(EVIDENCE_ROOT / "evaluator_protocol.json").get(
            "post_replay_tuning_allowed"
        ) is False,
        "runtime_not_run": int(runtime.get("runtime_replay_model_call_count") or 0) == 0,
        "no_training_adapter_hermes_or_default_change": (
            decision["new_training_allowed"] is False
            and decision["new_adapter_created"] is False
            and decision["hermes_attachment_allowed"] is False
            and decision["product_default_change_allowed"] is False
        ),
    }
    integrity = {
        "kind": "phase57_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    _write_json(replay_dir / "failure_analysis.json", _failure_analysis(report))
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase57-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_text(
        EVIDENCE_ROOT / "phase57-final-decision.md",
        f"""# Phase57 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。Phase51-55 历史 holdout 总准确率为 `{report.get('accuracy')}`，各阶段结果为 `{report.get('per_phase')}`，grounding validity 为 `{report.get('raw_grounding_validity_rate')}`。

## 冻结边界

- Phase56 prompt、JSON schema、quote mask、clause grounding 和 deterministic composer 未修改。
- 历史 phase、case id 与 gold label 对两个 judge 全部隐藏。
- 历史 replay 结果不得用于本阶段调参；失败即 hold。
- Phase57 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径。
- 所有输入均为历史 simulated evaluator fixtures，不是 actual user feedback，也不用于训练。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase57-runbook.md",
        """# Phase57 Runbook

Start an isolated four-slot Ollama service in terminal A:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

Run the frozen replay in terminal B:

```bash
.venv/bin/python tools/phase57_prepare.py --clean-evidence
.venv/bin/python tools/phase57_historical_replay.py --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase57_finalize_evidence.py
.venv/bin/python tools/phase57_validate.py
```

The historical replay is a one-shot external qualification. Its results may be analyzed and archived, but never used to tune the frozen Phase56 evaluator in Phase57.
""",
    )
    next_goal = (
        "Build Phase58 as one minimal runtime-contract A/B using the now externally qualified Phase56 evaluator. Freeze a small "
        "representative simulated-usage set before calls; compare the current runtime path against exactly one boundary-contract "
        "candidate with real model outputs; require no safety, citation, structure, provenance, or user-outcome regression and at "
        "least one measurable boundary improvement. Keep the candidate shadow-only and manual-review-only. Do not train, attach "
        "Hermes, change the product default, or claim actual user benefit."
        if decision["phase58_minimal_runtime_ab_design_eligible"]
        else
        "Build Phase58 from the sealed Phase57 historical replay failures. Do not tune on individual replay rows. Introduce one "
        "structural evaluator change justified by the aggregated failure class, then freeze completely new calibration and holdout "
        "before any model calls. Do not run runtime A/B, train, attach Hermes, or change the product default."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}\n")
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase57_finalization_state",
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
