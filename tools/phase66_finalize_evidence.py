#!/usr/bin/env python3
"""Finalize Phase66 evidence without tuning after external model calls."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase56_evidence_span_grounded_atomic import PHASE56_TYPED_FIELDS
from pfe_core.phase66_external_distribution_regression import build_phase66_decision


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase66-external-distribution-regression"
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
    return [
        dict(json.loads(line))
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


def _result_count(directory: Path) -> int:
    return sum(
        len(_read_jsonl(path))
        for path in directory.glob("judge_typed_wire_results_*.jsonl")
    )


def _aggregate_failure_analysis(report: Mapping[str, Any]) -> dict[str, Any]:
    transitions = Counter()
    per_phase = Counter()
    accept_field_values = Counter()
    hard_rejects = Counter()
    conflict_fields = Counter()
    for row in report.get("details") or []:
        if row.get("passed") is False:
            transition = f"{row.get('expected_label')}->{row.get('actual_label') or 'incomplete'}"
            transitions[transition] += 1
            per_phase[f"{row.get('phase')}:{transition}"] += 1
        if row.get("expected_label") == "accept" and row.get("passed") is False:
            hard_rejects[f"actual={row.get('actual_label') or 'incomplete'}:hard={bool(row.get('hard_reject'))}"] += 1
            grounded = dict(row.get("grounded_consensus") or {})
            for field in PHASE56_TYPED_FIELDS:
                accept_field_values[
                    f"actual={row.get('actual_label') or 'incomplete'}:{field}={grounded.get(field) or 'incomplete'}"
                ] += 1
        for field, detail in dict(row.get("field_consensus") or {}).items():
            if detail.get("candidate_value_conflict") is True:
                conflict_fields[field] += 1

    raw_shapes = Counter()
    for path in EVIDENCE_ROOT.glob(
        "evidence-historical-replay/typed_wire_failure_attempts_*.jsonl"
    ):
        for row in _read_jsonl(path):
            raw = str(row.get("raw_response") or "")
            if "=" in raw or "@" in raw:
                shape = "expanded_candidate_annotation"
            elif ";" in raw:
                shape = "multiple_candidate_ids"
            elif "rNNN" in raw or "uNNN" in raw or "sNNN" in raw:
                shape = "placeholder_candidate_id"
            elif "|u1|" in raw or "|r1" in raw or "|s1|" in raw:
                shape = "non_padded_candidate_id"
            else:
                shape = "other_invalid_wire"
            raw_shapes[shape] += 1
    return {
        "kind": "phase66_aggregate_external_regression_failure_taxonomy",
        "status": "sealed_no_phase66_tuning",
        "label_failure_count": sum(transitions.values()),
        "label_transition_counts": dict(sorted(transitions.items())),
        "per_phase_transition_counts": dict(sorted(per_phase.items())),
        "accept_failure_field_value_counts": dict(
            sorted(accept_field_values.items())
        ),
        "accept_failure_hard_reject_counts": dict(sorted(hard_rejects.items())),
        "candidate_conflict_field_counts": dict(sorted(conflict_fields.items())),
        "wire_failure_shape_counts": dict(sorted(raw_shapes.items())),
        "individual_failure_rows_included": False,
        "post_replay_evaluator_tuning_performed": False,
    }


def _split_failure_analysis(report: Mapping[str, Any], kind: str) -> dict[str, Any]:
    failures = [row for row in report.get("details") or [] if row.get("passed") is False]
    transitions = Counter(
        f"{row.get('expected_label')}->{row.get('actual_label') or 'incomplete'}"
        for row in failures
    )
    return {
        "kind": kind,
        "status": "sealed_no_post_call_tuning",
        "label_failure_count": len(failures),
        "label_transition_counts": dict(sorted(transitions.items())),
        "schema_failure_count": int(report.get("schema_failure_count") or 0),
        "candidate_value_conflict_count": int(
            report.get("candidate_value_conflict_count") or 0
        ),
        "false_accept_count": int(
            report.get("false_accept_count_on_reject_cases") or 0
        ),
        "individual_failure_rows_preserved_in_report": True,
        "post_call_tuning_performed": False,
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
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "phase66_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": digest,
    }


def main() -> int:
    preflight_dir = EVIDENCE_ROOT / "evidence-typed-wire-preflight"
    external_dir = EVIDENCE_ROOT / "evidence-external-holdout"
    historical_dir = EVIDENCE_ROOT / "evidence-historical-replay"
    phase65 = _read_json(
        EVIDENCE_ROOT / "evidence-baseline/phase65_canonical_snapshot.json"
    )
    phase64 = _read_json(
        EVIDENCE_ROOT / "evidence-baseline/phase64_historical_baseline.json"
    )
    preflight = _read_json(preflight_dir / "preflight_report.json")
    external = _read_json(external_dir / "candidate_evaluator_report.json")
    historical = _read_json(historical_dir / "historical_replay_report.json")
    external_integrity = _read_json(external_dir / "external_integrity.json")
    historical_integrity = _read_json(historical_dir / "historical_integrity.json")
    external_audit = _read_json(external_dir / "fixture_semantic_audit.json")
    external_hard = _read_json(external_dir / "hard_rule_compatibility.json")
    decision = build_phase66_decision(
        phase65_snapshot=phase65,
        external_integrity=external_integrity,
        historical_integrity=historical_integrity,
        preflight_report=preflight,
        external_report=external,
        historical_report=historical,
        external_audit=external_audit,
        external_hard_compatibility=external_hard,
    )

    preflight_calls = _result_count(preflight_dir)
    external_calls = _result_count(external_dir)
    historical_calls = _result_count(historical_dir)
    historical_failed_items = int(historical.get("failure_count") or 0)
    historical_raw_failures = int(historical.get("raw_failure_attempt_count") or 0)
    comparison = {
        "kind": "phase66_external_distribution_comparison",
        "phase64_historical_baseline": phase64,
        "phase65_fresh_holdout": {
            "accuracy": phase65.get("phase65_holdout_accuracy"),
            "recommendation": phase65.get("phase65_recommendation"),
        },
        "phase66_fresh_external_holdout": {
            "status": external.get("status"),
            "accuracy": external.get("accuracy"),
            "typed_exact_match_rate": external.get("typed_exact_match_rate"),
            "candidate_selection_exact_match_rate": external.get(
                "candidate_selection_exact_match_rate"
            ),
            "raw_judge_typed_exact_match_rate": external.get(
                "raw_judge_typed_exact_match_rate"
            ),
            "false_accept_count": external.get(
                "false_accept_count_on_reject_cases"
            ),
            "schema_failure_count": external.get("schema_failure_count"),
            "candidate_value_conflict_count": external.get(
                "candidate_value_conflict_count"
            ),
            "actual_model_output_count": external_calls,
        },
        "phase66_historical_distribution": {
            "status": historical.get("status"),
            "accuracy": historical.get("accuracy"),
            "accuracy_delta_from_phase64": historical.get(
                "accuracy_delta_from_phase64"
            ),
            "per_phase": historical.get("per_phase"),
            "per_category": historical.get("per_category"),
            "false_accept_count": historical.get(
                "false_accept_count_on_reject_cases"
            ),
            "schema_failure_count": historical.get("schema_failure_count"),
            "candidate_value_conflict_count": historical.get(
                "candidate_value_conflict_count"
            ),
            "successful_model_output_count": historical_calls,
            "failed_judge_item_count": historical_failed_items,
            "actual_judge_attempt_count": historical_calls
            + historical_raw_failures,
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
        "phase65_canonical_snapshot_passed": phase65.get("passed") is True,
        "external_integrity_passed": external_integrity.get("passed") is True,
        "historical_integrity_passed": historical_integrity.get("passed") is True,
        "all_freeze_checks_passed": all(
            _read_json(directory / "freeze_check.json").get("passed") is True
            for directory in (preflight_dir, external_dir, historical_dir)
        ),
        "preflight_complete": preflight_calls == 12
        and preflight.get("status") == "passed",
        "external_holdout_complete": external_calls == 300
        and external.get("completed_item_count") == 150,
        "historical_replay_outcomes_complete": (
            historical_calls + historical_failed_items == 1116
            and int(historical.get("completed_item_count") or 0)
            + historical_failed_items
            == 558
            and int(historical.get("judge_item_outcome_count") or 0) == 1116
        ),
        "raw_failures_preserved": historical.get("raw_failures_preserved") is True,
        "decision_matches_frozen_results": decision.get("recommendation")
        == "hold_phase66_external_distribution_regression",
        "no_post_call_tuning": _read_json(
            EVIDENCE_ROOT / "evaluator_protocol.json"
        ).get("post_model_call_tuning_allowed")
        is False,
        "no_runtime_training_adapter_hermes_or_default_change": (
            decision.get("runtime_replay_allowed_in_phase66") is False
            and decision.get("new_training_allowed") is False
            and decision.get("new_adapter_created") is False
            and decision.get("hermes_attachment_allowed") is False
            and decision.get("product_default_change_allowed") is False
        ),
    }
    integrity = {
        "kind": "phase66_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }

    _write_json(
        external_dir / "failure_analysis.json",
        _split_failure_analysis(external, "phase66_external_holdout_failure_analysis"),
    )
    _write_json(
        historical_dir / "failure_analysis.json",
        _split_failure_analysis(
            historical, "phase66_historical_distribution_failure_analysis"
        ),
    )
    _write_json(
        EVIDENCE_ROOT / "aggregate_failure_taxonomy.json",
        _aggregate_failure_analysis(historical),
    )
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase66-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_text(
        EVIDENCE_ROOT / "phase66-final-decision.md",
        f"""# Phase66 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。全新外部分布 holdout 为 `{external.get('accuracy')}`，但 Phase51-55 历史分布仅为 `{historical.get('accuracy')}`，相对 Phase64 `{phase64.get('accuracy')}` 只提升 `{historical.get('accuracy_delta_from_phase64')}`，未达到冻结的 `{historical.get('material_accuracy_delta_gate')}` 提升门槛。

## 真实证据

- 新 holdout：150/150，false accepts `{external.get('false_accept_count_on_reject_cases')}`，schema failures `{external.get('schema_failure_count')}`，candidate conflicts `{external.get('candidate_value_conflict_count')}`。
- 历史 replay：完成 `{historical.get('completed_item_count')}`/558，9 个 judge-item 在两次重试后仍格式失败，schema failures `{historical.get('schema_failure_count')}`，candidate conflicts `{historical.get('candidate_value_conflict_count')}`，false accepts `{historical.get('false_accept_count_on_reject_cases')}`。
- 各阶段准确率：`{historical.get('per_phase')}`。

## 边界

- Phase65 candidate rule、Phase63 wire、Phase62 consensus、Phase56 composer 和所有门槛在调用前冻结，调用后未修改。
- 所有输入都是 simulated evaluator fixtures，不是 actual user feedback，也不用于训练。
- Phase66 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改默认路径、不自动 promote。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase66-runbook.md",
        """# Phase66 Runbook

Start an isolated Ollama service:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

Freeze and run the one-shot stages:

```bash
.venv/bin/python tools/phase66_prepare.py --clean-evidence
.venv/bin/python tools/phase66_execute.py --stage preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase66_execute.py --stage external_holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase66_execute.py --stage historical_replay --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase66_finalize_evidence.py
.venv/bin/python tools/phase66_validate.py
```

Do not rerun or tune a scored split after revealing its labels. Failed wire attempts remain evidence and are not parser-normalized.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

Build Phase67 from the sealed Phase66 aggregate failure taxonomy. Keep the fresh external holdout and all Phase51-55 rows sealed. Introduce at most one structural correction that addresses the dominant historical accept-to-edit/reject failure class without weakening hard safety, wire validation, false-accept gates, or candidate grounding. Freeze entirely new calibration and holdout before calls, then require a second historical distribution replay to materially exceed Phase66 before any runtime A/B. Do not train, attach Hermes, change product defaults, auto-promote, or claim actual user benefit.
""",
    )
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase66_finalization_state",
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
