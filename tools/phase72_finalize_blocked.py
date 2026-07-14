#!/usr/bin/env python3
"""Package Phase72 after the frozen wire preflight blocks downstream work."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase72-deterministic-boundary-serializer"
DYNAMIC = {
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
        if path.is_file() and path.name not in DYNAMIC:
            files.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "kind": "phase72_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> int:
    snapshot = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase71_hold_snapshot.json")
    preparation = _read_json(EVIDENCE_ROOT / "preparation_decision.json")
    freeze = _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json")
    freeze_check = _read_json(
        EVIDENCE_ROOT / "evidence-sparse-preflight/freeze_check.json"
    )
    report = _read_json(
        EVIDENCE_ROOT / "evidence-sparse-preflight/evaluator_report.json"
    )
    frozen_sources = dict(freeze.get("source_sha256") or {})
    changed_sources = []
    for name, expected in frozen_sources.items():
        source_path = _read_json(EVIDENCE_ROOT / "source_manifest.json").get(
            "source_paths", {}
        ).get(name)
        if source_path:
            actual = _sha256(REPO_ROOT / str(source_path))
        else:
            source_map = {
                "phase46_generator_helpers": "tools/phase46_qwen3_4b_generate.py",
                "phase53_hard_detector": "pfe-core/pfe_core/phase53_evaluator_scope_recovery.py",
                "phase56_grounder_composer": "pfe-core/pfe_core/phase56_evidence_span_grounded_atomic.py",
                "phase59_candidates": "pfe-core/pfe_core/phase59_proposition_addressed_grounding.py",
                "phase62_consensus": "pfe-core/pfe_core/phase62_risk_asymmetric_candidate_consensus.py",
                "phase63_typed_wire": "pfe-core/pfe_core/phase63_field_typed_candidate_wire.py",
                "phase70_runtime_core": "pfe-core/pfe_core/phase70_structured_boundary_contract.py",
                "phase70_generate": "tools/phase70_generate.py",
                "phase70_execute_eval": "tools/phase70_execute_eval.py",
                "phase70_core": "pfe-core/pfe_core/phase72_deterministic_boundary_serializer.py",
                "phase70_prepare_product_eval": "tools/phase72_deterministic_boundary_serializer.py",
                "phase70_finalize": "tools/phase72_deterministic_boundary_serializer.py",
                "phase72_core": "pfe-core/pfe_core/phase72_deterministic_boundary_serializer.py",
                "phase72_driver": "tools/phase72_deterministic_boundary_serializer.py",
            }
            actual = _sha256(REPO_ROOT / source_map[name])
        if actual != expected:
            changed_sources.append(name)
    packaging_audit = {
        "kind": "phase72_post_call_packaging_audit",
        "passed": not changed_sources,
        "frozen_source_changes": sorted(changed_sources),
        "packager_was_not_part_of_model_call_freeze": True,
        "packager_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "packager_sha256": _sha256(Path(__file__).resolve()),
        "created_at": _utcnow(),
    }
    downstream_paths = (
        EVIDENCE_ROOT / "evidence-phase68-regression/evaluator_report.json",
        EVIDENCE_ROOT / "evidence-real-generation/metrics.json",
        EVIDENCE_ROOT / "evidence-product-eval/evaluator_report.json",
    )
    checks = {
        "phase71_snapshot_passed": snapshot.get("passed") is True,
        "preparation_was_ready": preparation.get("status")
        == "ready_for_sparse_transport_preflight",
        "model_call_freeze_passed": freeze_check.get("passed") is True,
        "wire_preflight_not_qualified": report.get("status") == "not_qualified",
        "34_of_36_real_outputs_preserved": report.get("successful_model_output_count")
        == 34
        and report.get("expected_model_output_count") == 36,
        "two_exhausted_failures_preserved": report.get("failure_count") == 2
        and report.get("raw_failure_attempt_count") == 4
        and report.get("raw_failures_preserved") is True,
        "downstream_work_not_run": not any(path.exists() for path in downstream_paths),
        "all_frozen_sources_unchanged": packaging_audit["passed"],
    }
    integrity = {
        "kind": "phase72_evidence_integrity",
        "passed": all(checks.values()),
        "experiment_succeeded": False,
        "blocked_evidence_complete": all(checks.values()),
        "failed_stage": "wire_preflight",
        "checks": checks,
        "created_at": _utcnow(),
    }
    decision = {
        "kind": "phase72_final_decision",
        "status": "hold_phase72_deterministic_boundary_serializer",
        "recommendation": "hold_phase72_deterministic_boundary_serializer",
        "experiment_status": "blocked_at_wire_preflight",
        "failed_checks": ["wire_preflight_qualified"],
        "blocker": {
            "successful_model_output_count": report.get("successful_model_output_count"),
            "expected_model_output_count": report.get("expected_model_output_count"),
            "failure_count": report.get("failure_count"),
            "failures": report.get("failures"),
            "raw_failure_attempt_count": report.get("raw_failure_attempt_count"),
            "observed_invalid_shapes": [
                "PFE2|s001=exclude_actual@c001|none only|r001=does_not_establish@c002",
                "PFE2|s001=exclude_actual@c001|u001=suspended_or_negated@c002|none only",
            ],
        },
        "phase73_nondefault_api_canary_eligible": False,
        "product_default_change_allowed": False,
        "training_allowed": False,
        "adapter_created": False,
        "hermes_attachment_allowed": False,
        "auto_promote_allowed": False,
    }
    comparison = {
        "kind": "phase72_deterministic_serializer_comparison",
        "experiment_status": "blocked_at_wire_preflight",
        "wire_preflight": {
            "status": report.get("status"),
            "accuracy": report.get("accuracy"),
            "successful_model_output_count": report.get("successful_model_output_count"),
            "failure_count": report.get("failure_count"),
            "schema_failure_count": report.get("schema_failure_count"),
        },
        "phase68_regression": {"status": "not_run_prerequisite_failed"},
        "generation": {"status": "not_run_prerequisite_failed"},
        "product_eval": {"status": "not_run_prerequisite_failed"},
        "actual_judge_output_counts": {
            "sparse_preflight": report.get("successful_model_output_count"),
            "phase68_regression": 0,
            "product": 0,
        },
        "actual_generation_call_count": 0,
        "actual_model_output_count_total": int(
            report.get("successful_model_output_count") or 0
        ),
        "recommendation": decision["recommendation"],
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }
    _write_json(EVIDENCE_ROOT / "post_call_packaging_audit.json", packaging_audit)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_json(EVIDENCE_ROOT / "phase72-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase72_training_attempt",
            "status": "not_run_by_design",
            "reason": "Frozen evaluator prerequisite did not qualify.",
            "adapter_created": False,
        },
    )
    _write_text(
        EVIDENCE_ROOT / "output_examples.md",
        """# Phase72 Output Examples

Product generation was not run because the frozen wire preflight failed. The valid outputs and all four raw invalid wire attempts are preserved under `evidence-sparse-preflight/`.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase72-final-decision.md",
        f"""# Phase72 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。explicit allowed-token wire 只完成 34/36 个真实 judge 输出，两个缺字段组合在两次冻结重试中都复制了完整 candidate descriptor，因此未运行 regression、Qwen3-4B generation 或 product eval。

## 边界

这是 evaluator protocol failure evidence，不是 serializer 产品结果、训练收益或真实用户反馈。没有训练、adapter、Hermes、默认切换或自动 promote。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase72-runbook.md",
        """# Phase72 Runbook

```bash
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py prepare --clean-evidence
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py eval --stage sparse_preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase72_finalize_blocked.py
.venv/bin/python tools/phase72_validate_blocked.py
```

Do not run downstream stages or edit the frozen wire protocol after this failure.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

Build Phase73 as a newly frozen typed-wire normalization protocol. Accept a verbose descriptor only when every segment exactly matches a listed typed candidate (`typed_id=value@clause`) or the literal `none only` for a field with zero candidates, normalize it to internal candidate IDs, reject all mismatches or cross-field values, then requalify on fresh stress fixtures and Phase68 regression before resuming the deterministic serializer A/B. Do not reuse Phase72 failures as passing outputs.
""",
    )
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    state = {
        "kind": "phase72_finalization_state",
        "status": "blocked",
        "recommendation": decision["recommendation"],
        "evidence_integrity_passed": integrity["passed"],
        "experiment_succeeded": False,
        "manifest_file_count": manifest["file_count"],
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "finalization_state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
