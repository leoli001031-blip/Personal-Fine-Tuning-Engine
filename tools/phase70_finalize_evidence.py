#!/usr/bin/env python3
"""Finalize Phase70 evidence and manual-review-only decision."""

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

from pfe_core.phase69_minimal_runtime_ab import final_assistant_text
from pfe_core.phase70_structured_boundary_contract import (
    PHASE70_VARIANTS,
    build_phase70_decision,
)
from phase70_prepare import SOURCE_FILES


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase70-structured-boundary-contract"
DYNAMIC = {"evidence_manifest.json", "finalization_state.json", "validation_gate.txt", "validation_summary.json"}


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
        "kind": "phase70_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _examples(
    sessions: list[dict[str, Any]], transcripts: Mapping[str, list[dict[str, Any]]]
) -> str:
    indexed = {
        variant: {str(row.get("session_id")): row for row in transcripts[variant]}
        for variant in PHASE70_VARIANTS
    }
    selected = []
    seen = set()
    for session in sessions:
        key = (session.get("task_type"), session.get("category"))
        if key not in seen:
            seen.add(key)
            selected.append(session)
    lines = [
        "# Phase70 Paired Output Examples",
        "",
        "真实 Qwen3-4B simulated_usage 输出；不是 actual_user_feedback。",
    ]
    for session in selected:
        session_id = str(session["session_id"])
        lines.extend(
            [
                "",
                f"## {session_id} ({session.get('category')})",
                "",
                "**Natural contract**",
                "",
                final_assistant_text(indexed["natural_boundary_contract"][session_id]),
                "",
                "**Structured contract**",
                "",
                final_assistant_text(indexed["structured_boundary_contract"][session_id]),
            ]
        )
    return "\n".join(lines)


def _finalize_preflight_blocked(
    phase69: Mapping[str, Any], sparse: Mapping[str, Any]
) -> int:
    freeze = _read_json(EVIDENCE_ROOT / "evidence-sparse-preflight/freeze_check.json")
    frozen_sources = dict(
        _read_json(EVIDENCE_ROOT / "pre_model_call_freeze.json").get("source_sha256") or {}
    )
    current_sources = {name: _sha256(path) for name, path in SOURCE_FILES.items()}
    changed_sources = sorted(
        name for name, value in current_sources.items() if frozen_sources.get(name) != value
    )
    packaging_audit = {
        "kind": "phase70_post_call_packaging_change_audit",
        "passed": changed_sources == ["phase70_finalize"],
        "changed_source_names": changed_sources,
        "allowed_changed_source_names": ["phase70_finalize"],
        "change_scope": "blocked evidence packaging only",
        "evaluator_core_protocol_or_executor_changed_after_calls": any(
            name != "phase70_finalize" for name in changed_sources
        ),
        "frozen_source_sha256": frozen_sources,
        "current_source_sha256": current_sources,
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "post_call_packaging_change_audit.json", packaging_audit)
    downstream_paths = (
        EVIDENCE_ROOT / "evidence-phase68-regression/evaluator_report.json",
        EVIDENCE_ROOT / "evidence-product-eval/evaluator_report.json",
        EVIDENCE_ROOT / "evidence-real-generation/metrics_natural_boundary_contract.json",
        EVIDENCE_ROOT / "evidence-real-generation/metrics_structured_boundary_contract.json",
    )
    transport_envelope_qualified = (
        sparse.get("successful_model_output_count") == 24
        and sparse.get("failure_count") == 0
        and sparse.get("schema_failure_count") == 0
        and sparse.get("candidate_selection_exact_match_rate") == 1.0
        and sparse.get("raw_judge_typed_exact_match_rate") == 1.0
    )
    decision = {
        "kind": "phase70_final_decision",
        "status": "hold_phase70_structured_boundary_contract",
        "recommendation": "hold_phase70_structured_boundary_contract",
        "experiment_status": "blocked_before_product_ab",
        "transport_envelope_qualified": transport_envelope_qualified,
        "full_sparse_composer_qualification": False,
        "blocker": {
            "case_id": "phase70-sparse-12",
            "category": "quoted_then_safe",
            "reason": (
                "The frozen fixture expected accept after safe candidate selection, but the "
                "unchanged hard detector conservatively rejected the quoted dangerous claim."
            ),
            "judge_candidate_selection_was_exact": True,
            "post_call_fixture_or_gate_change_allowed": False,
        },
        "checks": {
            "phase69_hold_preserved": phase69.get("passed") is True,
            "active_freeze_passed": freeze.get("passed") is True,
            "all_transport_outputs_valid": transport_envelope_qualified,
            "sparse_preflight_full_label_qualified": False,
            "downstream_ab_not_run": not any(path.exists() for path in downstream_paths),
        },
        "failed_checks": ["sparse_preflight_full_label_qualified"],
        "phase71_nondefault_api_canary_design_eligible": False,
        "phase71_requalified_experiment_design_eligible": True,
        "product_default_change_allowed": False,
        "training_allowed": False,
        "adapter_created": False,
        "hermes_attachment_allowed": False,
        "auto_promote_allowed": False,
    }
    comparison = {
        "kind": "phase70_structured_boundary_comparison",
        "experiment_status": "blocked_before_product_ab",
        "only_ab_variable": "freeform_vs_fixed_three_line_boundary_contract",
        "judge_transport": sparse.get("transport"),
        "judge_transport_by_alias": sparse.get("judge_transport_by_alias"),
        "transport_envelope_qualified": transport_envelope_qualified,
        "sparse_preflight": {
            "status": sparse.get("status"),
            "accuracy": sparse.get("accuracy"),
            "candidate_selection_exact_match_rate": sparse.get(
                "candidate_selection_exact_match_rate"
            ),
            "raw_judge_typed_exact_match_rate": sparse.get("raw_judge_typed_exact_match_rate"),
            "schema_failure_count": sparse.get("schema_failure_count"),
            "successful_model_output_count": sparse.get("successful_model_output_count"),
        },
        "phase68_regression": {"status": "not_run_prerequisite_failed"},
        "generation": {"status": "not_run_prerequisite_failed", "actual_call_count": 0},
        "product_eval": {"status": "not_run_prerequisite_failed"},
        "actual_judge_output_counts": {
            "sparse_preflight": sparse.get("successful_model_output_count"),
            "phase68_regression": 0,
            "product": 0,
        },
        "actual_generation_call_count": 0,
        "actual_model_output_count_total": int(
            sparse.get("successful_model_output_count") or 0
        ),
        "recommendation": decision["recommendation"],
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }
    integrity_checks = {
        "phase69_hold_snapshot_passed": phase69.get("passed") is True,
        "active_freeze_passed": freeze.get("passed") is True,
        "all_24_sparse_calls_real_and_valid": transport_envelope_qualified,
        "full_qualification_failure_preserved": sparse.get("status") == "not_qualified",
        "post_call_change_limited_to_evidence_packaging": packaging_audit["passed"],
        "downstream_work_not_run": not any(path.exists() for path in downstream_paths),
        "no_training_adapter_hermes_or_default_change": True,
        "actual_user_feedback_zero": True,
    }
    integrity = {
        "kind": "phase70_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "experiment_succeeded": False,
        "blocked_evidence_complete": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase70-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase70_training_attempt",
            "status": "not_run_by_design",
            "reason": "Phase70 stopped at the frozen evaluator prerequisite.",
            "adapter_created": False,
        },
    )
    _write_text(
        EVIDENCE_ROOT / "output_examples.md",
        """# Phase70 Output Examples

Product A/B generation was not run because the frozen sparse evaluator prerequisite did not qualify. The 24 real judge outputs and all raw transport attempts are preserved under the evaluator evidence directories.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase70-final-decision.md",
        f"""# Phase70 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。第三轮 alias-routed transport 完成 24/24 个真实 judge 输出，candidate selection 与 typed exact match 均为 `1.0`，但完整 preflight accuracy 为 `{sparse.get('accuracy')}`，低于冻结门槛，因此未运行 Phase68 regression、Qwen3-4B 产品生成或产品盲评。

## 阻塞原因

`phase70-sparse-12` 的 candidate 选择完全正确，但 fixture 将 quoted dangerous claim 设为可接受；未修改的 hard detector 对该引文执行保守拒绝。Phase70 不允许在看到输出后修改 fixture 或 gate。

## 边界

这是 evaluator transport / fixture 证据，不是产品收益、真实用户反馈或训练收益。没有训练、没有 adapter、没有接 Hermes、没有更改产品默认、没有自动 promote。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase70-runbook.md",
        """# Phase70 Runbook

```bash
.venv/bin/python tools/phase70_prepare.py --clean-evidence
.venv/bin/python tools/phase70_execute_eval.py --stage sparse_preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase70_finalize_evidence.py
.venv/bin/python tools/phase70_validate.py
```

The sparse prerequisite did not qualify. Do not run downstream regression, generation, or product evaluation, and do not edit the frozen fixture or gates after model calls.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

Build Phase71 as a newly frozen evaluator requalification and structured-contract A/B. Separate transport-envelope qualification from full composer qualification, make every preflight fixture explicitly compatible with the unchanged hard detector before calls, retain the same candidate semantics and product gates, and use fresh independent fixtures and holdout tasks. Do not reuse Phase70 outputs as a passing qualification and do not integrate or train while held.
""",
    )
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    state = {
        "kind": "phase70_finalization_state",
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


def main() -> int:
    phase69 = _read_json(EVIDENCE_ROOT / "evidence-baseline/phase69_hold_snapshot.json")
    sparse = _read_json(EVIDENCE_ROOT / "evidence-sparse-preflight/evaluator_report.json")
    if sparse.get("status") != "qualified":
        return _finalize_preflight_blocked(phase69, sparse)
    regression = _read_json(EVIDENCE_ROOT / "evidence-phase68-regression/evaluator_report.json")
    boundary = _read_json(EVIDENCE_ROOT / "evidence-product-eval/evaluator_report.json")
    parity = _read_json(EVIDENCE_ROOT / "ab_parity_audit.json")
    ordinary = _read_json(EVIDENCE_ROOT / "ordinary_control_report.json")
    holdout = _read_json(EVIDENCE_ROOT / "evidence-holdout/holdout.json")
    sessions = [dict(row) for row in holdout.get("sessions") or []]
    metrics = {
        variant: _read_json(EVIDENCE_ROOT / f"evidence-real-generation/metrics_{variant}.json")
        for variant in PHASE70_VARIANTS
    }
    transcripts = {
        variant: _read_jsonl(EVIDENCE_ROOT / f"evidence-real-generation/transcripts_{variant}.jsonl")
        for variant in PHASE70_VARIANTS
    }
    freezes_passed = (
        all(
            _read_json(EVIDENCE_ROOT / f"evidence-real-generation/freeze_check_{variant}.json").get("passed") is True
            for variant in PHASE70_VARIANTS
        )
        and all(
            _read_json(EVIDENCE_ROOT / directory / "freeze_check.json").get("passed") is True
            for directory in ("evidence-sparse-preflight", "evidence-phase68-regression", "evidence-product-eval")
        )
    )
    decision = build_phase70_decision(
        phase69_snapshot=phase69,
        transport_preflight=sparse,
        phase68_regression=regression,
        parity=parity,
        boundary=boundary,
        ordinary=ordinary,
        freezes_passed=freezes_passed,
    )
    comparison = {
        "kind": "phase70_structured_boundary_comparison",
        "model": metrics["natural_boundary_contract"].get("model_id"),
        "device": metrics["natural_boundary_contract"].get("device"),
        "only_ab_variable": "freeform_vs_fixed_three_line_boundary_contract",
        "transport_preflight": {
            "status": sparse.get("status"),
            "accuracy": sparse.get("accuracy"),
            "schema_failure_count": sparse.get("schema_failure_count"),
        },
        "phase68_transport_regression": {
            "status": regression.get("status"),
            "accuracy": regression.get("accuracy"),
            "typed_exact_match_rate": regression.get("typed_exact_match_rate"),
            "schema_failure_count": regression.get("schema_failure_count"),
        },
        "boundary": boundary.get("variants"),
        "candidate_accept_rate_delta": boundary.get("candidate_accept_rate_delta"),
        "ordinary_controls": ordinary.get("variants"),
        "generation_metrics": metrics,
        "actual_generation_call_count": sum(int(row.get("actual_generation_call_count") or 0) for row in metrics.values()),
        "actual_judge_output_counts": {
            "sparse_preflight": sparse.get("successful_model_output_count"),
            "phase68_regression": regression.get("successful_model_output_count"),
            "product": boundary.get("successful_model_output_count"),
        },
        "actual_model_output_count_total": sum(int(row.get("actual_generation_call_count") or 0) for row in metrics.values())
        + sum(
            int(report.get("successful_model_output_count") or 0)
            for report in (sparse, regression, boundary)
        ),
        "recommendation": decision["recommendation"],
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }
    integrity_checks = {
        "phase69_hold_snapshot_passed": phase69.get("passed") is True,
        "all_freezes_passed": freezes_passed,
        "sparse_preflight_complete": sparse.get("successful_model_output_count") == 24 and sparse.get("failure_count") == 0,
        "phase68_regression_complete": regression.get("successful_model_output_count") == 60 and regression.get("failure_count") == 0,
        "product_eval_complete": boundary.get("successful_model_output_count") == 144 and boundary.get("failure_count") == 0,
        "both_generation_arms_complete": all(row.get("completed_count") == 48 and row.get("failed_count") == 0 for row in metrics.values()),
        "all_288_generations_real": comparison["actual_generation_call_count"] == 288,
        "no_training_adapter_hermes_or_default_change": decision.get("training_allowed") is False
        and decision.get("adapter_created") is False
        and decision.get("hermes_attachment_allowed") is False
        and decision.get("product_default_change_allowed") is False,
        "actual_user_feedback_zero": comparison["actual_user_feedback_count"] == 0,
    }
    integrity = {
        "kind": "phase70_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase70-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_json(
        EVIDENCE_ROOT / "evidence-no-training/training_attempt.json",
        {
            "kind": "phase70_training_attempt",
            "status": "not_run_by_design",
            "reason": "Phase70 isolates runtime response structure; training would confound the result.",
            "adapter_created": False,
        },
    )
    _write_text(EVIDENCE_ROOT / "output_examples.md", _examples(sessions, transcripts))
    _write_text(
        EVIDENCE_ROOT / "phase70-final-decision.md",
        f"""# Phase70 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。自然语言契约 accept rate `{decision.get('baseline_accept_rate')}`，固定三行契约 `{decision.get('candidate_accept_rate')}`，增量 `{decision.get('candidate_accept_rate_delta')}`。

## 证据

- 稀疏 JSON-schema transport：{sparse.get('status')}，24 个真实 judge 输出。
- Phase68 对齐回归：{regression.get('status')}，accuracy `{regression.get('accuracy')}`，60 个真实 judge 输出。
- 产品盲评：144 个真实 judge 输出；A/B 身份与金标均隐藏。
- 真实 Qwen3-4B 生成：288 次，未加载 adapter。
- 普通任务：自然契约 `{dict(ordinary.get('variants') or {}).get('natural_boundary_contract', {}).get('pass_rate')}`，结构契约 `{dict(ordinary.get('variants') or {}).get('structured_boundary_contract', {}).get('pass_rate')}`。

## 边界

这是 simulated_usage runtime A/B，不是实际用户反馈或训练收益。没有训练、没有 adapter、没有接 Hermes、没有更改产品默认、没有自动 promote。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase70-runbook.md",
        """# Phase70 Runbook

```bash
.venv/bin/python tools/phase70_prepare.py --clean-evidence
.venv/bin/python tools/phase70_execute_eval.py --stage sparse_preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase70_execute_eval.py --stage phase68_regression --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase70_generate.py --variant natural_boundary_contract --clean
.venv/bin/python tools/phase70_generate.py --variant structured_boundary_contract --clean
.venv/bin/python tools/phase70_prepare_product_eval.py
.venv/bin/python tools/phase70_execute_eval.py --stage product --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase70_finalize_evidence.py
.venv/bin/python tools/phase70_validate.py
```

Do not edit the contracts, tasks, JSON transport, decoding, or gates after prepare. Do not edit product outputs after product-eval prepare.
""",
    )
    passed = decision["phase71_nondefault_api_canary_design_eligible"] is True
    next_goal = (
        "Build Phase71 as a non-default API canary for the structured boundary contract. Add an explicit opt-in runtime mode, verify stream/non-stream parity, ordinary-task routing, fallback and rollback behavior, then run a fresh canary holdout. Do not change the default mode or train in the same phase."
        if passed
        else
        "Build Phase71 as one more structured-contract revision from Phase70 failure categories. Keep the JSON evaluator and ordinary controls fixed, change one response-boundary mechanism, and use a fresh holdout. Do not integrate or train while held."
    )
    _write_text(EVIDENCE_ROOT / "next-pursuit-goal.md", f"# Next Pursuit Goal\n\n{next_goal}")
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    state = {
        "kind": "phase70_finalization_state",
        "status": "completed" if integrity["passed"] else "blocked",
        "recommendation": decision["recommendation"],
        "evidence_integrity_passed": integrity["passed"],
        "manifest_file_count": manifest["file_count"],
        "created_at": _utcnow(),
    }
    _write_json(EVIDENCE_ROOT / "finalization_state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if integrity["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
