#!/usr/bin/env python3
"""Finalize Phase67 deterministic contract-audit evidence."""

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

from pfe_core.phase67_historical_contract_compatibility_audit import build_phase67_decision


EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase67-historical-contract-compatibility-audit"
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
        "kind": "phase67_evidence_manifest",
        "file_count": len(files),
        "files": files,
        "manifest_sha256": digest,
    }


def main() -> int:
    snapshot = _read_json(
        EVIDENCE_ROOT / "evidence-baseline/phase66_canonical_snapshot.json"
    )
    matrix = _read_json(EVIDENCE_ROOT / "contract_compatibility_matrix.json")
    partition = _read_json(EVIDENCE_ROOT / "historical_partition.json")
    interpretation = _read_json(EVIDENCE_ROOT / "metric_interpretation.json")
    source_audit = _read_json(EVIDENCE_ROOT / "source_contract_audit.json")
    preparation = _read_json(EVIDENCE_ROOT / "preparation_decision.json")
    model_status = _read_json(
        EVIDENCE_ROOT / "evidence-no-model-calls/model_call_status.json"
    )
    decision = build_phase67_decision(
        phase66_snapshot=snapshot,
        contract_matrix=matrix,
        historical_partition=partition,
        metric_interpretation=interpretation,
        source_contract_audit=source_audit,
    )
    aligned = dict(interpretation.get("aligned_legacy_phase55_regression") or {})
    fresh = dict(interpretation.get("current_contract_fresh_external") or {})
    comparison = {
        "kind": "phase67_contract_aware_evidence_comparison",
        "phase66_fresh_current_contract": fresh,
        "aligned_legacy_phase55_regression": aligned,
        "legacy_diagnostic_only": {
            "phases": partition.get("legacy_diagnostic_only_phases"),
            "case_count": partition.get("legacy_diagnostic_only_count"),
            "phase66_all_phase_accuracy": interpretation.get(
                "phase66_all_phase_accuracy_retained_as_diagnostic"
            ),
        },
        "automatic_relabel_count": partition.get("automatic_relabel_count"),
        "current_evaluator_qualified_for_runtime_ab": False,
        "recommendation": decision["recommendation"],
        "model_call_count": model_status.get("model_call_count"),
        "actual_user_feedback_count": 0,
        "training_executed": False,
        "adapter_created": False,
        "runtime_replay_executed": False,
        "hermes_attached": False,
        "product_default_changed": False,
    }
    integrity_checks = {
        "preparation_ready": preparation.get("status")
        == "ready_for_deterministic_audit_finalization",
        "phase66_snapshot_passed": snapshot.get("passed") is True,
        "contract_matrix_passed": matrix.get("passed") is True,
        "historical_partition_passed": partition.get("passed") is True,
        "metric_interpretation_passed": interpretation.get("passed") is True,
        "source_contract_audit_passed": source_audit.get("passed") is True,
        "only_phase55_aligned": partition.get("aligned_legacy_regression_phases")
        == ["phase55"],
        "diagnostic_only_count_exact": int(
            partition.get("legacy_diagnostic_only_count") or 0
        )
        == 408,
        "automatic_relabel_count_zero": int(
            partition.get("automatic_relabel_count") or 0
        )
        == 0,
        "model_call_count_zero": int(model_status.get("model_call_count") or 0) == 0,
        "runtime_training_adapter_unchanged": decision.get("runtime_ab_allowed")
        is False
        and decision.get("training_allowed") is False
        and decision.get("adapter_created") is False,
    }
    integrity = {
        "kind": "phase67_evidence_integrity",
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "created_at": _utcnow(),
    }

    _write_json(EVIDENCE_ROOT / "comparison_summary.json", comparison)
    _write_json(EVIDENCE_ROOT / "phase67-final-decision.json", decision)
    _write_json(EVIDENCE_ROOT / "evidence_integrity.json", integrity)
    _write_text(
        EVIDENCE_ROOT / "phase67-final-decision.md",
        f"""# Phase67 Final Decision

## 结论

最终 recommendation 为 **{decision['recommendation']}**。Phase67 完成的是历史标签契约审计，不是一次新的模型评测。Phase51-54 共 408 条旧样本只能保留为 legacy diagnostic；只有 Phase55 的 150 条样本与当前三原子 accept 契约直接对齐。

## 证据解释

- Phase66 全新 current-contract holdout 准确率为 `{fresh.get('accuracy')}`。
- 对齐的 Phase55 regression 准确率为 `{aligned.get('accuracy')}`，低于冻结的 `0.95` 门槛，因此 evaluator 仍不具备 runtime A/B 资格。
- Phase51-54 的合并历史准确率继续保留，但不再冒充当前契约 gold。
- 没有自动重标旧数据；automatic relabel count 为 `0`。

## 边界

- 本阶段模型调用 `0` 次，不做 runtime replay、不训练、不创建 adapter、不接 Hermes、不改产品默认路径。
- 所有历史输入仍是 simulated evaluator fixtures，不是 actual user feedback，也不得进入训练。
- 审计结论只允许下一阶段基于 Phase55 对齐聚合失败做一次候选恢复，不允许自动 promote。
""",
    )
    _write_text(
        EVIDENCE_ROOT / "phase67-runbook.md",
        """# Phase67 Runbook

Phase67 is deterministic and makes no model calls:

```bash
.venv/bin/python tools/phase67_prepare.py --clean-evidence
.venv/bin/python tools/phase67_finalize_evidence.py
.venv/bin/python tools/phase67_validate.py
```

Do not relabel individual Phase51-55 rows. Only Phase55 labels may be used as aligned legacy regression under the current three-atom contract; Phase51-54 remain diagnostic-only.
""",
    )
    _write_text(
        EVIDENCE_ROOT / "next-pursuit-goal.md",
        """# Next Pursuit Goal

Build Phase68 as one aligned candidate-scope recovery. Use only aggregate failure classes from sealed Phase55 regression plus entirely new current-contract calibration and holdout. Keep all individual Phase55 rows sealed until the new candidate rule and split are frozen. Preserve the three-atom deterministic composer, the 0.95 aligned-regression gate, zero false accepts, zero schema failures, and zero candidate conflicts. Do not use Phase51-54 labels as current gold. Do not run runtime A/B, train, attach Hermes, change defaults, or auto-promote unless evaluator qualification is independently established first.
""",
    )
    manifest = _manifest()
    _write_json(EVIDENCE_ROOT / "evidence_manifest.json", manifest)
    finalization = {
        "kind": "phase67_finalization_state",
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
