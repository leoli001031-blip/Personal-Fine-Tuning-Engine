#!/usr/bin/env python3
"""Generate and validate the Phase111-112 evidence/eval closure."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.phase111_112_evidence_failure_eval import (
    audit_holdout_isolation,
    build_arm_comparison_contract,
    build_failure_taxonomy,
    build_phase112_cases,
    file_sha256,
    load_claim_ledger,
    load_eval_manifest,
    validate_claim_ledger,
    validate_eval_manifest,
    validate_phase112_cases,
)


DEFAULT_SOURCE_ROOT = Path(
    "/Users/lichenhao/Documents/Codex/2026-08-09/new-chat-2/outputs/"
    "harvest-20260809/pfe-fde"
)
DEFAULT_EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase111-112-evidence-ci-failure-eval-loop"
PHASE110_DECISION = (
    REPO_ROOT
    / "docs/demo/phase110-task-grounded-sft-dpo-causal-proof/phase110-final-decision.json"
)
PHASE85_TEST = REPO_ROOT / "tests/test_phase85_driver_safety.py"
PHASE85_FROZEN_HASH = "02de4e86e2d4b018b4a100cfc310b47c6782cae33790e70814dcea8ff425139f"
IMPLEMENTATION_SOURCES = (
    REPO_ROOT / "pfe-core/pfe_core/phase111_112_evidence_failure_eval.py",
    REPO_ROOT / "tools/phase111_112_evidence_failure_eval.py",
    REPO_ROOT / "tests/test_phase111_112_evidence_failure_eval.py",
    REPO_ROOT / "tests/test_phase111_112_driver_safety.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        dict(value)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in (json.loads(line),)
        if isinstance(value, Mapping)
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _safe_clean(root: Path) -> None:
    resolved = root.resolve()
    allowed_parent = (REPO_ROOT / "docs/demo").resolve()
    if resolved == allowed_parent or not resolved.is_relative_to(allowed_parent):
        raise ValueError(f"refusing to clean outside docs/demo: {resolved}")
    if resolved.name != "phase111-112-evidence-ci-failure-eval-loop":
        raise ValueError(f"refusing unexpected evidence root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _count_failure_modes(path: Path) -> int:
    return len(re.findall(r"^## FM-\d+", path.read_text(encoding="utf-8"), re.M))


def _phase110_contract() -> dict[str, Any]:
    decision = _read_json(PHASE110_DECISION)
    expected = {
        "status": "archive_phase110_sft_not_qualified",
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
        "actual_user_feedback_count": 0,
    }
    actual = {key: decision.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Phase110 frozen decision drifted: {actual}")
    return actual


def generate(source_root: Path, evidence_root: Path, *, clean: bool) -> dict[str, Any]:
    if clean:
        _safe_clean(evidence_root)
    claims_path = source_root / "claim-evidence.csv"
    evals_path = source_root / "eval-briefs.jsonl"
    failure_modes_path = source_root / "failure-modes.md"
    authorization_path = source_root / "authorization-matrix.md"
    for path in (claims_path, evals_path, failure_modes_path, authorization_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    claims = load_claim_ledger(claims_path)
    evals = load_eval_manifest(evals_path)
    cases = build_phase112_cases()
    taxonomy = build_failure_taxonomy()
    claim_report = validate_claim_ledger(claims, expected_count=28)
    eval_report = validate_eval_manifest(evals, expected_count=30)
    case_report = validate_phase112_cases(cases, expected_count=70)
    isolation = audit_holdout_isolation(cases, [])
    if not isolation["passed"]:
        raise ValueError("Phase112 holdout isolation failed")
    phase110 = _phase110_contract()
    current_phase85_hash = file_sha256(PHASE85_TEST)
    if current_phase85_hash != PHASE85_FROZEN_HASH:
        raise ValueError("Phase85 frozen driver test hash drifted")

    source_manifest = {
        "kind": "phase111_read_only_source_manifest",
        "sources": [
            {"path": str(path), "sha256": file_sha256(path), "read_only": True}
            for path in (claims_path, evals_path, failure_modes_path, authorization_path)
        ],
        "claim_count": len(claims),
        "eval_brief_count": len(evals),
        "source_failure_mode_count": _count_failure_modes(failure_modes_path),
        "raw_private_body_copied": False,
        "implementation_sources": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
            for path in IMPLEMENTATION_SOURCES
        ],
    }
    ci_report = {
        "kind": "phase111_cross_platform_pytest_temp_contract",
        "historical_linux_failure": "test_safe_clean_directory_allows_only_strict_allowlisted_descendants",
        "root_cause": "pytest tmp_path was a descendant of the production /tmp cleanup allowlist",
        "mitigation": "Fast beta pytest basetemp is under github.workspace, outside OS temp allowlists",
        "phase85_test_sha256": current_phase85_hash,
        "phase85_frozen_sha256": PHASE85_FROZEN_HASH,
        "phase85_frozen_test_unchanged": True,
        "gate_threshold_changed": False,
        "test_assertion_changed": False,
    }
    remote_ci_path = evidence_root / "evidence-ci/remote-fast-beta.json"
    remote_ci = _read_json(remote_ci_path) if remote_ci_path.is_file() else None
    if remote_ci is not None and (
        remote_ci.get("status") != "pass" or not remote_ci.get("run_url")
    ):
        raise ValueError(f"invalid remote Fast beta evidence: {remote_ci}")
    schema_report = {
        "kind": "phase111_112_schema_report",
        "claim_ledger": claim_report,
        "eval_manifest": eval_report,
        "phase112_cases": case_report,
        "holdout_integrity": isolation,
        "evidence_class_counts": dict(
            sorted(Counter(row["evidence_class"] for row in claims).items())
        ),
        "passed": True,
    }
    final_decision = {
        "kind": "phase111_112_final_decision",
        "status": "phase111_112_evidence_eval_ready_no_training",
        "recommendation": "proceed_to_phase113_only_after_manual_review",
        "phase110_frozen_decision": phase110,
        "phase111_ci_status": (
            "remote_fast_beta_pass"
            if remote_ci is not None
            else "local_reproduction_fixed_remote_gate_pending"
        ),
        "remote_fast_beta": remote_ci,
        "phase112_eval_status": "deterministic_contract_ready_no_new_inference",
        "model_call_count": 0,
        "training_run_count": 0,
        "actual_user_feedback_count": 0,
        "private_source_body_count": 0,
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }

    _write_json(evidence_root / "evidence-source/source-manifest.json", source_manifest)
    _write_jsonl(evidence_root / "evidence-ledger/claim-evidence.jsonl", claims)
    _write_jsonl(evidence_root / "evidence-eval/eval-briefs.jsonl", evals)
    _write_json(evidence_root / "evidence-eval/failure-taxonomy.json", taxonomy)
    _write_jsonl(evidence_root / "evidence-eval/phase112-eval-cases.jsonl", cases)
    _write_json(evidence_root / "evidence-eval/holdout-integrity-check.json", isolation)
    _write_json(evidence_root / "evidence-eval/arm-comparison-contract.json", build_arm_comparison_contract())
    _write_json(evidence_root / "phase111-ci-reproducibility.json", ci_report)
    _write_json(evidence_root / "schema-report.json", schema_report)
    _write_json(evidence_root / "phase111-112-final-decision.json", final_decision)
    return final_decision


def validate(evidence_root: Path) -> dict[str, Any]:
    claims = _read_jsonl(evidence_root / "evidence-ledger/claim-evidence.jsonl")
    evals = _read_jsonl(evidence_root / "evidence-eval/eval-briefs.jsonl")
    cases = _read_jsonl(evidence_root / "evidence-eval/phase112-eval-cases.jsonl")
    claim_report = validate_claim_ledger(claims, expected_count=28)
    eval_report = validate_eval_manifest(evals, expected_count=30)
    case_report = validate_phase112_cases(cases, expected_count=70)
    isolation = _read_json(evidence_root / "evidence-eval/holdout-integrity-check.json")
    if isolation.get("collision_count") != 0 or isolation.get("passed") is not True:
        raise ValueError("stored holdout integrity check failed")
    decision = _read_json(evidence_root / "phase111-112-final-decision.json")
    required_zero = (
        "model_call_count",
        "training_run_count",
        "actual_user_feedback_count",
        "private_source_body_count",
    )
    if any(decision.get(key) != 0 for key in required_zero):
        raise ValueError(f"forbidden Phase111-112 activity recorded: {decision}")
    if decision.get("product_gate_qualified") is not False:
        raise ValueError("Phase111-112 cannot qualify the product gate")
    _phase110_contract()
    if file_sha256(PHASE85_TEST) != PHASE85_FROZEN_HASH:
        raise ValueError("Phase85 frozen driver test hash drifted")
    return {
        "status": "validated",
        "claim_ledger": claim_report,
        "eval_manifest": eval_report,
        "phase112_cases": case_report,
        "holdout_integrity": isolation,
        "phase110_unchanged": True,
        "phase85_frozen_test_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "validate"), nargs="?", default="generate")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    result = (
        generate(args.source_root, args.evidence_root, clean=args.clean)
        if args.command == "generate"
        else validate(args.evidence_root)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
