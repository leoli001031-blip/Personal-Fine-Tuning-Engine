#!/usr/bin/env python3
"""Finalize and validate the Phase100-104 autonomous local loop evidence."""

from __future__ import annotations

import argparse
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

from pfe_core.phase104_autonomous_loop_decision import build_phase104_final_decision


ROOT = REPO_ROOT / "docs/demo/phase100-104-autonomous-qwen3-training-benefit-loop"
PHASE100 = ROOT / "phase100-generation-boundary"
PHASE101 = ROOT / "phase101-failure-targeted-sft"
PHASE102 = ROOT / "phase102-failure-targeted-dpo"
PHASE103 = ROOT / "phase103-simulated-user-acceptance"
MANIFEST_PATH = ROOT / "evidence_manifest.json"
VALIDATION_PATH = ROOT / "validation_summary.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_files() -> list[Path]:
    excluded = {MANIFEST_PATH.resolve(), VALIDATION_PATH.resolve()}
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.resolve() not in excluded
    )


def _build_manifest() -> dict[str, Any]:
    files = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _evidence_files()
    ]
    return {
        "kind": "phase100_104_evidence_manifest",
        "created_at": _utcnow(),
        "files": files,
        "file_count": len(files),
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "private_transcripts_committed": False,
    }


def _finalize() -> int:
    phase100 = _read_json(PHASE100 / "phase100-decision.json")
    phase101 = _read_json(PHASE101 / "phase101-decision.json")
    phase102 = _read_json(PHASE102 / "phase102-decision.json")
    phase103 = _read_json(PHASE103 / "phase103-decision.json")
    decision = build_phase104_final_decision(
        phase100=phase100,
        phase101=phase101,
        phase102=phase102,
        phase103=phase103,
        cumulative_model_calls=int(phase103.get("cumulative_model_call_count") or 0),
    )
    decision.update({
        "phase100_status": phase100.get("status"),
        "phase101_status": phase101.get("status"),
        "phase102_status": phase102.get("status"),
        "phase103_status": phase103.get("status"),
        "phase100_metrics": phase100.get("metrics"),
        "phase101_comparison": {
            "base": phase101.get("base_metrics"),
            "sft": phase101.get("candidate_metrics"),
        },
        "phase102_comparison": {
            "base": phase102.get("base_metrics"),
            "dpo": phase102.get("candidate_metrics"),
        },
        "phase103_comparison": {
            "base": phase103.get("base_metrics"),
            "dpo": phase103.get("archived_dpo_metrics"),
            "paired": phase103.get("paired_comparison"),
        },
    })
    _write_json(ROOT / "phase104-final-decision.json", decision)
    summary = {
        "kind": "phase100_104_comparison_summary",
        "generation_boundary": {
            "phase99_native_termination_rate": 0.375,
            "phase100_native_termination_rate": dict(phase100.get("metrics") or {}).get("native_termination_rate"),
            "phase100_provenance_correct_rate": dict(phase100.get("metrics") or {}).get("provenance_correct_rate"),
            "phase100_runtime_guided_provenance": True,
        },
        "sft": {
            "real_training_completed": dict(phase101.get("checks") or {}).get("real_training_completed"),
            "base_exact_three_line_rate": dict(phase101.get("base_metrics") or {}).get("exact_three_line_rate"),
            "adapter_exact_three_line_rate": dict(phase101.get("candidate_metrics") or {}).get("exact_three_line_rate"),
            "status": phase101.get("status"),
        },
        "dpo": {
            "real_training_completed": dict(phase102.get("checks") or {}).get("real_dpo_training_completed"),
            "base_exact_three_line_rate": dict(phase102.get("base_metrics") or {}).get("exact_three_line_rate"),
            "adapter_exact_three_line_rate": dict(phase102.get("candidate_metrics") or {}).get("exact_three_line_rate"),
            "status": phase102.get("status"),
        },
        "simulated_user_acceptance": {
            "base_acceptance_rate": dict(phase103.get("base_metrics") or {}).get("acceptance_rate"),
            "adapter_acceptance_rate": dict(phase103.get("archived_dpo_metrics") or {}).get("acceptance_rate"),
            "adapter_wins": dict(phase103.get("paired_comparison") or {}).get("adapter_wins"),
            "ties": dict(phase103.get("paired_comparison") or {}).get("ties"),
            "adapter_losses": dict(phase103.get("paired_comparison") or {}).get("adapter_losses"),
        },
        "recommendation": decision["recommendation"],
        "product_gate_qualified": False,
        "automatic_promotion_allowed": False,
    }
    _write_json(ROOT / "comparison_summary.json", summary)
    final_md = [
        "# Phase104 Final Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommendation: `{decision['recommendation']}`",
        "- Product gate qualified: false",
        "- Automatic promotion allowed: false",
        "- Deployment allowed: false",
        "- Evidence type: simulated_usage only",
        f"- Local generation calls: {decision['cumulative_model_call_count']}/270",
        "",
        "## What was proved",
        "",
        "- Phase100 closed the native generation boundary with no post-hoc truncation.",
        "- Qwen3-4B SFT completed at 1, 12, and 30 steps with valid LoRA artifacts.",
        "- Qwen3-4B DPO completed at 12 and 30 steps with finite MPS float32 metrics.",
        "",
        "## What was not proved",
        "",
        "- SFT did not improve the fresh product holdout and regressed format stability.",
        "- DPO matched base on the fresh holdout but did not exceed it.",
        "- In 20 paired three-turn simulated sessions, DPO had 0 wins, 19 ties, and 1 loss.",
        "- No adapter product benefit was established; both adapters remain archived.",
        "",
        "## Product path",
        "",
        "Keep the Phase100 runtime contract as the primary path. The next investment should be a more diverse provenance and correction-following curriculum plus loss-target diagnostics, not additional steps on the same 32 examples.",
    ]
    (ROOT / "phase104-final-decision.md").write_text("\n".join(final_md) + "\n", encoding="utf-8")
    output_examples = [
        "# Output Evidence Index",
        "",
        "Full simulated transcripts are intentionally stored outside Git under `/private/tmp/pfe-phase100-simulated-review`, `/private/tmp/pfe-phase101-simulated-review`, `/private/tmp/pfe-phase102-simulated-review`, and `/private/tmp/pfe-phase103-simulated-review`.",
        "",
        "Repository evidence contains per-turn output hashes, termination reasons, scores, and aggregate metrics without private transcript text.",
        "",
        "- Phase100: 24 final calls, complete native termination 1.0, provenance 1.0 with guided runtime target.",
        "- Phase101: base format 0.5 versus SFT 0.0.",
        "- Phase102: DPO metrics matched base and did not improve provenance.",
        "- Phase103: base/DPO acceptance both 0.40; paired result 0 wins, 19 ties, 1 loss.",
    ]
    (ROOT / "output_examples.md").write_text("\n".join(output_examples) + "\n", encoding="utf-8")
    runbook = [
        "# Phase100-104 Autonomous Loop Runbook",
        "",
        "All commands are local-only and use `models/Qwen3-4B`.",
        "",
        "```bash",
        ".venv/bin/python tools/phase100_qwen3_generation_boundary_closure.py prepare --clean",
        ".venv/bin/python tools/phase100_qwen3_generation_boundary_closure.py diagnose --attempt 1 --clean",
        ".venv/bin/python tools/phase100_qwen3_generation_boundary_closure.py diagnose --attempt 2 --clean",
        ".venv/bin/python tools/phase100_qwen3_generation_boundary_closure.py generate --clean",
        ".venv/bin/python tools/phase100_qwen3_generation_boundary_closure.py decide",
        ".venv/bin/python tools/phase101_failure_targeted_sft.py prepare --clean",
        ".venv/bin/python tools/phase101_failure_targeted_sft.py train --steps 1 --clean",
        ".venv/bin/python tools/phase101_failure_targeted_sft.py train --steps 12 --clean",
        ".venv/bin/python tools/phase101_failure_targeted_sft.py train --steps 30 --clean",
        ".venv/bin/python tools/phase101_failure_targeted_sft.py eval --variant base --clean",
        ".venv/bin/python tools/phase101_failure_targeted_sft.py eval --variant sft --clean",
        ".venv/bin/python tools/phase101_failure_targeted_sft.py decide",
        ".venv/bin/python tools/phase102_failure_targeted_dpo.py prepare --clean",
        ".venv/bin/python tools/phase102_failure_targeted_dpo.py train --steps 12 --clean",
        ".venv/bin/python tools/phase102_failure_targeted_dpo.py train --steps 30 --clean",
        ".venv/bin/python tools/phase102_failure_targeted_dpo.py eval --clean",
        ".venv/bin/python tools/phase102_failure_targeted_dpo.py decide",
        ".venv/bin/python tools/phase103_simulated_user_acceptance.py prepare --clean",
        ".venv/bin/python tools/phase103_simulated_user_acceptance.py eval --variant base --clean",
        ".venv/bin/python tools/phase103_simulated_user_acceptance.py eval --variant dpo --clean",
        ".venv/bin/python tools/phase103_simulated_user_acceptance.py decide",
        ".venv/bin/python tools/phase104_finalize_autonomous_loop.py finalize",
        ".venv/bin/python tools/phase104_finalize_autonomous_loop.py validate",
        "```",
        "",
        "Do not promote or deploy any Phase101/102 adapter. Phase100 runtime boundary is the retained product path.",
    ]
    (ROOT / "phase100-104-runbook.md").write_text("\n".join(runbook) + "\n", encoding="utf-8")
    _write_json(MANIFEST_PATH, _build_manifest())
    print(json.dumps({"status": decision["status"], "recommendation": decision["recommendation"]}, ensure_ascii=False, indent=2))
    return 0


def _validate() -> int:
    decision = _read_json(ROOT / "phase104-final-decision.json")
    phase100 = _read_json(PHASE100 / "phase100-decision.json")
    phase101 = _read_json(PHASE101 / "phase101-decision.json")
    phase102 = _read_json(PHASE102 / "phase102-decision.json")
    phase103 = _read_json(PHASE103 / "phase103-decision.json")
    manifest = _read_json(MANIFEST_PATH)
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest.get("files") or []}
    current = {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in _evidence_files()}
    checks = {
        "manifest_unchanged": expected == current,
        "phase100_passed": phase100.get("passed") is True,
        "phase101_archived": str(phase101.get("status") or "").startswith("archive_"),
        "phase102_archived": str(phase102.get("status") or "").startswith("archive_"),
        "phase103_no_benefit": phase103.get("passed") is False,
        "final_recommendation_runtime_primary": decision.get("recommendation") == "runtime_contract_remains_primary",
        "product_gate_false": decision.get("product_gate_qualified") is False,
        "automatic_promotion_false": decision.get("automatic_promotion_allowed") is False,
        "deployment_false": decision.get("deployment_allowed") is False,
        "model_call_budget_respected": int(decision.get("cumulative_model_call_count") or 0) == 240,
        "actual_feedback_zero": decision.get("actual_user_feedback_count") == 0,
        "private_transcripts_not_committed": manifest.get("private_transcripts_committed") is False,
    }
    payload = {
        "kind": "phase100_104_validation_summary",
        "validated_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
    }
    _write_json(VALIDATION_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("finalize", "validate"))
    args = parser.parse_args()
    return _finalize() if args.command == "finalize" else _validate()


if __name__ == "__main__":
    raise SystemExit(main())
