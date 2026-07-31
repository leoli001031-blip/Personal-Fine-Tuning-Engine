#!/usr/bin/env python3
"""Run Phase16 minimal real DPO runtime proof."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from pfe_core.errors import TrainingError
from pfe_core.trainer.executors import execute_dpo_training, probe_trainer_executor


PHASE16_DOCS_DIR = Path("docs/demo/phase16-dpo-runtime-proof")
PHASE15_DOCS_DIR = Path("docs/demo/phase15-true-preference-boundary-training")


def _load_local_tool(module_name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase15 = _load_local_tool("phase15_preference_boundary_training", "phase15_preference_boundary_training.py")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def dpo_runtime_preflight() -> dict[str, Any]:
    modules = ("torch", "transformers", "peft", "accelerate", "trl", "datasets")
    availability = {module: _module_available(module) for module in modules}
    try:
        probe = probe_trainer_executor("dpo", allow_mock_fallback=False).to_dict()
        probe_ready = bool(probe.get("ready"))
        probe_error = None
    except TrainingError as exc:
        probe = {}
        probe_ready = False
        probe_error = str(exc)
    missing = [module for module, ready in availability.items() if not ready]
    return {
        "kind": "phase16_dpo_runtime_preflight",
        "module_availability": availability,
        "missing_modules": missing,
        "strict_probe_ready": probe_ready,
        "strict_probe": probe,
        "strict_probe_error": probe_error,
        "ready": not missing and probe_ready,
        "bitsandbytes_required_for_this_proof": False,
        "created_at": _utcnow_iso(),
    }


def load_or_build_phase15_samples(*, evidence_dir: Path, phase15_evidence_dir: Path, pair_limit: int) -> dict[str, Any]:
    samples_path = phase15_evidence_dir / "dpo_samples.jsonl"
    quality_path = phase15_evidence_dir / "quality_report.json"
    if not samples_path.exists() or not quality_path.exists():
        phase15.build_phase15_preference_dataset(
            evidence_dir=phase15_evidence_dir,
            phase14_evidence_dir=Path("docs/demo/phase14-hard-negative-boundary-training/evidence-real-qwen3-8b-hard-negative-v2"),
            pair_limit=max(pair_limit, 80),
        )
    samples = _read_jsonl(samples_path)
    quality = _read_json(quality_path)
    selected = samples[:pair_limit]
    _write_jsonl(evidence_dir / "selected_dpo_samples.jsonl", selected)
    return {
        "kind": "phase16_phase15_sample_selection",
        "source_samples_path": str(samples_path),
        "source_quality_path": str(quality_path),
        "source_quality": quality,
        "source_sample_count": len(samples),
        "selected_sample_count": len(selected),
        "selected_samples_path": str(evidence_dir / "selected_dpo_samples.jsonl"),
        "created_at": _utcnow_iso(),
    }


def build_tiny_dpo_job_spec(
    *,
    samples: list[Mapping[str, Any]],
    base_model: str,
    output_dir: Path,
    epochs: int,
    beta: float,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, Any]:
    examples = [
        {
            "sample_id": item.get("sample_id"),
            "instruction": item.get("instruction"),
            "chosen": item.get("chosen"),
            "rejected": item.get("rejected"),
            "sample_type": "dpo",
        }
        for item in samples
    ]
    return {
        "backend": "dpo",
        "execution_backend": "dpo",
        "execution_executor": "dpo",
        "executor_mode": "real_import",
        "dry_run": True,
        "output_dir": str(output_dir),
        "recipe": {
            "training": {
                "method": "lora",
                "epochs": epochs,
                "train_type": "dpo",
                "base_model": base_model,
                "num_train_samples": len(examples),
                "output_dir": str(output_dir),
            },
            "peft": {
                "trainer_class": "trl.DPOTrainer",
                "dpo_config": {
                    "beta": beta,
                    "label_smoothing": 0.0,
                    "max_length": max_length,
                    "max_prompt_length": max_prompt_length,
                },
            },
        },
        "training_examples": examples,
        "phase16": {
            "proof_scope": "runtime_executor_only_not_product_adapter",
            "source_phase": "phase15",
        },
    }


def validate_dpo_artifact(result: Mapping[str, Any]) -> dict[str, Any]:
    artifact_dir = Path(str(result.get("artifact_dir") or ""))
    artifacts = _dict(result.get("artifacts"))
    adapter_config = Path(str(artifacts.get("adapter_config") or artifact_dir / "adapter_config.json"))
    adapter_model = Path(str(artifacts.get("adapter_model") or artifact_dir / "adapter_model.safetensors"))
    return {
        "kind": "phase16_dpo_artifact_validation",
        "artifact_dir": str(artifact_dir) if str(artifact_dir) else "",
        "artifact_dir_exists": artifact_dir.exists(),
        "adapter_config_exists": adapter_config.exists(),
        "adapter_model_exists": adapter_model.exists(),
        "valid": artifact_dir.exists() and adapter_config.exists() and adapter_model.exists(),
        "created_at": _utcnow_iso(),
    }


def run_dpo_runtime_proof(
    *,
    evidence_dir: Path,
    job_spec: Mapping[str, Any],
    preflight: Mapping[str, Any],
    run_real_dpo_proof: bool,
) -> dict[str, Any]:
    dry_run = execute_dpo_training(job_spec=job_spec, dry_run=True)
    _write_json(evidence_dir / "dpo_dry_run_plan.json", {"kind": "phase16_dpo_dry_run_plan", "result": dry_run, "created_at": _utcnow_iso()})
    if not run_real_dpo_proof:
        payload = {
            "kind": "phase16_real_dpo_runtime_proof",
            "real_training": "not_started",
            "training_run": False,
            "skip_reason": "skip_real_dpo_proof",
            "dry_run_result": dry_run,
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "training_attempt.json", payload)
        _write_json(evidence_dir / "train_log.json", payload)
        return payload
    if not preflight.get("ready"):
        payload = {
            "kind": "phase16_real_dpo_runtime_proof",
            "real_training": "blocked",
            "training_run": False,
            "blocked_reason": "dpo_runtime_dependencies_not_ready",
            "missing_modules": list(preflight.get("missing_modules") or []),
            "dry_run_result": dry_run,
            "preflight": dict(preflight),
            "created_at": _utcnow_iso(),
        }
        _write_json(evidence_dir / "training_attempt.json", payload)
        _write_json(evidence_dir / "train_log.json", payload)
        return payload
    result = execute_dpo_training(job_spec={**dict(job_spec), "dry_run": False}, dry_run=False)
    artifact_validation = validate_dpo_artifact(result)
    payload = {
        "kind": "phase16_real_dpo_runtime_proof",
        "real_training": "completed" if result.get("status") == "completed" else "failed",
        "training_run": True,
        "result": result,
        "artifact_validation": artifact_validation,
        "proof_scope": "runtime_executor_only_not_product_adapter",
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "training_attempt.json", payload)
    _write_json(evidence_dir / "train_log.json", payload)
    _write_json(evidence_dir / "artifact_validation.json", artifact_validation)
    return payload


def phase16_decision(*, preflight: Mapping[str, Any], sample_selection: Mapping[str, Any], training_attempt: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    artifact_validation = _dict(training_attempt.get("artifact_validation"))
    if not preflight.get("ready"):
        reasons.append("dpo_runtime_dependencies_not_ready")
    if int(sample_selection.get("selected_sample_count", 0)) <= 0:
        reasons.append("no_phase15_dpo_samples_selected")
    if training_attempt.get("real_training") != "completed":
        reasons.append("real_dpo_runtime_proof_not_completed")
    if training_attempt.get("real_training") == "completed" and not artifact_validation.get("valid"):
        reasons.append("dpo_adapter_artifact_missing")
    if reasons:
        return {
            "kind": "phase16_dpo_runtime_decision",
            "status": "blocked",
            "recommendation": "archive",
            "promotion_allowed": False,
            "auto_promotion_allowed": False,
            "manual_review_required": False,
            "reasons": sorted(set(reasons)),
            "created_at": _utcnow_iso(),
        }
    return {
        "kind": "phase16_dpo_runtime_decision",
        "status": "runtime_proof_passed",
        "recommendation": "proceed_to_qwen_dpo_probe_after_manual_review",
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
        "manual_review_required": True,
        "reasons": ["tiny_model_runtime_proof_passed", "not_a_product_adapter", "qwen_boundary_eval_required"],
        "created_at": _utcnow_iso(),
    }


def _write_runbook(docs_dir: Path) -> str:
    text = """# Phase16 DPO Runtime Proof Runbook

Phase16 proves that the DPO runtime can execute a real `trl.DPOTrainer` job and materialize an adapter artifact. This is a runtime proof only, not a product adapter quality claim.

## Default Smoke

```bash
.venv/bin/python tools/phase16_dpo_runtime_proof.py \\
  --evidence-dir docs/demo/phase16-dpo-runtime-proof/evidence \\
  --clean-evidence \\
  --skip-real-dpo-proof
```

## Real Tiny DPO Runtime Proof

```bash
.venv/bin/python tools/phase16_dpo_runtime_proof.py \\
  --evidence-dir docs/demo/phase16-dpo-runtime-proof/evidence-real-dpo-tiny \\
  --clean-evidence \\
  --run-real-dpo-proof \\
  --train-sample-limit 2 \\
  --training-output-dir trainer_job_outputs/phase16-dpo-runtime-proof-tiny
```

Passing this proof only permits a later Qwen DPO probe after manual review. It never promotes a product adapter by itself.
"""
    path = docs_dir / "phase16-runbook.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _write_final_decision(docs_dir: Path, report: Mapping[str, Any]) -> str:
    preflight = _dict(report.get("dpo_preflight"))
    sample_selection = _dict(report.get("sample_selection"))
    training = _dict(report.get("training_attempt"))
    artifact = _dict(training.get("artifact_validation"))
    decision = _dict(report.get("decision"))
    text = (
        "# Phase16 Final Decision\n\n"
        "## Goal\n\n"
        "- Prove the DPO runtime can run a real `trl.DPOTrainer` job.\n"
        "- Keep this separate from product adapter quality or Qwen boundary eval.\n\n"
        "## Runtime\n\n"
        f"- DPO preflight ready: {preflight.get('ready')}\n"
        f"- Missing modules: {preflight.get('missing_modules')}\n"
        f"- BitsAndBytes required for this proof: {preflight.get('bitsandbytes_required_for_this_proof')}\n\n"
        "## Data\n\n"
        f"- Source Phase15 samples: {sample_selection.get('source_sample_count')}\n"
        f"- Selected samples: {sample_selection.get('selected_sample_count')}\n\n"
        "## Training\n\n"
        f"- Real training: {training.get('real_training')}\n"
        f"- Training run: {training.get('training_run')}\n"
        f"- Artifact valid: {artifact.get('valid')}\n"
        f"- Artifact dir: {artifact.get('artifact_dir')}\n\n"
        "## Decision\n\n"
        f"- Recommendation: {decision.get('recommendation')}\n"
        f"- Status: {decision.get('status')}\n"
        f"- Reasons: {decision.get('reasons')}\n\n"
        "Phase16 can pass only as a runtime proof. It cannot promote a product adapter. The next product step is a small Qwen DPO probe with boundary holdout eval.\n"
    )
    path = docs_dir / "phase16-final-decision.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase16 DPO runtime proof.")
    parser.add_argument("--evidence-dir", type=Path, default=PHASE16_DOCS_DIR / "evidence")
    parser.add_argument("--phase15-evidence-dir", type=Path, default=PHASE15_DOCS_DIR / "evidence-real-dpo-preflight")
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--skip-real-dpo-proof", action="store_true")
    parser.add_argument("--run-real-dpo-proof", action="store_true")
    parser.add_argument("--train-sample-limit", type=int, default=2)
    parser.add_argument("--base-model", default="hf-internal-testing/tiny-random-gpt2")
    parser.add_argument("--training-output-dir", type=Path, default=Path("trainer_job_outputs/phase16-dpo-runtime-proof-tiny"))
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-max-length", type=int, default=128)
    parser.add_argument("--dpo-max-prompt-length", type=int, default=96)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    evidence_dir = args.evidence_dir.expanduser().resolve()
    docs_dir = evidence_dir.parent if evidence_dir.name.startswith("evidence") else evidence_dir
    if evidence_dir.exists() and args.clean_evidence:
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    if args.training_output_dir.exists() and args.clean_evidence and args.run_real_dpo_proof:
        shutil.rmtree(args.training_output_dir)
    _write_runbook(docs_dir)

    preflight = dpo_runtime_preflight()
    _write_json(evidence_dir / "dpo_preflight.json", preflight)
    sample_selection = load_or_build_phase15_samples(
        evidence_dir=evidence_dir,
        phase15_evidence_dir=args.phase15_evidence_dir.expanduser().resolve(),
        pair_limit=args.train_sample_limit,
    )
    selected_samples = _read_jsonl(evidence_dir / "selected_dpo_samples.jsonl")
    job_spec = build_tiny_dpo_job_spec(
        samples=selected_samples,
        base_model=args.base_model,
        output_dir=args.training_output_dir.expanduser().resolve(),
        epochs=args.dpo_epochs,
        beta=args.dpo_beta,
        max_length=args.dpo_max_length,
        max_prompt_length=args.dpo_max_prompt_length,
    )
    _write_json(evidence_dir / "dpo_job_spec.json", job_spec)
    training_attempt = run_dpo_runtime_proof(
        evidence_dir=evidence_dir,
        job_spec=job_spec,
        preflight=preflight,
        run_real_dpo_proof=bool(args.run_real_dpo_proof and not args.skip_real_dpo_proof),
    )
    decision = phase16_decision(preflight=preflight, sample_selection=sample_selection, training_attempt=training_attempt)
    _write_json(evidence_dir / "decision.json", decision)
    comparison = {
        "kind": "phase16_dpo_runtime_proof_summary",
        "dpo_preflight": preflight,
        "sample_selection": sample_selection,
        "training_attempt": training_attempt,
        "decision": decision,
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_dir / "comparison_summary.json", comparison)
    _write_json(docs_dir / "comparison_summary.json", comparison)
    final_decision = _write_final_decision(docs_dir, comparison)
    comparison["phase16_final_decision_path"] = final_decision
    _write_json(evidence_dir / "comparison_summary.json", comparison)
    _write_json(docs_dir / "comparison_summary.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
