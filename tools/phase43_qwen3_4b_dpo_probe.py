#!/usr/bin/env python3
"""Run the optional evidenced Phase43 Qwen3-4B 12-step DPO probe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.phase43_personal_preference_benefit import build_phase43_dpo_job_spec
from pfe_core.trainer.executors import execute_dpo_training, probe_trainer_executor


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase43-qwen3-4b-personal-preference-benefit-proof"
DPO_EVIDENCE = EVIDENCE_ROOT / "evidence-training-dpo"
CANDIDATE_PATH = EVIDENCE_ROOT / "evidence-candidate-review" / "selected_preference_pairs.jsonl"
MODEL_SELECTION_PATH = EVIDENCE_ROOT / "evidence-baseline" / "model_selection.json"
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    steps = max(1, int(args.steps))
    output_dir = REPO_ROOT / "trainer_job_outputs" / f"phase43-qwen3-4b-dpo-{steps}step"
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    if args.clean and DPO_EVIDENCE.exists():
        for path in DPO_EVIDENCE.iterdir():
            if path.name != "training_manifest.json" and not path.name.startswith("attempt-"):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    DPO_EVIDENCE.mkdir(parents=True, exist_ok=True)

    candidates = _read_jsonl(CANDIDATE_PATH)
    model_selection = _read_json(MODEL_SELECTION_PATH)
    try:
        executor_probe = probe_trainer_executor("dpo", allow_mock_fallback=False).to_dict()
    except Exception as exc:
        executor_probe = {"ready": False, "error": f"{exc.__class__.__name__}: {exc}"}
    try:
        import torch

        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        mps_available = False
    preflight = {
        "kind": "phase43_qwen3_4b_dpo_preflight",
        "model_selected": model_selection.get("status") == "selected",
        "memory_preflight_passed": model_selection.get("dpo_memory_preflight_passed") is True,
        "executor_probe": executor_probe,
        "executor_ready": executor_probe.get("ready") is True,
        "mps_available": mps_available,
        "candidate_count": len(candidates),
        "chosen_rejected_boundaries_valid": all(
            row.get("instruction") and row.get("chosen") and row.get("rejected") and row.get("chosen") != row.get("rejected")
            for row in candidates
        ),
    }
    preflight["ready"] = all(
        (
            preflight["model_selected"],
            preflight["memory_preflight_passed"],
            preflight["executor_ready"],
            preflight["mps_available"],
            preflight["candidate_count"] >= 12,
            preflight["chosen_rejected_boundaries_valid"],
        )
    )
    _write_json(DPO_EVIDENCE / "dpo_preflight.json", preflight)

    job_spec = build_phase43_dpo_job_spec(
        pairs=candidates,
        base_model=str(MODEL_PATH),
        output_dir=str(output_dir),
        max_steps=steps,
    )
    _write_json(DPO_EVIDENCE / "training_manifest.json", job_spec)
    dry_run = execute_dpo_training(job_spec=job_spec, dry_run=True)
    _write_json(DPO_EVIDENCE / "dpo_dry_run_plan.json", dry_run)
    if preflight.get("ready") is not True:
        attempt = {
            "kind": "phase43_qwen3_4b_dpo_training_attempt",
            "status": "blocked",
            "real_training": False,
            "requested_steps": steps,
            "reason": "dpo_preflight_failed",
            "preflight": preflight,
            "actual_product_benefit_claim_allowed": False,
            "auto_promotion_allowed": False,
        }
        _write_json(DPO_EVIDENCE / "training_attempt.json", attempt)
        _write_json(DPO_EVIDENCE / "train_log.json", attempt)
        print(json.dumps(attempt, ensure_ascii=False, indent=2))
        return 2

    started = time.perf_counter()
    started_at = _utcnow()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = execute_dpo_training(job_spec=job_spec, dry_run=False)
    real = dict(result.get("real_execution") or {})
    artifact_dir = Path(str(real.get("artifact_dir") or result.get("artifact_dir") or ""))
    adapter_path = artifact_dir / "adapter_model.safetensors"
    if artifact_dir.exists():
        validation = validate_adapter_artifact(
            artifact_dir,
            {"artifact_name": adapter_path.name, "artifact_format": "peft_lora"},
        )
    else:
        validation = {"valid": False, "reason": "artifact_dir_missing"}
    validation.update(
        {
            "artifact_dir": str(artifact_dir),
            "adapter_path": str(adapter_path),
            "sha256": _sha256(adapter_path) if adapter_path.exists() else None,
            "parameters_updated": real.get("parameters_updated"),
            "steps": real.get("steps"),
        }
    )
    completed = (
        result.get("status") == "completed"
        and real.get("success") is True
        and real.get("parameters_updated") is True
        and int(real.get("steps") or 0) >= steps
        and validation.get("valid") is True
    )
    attempt = {
        "kind": "phase43_qwen3_4b_dpo_training_attempt",
        "status": "completed" if completed else "failed",
        "real_training": completed,
        "model": str(MODEL_PATH),
        "requested_steps": steps,
        "started_at": started_at,
        "finished_at": _utcnow(),
        "duration_seconds": round(time.perf_counter() - started, 4),
        "max_rss_before_bytes": rss_before,
        "max_rss_after_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "result_status": result.get("status"),
        "error": result.get("error"),
        "execution": real,
        "adapter_validation": validation,
        "actual_user_feedback": False,
        "simulated_lab_benefit_not_yet_evaluated": True,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(DPO_EVIDENCE / "training_attempt.json", attempt)
    _write_json(DPO_EVIDENCE / "adapter_validation.json", validation)
    _write_json(DPO_EVIDENCE / "train_log.json", {"loss_history": real.get("loss_history") or [], "result": result})
    _write_json(DPO_EVIDENCE / "loss_history.json", real.get("loss_history") or [])
    _write_json(
        DPO_EVIDENCE / "parameter_fingerprint_before_after.json",
        {
            "before": real.get("parameter_fingerprint_before"),
            "after": real.get("parameter_fingerprint_after"),
            "parameters_updated": real.get("parameters_updated"),
        },
    )
    print(json.dumps({key: attempt.get(key) for key in ("status", "requested_steps", "duration_seconds", "error")}, indent=2))
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
