#!/usr/bin/env python3
"""Freeze Phase42 and prepare Phase43 candidates, holdout, and model evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.phase43_personal_preference_benefit import (
    build_holdout_integrity_check,
    build_phase43_dpo_job_spec,
    build_phase43_holdout_sessions,
    build_phase43_preference_taxonomy,
    build_phase43_sft_job_spec,
    review_phase41_v2_candidates,
)


EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase43-qwen3-4b-personal-preference-benefit-proof"
PHASE42_ROOT = REPO_ROOT / "docs" / "demo" / "phase42-trustworthy-training-runtime-hardening"
PHASE41_V2_PATH = (
    PHASE42_ROOT / "evidence-candidate-quality" / "phase41_v2_selected_preference_pairs.jsonl"
)
MODEL_PATH = REPO_ROOT / "models" / "Qwen3-4B"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(args: list[str], *, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:
        return {"command": args, "returncode": None, "error": f"{exc.__class__.__name__}: {exc}"}


def _module_versions() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("torch", "transformers", "peft", "accelerate", "trl", "datasets", "safetensors", "mlx", "mlx_lm"):
        try:
            module = importlib.import_module(name)
            result[name] = {"available": True, "version": getattr(module, "__version__", "installed")}
        except Exception as exc:
            result[name] = {"available": False, "error": f"{exc.__class__.__name__}: {exc}"}
    return result


def _system_profile() -> dict[str, Any]:
    memory = _command(["sysctl", "-n", "hw.memsize"])
    try:
        memory_bytes = int(str(memory.get("stdout") or "0").strip())
    except ValueError:
        memory_bytes = 0
    disk = shutil.disk_usage(REPO_ROOT)
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:
        torch_info = {"error": str(exc)}
    return {
        "created_at": _utcnow(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "memory_bytes": memory_bytes,
        "memory_gib": round(memory_bytes / 1024**3, 3) if memory_bytes else None,
        "disk_free_gib": round(disk.free / 1024**3, 3),
        "torch_device": torch_info,
    }


def _model_selection(*, full_hash: bool) -> dict[str, Any]:
    config_path = MODEL_PATH / "config.json"
    index_path = MODEL_PATH / "model.safetensors.index.json"
    if not config_path.exists() or not index_path.exists():
        return {
            "kind": "phase43_model_selection",
            "status": "blocked",
            "selected_model": None,
            "reason": "local_qwen3_4b_incomplete",
            "model_path": str(MODEL_PATH),
        }
    config = _read_json(config_path)
    index = _read_json(index_path)
    shards = sorted(MODEL_PATH.glob("*.safetensors"))
    files = []
    for path in shards:
        files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path) if full_hash else None,
            }
        )
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    total_size = int(dict(index.get("metadata") or {}).get("total_size") or sum(item["size_bytes"] for item in files))
    system = _system_profile()
    memory_gib = float(system.get("memory_gib") or 0.0)
    sft_estimate = 24.0
    dpo_estimate = 72.0
    return {
        "kind": "phase43_model_selection",
        "status": "selected" if len(files) == 3 else "blocked",
        "selected_model": "Qwen3-4B" if len(files) == 3 else None,
        "model_path": str(MODEL_PATH),
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
        "declared_dtype": config.get("torch_dtype") or config.get("dtype"),
        "parameter_scale": "4B",
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "weight_size_bytes": total_size,
        "weight_size_gib": round(total_size / 1024**3, 3),
        "shard_count": len(files),
        "files": files,
        "full_shard_hashes_computed": full_hash,
        "model_bundle_sha256": combined,
        "sft_estimated_peak_memory_gib": sft_estimate,
        "sft_memory_preflight_passed": memory_gib >= sft_estimate * 1.5,
        "dpo_estimated_peak_memory_gib": dpo_estimate,
        "dpo_memory_preflight_passed": memory_gib >= dpo_estimate * 1.5,
        "dpo_runtime_note": "Current executor capability must still be checked before a real DPO launch.",
        "system_profile": system,
        "selection_reason": "local unquantized Qwen3-4B is complete and fits the 128 GiB host SFT budget",
        "download_performed": False,
        "qwen27b_training_allowed": False,
    }


def _baseline(workspace: str) -> dict[str, Any]:
    store = AdapterStore(home=Path.home() / ".pfe", workspace=workspace)
    records = store.list_version_records()
    raw_version_006 = next((dict(row) for row in records if str(row.get("version")) == "20260617-006"), {})
    version_006 = {
        key: raw_version_006.get(key)
        for key in (
            "version",
            "workspace",
            "state",
            "base_model",
            "artifact_format",
            "artifact_path",
            "adapter_dir",
            "created_at",
            "updated_at",
            "archived_at",
            "num_samples",
        )
        if key in raw_version_006
    }
    phase42_final = _read_json(PHASE42_ROOT / "phase42-final-decision.json")
    phase42_training = _read_json(PHASE42_ROOT / "evidence-real-training" / "training_attempt.json")
    phase41_v2 = _read_json(PHASE42_ROOT / "evidence-candidate-quality" / "phase41_v2_manifest.json")
    return {
        "kind": "phase43_frozen_phase42_baseline",
        "created_at": _utcnow(),
        "git": {
            "head": _command(["git", "rev-parse", "HEAD"]),
            "branch": _command(["git", "branch", "--show-current"]),
            "status": _command(["git", "status", "--short", "--branch"]),
        },
        "phase42_final_decision": phase42_final,
        "phase42_training_attempt": phase42_training,
        "phase41_v2_manifest": phase41_v2,
        "phase41_v2_source_path": str(PHASE41_V2_PATH),
        "phase41_v2_source_sha256": _sha256(PHASE41_V2_PATH),
        "adapter_store": {
            "workspace": workspace,
            "current_latest_version": store.current_latest_version(),
            "version_20260617_006": version_006,
            "version_006_remains_archived": version_006.get("state") == "archived",
            "version_006_is_not_latest": store.current_latest_version() != "20260617-006",
        },
        "pfe_next": _command([str(REPO_ROOT / ".venv" / "bin" / "pfe"), "next", "--workspace", workspace]),
        "runtime_processes": _command(["pgrep", "-fl", "ollama|pfe|phase43|python.*train"]),
        "ollama": {
            "list": _command(["ollama", "list"]),
            "gemma4_show": _command(["ollama", "show", "gemma4:31b"]),
            "qwen36_show": _command(["ollama", "show", "qwen3.6"]),
        },
        "dependencies": _module_versions(),
        "system": _system_profile(),
        "phase42_canonical_evidence_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="user_default")
    parser.add_argument("--holdout-count", type=int, default=40)
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--skip-full-model-hash", action="store_true")
    args = parser.parse_args()

    if args.clean_evidence and EVIDENCE_ROOT.exists():
        shutil.rmtree(EVIDENCE_ROOT)
    for name in (
        "evidence-baseline",
        "evidence-candidate-review",
        "evidence-training-sft",
        "evidence-training-dpo",
        "evidence-holdout",
        "evidence-blind-eval",
    ):
        (EVIDENCE_ROOT / name).mkdir(parents=True, exist_ok=True)

    baseline = _baseline(args.workspace)
    model_selection = _model_selection(full_hash=not args.skip_full_model_hash)
    taxonomy = build_phase43_preference_taxonomy()
    holdout = build_phase43_holdout_sessions(args.holdout_count)
    source_candidates = _read_jsonl(PHASE41_V2_PATH)
    review = review_phase41_v2_candidates(source_candidates, holdout_sessions=holdout["sessions"])
    selected = list(review.get("selected_preference_pairs") or [])
    integrity = build_holdout_integrity_check(selected, holdout["sessions"])

    sft_job = build_phase43_sft_job_spec(
        pairs=selected,
        base_model=str(MODEL_PATH),
        output_dir=str(REPO_ROOT / "trainer_job_outputs" / "phase43-qwen3-4b-sft-12step"),
        max_steps=12,
    )
    dpo_job = build_phase43_dpo_job_spec(
        pairs=selected,
        base_model=str(MODEL_PATH),
        output_dir=str(REPO_ROOT / "trainer_job_outputs" / "phase43-qwen3-4b-dpo-12step"),
        max_steps=12,
    )

    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "baseline_snapshot.json", baseline)
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "model_selection.json", model_selection)
    _write_json(EVIDENCE_ROOT / "evidence-candidate-review" / "preference_taxonomy.json", taxonomy)
    _write_json(EVIDENCE_ROOT / "evidence-candidate-review" / "candidate_review.json", review)
    _write_jsonl(EVIDENCE_ROOT / "evidence-candidate-review" / "selected_preference_pairs.jsonl", selected)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout.json", holdout)
    _write_json(EVIDENCE_ROOT / "evidence-holdout" / "holdout_integrity_check.json", integrity)
    _write_json(
        EVIDENCE_ROOT / "evidence-holdout" / "source_manifest.json",
        {
            "kind": "phase43_source_manifest",
            "training_source": str(PHASE41_V2_PATH),
            "training_source_sha256": _sha256(PHASE41_V2_PATH),
            "holdout_source": "phase43_deterministic_simulated_multiturn_holdout",
            "holdout_manifest_sha256": holdout["manifest_sha256"],
            "holdout_not_for_training": True,
            "actual_user_feedback": False,
            "actual_product_benefit_claim_allowed": False,
        },
    )
    _write_json(EVIDENCE_ROOT / "evidence-training-sft" / "training_manifest.json", sft_job)
    _write_json(EVIDENCE_ROOT / "evidence-training-dpo" / "training_manifest.json", dpo_job)

    ready = (
        model_selection.get("status") == "selected"
        and model_selection.get("sft_memory_preflight_passed") is True
        and review.get("approved_count", 0) >= 12
        and integrity.get("passed") is True
        and baseline["adapter_store"]["version_006_remains_archived"] is True
        and baseline["adapter_store"]["version_006_is_not_latest"] is True
    )
    preparation = {
        "kind": "phase43_preparation_decision",
        "status": "ready_for_1_step_probe" if ready else "blocked",
        "checks": {
            "qwen3_4b_selected": model_selection.get("status") == "selected",
            "sft_memory_preflight_passed": model_selection.get("sft_memory_preflight_passed") is True,
            "candidate_review_threshold_met": review.get("approved_count", 0) >= 12,
            "holdout_integrity_passed": integrity.get("passed") is True,
            "bad_adapter_006_remains_archived": baseline["adapter_store"]["version_006_remains_archived"] is True,
            "bad_adapter_006_not_latest": baseline["adapter_store"]["version_006_is_not_latest"] is True,
        },
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(EVIDENCE_ROOT / "evidence-baseline" / "preparation_decision.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
