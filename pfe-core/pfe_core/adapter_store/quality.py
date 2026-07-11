"""Strict adapter artifact and serving-quality validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SERVING_QUALITY_REPORT_FILENAME = "serving_quality_report.json"
MIN_SERVING_HOLDOUT_COUNT = 20


def _candidate_artifact_paths(version_dir: Path, manifest: Mapping[str, Any]) -> list[Path]:
    artifact_name = str(manifest.get("artifact_name") or "adapter_model.safetensors")
    candidates = [
        version_dir / artifact_name,
        version_dir / "adapter_model.safetensors",
        version_dir / "peft_lora" / "adapter_model.safetensors",
        version_dir / "dpo_adapter" / "adapter_model.safetensors",
        version_dir / "adapter_model.gguf",
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def validate_adapter_artifact(version_dir: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that an adapter artifact is materialized and machine-readable."""

    root = Path(version_dir).expanduser()
    existing = next((path for path in _candidate_artifact_paths(root, manifest) if path.is_file()), None)
    if existing is None:
        return {
            "valid": False,
            "reason": "adapter_artifact_missing",
            "checked_paths": [str(path) for path in _candidate_artifact_paths(root, manifest)],
        }

    if existing.suffix == ".gguf":
        header = existing.read_bytes()[:4]
        valid = header == b"GGUF" and existing.stat().st_size > 32
        return {
            "valid": valid,
            "reason": None if valid else "invalid_gguf_artifact",
            "path": str(existing),
            "format": "gguf",
            "size_bytes": existing.stat().st_size,
        }

    if existing.suffix != ".safetensors":
        return {
            "valid": False,
            "reason": "unsupported_adapter_artifact_format",
            "path": str(existing),
        }

    try:
        from safetensors import safe_open

        with safe_open(str(existing), framework="pt", device="cpu") as handle:
            tensor_names = list(handle.keys())
            parameter_count = 0
            for name in tensor_names:
                tensor = handle.get_tensor(name)
                parameter_count += int(tensor.numel())
    except Exception as exc:
        return {
            "valid": False,
            "reason": "invalid_safetensors_artifact",
            "path": str(existing),
            "error": f"{exc.__class__.__name__}: {exc}",
        }

    lora_tensor_count = sum("lora_" in name.lower() for name in tensor_names)
    valid = bool(tensor_names) and parameter_count > 0 and lora_tensor_count > 0
    return {
        "valid": valid,
        "reason": None if valid else "safetensors_missing_lora_tensors",
        "path": str(existing),
        "format": "safetensors",
        "size_bytes": existing.stat().st_size,
        "tensor_count": len(tensor_names),
        "lora_tensor_count": lora_tensor_count,
        "parameter_count": parameter_count,
    }

def load_serving_quality_report(version_dir: str | Path) -> dict[str, Any]:
    path = Path(version_dir).expanduser() / SERVING_QUALITY_REPORT_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def evaluate_serving_quality_gate(
    *,
    version_dir: str | Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate persisted quality evidence before an adapter can serve."""

    quality_report = dict(report or load_serving_quality_report(version_dir))
    artifact = validate_adapter_artifact(version_dir, manifest)
    holdout = quality_report.get("holdout") if isinstance(quality_report.get("holdout"), Mapping) else {}
    holdout_count = int(holdout.get("count") or quality_report.get("holdout_count") or 0)
    leakage = bool(
        quality_report.get("training_leakage_detected")
        or (quality_report.get("scores") or {}).get("training_leakage_detected")
    )
    reasons: list[str] = []
    if not quality_report:
        reasons.append("serving_quality_report_missing")
    if quality_report and quality_report.get("passed") is not True:
        reasons.append("serving_quality_report_not_passed")
    if not artifact.get("valid"):
        reasons.append(str(artifact.get("reason") or "adapter_artifact_invalid"))
    if holdout_count < MIN_SERVING_HOLDOUT_COUNT:
        reasons.append("insufficient_generic_holdout")
    if holdout.get("passed") is not True:
        reasons.append("generic_holdout_not_passed")
    if leakage:
        reasons.append("training_leakage_detected")
    return {
        "kind": "pfe_adapter_serving_quality_gate",
        "version": manifest.get("version"),
        "passed": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "artifact_validation": artifact,
        "holdout_count": holdout_count,
        "training_leakage_detected": leakage,
        "report_path": str(Path(version_dir).expanduser() / SERVING_QUALITY_REPORT_FILENAME),
    }
