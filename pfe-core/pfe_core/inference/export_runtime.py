"""Lightweight local execution helpers for export planning.

This layer sits on top of :mod:`pfe_core.inference.export`. Phase 0 keeps the
logic local and safe: planners produce dry-run specs, and materialization
helpers only write standard placeholder artifacts plus audit files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from ..adapter_store.lifecycle import AdapterArtifactFormat
from .backends import normalize_artifact_format, normalize_backend_name
from .export import (
    ExportPlan,
    export_contract_roles,
    export_artifact_name_for_format,
    export_directory_name_for_format,
    plan_export,
    required_export_format_for_backend,
)


def describe_llama_cpp_export_constraint(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None,
) -> str:
    """Explain the local constraint for llama.cpp export planning."""

    backend = normalize_backend_name(target_backend)
    source_artifact = normalize_artifact_format(source_artifact_format)
    required_format = required_export_format_for_backend(backend)
    if required_format is None:
        return f"{backend} does not require a merged GGUF export."
    if source_artifact == required_format:
        return (
            "llama.cpp is compatible with this merged GGUF artifact already; "
            "do not load LoRA safetensors directly on the llama.cpp path."
        )
    return (
        "llama.cpp targets must be exported from a LoRA adapter to merged GGUF first; "
        "do not load LoRA safetensors directly on the llama.cpp path."
    )


def resolve_export_output_dir(
    *,
    adapter_dir: str | Path | None,
    artifact_directory: str,
) -> str | None:
    """Resolve the standard export output directory without creating it."""

    if adapter_dir is None:
        return None
    return str(Path(adapter_dir).expanduser() / artifact_directory)


def _base_model_reference_kind(source_model_path: str | Path | None) -> str | None:
    if source_model_path is None:
        return None
    return "local_base_model" if _looks_like_local_path(source_model_path) else "base_model_id"


@dataclass(frozen=True)
class ExportRuntimeSpec:
    """Dry-run export specification for local execution."""

    target_backend: str
    source_artifact_format: str
    target_artifact_format: str
    required: bool
    dry_run: bool = True
    artifact_name: str = "adapter_model.safetensors"
    artifact_directory: str = "peft_lora"
    output_dir: str | None = None
    manifest_updates: dict[str, Any] = field(default_factory=dict)
    constraint: str = ""
    reason: str = ""
    backend_plan: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaterializedExportPlan:
    """Renderable export plan for display or later filesystem materialization."""

    target_backend: str
    source_artifact_format: str
    target_artifact_format: str
    required: bool
    dry_run: bool = True
    artifact_name: str = "adapter_model.safetensors"
    artifact_directory: str = "peft_lora"
    output_dir: str | None = None
    placeholder_files: list[str] = field(default_factory=list)
    manifest_patch: dict[str, Any] = field(default_factory=dict)
    manifest_patch_description: list[str] = field(default_factory=list)
    constraint: str = ""
    reason: str = ""
    backend_plan: dict[str, Any] = field(default_factory=dict)
    runtime_spec: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaterializedExportWriteResult:
    """Result of writing a materialized export plan to disk."""

    output_dir: str
    artifact_path: str
    export_plan_path: str
    notes_path: str
    written_files: list[str] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaCppExportToolResolution:
    """Outcome of parsing/checking a llama.cpp export tool path."""

    requested_path: str | None
    resolved_path: str | None
    exists: bool
    executable: bool
    source: str
    checked_paths: list[str] = field(default_factory=list)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaCppExportCommandPlan:
    """Command plan for a real merged GGUF export invocation."""

    tool_resolution: dict[str, Any]
    toolchain_status: str
    toolchain_validation: dict[str, Any]
    command: list[str]
    working_dir: str
    input_path: str
    output_dir: str
    output_artifact_path: str
    artifact_name: str
    target_artifact_format: str
    dry_run: bool = False
    audit: dict[str, Any] = field(default_factory=dict)
    command_metadata: dict[str, Any] = field(default_factory=dict)
    output_artifact_validation: dict[str, Any] = field(default_factory=dict)
    toolchain_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaCppExportExecutionResult:
    """Result of attempting to execute the real export command."""

    status: str
    outcome: str | None
    attempted: bool
    success: bool
    returncode: int | None
    exit_code: int | None
    command: list[str]
    working_dir: str
    tool_resolution: dict[str, Any]
    stdout: str = ""
    stderr: str = ""
    output_dir: str | None = None
    output_artifact_path: str | None = None
    failure_category: str | None = None
    failure_reason: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)
    command_metadata: dict[str, Any] = field(default_factory=dict)
    output_artifact_validation: dict[str, Any] = field(default_factory=dict)
    audit_summary: dict[str, Any] = field(default_factory=dict)
    toolchain_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaCppExportArtifactValidation:
    """Validation result for the output artifact path."""

    path: str | None
    exists: bool
    is_file: bool
    size_bytes: int | None
    valid: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plan_get(plan: Any, key: str, default: Any = None) -> Any:
    if isinstance(plan, Mapping):
        return plan.get(key, default)
    return getattr(plan, key, default)


def _classify_llama_cpp_export_failure(
    *,
    toolchain_status: str,
    attempted: bool,
    success: bool,
    returncode: int | None,
    output_artifact_validation: Mapping[str, Any] | Any,
    timeout: bool = False,
    execution_error: str | None = None,
    stderr_text: str | None = None,
) -> tuple[str, str | None]:
    """Classify export failures into stable machine-readable categories."""

    if toolchain_status in {"tool_missing", "tool_invalid"}:
        return toolchain_status, _plan_get(output_artifact_validation, "reason", None)
    if timeout:
        return "execution_timeout", execution_error or "llama.cpp export command timed out"
    if execution_error is not None:
        return "execution_error", execution_error
    if not attempted:
        return "planned", None
    if success and bool(_plan_get(output_artifact_validation, "valid", False)):
        return "success", None
    if returncode not in (None, 0):
        stderr_summary = _summarize_stderr_text(stderr_text)
        if stderr_summary:
            return "command_failed", f"llama.cpp export command exited with exit code {returncode}: {stderr_summary}"
        return "command_failed", f"llama.cpp export command exited with exit code {returncode}"
    artifact_reason = str(_plan_get(output_artifact_validation, "reason", "") or "").strip()
    if artifact_reason:
        lowered = artifact_reason.lower()
        if "empty" in lowered:
            return "artifact_empty", artifact_reason
        if "missing" in lowered:
            return "artifact_missing", artifact_reason
    if _plan_get(output_artifact_validation, "exists", False):
        return "artifact_invalid", artifact_reason or "output artifact failed validation"
    return "artifact_missing", artifact_reason or "output artifact is missing"


def _summarize_stderr_text(stderr_text: str | None) -> str | None:
    if not stderr_text:
        return None
    lines = [line.strip() for line in str(stderr_text).splitlines() if line.strip()]
    if not lines:
        return None
    for line in reversed(lines):
        lowered = line.lower()
        if lowered.startswith("traceback"):
            continue
        if "warning" in lowered and "error" not in lowered and "exception" not in lowered:
            continue
        if len(line) > 220:
            return line[:217] + "..."
        return line
    line = lines[-1]
    return line[:217] + "..." if len(line) > 220 else line


def validate_llama_cpp_export_toolchain(tool_resolution: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Classify the toolchain state for command planning and execution."""

    exists = bool(_plan_get(tool_resolution, "exists", False))
    executable = bool(_plan_get(tool_resolution, "executable", False))
    runnable_with_python = bool(_plan_get(tool_resolution, "metadata", {}).get("runnable_with_python", False))
    resolved_path = _plan_get(tool_resolution, "resolved_path", None)
    reason = _plan_get(tool_resolution, "reason", "")
    searched_env = list(_plan_get(tool_resolution, "metadata", {}).get("searched_env", []) or [])
    env_name = _plan_get(tool_resolution, "metadata", {}).get("env_name")
    if not exists:
        status = "tool_missing"
        allowed = False
        reason = reason or "llama.cpp export tool not found"
    elif not executable and not runnable_with_python:
        status = "tool_invalid"
        allowed = False
        reason = reason or "llama.cpp export tool exists but is not executable"
    else:
        status = "planned"
        allowed = True
        reason = reason or ("llama.cpp export tool can be launched with Python" if runnable_with_python else "llama.cpp export tool is ready")
    recommended_action = _llama_cpp_export_toolchain_recommended_action(
        status=status,
        resolved_path=resolved_path,
        runnable_with_python=runnable_with_python,
        exists=exists,
        executable=executable,
        searched_env=searched_env,
    )
    return {
        "status": status,
        "allowed": allowed,
        "ready": allowed,
        "readiness_state": "ready" if allowed else "blocked",
        "exists": exists,
        "executable": executable,
        "runnable_with_python": runnable_with_python,
        "resolved_path": resolved_path,
        "reason": reason,
        "checked_paths": list(_plan_get(tool_resolution, "checked_paths", []) or []),
        "source": _plan_get(tool_resolution, "source", ""),
        "searched_env": searched_env,
        "env_name": env_name,
        "recommended_action": recommended_action,
    }


def _llama_cpp_export_toolchain_recommended_action(
    *,
    status: str,
    resolved_path: str | None,
    runnable_with_python: bool,
    exists: bool,
    executable: bool,
    searched_env: list[str],
) -> str:
    if status == "planned":
        if runnable_with_python:
            return (
                f"Invoke the resolved Python converter with {sys.executable}; "
                f"resolved_path={resolved_path}"
            )
        return "No action needed; the llama.cpp export tool is ready."
    if status == "tool_missing":
        env_hint = ", ".join(searched_env) if searched_env else "PFE_LLAMA_CPP_EXPORT_TOOL"
        return f"Set {env_hint} or pass tool_path explicitly to point at the llama.cpp export tool."
    if status == "tool_invalid":
        if exists and not executable:
            return (
                "Make the resolved export tool executable, or point tool_path at a runnable "
                ".py converter / PATH command."
            )
        return "Point tool_path at a runnable llama.cpp export converter or expose it on PATH."
    return "Inspect the toolchain probe output for the next actionable step."


def validate_llama_cpp_export_output_artifact(
    output_artifact_path: str | Path | None,
    *,
    required: bool = False,
) -> LlamaCppExportArtifactValidation:
    """Validate the output artifact path for the runner audit."""

    if output_artifact_path is None:
        reason = "output artifact path is missing"
        return LlamaCppExportArtifactValidation(
            path=None,
            exists=False,
            is_file=False,
            size_bytes=None,
            valid=not required,
            reason=reason,
            metadata={"required": required},
        )

    path = Path(output_artifact_path).expanduser()
    exists = path.exists()
    is_file = path.is_file()
    size_bytes = path.stat().st_size if exists and is_file else None
    if exists and is_file and size_bytes == 0:
        reason = "output artifact is empty"
        valid = False
    elif exists and is_file:
        reason = "output artifact exists"
        valid = True
    elif required:
        reason = "required output artifact is missing"
        valid = False
    else:
        reason = "output artifact is not present yet"
        valid = False
    return LlamaCppExportArtifactValidation(
        path=str(path),
        exists=exists,
        is_file=is_file,
        size_bytes=size_bytes,
        valid=valid,
        reason=reason,
        metadata={"required": required},
    )


def summarize_llama_cpp_export_audit(
    *,
    toolchain_status: str,
    execution_status: str,
    outcome: str | None,
    output_artifact_validation: Mapping[str, Any] | Any,
    command_metadata: Mapping[str, Any] | Any = None,
    failure_category: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Produce a stable audit summary for mainline consumption."""

    artifact_status = _plan_get(output_artifact_validation, "valid", False)
    artifact_reason = _plan_get(output_artifact_validation, "reason", "")
    summary = execution_status if execution_status in {
        "planned",
        "executed",
        "tool_missing",
        "tool_invalid",
        "command_failed",
        "artifact_missing",
        "artifact_empty",
        "artifact_invalid",
        "execution_timeout",
        "execution_error",
    } else "planned"
    return {
        "summary": summary,
        "toolchain_status": toolchain_status,
        "execution_status": execution_status,
        "outcome": outcome,
        "failure_category": failure_category or summary,
        "failure_reason": failure_reason,
        "artifact_valid": bool(artifact_status),
        "artifact_reason": artifact_reason,
        "command_metadata": dict(command_metadata or {}),
    }


def summarize_llama_cpp_export_toolchain(
    *,
    toolchain_status: str,
    toolchain_validation: Mapping[str, Any] | Any,
    output_artifact_validation: Mapping[str, Any] | Any,
    command_metadata: Mapping[str, Any] | Any = None,
    execution_status: str | None = None,
    outcome: str | None = None,
    failure_category: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Build a stable toolchain summary for mainline consumption."""

    validation = dict(toolchain_validation or {})
    artifact = dict(output_artifact_validation or {})
    command_metadata_dict = dict(command_metadata or {})
    execution_status = execution_status or toolchain_status
    ready = bool(validation.get("allowed", False))
    probe_summary = {
        "status": validation.get("status", toolchain_status),
        "ready": ready,
        "readiness_state": validation.get("readiness_state", "ready" if ready else "blocked"),
        "source": validation.get("source", ""),
        "resolved_path": validation.get("resolved_path"),
        "checked_paths": list(validation.get("checked_paths", []) or []),
        "searched_env": list(validation.get("searched_env", []) or []),
        "env_name": validation.get("env_name"),
        "exists": bool(validation.get("exists", False)),
        "executable": bool(validation.get("executable", False)),
        "runnable_with_python": bool(validation.get("runnable_with_python", False)),
        "reason": validation.get("reason", ""),
        "recommended_action": validation.get("recommended_action", ""),
    }
    audit_summary = summarize_llama_cpp_export_audit(
        toolchain_status=toolchain_status,
        execution_status=execution_status,
        outcome=outcome,
        output_artifact_validation=artifact,
        command_metadata=command_metadata_dict,
        failure_category=failure_category,
        failure_reason=failure_reason,
    )
    artifact_contract = command_metadata_dict.get("artifact_contract")
    return {
        "summary": audit_summary["summary"],
        "status": toolchain_status,
        "toolchain_status": toolchain_status,
        "toolchain_reason": validation.get("reason", ""),
        "toolchain_allowed": bool(validation.get("allowed", False)),
        "toolchain_ready": ready,
        "toolchain_readiness_state": probe_summary["readiness_state"],
        "toolchain_exists": bool(validation.get("exists", False)),
        "toolchain_executable": bool(validation.get("executable", False)),
        "toolchain_resolved_path": validation.get("resolved_path"),
        "toolchain_checked_paths": list(validation.get("checked_paths", []) or []),
        "toolchain_searched_env": list(validation.get("searched_env", []) or []),
        "output_artifact_path": artifact.get("path"),
        "output_artifact_valid": bool(artifact.get("valid", False)),
        "output_artifact_exists": bool(artifact.get("exists", False)),
        "output_artifact_is_file": bool(artifact.get("is_file", False)),
        "output_artifact_size_bytes": artifact.get("size_bytes"),
        "output_artifact_reason": artifact.get("reason", ""),
        "execution_status": execution_status,
        "outcome": outcome,
        "failure_category": audit_summary["failure_category"],
        "failure_reason": failure_reason or audit_summary["failure_reason"],
        "audit_summary": audit_summary,
        "probe_summary": probe_summary,
        "artifact_contract": artifact_contract,
        "source_artifact_role": command_metadata_dict.get("source_artifact_role"),
        "target_artifact_role": command_metadata_dict.get("target_artifact_role"),
        "base_model_role": command_metadata_dict.get("base_model_role", "base_model"),
        "recommended_action": validation.get("recommended_action", ""),
        "command_metadata": command_metadata_dict,
        "toolchain_validation": validation,
        "output_artifact_validation": artifact,
    }


def probe_llama_cpp_export_toolchain(
    *,
    tool_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Probe the llama.cpp export toolchain without requiring a model path."""

    tool_resolution = resolve_llama_cpp_export_tool_path(tool_path, env=env)
    toolchain_validation = validate_llama_cpp_export_toolchain(tool_resolution)
    return {
        "summary": toolchain_validation["status"],
        "status": toolchain_validation["status"],
        "ready": toolchain_validation["ready"],
        "readiness_state": toolchain_validation["readiness_state"],
        "recommended_action": toolchain_validation["recommended_action"],
        "reason": toolchain_validation["reason"],
        "source": toolchain_validation["source"],
        "resolved_path": toolchain_validation["resolved_path"],
        "checked_paths": toolchain_validation["checked_paths"],
        "searched_env": toolchain_validation["searched_env"],
        "env_name": toolchain_validation["env_name"],
        "exists": toolchain_validation["exists"],
        "executable": toolchain_validation["executable"],
        "runnable_with_python": toolchain_validation["runnable_with_python"],
        "tool_resolution": tool_resolution.to_dict(),
        "toolchain_validation": toolchain_validation,
    }


def _manifest_patch_description(manifest_updates: Mapping[str, Any]) -> list[str]:
    description = [
        "update adapter_manifest.json with export planning metadata",
        "preserve provenance fields: workspace, adapter_dir, source_adapter_version, source_model, training_run_id, num_samples",
    ]
    if "artifact_format" in manifest_updates:
        description.append(f"set artifact_format={manifest_updates['artifact_format']}")
    if "artifact_name" in manifest_updates:
        description.append(f"set artifact_name={manifest_updates['artifact_name']}")
    if "inference_backend" in manifest_updates:
        description.append(f"set inference_backend={manifest_updates['inference_backend']}")
    if manifest_updates.get("requires_export"):
        description.append("mark requires_export=true so the caller knows a conversion was needed")
    export_payload = manifest_updates.get("export", {})
    if isinstance(export_payload, Mapping):
        if "target_backend" in export_payload:
            description.append(f"record export.target_backend={export_payload['target_backend']}")
        if "target_artifact_format" in export_payload:
            description.append(
                f"record export.target_artifact_format={export_payload['target_artifact_format']}"
            )
        if "export_directory" in export_payload:
            description.append(f"record export.export_directory={export_payload['export_directory']}")
        if "source_artifact_role" in export_payload:
            description.append(f"record export.source_artifact_role={export_payload['source_artifact_role']}")
        if "target_artifact_role" in export_payload:
            description.append(f"record export.target_artifact_role={export_payload['target_artifact_role']}")
        if "base_model_role" in export_payload:
            description.append(f"record export.base_model_role={export_payload['base_model_role']}")
    return description


def _placeholder_files(artifact_name: str) -> list[str]:
    return [
        "adapter_manifest.json",
        artifact_name,
        "export_plan.json",
        "EXPORT_NOTES.txt",
    ]


def _export_tool_env_candidates() -> list[str]:
    return [
        "PFE_LLAMA_CPP_EXPORT_TOOL",
        "LLAMA_CPP_EXPORT_TOOL",
        "LLAMA_CPP_PATH",
        "LLAMA_CPP_BIN",
    ]


def _llama_cpp_export_tool_names() -> list[str]:
    return [
        "convert_lora_to_gguf.py",
        "convert_hf_to_gguf.py",
        "convert-hf-to-gguf.py",
        "llama-export-lora",
        "llama_export_lora",
    ]


def _llama_cpp_converter_kind(tool_path: str | Path | None) -> str:
    if tool_path is None:
        return "unknown"
    name = Path(str(tool_path)).name.lower()
    if "convert_lora_to_gguf" in name:
        return "lora"
    if "convert_hf_to_gguf" in name or "convert-hf-to-gguf" in name:
        return "hf"
    return "unknown"


def _llama_cpp_outtype(
    *,
    converter_kind: str,
    env: Mapping[str, str] | None = None,
) -> str:
    env_mapping = dict(os.environ if env is None else env)
    value = str(env_mapping.get("PFE_LLAMA_CPP_OUTTYPE", "auto")).strip().lower()
    if value in {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}:
        return value if value != "auto" or converter_kind != "lora" else "f16"
    return "f16" if converter_kind == "lora" else "auto"


def _looks_like_local_path(path: str | Path | None) -> bool:
    if path is None:
        return False
    return Path(str(path)).expanduser().exists()


def _should_add_no_lazy(
    *,
    converter_kind: str,
    source_model_path: str | Path | None,
    extra_args: list[str] | None,
    env: Mapping[str, str] | None = None,
) -> bool:
    del source_model_path  # reserved for future per-model compatibility rules
    if converter_kind != "lora":
        return False
    if extra_args and "--no-lazy" in extra_args:
        return False
    env_mapping = dict(os.environ if env is None else env)
    configured = str(env_mapping.get("PFE_LLAMA_CPP_NO_LAZY", "auto")).strip().lower()
    if configured in {"1", "true", "yes", "on"}:
        return True
    return False


def _tool_resolution_payload(
    *,
    requested_path: str | None,
    resolved_path: str | None,
    source: str,
    checked_paths: list[str],
    reason: str,
    searched_env: list[str] | None = None,
    env_name: str | None = None,
) -> LlamaCppExportToolResolution:
    path = Path(resolved_path).expanduser() if resolved_path is not None else None
    exists = bool(path is not None and path.exists())
    is_python_script = bool(path is not None and path.is_file() and path.suffix == ".py")
    executable = bool(path is not None and path.is_file() and os.access(path, os.X_OK))
    return LlamaCppExportToolResolution(
        requested_path=requested_path,
        resolved_path=str(path) if path is not None else None,
        exists=exists,
        executable=executable,
        source=source,
        checked_paths=checked_paths,
        reason=reason,
        metadata={
            "from_env": bool(env_name),
            "env_name": env_name,
            "is_python_script": is_python_script,
            "runnable_with_python": bool(exists and is_python_script),
            "searched_env": list(searched_env or []),
        },
    )


def resolve_llama_cpp_export_tool_path(
    tool_path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> LlamaCppExportToolResolution:
    """Parse and check the llama.cpp export tool path.

    The helper accepts an explicit path first, then common environment
    variables, and finally falls back to ``shutil.which`` for a command name.
    """

    env_mapping = dict(os.environ if env is None else env)
    checked_paths: list[str] = []
    searched_env = _export_tool_env_candidates()
    requested_path = str(tool_path) if tool_path is not None else None

    candidate_values: list[tuple[str, str | None]] = []
    if tool_path is not None:
        candidate_values.append((str(tool_path), None))
    for name in searched_env:
        value = env_mapping.get(name)
        if value:
            candidate_values.append((value, name))

    for candidate, env_name in candidate_values:
        candidate_path = Path(candidate).expanduser()
        checked_paths.append(str(candidate_path))
        if candidate_path.is_dir():
            for tool_name in _llama_cpp_export_tool_names():
                nested = candidate_path / tool_name
                checked_paths.append(str(nested))
                if nested.exists():
                    return _tool_resolution_payload(
                        requested_path=requested_path,
                        resolved_path=str(nested),
                        source="explicit_dir" if tool_path is not None and str(tool_path) == candidate else "env_dir",
                        checked_paths=checked_paths,
                        reason="resolved llama.cpp export tool inside configured directory",
                        searched_env=searched_env,
                        env_name=env_name,
                    )
            continue
        if candidate_path.exists():
            return _tool_resolution_payload(
                requested_path=requested_path,
                resolved_path=str(candidate_path),
                source="explicit" if tool_path is not None and str(tool_path) == candidate else "env",
                checked_paths=checked_paths,
                reason="resolved explicit tool path" if os.access(candidate_path, os.X_OK) or candidate_path.suffix == ".py" else "tool path exists but is not directly runnable",
                searched_env=searched_env,
                env_name=env_name,
            )
        found = shutil.which(candidate)
        if found:
            checked_paths.append(found)
            return _tool_resolution_payload(
                requested_path=requested_path,
                resolved_path=found,
                source="which",
                checked_paths=checked_paths,
                reason="resolved via PATH lookup",
                searched_env=searched_env,
                env_name=env_name,
            )

    if not candidate_values:
        workspace_tool_dir = Path.cwd() / "tools" / "llama.cpp"
        checked_paths.append(str(workspace_tool_dir))
        if workspace_tool_dir.is_dir():
            for tool_name in _llama_cpp_export_tool_names():
                nested = workspace_tool_dir / tool_name
                checked_paths.append(str(nested))
                if nested.exists():
                    return _tool_resolution_payload(
                        requested_path=requested_path,
                        resolved_path=str(nested),
                        source="workspace_dir",
                        checked_paths=checked_paths,
                        reason="resolved llama.cpp export tool from workspace tools/llama.cpp",
                        searched_env=searched_env,
                        env_name=None,
                    )

    return LlamaCppExportToolResolution(
        requested_path=requested_path,
        resolved_path=None,
        exists=False,
        executable=False,
        source="missing",
        checked_paths=checked_paths,
        reason="llama.cpp export tool not found; set PFE_LLAMA_CPP_EXPORT_TOOL or pass tool_path explicitly",
        metadata={"searched_env": searched_env},
    )


def build_llama_cpp_export_command_plan(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None,
    adapter_dir: str | Path,
    tool_path: str | Path | None = None,
    source_model_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> LlamaCppExportCommandPlan:
    """Generate a real export command plan for the llama.cpp path."""

    plan = materialize_export_plan(
        target_backend=target_backend,
        source_artifact_format=source_artifact_format,
        adapter_dir=adapter_dir,
    )
    tool_resolution = resolve_llama_cpp_export_tool_path(tool_path, env=env)
    toolchain_validation = validate_llama_cpp_export_toolchain(tool_resolution)
    input_path = str(Path(adapter_dir).expanduser())
    output_dir = plan.output_dir or resolve_export_output_dir(adapter_dir=adapter_dir, artifact_directory=plan.artifact_directory)
    if output_dir is None:
        raise ValueError("adapter_dir is required to build a llama.cpp export command")
    if normalize_backend_name(target_backend) == "llama_cpp" and plan.target_artifact_format != AdapterArtifactFormat.GGUF_MERGED.value:
        raise ValueError("llama.cpp export commands must target gguf_merged artifacts")
    output_artifact_path = str(Path(output_dir) / plan.artifact_name)

    resolved_command = tool_resolution.resolved_path or str(tool_path or "llama_cpp_export_tool")
    converter_kind = _llama_cpp_converter_kind(resolved_command)
    outtype = _llama_cpp_outtype(converter_kind=converter_kind, env=env)
    add_no_lazy = _should_add_no_lazy(
        converter_kind=converter_kind,
        source_model_path=source_model_path,
        extra_args=extra_args,
        env=env,
    )
    base_model_reference_kind = _base_model_reference_kind(source_model_path)
    artifact_contract = export_contract_roles(
        source_artifact_format=plan.source_artifact_format,
        target_artifact_format=plan.target_artifact_format,
    )
    command: list[str] = []
    if str(resolved_command).endswith(".py"):
        command.extend([sys.executable, str(resolved_command)])
    else:
        command.append(str(resolved_command))
    if converter_kind == "lora":
        command.extend(["--outfile", output_artifact_path, "--outtype", outtype])
        if source_model_path is not None:
            if _looks_like_local_path(source_model_path):
                command.extend(["--base", str(Path(source_model_path).expanduser())])
            else:
                command.extend(["--base-model-id", str(source_model_path)])
    elif converter_kind == "hf":
        command.extend(["--outfile", output_artifact_path, "--outtype", outtype])
    else:
        command.extend(
            [
                "--input",
                str(Path(source_model_path or adapter_dir).expanduser()),
                "--output-dir",
                output_dir,
                "--output-name",
                plan.artifact_name,
                "--target-format",
                plan.target_artifact_format,
            ]
        )
    if add_no_lazy:
        command.append("--no-lazy")
    if extra_args:
        command.extend(extra_args)
    if converter_kind == "lora":
        command.append(str(Path(adapter_dir).expanduser()))
    elif converter_kind == "hf":
        model_ref = str(Path(source_model_path).expanduser()) if _looks_like_local_path(source_model_path) else str(source_model_path or adapter_dir)
        command.append(model_ref)

    audit = {
        "status": toolchain_validation["status"],
        "tool_available": toolchain_validation["allowed"],
        "tool_reason": toolchain_validation["reason"],
        "required_export": plan.required,
        "constraint": plan.constraint,
        "materialized_plan": plan.to_dict(),
        "placeholder_only": False,
        "execution_mode": "planned",
    }
    artifact_validation = validate_llama_cpp_export_output_artifact(output_artifact_path, required=False).to_dict()
    toolchain_summary = summarize_llama_cpp_export_toolchain(
        toolchain_status=toolchain_validation["status"],
        toolchain_validation=toolchain_validation,
        output_artifact_validation=artifact_validation,
        command_metadata={
            "planned": True,
            "execution_state": "planned",
            "toolchain_status": toolchain_validation["status"],
            "target_backend": normalize_backend_name(target_backend),
            "output_artifact_name": plan.artifact_name,
            "converter_kind": converter_kind,
            "outtype": outtype,
            "no_lazy": add_no_lazy or bool(extra_args and "--no-lazy" in extra_args),
            "source_artifact_role": artifact_contract["source_artifact_role"],
            "target_artifact_role": artifact_contract["target_artifact_role"],
            "base_model_role": artifact_contract["base_model_role"],
            "artifact_contract": artifact_contract,
            "base_model_reference_kind": base_model_reference_kind,
        },
        execution_status="planned",
        outcome="planned",
    )
    return LlamaCppExportCommandPlan(
        tool_resolution=tool_resolution.to_dict(),
        toolchain_status=str(toolchain_validation["status"]),
        toolchain_validation=toolchain_validation,
        command=command,
        working_dir=str(Path(adapter_dir).expanduser()),
        input_path=input_path,
        output_dir=str(output_dir),
        output_artifact_path=output_artifact_path,
        artifact_name=plan.artifact_name,
        target_artifact_format=plan.target_artifact_format,
        audit=audit,
        command_metadata={
            "planned": True,
            "execution_state": "planned",
            "toolchain_status": toolchain_validation["status"],
            "target_backend": normalize_backend_name(target_backend),
            "output_artifact_name": plan.artifact_name,
            "converter_kind": converter_kind,
            "outtype": outtype,
            "no_lazy": add_no_lazy or bool(extra_args and "--no-lazy" in extra_args),
            "source_artifact_role": artifact_contract["source_artifact_role"],
            "target_artifact_role": artifact_contract["target_artifact_role"],
            "base_model_role": artifact_contract["base_model_role"],
            "artifact_contract": artifact_contract,
            "base_model_reference_kind": base_model_reference_kind,
        },
        output_artifact_validation=artifact_validation,
        toolchain_summary=toolchain_summary,
        metadata={
            "backend": normalize_backend_name(target_backend),
            "source_artifact_format": normalize_artifact_format(source_artifact_format),
            "extra_args_count": len(extra_args or []),
            "converter_kind": converter_kind,
            "base_model_reference": str(source_model_path) if source_model_path is not None else None,
            "base_model_reference_kind": base_model_reference_kind,
            "source_artifact_role": artifact_contract["source_artifact_role"],
            "target_artifact_role": artifact_contract["target_artifact_role"],
            "base_model_role": artifact_contract["base_model_role"],
            "artifact_contract": artifact_contract,
            "no_lazy": add_no_lazy or bool(extra_args and "--no-lazy" in extra_args),
        },
    )


def run_export_command_plan(
    plan: Any,
    *,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> LlamaCppExportExecutionResult:
    """Run a command plan with consistent audit semantics.

    This helper stays backend-agnostic enough for the mainline to reuse while
    remaining dependency-free in Phase 0.
    """

    tool_resolution = _plan_get(plan, "tool_resolution", {}) or {}
    command = list(_plan_get(plan, "command", []))
    working_dir = str(_plan_get(plan, "working_dir", "."))
    output_dir = _plan_get(plan, "output_dir")
    output_artifact_path = _plan_get(plan, "output_artifact_path")
    audit = dict(_plan_get(plan, "audit", {}) or {})
    toolchain_validation = validate_llama_cpp_export_toolchain(tool_resolution)
    metadata = dict(_plan_get(plan, "metadata", {}) or {})
    command_metadata = dict(_plan_get(plan, "command_metadata", {}) or {})
    output_artifact_validation = validate_llama_cpp_export_output_artifact(
        output_artifact_path,
        required=bool(_plan_get(plan, "target_artifact_format", "") == AdapterArtifactFormat.GGUF_MERGED.value),
    ).to_dict()

    if toolchain_validation["status"] == "tool_missing":
        failure_category, failure_reason = _classify_llama_cpp_export_failure(
            toolchain_status="tool_missing",
            attempted=False,
            success=False,
            returncode=None,
            output_artifact_validation=output_artifact_validation,
        )
        failure_reason = f"{toolchain_validation['reason']} (missing or not executable)"
        audit_summary = summarize_llama_cpp_export_audit(
            toolchain_status="tool_missing",
            execution_status="tool_missing",
            outcome=None,
            output_artifact_validation=output_artifact_validation,
            command_metadata=command_metadata,
            failure_category=failure_category,
            failure_reason=failure_reason,
        )
        return LlamaCppExportExecutionResult(
            status="tool_missing",
            outcome=None,
            attempted=False,
            success=False,
            returncode=None,
            exit_code=None,
            command=command,
            working_dir=working_dir,
            tool_resolution=tool_resolution,
            output_dir=output_dir,
            output_artifact_path=output_artifact_path,
            failure_category=failure_category,
            failure_reason=failure_reason,
            audit={
                **audit,
                "status": "tool_missing",
                "failure_reason": failure_reason,
                "failure_category": failure_category,
            },
            command_metadata={**command_metadata, "execution_state": "blocked"},
            output_artifact_validation=output_artifact_validation,
            audit_summary=audit_summary,
            toolchain_summary=summarize_llama_cpp_export_toolchain(
                toolchain_status="tool_missing",
                toolchain_validation=toolchain_validation,
                output_artifact_validation=output_artifact_validation,
                command_metadata={**command_metadata, "execution_state": "blocked"},
                execution_status="tool_missing",
                outcome=None,
                failure_category=failure_category,
                failure_reason=failure_reason,
            ),
            metadata={**metadata, "execution_mode": "skipped", "failure_category": failure_category},
        )
    if toolchain_validation["status"] == "tool_invalid":
        failure_category, failure_reason = _classify_llama_cpp_export_failure(
            toolchain_status="tool_invalid",
            attempted=False,
            success=False,
            returncode=None,
            output_artifact_validation=output_artifact_validation,
        )
        failure_reason = f"{toolchain_validation['reason']} (missing or not executable)"
        audit_summary = summarize_llama_cpp_export_audit(
            toolchain_status="tool_invalid",
            execution_status="tool_invalid",
            outcome=None,
            output_artifact_validation=output_artifact_validation,
            command_metadata=command_metadata,
            failure_category=failure_category,
            failure_reason=failure_reason,
        )
        return LlamaCppExportExecutionResult(
            status="tool_invalid",
            outcome=None,
            attempted=False,
            success=False,
            returncode=None,
            exit_code=None,
            command=command,
            working_dir=working_dir,
            tool_resolution=tool_resolution,
            output_dir=output_dir,
            output_artifact_path=output_artifact_path,
            failure_category=failure_category,
            failure_reason=failure_reason,
            audit={
                **audit,
                "status": "tool_invalid",
                "failure_reason": failure_reason,
                "failure_category": failure_category,
            },
            command_metadata={**command_metadata, "execution_state": "blocked"},
            output_artifact_validation=output_artifact_validation,
            audit_summary=audit_summary,
            toolchain_summary=summarize_llama_cpp_export_toolchain(
                toolchain_status="tool_invalid",
                toolchain_validation=toolchain_validation,
                output_artifact_validation=output_artifact_validation,
                command_metadata={**command_metadata, "execution_state": "blocked"},
                execution_status="tool_invalid",
                outcome=None,
                failure_category=failure_category,
                failure_reason=failure_reason,
            ),
            metadata={**metadata, "execution_mode": "skipped", "failure_category": failure_category},
        )

    if dry_run:
        failure_category, failure_reason = _classify_llama_cpp_export_failure(
            toolchain_status=toolchain_validation["status"],
            attempted=False,
            success=True,
            returncode=None,
            output_artifact_validation=output_artifact_validation,
        )
        audit_summary = summarize_llama_cpp_export_audit(
            toolchain_status=toolchain_validation["status"],
            execution_status="planned",
            outcome="planned",
            output_artifact_validation=output_artifact_validation,
            command_metadata=command_metadata,
            failure_category=failure_category,
            failure_reason=failure_reason,
        )
        return LlamaCppExportExecutionResult(
            status="planned",
            outcome="planned",
            attempted=False,
            success=True,
            returncode=None,
            exit_code=None,
            command=command,
            working_dir=working_dir,
            tool_resolution=tool_resolution,
            output_dir=output_dir,
            output_artifact_path=output_artifact_path,
            failure_category=failure_category,
            failure_reason=failure_reason,
            audit={
                **audit,
                "status": "planned",
                "stdout_length": 0,
                "stderr_length": 0,
                "exit_code": None,
                "failure_category": failure_category,
            },
            command_metadata={**command_metadata, "execution_state": "planned"},
            output_artifact_validation=output_artifact_validation,
            audit_summary=audit_summary,
            toolchain_summary=summarize_llama_cpp_export_toolchain(
                toolchain_status=toolchain_validation["status"],
                toolchain_validation=toolchain_validation,
                output_artifact_validation=output_artifact_validation,
                command_metadata={**command_metadata, "execution_state": "planned"},
                execution_status="planned",
                outcome="planned",
                failure_category=failure_category,
                failure_reason=failure_reason,
            ),
            metadata={**metadata, "execution_mode": "dry_run", "failure_category": failure_category},
        )

    if output_dir is not None:
        Path(str(output_dir)).expanduser().mkdir(parents=True, exist_ok=True)

    try:
        completed = subprocess.run(
            command,
            cwd=working_dir,
            env=dict(os.environ if env is None else env),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        execution_error = None
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        completed = exc
        execution_error = f"{exc.__class__.__name__}: {exc}"
        timed_out = True
    except Exception as exc:
        completed = None
        execution_error = f"{exc.__class__.__name__}: {exc}"
        timed_out = False

    stdout = getattr(completed, "stdout", "") or ""
    stderr = getattr(completed, "stderr", "") or ""
    returncode = getattr(completed, "returncode", None) if completed is not None else None
    output_artifact_validation = validate_llama_cpp_export_output_artifact(
        output_artifact_path,
        required=True,
    ).to_dict()
    success = bool(returncode == 0 and not timed_out and execution_error is None)
    failure_category, failure_reason = _classify_llama_cpp_export_failure(
        toolchain_status=toolchain_validation["status"],
        attempted=True,
        success=success,
        returncode=returncode,
        output_artifact_validation=output_artifact_validation,
        timeout=timed_out,
        execution_error=execution_error,
        stderr_text=stderr,
    )
    execution_status = "executed" if success else failure_category
    outcome = "success" if success else ("timeout" if timed_out else "failed")
    audit_summary = summarize_llama_cpp_export_audit(
        toolchain_status=toolchain_validation["status"],
        execution_status=execution_status,
        outcome=outcome,
        output_artifact_validation=output_artifact_validation,
        command_metadata=command_metadata,
        failure_category=failure_category,
        failure_reason=failure_reason,
    )
    toolchain_summary = summarize_llama_cpp_export_toolchain(
        toolchain_status=toolchain_validation["status"],
        toolchain_validation=toolchain_validation,
        output_artifact_validation=output_artifact_validation,
        command_metadata={**command_metadata, "execution_state": "executed"},
        execution_status=execution_status,
        outcome=outcome,
        failure_category=failure_category,
        failure_reason=failure_reason,
    )
    return LlamaCppExportExecutionResult(
        status=execution_status,
        outcome=outcome,
        attempted=True,
        success=success,
        returncode=returncode,
        exit_code=returncode,
        command=command,
        working_dir=working_dir,
        tool_resolution=tool_resolution,
        stdout=stdout,
        stderr=stderr,
        output_dir=output_dir,
        output_artifact_path=output_artifact_path,
        failure_category=failure_category,
        failure_reason=failure_reason,
        audit={
            **audit,
            "status": execution_status,
            "outcome": outcome,
            "stdout_length": len(stdout),
            "stderr_length": len(stderr),
            "exit_code": returncode,
            "failure_category": failure_category,
            "failure_reason": failure_reason,
        },
        command_metadata={**command_metadata, "execution_state": "executed"},
        output_artifact_validation=output_artifact_validation,
        audit_summary=audit_summary,
        toolchain_summary=toolchain_summary,
        metadata={**metadata, "execution_mode": "subprocess", "failure_category": failure_category},
    )


def execute_llama_cpp_export_command(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None,
    adapter_dir: str | Path,
    tool_path: str | Path | None = None,
    source_model_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
    timeout_seconds: float | None = None,
) -> LlamaCppExportExecutionResult:
    """Execute the llama.cpp export command when the tool is available.

    If the tool is missing or not executable, the helper returns a structured
    failure result with audit information instead of raising.
    """

    plan = build_llama_cpp_export_command_plan(
        target_backend=target_backend,
        source_artifact_format=source_artifact_format,
        adapter_dir=adapter_dir,
        tool_path=tool_path,
        source_model_path=source_model_path,
        env=env,
        extra_args=extra_args,
    )
    return run_export_command_plan(
        plan,
        dry_run=dry_run,
        env=env,
        timeout_seconds=timeout_seconds,
    )


def _render_export_notes(plan: MaterializedExportPlan, artifact_path: Path) -> str:
    artifact_contract = plan.runtime_spec.get("metadata", {}).get("artifact_contract", {}) if isinstance(plan.runtime_spec, Mapping) else {}
    lines = [
        "PFE export materialization (dry-run + local placeholder write)",
        f"target_backend: {plan.target_backend}",
        f"source_artifact_format: {plan.source_artifact_format}",
        f"target_artifact_format: {plan.target_artifact_format}",
        f"source_artifact_role: {artifact_contract.get('source_artifact_role', 'unknown')}",
        f"target_artifact_role: {artifact_contract.get('target_artifact_role', 'unknown')}",
        f"base_model_role: {artifact_contract.get('base_model_role', 'base_model')}",
        f"artifact_directory: {plan.artifact_directory}",
        f"artifact_name: {plan.artifact_name}",
        f"output_dir: {plan.output_dir or artifact_path.parent}",
        f"artifact_path: {artifact_path}",
        f"required_export: {plan.required}",
        f"constraint: {plan.constraint}",
        f"reason: {plan.reason}",
        "",
        "manifest patch:",
    ]
    lines.extend(f"- {item}" for item in plan.manifest_patch_description)
    lines.extend(
        [
            "",
            "placeholders:",
        ]
    )
    lines.extend(f"- {item}" for item in plan.placeholder_files)
    return "\n".join(lines) + "\n"


def _is_placeholder_export_artifact(path: Path) -> bool:
    try:
        prefix = path.read_bytes()[:128]
    except Exception:
        return False
    return b"PFE placeholder artifact" in prefix


def build_export_runtime_spec(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None = None,
    adapter_dir: str | Path | None = None,
    workspace: str | None = None,
    source_adapter_version: str | None = None,
    source_model: str | None = None,
    training_run_id: str | None = None,
    num_samples: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> ExportRuntimeSpec:
    """Build a dry-run export spec that mirrors the intended runtime layout."""

    plan: ExportPlan = plan_export(
        target_backend=target_backend,
        source_artifact_format=source_artifact_format,
        workspace=workspace,
        adapter_dir=str(adapter_dir) if adapter_dir is not None else None,
        source_adapter_version=source_adapter_version,
        source_model=source_model,
        training_run_id=training_run_id,
        num_samples=num_samples,
        extra_metadata=extra_metadata,
    )
    output_dir = resolve_export_output_dir(adapter_dir=adapter_dir, artifact_directory=plan.artifact_directory)
    return ExportRuntimeSpec(
        target_backend=normalize_backend_name(target_backend),
        source_artifact_format=normalize_artifact_format(source_artifact_format),
        target_artifact_format=plan.target_artifact_format,
        required=plan.required,
        artifact_name=plan.artifact_name,
        artifact_directory=plan.artifact_directory,
        output_dir=output_dir,
        manifest_updates=plan.manifest_updates,
        constraint=describe_llama_cpp_export_constraint(
            target_backend=target_backend,
            source_artifact_format=source_artifact_format,
        ),
        reason=plan.reason,
        backend_plan=plan.to_dict(),
        metadata={
            "dry_run": True,
            "output_dir_resolved": output_dir is not None,
            "export_directory_name": export_directory_name_for_format(plan.target_artifact_format),
            "artifact_name": export_artifact_name_for_format(plan.target_artifact_format),
            "source_artifact_role": plan.metadata.get("source_artifact_role"),
            "target_artifact_role": plan.metadata.get("target_artifact_role"),
            "base_model_role": plan.metadata.get("base_model_role", "base_model"),
            "artifact_contract": plan.metadata.get("artifact_contract"),
        },
    )


def dry_run_export_spec(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper returning a serializable dry-run export spec."""

    return build_export_runtime_spec(
        target_backend=target_backend,
        source_artifact_format=source_artifact_format,
        **kwargs,
    ).to_dict()


def materialize_export_plan(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None = None,
    adapter_dir: str | Path | None = None,
    workspace: str | None = None,
    source_adapter_version: str | None = None,
    source_model: str | None = None,
    training_run_id: str | None = None,
    num_samples: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> MaterializedExportPlan:
    """Render the export plan into a materialization-friendly description.

    This does not write files. It only expands the plan into:
    - standard output directory
    - placeholder file list
    - manifest patch description
    - dry-run spec payload for display or downstream persistence
    """

    runtime_spec = build_export_runtime_spec(
        target_backend=target_backend,
        source_artifact_format=source_artifact_format,
        adapter_dir=adapter_dir,
        workspace=workspace,
        source_adapter_version=source_adapter_version,
        source_model=source_model,
        training_run_id=training_run_id,
        num_samples=num_samples,
        extra_metadata=extra_metadata,
    )
    return MaterializedExportPlan(
        target_backend=runtime_spec.target_backend,
        source_artifact_format=runtime_spec.source_artifact_format,
        target_artifact_format=runtime_spec.target_artifact_format,
        required=runtime_spec.required,
        artifact_name=runtime_spec.artifact_name,
        artifact_directory=runtime_spec.artifact_directory,
        output_dir=runtime_spec.output_dir,
        placeholder_files=_placeholder_files(runtime_spec.artifact_name),
        manifest_patch=runtime_spec.manifest_updates,
        manifest_patch_description=_manifest_patch_description(runtime_spec.manifest_updates),
        constraint=runtime_spec.constraint,
        reason=runtime_spec.reason,
        backend_plan=runtime_spec.backend_plan,
        runtime_spec=runtime_spec.to_dict(),
        metadata={
            **runtime_spec.metadata,
            "materialized": True,
            "placeholder_count": len(_placeholder_files(runtime_spec.artifact_name)),
        },
    )


def materialized_export_plan(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper returning a serializable materialized export plan."""

    return materialize_export_plan(
        target_backend=target_backend,
        source_artifact_format=source_artifact_format,
        **kwargs,
    ).to_dict()


def write_materialized_export_plan(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None = None,
    adapter_dir: str | Path | None = None,
    workspace: str | None = None,
    source_adapter_version: str | None = None,
    source_model: str | None = None,
    training_run_id: str | None = None,
    num_samples: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> MaterializedExportWriteResult:
    """Materialize the export plan into a local placeholder layout.

    This helper writes only metadata and placeholder files:
    - standard export directory
    - placeholder artifact file
    - export_plan.json
    - EXPORT_NOTES.txt
    """

    plan = materialize_export_plan(
        target_backend=target_backend,
        source_artifact_format=source_artifact_format,
        adapter_dir=adapter_dir,
        workspace=workspace,
        source_adapter_version=source_adapter_version,
        source_model=source_model,
        training_run_id=training_run_id,
        num_samples=num_samples,
        extra_metadata=extra_metadata,
    )
    if adapter_dir is None:
        raise ValueError("adapter_dir is required to materialize an export plan locally")

    output_dir = Path(adapter_dir).expanduser() / plan.artifact_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / plan.artifact_name
    export_plan_path = output_dir / "export_plan.json"
    notes_path = output_dir / "EXPORT_NOTES.txt"
    artifact_preexisting = artifact_path.exists()
    artifact_preserved = False
    if artifact_preexisting and artifact_path.is_file() and not _is_placeholder_export_artifact(artifact_path):
        artifact_preserved = True
    else:
        artifact_path.write_text(
            "\n".join(
                [
                    "PFE placeholder artifact",
                    "This file marks where the exported adapter payload will be written.",
                    f"target_backend={plan.target_backend}",
                    f"target_artifact_format={plan.target_artifact_format}",
                    f"required={plan.required}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    export_plan_path.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    notes_path.write_text(_render_export_notes(plan, artifact_path), encoding="utf-8")

    written_files = [str(artifact_path), str(export_plan_path), str(notes_path)]
    return MaterializedExportWriteResult(
        output_dir=str(output_dir),
        artifact_path=str(artifact_path),
        export_plan_path=str(export_plan_path),
        notes_path=str(notes_path),
        written_files=written_files,
        plan=plan.to_dict(),
        metadata={
            "written_count": len(written_files),
            "placeholder_files": plan.placeholder_files,
            "materialized": True,
            "artifact_preexisting": artifact_preexisting,
            "artifact_preserved": artifact_preserved,
            "artifact_state": "preserved" if artifact_preserved else "placeholder_written",
        },
    )


def export_runtime_summary(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Alias for callers that prefer a more explicit runtime naming."""

    return dry_run_export_spec(
        target_backend=target_backend,
        source_artifact_format=source_artifact_format,
        **kwargs,
    )
