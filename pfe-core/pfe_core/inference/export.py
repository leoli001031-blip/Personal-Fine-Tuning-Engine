"""Export planning helpers for inference artifacts.

The Phase 0 implementation keeps export planning lightweight and local.  The
mainline can use this module to decide whether an adapter must be converted to
merged GGUF before serving on the ``llama.cpp`` path, while keeping the
actual exporter out of the skeleton.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from ..adapter_store.lifecycle import (
    AdapterArtifactFormat,
    artifact_role_for_format,
    llama_cpp_requires_merged_gguf,
)
from .backends import normalize_artifact_format, normalize_backend_name

EXPORT_MANIFEST_EXPORT_KEY = "export"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def export_artifact_name_for_format(artifact_format: str | AdapterArtifactFormat | None) -> str:
    """Return the canonical artifact filename for a planned export format."""

    normalized = normalize_artifact_format(artifact_format)
    if normalized == AdapterArtifactFormat.GGUF_MERGED.value:
        return "adapter_model.gguf"
    if normalized == AdapterArtifactFormat.MLX_LORA.value:
        return "adapter_model.npz"
    return "adapter_model.safetensors"


def export_directory_name_for_format(artifact_format: str | AdapterArtifactFormat | None) -> str:
    """Return the canonical directory hint for the exported payload."""

    normalized = normalize_artifact_format(artifact_format)
    if normalized == AdapterArtifactFormat.GGUF_MERGED.value:
        return "gguf_merged"
    if normalized == AdapterArtifactFormat.MLX_LORA.value:
        return "mlx_lora"
    return "peft_lora"


def export_required_for_backend(
    target_backend: str | None,
    artifact_format: str | AdapterArtifactFormat | None,
) -> bool:
    """Check whether the requested backend needs a format conversion first."""

    normalized_backend = normalize_backend_name(target_backend)
    normalized_artifact = normalize_artifact_format(artifact_format)
    if not llama_cpp_requires_merged_gguf(normalized_backend):
        return False
    return normalized_artifact != AdapterArtifactFormat.GGUF_MERGED.value


def required_export_format_for_backend(target_backend: str | None) -> str | None:
    """Return the artifact format required by the backend, if any."""

    if llama_cpp_requires_merged_gguf(normalize_backend_name(target_backend)):
        return AdapterArtifactFormat.GGUF_MERGED.value
    return None


def export_contract_roles(
    *,
    source_artifact_format: str | AdapterArtifactFormat | None,
    target_artifact_format: str | AdapterArtifactFormat,
) -> dict[str, str]:
    """Return the semantic roles used by the export contract."""

    return {
        "source_artifact_role": artifact_role_for_format(source_artifact_format),
        "target_artifact_role": artifact_role_for_format(target_artifact_format),
        "base_model_role": "base_model",
    }


def manifest_export_fields(
    *,
    target_backend: str,
    source_artifact_format: str | AdapterArtifactFormat | None,
    target_artifact_format: str | AdapterArtifactFormat,
    workspace: str | None = None,
    adapter_dir: str | None = None,
    source_adapter_version: str | None = None,
    source_model: str | None = None,
    training_run_id: str | None = None,
    num_samples: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the manifest update payload for an export run.

    The manifest should continue to retain the core identity and provenance
    fields:
    - ``workspace``
    - ``adapter_dir``
    - ``source_adapter_version``
    - ``source_model``
    - ``training_run_id``
    - ``num_samples``
    - ``artifact_format``
    - ``artifact_name``
    - ``inference_backend``
    - ``requires_export``
    - ``artifact_contract``
    - ``metadata.export``
    """

    target_artifact = normalize_artifact_format(target_artifact_format)
    source_artifact = normalize_artifact_format(source_artifact_format)
    contract_roles = export_contract_roles(
        source_artifact_format=source_artifact,
        target_artifact_format=target_artifact,
    )
    export_metadata: dict[str, Any] = {
        "exported_at": utc_now_iso(),
        "target_backend": normalize_backend_name(target_backend),
        "source_artifact_format": source_artifact,
        "target_artifact_format": target_artifact,
        "export_required": export_required_for_backend(target_backend, source_artifact),
        "export_directory": export_directory_name_for_format(target_artifact),
        **contract_roles,
    }
    if extra_metadata:
        export_metadata.update(dict(extra_metadata))

    payload: dict[str, Any] = {
        "workspace": workspace,
        "adapter_dir": adapter_dir,
        "source_adapter_version": source_adapter_version,
        "source_model": source_model,
        "training_run_id": training_run_id,
        "num_samples": num_samples,
        "artifact_format": target_artifact,
        "artifact_name": export_artifact_name_for_format(target_artifact),
        "inference_backend": normalize_backend_name(target_backend),
        "requires_export": export_required_for_backend(target_backend, source_artifact),
        "artifact_contract": contract_roles,
        EXPORT_MANIFEST_EXPORT_KEY: export_metadata,
    }
    return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class ExportPlan:
    """Lightweight export planning result."""

    target_backend: str
    source_artifact_format: str
    target_artifact_format: str
    required: bool
    artifact_name: str
    artifact_directory: str
    manifest_updates: dict[str, Any]
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def plan_export(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None = None,
    workspace: str | None = None,
    adapter_dir: str | None = None,
    source_adapter_version: str | None = None,
    source_model: str | None = None,
    training_run_id: str | None = None,
    num_samples: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> ExportPlan:
    """Plan a backend-specific export without touching the filesystem."""

    normalized_target_backend = normalize_backend_name(target_backend)
    normalized_source_artifact = normalize_artifact_format(source_artifact_format)
    required_format = required_export_format_for_backend(normalized_target_backend)
    if required_format is None:
        target_artifact_format = normalized_source_artifact
        required = False
        reason = "backend does not require a format conversion"
    else:
        target_artifact_format = required_format
        required = normalized_source_artifact != required_format
        reason = "llama.cpp path requires merged GGUF export"

    manifest_updates = manifest_export_fields(
        target_backend=normalized_target_backend,
        source_artifact_format=normalized_source_artifact,
        target_artifact_format=target_artifact_format,
        workspace=workspace,
        adapter_dir=adapter_dir,
        source_adapter_version=source_adapter_version,
        source_model=source_model,
        training_run_id=training_run_id,
        num_samples=num_samples,
        extra_metadata=extra_metadata,
    )
    metadata = {
        "target_backend": normalized_target_backend,
        "source_artifact_format": normalized_source_artifact,
        "target_artifact_format": target_artifact_format,
        "required_format": required_format,
        "source_artifact_role": artifact_role_for_format(normalized_source_artifact),
        "target_artifact_role": artifact_role_for_format(target_artifact_format),
        "base_model_role": "base_model",
        "artifact_contract": export_contract_roles(
            source_artifact_format=normalized_source_artifact,
            target_artifact_format=target_artifact_format,
        ),
    }
    if extra_metadata:
        metadata["extra_metadata_keys"] = sorted(extra_metadata.keys())

    return ExportPlan(
        target_backend=normalized_target_backend,
        source_artifact_format=normalized_source_artifact,
        target_artifact_format=target_artifact_format,
        required=required,
        artifact_name=export_artifact_name_for_format(target_artifact_format),
        artifact_directory=export_directory_name_for_format(target_artifact_format),
        manifest_updates=manifest_updates,
        reason=reason,
        metadata=metadata,
    )


def export_summary(
    *,
    target_backend: str | None,
    source_artifact_format: str | AdapterArtifactFormat | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper returning a serializable export plan."""

    return plan_export(
        target_backend=target_backend,
        source_artifact_format=source_artifact_format,
        **kwargs,
    ).to_dict()
