"""Inference backend selection helpers.

This module keeps backend selection independent from runtime execution so the
mainline can decide between ``transformers``, ``mlx``, ``llama_cpp`` and
``auto`` without importing heavyweight backends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from platform import machine as platform_machine
from platform import system as platform_system
from typing import Any, Mapping

from ..adapter_store.lifecycle import AdapterArtifactFormat, llama_cpp_requires_merged_gguf

InferenceBackendName = str

SUPPORTED_INFERENCE_BACKENDS: tuple[str, ...] = ("auto", "transformers", "mlx", "llama_cpp")
DEFAULT_LOCAL_BACKEND: str = "transformers"
LLAMA_CPP_BACKEND_ALIASES: tuple[str, ...] = ("llama", "llama.cpp", "llama_cpp")
TRANSFORMERS_BACKEND_ALIASES: tuple[str, ...] = ("hf", "huggingface", "transformers")
MLX_BACKEND_ALIASES: tuple[str, ...] = ("mlx_lm", "mlx-lm", "mlx")


def normalize_backend_name(value: str | None) -> str:
    """Normalize backend names to the canonical local planning vocabulary."""

    if value is None:
        return "auto"
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in LLAMA_CPP_BACKEND_ALIASES:
        return "llama_cpp"
    if normalized in TRANSFORMERS_BACKEND_ALIASES:
        return "transformers"
    if normalized in MLX_BACKEND_ALIASES:
        return "mlx"
    return normalized


def backend_requires_gguf_merged(backend: str | None) -> bool:
    """Return whether the backend can only consume ``gguf_merged`` artifacts."""

    return llama_cpp_requires_merged_gguf(normalize_backend_name(backend))


def backend_supports_artifact_format(backend: str | None, artifact_format: str | AdapterArtifactFormat | None) -> bool:
    """Best-effort compatibility check for a planned backend/artifact pair."""

    normalized_backend = normalize_backend_name(backend)
    normalized_artifact = normalize_artifact_format(artifact_format)
    if normalized_backend in {"auto", "transformers", "mlx"}:
        return normalized_artifact != AdapterArtifactFormat.GGUF_MERGED.value
    if normalized_backend == "llama_cpp":
        return normalized_artifact == AdapterArtifactFormat.GGUF_MERGED.value
    return True


def normalize_artifact_format(value: str | AdapterArtifactFormat | None) -> str:
    if value is None:
        return AdapterArtifactFormat.PEFT_LORA.value
    if isinstance(value, AdapterArtifactFormat):
        return value.value
    return str(value).strip().lower().replace("-", "_")


def _looks_like_apple_silicon(system_name: str, machine_name: str) -> bool:
    return system_name == "Darwin" and machine_name in {"arm64", "aarch64"}


def _coerce_manifest_value(manifest: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if not manifest:
        return default
    return manifest.get(key, default)


@dataclass(frozen=True)
class BackendDecision:
    """Outcome of the backend selection helper."""

    requested_backend: str
    selected_backend: str
    reason: str
    requires_export: bool = False
    required_artifact_format: str | None = None
    compatible_artifact_formats: tuple[str, ...] = (
        AdapterArtifactFormat.PEFT_LORA.value,
        AdapterArtifactFormat.MLX_LORA.value,
        AdapterArtifactFormat.GGUF_MERGED.value,
    )
    preferred_device: str = "auto"
    strict_local: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compatible_artifact_formats"] = list(self.compatible_artifact_formats)
        return payload


def plan_inference_backend(
    *,
    requested_backend: str | None = "auto",
    artifact_format: str | AdapterArtifactFormat | None = None,
    manifest: Mapping[str, Any] | None = None,
    device: str = "auto",
    prefer_local: bool = True,
    platform_name: str | None = None,
    machine_name: str | None = None,
) -> BackendDecision:
    """Decide which local inference backend to use.

    Priority order:
    1. Explicit backend request.
    2. Merged GGUF artifacts route to ``llama_cpp``.
    3. Apple Silicon prefers ``mlx`` when available.
    4. Fallback to ``transformers``.
    """

    requested = normalize_backend_name(requested_backend)
    manifest_backend = normalize_backend_name(_coerce_manifest_value(manifest, "inference_backend", None))
    manifest_artifact = normalize_artifact_format(_coerce_manifest_value(manifest, "artifact_format", artifact_format))
    platform_name = platform_name or platform_system()
    machine_name = machine_name or platform_machine()

    if requested != "auto":
        selected = requested
        requires_export = backend_requires_gguf_merged(selected) and manifest_artifact != AdapterArtifactFormat.GGUF_MERGED.value
        reason = f"explicit backend request: {selected}"
        if selected == "llama_cpp" and manifest_artifact != AdapterArtifactFormat.GGUF_MERGED.value:
            reason += " (merged GGUF export required)"
        return BackendDecision(
            requested_backend=requested,
            selected_backend=selected,
            reason=reason,
            requires_export=requires_export,
            required_artifact_format=AdapterArtifactFormat.GGUF_MERGED.value if requires_export else None,
            preferred_device=device,
            strict_local=prefer_local,
            metadata={
                "platform": platform_name,
                "machine": machine_name,
                "manifest_backend": manifest_backend,
                "manifest_artifact_format": manifest_artifact,
            },
        )

    if manifest_backend != "auto":
        if manifest_artifact == AdapterArtifactFormat.GGUF_MERGED.value:
            selected = "llama_cpp"
            reason = "manifest artifact is merged GGUF, so llama.cpp is the compatible local backend"
            return BackendDecision(
                requested_backend=requested,
                selected_backend=selected,
                reason=reason,
                requires_export=False,
                required_artifact_format=AdapterArtifactFormat.GGUF_MERGED.value,
                preferred_device=device,
                strict_local=prefer_local,
                metadata={
                    "platform": platform_name,
                    "machine": machine_name,
                    "manifest_backend": manifest_backend,
                    "manifest_artifact_format": manifest_artifact,
                },
            )

        if backend_supports_artifact_format(manifest_backend, manifest_artifact):
            selected = manifest_backend
            reason = f"manifest already prefers {selected}"
            return BackendDecision(
                requested_backend=requested,
                selected_backend=selected,
                reason=reason,
                requires_export=False,
                preferred_device=device,
                strict_local=prefer_local,
                metadata={
                    "platform": platform_name,
                    "machine": machine_name,
                    "manifest_backend": manifest_backend,
                    "manifest_artifact_format": manifest_artifact,
                },
            )

        selected = manifest_backend
        reason = f"manifest prefers {selected}, but artifact format requires export"
        return BackendDecision(
            requested_backend=requested,
            selected_backend=selected,
            reason=reason,
            requires_export=backend_requires_gguf_merged(selected),
            required_artifact_format=AdapterArtifactFormat.GGUF_MERGED.value
            if backend_requires_gguf_merged(selected)
            else None,
            preferred_device=device,
            strict_local=prefer_local,
            metadata={
                "platform": platform_name,
                "machine": machine_name,
                "manifest_backend": manifest_backend,
                "manifest_artifact_format": manifest_artifact,
            },
        )

    if manifest_artifact == AdapterArtifactFormat.GGUF_MERGED.value:
        selected = "llama_cpp"
        reason = "artifact format is merged GGUF, so llama.cpp is the most direct local backend"
        return BackendDecision(
            requested_backend=requested,
            selected_backend=selected,
            reason=reason,
            requires_export=False,
            required_artifact_format=AdapterArtifactFormat.GGUF_MERGED.value,
            preferred_device=device,
            strict_local=prefer_local,
            metadata={
                "platform": platform_name,
                "machine": machine_name,
                "manifest_backend": manifest_backend,
                "manifest_artifact_format": manifest_artifact,
            },
        )

    if _looks_like_apple_silicon(platform_name, machine_name) and device in {"auto", "mps", "metal"}:
        selected = "mlx"
        reason = "Apple Silicon local runtime prefers mlx for the auto path"
    else:
        selected = DEFAULT_LOCAL_BACKEND
        reason = "fallback local backend is transformers"

    if manifest_backend in {"transformers", "mlx", "llama_cpp"}:
        reason += f"; manifest already advertises {manifest_backend}"

    return BackendDecision(
        requested_backend=requested,
        selected_backend=selected,
        reason=reason,
        requires_export=False,
        preferred_device=device,
        strict_local=prefer_local,
        metadata={
            "platform": platform_name,
            "machine": machine_name,
            "manifest_backend": manifest_backend,
            "manifest_artifact_format": manifest_artifact,
        },
    )


def backend_aliases(backend: str | None) -> tuple[str, ...]:
    """Return the known aliases for a canonical backend name."""

    normalized = normalize_backend_name(backend)
    if normalized == "llama_cpp":
        return LLAMA_CPP_BACKEND_ALIASES
    if normalized == "mlx":
        return MLX_BACKEND_ALIASES
    if normalized == "transformers":
        return TRANSFORMERS_BACKEND_ALIASES
    if normalized == "auto":
        return ("auto",)
    return (normalized,)


def summarize_backend_plan(
    *,
    requested_backend: str | None = "auto",
    artifact_format: str | AdapterArtifactFormat | None = None,
    manifest: Mapping[str, Any] | None = None,
    device: str = "auto",
    prefer_local: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper for callers that want a dict instead of a dataclass."""

    return plan_inference_backend(
        requested_backend=requested_backend,
        artifact_format=artifact_format,
        manifest=manifest,
        device=device,
        prefer_local=prefer_local,
    ).to_dict()
