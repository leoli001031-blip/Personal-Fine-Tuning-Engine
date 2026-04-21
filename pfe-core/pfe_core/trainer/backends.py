"""Trainer backend capability and planning helpers.

This module is intentionally pure and lightweight so the mainline can ask
questions like "which backend should I use for this train type on this
machine?" without importing heavyweight training stacks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from platform import machine as platform_machine
from platform import system as platform_system
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

TRAIN_TYPES: Tuple[str, ...] = ("sft", "dpo")
TRAINING_BACKENDS: Tuple[str, ...] = ("mock_local", "peft", "unsloth", "mlx")
DEFAULT_BACKEND: str = "mock_local"
SUPPORTED_DEVICE_PREFERENCES: Tuple[str, ...] = ("auto", "cpu", "cuda", "mps")
LLAMA_CPP_BACKEND: str = "llama_cpp"
GGUF_MERGED_FORMAT: str = "gguf_merged"
PEFT_LORA_FORMAT: str = "peft_lora"
MLX_LORA_FORMAT: str = "mlx_lora"

BACKEND_ALIASES: Dict[str, Tuple[str, ...]] = {
    "mock_local": ("mock", "mock_local", "local_mock"),
    "peft": ("peft", "hf_peft", "transformers_peft"),
    "unsloth": ("unsloth", "fast_lora", "unsloth_lora"),
    "mlx": ("mlx", "mlx_lm", "mlx-lm", "mlx_lora"),
}


def normalize_train_type(value: Optional[str]) -> str:
    normalized = (value or "sft").strip().lower().replace("-", "_")
    if normalized not in TRAIN_TYPES:
        raise ValueError(f"unsupported train_type: {value}")
    return normalized


def normalize_backend_name(value: Optional[str]) -> str:
    normalized = (value or "auto").strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return "auto"
    for canonical, aliases in BACKEND_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized


def normalize_device_preference(value: Optional[str]) -> str:
    normalized = (value or "auto").strip().lower().replace("-", "_")
    if normalized in {"", "auto"}:
        return "auto"
    if normalized not in SUPPORTED_DEVICE_PREFERENCES:
        raise ValueError(f"unsupported device preference: {value}")
    return normalized


def normalize_platform_name(value: Optional[str]) -> str:
    if not value:
        return platform_system()
    return value.strip()


def looks_like_apple_silicon(
    platform_name: Optional[str] = None,
    machine_name: Optional[str] = None,
) -> bool:
    platform_name = normalize_platform_name(platform_name)
    machine_name = (machine_name or platform_machine()).strip().lower()
    return platform_name == "Darwin" and machine_name in {"arm64", "aarch64"}


def _coerce_runtime_value(runtime: Optional[Mapping[str, Any]], key: str, default: Any = None) -> Any:
    if not runtime:
        return default
    return runtime.get(key, default)


def _runtime_packages(runtime: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    packages = _coerce_runtime_value(runtime, "installed_packages", {})
    if isinstance(packages, Mapping):
        return {str(key): bool(value) for key, value in packages.items()}
    return {}


def _runtime_dependency_versions(runtime: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    versions = _coerce_runtime_value(runtime, "dependency_versions", {})
    if isinstance(versions, Mapping):
        return {str(key): str(value) for key, value in versions.items() if value is not None}
    return {}


def _missing_dependencies(required: Sequence[str], runtime: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
    packages = _runtime_packages(runtime)
    missing = [name for name in required if not packages.get(name, False)]
    return tuple(missing)


@dataclass(frozen=True)
class BackendCapability:
    """Static capability description for a trainer backend."""

    name: str
    supports_sft: bool
    supports_dpo: bool
    artifact_format: str
    supported_devices: Tuple[str, ...]
    preferred_on: Tuple[str, ...]
    required_dependencies: Tuple[str, ...] = ()
    optional_dependencies: Tuple[str, ...] = ()
    supports_cpu_only: bool = True
    supports_cuda: bool = True
    supports_apple_silicon: bool = True
    requires_export_for_llama_cpp: bool = False
    notes: Tuple[str, ...] = ()

    def supports_train_type(self, train_type: str) -> bool:
        normalized = normalize_train_type(train_type)
        return self.supports_sft if normalized == "sft" else self.supports_dpo

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["required_dependencies"] = list(self.required_dependencies)
        payload["optional_dependencies"] = list(self.optional_dependencies)
        payload["supported_devices"] = list(self.supported_devices)
        payload["preferred_on"] = list(self.preferred_on)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class BackendPlan:
    """Planner output that the trainer service can consume directly."""

    requested_backend: str
    recommended_backend: str
    train_type: str
    device_preference: str
    runtime_device: str
    capability: BackendCapability
    missing_dependencies: Tuple[str, ...] = ()
    available: bool = True
    reason: str = ""
    reasons: Tuple[str, ...] = ()
    requires_export_step: bool = False
    export_steps: Tuple[str, ...] = ()
    export_format: Optional[str] = None
    export_backend: Optional[str] = None
    alternative_backends: Tuple[str, ...] = ()
    runtime_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["capability"] = self.capability.to_dict()
        payload["missing_dependencies"] = list(self.missing_dependencies)
        payload["reasons"] = list(self.reasons)
        payload["export_steps"] = list(self.export_steps)
        payload["alternative_backends"] = list(self.alternative_backends)
        return payload


BACKEND_CAPABILITIES: Dict[str, BackendCapability] = {
    "mock_local": BackendCapability(
        name="mock_local",
        supports_sft=True,
        supports_dpo=True,
        artifact_format=PEFT_LORA_FORMAT,
        supported_devices=("cpu", "cuda", "mps"),
        preferred_on=("cpu", "cuda", "mps"),
        required_dependencies=(),
        optional_dependencies=(),
        supports_cpu_only=True,
        supports_cuda=True,
        supports_apple_silicon=True,
        requires_export_for_llama_cpp=False,
        notes=(
            "phase-0 fallback backend",
            "keeps the trainer runnable without external finetuning dependencies",
        ),
    ),
    "peft": BackendCapability(
        name="peft",
        supports_sft=True,
        supports_dpo=True,
        artifact_format=PEFT_LORA_FORMAT,
        supported_devices=("cpu", "cuda", "mps"),
        preferred_on=("cpu", "cuda"),
        required_dependencies=("torch", "transformers", "peft", "accelerate"),
        optional_dependencies=("trl", "datasets"),
        supports_cpu_only=True,
        supports_cuda=True,
        supports_apple_silicon=True,
        requires_export_for_llama_cpp=True,
        notes=(
            "general-purpose local LoRA backend",
            "best default when DPO is requested",
        ),
    ),
    "unsloth": BackendCapability(
        name="unsloth",
        supports_sft=True,
        supports_dpo=False,
        artifact_format=PEFT_LORA_FORMAT,
        supported_devices=("cuda",),
        preferred_on=("cuda",),
        required_dependencies=("torch", "transformers", "unsloth"),
        optional_dependencies=("peft", "accelerate", "trl"),
        supports_cpu_only=False,
        supports_cuda=True,
        supports_apple_silicon=False,
        requires_export_for_llama_cpp=True,
        notes=(
            "CUDA-optimized fine-tuning path",
            "planner treats DPO as unsupported until the backend explicitly advertises it",
        ),
    ),
    "mlx": BackendCapability(
        name="mlx",
        supports_sft=True,
        supports_dpo=False,
        artifact_format=MLX_LORA_FORMAT,
        supported_devices=("mps",),
        preferred_on=("mps",),
        required_dependencies=("mlx", "mlx_lm"),
        optional_dependencies=("torch",),
        supports_cpu_only=False,
        supports_cuda=False,
        supports_apple_silicon=True,
        requires_export_for_llama_cpp=True,
        notes=(
            "Apple Silicon first path",
            "sft-native backend for local macOS workloads",
        ),
    ),
}


def available_backends() -> Tuple[str, ...]:
    return tuple(TRAINING_BACKENDS)


def backend_capabilities() -> Dict[str, BackendCapability]:
    return dict(BACKEND_CAPABILITIES)


def is_known_backend_name(name: Optional[str]) -> bool:
    return normalize_backend_name(name) in BACKEND_CAPABILITIES


def get_backend_capability(name: Optional[str]) -> BackendCapability:
    canonical = normalize_backend_name(name)
    return BACKEND_CAPABILITIES.get(canonical, BACKEND_CAPABILITIES[DEFAULT_BACKEND])


def backend_missing_dependencies(
    name: Optional[str],
    runtime: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, ...]:
    capability = get_backend_capability(name)
    return _missing_dependencies(capability.required_dependencies, runtime)


def backend_is_supported_on_runtime(
    name: Optional[str],
    *,
    train_type: str,
    runtime: Optional[Mapping[str, Any]] = None,
    device_preference: str = "auto",
) -> bool:
    capability = get_backend_capability(name)
    if not capability.supports_train_type(train_type):
        return False

    device = normalize_device_preference(device_preference)
    runtime_device = _coerce_runtime_value(runtime, "runtime_device", "cpu")
    apple_silicon = bool(_coerce_runtime_value(runtime, "apple_silicon", False))
    cuda_available = bool(_coerce_runtime_value(runtime, "cuda_available", False))
    mps_available = bool(_coerce_runtime_value(runtime, "mps_available", False))

    if device == "cpu" and not capability.supports_cpu_only:
        return False
    if device == "cuda" and not capability.supports_cuda:
        return False
    if device == "mps" and not capability.supports_apple_silicon:
        return False

    if capability.name == "unsloth" and not cuda_available:
        return False
    if capability.name == "mlx" and not (apple_silicon or mps_available or runtime_device == "mps"):
        return False
    return True


def _device_from_runtime(runtime: Optional[Mapping[str, Any]]) -> str:
    if not runtime:
        return "cpu"
    runtime_device = _coerce_runtime_value(runtime, "runtime_device", None)
    if isinstance(runtime_device, str) and runtime_device:
        return runtime_device
    if bool(_coerce_runtime_value(runtime, "cuda_available", False)):
        return "cuda"
    if bool(_coerce_runtime_value(runtime, "mps_available", False)):
        return "mps"
    return "cpu"


def _rank_backends_for_context(
    *,
    train_type: str,
    runtime: Optional[Mapping[str, Any]],
    device_preference: str,
) -> Tuple[str, ...]:
    device = normalize_device_preference(device_preference)
    apple_silicon = bool(_coerce_runtime_value(runtime, "apple_silicon", False))
    cuda_available = bool(_coerce_runtime_value(runtime, "cuda_available", False))
    mps_available = bool(_coerce_runtime_value(runtime, "mps_available", False))

    candidates = []
    if train_type == "dpo":
        if cuda_available:
            candidates.extend(["unsloth", "peft"])
        else:
            candidates.append("peft")
        candidates.append("mock_local")
    else:
        if device == "mps" or (device == "auto" and apple_silicon and mps_available):
            candidates.extend(["mlx", "peft", "mock_local"])
        elif device == "cuda" or cuda_available:
            candidates.extend(["unsloth", "peft", "mock_local"])
        else:
            candidates.extend(["peft", "mock_local"])
            if apple_silicon and mps_available:
                candidates.insert(0, "mlx")

    ordered: list[str] = []
    for name in candidates:
        if name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def plan_training_backend(
    *,
    train_type: str = "sft",
    runtime: Optional[Mapping[str, Any]] = None,
    backend_hint: Optional[str] = None,
    device_preference: str = "auto",
    target_inference_backend: Optional[str] = None,
    allow_mock_fallback: bool = True,
) -> BackendPlan:
    """Plan the most suitable training backend for the current runtime.

    The result is deliberately compact and serializable so the trainer service
    can call ``plan.to_dict()`` or inspect the individual fields directly.
    """

    normalized_train_type = normalize_train_type(train_type)
    requested_backend = normalize_backend_name(backend_hint)
    normalized_device = normalize_device_preference(device_preference)
    runtime = runtime or {}

    platform_name = normalize_platform_name(_coerce_runtime_value(runtime, "platform_name", platform_system()))
    machine_name = str(_coerce_runtime_value(runtime, "machine", platform_machine()))
    apple_silicon = bool(_coerce_runtime_value(runtime, "apple_silicon", looks_like_apple_silicon(platform_name, machine_name)))
    cuda_available = bool(_coerce_runtime_value(runtime, "cuda_available", False))
    mps_available = bool(_coerce_runtime_value(runtime, "mps_available", False))
    runtime_device = _device_from_runtime(runtime)

    runtime_summary = {
        "platform_name": platform_name,
        "machine": machine_name,
        "apple_silicon": apple_silicon,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "runtime_device": runtime_device,
        "device_preference": normalized_device,
        "installed_packages": _runtime_packages(runtime),
        "dependency_versions": _runtime_dependency_versions(runtime),
    }

    available = []
    unavailable_reasons: Dict[str, Tuple[str, ...]] = {}
    for backend_name in TRAINING_BACKENDS:
        capability = BACKEND_CAPABILITIES[backend_name]
        missing = _missing_dependencies(capability.required_dependencies, runtime)
        supported = backend_is_supported_on_runtime(
            backend_name,
            train_type=normalized_train_type,
            runtime=runtime_summary,
            device_preference=normalized_device,
        )
        if missing or not supported:
            reasons = []
            if missing:
                reasons.append("missing dependencies: " + ", ".join(missing))
            if not capability.supports_train_type(normalized_train_type):
                reasons.append(f"{backend_name} does not support {normalized_train_type.upper()}")
            if normalized_train_type == "dpo" and backend_name in {"unsloth", "mlx"}:
                reasons.append(f"{backend_name} is not marked as DPO-capable")
            unavailable_reasons[backend_name] = tuple(reasons) if reasons else ("unavailable on this runtime",)
            continue
        available.append(backend_name)

    ordered = list(_rank_backends_for_context(
        train_type=normalized_train_type,
        runtime=runtime_summary,
        device_preference=normalized_device,
    ))
    if allow_mock_fallback and "mock_local" not in ordered:
        ordered.append("mock_local")

    selected_backend = None
    selection_reason = ""
    selection_reasons: Tuple[str, ...] = ()
    missing_dependencies: Tuple[str, ...] = ()

    if requested_backend != "auto":
        if not is_known_backend_name(requested_backend):
            fallback_order = list(ordered)
            fallback_backend = fallback_order[0] if fallback_order else DEFAULT_BACKEND
            selected_backend = fallback_backend
            capability = get_backend_capability(selected_backend)
            missing_dependencies = _missing_dependencies(capability.required_dependencies, runtime_summary)
            selection_reason = f"unknown backend '{backend_hint}', falling back to {fallback_backend}"
            selection_reasons = (selection_reason,)
        else:
            selected_backend = requested_backend
            capability = get_backend_capability(selected_backend)
            missing_dependencies = _missing_dependencies(capability.required_dependencies, runtime_summary)
            supported = backend_is_supported_on_runtime(
                selected_backend,
                train_type=normalized_train_type,
                runtime=runtime_summary,
                device_preference=normalized_device,
            )
            if supported and not missing_dependencies:
                selection_reason = f"explicit backend request: {selected_backend}"
                selection_reasons = (selection_reason,)
            else:
                fallback_order = [name for name in ordered if name != selected_backend]
                fallback_backend = fallback_order[0] if fallback_order else DEFAULT_BACKEND
                fallback_capability = get_backend_capability(fallback_backend)
                fallback_missing = _missing_dependencies(fallback_capability.required_dependencies, runtime_summary)
                selected_backend = fallback_backend
                missing_dependencies = fallback_missing
                selection_reason = f"explicit backend '{backend_hint}' is unavailable, falling back to {fallback_backend}"
                reasons = [selection_reason]
                if missing_dependencies:
                    reasons.append("missing dependencies for fallback: " + ", ".join(missing_dependencies))
                if not supported:
                    reasons.append("requested backend is not supported by this runtime or train type")
                selection_reasons = tuple(reasons)
    else:
        for backend_name in ordered:
            capability = get_backend_capability(backend_name)
            missing = _missing_dependencies(capability.required_dependencies, runtime_summary)
            if backend_name not in available:
                continue
            if not capability.supports_train_type(normalized_train_type):
                continue
            if backend_name == "mlx" and normalized_device == "cuda":
                continue
            if backend_name == "unsloth" and not cuda_available:
                continue
            selected_backend = backend_name
            missing_dependencies = missing
            break
        if selected_backend is None:
            selected_backend = DEFAULT_BACKEND
            capability = get_backend_capability(selected_backend)
            missing_dependencies = _missing_dependencies(capability.required_dependencies, runtime_summary)
        selection_reason = f"auto-selected {selected_backend} for {normalized_train_type.upper()}"
        selection_reasons = (selection_reason,)

    capability = get_backend_capability(selected_backend)
    if requested_backend == "auto" and selected_backend != "mock_local":
        if selected_backend == "mlx":
            selection_reasons += ("Apple Silicon local runtime prefers mlx for SFT",)
        elif selected_backend == "unsloth":
            selection_reasons += ("CUDA runtime prefers unsloth for fast local finetuning",)
        elif normalized_train_type == "dpo":
            selection_reasons += ("DPO currently routes to the most explicit local trainer backend",)

    if selected_backend == "mock_local":
        selection_reasons += ("mock_local is the fallback backend and does not require external trainer dependencies",)

    target_backend = normalize_backend_name(target_inference_backend)
    requires_export_step = False
    export_steps: Tuple[str, ...] = ()
    export_format: Optional[str] = None
    export_backend: Optional[str] = None
    if target_backend == LLAMA_CPP_BACKEND:
        export_backend = LLAMA_CPP_BACKEND
        export_format = GGUF_MERGED_FORMAT
        if capability.artifact_format != GGUF_MERGED_FORMAT:
            requires_export_step = True
            export_steps = ("gguf_merged_export",)
            selection_reasons += ("llama.cpp serving requires gguf_merged export",)

    alternative_backends = tuple(name for name in ordered if name != selected_backend)
    return BackendPlan(
        requested_backend=requested_backend,
        recommended_backend=selected_backend,
        train_type=normalized_train_type,
        device_preference=normalized_device,
        runtime_device=runtime_device,
        capability=capability,
        missing_dependencies=missing_dependencies,
        available=selected_backend in available or selected_backend == "mock_local",
        reason=selection_reason,
        reasons=selection_reasons,
        requires_export_step=requires_export_step,
        export_steps=export_steps,
        export_format=export_format,
        export_backend=export_backend,
        alternative_backends=alternative_backends,
        runtime_summary=runtime_summary,
    )


def summarize_backend_plan(
    *,
    train_type: str = "sft",
    runtime: Optional[Mapping[str, Any]] = None,
    backend_hint: Optional[str] = None,
    device_preference: str = "auto",
    target_inference_backend: Optional[str] = None,
    allow_mock_fallback: bool = True,
) -> Dict[str, Any]:
    """Return a serializable planner summary for CLI and service callers."""

    return plan_training_backend(
        train_type=train_type,
        runtime=runtime,
        backend_hint=backend_hint,
        device_preference=device_preference,
        target_inference_backend=target_inference_backend,
        allow_mock_fallback=allow_mock_fallback,
    ).to_dict()


def describe_training_backends() -> Dict[str, Dict[str, Any]]:
    """Return the static capability matrix as plain dictionaries."""

    return {name: capability.to_dict() for name, capability in BACKEND_CAPABILITIES.items()}


select_training_backend = plan_training_backend
get_training_backend_plan = summarize_backend_plan
