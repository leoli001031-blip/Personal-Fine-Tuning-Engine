"""Runtime probing helpers for trainer backend planning.

The mainline can use this module to collect local environment facts and feed
them into :mod:`pfe_core.trainer.backends` without importing any heavyweight
finetuning libraries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from importlib.metadata import PackageNotFoundError
from importlib.util import find_spec
from platform import machine as platform_machine
from platform import processor as platform_processor
from platform import system as platform_system
import sys
from typing import Any, Dict, Mapping, Optional, Tuple

from .backends import BackendPlan, describe_training_backends, plan_training_backend

_PROBED_PACKAGES: Tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "unsloth",
    "mlx",
    "mlx_lm",
    "accelerate",
    "trl",
    "datasets",
)


def _probe_package(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except Exception:
        return False


def _probe_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _probe_requires_python(name: str) -> Optional[str]:
    try:
        metadata = importlib_metadata.metadata(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None
    requires_python = metadata.get("Requires-Python")
    return str(requires_python) if requires_python else None


def _python_supported_for_requires_python(requires_python: Optional[str]) -> Optional[bool]:
    if not requires_python:
        return None

    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except Exception:
        return None

    try:
        current_version = Version("%s.%s.%s" % tuple(sys.version_info[:3]))
        return current_version in SpecifierSet(requires_python)
    except Exception:
        return None


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_mps_available() -> bool:
    try:
        import torch

        backends = getattr(torch, "backends", None)
        if backends is None:
            return False
        mps = getattr(backends, "mps", None)
        if mps is None:
            return False
        return bool(mps.is_available())
    except Exception:
        return False


@dataclass(frozen=True)
class TrainerRuntimeContext:
    """Snapshot of the local environment used by the trainer planner."""

    platform_name: str
    machine: str
    processor: str
    python_version: str
    apple_silicon: bool
    cuda_available: bool
    mps_available: bool
    cpu_only: bool
    runtime_device: str
    installed_packages: Dict[str, bool] = field(default_factory=dict)
    dependency_versions: Dict[str, str] = field(default_factory=dict)
    requires_python: Dict[str, str] = field(default_factory=dict)
    python_supported: Dict[str, Optional[bool]] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload

    def to_runtime_mapping(self) -> Dict[str, Any]:
        payload = self.to_dict()
        payload["runtime_device"] = self.runtime_device
        return payload


def detect_trainer_runtime() -> TrainerRuntimeContext:
    platform_name = platform_system()
    machine = platform_machine()
    processor = platform_processor()
    python_version = "%s.%s.%s" % tuple(sys.version_info[:3])
    apple_silicon = platform_name == "Darwin" and machine.lower() in {"arm64", "aarch64"}
    cuda_available = _torch_cuda_available()
    mps_available = _torch_mps_available()
    cpu_only = not cuda_available and not mps_available
    runtime_device = "cuda" if cuda_available else "mps" if mps_available else "cpu"

    installed_packages: Dict[str, bool] = {}
    dependency_versions: Dict[str, str] = {}
    requires_python_by_package: Dict[str, str] = {}
    python_supported: Dict[str, Optional[bool]] = {}
    for package in _PROBED_PACKAGES:
        installed_packages[package] = _probe_package(package)
        if not installed_packages[package]:
            continue
        version = _probe_version(package)
        if version is not None:
            dependency_versions[package] = version
        package_requires_python = _probe_requires_python(package)
        if package_requires_python is not None:
            requires_python_by_package[package] = package_requires_python
            python_supported[package] = _python_supported_for_requires_python(package_requires_python)

    notes = []
    if apple_silicon:
        notes.append("detected Apple Silicon host")
    if cuda_available:
        notes.append("torch reports CUDA available")
    if mps_available:
        notes.append("torch reports MPS available")
    if cpu_only:
        notes.append("runtime is CPU-only for trainer planning")

    return TrainerRuntimeContext(
        platform_name=platform_name,
        machine=machine,
        processor=processor,
        python_version=python_version,
        apple_silicon=apple_silicon,
        cuda_available=cuda_available,
        mps_available=mps_available,
        cpu_only=cpu_only,
        runtime_device=runtime_device,
        installed_packages=installed_packages,
        dependency_versions=dependency_versions,
        requires_python=requires_python_by_package,
        python_supported=python_supported,
        notes=tuple(notes),
    )


def plan_trainer_backend(
    *,
    train_type: str = "sft",
    backend_hint: Optional[str] = None,
    device_preference: str = "auto",
    target_inference_backend: Optional[str] = None,
    allow_mock_fallback: bool = True,
    runtime: Optional[Mapping[str, Any]] = None,
) -> BackendPlan:
    """Plan the trainer backend using either an explicit runtime or a probe."""

    runtime_mapping = runtime or detect_trainer_runtime().to_runtime_mapping()
    return plan_training_backend(
        train_type=train_type,
        runtime=runtime_mapping,
        backend_hint=backend_hint,
        device_preference=device_preference,
        target_inference_backend=target_inference_backend,
        allow_mock_fallback=allow_mock_fallback,
    )


def summarize_trainer_backend_plan(
    *,
    train_type: str = "sft",
    backend_hint: Optional[str] = None,
    device_preference: str = "auto",
    target_inference_backend: Optional[str] = None,
    allow_mock_fallback: bool = True,
    runtime: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper that returns a plain dictionary."""

    return plan_trainer_backend(
        train_type=train_type,
        backend_hint=backend_hint,
        device_preference=device_preference,
        target_inference_backend=target_inference_backend,
        allow_mock_fallback=allow_mock_fallback,
        runtime=runtime,
    ).to_dict()


def trainer_runtime_summary() -> Dict[str, Any]:
    """Return a serializable summary of the local trainer runtime."""

    context = detect_trainer_runtime()
    payload = context.to_dict()
    payload["capabilities"] = describe_training_backends()
    return payload


get_trainer_backend_plan = summarize_trainer_backend_plan
