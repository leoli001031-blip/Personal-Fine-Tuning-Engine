"""Automatic backend selector for training efficiency optimization.

This module provides intelligent backend selection based on:
- Hardware capabilities (Apple Silicon, CUDA, CPU)
- Training type (SFT, DPO)
- Available dependencies
- User preferences
- Model compatibility

The selector automatically chooses the optimal backend for the current
runtime environment to maximize training efficiency.

Example:
    >>> from pfe_core.trainer.backend_selector import AutoBackendSelector
    >>> selector = AutoBackendSelector()
    >>> backend = selector.select_backend(
    ...     train_type="sft",
    ...     base_model="unsloth/Llama-3.2-1B"
    ... )
    >>> print(backend)  # "unsloth" on CUDA, "mlx" on Apple Silicon
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from ..errors import TrainingError
from .backends import (
    BACKEND_CAPABILITIES,
    BackendCapability,
    normalize_backend_name,
    normalize_train_type,
)


@dataclass
class BackendSelectionResult:
    """Result of backend selection.

    Attributes:
        selected_backend: The chosen backend name
        confidence: Selection confidence score (0-1)
        reason: Human-readable selection reason
        alternatives: Alternative backends that could work
        requirements: Required dependencies
        estimated_performance: Relative performance estimate
        warnings: Any warnings about the selection
    """

    selected_backend: str
    confidence: float
    reason: str
    alternatives: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    estimated_performance: str = "unknown"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_backend": self.selected_backend,
            "confidence": self.confidence,
            "reason": self.reason,
            "alternatives": self.alternatives,
            "requirements": self.requirements,
            "estimated_performance": self.estimated_performance,
            "warnings": self.warnings,
        }


@dataclass
class HardwareProfile:
    """Hardware profile for backend selection.

    Attributes:
        platform: Operating system platform
        machine: Machine architecture
        is_apple_silicon: Whether running on Apple Silicon
        cuda_available: Whether CUDA is available
        cuda_version: CUDA version if available
        mps_available: Whether Metal Performance Shaders is available
        cpu_count: Number of CPU cores
        memory_gb: Available memory in GB
    """

    platform: str = ""
    machine: str = ""
    is_apple_silicon: bool = False
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    mps_available: bool = False
    cpu_count: int = 0
    memory_gb: Optional[float] = None

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Detect hardware profile from current system."""
        profile = cls(
            platform=platform.system().lower(),
            machine=platform.machine().lower(),
        )

        # Detect Apple Silicon
        profile.is_apple_silicon = (
            profile.platform == "darwin" and profile.machine in ("arm64", "aarch64")
        )

        # Detect CUDA
        try:
            import torch

            profile.cuda_available = torch.cuda.is_available()
            if profile.cuda_available:
                profile.cuda_version = torch.version.cuda
        except ImportError:
            pass

        # Detect MPS (Apple Silicon)
        if profile.is_apple_silicon:
            try:
                import torch

                profile.mps_available = torch.backends.mps.is_available()
            except (ImportError, AttributeError):
                # Try MLX as alternative indicator
                try:
                    import mlx.core as mx

                    profile.mps_available = True
                except ImportError:
                    pass

        # CPU count
        try:
            import os

            profile.cpu_count = os.cpu_count() or 0
        except Exception:
            pass

        # Memory info
        try:
            import psutil

            profile.memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass

        return profile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "machine": self.machine,
            "is_apple_silicon": self.is_apple_silicon,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "mps_available": self.mps_available,
            "cpu_count": self.cpu_count,
            "memory_gb": self.memory_gb,
        }


@dataclass
class DependencyProfile:
    """Dependency availability profile.

    Attributes:
        torch: Whether PyTorch is available
        transformers: Whether transformers is available
        peft: Whether PEFT is available
        unsloth: Whether Unsloth is available
        mlx: Whether MLX is available
        mlx_lm: Whether mlx_lm is available
        trl: Whether TRL is available
        accelerate: Whether Accelerate is available
    """

    torch: bool = False
    transformers: bool = False
    peft: bool = False
    unsloth: bool = False
    mlx: bool = False
    mlx_lm: bool = False
    trl: bool = False
    accelerate: bool = False
    bitsandbytes: bool = False

    @classmethod
    def detect(cls) -> "DependencyProfile":
        """Detect available dependencies."""
        profile = cls()

        packages = [
            "torch",
            "transformers",
            "peft",
            "unsloth",
            "mlx",
            "mlx_lm",
            "trl",
            "accelerate",
            "bitsandbytes",
        ]

        for package in packages:
            try:
                import importlib.util

                available = importlib.util.find_spec(package) is not None
                setattr(profile, package, available)
            except Exception:
                pass

        return profile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "torch": self.torch,
            "transformers": self.transformers,
            "peft": self.peft,
            "unsloth": self.unsloth,
            "mlx": self.mlx,
            "mlx_lm": self.mlx_lm,
            "trl": self.trl,
            "accelerate": self.accelerate,
            "bitsandbytes": self.bitsandbytes,
        }


class AutoBackendSelector:
    """Automatic backend selector for optimal training performance.

    This class analyzes the current runtime environment and selects the
    most appropriate training backend based on hardware capabilities,
    available dependencies, and training requirements.
    """

    # Backend priority by hardware
    APPLE_SILICON_PRIORITY = ["mlx", "peft", "mock_local"]
    CUDA_PRIORITY = ["unsloth", "peft", "mock_local"]
    CPU_PRIORITY = ["peft", "mock_local"]

    # Backend support by train type
    TRAIN_TYPE_SUPPORT = {
        "sft": ["mlx", "unsloth", "peft", "mock_local"],
        "dpo": ["peft", "unsloth", "mock_local"],
    }

    # Performance estimates (relative to peft baseline)
    PERFORMANCE_ESTIMATES = {
        "mlx": {"apple_silicon": "5-10x faster", "cuda": "N/A", "cpu": "N/A"},
        "unsloth": {"apple_silicon": "N/A", "cuda": "2-5x faster", "cpu": "N/A"},
        "peft": {"apple_silicon": "baseline", "cuda": "baseline", "cpu": "baseline"},
        "mock_local": {"apple_silicon": "no training", "cuda": "no training", "cpu": "no training"},
    }

    def __init__(
        self,
        hardware: Optional[HardwareProfile] = None,
        dependencies: Optional[DependencyProfile] = None,
    ):
        self.hardware = hardware or HardwareProfile.detect()
        self.dependencies = dependencies or DependencyProfile.detect()

    def get_backend_priority(self, train_type: str = "sft") -> List[str]:
        """Get backend priority list for current hardware and train type.

        Args:
            train_type: Type of training (sft or dpo)

        Returns:
            Ordered list of backend names by priority
        """
        train_type = normalize_train_type(train_type)

        # Get hardware-specific priority
        if self.hardware.is_apple_silicon and self.hardware.mps_available:
            priority = list(self.APPLE_SILICON_PRIORITY)
        elif self.hardware.cuda_available:
            priority = list(self.CUDA_PRIORITY)
        else:
            priority = list(self.CPU_PRIORITY)

        # Filter by train type support
        supported = self.TRAIN_TYPE_SUPPORT.get(train_type, ["peft", "mock_local"])

        # Reorder based on availability
        available = []
        unavailable = []

        for backend in priority:
            if backend in supported:
                if self._is_backend_available(backend):
                    available.append(backend)
                else:
                    unavailable.append(backend)

        # Add any supported backends not in priority list
        for backend in supported:
            if backend not in priority:
                if self._is_backend_available(backend):
                    available.append(backend)
                else:
                    unavailable.append(backend)

        return available + unavailable

    def _is_backend_available(self, backend: str) -> bool:
        """Check if a backend is available given current dependencies.

        Args:
            backend: Backend name

        Returns:
            True if backend is available
        """
        backend = normalize_backend_name(backend)

        if backend == "mock_local":
            return True

        if backend == "mlx":
            return self.dependencies.mlx and self.dependencies.mlx_lm

        if backend == "unsloth":
            return (
                self.dependencies.unsloth
                and self.dependencies.torch
                and self.hardware.cuda_available
            )

        if backend == "peft":
            return (
                self.dependencies.peft
                and self.dependencies.torch
                and self.dependencies.transformers
            )

        return False

    def _get_backend_requirements(self, backend: str) -> List[str]:
        """Get list of requirements for a backend.

        Args:
            backend: Backend name

        Returns:
            List of required package names
        """
        backend = normalize_backend_name(backend)

        requirements_map = {
            "mlx": ["mlx", "mlx_lm"],
            "unsloth": ["unsloth", "torch", "transformers", "trl"],
            "peft": ["torch", "transformers", "peft", "accelerate"],
            "mock_local": [],
        }

        return requirements_map.get(backend, [])

    def _get_missing_requirements(self, backend: str) -> List[str]:
        """Get missing requirements for a backend.

        Args:
            backend: Backend name

        Returns:
            List of missing package names
        """
        requirements = self._get_backend_requirements(backend)
        deps_dict = self.dependencies.to_dict()
        return [req for req in requirements if not deps_dict.get(req, False)]

    def _estimate_performance(self, backend: str) -> str:
        """Estimate performance for a backend on current hardware.

        Args:
            backend: Backend name

        Returns:
            Performance estimate string
        """
        backend = normalize_backend_name(backend)

        if self.hardware.is_apple_silicon:
            return self.PERFORMANCE_ESTIMATES.get(backend, {}).get("apple_silicon", "unknown")
        elif self.hardware.cuda_available:
            return self.PERFORMANCE_ESTIMATES.get(backend, {}).get("cuda", "unknown")
        else:
            return self.PERFORMANCE_ESTIMATES.get(backend, {}).get("cpu", "unknown")

    def select_backend(
        self,
        train_type: str = "sft",
        base_model: Optional[str] = None,
        backend_hint: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> BackendSelectionResult:
        """Select the optimal backend for training.

        Args:
            train_type: Type of training (sft or dpo)
            base_model: Optional base model name for compatibility checking
            backend_hint: Optional user-specified backend preference
            allow_fallback: Whether to allow fallback to mock_local

        Returns:
            BackendSelectionResult with selected backend and metadata
        """
        train_type = normalize_train_type(train_type)

        # Handle explicit backend hint
        if backend_hint:
            backend_hint = normalize_backend_name(backend_hint)
            if backend_hint != "auto":
                if self._is_backend_available(backend_hint):
                    return BackendSelectionResult(
                        selected_backend=backend_hint,
                        confidence=1.0,
                        reason=f"User-specified backend '{backend_hint}' is available",
                        alternatives=[],
                        requirements=self._get_backend_requirements(backend_hint),
                        estimated_performance=self._estimate_performance(backend_hint),
                    )
                elif not allow_fallback:
                    missing = self._get_missing_requirements(backend_hint)
                    raise TrainingError(
                        f"Requested backend '{backend_hint}' is not available. "
                        f"Missing dependencies: {', '.join(missing)}"
                    )
                else:
                    # Will fall through to auto-selection
                    pass

        # Get prioritized list
        priority = self.get_backend_priority(train_type)

        if not priority:
            if allow_fallback:
                return BackendSelectionResult(
                    selected_backend="mock_local",
                    confidence=1.0,
                    reason="No suitable backend found, using mock fallback",
                    alternatives=[],
                    requirements=[],
                    estimated_performance="no training",
                    warnings=["No training backends available"],
                )
            else:
                raise TrainingError("No suitable training backend available")

        # Select best available
        selected = priority[0]
        alternatives = priority[1:]

        # Build warnings
        warnings = []
        if selected == "mock_local" and len(priority) > 1:
            warnings.append("Using mock backend - no actual training will occur")

        # Check model compatibility
        model_warning = self._check_model_compatibility(selected, base_model)
        if model_warning:
            warnings.append(model_warning)

        # Build reason
        if self.hardware.is_apple_silicon and selected == "mlx":
            reason = "MLX selected for optimal Apple Silicon performance"
        elif self.hardware.cuda_available and selected == "unsloth":
            reason = "Unsloth selected for optimal CUDA performance"
        elif selected == "peft":
            reason = "PEFT selected as general-purpose fallback"
        elif selected == "mock_local":
            reason = "Mock backend selected (no training dependencies available)"
        else:
            reason = f"{selected} selected based on hardware and dependencies"

        return BackendSelectionResult(
            selected_backend=selected,
            confidence=0.95 if selected != "mock_local" else 0.5,
            reason=reason,
            alternatives=alternatives,
            requirements=self._get_backend_requirements(selected),
            estimated_performance=self._estimate_performance(selected),
            warnings=warnings,
        )

    def _check_model_compatibility(
        self, backend: str, base_model: Optional[str]
    ) -> Optional[str]:
        """Check if a base model is compatible with a backend.

        Args:
            backend: Backend name
            base_model: Base model name

        Returns:
            Warning message if compatibility issue found, None otherwise
        """
        if not base_model:
            return None

        backend = normalize_backend_name(backend)
        model_lower = base_model.lower()

        # MLX compatibility
        if backend == "mlx":
            mlx_compatible = [
                "mlx-community",
                "mistral",
                "llama",
                "qwen",
                "phi",
                "gemma",
            ]
            if not any(x in model_lower for x in mlx_compatible):
                return f"Model '{base_model}' may not be MLX-compatible"

        # Unsloth compatibility
        if backend == "unsloth":
            unsloth_prefixes = ["unsloth/"]
            if not any(base_model.startswith(x) for x in unsloth_prefixes):
                return f"Consider using unsloth/ models for best performance with Unsloth"

        return None

    def get_recommendation_report(self, train_type: str = "sft") -> Dict[str, Any]:
        """Get a comprehensive recommendation report.

        Args:
            train_type: Type of training

        Returns:
            Dictionary with full recommendation details
        """
        selection = self.select_backend(train_type)

        # Get all backends with their status
        all_backends = {}
        for backend in ["mlx", "unsloth", "peft", "mock_local"]:
            available = self._is_backend_available(backend)
            all_backends[backend] = {
                "available": available,
                "missing_requirements": (
                    [] if available else self._get_missing_requirements(backend)
                ),
                "estimated_performance": self._estimate_performance(backend),
            }

        return {
            "recommendation": selection.to_dict(),
            "hardware": self.hardware.to_dict(),
            "dependencies": self.dependencies.to_dict(),
            "all_backends": all_backends,
            "priority_order": self.get_backend_priority(train_type),
        }


def select_optimal_backend(
    train_type: str = "sft",
    base_model: Optional[str] = None,
    backend_hint: Optional[str] = None,
    runtime: Optional[Mapping[str, Any]] = None,
) -> str:
    """Select the optimal backend for training.

    This is a convenience function for simple backend selection.

    Args:
        train_type: Type of training (sft or dpo)
        base_model: Optional base model name
        backend_hint: Optional user preference
        runtime: Optional runtime context dictionary

    Returns:
        Selected backend name
    """
    # Use runtime context if provided
    if runtime:
        hardware = HardwareProfile(
            platform=runtime.get("platform_name", "").lower(),
            machine=runtime.get("machine", "").lower(),
            is_apple_silicon=runtime.get("apple_silicon", False),
            cuda_available=runtime.get("cuda_available", False),
            mps_available=runtime.get("mps_available", False),
        )

        packages = runtime.get("installed_packages", {})
        dependencies = DependencyProfile(
            torch=packages.get("torch", False),
            transformers=packages.get("transformers", False),
            peft=packages.get("peft", False),
            unsloth=packages.get("unsloth", False),
            mlx=packages.get("mlx", False),
            mlx_lm=packages.get("mlx_lm", False),
            trl=packages.get("trl", False),
            accelerate=packages.get("accelerate", False),
        )

        selector = AutoBackendSelector(hardware=hardware, dependencies=dependencies)
    else:
        selector = AutoBackendSelector()

    result = selector.select_backend(
        train_type=train_type,
        base_model=base_model,
        backend_hint=backend_hint,
    )

    return result.selected_backend


def get_backend_selection_summary(
    train_type: str = "sft",
    runtime: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Get a summary of backend selection for the current environment.

    Args:
        train_type: Type of training
        runtime: Optional runtime context

    Returns:
        Dictionary with selection summary
    """
    if runtime:
        hardware = HardwareProfile(
            platform=runtime.get("platform_name", "").lower(),
            machine=runtime.get("machine", "").lower(),
            is_apple_silicon=runtime.get("apple_silicon", False),
            cuda_available=runtime.get("cuda_available", False),
            mps_available=runtime.get("mps_available", False),
        )

        packages = runtime.get("installed_packages", {})
        dependencies = DependencyProfile(
            torch=packages.get("torch", False),
            transformers=packages.get("transformers", False),
            peft=packages.get("peft", False),
            unsloth=packages.get("unsloth", False),
            mlx=packages.get("mlx", False),
            mlx_lm=packages.get("mlx_lm", False),
            trl=packages.get("trl", False),
            accelerate=packages.get("accelerate", False),
        )

        selector = AutoBackendSelector(hardware=hardware, dependencies=dependencies)
    else:
        selector = AutoBackendSelector()

    return selector.get_recommendation_report(train_type)
