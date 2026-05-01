"""Doctor trainer package readiness formatting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .doctor_formatting_deps import DoctorFormattingDeps


def _format_doctor_trainer_deps(runtime: Any, deps: DoctorFormattingDeps) -> str | None:
    mapping = deps.coerce_mapping(runtime)
    if mapping is None:
        return None

    installed_packages = deps.coerce_mapping(mapping.get("installed_packages")) or {}
    if not installed_packages:
        return None

    required_packages = ("torch", "transformers", "peft", "accelerate", "trl", "datasets")
    optional_packages = ("unsloth", "mlx", "mlx_lm")
    ready = all(bool(installed_packages.get(name, False)) for name in required_packages)

    parts = [f"ready={deps.format_scalar(ready)}"]
    missing = [name for name in required_packages if not installed_packages.get(name, False)]
    if missing:
        parts.append(f"missing={deps.format_scalar(missing)}")
    for name in (*required_packages, *optional_packages):
        if name in installed_packages:
            parts.append(f"{name}={deps.format_scalar(installed_packages.get(name))}")
    python_version = deps.pick_first(mapping, "python_version")
    if python_version is not None:
        parts.append(f"python_version={deps.format_scalar(python_version)}")
    requires_python = _format_doctor_package_mapping(
        mapping.get("requires_python"),
        preferred_order=(*required_packages, *optional_packages),
        deps=deps,
    )
    if requires_python is not None:
        parts.append(f"requires_python={requires_python}")
    python_supported = _format_doctor_package_mapping(
        mapping.get("python_supported"),
        preferred_order=(*required_packages, *optional_packages),
        deps=deps,
    )
    if python_supported is not None:
        parts.append(f"python_supported={python_supported}")
    runtime_device = deps.pick_first(mapping, "runtime_device")
    if runtime_device is not None:
        parts.append(f"runtime_device={deps.format_scalar(runtime_device)}")
    return "trainer deps: " + " | ".join(parts)


def _format_doctor_package_mapping(
    mapping: Any,
    *,
    preferred_order: Sequence[str],
    deps: DoctorFormattingDeps,
) -> str | None:
    coerced = deps.coerce_mapping(mapping)
    if coerced is None:
        return None

    parts: list[str] = []
    seen: set[str] = set()
    for name in preferred_order:
        if name in coerced:
            parts.append(f"{name}={deps.format_scalar(coerced.get(name))}")
            seen.add(name)
    for name in sorted(coerced):
        if name not in seen:
            parts.append(f"{name}={deps.format_scalar(coerced.get(name))}")
    return ", ".join(parts) if parts else None


__all__ = ["_format_doctor_package_mapping", "_format_doctor_trainer_deps"]
