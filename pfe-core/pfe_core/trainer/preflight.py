"""Training preflight checks before launching real training jobs."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


class TrainingPreflight:
    def __init__(self, job_spec: Mapping[str, Any]) -> None:
        self.job_spec = dict(job_spec)
        recipe = dict(job_spec.get("recipe") or {})
        training_recipe = dict(recipe.get("training") or {})
        self.backend = str(
            job_spec.get("execution_executor")
            or job_spec.get("backend")
            or "mock_local"
        )
        self.base_model = job_spec.get("base_model") or training_recipe.get("base_model") or ""
        self.workspace = job_spec.get("workspace") or "user_default"

    @staticmethod
    def _available_memory_gb() -> float | None:
        try:
            import psutil

            return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return None

    @staticmethod
    def _local_model_weight_size_gb(base_model: str) -> float | None:
        if not base_model:
            return None
        model_path = Path(base_model).expanduser()
        if not model_path.exists():
            return None
        if model_path.is_file():
            return model_path.stat().st_size / (1024**3)

        direct_weights = [
            path
            for pattern in ("*.safetensors", "*.bin")
            for path in model_path.glob(pattern)
            if path.is_file()
        ]
        weight_files = direct_weights or [
            path
            for pattern in ("**/*.safetensors", "**/*.bin")
            for path in model_path.glob(pattern)
            if path.is_file()
        ]
        if not weight_files:
            return None
        return sum(path.stat().st_size for path in weight_files) / (1024**3)

    def _mlx_memory_check(self) -> dict[str, Any]:
        available_gb = self._available_memory_gb()
        model_size_gb = self._local_model_weight_size_gb(str(self.base_model))
        if available_gb is None or model_size_gb is None:
            return {
                "ok": True,
                "status": "skipped",
                "available_gb": None if available_gb is None else round(available_gb, 2),
                "model_size_gb": None if model_size_gb is None else round(model_size_gb, 2),
                "reason": "memory_or_local_model_size_unavailable",
            }

        required_gb = max(2.0, model_size_gb * 1.5)
        ok = available_gb >= required_gb
        return {
            "ok": ok,
            "status": "ok" if ok else "blocked",
            "available_gb": round(available_gb, 2),
            "model_size_gb": round(model_size_gb, 2),
            "estimated_required_gb": round(required_gb, 2),
        }

    def check(self) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        reasons: list[str] = []

        # 1. Python version
        py_ok = sys.version_info >= (3, 10)
        checks["python_version"] = {"ok": py_ok, "version": sys.version.split()[0]}
        if not py_ok:
            reasons.append("python_version_too_low")

        # 2. Backend allowlist
        allowed = {"mock_local", "mlx", "peft", "unsloth", "dpo"}
        backend_ok = self.backend in allowed
        checks["backend_allowed"] = {"ok": backend_ok, "backend": self.backend}
        if not backend_ok:
            reasons.append(f"backend_not_allowed:{self.backend}")

        # 3. Real training environment variable
        real_enabled = os.getenv("PFE_REAL_TRAINING", "").lower() in ("1", "true", "yes")
        checks["real_training_env"] = {"ok": real_enabled}
        if not real_enabled and self.backend != "mock_local":
            reasons.append("real_training_disabled")

        # 4. Base model exists
        model_ok = False
        if self.base_model:
            model_path = Path(self.base_model).expanduser()
            model_ok = model_path.exists() or bool(self.base_model.strip())
        checks["base_model"] = {"ok": model_ok, "path": self.base_model}
        if not model_ok and self.backend != "mock_local":
            reasons.append("missing_base_model")

        # 5. Dependency imports
        dep_checks = {}
        if self.backend == "mlx":
            if importlib.util.find_spec("mlx") is not None and importlib.util.find_spec("mlx.core") is not None:
                dep_checks["mlx_import"] = {"ok": True}
            else:
                dep_checks["mlx_import"] = {"ok": False}
                reasons.append("mlx_import_failed")
        elif self.backend == "peft":
            if all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft")):
                dep_checks["peft_import"] = {"ok": True}
            else:
                dep_checks["peft_import"] = {"ok": False}
                reasons.append("peft_import_failed")
        elif self.backend == "dpo":
            if all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft", "trl")):
                dep_checks["dpo_import"] = {"ok": True}
            else:
                dep_checks["dpo_import"] = {"ok": False}
                reasons.append("dpo_import_failed")
        elif self.backend == "unsloth":
            if importlib.util.find_spec("unsloth") is not None:
                dep_checks["unsloth_import"] = {"ok": True}
            else:
                dep_checks["unsloth_import"] = {"ok": False}
                reasons.append("unsloth_import_failed")
        checks["dependencies"] = dep_checks

        # 6. Output directory writable
        output_dir = Path(self.job_spec.get("output_dir", "~/.pfe/adapters") or "~/.pfe/adapters").expanduser()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            writable = os.access(output_dir, os.W_OK)
        except Exception:
            writable = False
        checks["output_writable"] = {"ok": writable, "path": str(output_dir)}
        if not writable:
            reasons.append("output_not_writable")

        # 7. Disk space (require 5GB free)
        try:
            stat = shutil.disk_usage(str(output_dir))
            free_gb = stat.free / (1024 ** 3)
            disk_ok = free_gb >= 5.0
        except Exception:
            free_gb = 0.0
            disk_ok = False
        checks["disk_space"] = {"ok": disk_ok, "free_gb": round(free_gb, 2)}
        if not disk_ok:
            reasons.append("insufficient_disk_space")

        if self.backend == "mlx":
            memory_check = self._mlx_memory_check()
            checks["memory"] = memory_check
            if not memory_check.get("ok", True):
                reasons.append("insufficient_memory")

        # 8. Training samples > 0
        samples = list(self.job_spec.get("training_examples") or [])
        samples_ok = len(samples) > 0
        checks["training_samples"] = {"ok": samples_ok, "count": len(samples)}
        if not samples_ok:
            reasons.append("no_training_samples")

        # 9. MLX smoke (only for mlx backend). GPU smoke can abort the process
        # on some Metal failures, so keep it opt-in for the parent preflight.
        if self.backend == "mlx":
            if os.getenv("PFE_MLX_PREFLIGHT_SMOKE", "").lower() in ("1", "true", "yes"):
                from .mlx_backend import mlx_smoke_check

                smoke = mlx_smoke_check()
            else:
                smoke = {
                    "status": "skipped",
                    "check": "mlx_gpu_smoke",
                    "reason": "gpu_smoke_runs_in_training_subprocess",
                }
            checks["mlx_smoke"] = smoke
            if smoke.get("status") != "ok":
                if smoke.get("status") != "skipped":
                    reasons.append(smoke.get("reason", "mlx_smoke_failed"))

        status = "ok" if not reasons else "blocked"
        return {
            "status": status,
            "ready": status == "ok",
            "stage": "preflight",
            "backend": self.backend,
            "reasons": reasons,
            "checks": checks,
        }
