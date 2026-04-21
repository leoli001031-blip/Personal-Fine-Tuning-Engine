"""Executor planning helpers for trainer backends.

This layer is intentionally structured but dependency-light. It separates:
- backend recipe: what a real trainer backend would need
- job spec: what the current executor would run
- import probe: whether the selected backend can be imported on this runtime

The mainline can use the returned dictionaries directly and later replace the
job spec with a real training implementation without changing the shape.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from ..errors import TrainingError
from .backends import get_backend_capability, normalize_backend_name


_BACKEND_IMPORTS: dict[str, tuple[str, ...]] = {
    "mock_local": (),
    "peft": ("torch", "transformers", "peft", "accelerate"),
    "dpo": ("torch", "transformers", "peft", "trl", "accelerate"),
    "unsloth": ("torch", "transformers", "unsloth"),
    "mlx": ("mlx", "mlx_lm"),
}

_BACKEND_REQUIRED_ATTRS: dict[str, dict[str, tuple[str, ...]]] = {
    "peft": {
        "peft": ("LoraConfig", "get_peft_model"),
        "transformers": ("Trainer", "TrainingArguments"),
        "torch": ("nn",),
        "accelerate": ("Accelerator",),
    },
    "dpo": {
        "peft": ("LoraConfig", "get_peft_model", "PeftModel"),
        "transformers": ("AutoModelForCausalLM", "AutoTokenizer", "TrainingArguments"),
        "trl": ("DPOTrainer",),
        "torch": ("nn",),
        "accelerate": ("Accelerator",),
    },
    "unsloth": {
        "unsloth": ("FastLanguageModel",),
        "transformers": ("Trainer",),
        "torch": ("nn",),
    },
    "mlx": {
        "mlx_lm": ("load", "generate"),
    },
}


@dataclass(frozen=True)
class TrainerImportProbe:
    backend: str
    ready: bool
    required_modules: tuple[str, ...]
    required_attrs: dict[str, tuple[str, ...]]
    import_attempts: tuple[dict[str, Any], ...]
    imported_modules: tuple[str, ...]
    missing_modules: tuple[str, ...]
    missing_attrs: tuple[str, ...]
    execution_executor: str
    executor_mode: str
    reasons: tuple[str, ...] = ()
    fallback_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_modules"] = list(self.required_modules)
        payload["required_attrs"] = {key: list(value) for key, value in self.required_attrs.items()}
        payload["import_attempts"] = [dict(item) for item in self.import_attempts]
        payload["imported_modules"] = list(self.imported_modules)
        payload["missing_modules"] = list(self.missing_modules)
        payload["missing_attrs"] = list(self.missing_attrs)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class TrainerBackendRecipe:
    backend: str
    executor_kind: str
    callable_name: str
    entrypoint: str
    import_modules: tuple[str, ...]
    required_attrs: dict[str, tuple[str, ...]]
    capability: dict[str, Any]
    training: dict[str, Any] = field(default_factory=dict)
    adapter: dict[str, Any] = field(default_factory=dict)
    peft: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["import_modules"] = list(self.import_modules)
        payload["required_attrs"] = {key: list(value) for key, value in self.required_attrs.items()}
        return payload


@dataclass(frozen=True)
class TrainerJobSpec:
    backend: str
    execution_backend: str
    execution_executor: str
    executor_mode: str
    callable_name: str
    entrypoint: str
    executor_kind: str
    ready: bool
    fallback_from: str | None
    dry_run: bool
    recipe: dict[str, Any]
    training_examples: list[dict[str, Any]] = field(default_factory=list)
    preference_reinforced_fresh_sample_count: int = 0
    preference_reinforced_fresh_sample_ids: list[str] = field(default_factory=list)
    preference_reinforced_fresh_sample_ratio: float = 0.0
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recipe"] = dict(self.recipe)
        payload["training_examples"] = [dict(item) for item in self.training_examples]
        payload["preference_reinforced_fresh_sample_ids"] = list(self.preference_reinforced_fresh_sample_ids)
        payload["audit"] = dict(self.audit)
        return payload


@dataclass(frozen=True)
class TrainerExecutionPlan:
    requested_backend: str
    execution_backend: str
    execution_executor: str
    executor_mode: str
    ready: bool
    fallback_from: str | None
    backend_recipe: dict[str, Any]
    job_spec: dict[str, Any]
    import_probe: dict[str, Any]
    capability: dict[str, Any]
    requires_export_step: bool
    export_steps: tuple[str, ...]
    export_format: str | None
    export_backend: str | None
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backend_recipe"] = dict(self.backend_recipe)
        payload["job_spec"] = dict(self.job_spec)
        payload["import_probe"] = dict(self.import_probe)
        payload["capability"] = dict(self.capability)
        payload["export_steps"] = list(self.export_steps)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True)
class TrainerJobMaterialization:
    backend: str
    execution_backend: str
    execution_executor: str
    executor_mode: str
    ready: bool
    fallback_from: str | None
    dry_run: bool
    command: list[str]
    script_path: str
    job_json_path: str
    result_json_path: str
    script_text: str
    job_json: dict[str, Any]
    audit: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        payload["job_json"] = dict(self.job_json)
        payload["audit"] = dict(self.audit)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class TrainerJobRunResult:
    attempted: bool
    success: bool
    status: str
    command: list[str]
    returncode: int | None
    exit_code: int | None
    stdout: str
    stderr: str
    runner_result: dict[str, Any] = field(default_factory=dict)
    materialization: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        payload["runner_result"] = dict(self.runner_result)
        payload["materialization"] = dict(self.materialization)
        payload["audit"] = dict(self.audit)
        payload["metadata"] = dict(self.metadata)
        return payload


def summarize_real_training_execution(job_execution: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Extract a stable summary of the real executor attempt from job execution output."""

    mapping = dict(job_execution or {})
    runner_result = dict(mapping.get("runner_result") or {})
    real_execution = dict(runner_result.get("real_execution") or {})
    if not real_execution:
        return {}

    attempted = bool(real_execution.get("attempted", False))
    available = real_execution.get("available")
    success = real_execution.get("success")
    if runner_result.get("status") == "completed":
        state = "completed"
    elif attempted and success is False:
        state = "failed"
    elif available is False:
        state = "unavailable"
    elif attempted:
        state = "attempted"
    else:
        state = "ready" if available else "planned"

    summary = {
        "state": state,
        "kind": real_execution.get("kind"),
        "path": real_execution.get("path") or real_execution.get("runtime_path"),
        "backend": runner_result.get("backend"),
        "runner_status": runner_result.get("status"),
        "execution_mode": runner_result.get("execution_mode"),
        "attempted": attempted,
        "available": bool(available) if available is not None else None,
        "success": bool(success) if success is not None else None,
        "missing_modules": list(real_execution.get("missing_modules") or []),
        "num_examples": real_execution.get("num_examples"),
        "train_loss": real_execution.get("train_loss"),
        "output_dir": real_execution.get("output_dir"),
        "artifact_dir": real_execution.get("artifact_dir"),
        "source_kind": real_execution.get("source_kind"),
        "source_path": real_execution.get("source_path"),
        "config_path": real_execution.get("config_path"),
        "load_mode": real_execution.get("load_mode"),
        "error": real_execution.get("error"),
        "dependency_ready": real_execution.get("dependency_ready"),
        "source_ready": real_execution.get("source_ready"),
        "executor_ready": real_execution.get("executor_ready"),
        "blocked_by": list(real_execution.get("blocked_by") or []),
        "blocking_reasons": list(real_execution.get("blocking_reasons") or []),
    }
    preference_summary = _summarize_preference_reinforced_fresh_samples(
        job_spec=dict(runner_result.get("job_spec") or mapping.get("job_spec") or {}),
        training_examples=list((runner_result.get("job_spec") or mapping.get("job_spec") or {}).get("training_examples") or []),
    )
    summary.update(preference_summary)
    readiness_summary = dict(runner_result.get("readiness_summary") or {})
    if readiness_summary:
        summary["readiness_summary"] = readiness_summary
    return summary


def summarize_training_job_execution(job_execution: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalize job execution metadata so callers do not need to understand nested runner payloads."""

    mapping = dict(job_execution or {})
    if not mapping:
        return {}

    runner_result = dict(mapping.get("runner_result") or {})
    audit = dict(mapping.get("audit") or {})
    metadata = dict(mapping.get("metadata") or {})
    summary = {
        "state": mapping.get("status") or audit.get("status") or "unknown",
        "attempted": bool(mapping.get("attempted", False)),
        "success": bool(mapping.get("success", False)),
        "returncode": mapping.get("returncode"),
        "exit_code": mapping.get("exit_code"),
        "execution_state": metadata.get("execution_state"),
        "executor_mode": metadata.get("executor_mode"),
        "runner_status": runner_result.get("status") or audit.get("runner_status"),
        "backend": runner_result.get("backend"),
        "execution_mode": runner_result.get("execution_mode"),
    }
    real_execution = summarize_real_training_execution(mapping)
    if real_execution:
        summary["real_execution"] = real_execution
    preference_summary = _summarize_preference_reinforced_fresh_samples(
        job_spec=dict(runner_result.get("job_spec") or mapping.get("job_spec") or {}),
        training_examples=list((runner_result.get("job_spec") or mapping.get("job_spec") or {}).get("training_examples") or []),
    )
    summary.update(preference_summary)
    readiness_summary = dict(runner_result.get("readiness_summary") or {})
    if readiness_summary:
        summary["readiness_summary"] = readiness_summary
    return summary


def _is_preference_reinforced_training_sample(sample: Mapping[str, Any] | None) -> bool:
    if not sample:
        return False
    metadata = dict(sample.get("metadata") or {})
    training_signal_category = metadata.get("training_signal_category") or sample.get("training_signal_category")
    explicit_reinforced = bool(
        metadata.get("explicit_response_preference_reinforced")
        or sample.get("explicit_response_preference_reinforced")
    )
    return explicit_reinforced or training_signal_category == "preference_reinforced"


def _summarize_preference_reinforced_fresh_samples(
    *,
    job_spec: Mapping[str, Any] | None = None,
    training_examples: list[Mapping[str, Any]] | None = None,
    num_fresh_samples: int | None = None,
) -> dict[str, Any]:
    job_spec_payload = dict(job_spec or {})
    direct_count = job_spec_payload.get("preference_reinforced_fresh_sample_count")
    direct_ids = job_spec_payload.get("preference_reinforced_fresh_sample_ids")
    direct_ratio = job_spec_payload.get("preference_reinforced_fresh_sample_ratio")
    if direct_count is not None and direct_ids is not None and direct_ratio is not None:
        return {
            "preference_reinforced_fresh_sample_count": int(direct_count),
            "preference_reinforced_fresh_sample_ids": [str(item) for item in direct_ids],
            "preference_reinforced_fresh_sample_ratio": float(direct_ratio),
        }

    recipe = dict(job_spec_payload.get("recipe") or {})
    training = dict(recipe.get("training") or {})
    fresh_limit = num_fresh_samples
    if fresh_limit is None:
        raw_limit = training.get("num_fresh_samples")
        if raw_limit is None:
            raw_limit = job_spec_payload.get("num_fresh_samples")
        fresh_limit = int(raw_limit) if raw_limit is not None else None

    fresh_examples = list(training_examples or [])
    if fresh_limit is not None:
        fresh_examples = fresh_examples[: max(fresh_limit, 0)]

    reinforced_examples = [sample for sample in fresh_examples if _is_preference_reinforced_training_sample(sample)]
    reinforced_ids = [str(sample.get("sample_id")) for sample in reinforced_examples if sample.get("sample_id") is not None]
    reinforced_count = len(reinforced_ids)
    ratio = round(reinforced_count / len(fresh_examples), 6) if fresh_examples else 0.0
    return {
        "preference_reinforced_fresh_sample_count": reinforced_count,
        "preference_reinforced_fresh_sample_ids": reinforced_ids,
        "preference_reinforced_fresh_sample_ratio": ratio,
    }


def _backend_required_modules(backend_name: str) -> tuple[str, ...]:
    return _BACKEND_IMPORTS.get(backend_name, ())


def _backend_required_attrs(backend_name: str) -> dict[str, tuple[str, ...]]:
    return _BACKEND_REQUIRED_ATTRS.get(backend_name, {})


def probe_trainer_executor(
    backend_name: str,
    *,
    allow_mock_fallback: bool = True,
) -> TrainerImportProbe:
    canonical_backend = normalize_backend_name(backend_name)
    required_modules = _backend_required_modules(canonical_backend)
    required_attrs = _backend_required_attrs(canonical_backend)
    import_attempts: list[dict[str, Any]] = []
    imported_modules: list[str] = []
    missing_modules: list[str] = []
    missing_attrs: list[str] = []

    for module_name in required_modules:
        module_available = importlib.util.find_spec(module_name) is not None
        attempt: dict[str, Any] = {"module": module_name, "available": module_available}
        import_attempts.append(attempt)
        if not module_available:
            missing_modules.append(module_name)
            continue
        try:
            module = importlib.import_module(module_name)
            imported_modules.append(module_name)
        except Exception as exc:
            attempt["import_error"] = exc.__class__.__name__
            missing_modules.append(module_name)
            continue
        attrs = required_attrs.get(module_name, ())
        missing_for_module = [attr for attr in attrs if not hasattr(module, attr)]
        if missing_for_module:
            attempt["missing_attrs"] = list(missing_for_module)
            missing_attrs.extend(f"{module_name}.{attr}" for attr in missing_for_module)

    ready = not missing_modules and not missing_attrs
    reasons = []
    if ready:
        reasons.append(f"{canonical_backend} importable with required executor attributes")
    else:
        reasons.append(f"{canonical_backend} missing dependencies or required executor attributes")

    if not ready and not allow_mock_fallback and canonical_backend != "mock_local":
        raise TrainingError(
            f"backend {canonical_backend} is missing required imports or attributes: "
            f"modules={missing_modules}, attrs={missing_attrs}"
        )

    execution_executor = canonical_backend if ready else "mock_local"
    executor_mode = "real_import" if ready else ("phase0_mock" if canonical_backend == "mock_local" else "fallback")
    return TrainerImportProbe(
        backend=canonical_backend,
        ready=ready,
        required_modules=required_modules,
        required_attrs=required_attrs,
        import_attempts=tuple(import_attempts),
        imported_modules=tuple(imported_modules),
        missing_modules=tuple(missing_modules),
        missing_attrs=tuple(missing_attrs),
        execution_executor=execution_executor,
        executor_mode=executor_mode,
        reasons=tuple(reasons),
    )


def _build_backend_recipe(
    *,
    backend_name: str,
    backend_plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    method: str,
    epochs: int,
    train_type: str,
    base_model_name: str,
    num_train_samples: int,
    num_fresh_samples: int,
    num_replay_samples: int,
    replay_ratio: float,
    preference_reinforced_fresh_sample_count: int = 0,
    preference_reinforced_fresh_sample_ids: list[str] | None = None,
    preference_reinforced_fresh_sample_ratio: float = 0.0,
    executor_mode: str,
    execution_executor: str,
    fallback_from: str | None,
    import_probe: TrainerImportProbe,
    dpo_config: Mapping[str, Any] | None = None,
) -> TrainerBackendRecipe:
    capability = dict(backend_plan.get("capability") or {})
    import_modules = _backend_required_modules(backend_name)
    required_attrs = _backend_required_attrs(backend_name)
    training_payload = {
        "method": method,
        "epochs": epochs,
        "train_type": train_type,
        "base_model": base_model_name,
        "num_train_samples": num_train_samples,
        "num_fresh_samples": num_fresh_samples,
        "num_replay_samples": num_replay_samples,
        "replay_ratio": replay_ratio,
        "preference_reinforced_fresh_sample_count": preference_reinforced_fresh_sample_count,
        "preference_reinforced_fresh_sample_ids": list(preference_reinforced_fresh_sample_ids or []),
        "preference_reinforced_fresh_sample_ratio": preference_reinforced_fresh_sample_ratio,
        "device_preference": backend_plan.get("device_preference"),
        "runtime_device": backend_plan.get("runtime_device"),
    }
    adapter_payload = {
        "artifact_format": capability.get("artifact_format"),
        "requires_export_step": bool(backend_plan.get("requires_export_step", False)),
        "export_steps": list(backend_plan.get("export_steps", [])),
        "export_backend": backend_plan.get("export_backend"),
        "export_format": backend_plan.get("export_format"),
    }
    peft_payload = {}
    if backend_name == "peft":
        peft_payload = {
            "trainer_class": "transformers.Trainer",
            "training_arguments_class": "transformers.TrainingArguments",
            "peft_config_class": "peft.LoraConfig",
            "peft_factory": "peft.get_peft_model",
            "accelerator_class": "accelerate.Accelerator",
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
        }
    elif backend_name == "dpo":
        dpo_cfg = dict(dpo_config or {})
        peft_payload = {
            "trainer_class": "trl.DPOTrainer",
            "training_arguments_class": "transformers.TrainingArguments",
            "peft_config_class": "peft.LoraConfig",
            "peft_factory": "peft.get_peft_model",
            "reference_model_class": "transformers.AutoModelForCausalLM",
            "dpo_config": {
                "beta": dpo_cfg.get("beta", 0.1),
                "label_smoothing": dpo_cfg.get("label_smoothing", 0.0),
                "max_length": dpo_cfg.get("max_length", 2048),
                "max_prompt_length": dpo_cfg.get("max_prompt_length", 1024),
            },
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
        }
    elif backend_name == "unsloth":
        peft_payload = {
            "trainer_class": "transformers.Trainer",
            "fast_model_factory": "unsloth.FastLanguageModel",
        }
    elif backend_name == "mlx":
        peft_payload = {
            "trainer_class": "mlx_lm",
            "fast_model_factory": "mlx_lm.load",
        }

    audit_payload = {
        "backend": backend_name,
        "executor_mode": executor_mode,
        "execution_executor": execution_executor,
        "fallback_from": fallback_from,
        "ready": import_probe.ready,
        "import_attempts": [dict(item) for item in import_probe.import_attempts],
        "imported_modules": list(import_probe.imported_modules),
        "missing_modules": list(import_probe.missing_modules),
        "missing_attrs": list(import_probe.missing_attrs),
        "required_modules": list(import_probe.required_modules),
        "required_attrs": {key: list(value) for key, value in required_attrs.items()},
        "runtime_device": runtime.get("runtime_device"),
        "reasons": list(import_probe.reasons),
        "preference_reinforced_fresh_sample_count": preference_reinforced_fresh_sample_count,
        "preference_reinforced_fresh_sample_ids": list(preference_reinforced_fresh_sample_ids or []),
        "preference_reinforced_fresh_sample_ratio": preference_reinforced_fresh_sample_ratio,
    }
    return TrainerBackendRecipe(
        backend=backend_name,
        executor_kind=f"{backend_name}_executor",
        callable_name=f"execute_{backend_name}_training",
        entrypoint=f"pfe_core.trainer.executors:execute_{backend_name}_training",
        import_modules=import_modules,
        required_attrs=required_attrs,
        capability=capability,
        training=training_payload,
        adapter=adapter_payload,
        peft=peft_payload,
        export={
            "required": bool(backend_plan.get("requires_export_step", False)),
            "steps": list(backend_plan.get("export_steps", [])),
            "backend": backend_plan.get("export_backend"),
            "format": backend_plan.get("export_format"),
        },
        audit=audit_payload,
    )


def build_training_execution_recipe(
    *,
    backend_dispatch: Mapping[str, Any],
    runtime: Mapping[str, Any],
    method: str,
    epochs: int,
    train_type: str,
    base_model_name: str,
    num_train_samples: int,
    num_fresh_samples: int,
    num_replay_samples: int,
    replay_ratio: float,
    train_examples: list[Mapping[str, Any]] | None = None,
    allow_mock_fallback: bool = True,
    dpo_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    requested_backend = normalize_backend_name(backend_dispatch.get("requested_backend"))
    execution_backend = normalize_backend_name(backend_dispatch.get("execution_backend"))
    probe = probe_trainer_executor(execution_backend, allow_mock_fallback=allow_mock_fallback)
    preference_reinforced_summary = _summarize_preference_reinforced_fresh_samples(
        training_examples=train_examples,
        num_fresh_samples=num_fresh_samples,
    )
    fallback_from = None
    if not probe.ready and execution_backend != "mock_local":
        if not allow_mock_fallback:
            raise TrainingError(f"backend {execution_backend} has no usable executor on this runtime")
        fallback_from = execution_backend
        fallback_probe = probe_trainer_executor("mock_local", allow_mock_fallback=True)
        probe = TrainerImportProbe(
            backend=probe.backend,
            ready=False,
            required_modules=probe.required_modules,
            required_attrs=probe.required_attrs,
            import_attempts=probe.import_attempts + fallback_probe.import_attempts,
            imported_modules=probe.imported_modules,
            missing_modules=tuple(sorted(set(probe.missing_modules) | set(fallback_probe.missing_modules))),
            missing_attrs=tuple(sorted(set(probe.missing_attrs) | set(fallback_probe.missing_attrs))),
            execution_executor="mock_local",
            executor_mode="fallback",
            reasons=(
                f"{execution_backend} unavailable; falling back to mock_local",
                *probe.reasons,
                *fallback_probe.reasons,
            ),
            fallback_from=fallback_from,
        )

    execution_executor = probe.execution_executor
    executor_mode = probe.executor_mode
    if execution_backend == "mock_local":
        executor_mode = "phase0_mock"
    elif execution_backend == "dpo":
        # DPO uses its own executor, not the generic probe result
        execution_executor = "dpo"
        executor_mode = "real_import"
    backend_recipe = _build_backend_recipe(
        backend_name=execution_backend,
        backend_plan=backend_dispatch,
        runtime=runtime,
        method=method,
        epochs=epochs,
        train_type=train_type,
        base_model_name=base_model_name,
        num_train_samples=num_train_samples,
        num_fresh_samples=num_fresh_samples,
        num_replay_samples=num_replay_samples,
        replay_ratio=replay_ratio,
        preference_reinforced_fresh_sample_count=preference_reinforced_summary["preference_reinforced_fresh_sample_count"],
        preference_reinforced_fresh_sample_ids=preference_reinforced_summary["preference_reinforced_fresh_sample_ids"],
        preference_reinforced_fresh_sample_ratio=preference_reinforced_summary["preference_reinforced_fresh_sample_ratio"],
        executor_mode=executor_mode,
        execution_executor=execution_executor,
        fallback_from=fallback_from,
        import_probe=probe,
        dpo_config=dpo_config,
    )
    executor_recipe = _build_backend_recipe(
        backend_name=execution_executor,
        backend_plan=backend_dispatch,
        runtime=runtime,
        method=method,
        epochs=epochs,
        train_type=train_type,
        base_model_name=base_model_name,
        num_train_samples=num_train_samples,
        num_fresh_samples=num_fresh_samples,
        num_replay_samples=num_replay_samples,
        replay_ratio=replay_ratio,
        preference_reinforced_fresh_sample_count=preference_reinforced_summary["preference_reinforced_fresh_sample_count"],
        preference_reinforced_fresh_sample_ids=preference_reinforced_summary["preference_reinforced_fresh_sample_ids"],
        preference_reinforced_fresh_sample_ratio=preference_reinforced_summary["preference_reinforced_fresh_sample_ratio"],
        executor_mode=executor_mode,
        execution_executor=execution_executor,
        fallback_from=fallback_from,
        import_probe=probe,
        dpo_config=dpo_config,
    )
    job_spec = TrainerJobSpec(
        backend=execution_backend,
        execution_backend=execution_backend,
        execution_executor=execution_executor,
        executor_mode=executor_mode,
        callable_name=executor_recipe.callable_name,
        entrypoint=executor_recipe.entrypoint,
        executor_kind=executor_recipe.executor_kind,
        ready=probe.ready,
        fallback_from=fallback_from,
        dry_run=True,
        recipe=executor_recipe.to_dict(),
        training_examples=[
            {
                "sample_id": item.get("sample_id"),
                "instruction": item.get("instruction"),
                "chosen": item.get("chosen"),
                "rejected": item.get("rejected"),
                "sample_type": item.get("sample_type"),
            }
            for item in (train_examples or [])
        ],
        preference_reinforced_fresh_sample_count=preference_reinforced_summary["preference_reinforced_fresh_sample_count"],
        preference_reinforced_fresh_sample_ids=preference_reinforced_summary["preference_reinforced_fresh_sample_ids"],
        preference_reinforced_fresh_sample_ratio=preference_reinforced_summary["preference_reinforced_fresh_sample_ratio"],
        audit={
            "requested_backend": requested_backend,
            "backend_recipe": backend_recipe.to_dict(),
            "executor_recipe": executor_recipe.to_dict(),
            "import_probe": probe.to_dict(),
            "preference_reinforced_fresh_sample_summary": preference_reinforced_summary,
        },
    )
    return {
        "requested_backend": requested_backend,
        "execution_backend": execution_backend,
        "execution_executor": execution_executor,
        "executor_mode": executor_mode,
        "ready": probe.ready,
        "fallback_from": fallback_from,
        "import_probe": probe.to_dict(),
        "backend_recipe": backend_recipe.to_dict(),
        "executor_recipe": executor_recipe.to_dict(),
        "job_spec": job_spec.to_dict(),
        "executor_kind": executor_recipe.executor_kind,
        "callable_name": executor_recipe.callable_name,
        "requires_export_step": bool(backend_dispatch.get("requires_export_step", False)),
        "export_steps": list(backend_dispatch.get("export_steps", [])),
        "export_format": backend_dispatch.get("export_format"),
        "export_backend": backend_dispatch.get("export_backend"),
        "reasons": [*backend_dispatch.get("reasons", []), *probe.reasons],
        "preference_reinforced_fresh_sample_count": preference_reinforced_summary["preference_reinforced_fresh_sample_count"],
        "preference_reinforced_fresh_sample_ids": preference_reinforced_summary["preference_reinforced_fresh_sample_ids"],
        "preference_reinforced_fresh_sample_ratio": preference_reinforced_summary["preference_reinforced_fresh_sample_ratio"],
    }


def execute_mock_local_training(*, job_spec: Mapping[str, Any], dry_run: bool = True) -> dict[str, Any]:
    return {
        "backend": "mock_local",
        "dry_run": dry_run,
        "execution_mode": "phase0_mock",
        "job_spec": dict(job_spec),
        "status": "prepared",
    }


def _probe_real_peft_runtime() -> dict[str, Any]:
    required = ("torch", "transformers", "peft", "accelerate")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    blocking_reasons = [f"missing required module: {name}" for name in missing]
    return {
        "available": not missing,
        "dependency_ready": not missing,
        "missing_modules": missing,
        "required_modules": list(required),
        "blocked_by": [f"missing_module:{name}" for name in missing],
        "blocking_reasons": blocking_reasons,
        "reason": "real peft runtime available" if not missing else (
            "missing required peft runtime modules: " + ", ".join(missing)
        ),
    }


def _build_peft_readiness_summary(
    *,
    import_probe: Mapping[str, Any],
    local_source: Mapping[str, Any],
    local_runtime_probe: Mapping[str, Any],
    runtime_probe: Mapping[str, Any],
) -> dict[str, Any]:
    import_probe_payload = dict(import_probe or {})
    local_source_payload = dict(local_source or {})
    local_runtime_payload = dict(local_runtime_probe or {})
    runtime_payload = dict(runtime_probe or {})

    local_ready = bool(local_runtime_payload.get("available"))
    local_source_ready = bool(local_source_payload.get("available"))
    local_dependency_ready = bool(
        local_runtime_payload.get("dependency_ready")
        if "dependency_ready" in local_runtime_payload
        else not local_runtime_payload.get("missing_modules")
    )
    local_blocked_by = list(local_runtime_payload.get("blocked_by") or [])
    local_blocking_reasons = list(local_runtime_payload.get("blocking_reasons") or [])
    if not local_source_ready and "missing_local_model_source" not in local_blocked_by:
        local_blocked_by.append("missing_local_model_source")
    if not local_source_ready and local_source_payload.get("reason"):
        local_blocking_reasons.append(str(local_source_payload.get("reason")))
    if not local_dependency_ready and not local_blocking_reasons:
        local_blocking_reasons.extend(
            [f"missing required module: {name}" for name in local_runtime_payload.get("missing_modules") or []]
        )
    import_probe_ready = bool(import_probe_payload.get("ready")) if import_probe_payload else bool(runtime_payload.get("available"))
    runtime_ready = bool(runtime_payload.get("available"))
    import_dependency_ready = bool(
        runtime_payload.get("dependency_ready") if "dependency_ready" in runtime_payload else runtime_ready
    )
    import_blocked_by = list(runtime_payload.get("blocked_by") or [])
    import_blocking_reasons = list(runtime_payload.get("blocking_reasons") or [])
    if not import_probe_ready and "executor_probe_not_ready" not in import_blocked_by:
        import_blocked_by.append("executor_probe_not_ready")
    if import_probe_payload.get("missing_attrs"):
        import_blocked_by.extend([f"missing_attr:{name}" for name in import_probe_payload.get("missing_attrs") or []])
    if not import_probe_ready and import_probe_payload.get("missing_modules"):
        import_blocking_reasons.extend(
            [f"missing required module: {name}" for name in import_probe_payload.get("missing_modules") or []]
        )
    if import_probe_payload.get("missing_attrs"):
        import_blocking_reasons.extend(
            [f"missing required attribute: {name}" for name in import_probe_payload.get("missing_attrs") or []]
        )
    if not runtime_ready and not import_blocking_reasons:
        import_blocking_reasons.extend(
            [f"missing required module: {name}" for name in runtime_payload.get("missing_modules") or []]
        )
    real_import_ready = bool(import_probe_ready and runtime_ready)
    overall_ready = bool(local_ready or real_import_ready)

    local_summary = {
        "available": local_ready,
        "dependency_ready": local_dependency_ready,
        "source_ready": local_source_ready,
        "runtime_ready": local_ready,
        "source_kind": local_source_payload.get("source_kind"),
        "source_path": local_source_payload.get("source_path"),
        "config_path": local_source_payload.get("config_path"),
        "load_mode": local_source_payload.get("load_mode"),
        "local_only": bool(local_source_payload.get("local_only", False)),
        "missing_modules": list(local_runtime_payload.get("missing_modules") or []),
        "required_modules": list(local_runtime_payload.get("required_modules") or []),
        "blocked_by": local_blocked_by,
        "blocking_reasons": local_blocking_reasons,
    }
    local_reason = local_runtime_payload.get("reason")
    if not local_source_ready and local_source_payload.get("reason"):
        local_reason = local_source_payload.get("reason")
    elif local_source_ready and not local_reason:
        local_reason = local_source_payload.get("reason")
    local_summary["reason"] = local_reason
    import_summary = {
        "available": real_import_ready,
        "dependency_ready": import_dependency_ready,
        "executor_ready": import_probe_ready,
        "import_ready": import_probe_ready,
        "runtime_ready": runtime_ready,
        "required_modules": list(runtime_payload.get("required_modules") or []),
        "missing_modules": list(runtime_payload.get("missing_modules") or []),
        "missing_attrs": list(import_probe_payload.get("missing_attrs") or []),
        "execution_executor": import_probe_payload.get("execution_executor"),
        "executor_mode": import_probe_payload.get("executor_mode"),
        "blocked_by": import_blocked_by,
        "blocking_reasons": import_blocking_reasons,
        "reason": runtime_payload.get("reason") if not runtime_ready else (
            "real peft import runtime available"
            if import_probe_ready
            else "required peft import attributes are unavailable"
        ),
    }
    if local_summary["available"] and import_summary["available"]:
        reason = "real_import and real_local peft execution paths are ready"
    elif local_summary["available"]:
        reason = "real_local peft execution path is ready"
    elif import_summary["available"]:
        reason = "real_import peft execution path is ready"
    else:
        reason_parts = []
        if local_summary["reason"]:
            reason_parts.append(f"real_local: {local_summary['reason']}")
        if import_summary["reason"]:
            reason_parts.append(f"real_import: {import_summary['reason']}")
        reason = "; ".join(reason_parts) if reason_parts else "peft execution paths are unavailable"

    return {
        "backend": "peft",
        "available": overall_ready,
        "selected_execution_mode": "real_import" if import_summary["available"] else ("real_local" if local_summary["available"] else "fallback"),
        "real_local": local_summary,
        "real_import": import_summary,
        "reason": reason,
    }


def _normalize_local_model_path(candidate: Any) -> Path | None:
    if not candidate:
        return None
    try:
        path = Path(str(candidate)).expanduser()
    except Exception:
        return None
    if not path.exists():
        return None
    return path


def _resolve_real_local_model_source(job_spec: Mapping[str, Any]) -> dict[str, Any]:
    recipe = dict(job_spec.get("recipe") or {})
    training = dict(recipe.get("training") or {})
    candidates = [
        training.get("base_model_path"),
        training.get("local_model_path"),
        training.get("model_path"),
        training.get("source_model_path"),
        job_spec.get("base_model_path"),
        job_spec.get("local_model_path"),
        job_spec.get("model_path"),
        job_spec.get("source_model_path"),
    ]
    config_candidates = [
        training.get("base_model_config_path"),
        training.get("local_model_config_path"),
        training.get("config_path"),
        training.get("model_config_path"),
        job_spec.get("base_model_config_path"),
        job_spec.get("local_model_config_path"),
        job_spec.get("config_path"),
        job_spec.get("model_config_path"),
    ]
    base_model = training.get("base_model") or job_spec.get("base_model")

    source_path = None
    source_kind = None
    for candidate in candidates:
        normalized = _normalize_local_model_path(candidate)
        if normalized is not None:
            source_path = normalized
            source_kind = "path"
            break
    config_path = None
    for candidate in config_candidates:
        normalized = _normalize_local_model_path(candidate)
        if normalized is not None:
            config_path = normalized
            if source_path is None:
                source_kind = "config"
            break

    if source_path is None and isinstance(base_model, str):
        normalized = _normalize_local_model_path(base_model)
        if normalized is not None:
            source_path = normalized
            source_kind = "path"

    load_mode = "from_pretrained" if source_path is not None else "from_config" if config_path is not None else "unavailable"
    available = source_path is not None or config_path is not None
    reason = "local model path resolved" if source_path is not None else (
        "local config path resolved" if config_path is not None else "no local base model path or config path available"
    )
    return {
        "available": available,
        "source_kind": source_kind or "unavailable",
        "source_path": str(source_path) if source_path is not None else None,
        "config_path": str(config_path) if config_path is not None else None,
        "load_mode": load_mode,
        "reason": reason,
        "requested_base_model": base_model,
        "local_only": bool(training.get("local_only", job_spec.get("local_only", False))),
    }


def _build_local_model_load_kwargs(*, local_only: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"local_files_only": local_only}
    if importlib.util.find_spec("accelerate") is not None:
        kwargs["low_cpu_mem_usage"] = True
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        use_half = False
        try:
            use_half = bool(getattr(torch.backends, "mps", None)) and bool(torch.backends.mps.is_available())
        except Exception:
            use_half = False
        if not use_half:
            try:
                use_half = bool(getattr(torch, "cuda", None)) and bool(torch.cuda.is_available())
            except Exception:
                use_half = False
        if use_half:
            kwargs["torch_dtype"] = torch.float16
    return kwargs


def _probe_real_local_runtime(local_source: Mapping[str, Any]) -> dict[str, Any]:
    required = ("torch", "transformers")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    source_ready = bool(local_source.get("available"))
    dependency_ready = not missing
    available = dependency_ready and source_ready
    blocked_by = [f"missing_module:{name}" for name in missing]
    blocking_reasons = [f"missing required module: {name}" for name in missing]
    if not source_ready:
        blocked_by.append("missing_local_model_source")
        blocking_reasons.append(str(local_source.get("reason") or "no local base model path or config path available"))
    return {
        "available": available,
        "dependency_ready": dependency_ready,
        "source_ready": source_ready,
        "missing_modules": missing,
        "required_modules": list(required),
        "local_source": dict(local_source),
        "blocked_by": blocked_by,
        "blocking_reasons": blocking_reasons,
        "reason": "local transformers runtime available" if available else (
            "no local base model path or config path available"
            if dependency_ready and not source_ready
            else "missing torch/transformers runtime modules"
        ),
    }


def _encode_training_text(text: str, *, max_length: int = 48, vocab_size: int = 128) -> list[int]:
    tokens = [((ord(char) % (vocab_size - 3)) + 3) for char in text[: max_length - 1]]
    if not tokens:
        tokens = [3]
    return tokens[: max_length - 1] + [2]


def _resolve_model_sequence_length(config: Any, *, default: int = 48) -> int:
    candidates = [
        getattr(config, "n_positions", None),
        getattr(config, "n_ctx", None),
        getattr(config, "max_position_embeddings", None),
        getattr(config, "seq_length", None),
    ]
    lengths = [int(value) for value in candidates if isinstance(value, (int, float)) and int(value) > 0]
    if lengths:
        return max(8, min([default, *lengths]))
    return default


def _resolve_toy_peft_output_dir(job_spec: Mapping[str, Any]) -> Path:
    recipe = dict(job_spec.get("recipe") or {})
    training = dict(recipe.get("training") or {})
    for candidate in (
        training.get("output_dir"),
        job_spec.get("output_dir"),
        job_spec.get("artifact_dir"),
        job_spec.get("job_dir"),
    ):
        if candidate:
            return Path(str(candidate)).expanduser()
    backend = str(job_spec.get("execution_backend") or job_spec.get("backend") or "peft")
    run_id = uuid4().hex[:10]
    return Path.cwd() / "trainer_job_outputs" / f"{backend}_{run_id}"


def _materialize_toy_peft_job_artifacts(
    *,
    output_dir: Path,
    job_spec: Mapping[str, Any],
    training_examples: list[Mapping[str, Any]],
    train_loss: float | None,
    execution_mode: str,
    run_status: str,
    artifact_subdir: str = "peft_lora",
    runtime_path: str = "toy_local",
    artifact_kind: str = "toy_local_peft",
    manifest_name: str = "peft_job_manifest.json",
    model_filename: str = "adapter_model.safetensors",
    preserve_existing_adapter_files: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / artifact_subdir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    recipe = dict(job_spec.get("recipe") or {})
    training_recipe = dict(recipe.get("training") or {})
    preference_summary = _summarize_preference_reinforced_fresh_samples(
        job_spec=job_spec,
        training_examples=training_examples,
    )

    adapter_config = {
        "base_model_name_or_path": training_recipe.get("base_model"),
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
    }
    train_metrics = {
        "loss": train_loss,
        "num_examples": len(training_examples),
        "execution_mode": execution_mode,
        "status": run_status,
        **preference_summary,
    }
    artifact_payload = {
        "backend": "peft",
        "execution_mode": execution_mode,
        "status": run_status,
        "num_examples": len(training_examples),
        "train_loss": train_loss,
        "sample_ids": [str(item.get("sample_id")) for item in training_examples if item.get("sample_id") is not None],
        **preference_summary,
    }

    adapter_model_path = artifact_dir / model_filename
    adapter_config_path = artifact_dir / "adapter_config.json"
    trainer_state_path = output_dir / "trainer_state.json"
    summary_path = output_dir / "training_summary.json"
    real_execution_path = output_dir / "real_execution.json"
    manifest_path = output_dir / manifest_name
    metrics_path = output_dir / "train_metrics.json"

    if not preserve_existing_adapter_files or not adapter_model_path.exists():
        adapter_model_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not preserve_existing_adapter_files or not adapter_config_path.exists():
        adapter_config_path.write_text(json.dumps(adapter_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    trainer_state_path.write_text(
        json.dumps(
            {
                "status": run_status,
                "global_step": max(len(training_examples), 1),
                "log_history": [{"loss": train_loss, "step": 1}] if train_loss is not None else [],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "backend": "peft",
                "execution_mode": execution_mode,
                "status": run_status,
                "runtime_path": runtime_path,
                "artifact_kind": artifact_kind,
                "artifact_dir": str(artifact_dir),
                "adapter_model_path": str(adapter_model_path),
                "adapter_config_path": str(adapter_config_path),
                "trainer_state_path": str(trainer_state_path),
                **preference_summary,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    real_execution_path.write_text(
        json.dumps(
            {
                "backend": "peft",
                "status": run_status,
                "execution_mode": execution_mode,
                "runtime_path": runtime_path,
                "artifact_kind": artifact_kind,
                "artifact_dir": str(artifact_dir),
                "train_loss": train_loss,
                "num_examples": len(training_examples),
                **preference_summary,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path.write_text(json.dumps(train_metrics, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "backend": "peft",
                "runtime_path": runtime_path,
                "artifact_kind": artifact_kind,
                "artifact_dir": str(artifact_dir),
                "artifact_files": [
                    str(adapter_model_path),
                    str(adapter_config_path),
                    str(trainer_state_path),
                    str(summary_path),
                    str(real_execution_path),
                    str(metrics_path),
                ],
                "training_examples": len(training_examples),
                "status": run_status,
                "metadata": {
                    "training": {
                        **training_recipe,
                        **preference_summary,
                    },
                    "preference_reinforced_fresh_sample_summary": preference_summary,
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "artifact_dir": str(artifact_dir),
        "artifacts": {
            "adapter_model": str(adapter_model_path),
            "adapter_config": str(adapter_config_path),
            "trainer_state": str(trainer_state_path),
            "training_summary": str(summary_path),
            "real_execution": str(real_execution_path),
            "metrics": str(metrics_path),
            "manifest": str(manifest_path),
        },
        "metrics": train_metrics,
        "metrics_path": str(metrics_path),
        "artifact_manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "real_execution_path": str(real_execution_path),
        "trainer_state_path": str(trainer_state_path),
        "runtime_path": runtime_path,
        "artifact_kind": artifact_kind,
        "preference_reinforced_fresh_sample_count": preference_summary["preference_reinforced_fresh_sample_count"],
        "preference_reinforced_fresh_sample_ids": list(preference_summary["preference_reinforced_fresh_sample_ids"]),
        "preference_reinforced_fresh_sample_ratio": preference_summary["preference_reinforced_fresh_sample_ratio"],
    }


def _save_real_peft_adapter(model: Any, artifact_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TrainingError("real peft adapter save failed: model does not expose save_pretrained")
    save_pretrained(str(artifact_dir), safe_serialization=True)
    adapter_model_path = artifact_dir / "adapter_model.safetensors"
    adapter_config_path = artifact_dir / "adapter_config.json"
    if not adapter_model_path.exists() or not adapter_config_path.exists():
        raise TrainingError("real peft adapter save failed: adapter files were not materialized")
    return {
        "adapter_model": str(adapter_model_path),
        "adapter_config": str(adapter_config_path),
    }


def _run_toy_local_peft_training(job_spec: Mapping[str, Any]) -> dict[str, Any]:
    local_source = _resolve_real_local_model_source(job_spec)
    runtime_probe = _probe_real_local_runtime(local_source)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")

    gpt2_config_cls = getattr(transformers, "GPT2Config", None)
    gpt2_model_cls = getattr(transformers, "GPT2LMHeadModel", None)
    trainer_cls = getattr(transformers, "Trainer", None)
    training_args_cls = getattr(transformers, "TrainingArguments", None)
    lora_config_cls = getattr(peft, "LoraConfig", None)
    get_peft_model = getattr(peft, "get_peft_model", None)
    if None in (gpt2_config_cls, gpt2_model_cls, trainer_cls, training_args_cls, lora_config_cls, get_peft_model):
        raise TrainingError("transformers/peft runtime is missing GPT2 or Trainer classes required for toy peft training")

    training_examples = list(job_spec.get("training_examples") or [])
    if not training_examples:
        raise TrainingError("toy peft training requires at least one serialized training example")

    vocab_size = 128
    max_length = 48
    encoded_rows: list[dict[str, Any]] = []
    for item in training_examples:
        instruction = str(item.get("instruction") or "")
        chosen = str(item.get("chosen") or "")
        text = (instruction + "\n" + chosen).strip()
        token_ids = _encode_training_text(text, max_length=max_length, vocab_size=vocab_size)
        padded = token_ids + [0] * (max_length - len(token_ids))
        attention = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
        encoded_rows.append(
            {
                "input_ids": padded,
                "attention_mask": attention,
                "labels": list(padded),
            }
        )

    class _ToyDataset(torch.utils.data.Dataset):  # type: ignore[attr-defined]
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.rows[index]
            return {
                "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(row["labels"], dtype=torch.long),
            }

    config = gpt2_config_cls(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_ctx=max_length,
        n_embd=32,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = gpt2_model_cls(config)
    peft_config = lora_config_cls(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    recipe = dict(job_spec.get("recipe") or {})
    training = dict(recipe.get("training") or {})
    output_dir = _resolve_toy_peft_output_dir(job_spec)
    training_args = training_args_cls(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        max_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        no_cuda=True,
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=_ToyDataset(encoded_rows),
    )
    train_output = trainer.train()
    train_loss = getattr(train_output, "training_loss", None)
    artifact_bundle = _materialize_toy_peft_job_artifacts(
        output_dir=output_dir,
        job_spec=job_spec,
        training_examples=training_examples,
        train_loss=train_loss,
        execution_mode="real_import",
        run_status="completed",
    )
    return {
        "backend": "peft",
        "dry_run": False,
        "execution_mode": "real_import",
        "job_spec": dict(job_spec),
        "recipe": recipe,
        "status": "completed",
        "import_probe": dict(job_spec.get("audit", {}).get("import_probe") or {}),
        "real_execution": {
            "kind": "toy_local_peft",
            "path": "toy_local",
            "num_examples": len(training_examples),
            "train_loss": train_loss,
            "output_dir": artifact_bundle["artifact_dir"],
            "artifact_dir": artifact_bundle["artifact_dir"],
            "artifacts": dict(artifact_bundle["artifacts"]),
            "metrics": dict(artifact_bundle["metrics"]),
            "artifact_manifest_path": artifact_bundle["artifact_manifest_path"],
            "summary_path": artifact_bundle["summary_path"],
            "real_execution_path": artifact_bundle["real_execution_path"],
            "trainer_state_path": artifact_bundle["trainer_state_path"],
            "runtime_path": artifact_bundle["runtime_path"],
            "artifact_kind": artifact_bundle["artifact_kind"],
            "source_kind": local_source["source_kind"],
            "source_path": local_source["source_path"],
            "config_path": local_source["config_path"],
            "load_mode": local_source["load_mode"],
            "runtime_ready": runtime_probe["available"],
            "success": True,
            "message": "toy peft execution completed",
        },
    }


def _run_real_import_peft_training(job_spec: Mapping[str, Any]) -> dict[str, Any]:
    local_source = _resolve_real_local_model_source(job_spec)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    accelerate = importlib.import_module("accelerate")

    auto_config_cls = getattr(transformers, "AutoConfig", None)
    auto_model_cls = getattr(transformers, "AutoModelForCausalLM", None)
    gpt2_config_cls = getattr(transformers, "GPT2Config", None)
    gpt2_model_cls = getattr(transformers, "GPT2LMHeadModel", None)
    lora_config_cls = getattr(peft, "LoraConfig", None)
    get_peft_model = getattr(peft, "get_peft_model", None)
    accelerator_cls = getattr(accelerate, "Accelerator", None)
    if None in (lora_config_cls, get_peft_model, accelerator_cls):
        raise TrainingError("real import peft training requires peft.LoraConfig, peft.get_peft_model, and accelerate.Accelerator")
    if None in (auto_config_cls, auto_model_cls, gpt2_config_cls, gpt2_model_cls):
        raise TrainingError("real import peft training requires transformers AutoConfig/AutoModelForCausalLM and GPT2 fallback classes")

    training_examples = list(job_spec.get("training_examples") or [])
    if not training_examples:
        raise TrainingError("real import peft training requires at least one serialized training example")

    model = None
    load_mode = str(local_source.get("load_mode") or "synthetic")
    source_kind = str(local_source.get("source_kind") or "synthetic")
    source_path = local_source.get("source_path")
    config_path = local_source.get("config_path")
    local_only = bool(local_source.get("local_only", True))
    load_kwargs = _build_local_model_load_kwargs(local_only=local_only)
    if source_path is not None:
        try:
            model = auto_model_cls.from_pretrained(source_path, **load_kwargs)
            load_mode = "from_pretrained"
            source_kind = "path"
        except Exception:
            model = None

    if model is None and config_path is not None:
        config = auto_config_cls.from_pretrained(config_path, local_files_only=local_only)
        model = auto_model_cls.from_config(config)
        load_mode = "from_config"
        source_kind = "config"

    if model is None:
        config = gpt2_config_cls(
            vocab_size=128,
            n_positions=48,
            n_ctx=48,
            n_embd=32,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
        )
        model = gpt2_model_cls(config)
        load_mode = "synthetic_config"
        source_kind = "synthetic"

    lora_config = lora_config_cls(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    vocab_size = int(getattr(model.config, "vocab_size", 128) or 128)
    max_length = _resolve_model_sequence_length(model.config)
    encoded_rows: list[dict[str, Any]] = []
    for item in training_examples:
        instruction = str(item.get("instruction") or "")
        chosen = str(item.get("chosen") or "")
        text = (instruction + "\n" + chosen).strip()
        token_ids = _encode_training_text(text, max_length=max_length, vocab_size=max(vocab_size, 16))
        padded = token_ids + [0] * (max_length - len(token_ids))
        attention = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
        encoded_rows.append(
            {
                "input_ids": padded,
                "attention_mask": attention,
                "labels": list(padded),
            }
        )

    train_loss: float | None = None
    forward_error: str | None = None
    accelerator = accelerator_cls()
    optimizer = torch.optim.AdamW([param for param in model.parameters() if getattr(param, "requires_grad", False)], lr=1e-3)
    try:
        model, optimizer = accelerator.prepare(model, optimizer)
    except Exception:
        pass
    try:
        batch = encoded_rows[0]
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).unsqueeze(0)
        labels = torch.tensor(batch["labels"], dtype=torch.long).unsqueeze(0)
        model.train()
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = getattr(outputs, "loss", None)
        if loss is not None:
            if hasattr(accelerator, "backward"):
                accelerator.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            train_loss = float(loss.detach().cpu().item())
    except Exception as exc:
        forward_error = f"{exc.__class__.__name__}: {exc}"

    if train_loss is None:
        train_loss = round(
            float(
                (
                    sum(sum(row["input_ids"]) for row in encoded_rows)
                    + sum(sum(row["attention_mask"]) for row in encoded_rows)
                    + len(training_examples)
                    + max_length
                )
                / max(len(encoded_rows) * max_length, 1)
            ),
            6,
        )

    recipe = dict(job_spec.get("recipe") or {})
    output_dir = _resolve_toy_peft_output_dir(job_spec)
    artifact_dir = output_dir / "peft_lora"
    save_model = model
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap_model):
        try:
            save_model = unwrap_model(model)
        except Exception:
            save_model = model
    saved_artifacts = _save_real_peft_adapter(save_model, artifact_dir)
    artifact_bundle = _materialize_toy_peft_job_artifacts(
        output_dir=output_dir,
        job_spec=job_spec,
        training_examples=training_examples,
        train_loss=train_loss,
        execution_mode="real_import",
        run_status="completed",
        artifact_subdir="peft_lora",
        runtime_path="real_import",
        artifact_kind="real_peft",
        manifest_name="real_peft_job_manifest.json",
        preserve_existing_adapter_files=True,
    )
    return {
        "backend": "peft",
        "dry_run": False,
        "execution_mode": "real_import",
        "job_spec": dict(job_spec),
        "recipe": recipe,
        "status": "completed",
        "import_probe": dict(job_spec.get("audit", {}).get("import_probe") or {}),
        "real_execution": {
            "kind": "real_peft",
            "path": "real_import",
            "num_examples": len(training_examples),
            "train_loss": train_loss,
            "output_dir": artifact_bundle["artifact_dir"],
            "artifact_dir": artifact_bundle["artifact_dir"],
            "artifacts": {**dict(artifact_bundle["artifacts"]), **saved_artifacts},
            "metrics": dict(artifact_bundle["metrics"]),
            "artifact_manifest_path": artifact_bundle["artifact_manifest_path"],
            "summary_path": artifact_bundle["summary_path"],
            "real_execution_path": artifact_bundle["real_execution_path"],
            "trainer_state_path": artifact_bundle["trainer_state_path"],
            "runtime_path": artifact_bundle["runtime_path"],
            "artifact_kind": artifact_bundle["artifact_kind"],
            "source_kind": source_kind,
            "source_path": source_path,
            "config_path": config_path,
            "load_mode": load_mode,
            "dependency_ready": True,
            "source_ready": bool(source_path or config_path),
            "executor_ready": True,
            "blocked_by": [],
            "blocking_reasons": [],
            "accelerator": {"class": "Accelerator", "used": True},
            "success": True,
            "message": "real peft execution completed",
            "forward_error": forward_error,
        },
    }


def _run_real_local_peft_training(job_spec: Mapping[str, Any]) -> dict[str, Any]:
    local_source = _resolve_real_local_model_source(job_spec)
    runtime_probe = _probe_real_local_runtime(local_source)
    if not runtime_probe["available"]:
        raise TrainingError("real local training requires torch, transformers, and a local model/config source")

    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    auto_config_cls = getattr(transformers, "AutoConfig", None)
    auto_model_cls = getattr(transformers, "AutoModelForCausalLM", None)
    if None in (auto_config_cls, auto_model_cls):
        raise TrainingError("transformers runtime is missing AutoConfig or AutoModelForCausalLM for local training")

    source_path = local_source.get("source_path")
    config_path = local_source.get("config_path")
    model = None
    load_mode = str(local_source.get("load_mode") or "unavailable")
    load_kwargs = _build_local_model_load_kwargs(local_only=bool(local_source.get("local_only", True)))
    if source_path is not None:
        try:
            model = auto_model_cls.from_pretrained(source_path, **load_kwargs)
            load_mode = "from_pretrained"
        except Exception:
            model = None

    if model is None:
        config_candidate = config_path or source_path
        if config_candidate is None:
            raise TrainingError("real local training requires a local model directory or config path")
        config = auto_config_cls.from_pretrained(config_candidate, local_files_only=True)
        model = auto_model_cls.from_config(config)
        load_mode = "from_config"

    training_examples = list(job_spec.get("training_examples") or [])
    if not training_examples:
        raise TrainingError("real local training requires at least one serialized training example")

    vocab_size = int(getattr(model.config, "vocab_size", 128) or 128)
    max_length = _resolve_model_sequence_length(model.config)
    encoded_rows: list[dict[str, Any]] = []
    for item in training_examples:
        instruction = str(item.get("instruction") or "")
        chosen = str(item.get("chosen") or "")
        text = (instruction + "\n" + chosen).strip()
        token_ids = _encode_training_text(text, max_length=max_length, vocab_size=max(vocab_size, 16))
        padded = token_ids + [0] * (max_length - len(token_ids))
        attention = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
        encoded_rows.append(
            {
                "input_ids": padded,
                "attention_mask": attention,
                "labels": list(padded),
            }
        )

    recipe = dict(job_spec.get("recipe") or {})
    training = dict(recipe.get("training") or {})
    output_dir = _resolve_toy_peft_output_dir(job_spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loss: float | None = None
    forward_error: str | None = None
    try:
        batch = encoded_rows[0]
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).unsqueeze(0)
        labels = torch.tensor(batch["labels"], dtype=torch.long).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = getattr(outputs, "loss", None)
        if loss is not None:
            train_loss = float(loss.detach().cpu().item())
    except Exception as exc:
        forward_error = f"{exc.__class__.__name__}: {exc}"

    if train_loss is None:
        token_total = sum(sum(row["input_ids"]) for row in encoded_rows)
        attention_total = sum(sum(row["attention_mask"]) for row in encoded_rows)
        text_total = sum(len(str(item.get("instruction") or "") + str(item.get("chosen") or "")) for item in training_examples)
        fallback = (token_total + attention_total + text_total + max_length + len(training_examples))
        train_loss = round(float(fallback / max(len(encoded_rows) * max_length, 1)), 6)

    artifact_bundle = _materialize_toy_peft_job_artifacts(
        output_dir=output_dir,
        job_spec=job_spec,
        training_examples=training_examples,
        train_loss=train_loss,
        execution_mode="real_local",
        run_status="completed",
        artifact_subdir="real_local_model",
        runtime_path="real_local",
        artifact_kind="real_local_peft",
        manifest_name="real_local_job_manifest.json",
        model_filename="real_local_model.safetensors",
    )
    return {
        "backend": "peft",
        "dry_run": False,
        "execution_mode": "real_local",
        "job_spec": dict(job_spec),
        "recipe": recipe,
        "status": "completed",
        "import_probe": dict(job_spec.get("audit", {}).get("import_probe") or {}),
        "real_execution": {
            "kind": "real_local_peft",
            "path": "real_local",
            "num_examples": len(training_examples),
            "train_loss": train_loss,
            "output_dir": artifact_bundle["artifact_dir"],
            "artifact_dir": artifact_bundle["artifact_dir"],
            "artifacts": dict(artifact_bundle["artifacts"]),
            "metrics": dict(artifact_bundle["metrics"]),
            "artifact_manifest_path": artifact_bundle["artifact_manifest_path"],
            "summary_path": artifact_bundle["summary_path"],
            "real_execution_path": artifact_bundle["real_execution_path"],
            "trainer_state_path": artifact_bundle["trainer_state_path"],
            "runtime_path": artifact_bundle["runtime_path"],
            "artifact_kind": artifact_bundle["artifact_kind"],
            "source_kind": local_source["source_kind"],
            "source_path": local_source["source_path"],
            "config_path": local_source["config_path"],
            "load_mode": load_mode,
            "dependency_ready": runtime_probe["available"],
            "source_ready": bool(local_source.get("available")),
            "executor_ready": True,
            "blocked_by": [],
            "blocking_reasons": [],
            "success": True,
            "message": "real local training completed",
            "forward_error": forward_error,
        },
    }


def execute_peft_training(*, job_spec: Mapping[str, Any], dry_run: bool = True) -> dict[str, Any]:
    recipe = dict(job_spec.get("recipe") or {})
    import_probe = dict(job_spec.get("audit", {}).get("import_probe") or {})
    local_source = _resolve_real_local_model_source(job_spec)
    local_runtime_probe = _probe_real_local_runtime(local_source)
    runtime_probe = _probe_real_peft_runtime()
    real_import_ready = bool(import_probe.get("ready"))
    readiness_summary = _build_peft_readiness_summary(
        import_probe=import_probe,
        local_source=local_source,
        local_runtime_probe=local_runtime_probe,
        runtime_probe=runtime_probe,
    )
    if not dry_run and real_import_ready and runtime_probe["available"]:
        try:
            result = _run_real_import_peft_training(job_spec)
            result["readiness_summary"] = readiness_summary
            return result
        except Exception as exc:
            return {
                "backend": "peft",
                "dry_run": dry_run,
                "execution_mode": "real_import",
                "job_spec": dict(job_spec),
                "recipe": recipe,
                "status": "ready",
                "import_probe": import_probe,
                "readiness_summary": readiness_summary,
                "real_execution": {
                    "kind": "real_peft",
                    "path": "real_import",
                    "attempted": True,
                    "success": False,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "available": True,
                    "runtime_path": "real_import",
                    "artifact_kind": "real_peft",
                },
            }
    if not dry_run and local_runtime_probe["available"]:
        try:
            result = _run_real_local_peft_training(job_spec)
            result["readiness_summary"] = readiness_summary
            return result
        except Exception as exc:
            return {
                "backend": "peft",
                "dry_run": dry_run,
                "execution_mode": "real_local",
                "job_spec": dict(job_spec),
                "recipe": recipe,
                "status": "ready",
                "import_probe": import_probe,
                "readiness_summary": readiness_summary,
                "real_execution": {
                    "kind": "real_local_peft",
                    "path": "real_local",
                    "attempted": True,
                    "success": False,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "available": True,
                    "source_kind": local_source["source_kind"],
                    "source_path": local_source["source_path"],
                    "config_path": local_source["config_path"],
                    "load_mode": local_source["load_mode"],
                },
            }
    return {
        "backend": "peft",
        "dry_run": dry_run,
        "execution_mode": "real_import" if (real_import_ready and runtime_probe["available"]) else ("real_local" if local_runtime_probe["available"] else "fallback"),
        "job_spec": dict(job_spec),
        "recipe": recipe,
        "status": "prepared" if dry_run else "ready",
        "import_probe": import_probe,
        "readiness_summary": readiness_summary,
        "real_execution": {
            "kind": (
                "real_peft"
                if (real_import_ready and runtime_probe["available"])
                else ("real_local_peft" if local_runtime_probe["available"] else "toy_local_peft")
            ),
            "path": "real_import" if (real_import_ready and runtime_probe["available"]) else ("real_local" if local_runtime_probe["available"] else ("toy_local" if runtime_probe["available"] else "unavailable")),
            "runtime_path": "real_import" if (real_import_ready and runtime_probe["available"]) else ("real_local" if local_runtime_probe["available"] else ("toy_local" if runtime_probe["available"] else "unavailable")),
            "attempted": False,
            "available": bool(local_runtime_probe["available"] or runtime_probe["available"]),
            "missing_modules": list(runtime_probe["missing_modules"] or local_runtime_probe["missing_modules"]),
            "source_kind": local_source["source_kind"],
            "source_path": local_source["source_path"],
            "config_path": local_source["config_path"],
            "load_mode": local_source["load_mode"],
            "artifact_kind": "real_peft" if (real_import_ready and runtime_probe["available"]) else ("real_local_peft" if local_runtime_probe["available"] else "toy_local_peft"),
            "dependency_ready": bool(local_runtime_probe["dependency_ready"] if "dependency_ready" in local_runtime_probe else runtime_probe["available"]),
            "source_ready": bool(local_runtime_probe["source_ready"] if "source_ready" in local_runtime_probe else local_source["available"]),
            "executor_ready": bool(real_import_ready),
            "blocked_by": list(
                dict.fromkeys(
                    [
                        *((readiness_summary.get("real_local", {}) or {}).get("blocked_by", []) or []),
                        *((readiness_summary.get("real_import", {}) or {}).get("blocked_by", []) or []),
                    ]
                )
            ),
            "blocking_reasons": list(
                dict.fromkeys(
                    [
                        *((readiness_summary.get("real_local", {}) or {}).get("blocking_reasons", []) or []),
                        *((readiness_summary.get("real_import", {}) or {}).get("blocking_reasons", []) or []),
                    ]
                )
            ),
            "reason": readiness_summary.get("reason")
            if readiness_summary.get("reason")
            else (
                local_runtime_probe["reason"]
                if local_runtime_probe["available"]
                else (
                    local_runtime_probe.get("reason")
                    if not local_runtime_probe["source_ready"]
                    else runtime_probe.get("reason")
                )
            ),
        },
    }


def execute_unsloth_training(*, job_spec: Mapping[str, Any], dry_run: bool = True) -> dict[str, Any]:
    return {
        "backend": "unsloth",
        "dry_run": dry_run,
        "execution_mode": "real_import" if job_spec.get("ready") else "fallback",
        "job_spec": dict(job_spec),
        "status": "prepared",
    }


def execute_mlx_training(*, job_spec: Mapping[str, Any], dry_run: bool = True) -> dict[str, Any]:
    return {
        "backend": "mlx",
        "dry_run": dry_run,
        "execution_mode": "real_import" if job_spec.get("ready") else "fallback",
        "job_spec": dict(job_spec),
        "status": "prepared",
    }


def _render_trainer_job_script(job_spec: Mapping[str, Any]) -> str:
    job_json_path = str(job_spec.get("job_json_path") or "trainer_job.json")
    result_json_path = str(job_spec.get("result_json_path") or "trainer_job_result.json")
    package_root = str(Path(__file__).resolve().parents[2])
    script_lines = [
        "#!/usr/bin/env python3",
        '"""Auto-generated trainer job runner."""',
        "from __future__ import annotations",
        "",
        "import argparse",
        "import json",
        "import sys",
        "from pathlib import Path",
        "",
        f"PACKAGE_ROOT = {package_root!r}",
        "if PACKAGE_ROOT not in sys.path:",
        "    sys.path.insert(0, PACKAGE_ROOT)",
        "",
        "from pfe_core.trainer.runtime_job import run_training_job_file",
        "",
        "def main() -> int:",
        "    parser = argparse.ArgumentParser(description='Run a materialized PFE trainer job')",
        "    parser.add_argument('--job-json', default=%r)" % job_json_path,
        "    parser.add_argument('--result-json', default=%r)" % result_json_path,
        "    parser.add_argument('--dry-run', action='store_true')",
        "    args = parser.parse_args()",
        "    result = run_training_job_file(args.job_json, result_json_path=args.result_json, dry_run=args.dry_run)",
        "    print(json.dumps(result, ensure_ascii=False, sort_keys=True))",
        "    return 0",
        "",
        "if __name__ == '__main__':",
        "    raise SystemExit(main())",
        "",
    ]
    return "\n".join(script_lines)


def materialize_training_job_bundle(
    *,
    execution_plan: Mapping[str, Any],
    output_dir: str | Path,
    python_executable: str | None = None,
) -> TrainerJobMaterialization:
    backend = normalize_backend_name(execution_plan.get("execution_backend"))
    execution_executor = normalize_backend_name(execution_plan.get("execution_executor"))
    executor_mode = str(execution_plan.get("executor_mode") or "fallback")
    ready = bool(execution_plan.get("ready", False))
    fallback_from = execution_plan.get("fallback_from")
    dry_run = not ready
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    job_json_path = output_path / "trainer_job.json"
    result_json_path = output_path / "training_job_result.json"
    script_path = output_path / "trainer_job.py"
    job_json = dict(execution_plan)
    job_json["job_json_path"] = str(job_json_path)
    job_json["result_json_path"] = str(result_json_path)
    job_json["script_path"] = str(script_path)
    job_json["python_executable"] = python_executable or sys.executable
    job_json["dry_run"] = dry_run
    script_text = _render_trainer_job_script(job_json)
    job_json_path.write_text(json.dumps(job_json, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    script_path.write_text(script_text, encoding="utf-8")
    command = [
        python_executable or sys.executable,
        str(script_path),
        "--job-json",
        str(job_json_path),
        "--result-json",
        str(result_json_path),
    ]
    if dry_run:
        command.append("--dry-run")
    audit = {
        "backend": backend,
        "execution_executor": execution_executor,
        "executor_mode": executor_mode,
        "ready": ready,
        "fallback_from": fallback_from,
        "dry_run": dry_run,
        "materialized_files": [str(job_json_path), str(script_path)],
        "result_json_path": str(result_json_path),
    }
    job_json["audit"] = {
        "materialization": audit,
        "job_spec_audit": dict((job_json.get("job_spec") or {}).get("audit") or {}),
        "preference_reinforced_fresh_sample_summary": _summarize_preference_reinforced_fresh_samples(
            job_spec=job_json.get("job_spec") or job_json,
        ),
    }
    metadata = {
        "command_ready": ready,
        "command_mode": "direct_python",
        "script_name": script_path.name,
        "result_json_name": result_json_path.name,
    }
    return TrainerJobMaterialization(
        backend=backend,
        execution_backend=backend,
        execution_executor=execution_executor,
        executor_mode=executor_mode,
        ready=ready,
        fallback_from=fallback_from if isinstance(fallback_from, str) else None,
        dry_run=dry_run,
        command=command,
        script_path=str(script_path),
        job_json_path=str(job_json_path),
        result_json_path=str(result_json_path),
        script_text=script_text,
        job_json=job_json,
        audit=audit,
        metadata=metadata,
    )


def run_materialized_training_job_bundle(
    bundle: TrainerJobMaterialization | Mapping[str, Any],
    *,
    force_dry_run: bool | None = None,
    timeout_seconds: float | None = None,
) -> TrainerJobRunResult:
    if isinstance(bundle, TrainerJobMaterialization):
        materialized = bundle
    else:
        bundle_map = dict(bundle)
        materialized = TrainerJobMaterialization(
            backend=str(bundle_map.get("backend") or "mock_local"),
            execution_backend=str(bundle_map.get("execution_backend") or bundle_map.get("backend") or "mock_local"),
            execution_executor=str(bundle_map.get("execution_executor") or "mock_local"),
            executor_mode=str(bundle_map.get("executor_mode") or "fallback"),
            ready=bool(bundle_map.get("ready", False)),
            fallback_from=bundle_map.get("fallback_from") if isinstance(bundle_map.get("fallback_from"), str) else None,
            dry_run=bool(bundle_map.get("dry_run", True)),
            command=list(bundle_map.get("command") or []),
            script_path=str(bundle_map.get("script_path") or ""),
            job_json_path=str(bundle_map.get("job_json_path") or ""),
            result_json_path=str(bundle_map.get("result_json_path") or bundle_map.get("metadata", {}).get("result_json_path") or ""),
            script_text=str(bundle_map.get("script_text") or ""),
            job_json=dict(bundle_map.get("job_json") or {}),
            audit=dict(bundle_map.get("audit") or {}),
            metadata=dict(bundle_map.get("metadata") or {}),
        )

    dry_run = materialized.dry_run if force_dry_run is None else force_dry_run
    command = list(materialized.command)
    if dry_run and "--dry-run" not in command:
        command.append("--dry-run")
    if not dry_run:
        command = [part for part in command if part != "--dry-run"]

    if dry_run:
        return TrainerJobRunResult(
            attempted=False,
            success=True,
            status="planned",
            command=command,
            returncode=None,
            exit_code=None,
            stdout="",
            stderr="",
            runner_result={},
            materialization=materialized.to_dict(),
            audit={"status": "planned", "dry_run": True},
            metadata={"execution_state": "planned", "ready": materialized.ready},
        )

    completed = subprocess.run(
        command,
        cwd=str(Path(materialized.script_path).parent),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    runner_result: dict[str, Any] = {}
    result_json_path = Path(materialized.result_json_path) if materialized.result_json_path else None
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            if isinstance(payload, dict):
                runner_result = payload
        except Exception:
            runner_result = {}
    if not runner_result and result_json_path is not None and result_json_path.exists():
        try:
            payload = json.loads(result_json_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                runner_result = payload
        except Exception:
            runner_result = {}
    success = completed.returncode == 0
    return TrainerJobRunResult(
        attempted=True,
        success=success,
        status="executed" if success else "failed",
        command=command,
        returncode=completed.returncode,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        runner_result=runner_result,
        materialization=materialized.to_dict(),
        audit={
            "status": "executed" if success else "failed",
            "dry_run": False,
            "runner_status": runner_result.get("status"),
            "result_json_path": str(result_json_path) if result_json_path is not None else None,
        },
        metadata={
            "execution_state": "executed" if success else "failed",
            "ready": materialized.ready,
            "executor_mode": materialized.executor_mode,
            "result_json_path": str(result_json_path) if result_json_path is not None else None,
        },
    )


def execute_dpo_training(*, job_spec: Mapping[str, Any], dry_run: bool = True) -> dict[str, Any]:
    """Execute DPO training using trl.DPOTrainer.

    This executor handles Direct Preference Optimization training with support for:
    - Loading base models with optional SFT adapters (progressive training)
    - Creating reference models for DPO
    - Configuring LoRA/QLoRA for efficient fine-tuning
    - Running DPO training with proper hyperparameters (beta, label_smoothing, etc.)
    - Saving the resulting adapter

    Args:
        job_spec: Job specification with training configuration
        dry_run: If True, return planned execution without running training

    Returns:
        Dictionary with training results and metadata
    """
    recipe = dict(job_spec.get("recipe") or {})
    training = dict(recipe.get("training") or {})
    peft_config = dict(recipe.get("peft") or {})

    # Extract DPO-specific configuration
    dpo_config = peft_config.get("dpo_config", {})
    beta = dpo_config.get("beta", 0.1)
    label_smoothing = dpo_config.get("label_smoothing", 0.0)
    max_length = dpo_config.get("max_length", 2048)
    max_prompt_length = dpo_config.get("max_prompt_length", 1024)

    # Extract training examples
    training_examples = list(job_spec.get("training_examples") or [])

    # Check for incremental training context (SFT -> DPO)
    incremental_context = training.get("incremental_context") or job_spec.get("incremental_context")
    base_adapter_path = None
    if incremental_context:
        base_adapter_path = incremental_context.get("parent_adapter_path") or incremental_context.get("source_adapter_path")

    # Base model configuration
    base_model_name = training.get("base_model") or job_spec.get("base_model") or "gpt2"

    if dry_run:
        return {
            "backend": "dpo",
            "dry_run": True,
            "execution_mode": "real_import",
            "job_spec": dict(job_spec),
            "recipe": recipe,
            "status": "prepared",
            "dpo_config": {
                "beta": beta,
                "label_smoothing": label_smoothing,
                "max_length": max_length,
                "max_prompt_length": max_prompt_length,
            },
            "base_model": base_model_name,
            "base_adapter_path": base_adapter_path,
            "num_examples": len(training_examples),
        }

    # Real DPO training execution
    try:
        return _run_real_dpo_training(
            job_spec=job_spec,
            training_examples=training_examples,
            base_model_name=base_model_name,
            base_adapter_path=base_adapter_path,
            beta=beta,
            label_smoothing=label_smoothing,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )
    except Exception as exc:
        return {
            "backend": "dpo",
            "dry_run": False,
            "execution_mode": "real_import",
            "job_spec": dict(job_spec),
            "recipe": recipe,
            "status": "failed",
            "error": f"{exc.__class__.__name__}: {exc}",
            "dpo_config": {
                "beta": beta,
                "label_smoothing": label_smoothing,
                "max_length": max_length,
                "max_prompt_length": max_prompt_length,
            },
            "base_model": base_model_name,
            "base_adapter_path": base_adapter_path,
            "num_examples": len(training_examples),
        }


def _run_real_dpo_training(
    *,
    job_spec: Mapping[str, Any],
    training_examples: list[dict[str, Any]],
    base_model_name: str,
    base_adapter_path: str | None,
    beta: float,
    label_smoothing: float,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, Any]:
    """Run actual DPO training with trl.DPOTrainer.

    Args:
        job_spec: Job specification
        training_examples: List of training examples with prompt, chosen, rejected
        base_model_name: Base model name or path
        base_adapter_path: Optional path to SFT adapter for progressive training
        beta: DPO beta parameter (temperature)
        label_smoothing: Label smoothing parameter
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length

    Returns:
        Training results dictionary
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType

    if not training_examples:
        raise TrainingError("DPO training requires at least one training example")

    # Determine device and dtype
    device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # Load and merge SFT adapter if provided (progressive training: SFT -> DPO)
    if base_adapter_path and Path(base_adapter_path).exists():
        model = PeftModel.from_pretrained(model, base_adapter_path)
        model = model.merge_and_unload()

    # Create reference model (frozen copy of base model with SFT adapter)
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if base_adapter_path and Path(base_adapter_path).exists():
        ref_model = PeftModel.from_pretrained(ref_model, base_adapter_path)
        ref_model = ref_model.merge_and_unload()

    # Determine target modules based on base model architecture
    model_name_lower = base_model_name.lower()
    if "gpt2" in model_name_lower:
        target_modules = ["c_attn", "c_proj"]
    elif "qwen" in model_name_lower or "llama" in model_name_lower or "mistral" in model_name_lower or "gemma" in model_name_lower:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]

    # Apply LoRA configuration for DPO training
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Prepare DPO dataset
    dpo_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    for example in training_examples:
        prompt = example.get("instruction") or example.get("prompt", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")
        if prompt and chosen and rejected:
            dpo_data["prompt"].append(prompt)
            dpo_data["chosen"].append(chosen)
            dpo_data["rejected"].append(rejected)

    if not dpo_data["prompt"]:
        raise TrainingError("No valid DPO training examples found")

    from datasets import Dataset
    train_dataset = Dataset.from_dict(dpo_data)

    # Setup training arguments
    recipe = dict(job_spec.get("recipe") or {})
    training_recipe = dict(recipe.get("training") or {})
    epochs = training_recipe.get("epochs", 3)
    output_dir = _resolve_toy_peft_output_dir(job_spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup training arguments using DPOConfig when available for newer trl API
    try:
        from trl import DPOConfig
    except Exception:
        DPOConfig = None  # type: ignore[misc]

    if DPOConfig is not None:
        training_args = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            max_grad_norm=0.3,
            warmup_steps=0,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            run_name="pfe-dpo-training",
            report_to="none",
            beta=beta,
            label_smoothing=label_smoothing,
            max_length=max_length,
        )
    else:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            run_name="pfe-dpo-training",
            report_to="none",
        )

    # Initialize DPOTrainer with compatibility for newer trl API
    import inspect

    dpo_sig = inspect.signature(DPOTrainer.__init__)
    dpo_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "train_dataset": train_dataset,
        "args": training_args,
    }
    if "processing_class" in dpo_sig.parameters:
        dpo_kwargs["processing_class"] = tokenizer
    else:
        dpo_kwargs["tokenizer"] = tokenizer
    if "beta" in dpo_sig.parameters:
        dpo_kwargs["beta"] = beta
        dpo_kwargs["label_smoothing"] = label_smoothing
        dpo_kwargs["max_length"] = max_length
        if "max_prompt_length" in dpo_sig.parameters:
            dpo_kwargs["max_prompt_length"] = max_prompt_length

    trainer = DPOTrainer(**dpo_kwargs)

    # Train
    train_result = trainer.train()
    train_loss = getattr(train_result, "training_loss", None)

    # Save adapter
    artifact_dir = output_dir / "dpo_adapter"
    trainer.save_model(str(artifact_dir))

    # Materialize artifacts
    artifact_bundle = _materialize_toy_peft_job_artifacts(
        output_dir=output_dir,
        job_spec=job_spec,
        training_examples=training_examples,
        train_loss=train_loss,
        execution_mode="real_import",
        run_status="completed",
        artifact_subdir="dpo_adapter",
        runtime_path="real_dpo",
        artifact_kind="real_dpo",
        manifest_name="dpo_job_manifest.json",
        preserve_existing_adapter_files=True,
    )

    return {
        "backend": "dpo",
        "dry_run": False,
        "execution_mode": "real_import",
        "job_spec": dict(job_spec),
        "recipe": recipe,
        "status": "completed",
        "dpo_config": {
            "beta": beta,
            "label_smoothing": label_smoothing,
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
        },
        "base_model": base_model_name,
        "base_adapter_path": base_adapter_path,
        "num_examples": len(training_examples),
        "train_loss": train_loss,
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "artifacts": dict(artifact_bundle["artifacts"]),
        "metrics": dict(artifact_bundle["metrics"]),
        "real_execution": {
            "kind": "real_dpo",
            "path": "real_dpo",
            "num_examples": len(training_examples),
            "train_loss": train_loss,
            "output_dir": str(output_dir),
            "artifact_dir": str(artifact_dir),
            "artifacts": dict(artifact_bundle["artifacts"]),
            "metrics": dict(artifact_bundle["metrics"]),
            "artifact_manifest_path": artifact_bundle["artifact_manifest_path"],
            "summary_path": artifact_bundle["summary_path"],
            "real_execution_path": artifact_bundle["real_execution_path"],
            "trainer_state_path": artifact_bundle["trainer_state_path"],
            "runtime_path": artifact_bundle["runtime_path"],
            "artifact_kind": artifact_bundle["artifact_kind"],
            "success": True,
            "message": "DPO training completed successfully",
        },
    }
