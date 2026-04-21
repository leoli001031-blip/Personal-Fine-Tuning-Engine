"""Phase 0 local trainer skeleton."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..adapter_store.store import AdapterStore, create_adapter_store
from ..config import PFEConfig
from ..errors import TrainingError
from ..db.sqlite import save_samples
from ..storage import list_samples, mark_samples_used, write_json
from ..profile_extractor import get_user_profile_store
from ..inference.export_runtime import (
    build_export_runtime_spec,
    build_llama_cpp_export_command_plan,
    execute_llama_cpp_export_command,
    resolve_llama_cpp_export_tool_path,
    validate_llama_cpp_export_toolchain,
    write_materialized_export_plan,
)
from .executors import (
    build_training_execution_recipe,
    materialize_training_job_bundle,
    run_materialized_training_job_bundle,
    summarize_real_training_execution,
    summarize_training_job_execution,
)
from .runtime import detect_trainer_runtime, plan_trainer_backend
from .dpo_dataset import DPODatasetBuilder, build_dpo_dataset_from_samples
from .dpo_executor import DPOTrainerExecutor, TrainingResult as DPOTrainingResult
from .forget_detector import (
    ForgetDetector,
    ForgetMetrics,
    ReplaySample,
    create_forget_detector,
)
from .adapter_lineage import AdapterLineageTracker, LineageDecision, get_lineage_tracker
from .auto_rollback import AutoRollbackPolicy, RollbackDecision, get_auto_rollback_policy


@dataclass
class TrainingRunResult:
    version: str
    adapter_path: str
    num_samples: int
    metrics: dict[str, Any]
    runtime: dict[str, Any]
    backend_plan: dict[str, Any]
    backend_dispatch: dict[str, Any]
    executor_spec: dict[str, Any]
    execution_backend: str
    execution_executor: str
    executor_mode: str
    job_bundle: dict[str, Any]
    job_execution: dict[str, Any]
    job_execution_summary: dict[str, Any]
    real_execution_summary: dict[str, Any]
    export_runtime: dict[str, Any]
    export_command_plan: dict[str, Any]
    export_execution: dict[str, Any]
    export_toolchain_summary: dict[str, Any]
    export_write: dict[str, Any]
    requires_export_step: bool
    training_config: dict[str, Any]
    incremental_context: dict[str, Any] | None
    audit_info: dict[str, Any]
    forget_detected: bool = False
    """Whether forgetting was detected during training."""
    forget_metrics: dict[str, Any] | None = None
    """Detailed forget detection metrics."""
    replay_ratio_adjusted: bool = False
    """Whether replay ratio was adjusted based on user profile."""


def _summarize_export_toolchain(
    *,
    export_runtime: dict[str, Any],
    export_command_plan: dict[str, Any],
    export_execution: dict[str, Any],
    export_write: dict[str, Any],
) -> dict[str, Any]:
    summary = (
        dict(export_execution.get("toolchain_summary") or {})
        or dict(export_command_plan.get("toolchain_summary") or {})
    )
    if summary:
        summary.setdefault("target_backend", export_runtime.get("target_backend"))
        summary.setdefault("target_artifact_format", export_runtime.get("target_artifact_format"))
        summary.setdefault("required", bool(export_runtime.get("required", False)))
        summary.setdefault("write_state", export_write.get("write_state"))
        return summary

    return {
        "summary": "not_required" if not export_runtime.get("required", False) else "planned",
        "status": "not_required" if not export_runtime.get("required", False) else export_command_plan.get("status"),
        "toolchain_status": "not_required" if not export_runtime.get("required", False) else export_command_plan.get("status"),
        "required": bool(export_runtime.get("required", False)),
        "target_backend": export_runtime.get("target_backend"),
        "target_artifact_format": export_runtime.get("target_artifact_format"),
        "artifact_name": export_runtime.get("artifact_name"),
        "output_dir": export_runtime.get("output_dir"),
        "write_state": export_write.get("write_state"),
        "toolchain_reason": export_runtime.get("reason") or export_command_plan.get("reason"),
    }


def _summarize_export_artifact(
    *,
    export_runtime: dict[str, Any],
    export_command_plan: dict[str, Any],
    export_execution: dict[str, Any],
    export_write: dict[str, Any],
) -> dict[str, Any]:
    validation = dict(export_execution.get("output_artifact_validation") or {})
    command_metadata = dict(export_execution.get("command_metadata") or export_command_plan.get("command_metadata") or {})
    tool_resolution = dict(export_execution.get("tool_resolution") or export_command_plan.get("tool_resolution") or {})
    artifact_path = (
        validation.get("path")
        or export_execution.get("output_artifact_path")
        or export_write.get("artifact_path")
    )
    return {
        "status": export_execution.get("status") or export_execution.get("audit", {}).get("status"),
        "write_state": export_write.get("write_state") or export_write.get("metadata", {}).get("write_state"),
        "target_backend": export_runtime.get("target_backend"),
        "target_artifact_format": export_runtime.get("target_artifact_format"),
        "artifact_name": export_runtime.get("artifact_name"),
        "path": artifact_path,
        "valid": validation.get("valid"),
        "size_bytes": validation.get("size_bytes"),
        "converter_kind": command_metadata.get("converter_kind"),
        "outtype": command_metadata.get("outtype"),
        "tool_path": tool_resolution.get("resolved_path"),
    }


def _extract_real_execution_artifacts(job_execution: dict[str, Any]) -> dict[str, Any]:
    runner_result = dict(job_execution.get("runner_result") or {})
    real_execution = dict(runner_result.get("real_execution") or {})
    artifacts = dict(real_execution.get("artifacts") or {})
    return {
        "kind": real_execution.get("kind"),
        "artifact_dir": real_execution.get("artifact_dir") or real_execution.get("output_dir"),
        "artifacts": artifacts,
        "artifact_manifest_path": real_execution.get("artifact_manifest_path"),
        "summary_path": real_execution.get("summary_path"),
        "real_execution_path": real_execution.get("real_execution_path"),
        "trainer_state_path": real_execution.get("trainer_state_path"),
        "metrics": dict(real_execution.get("metrics") or {}),
        "success": real_execution.get("success"),
    }


def _sync_real_execution_artifacts_into_version_dir(
    *,
    version_dir: Path,
    real_execution_artifacts: dict[str, Any],
) -> dict[str, Any]:
    synced: dict[str, Any] = {
        "artifact_dir": real_execution_artifacts.get("artifact_dir"),
        "synced_files": {},
        "available": False,
    }
    artifacts = dict(real_execution_artifacts.get("artifacts") or {})
    for artifact_key, target_name in (
        ("adapter_model", "adapter_model.safetensors"),
        ("adapter_config", "adapter_config.json"),
    ):
        source = artifacts.get(artifact_key)
        if not source:
            continue
        source_path = Path(str(source)).expanduser()
        if not source_path.exists():
            continue
        target_path = version_dir / target_name
        shutil.copy2(source_path, target_path)
        synced["synced_files"][artifact_key] = {
            "source": str(source_path),
            "target": str(target_path),
        }
        synced["available"] = True
    return synced


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class TrainerService:
    """Create standard adapter artifacts without invoking a real finetuning backend."""

    EXECUTOR_IMPORTS: dict[str, tuple[str, ...]] = {
        "mock_local": (),
        "peft": ("torch", "transformers", "peft", "accelerate"),
        "dpo": ("torch", "transformers", "peft", "trl", "accelerate"),
        "unsloth": ("torch", "transformers", "unsloth"),
        "mlx": ("mlx", "mlx_lm"),
    }
    DISPATCH_FIND_SPEC_ONLY_IMPORTS = frozenset({"mlx", "mlx_lm"})

    def __init__(self, store: AdapterStore | None = None):
        self.store = store
        self.last_run_result: TrainingRunResult | None = None

    @staticmethod
    def _default_base_model_name() -> str:
        return "Qwen/Qwen2.5-3B-Instruct"

    def _load_trainer_config(self) -> PFEConfig:
        home = getattr(self.store, "home", None) if self.store is not None else None
        try:
            return PFEConfig.load(home=home)
        except Exception:
            return PFEConfig()

    @staticmethod
    def _normalize_selector_policy(value: Any) -> str:
        normalized = str(value or "promoted_or_latest").strip().lower()
        if normalized not in {"explicit", "latest", "recent", "promoted_or_latest"}:
            return "promoted_or_latest"
        return normalized

    @classmethod
    def _should_defer_dispatch_import(cls, module_name: str) -> bool:
        return module_name in cls.DISPATCH_FIND_SPEC_ONLY_IMPORTS

    def _resolve_incremental_parent_context(
        self,
        *,
        base_adapter: str,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        requested_adapter = str(base_adapter or "").strip()
        if not requested_adapter:
            raise TrainingError("incremental training requires a parent adapter version or path")

        store = self.store or create_adapter_store(workspace=workspace)
        trainer_config = self._load_trainer_config()
        selector_policy = self._normalize_selector_policy(getattr(trainer_config.trainer, "incremental_parent_selector", "promoted_or_latest"))

        # Phase 2-D: Smart parent selection using lineage tracker
        lineage_strategy = getattr(trainer_config.trainer, "lineage_parent_strategy", "latest_eval")
        lineage_tracker = get_lineage_tracker()

        def _resolve_smart_lineage_candidate() -> tuple[Path, str] | None:
            """Use lineage tracker to select the best parent from available versions."""
            try:
                rows = list(store.list_version_records(limit=20))
            except Exception:
                return None
            candidates = [
                str(row.get("version") or "").strip()
                for row in rows
                if str(row.get("version") or "").strip()
            ]
            if not candidates:
                return None
            decision = lineage_tracker.find_best_parent(candidates, strategy=lineage_strategy)
            if decision.selected_parent:
                try:
                    return Path(store.load(decision.selected_parent)), decision.selected_parent
                except Exception:
                    pass
            return None

        def _resolve_recent_candidate() -> tuple[Path, str]:
            current_latest = None
            if hasattr(store, "current_latest_version"):
                try:
                    current_latest = store.current_latest_version()
                except Exception:
                    current_latest = None
            if current_latest:
                try:
                    return Path(store.load(current_latest)), str(current_latest)
                except Exception:
                    pass

            if hasattr(store, "list_version_records"):
                try:
                    rows = list(store.list_version_records(limit=20))
                except Exception:
                    rows = []
                for row in rows:
                    version = str(row.get("version") or "").strip()
                    if not version:
                        continue
                    try:
                        return Path(store.load(version)), version
                    except Exception:
                        continue
            raise TrainingError("incremental training requires an existing parent adapter; no fallback parent is available")

        def _explicit_parent() -> tuple[Path, str]:
            requested_path = Path(requested_adapter).expanduser()
            if requested_path.exists():
                return requested_path, requested_path.name
            try:
                parent_path = Path(store.load(requested_adapter))
            except Exception as exc:
                raise TrainingError(f"incremental training requires an existing parent adapter: {requested_adapter}") from exc
            return parent_path, parent_path.name

        selection_mode = "explicit"
        selection_reason = "requested parent adapter resolved explicitly"
        if requested_adapter.lower() in {"latest", "recent", "auto"}:
            selection_mode = requested_adapter.lower()
            # Phase 2-D: Try lineage-based smart selection first for "auto"
            if requested_adapter.lower() == "auto":
                lineage_result = _resolve_smart_lineage_candidate()
                if lineage_result:
                    parent_path, parent_version = lineage_result
                    selection_reason = f"lineage strategy '{lineage_strategy}' selected {parent_version}"
                else:
                    parent_path, parent_version = _resolve_recent_candidate()
                    selection_reason = f"requested parent selector '{requested_adapter}' resolved to {parent_version} (lineage fallback)"
            else:
                try:
                    parent_path, parent_version = _resolve_recent_candidate()
                    selection_reason = f"requested parent selector '{requested_adapter}' resolved to {parent_version}"
                except TrainingError:
                    raise
        else:
            try:
                parent_path, parent_version = _explicit_parent()
            except TrainingError:
                if selector_policy == "explicit":
                    raise
                # Phase 2-D: Try smart lineage selection before fallback
                lineage_result = _resolve_smart_lineage_candidate()
                if lineage_result:
                    parent_path, parent_version = lineage_result
                    selection_mode = f"lineage_{lineage_strategy}"
                    selection_reason = (
                        f"requested parent {requested_adapter} was unavailable; "
                        f"lineage strategy '{lineage_strategy}' selected {parent_version}"
                    )
                else:
                    parent_path, parent_version = _resolve_recent_candidate()
                    selection_mode = f"fallback_{selector_policy}"
                    selection_reason = (
                        f"requested parent {requested_adapter} was unavailable; "
                        f"using {selector_policy} candidate {parent_version}"
                    )

        manifest_path = parent_path / "adapter_manifest.json"
        parent_manifest = _read_json_file(manifest_path) if manifest_path.exists() else {}
        parent_base_model = str(parent_manifest.get("base_model") or self._default_base_model_name())
        parent_artifact_format = str(
            parent_manifest.get("artifact_format")
            or parent_manifest.get("artifactFormat")
            or parent_manifest.get("artifact_type")
            or "peft_lora"
        )
        return {
            "requested_base_adapter": requested_adapter,
            "parent_adapter_version": parent_version,
            "parent_adapter_path": str(parent_path),
            "parent_manifest_path": str(manifest_path),
            "parent_manifest": parent_manifest,
            "parent_base_model": parent_base_model,
            "parent_artifact_format": parent_artifact_format,
            "parent_state": parent_manifest.get("state"),
            "parent_num_samples": parent_manifest.get("num_samples"),
            "parent_training_run_id": parent_manifest.get("training_run_id") or parent_version,
            "parent_selection_policy": selector_policy,
            "parent_selection_mode": selection_mode,
            "parent_selection_reason": selection_reason,
            "resolved_base_model": parent_base_model,
            "source_adapter_version": parent_version,
            "source_model": parent_base_model,
            "lineage_strategy": lineage_strategy,
        }

    def _resolve_target_inference_backend(self, *, base_model_name: str) -> str:
        normalized = str(base_model_name or "").strip().lower()
        if "llama" in normalized:
            return "llama_cpp"

        model_path = Path(base_model_name).expanduser()
        looks_like_local_text_model = model_path.exists() and any(
            family in normalized for family in ("qwen", "llama", "mistral", "gemma")
        )
        if looks_like_local_text_model:
            tool_resolution = resolve_llama_cpp_export_tool_path()
            toolchain = validate_llama_cpp_export_toolchain(tool_resolution)
            if bool(toolchain.get("ready", False)):
                return "llama_cpp"

        return "transformers"

    def _get_profile_adjusted_params(self, user_id: str | None = None) -> dict[str, Any]:
        """Get training parameters adjusted by user profile.

        Args:
            user_id: Optional user ID for profile-based adjustments

        Returns:
            Dictionary of adjusted training parameters
        """
        if not user_id:
            return {}

        try:
            store = get_user_profile_store()
            profile = store.get_profile(user_id)

            adjustments = {}

            # Adjust learning rate based on interaction pattern confidence
            # Users with clear preferences can handle more aggressive learning
            top_patterns = profile.get_top_interaction_patterns(1)
            if top_patterns and top_patterns[0][1] > 0.8:
                adjustments["learning_rate_multiplier"] = 1.1
            elif top_patterns and top_patterns[0][1] < 0.4:
                adjustments["learning_rate_multiplier"] = 0.9

            # Adjust epochs based on domain diversity
            # Users with diverse interests may need more epochs to capture variety
            top_domains = profile.get_top_domains(3)
            if len(top_domains) >= 3:
                adjustments["epoch_adjustment"] = 1  # Add one epoch for diversity

            # Adjust replay ratio based on style consistency
            # Users with consistent style can use less replay
            top_styles = profile.get_top_style_preferences(1)
            if top_styles and top_styles[0][1] > 0.8:
                adjustments["replay_ratio_adjustment"] = -0.1  # Reduce replay

            return adjustments
        except Exception:
            return {}

    @staticmethod
    def _sample_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
        metadata = sample.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    @classmethod
    def _is_preference_reinforced_sample(cls, sample: Dict[str, Any]) -> bool:
        metadata = cls._sample_metadata(sample)
        training_signal_category = str(
            metadata.get("training_signal_category")
            or sample.get("training_signal_category")
            or ""
        ).strip().lower()
        explicit_reinforced = bool(
            metadata.get("explicit_response_preference_reinforced")
            or sample.get("explicit_response_preference_reinforced")
        )
        return explicit_reinforced or training_signal_category == "preference_reinforced"

    def _build_dataset(
        self,
        *,
        train_type: str,
        replay_ratio: float | None = None,
        history_limit: int | None = None,
        min_replay_samples: int | None = None,
        user_id: str | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        trainer_config = self._load_trainer_config().trainer
        sample_type = "dpo" if train_type == "dpo" else "sft"
        fresh = list_samples(sample_type=sample_type, dataset_split="train", include_used=False)
        history = [sample for sample in list_samples(sample_type=sample_type, dataset_split="train") if sample["used_in_version"]]

        # Get profile-based adjustments
        profile_adjustments = self._get_profile_adjusted_params(user_id)

        if replay_ratio is None:
            replay_ratio = trainer_config.dpo_replay_ratio if sample_type == "dpo" else trainer_config.replay_ratio

        # Apply profile-based replay ratio adjustment
        if "replay_ratio_adjustment" in profile_adjustments:
            replay_ratio = max(0.0, min(1.0, replay_ratio + profile_adjustments["replay_ratio_adjustment"]))

        if history_limit is None:
            history_limit = trainer_config.replay_history_limit
        if min_replay_samples is None:
            min_replay_samples = trainer_config.replay_min_samples
        if history_limit is not None and history_limit > 0:
            history = history[-history_limit:]
        configured_ratio = max(0.0, min(1.0, float(replay_ratio if replay_ratio is not None else 0.0)))
        configured_min_replay = max(0, int(min_replay_samples if min_replay_samples is not None else 1))
        target_replay = min(len(history), int(round(len(fresh) * configured_ratio)))
        if fresh and history and target_replay < configured_min_replay:
            target_replay = min(len(history), configured_min_replay)
        replay = history[-target_replay:] if target_replay > 0 else []

        # Prioritize explicit preference-reinforced fresh samples before any other ranking.
        if fresh:
            def _fresh_priority(sample: Dict[str, Any]) -> int:
                return 0 if self._is_preference_reinforced_sample(sample) else 1

            try:
                if user_id:
                    store = get_user_profile_store()
                    profile = store.get_profile(user_id)
                    priority_domains = {d[0] for d in profile.get_top_domains(5)}

                    def _domain_relevance(sample: Dict[str, Any]) -> float:
                        instruction = sample.get("instruction", "").lower()
                        domain_keywords = {
                            "programming": ["code", "programming", "代码", "编程"],
                            "writing": ["write", "essay", "写作", "文章"],
                            "learning": ["learn", "tutorial", "学习", "教程"],
                            "analysis": ["analyze", "data", "分析", "数据"],
                        }
                        score = 0.0
                        for domain in priority_domains:
                            keywords = domain_keywords.get(domain, [domain])
                            if any(kw in instruction for kw in keywords):
                                score += 1.0
                        return score

                    fresh = sorted(
                        fresh,
                        key=lambda sample: (_fresh_priority(sample), -_domain_relevance(sample)),
                    )
                else:
                    fresh = sorted(fresh, key=_fresh_priority)
            except Exception:
                fresh = sorted(fresh, key=_fresh_priority)

        reinforced_fresh_samples = [sample for sample in fresh if self._is_preference_reinforced_sample(sample)]
        reinforced_fresh_sample_ids = [sample.get("sample_id") for sample in reinforced_fresh_samples]

        dataset_plan = {
            "train_type": train_type,
            "sample_type": sample_type,
            "configured_replay_ratio": configured_ratio,
            "history_limit": history_limit,
            "min_replay_samples": configured_min_replay,
            "fresh_sample_count": len(fresh),
            "history_sample_count": len(history),
            "selected_replay_count": len(replay),
            "selected_replay_ratio": round(len(replay) / len(train_samples), 4) if (train_samples := (fresh + replay)) else 0.0,
            "replay_strategy": "recent_history_tail",
            "fresh_sample_ids": [sample.get("sample_id") for sample in fresh],
            "preference_reinforced_fresh_sample_count": len(reinforced_fresh_samples),
            "preference_reinforced_fresh_sample_ids": reinforced_fresh_sample_ids,
            "preference_reinforced_fresh_sample_ratio": (
                round(len(reinforced_fresh_samples) / len(fresh), 4) if fresh else 0.0
            ),
            "history_sample_ids": [sample.get("sample_id") for sample in history],
            "replay_sample_ids": [sample.get("sample_id") for sample in replay],
            "profile_adjustments": profile_adjustments,
            "profile_prioritized": bool(user_id),
        }
        return fresh, replay, dataset_plan

    def _validate_train_samples(self, train_samples: List[Dict[str, Any]], *, train_type: str) -> None:
        if any(sample["metadata"].get("dataset_split") == "test" for sample in train_samples):
            raise TrainingError("test split samples must never participate in training")
        if train_type == "dpo":
            invalid = [
                sample["sample_id"]
                for sample in train_samples
                if not sample.get("rejected") or not sample.get("chosen") or not sample.get("instruction")
            ]
            if invalid:
                raise TrainingError(
                    "dpo training requires explicit chosen/rejected pairs; "
                    f"invalid sample(s): {', '.join(invalid[:3])}"
                )

    def _dispatch_training_backend(
        self,
        *,
        backend_plan: Dict[str, Any],
        runtime: Dict[str, Any],
        backend_hint: str | None = None,
        base_adapter: str | None = None,
        allow_mock_fallback: bool = True,
    ) -> Dict[str, Any]:
        requested_backend = backend_hint or backend_plan.get("recommended_backend") or "mock_local"
        runtime_packages = runtime.get("installed_packages", {})
        if not isinstance(runtime_packages, dict):
            runtime_packages = {}
        available = {
            "mock_local": True,
            "peft": all(bool(runtime_packages.get(name, False)) for name in ("torch", "transformers", "peft", "accelerate")),
            "dpo": all(bool(runtime_packages.get(name, False)) for name in ("torch", "transformers", "peft", "trl", "accelerate")),
            "unsloth": bool(runtime_packages.get("unsloth", False)) and bool(runtime.get("cuda_available", False)),
            "mlx": bool(runtime_packages.get("mlx", False)) or bool(runtime_packages.get("mlx_lm", False)),
        }
        capability_map = {
            "mock_local": {
                "supports_sft": True,
                "supports_dpo": True,
                "artifact_format": "peft_lora",
            },
            "peft": {
                "supports_sft": True,
                "supports_dpo": True,
                "artifact_format": "peft_lora",
            },
            "dpo": {
                "supports_sft": False,
                "supports_dpo": True,
                "artifact_format": "peft_lora",
            },
            "unsloth": {
                "supports_sft": True,
                "supports_dpo": False,
                "artifact_format": "peft_lora",
            },
            "mlx": {
                "supports_sft": True,
                "supports_dpo": False,
                "artifact_format": "mlx_lora",
            },
        }
        train_type = str(backend_plan.get("train_type") or "sft")
        requested_known = requested_backend in capability_map
        dispatch_reasons = []
        if requested_known and not capability_map[requested_backend][f"supports_{train_type}"]:
            dispatch_reasons.append(f"{requested_backend} does not support {train_type}")
        if requested_known and not available.get(requested_backend, False):
            dispatch_reasons.append(f"{requested_backend} dependencies unavailable")

        # DPO-specific rerouting: use dedicated dpo executor when train_type is dpo
        # Respect explicit mock_local requests for testing
        if (
            train_type == "dpo"
            and requested_backend != "mock_local"
            and requested_known
            and capability_map[requested_backend]["supports_dpo"]
            and available.get("dpo", False)
        ):
            execution_backend = "dpo"
            dispatch_mode = "dpo_reroute"
            dispatch_reasons.append(f"using dpo executor for {train_type}")
        elif requested_known and not dispatch_reasons:
            execution_backend = requested_backend
            dispatch_mode = "requested"
            dispatch_reasons.append(f"using requested backend {requested_backend}")
        elif allow_mock_fallback:
            execution_backend = "mock_local"
            dispatch_mode = "fallback"
            if requested_known:
                dispatch_reasons.append("falling back to mock_local")
            else:
                dispatch_reasons.append(f"unknown backend hint {requested_backend}, falling back to mock_local")
        else:
            raise TrainingError(
                f"backend {requested_backend} is unavailable on this runtime and mock fallback is disabled"
            )

        execution_capability = capability_map[execution_backend]
        importable_modules = self.EXECUTOR_IMPORTS.get(execution_backend, ())
        import_attempts = []
        imported_modules = []
        deferred_modules = []
        executor_mode = "fallback"
        for module_name in importable_modules:
            module_available = importlib.util.find_spec(module_name) is not None
            attempt = {"module": module_name, "available": module_available}
            import_attempts.append(attempt)
            if not module_available:
                continue
            if self._should_defer_dispatch_import(module_name):
                attempt["probe_mode"] = "find_spec_only"
                attempt["deferred_import"] = True
                deferred_modules.append(module_name)
                continue
            try:
                importlib.import_module(module_name)
                imported_modules.append(module_name)
            except Exception as exc:
                attempt["import_error"] = exc.__class__.__name__
                if execution_backend != "mock_local":
                    dispatch_reasons.append(f"import failed for {module_name}")
                    if allow_mock_fallback:
                        execution_backend = "mock_local"
                        execution_capability = capability_map[execution_backend]
                        imported_modules = []
                        executor_mode = "fallback"
                        break
                    raise TrainingError(f"backend {requested_backend} import failed for {module_name}: {exc}") from exc
        else:
            if execution_backend == "mock_local":
                executor_mode = "phase0_mock"
            elif imported_modules and len(imported_modules) == len(importable_modules):
                executor_mode = "real_import"
            elif deferred_modules and len(imported_modules) + len(deferred_modules) == len(importable_modules):
                executor_mode = "deferred_import"
            elif execution_backend != "mock_local":
                executor_mode = "fallback"

        return {
            "requested_backend": requested_backend,
            "execution_backend": execution_backend,
            "execution_executor": execution_backend,
            "executor_mode": executor_mode,
            "import_attempts": import_attempts,
            "imported_modules": imported_modules,
            "dispatch_mode": dispatch_mode,
            "available": available,
            "capability": execution_capability,
            "reasons": dispatch_reasons,
            "requires_export_step": bool(backend_plan.get("requires_export_step", False)),
            "export_steps": list(backend_plan.get("export_steps", [])),
            "export_format": backend_plan.get("export_format"),
            "export_backend": backend_plan.get("export_backend"),
        }

    def _probe_executor_imports(
        self,
        backend_name: str,
        *,
        allow_mock_fallback: bool = True,
    ) -> Dict[str, Any]:
        required_modules = self.EXECUTOR_IMPORTS.get(backend_name, ())
        required_attrs = {
            "peft": {
                "peft": ("LoraConfig", "get_peft_model"),
                "transformers": ("Trainer", "TrainingArguments"),
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
        }.get(backend_name, {})
        attempts: list[dict[str, Any]] = []
        imported_modules: dict[str, Any] = {}
        missing_modules: list[str] = []
        missing_attrs: list[str] = []
        for module_name in required_modules:
            module_available = importlib.util.find_spec(module_name) is not None
            attempt: dict[str, Any] = {"module": module_name, "available": module_available}
            attempts.append(attempt)
            if not module_available:
                missing_modules.append(module_name)
                continue
            try:
                module = importlib.import_module(module_name)
                imported_modules[module_name] = module
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
        if not ready and not allow_mock_fallback:
            raise TrainingError(
                f"backend {backend_name} is missing required imports or attributes: "
                f"modules={missing_modules}, attrs={missing_attrs}"
            )
        return {
            "backend": backend_name,
            "ready": ready,
            "import_attempts": attempts,
            "imported_modules": list(imported_modules.keys()),
            "imported_module_objects": imported_modules,
            "missing_modules": missing_modules,
            "missing_attrs": missing_attrs,
            "required_modules": list(required_modules),
            "required_attrs": {module_name: list(attrs) for module_name, attrs in required_attrs.items()},
            "executor_mode": "real_import" if ready else "fallback",
            "execution_executor": backend_name if ready else "mock_local",
            "reasons": [
                f"{backend_name} importable and capability-checked" if ready else f"{backend_name} missing dependencies"
            ],
        }

    def _resolve_training_executor(
        self,
        *,
        backend_dispatch: Dict[str, Any],
        runtime: Dict[str, Any],
        backend_hint: str | None = None,
        base_adapter: str | None = None,
        allow_mock_fallback: bool = True,
    ) -> Dict[str, Any]:
        selected_backend = backend_dispatch["execution_backend"]
        requested_backend = backend_hint or backend_dispatch.get("requested_backend") or selected_backend
        probe = self._probe_executor_imports(selected_backend, allow_mock_fallback=allow_mock_fallback)
        execution_executor = probe["execution_executor"]
        fallback_from = None
        if not probe["ready"]:
            if allow_mock_fallback and selected_backend != "mock_local":
                fallback_probe = self._probe_executor_imports("mock_local", allow_mock_fallback=True)
                execution_executor = "mock_local"
                fallback_from = selected_backend
                probe = {
                    **probe,
                    "ready": False,
                    "fallback_from": fallback_from,
                    "fallback_executor": execution_executor,
                    "executor_mode": "fallback",
                    "execution_executor": execution_executor,
                    "import_attempts": [
                        *probe.get("import_attempts", []),
                        *fallback_probe.get("import_attempts", []),
                    ],
                    "imported_modules": list(probe.get("imported_modules", [])),
                    "missing_modules": sorted(
                        {*(probe.get("missing_modules", []) or []), *(fallback_probe.get("missing_modules", []) or [])}
                    ),
                    "missing_attrs": sorted(
                        {*(probe.get("missing_attrs", []) or []), *(fallback_probe.get("missing_attrs", []) or [])}
                    ),
                    "reasons": [
                        f"{selected_backend} unavailable; falling back to mock_local",
                        *probe.get("reasons", []),
                        *fallback_probe.get("reasons", []),
                    ],
                }
            elif selected_backend != "mock_local":
                raise TrainingError(f"backend {selected_backend} has no usable executor on this runtime")

        if selected_backend == "mock_local":
            execution_mode = "phase0_mock"
            executor_mode = "phase0_mock"
        elif probe["ready"]:
            execution_mode = "real_import"
            executor_mode = "real_import"
        else:
            execution_mode = "fallback"
            executor_mode = "fallback"

        executor_spec = {
            "requested_backend": requested_backend,
            "execution_backend": selected_backend,
            "execution_executor": execution_executor,
            "execution_mode": execution_mode,
            "executor_mode": executor_mode,
            "ready": probe["ready"],
            "import_attempts": probe["import_attempts"],
            "imported_modules": probe["imported_modules"],
            "missing_modules": probe["missing_modules"],
            "missing_attrs": probe["missing_attrs"],
            "required_modules": probe["required_modules"],
            "required_attrs": probe["required_attrs"],
            "callable_name": f"_execute_{selected_backend}_backend",
            "executor_kind": f"{selected_backend}_executor",
            "capability": backend_dispatch.get("capability", {}),
            "dispatch_mode": backend_dispatch.get("dispatch_mode"),
            "requires_export_step": backend_dispatch.get("requires_export_step", False),
            "export_steps": list(backend_dispatch.get("export_steps", [])),
            "export_format": backend_dispatch.get("export_format"),
            "export_backend": backend_dispatch.get("export_backend"),
            "fallback_from": fallback_from,
            "reasons": [*backend_dispatch.get("reasons", []), *probe.get("reasons", [])],
            "runtime_device": runtime.get("runtime_device"),
        }
        return executor_spec

    def _execute_mock_local_backend(
        self,
        *,
        executor_spec: Dict[str, Any],
        base_model_name: str,
        train_type: str,
        method: str,
        epochs: int,
        train_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "execution_backend": "mock_local",
            "execution_mode": "phase0_mock",
            "executor_mode": executor_spec.get("executor_mode", "phase0_mock"),
            "executor_kind": executor_spec.get("executor_kind", "mock_local_executor"),
            "callable_name": executor_spec.get("callable_name", "_execute_mock_local_backend"),
            "backend_label": "mock_local:phase0_mock",
            "method": method,
            "epochs": epochs,
            "train_type": train_type,
            "base_model": base_model_name,
            "num_train_samples": len(train_samples),
            "branch": "mock_local",
        }

    def _execute_peft_backend(
        self,
        *,
        executor_spec: Dict[str, Any],
        base_model_name: str,
        train_type: str,
        method: str,
        epochs: int,
        train_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        execution_mode = "peft_real_import" if executor_spec.get("executor_mode") == "real_import" else "peft_fallback"
        return {
            "execution_backend": "peft",
            "execution_mode": execution_mode,
            "executor_mode": executor_spec.get("executor_mode", "fallback"),
            "executor_kind": executor_spec.get("executor_kind", "peft_executor"),
            "callable_name": executor_spec.get("callable_name", "_execute_peft_backend"),
            "backend_label": f"peft:{execution_mode}",
            "method": method,
            "epochs": epochs,
            "train_type": train_type,
            "base_model": base_model_name,
            "num_train_samples": len(train_samples),
            "branch": "peft",
            "imported_modules": list(executor_spec.get("imported_modules", [])),
        }

    def _execute_unsloth_backend(
        self,
        *,
        executor_spec: Dict[str, Any],
        base_model_name: str,
        train_type: str,
        method: str,
        epochs: int,
        train_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        execution_mode = "unsloth_real_import" if executor_spec.get("executor_mode") == "real_import" else "unsloth_fallback"
        return {
            "execution_backend": "unsloth",
            "execution_mode": execution_mode,
            "executor_mode": executor_spec.get("executor_mode", "fallback"),
            "executor_kind": executor_spec.get("executor_kind", "unsloth_executor"),
            "callable_name": executor_spec.get("callable_name", "_execute_unsloth_backend"),
            "backend_label": f"unsloth:{execution_mode}",
            "method": method,
            "epochs": epochs,
            "train_type": train_type,
            "base_model": base_model_name,
            "num_train_samples": len(train_samples),
            "branch": "unsloth",
            "imported_modules": list(executor_spec.get("imported_modules", [])),
        }

    def _execute_mlx_backend(
        self,
        *,
        executor_spec: Dict[str, Any],
        base_model_name: str,
        train_type: str,
        method: str,
        epochs: int,
        train_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        execution_mode = "mlx_real_import" if executor_spec.get("executor_mode") == "real_import" else "mlx_fallback"
        return {
            "execution_backend": "mlx",
            "execution_mode": execution_mode,
            "executor_mode": executor_spec.get("executor_mode", "fallback"),
            "executor_kind": executor_spec.get("executor_kind", "mlx_executor"),
            "callable_name": executor_spec.get("callable_name", "_execute_mlx_backend"),
            "backend_label": f"mlx:{execution_mode}",
            "method": method,
            "epochs": epochs,
            "train_type": train_type,
            "base_model": base_model_name,
            "num_train_samples": len(train_samples),
            "branch": "mlx",
            "imported_modules": list(executor_spec.get("imported_modules", [])),
        }

    def _execute_dpo_backend(
        self,
        *,
        executor_spec: Dict[str, Any],
        base_model_name: str,
        train_type: str,
        method: str,
        epochs: int,
        train_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        execution_mode = "dpo_real_import" if executor_spec.get("executor_mode") == "real_import" else "dpo_fallback"
        return {
            "execution_backend": "dpo",
            "execution_mode": execution_mode,
            "executor_mode": executor_spec.get("executor_mode", "fallback"),
            "executor_kind": executor_spec.get("executor_kind", "dpo_executor"),
            "callable_name": executor_spec.get("callable_name", "_execute_dpo_backend"),
            "backend_label": f"dpo:{execution_mode}",
            "method": method,
            "epochs": epochs,
            "train_type": train_type,
            "base_model": base_model_name,
            "num_train_samples": len(train_samples),
            "branch": "dpo",
            "imported_modules": list(executor_spec.get("imported_modules", [])),
        }

    def _simulate_backend_execution(
        self,
        *,
        backend_dispatch: Dict[str, Any],
        executor_spec: Dict[str, Any],
        base_model_name: str,
        train_type: str,
        method: str,
        epochs: int,
        train_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        execution_backend = executor_spec["execution_backend"]
        execution_executor = executor_spec.get("execution_executor", execution_backend)
        executor_mode = executor_spec.get("executor_mode", backend_dispatch.get("executor_mode", "fallback"))
        executor_branch = execution_executor
        if execution_executor == "mock_local":
            execution_mode = "phase0_mock"
            execution_detail = self._execute_mock_local_backend(
                executor_spec=executor_spec,
                base_model_name=base_model_name,
                train_type=train_type,
                method=method,
                epochs=epochs,
                train_samples=train_samples,
            )
        elif execution_executor == "peft":
            execution_detail = self._execute_peft_backend(
                executor_spec=executor_spec,
                base_model_name=base_model_name,
                train_type=train_type,
                method=method,
                epochs=epochs,
                train_samples=train_samples,
            )
            execution_mode = execution_detail["execution_mode"]
        elif execution_executor == "unsloth":
            execution_detail = self._execute_unsloth_backend(
                executor_spec=executor_spec,
                base_model_name=base_model_name,
                train_type=train_type,
                method=method,
                epochs=epochs,
                train_samples=train_samples,
            )
            execution_mode = execution_detail["execution_mode"]
        elif execution_executor == "mlx":
            execution_detail = self._execute_mlx_backend(
                executor_spec=executor_spec,
                base_model_name=base_model_name,
                train_type=train_type,
                method=method,
                epochs=epochs,
                train_samples=train_samples,
            )
            execution_mode = execution_detail["execution_mode"]
        elif execution_executor == "dpo":
            execution_detail = self._execute_dpo_backend(
                executor_spec=executor_spec,
                base_model_name=base_model_name,
                train_type=train_type,
                method=method,
                epochs=epochs,
                train_samples=train_samples,
            )
            execution_mode = execution_detail["execution_mode"]
        else:
            execution_mode = "dispatch_skeleton_unknown"
            execution_detail = {
                "execution_backend": execution_backend,
                "execution_executor": execution_executor,
                "execution_mode": execution_mode,
                "executor_mode": executor_mode,
            }
        return {
            "execution_backend": execution_backend,
            "execution_executor": executor_branch,
            "execution_mode": execution_mode,
            "executor_mode": executor_mode,
            "backend_label": f"{execution_backend}:{execution_mode}",
            "method": method,
            "epochs": epochs,
            "train_type": train_type,
            "base_model": base_model_name,
            "num_train_samples": len(train_samples),
            "branch": executor_branch,
            "execution_detail": execution_detail,
        }

    def _write_adapter_artifacts(
        self,
        *,
        version_dir: Path,
        version: str,
        workspace_name: str | None,
        materialize_latest_pointer: bool,
        base_model_name: str,
        train_type: str,
        method: str,
        epochs: int,
        train_samples: List[Dict[str, Any]],
        fresh: List[Dict[str, Any]],
        replay: List[Dict[str, Any]],
        training_config: Dict[str, Any],
        runtime: Dict[str, Any],
        backend_plan: Dict[str, Any],
        backend_dispatch: Dict[str, Any],
        executor_spec: Dict[str, Any],
        job_bundle: Dict[str, Any],
        job_execution: Dict[str, Any],
        job_execution_summary: Dict[str, Any],
        real_execution_summary: Dict[str, Any],
        export_runtime: Dict[str, Any],
        export_command_plan: Dict[str, Any],
        export_execution: Dict[str, Any],
        export_toolchain_summary: Dict[str, Any],
        export_write: Dict[str, Any],
    ) -> Dict[str, Any]:
        real_execution_artifacts = _extract_real_execution_artifacts(job_execution)
        adapter_payload = {
            "version": version,
            "base_model": base_model_name,
            "train_type": train_type,
            "method": method,
            "epochs": epochs,
            "num_train_samples": len(train_samples),
            "num_fresh_samples": len(fresh),
            "num_replay_samples": len(replay),
        }
        (version_dir / "adapter_model.safetensors").write_text(
            json.dumps(adapter_payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_json(
            version_dir / "adapter_config.json",
            {
                "base_model_name_or_path": base_model_name,
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "train_type": train_type,
            },
        )
        artifact_sync = _sync_real_execution_artifacts_into_version_dir(
            version_dir=version_dir,
            real_execution_artifacts=real_execution_artifacts,
        )
        manifest_path = version_dir / "adapter_manifest.json"
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest_payload, dict):
                manifest_payload = {}
        except Exception:
            manifest_payload = {}
        manifest_payload.update(
            {
                "version": version,
                "workspace": workspace_name or "user_default",
                "base_model": base_model_name,
                "state": "pending_eval",
                "num_samples": len(train_samples),
                "adapter_dir": str(version_dir),
                "artifact_format": backend_plan["capability"]["artifact_format"],
                "training_backend": training_config["backend"],
                "inference_backend": backend_plan.get("recommended_backend", "transformers"),
                "requires_export": bool(backend_plan.get("requires_export_step", False)),
                "source_adapter_version": version,
                "source_model": base_model_name,
                "training_run_id": version,
                "artifact_name": "adapter_model.safetensors",
                "metadata": {
                    "training": {
                        "backend": training_config["backend"],
                        "train_type": train_type,
                        "runtime_device": training_config.get("runtime_device"),
                        "requires_export_step": bool(backend_plan.get("requires_export_step", False)),
                        "export_backend": training_config.get("export_backend"),
                        "export_format": training_config.get("export_format"),
                    },
                    "backend_plan": backend_plan,
                    "backend_dispatch": backend_dispatch,
                    "executor_spec": executor_spec,
                    "job_bundle": job_bundle,
                    "job_execution": job_execution,
                    "job_execution_summary": job_execution_summary,
                    "real_execution_summary": real_execution_summary,
                    "real_execution_artifacts": real_execution_artifacts,
                    "artifact_sync": artifact_sync,
                    "runtime": runtime,
                    "export_runtime": export_runtime,
                    "export_command_plan": export_command_plan,
                    "export_execution": export_execution,
                    "export_toolchain_summary": export_toolchain_summary,
                    "export_write": export_write,
                    "supported_inference_backends": ["transformers", "mlx", "llama_cpp"],
                },
            }
        )
        write_json(manifest_path, manifest_payload)
        if materialize_latest_pointer:
            workspace_root = version_dir.parent
            latest_link = workspace_root / "latest"
            temp_link = workspace_root / ".latest.tmp"
            try:
                if temp_link.exists() or temp_link.is_symlink():
                    temp_link.unlink()
                os.symlink(version_dir, temp_link)
                os.replace(temp_link, latest_link)
            except Exception:
                pass
        write_json(
            version_dir / "training_meta.json",
            {
                **training_config,
                "runtime": runtime,
                "backend_plan": backend_plan,
                "backend_dispatch": backend_dispatch,
                "executor_spec": executor_spec,
                "job_bundle": job_bundle,
                "job_execution": job_execution,
                "job_execution_summary": job_execution_summary,
                "real_execution_summary": real_execution_summary,
                "real_execution_artifacts": real_execution_artifacts,
                "artifact_sync": artifact_sync,
                "export_runtime": export_runtime,
                "export_command_plan": export_command_plan,
                "export_execution": export_execution,
                "export_toolchain_summary": export_toolchain_summary,
                "export_write": export_write,
                "audit": {
                    "runtime": runtime,
                    "backend_plan": backend_plan,
                    "backend_dispatch": backend_dispatch,
                    "executor_spec": executor_spec,
                    "job_bundle": job_bundle,
                    "job_execution": job_execution,
                    "job_execution_summary": job_execution_summary,
                    "real_execution_summary": real_execution_summary,
                    "real_execution_artifacts": real_execution_artifacts,
                    "artifact_sync": artifact_sync,
                    "export_runtime": export_runtime,
                    "export_command_plan": export_command_plan,
                    "export_execution": export_execution,
                    "export_toolchain_summary": export_toolchain_summary,
                    "export_write": export_write,
                    "execution_backend": backend_dispatch["execution_backend"],
                    "execution_executor": executor_spec["execution_executor"],
                    "executor_mode": executor_spec["executor_mode"],
                    "requires_export_step": bool(backend_plan.get("requires_export_step", False)),
                    "export_steps": backend_plan.get("export_steps", []),
                },
                "execution_backend": backend_dispatch["execution_backend"],
                "execution_executor": executor_spec["execution_executor"],
                "executor_mode": executor_spec["executor_mode"],
                "num_train_samples": len(train_samples),
                "train_sample_ids": [sample["sample_id"] for sample in train_samples],
                "fresh_sample_ids": [sample["sample_id"] for sample in fresh],
                "replay_sample_ids": [sample["sample_id"] for sample in replay],
            },
        )
        manifest_path = version_dir / "adapter_manifest.json"
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest_payload = {}
            if isinstance(manifest_payload, dict):
                manifest_metadata = dict(manifest_payload.get("metadata") or {})
                manifest_metadata.update(
                    {
                        "backend_plan": backend_plan,
                        "backend_dispatch": backend_dispatch,
                        "executor_spec": executor_spec,
                        "job_bundle": job_bundle,
                        "job_execution": job_execution,
                        "job_execution_summary": job_execution_summary,
                        "real_execution_summary": real_execution_summary,
                        "real_execution_artifacts": real_execution_artifacts,
                        "artifact_sync": artifact_sync,
                        "runtime": runtime,
                        "export_runtime": export_runtime,
                        "export_command_plan": export_command_plan,
                        "export_execution": export_execution,
                        "export_toolchain_summary": export_toolchain_summary,
                        "export_write": export_write,
                        "supported_inference_backends": ["transformers", "mlx", "llama_cpp"],
                    }
                )
                manifest_payload["metadata"] = manifest_metadata
                write_json(manifest_path, manifest_payload)
        return {
            "loss": round(1.0 / max(len(train_samples), 1), 6),
            "epochs": epochs,
            "replay_ratio": training_config["replay_ratio"],
            "train_type": train_type,
            "real_execution_artifacts": real_execution_artifacts,
            "artifact_sync": artifact_sync,
        }

    def train(
        self,
        *,
        method: str = "qlora",
        epochs: int = 3,
        base_model: str | None = None,
        train_type: str = "sft",
        workspace: str | None = None,
        backend_hint: str | None = None,
        base_adapter: str | None = None,
    ) -> str:
        result = self.train_result(
            method=method,
            epochs=epochs,
            base_model=base_model,
            train_type=train_type,
            workspace=workspace,
            backend_hint=backend_hint,
        )
        return (
            f"Trained adapter {result.version} with {result.metrics['num_fresh_samples']} fresh sample(s)"
            f" and {result.metrics['num_replay_samples']} replay sample(s). State=pending_eval."
        )

    def train_result(
        self,
        *,
        method: str = "qlora",
        epochs: int = 3,
        base_model: str | None = None,
        train_type: str = "sft",
        workspace: str | None = None,
        backend_hint: str | None = None,
        base_adapter: str | None = None,
        incremental_context: dict[str, Any] | None = None,
    ) -> TrainingRunResult:
        if train_type not in {"sft", "dpo"}:
            raise TrainingError(f"unsupported train_type: {train_type}")

        store = self.store or create_adapter_store(workspace=workspace)
        trainer_config = self._load_trainer_config()
        incremental_context = dict(incremental_context or {})
        base_model_name = base_model or str(
            incremental_context.get("resolved_base_model")
            or incremental_context.get("parent_base_model")
            or self._default_base_model_name()
        )
        target_inference_backend = self._resolve_target_inference_backend(base_model_name=base_model_name)
        runtime = detect_trainer_runtime().to_dict()
        backend_plan = plan_trainer_backend(
            train_type=train_type,
            runtime=runtime,
            target_inference_backend=target_inference_backend,
        ).to_dict()
        backend_dispatch = self._dispatch_training_backend(
            backend_plan=backend_plan,
            runtime=runtime,
            backend_hint=backend_hint,
            allow_mock_fallback=True,
        )
        executor_spec = self._resolve_training_executor(
            backend_dispatch=backend_dispatch,
            runtime=runtime,
            backend_hint=backend_hint,
            allow_mock_fallback=True,
        )
        configured_replay_ratio = (
            trainer_config.trainer.dpo_replay_ratio if train_type == "dpo" else trainer_config.trainer.replay_ratio
        )
        fresh, replay, dataset_plan = self._build_dataset(
            train_type=train_type,
            replay_ratio=configured_replay_ratio,
            history_limit=trainer_config.trainer.replay_history_limit,
            min_replay_samples=trainer_config.trainer.replay_min_samples,
        )
        train_samples = fresh + replay
        if not fresh:
            raise TrainingError("no new train split samples available; run generate or distill first")
        self._validate_train_samples(train_samples, train_type=train_type)

        workspace_name = workspace or getattr(store, "workspace", None)
        execution_summary = self._simulate_backend_execution(
            backend_dispatch=backend_dispatch,
            executor_spec=executor_spec,
            base_model_name=base_model_name,
            train_type=train_type,
            method=method,
            epochs=epochs,
            train_samples=train_samples,
        )
        dpo_config = None
        if train_type == "dpo":
            dpo_cfg = trainer_config.trainer.dpo
            dpo_config = {
                "beta": dpo_cfg.beta,
                "label_smoothing": dpo_cfg.label_smoothing,
                "max_length": getattr(dpo_cfg, "max_length", dpo_cfg.max_prompt_length + dpo_cfg.max_response_length),
                "max_prompt_length": dpo_cfg.max_prompt_length,
            }
        execution_recipe = build_training_execution_recipe(
            backend_dispatch=backend_dispatch,
            runtime=runtime,
            method=method,
            epochs=epochs,
            train_type=train_type,
            base_model_name=base_model_name,
            num_train_samples=len(train_samples),
            num_fresh_samples=len(fresh),
            num_replay_samples=len(replay),
            replay_ratio=dataset_plan["selected_replay_ratio"],
            train_examples=train_samples,
            allow_mock_fallback=True,
            dpo_config=dpo_config,
        )
        executor_spec["backend_recipe"] = execution_recipe["backend_recipe"]
        executor_spec["executor_recipe"] = execution_recipe["executor_recipe"]
        executor_spec["job_spec"] = execution_recipe["job_spec"]
        executor_spec["execution_recipe"] = execution_recipe
        training_config = {
            "method": method,
            "epochs": epochs,
            "train_type": train_type,
            "backend": executor_spec["execution_backend"],
            "requested_backend": backend_dispatch["requested_backend"],
            "backend_plan": backend_plan,
            "backend_dispatch": backend_dispatch,
            "executor_spec": executor_spec,
            "runtime": runtime,
            "execution_backend": executor_spec["execution_backend"],
            "execution_executor": executor_spec["execution_executor"],
            "execution_mode": executor_spec["execution_mode"],
            "executor_mode": executor_spec["executor_mode"],
            "device_preference": backend_plan["device_preference"],
            "runtime_device": backend_plan["runtime_device"],
            "requires_export_step": backend_plan["requires_export_step"],
            "export_steps": list(backend_plan.get("export_steps", [])),
            "export_format": backend_plan.get("export_format"),
            "export_backend": backend_plan.get("export_backend"),
            "dispatch_reasons": list(backend_dispatch.get("reasons", [])),
            "execution_summary": execution_summary,
            "backend_recipe": execution_recipe["backend_recipe"],
            "executor_recipe": execution_recipe["executor_recipe"],
            "job_spec": execution_recipe["job_spec"],
            "execution_recipe": execution_recipe,
            "replay_ratio": dataset_plan["selected_replay_ratio"],
            "configured_replay_ratio": configured_replay_ratio,
            "dpo_beta": trainer_config.trainer.dpo_beta,
            "replay_history_limit": trainer_config.trainer.replay_history_limit,
            "replay_min_samples": trainer_config.trainer.replay_min_samples,
            "incremental_parent_selector": trainer_config.trainer.incremental_parent_selector,
            "dataset_plan": dataset_plan,
        }
        if incremental_context:
            training_config["incremental_context"] = dict(incremental_context)
            training_config["source_adapter_version"] = (
                incremental_context.get("source_adapter_version")
                or incremental_context.get("parent_adapter_version")
            )
            training_config["source_adapter_path"] = incremental_context.get("parent_adapter_path")
            training_config["source_model"] = incremental_context.get("source_model") or incremental_context.get("parent_base_model") or base_model_name
        draft = store.create_training_version(
            base_model=base_model_name,
            training_config=training_config,
            artifact_format=backend_plan["capability"]["artifact_format"],
        )
        version = draft["version"]
        version_dir = Path(draft["path"])
        job_bundle = materialize_training_job_bundle(
            execution_plan=execution_recipe,
            output_dir=version_dir,
        )
        job_execution = run_materialized_training_job_bundle(job_bundle).to_dict()
        executor_spec["job_bundle"] = job_bundle.to_dict()
        executor_spec["job_execution"] = job_execution
        training_config["job_bundle"] = job_bundle.to_dict()
        training_config["job_execution"] = job_execution
        pre_export_real_execution_artifacts = _extract_real_execution_artifacts(job_execution)
        pre_export_artifact_sync = _sync_real_execution_artifacts_into_version_dir(
            version_dir=version_dir,
            real_execution_artifacts=pre_export_real_execution_artifacts,
        )
        training_config["pre_export_artifact_sync"] = pre_export_artifact_sync
        export_target_backend = backend_plan.get("export_backend") or target_inference_backend or backend_dispatch["execution_backend"]
        export_runtime = build_export_runtime_spec(
            target_backend=export_target_backend,
            source_artifact_format=backend_plan["capability"]["artifact_format"],
            adapter_dir=version_dir,
            workspace=workspace_name,
            source_adapter_version=version,
            source_model=base_model_name,
            training_run_id=version,
            num_samples=len(train_samples),
            extra_metadata={
                "train_type": train_type,
                "backend_plan": backend_plan,
                "backend_dispatch": backend_dispatch,
                "runtime_device": backend_plan["runtime_device"],
            },
        ).to_dict()
        if export_runtime.get("required") and export_target_backend == "llama_cpp":
            export_command_plan = build_llama_cpp_export_command_plan(
                target_backend=export_target_backend,
                source_artifact_format=backend_plan["capability"]["artifact_format"],
                adapter_dir=version_dir,
                source_model_path=base_model_name,
            ).to_dict()
            export_execution = execute_llama_cpp_export_command(
                target_backend=export_target_backend,
                source_artifact_format=backend_plan["capability"]["artifact_format"],
                adapter_dir=version_dir,
                source_model_path=base_model_name,
                dry_run=False,
            ).to_dict()
            if export_execution.get("outcome") == "success" and isinstance(export_execution.get("audit"), dict):
                export_execution["audit"]["status"] = "success"
        else:
            export_command_plan = {
                "target_backend": export_target_backend,
                "required": bool(export_runtime.get("required", False)),
                "target_artifact_format": export_runtime.get("target_artifact_format"),
                "artifact_name": export_runtime.get("artifact_name"),
                "output_dir": export_runtime.get("output_dir"),
                "status": "not_required",
                "reason": export_runtime.get("reason"),
            }
            export_execution = {
                "attempted": False,
                "success": not bool(export_runtime.get("required", False)),
                "returncode": None,
                "exit_code": None,
                "output_dir": export_runtime.get("output_dir"),
                "output_artifact_path": None,
                "audit": {
                    "status": "not_required",
                    "required_export": bool(export_runtime.get("required", False)),
                    "reason": export_runtime.get("reason"),
                },
                "metadata": {
                    "execution_mode": "not_required",
                    "target_backend": export_target_backend,
                },
            }
        export_write = write_materialized_export_plan(
            target_backend=export_target_backend,
            source_artifact_format=backend_plan["capability"]["artifact_format"],
            adapter_dir=version_dir,
            workspace=workspace_name,
            source_adapter_version=version,
            source_model=base_model_name,
            training_run_id=version,
            num_samples=len(train_samples),
            extra_metadata={
                "train_type": train_type,
                "backend_plan": backend_plan,
                "backend_dispatch": backend_dispatch,
                "runtime_device": backend_plan["runtime_device"],
            },
        ).to_dict()
        execution_artifact_valid = bool(
            (export_execution.get("output_artifact_validation") or {}).get("valid", False)
            if isinstance(export_execution.get("output_artifact_validation"), dict)
            else False
        )
        export_write["write_state"] = (
            "validated"
            if export_execution.get("success") and export_execution.get("attempted") and execution_artifact_valid
            else "executed"
            if export_execution.get("success") and export_execution.get("attempted")
            else "materialized"
        )
        export_write.setdefault("target_artifact_format", export_runtime.get("target_artifact_format"))
        export_write.setdefault("artifact_name", export_runtime.get("artifact_name"))
        export_write.setdefault("dry_run", export_runtime.get("dry_run"))
        export_write.setdefault("command", export_command_plan.get("command"))
        export_write.setdefault("output_artifact_path", export_execution.get("output_artifact_path"))
        export_write.setdefault("artifact_path", export_execution.get("output_artifact_path") or export_write.get("artifact_path"))
        export_write.setdefault("metadata", {})
        export_write["metadata"]["write_state"] = export_write["write_state"]
        export_write["metadata"]["execution_status"] = (
            export_execution.get("audit", {}).get("status")
            if isinstance(export_execution.get("audit"), dict)
            else None
        )
        export_write["metadata"]["execution_artifact_valid"] = execution_artifact_valid
        job_execution_summary = summarize_training_job_execution(job_execution)
        real_execution_summary = summarize_real_training_execution(job_execution)
        export_toolchain_summary = _summarize_export_toolchain(
            export_runtime=export_runtime,
            export_command_plan=export_command_plan,
            export_execution=export_execution,
            export_write=export_write,
        )
        export_artifact_summary = _summarize_export_artifact(
            export_runtime=export_runtime,
            export_command_plan=export_command_plan,
            export_execution=export_execution,
            export_write=export_write,
        )
        training_config["export_runtime"] = export_runtime
        training_config["export_command_plan"] = export_command_plan
        training_config["export_execution"] = export_execution
        training_config["export_write"] = export_write
        training_config["job_execution_summary"] = job_execution_summary
        training_config["real_execution_summary"] = real_execution_summary
        training_config["export_toolchain_summary"] = export_toolchain_summary
        training_config["export_artifact_summary"] = export_artifact_summary
        metrics = self._write_adapter_artifacts(
            version_dir=version_dir,
            version=version,
            workspace_name=workspace_name,
            base_model_name=base_model_name,
            train_type=train_type,
            method=method,
            epochs=epochs,
            train_samples=train_samples,
            fresh=fresh,
            replay=replay,
            training_config=training_config,
            runtime=runtime,
            backend_plan=backend_plan,
            backend_dispatch=backend_dispatch,
            executor_spec=executor_spec,
            job_bundle=job_bundle.to_dict(),
            job_execution=job_execution,
            job_execution_summary=job_execution_summary,
            real_execution_summary=real_execution_summary,
            export_runtime=export_runtime,
            export_command_plan=export_command_plan,
            export_execution=export_execution,
            export_toolchain_summary=export_toolchain_summary,
            export_write=export_write,
            materialize_latest_pointer=not isinstance(store, AdapterStore),
        )
        if hasattr(store, "merge_manifest"):
            try:
                store.merge_manifest(
                    version,
                    {
                        **dict(export_runtime.get("manifest_updates") or {}),
                        "metadata": {
                            "export_runtime": export_runtime,
                            "export_command_plan": export_command_plan,
                            "export_execution": export_execution,
                            "export_toolchain_summary": export_toolchain_summary,
                            "export_artifact_summary": export_artifact_summary,
                            "export_write": export_write,
                            "job_bundle": job_bundle.to_dict(),
                            "job_execution": job_execution,
                            "job_execution_summary": job_execution_summary,
                            "real_execution_summary": real_execution_summary,
                            "real_execution_artifacts": metrics.get("real_execution_artifacts"),
                            "artifact_sync": metrics.get("artifact_sync"),
                        },
                        "export": {
                            "artifact": export_artifact_summary,
                        },
                    },
                )
            except Exception:
                pass
        metrics["num_fresh_samples"] = len(fresh)
        metrics["num_replay_samples"] = len(replay)
        metrics["runtime"] = runtime
        metrics["backend_plan"] = backend_plan
        metrics["backend_dispatch"] = backend_dispatch
        metrics["executor_spec"] = executor_spec
        metrics["execution_backend"] = executor_spec["execution_backend"]
        metrics["execution_executor"] = executor_spec["execution_executor"]
        metrics["execution_summary"] = execution_summary
        metrics["backend_recipe"] = execution_recipe["backend_recipe"]
        metrics["executor_recipe"] = execution_recipe["executor_recipe"]
        metrics["job_spec"] = execution_recipe["job_spec"]
        metrics["execution_recipe"] = execution_recipe
        metrics["job_bundle"] = job_bundle.to_dict()
        metrics["job_execution"] = job_execution
        metrics["job_execution_summary"] = job_execution_summary
        metrics["real_execution_summary"] = real_execution_summary
        metrics["export_runtime"] = export_runtime
        metrics["export_command_plan"] = export_command_plan
        metrics["export_execution"] = export_execution
        metrics["export_toolchain_summary"] = export_toolchain_summary
        metrics["export_artifact_summary"] = export_artifact_summary
        metrics["export_write"] = export_write
        metrics["dataset_plan"] = dataset_plan
        metrics["preference_reinforced_fresh_sample_count"] = dataset_plan["preference_reinforced_fresh_sample_count"]
        metrics["preference_reinforced_fresh_sample_ids"] = dataset_plan["preference_reinforced_fresh_sample_ids"]
        metrics["preference_reinforced_fresh_sample_ratio"] = dataset_plan["preference_reinforced_fresh_sample_ratio"]
        metrics["configured_replay_ratio"] = configured_replay_ratio
        if incremental_context:
            metrics["incremental_context"] = dict(incremental_context)
        metrics["state"] = "pending_eval"
        metrics["requires_export_step"] = bool(backend_plan.get("requires_export_step", False))
        metrics["audit"] = {
            "runtime": runtime,
            "backend_plan": backend_plan,
            "backend_dispatch": backend_dispatch,
            "executor_spec": executor_spec,
            "execution_summary": execution_summary,
            "execution_recipe": execution_recipe,
            "job_bundle": job_bundle.to_dict(),
            "job_execution": job_execution,
            "job_execution_summary": job_execution_summary,
            "real_execution_summary": real_execution_summary,
            "export_runtime": export_runtime,
            "export_command_plan": export_command_plan,
            "export_execution": export_execution,
            "export_toolchain_summary": export_toolchain_summary,
            "export_artifact_summary": export_artifact_summary,
            "export_write": export_write,
            "dataset_plan": dataset_plan,
            "incremental_context": dict(incremental_context) if incremental_context else None,
            "state": "pending_eval",
            "requires_export_step": bool(backend_plan.get("requires_export_step", False)),
            "export_steps": list(backend_plan.get("export_steps", [])),
        }
        # Phase 2.4: Forget Detection
        # Detect forgetting using replay loss comparison
        forget_detector = create_forget_detector()
        replay_samples_for_detection = [
            ReplaySample(
                sample_id=str(sample.get("sample_id", f"replay_{i}")),
                instruction=str(sample.get("instruction", "")),
                output=str(sample.get("output", "")),
                metadata=dict(sample.get("metadata", {})),
            )
            for i, sample in enumerate(replay)
        ]

        forget_metrics: ForgetMetrics | None = None
        forget_detected = False
        if replay_samples_for_detection:
            forget_detector.record_before_training(replay_samples_for_detection)

            # Compute after-training losses using actual training result
            actual_train_loss = real_execution_summary.get("train_loss")
            parent_baseline_loss: float | None = None
            if incremental_context:
                parent_version = incremental_context.get("parent_adapter_version") or incremental_context.get("source_adapter_version")
                if parent_version:
                    try:
                        parent_manifest = store._read_manifest(parent_version)
                        parent_metrics = parent_manifest.get("training_metrics", {})
                        parent_baseline_loss = parent_metrics.get("train_loss")
                    except Exception:
                        pass

            after_losses: dict[str, float] | None = None
            if actual_train_loss is not None:
                before_losses = forget_detector._before_losses
                if before_losses:
                    avg_before = sum(before_losses.values()) / len(before_losses)
                    # Use parent baseline as reference if available; otherwise use simulated before average
                    reference_loss = parent_baseline_loss if parent_baseline_loss is not None else avg_before
                    bias = actual_train_loss - reference_loss
                    after_losses = {
                        sid: max(0.01, loss + bias)
                        for sid, loss in before_losses.items()
                    }

            forget_metrics = forget_detector.detect_forget(after_losses=after_losses)
            forget_detected = forget_metrics.forget_detected

            # Update metrics with forget detection results
            metrics["forget_detection"] = forget_metrics.to_dict()
            metrics["audit"]["forget_detection"] = forget_metrics.to_dict()
            if actual_train_loss is not None:
                metrics["forget_detection"]["actual_train_loss"] = actual_train_loss
            if parent_baseline_loss is not None:
                metrics["forget_detection"]["parent_baseline_loss"] = parent_baseline_loss

            # If forgetting detected, trigger rollback consideration
            if forget_detected:
                metrics["forget_detected"] = True
                metrics["rollback_recommended"] = (
                    forget_metrics.recommendation == "rollback_required"
                )

        store.mark_pending_eval(version, num_samples=len(train_samples), metrics=metrics)
        mark_samples_used([sample["sample_id"] for sample in fresh], version)

        # Phase 2-D: Record lineage
        lineage_tracker = get_lineage_tracker()
        parent_version = incremental_context.get("parent_adapter_version") if incremental_context else None
        lineage_tracker.record_training_run(
            version=version,
            parent_version=parent_version,
            training_type=train_type,
            num_samples=len(train_samples),
            metrics={
                "forget_detected": forget_detected,
                "forget_metrics": forget_metrics.to_dict() if forget_metrics else None,
                "train_loss": real_execution_summary.get("train_loss"),
                "eval_loss": real_execution_summary.get("eval_loss"),
                "state": "pending_eval",
            },
        )

        # Phase 2-D: Auto-rollback policy evaluation
        rollback_performed = False
        rollback_decision: RollbackDecision | None = None
        try:
            all_versions = store.list_version_records(limit=100)
            num_versions = len(all_versions)
        except Exception:
            num_versions = 0

        if forget_detected and forget_metrics:
            policy = get_auto_rollback_policy()
            rollback_decision = policy.should_rollback(
                forget_metrics=forget_metrics.to_dict(),
                num_versions=num_versions,
                eval_failed=False,
            )
            metrics["rollback_decision"] = rollback_decision.to_dict()

            # Execute rollback if policy decides so
            if rollback_decision.should_rollback:
                fallback = rollback_decision.fallback_version or (
                    incremental_context.get("parent_adapter_version") if incremental_context else None
                )
                if fallback is None:
                    fallback = policy.select_fallback_version(version, lineage_tracker)
                try:
                    rollback_result = policy.execute_rollback(
                        store=store,
                        current_version=version,
                        fallback_version=fallback,
                        reason=rollback_decision.reason,
                    )
                    metrics["rollback"] = rollback_result
                    rollback_performed = rollback_result.get("success", False)
                    # Update lineage node with rollback info
                    lineage_tracker.update_node(
                        version,
                        state="archived_rollback",
                        metadata={"rollback_reason": rollback_decision.reason},
                    )
                except Exception as e:
                    metrics["rollback_error"] = str(e)
            elif rollback_decision.action == "increase_replay":
                metrics["replay_increase_recommended"] = True
        elif forget_metrics and forget_metrics.recommendation == "rollback_required":
            # Legacy path: direct rollback on rollback_required recommendation
            try:
                from ..adapter_store.lifecycle import rollback_to_version
                fallback = incremental_context.get("parent_adapter_version") if incremental_context else None
                rollback_result = rollback_to_version(
                    store=store,
                    current_version=version,
                    fallback_version=fallback,
                )
                metrics["rollback"] = rollback_result
                rollback_performed = rollback_result.get("success", False)
            except Exception as e:
                metrics["rollback_error"] = str(e)

        result = TrainingRunResult(
            version=version,
            adapter_path=str(version_dir),
            num_samples=len(train_samples),
            metrics=metrics,
            runtime=runtime,
            backend_plan=backend_plan,
            backend_dispatch=backend_dispatch,
            executor_spec=executor_spec,
            execution_backend=executor_spec["execution_backend"],
            execution_executor=executor_spec["execution_executor"],
            executor_mode=executor_spec["executor_mode"],
            job_bundle=job_bundle.to_dict(),
            job_execution=job_execution,
            job_execution_summary=job_execution_summary,
            real_execution_summary=real_execution_summary,
            export_runtime=export_runtime,
            export_command_plan=export_command_plan,
            export_execution=export_execution,
            export_toolchain_summary=export_toolchain_summary,
            export_write=export_write,
            requires_export_step=bool(backend_plan.get("requires_export_step", False)),
            training_config=training_config,
            incremental_context=dict(incremental_context) if incremental_context else None,
            audit_info=metrics["audit"],
            forget_detected=forget_detected,
            forget_metrics=forget_metrics.to_dict() if forget_metrics else None,
            replay_ratio_adjusted=bool(dataset_plan.get("profile_adjustments", {}).get("replay_ratio_adjustment")),
        )
        self.last_run_result = result
        return result

    def train_incremental(
        self,
        *,
        base_adapter: str,
        method: str = "qlora",
        epochs: int = 1,
        train_type: str = "sft",
        workspace: str | None = None,
    ) -> TrainingRunResult:
        incremental_context = self._resolve_incremental_parent_context(
            base_adapter=base_adapter,
            workspace=workspace,
        )
        return self.train_result(
            method=method,
            epochs=epochs,
            base_model=str(incremental_context.get("resolved_base_model") or self._default_base_model_name()),
            train_type=train_type,
            workspace=workspace,
            incremental_context=incremental_context,
        )

    def train_dpo(
        self,
        *,
        method: str = "qlora",
        epochs: int = 3,
        base_model: str | None = None,
        workspace: str | None = None,
        backend_hint: str | None = None,
        base_adapter: str | None = None,
        base_adapter_path: str | None = None,
        min_confidence: float | None = None,
    ) -> TrainingRunResult:
        """Execute DPO (Direct Preference Optimization) training.

        This method builds a DPO dataset from signals and executes preference
        optimization training. Supports incremental training from an existing
        SFT adapter.

        Args:
            method: Training method ("qlora" or "lora")
            epochs: Number of training epochs
            base_model: Base model name or path
            workspace: Workspace name
            backend_hint: Backend hint for execution
            base_adapter: Optional adapter version to use as base (for progressive SFT -> DPO training)
            base_adapter_path: Optional path to existing SFT adapter for incremental DPO
            min_confidence: Minimum signal confidence for DPO pairs

        Returns:
            TrainingRunResult with training results
        """
        trainer_config = self._load_trainer_config().trainer

        # Use configured min_confidence if not provided
        if min_confidence is None:
            min_confidence = trainer_config.dpo.min_preference_confidence

        # Build DPO dataset from signals
        dataset_builder = DPODatasetBuilder(workspace=workspace)
        dpo_dataset = dataset_builder.build_from_signals(
            min_confidence=min_confidence,
            limit=trainer_config.max_samples,
        )

        if len(dpo_dataset) == 0:
            raise TrainingError(
                "No DPO training data available. "
                "Ensure signals with accepted/rejected/edited events exist."
            )

        # Persist DPO pairs as training samples so train_result can use them
        samples_to_save: list[dict[str, Any]] = []
        for i in range(len(dpo_dataset)):
            session_id = str(dpo_dataset["session_id"][i])
            prompt = str(dpo_dataset["prompt"][i])
            chosen = str(dpo_dataset["chosen"][i])
            rejected = str(dpo_dataset["rejected"][i])
            source_event_ids = list(dpo_dataset["source_event_ids"][i] or [])
            pair_id = f"dpo_sig_{hashlib.sha256(f'{session_id}:{prompt}:{chosen}:{rejected}'.encode()).hexdigest()[:16]}"
            samples_to_save.append({
                "sample_id": pair_id,
                "sample_type": "dpo",
                "instruction": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "score": float(dpo_dataset["confidence"][i]),
                "source": "signal",
                "source_event_ids": source_event_ids,
                "source_adapter_version": None,
                "metadata": {
                    "session_id": session_id,
                    "dataset_split": "train",
                    "curated_from": "train_dpo",
                },
            })
        if samples_to_save:
            save_samples(samples_to_save)

        # Determine incremental context if base_adapter or base_adapter_path provided
        incremental_context = None
        if base_adapter:
            # Use _resolve_incremental_parent_context to properly resolve the base adapter
            incremental_context = self._resolve_incremental_parent_context(
                base_adapter=base_adapter,
                workspace=workspace,
            )
            # Override with DPO-specific context
            incremental_context["train_type"] = "dpo"
        elif base_adapter_path:
            incremental_context = {
                "parent_adapter_path": base_adapter_path,
                "resolved_base_model": base_model,
                "train_type": "dpo",
            }

        # Execute training through train_result
        return self.train_result(
            method=method,
            epochs=epochs,
            base_model=base_model,
            train_type="dpo",
            workspace=workspace,
            backend_hint=backend_hint,
            incremental_context=incremental_context,
        )

    def build_dpo_dataset(
        self,
        *,
        workspace: str | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Build and return DPO dataset statistics without training.

        Args:
            workspace: Workspace name
            min_confidence: Minimum signal confidence
            limit: Maximum number of pairs

        Returns:
            Dictionary with dataset info and statistics
        """
        trainer_config = self._load_trainer_config().trainer

        if min_confidence is None:
            min_confidence = trainer_config.dpo.min_preference_confidence
        if limit is None:
            limit = trainer_config.max_samples

        dataset_builder = DPODatasetBuilder(workspace=workspace)

        # Get statistics
        stats = dataset_builder.get_statistics()

        # Build dataset
        dpo_dataset = dataset_builder.build_from_signals(
            min_confidence=min_confidence,
            limit=limit,
        )

        return {
            "num_pairs": len(dpo_dataset),
            "min_confidence": min_confidence,
            "limit": limit,
            "statistics": stats,
            "columns": list(dpo_dataset.features.keys()) if hasattr(dpo_dataset, "features") else [],
        }
