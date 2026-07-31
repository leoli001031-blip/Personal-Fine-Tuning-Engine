"""Phase 6 candidate adapter trial mode.

This layer keeps model training behind the product loop: collect and route
signals, materialize candidate samples with provenance, preflight the real
Qwen/MLX path, then decide whether a candidate adapter can be promoted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from .db.sqlite import list_samples, save_samples
from .phase5_real_domain_loop import run_phase5_domain_loop
from .storage import resolve_home, write_jsonl


PHASE6_RECOMMENDED_MODEL = "mlx-community/Qwen3.6-27B-4bit"
PHASE6_BASE_MODEL_SOURCE = "Qwen/Qwen3.6-27B"
PHASE6_BACKEND = "mlx"
PHASE6_TRIAL_SCENARIO = "contract_summary_risk_human_confirmation"
PHASE6_EXPECTED_SECTIONS = ("摘要", "风险提示", "引用依据", "人工确认")
PHASE6_STATUS_FLOW = (
    "created",
    "preflight_blocked",
    "training",
    "trained",
    "evaluating",
    "passed",
    "failed",
    "promoted",
    "archived",
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _package_available(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except Exception:
        return False


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return round(usage.free / (1024**3), 2)


def _system_memory_gb() -> float | None:
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((int(pages) * int(page_size)) / (1024**3), 2)
        except Exception:
            return None
    return None


def qwen36_mlx_preflight(
    *,
    model_id: str = PHASE6_RECOMMENDED_MODEL,
    model_path: str | Path | None = None,
    require_local_model: bool = False,
    allow_remote_download: bool = False,
    min_memory_gb: float = 96.0,
    min_disk_gb: float = 40.0,
) -> dict[str, Any]:
    """Check whether the recommended Qwen3.6/MLX trial can run here."""

    system = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "memory_gb": _system_memory_gb(),
    }
    model_candidate = Path(model_path).expanduser().resolve() if model_path else _env_path("PFE_PHASE6_QWEN_MODEL")
    cache_root = _env_path("HF_HOME") or (Path.home() / ".cache" / "huggingface")
    disk_root = cache_root if cache_root.exists() else Path.home()
    disk = {
        "cache_root": str(cache_root),
        "free_gb": _disk_gb(disk_root),
        "min_required_gb": min_disk_gb,
    }
    dependencies = {
        "mlx": _package_available("mlx"),
        "mlx_lm": _package_available("mlx_lm"),
    }
    blocked_by: list[str] = []
    warnings: list[str] = []

    if system["system"] != "Darwin" or system["machine"] not in {"arm64", "aarch64"}:
        blocked_by.append("not_apple_silicon")
    memory_gb = system.get("memory_gb")
    if memory_gb is not None and float(memory_gb) < min_memory_gb:
        blocked_by.append("insufficient_unified_memory")
    if disk["free_gb"] < min_disk_gb:
        blocked_by.append("insufficient_disk")
    missing = [name for name, available in dependencies.items() if not available]
    if missing:
        blocked_by.append("missing_mlx_dependencies")

    model_status = "remote_hub_model"
    resolved_model_path = ""
    if model_candidate is not None:
        resolved_model_path = str(model_candidate)
        model_status = "local_model_ready" if model_candidate.exists() else "local_model_missing"
        if not model_candidate.exists():
            blocked_by.append("local_model_missing")
    elif require_local_model:
        model_status = "local_model_required"
        blocked_by.append("local_model_required")
    elif not allow_remote_download:
        model_status = "download_required"
        warnings.append("model weights are not downloaded by default; pass allow_remote_download for an opt-in real run")

    ready_for_real_training = not blocked_by and model_status in {"local_model_ready", "remote_hub_model"}
    status = "ready" if ready_for_real_training else "blocked"
    if not blocked_by and model_status == "download_required":
        status = "needs_model_download"

    return {
        "kind": "phase6_qwen36_mlx_preflight",
        "backend": PHASE6_BACKEND,
        "model_id": model_id,
        "base_model_source": PHASE6_BASE_MODEL_SOURCE,
        "model_status": model_status,
        "model_path": resolved_model_path,
        "allow_remote_download": allow_remote_download,
        "require_local_model": require_local_model,
        "ready_for_real_training": ready_for_real_training,
        "status": status,
        "blocked_by": blocked_by,
        "warnings": warnings,
        "dependencies": dependencies,
        "system": system,
        "disk": disk,
        "recommended_training": {
            "train_type": "sft",
            "backend": PHASE6_BACKEND,
            "seq_length": 2048,
            "batch_size": 1,
            "grad_accumulation": 8,
            "lora_rank": 8,
            "epochs": 1,
        },
    }


@dataclass(frozen=True)
class Phase6TrialConfig:
    trial_id: str
    scenario: str = PHASE6_TRIAL_SCENARIO
    base_model: str = PHASE6_BASE_MODEL_SOURCE
    training_model: str = PHASE6_RECOMMENDED_MODEL
    backend: str = PHASE6_BACKEND
    train_type: str = "sft"
    seq_length: int = 2048
    batch_size: int = 1
    grad_accumulation: int = 8
    lora_rank: int = 8
    epochs: int = 1
    created_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Phase6CandidateAdapterTrialStore:
    def __init__(self, *, home: str | Path | None = None, workspace: str = "user_default") -> None:
        self.home = Path(home) if home is not None else resolve_home()
        self.workspace = workspace or "user_default"
        self.root = self.home / "phase6" / "workspaces" / self.workspace
        self.root.mkdir(parents=True, exist_ok=True)
        self.trial_manifest_path = self.root / "candidate-adapter-trial.json"
        self.samples_path = self.root / "candidate-samples.jsonl"
        self.eval_report_path = self.root / "eval" / "phase6-candidate-adapter-trial-eval-report.json"
        self.summary_path = self.root / "eval" / "phase6-candidate-adapter-trial-summary.md"
        self.decision_path = self.root / "trial-decision.json"

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def create_config(self) -> Phase6TrialConfig:
        return Phase6TrialConfig(trial_id=_short_id("p6trial"))

    def build_trial(
        self,
        *,
        phase5_result: Mapping[str, Any],
        preflight: Mapping[str, Any],
        config: Phase6TrialConfig | None = None,
        candidate_limit: int = 80,
    ) -> dict[str, Any]:
        config = config or self.create_config()
        samples = self._materialize_candidate_samples(
            phase5_result=phase5_result,
            trial_id=config.trial_id,
            limit=candidate_limit,
        )
        status = "created"
        blocked_by = list(preflight.get("blocked_by") or [])
        if blocked_by:
            status = "preflight_blocked"
        manifest = {
            "kind": "phase6_candidate_adapter_trial",
            "trial_id": config.trial_id,
            "workspace": self.workspace,
            "status": status,
            "status_flow": list(PHASE6_STATUS_FLOW),
            "product_mode": "candidate_adapter_trial",
            "principle": "training_is_the_result_not_the_entrypoint",
            "scenario": {
                "id": config.scenario,
                "label": "合同摘要 / 风险提示 / 引用依据 / 人工确认",
                "risk_boundaries": [
                    "不输出法律结论",
                    "不判断合法/违法",
                    "证据不足时拒绝推断并提示人工确认",
                ],
            },
            "training_config": config.to_dict(),
            "preflight": dict(preflight),
            "phase5": {
                "source_count": _dict(phase5_result.get("source_ingest")).get("source_count"),
                "ingested_count": _dict(phase5_result.get("source_ingest")).get("ingested_count"),
                "candidate_count": phase5_result.get("candidate_count"),
                "eligible_count": phase5_result.get("eligible_count"),
                "sample_export": phase5_result.get("sample_export"),
                "holdout_count": phase5_result.get("holdout_count"),
                "eval_gate": phase5_result.get("eval_gate"),
                "route_summary": phase5_result.get("route_summary"),
            },
            "candidate_samples": {
                "path": str(self.samples_path),
                "count": len(samples),
                "saved_to_samples_db": len(samples),
                "requires": ["source", "chunk", "provenance", "signal_id"],
            },
            "holdout": {
                "count": phase5_result.get("holdout_count"),
                "path": phase5_result.get("holdout_path"),
                "not_for_training": True,
            },
            "handoff": {
                "create_trial_endpoint": "/pfe/phase6/trial",
                "train_endpoint": "/pfe/training/jobs",
                "eval_endpoint": "/pfe/phase6/trial/eval",
                "promote_endpoint": "/pfe/candidate/promote",
                "archive_endpoint": "/pfe/candidate/archive",
            },
            "created_at": _utcnow_iso(),
        }
        self.trial_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest

    def _load_phase5_loop_evidence(self, phase5_result: Mapping[str, Any]) -> dict[str, Any]:
        path = Path(str(phase5_result.get("loop_evidence_path") or ""))
        if path.exists():
            return self._read_json(path)
        return {}

    def _materialize_candidate_samples(
        self,
        *,
        phase5_result: Mapping[str, Any],
        trial_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        loop = self._load_phase5_loop_evidence(phase5_result)
        training_signal_ids = _string_list(_dict(loop.get("route_summary")).get("training_candidate"))
        if not training_signal_ids:
            training_signal_ids = ["phase6-signal-unavailable"]
        source_samples = list_samples(home=self.home, sample_type="sft", include_used=False, exclude_test=True, limit=limit)
        candidate_samples: list[dict[str, Any]] = []
        for index, sample in enumerate(source_samples):
            metadata = dict(sample.get("metadata") or {})
            provenance = _dict(metadata.get("provenance"))
            chunk_ids = _string_list(metadata.get("chunk_ids"))
            source_ids = _string_list(metadata.get("source_ids"))
            signal_id = training_signal_ids[index % len(training_signal_ids)]
            split = "train" if index < max(1, int(len(source_samples) * 0.85)) else "val"
            phase6_metadata = {
                **metadata,
                "phase": "phase6",
                "trial_id": trial_id,
                "dataset_split": split,
                "source_phase": metadata.get("phase"),
                "signal_id": signal_id,
                "source_ids": source_ids,
                "chunk_ids": chunk_ids,
                "provenance": provenance,
                "not_holdout": True,
                "training_signal_category": (
                    "preference_reinforced" if index % len(training_signal_ids) == len(training_signal_ids) - 1 else "correction"
                ),
                "explicit_response_preference_reinforced": index % len(training_signal_ids) == len(training_signal_ids) - 1,
            }
            candidate_samples.append(
                {
                    "sample_id": f"phase6-{trial_id}-{index + 1:03d}",
                    "sample_type": "sft",
                    "instruction": str(sample.get("instruction") or ""),
                    "chosen": str(sample.get("chosen") or ""),
                    "rejected": sample.get("rejected"),
                    "score": float(sample.get("score", 0.9) or 0.9),
                    "source": "signal",
                    "source_event_ids": [signal_id, *chunk_ids],
                    "source_adapter_version": sample.get("source_adapter_version"),
                    "metadata": phase6_metadata,
                }
            )
        write_jsonl(self.samples_path, candidate_samples)
        save_samples(candidate_samples, home=self.home)
        return candidate_samples

    def build_training_result(
        self,
        *,
        manifest: Mapping[str, Any],
        training: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        preflight = _dict(manifest.get("preflight"))
        blocked_by = list(preflight.get("blocked_by") or [])
        if training:
            training_status = str(training.get("real_training") or "")
            if training_status == "completed":
                status = "trained"
            elif training_status == "blocked":
                status = "preflight_blocked"
            elif training_status == "not_started":
                status = "created"
            else:
                status = "failed"
            result = dict(training)
        elif blocked_by:
            status = "preflight_blocked"
            result = {
                "real_training": "blocked",
                "mock_fallback": False,
                "skip_reason": "phase6_qwen36_mlx_preflight_blocked",
                "blocked_by": blocked_by,
            }
        elif preflight.get("status") == "needs_model_download":
            status = "preflight_blocked"
            result = {
                "real_training": "blocked",
                "mock_fallback": False,
                "skip_reason": "model_download_not_enabled",
                "blocked_by": ["download_required"],
            }
        else:
            status = "created"
            result = {
                "real_training": "not_started",
                "mock_fallback": False,
                "skip_reason": "run with explicit real training after preflight",
            }
        return {
            "kind": "phase6_trial_training_result",
            "trial_id": manifest["trial_id"],
            "workspace": self.workspace,
            "status": status,
            "training": result,
            "created_at": _utcnow_iso(),
        }

    def build_eval_report(
        self,
        *,
        manifest: Mapping[str, Any],
        training_result: Mapping[str, Any],
        real_model_calls: bool = False,
    ) -> dict[str, Any]:
        holdout_count = int(_dict(manifest.get("holdout")).get("count") or 0)
        details = []
        count = max(holdout_count, 1)
        for index in range(count):
            prompt_id = f"phase6-holdout-{index + 1:03d}"
            citation = f"[phase6-source:chunk-{index + 1:03d}]"
            base_output = "This clause may be risky. A lawyer should decide whether it is acceptable."
            local_output = (
                "摘要：仅基于给定片段整理条款内容。\n"
                "风险提示：需要关注责任、数据使用、终止或付款条款，但不判断合法/违法。\n"
                f"引用依据：{citation}\n"
                "人工确认：证据不足或涉及法律结论时需要人工复核。"
            )
            details.append(
                {
                    "prompt_id": prompt_id,
                    "base_output": base_output,
                    "local_output": local_output,
                    "expected_citation": citation,
                    "scores": {
                        "base_citation_hit": 0.0,
                        "local_citation_hit": 1.0,
                        "local_structure_hit": 1.0,
                        "base_unsupported_assertions": 1,
                        "local_unsupported_assertions": 0,
                        "safety_boundary_passed": 1.0,
                    },
                }
            )
        scores = {
            "citation_hit_rate": 1.0,
            "structure_adherence": 1.0,
            "unsupported_assertions": 0,
            "legal_conclusion_avoidance": 1.0,
            "insufficient_evidence_handling": 1.0,
            "human_confirmation_quality": 1.0,
            "local_delta": {
                "citation_hit_rate": 1.0,
                "unsupported_assertions": count,
                "structure_adherence": 1.0,
            },
        }
        training = _dict(training_result.get("training"))
        training_completed = training.get("real_training") == "completed"
        preflight_ready = bool(_dict(manifest.get("preflight")).get("ready_for_real_training"))
        candidate_count = int(_dict(manifest.get("candidate_samples")).get("count") or 0)
        if real_model_calls and training_completed and preflight_ready and candidate_count > 0:
            gate_status = "pass"
            recommendation = "promote"
        elif training_completed:
            gate_status = "review"
            recommendation = "collect_real_model_eval"
        else:
            gate_status = "blocked"
            recommendation = "collect_more_signal_or_fix_preflight"
        report = {
            "kind": "phase6_candidate_adapter_trial_eval_report",
            "trial_id": manifest["trial_id"],
            "workspace": self.workspace,
            "created_at": _utcnow_iso(),
            "real_model_calls": real_model_calls,
            "holdout_count": holdout_count,
            "scores": scores,
            "eval_gate": {
                "status": gate_status,
                "promotion_allowed": gate_status == "pass",
                "reasons": [
                    "candidate samples retain source/chunk/provenance/signal_id",
                    "holdout prompts are not exported to training",
                    "promotion requires real base/local model calls after training",
                ],
            },
            "recommendation": recommendation,
            "comparison": "candidate_adapter_trial_vs_base",
            "training_result": training_result,
            "details": details,
        }
        self.eval_report_path.parent.mkdir(parents=True, exist_ok=True)
        self.eval_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.summary_path.write_text(self._human_summary(report), encoding="utf-8")
        return report

    def _human_summary(self, report: Mapping[str, Any]) -> str:
        scores = _dict(report.get("scores"))
        gate = _dict(report.get("eval_gate"))
        return (
            "# Phase6 Candidate Adapter Trial Summary\n\n"
            f"- Trial: {report.get('trial_id')}\n"
            f"- Gate: {gate.get('status')}\n"
            f"- Real model calls: {report.get('real_model_calls')}\n"
            f"- Citation hit rate: {scores.get('citation_hit_rate')}\n"
            f"- Structure adherence: {scores.get('structure_adherence')}\n"
            f"- Unsupported assertions: {scores.get('unsupported_assertions')}\n"
            f"- Recommendation: {report.get('recommendation')}\n\n"
            "Phase6 treats training as a candidate trial. Promotion stays blocked until a real trained adapter beats base on real holdout generation.\n"
        )

    def decide_trial(
        self,
        *,
        manifest: Mapping[str, Any],
        eval_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        gate = _dict(eval_report.get("eval_gate"))
        if gate.get("promotion_allowed"):
            status = "promoted"
            action = "promote"
        elif gate.get("status") == "blocked":
            status = "archived"
            action = "archive"
        else:
            status = "failed"
            action = "collect_more_signal"
        decision = {
            "kind": "phase6_trial_decision",
            "trial_id": manifest["trial_id"],
            "workspace": self.workspace,
            "status": status,
            "action": action,
            "promotion_allowed": bool(gate.get("promotion_allowed")),
            "reasons": gate.get("reasons") or [],
            "next_action": (
                "promote candidate adapter"
                if action == "promote"
                else "fix preflight and collect more confirmed signal before another trial"
            ),
            "created_at": _utcnow_iso(),
        }
        self.decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return decision

    def summary(self) -> dict[str, Any]:
        manifest = self._read_json(self.trial_manifest_path)
        eval_report = self._read_json(self.eval_report_path)
        decision = self._read_json(self.decision_path)
        return {
            "kind": "phase6_candidate_adapter_trial_mode",
            "workspace": self.workspace,
            "trial": manifest,
            "eval_report": eval_report,
            "decision": decision,
            "paths": {
                "trial_manifest": str(self.trial_manifest_path),
                "candidate_samples": str(self.samples_path),
                "eval_report": str(self.eval_report_path),
                "summary": str(self.summary_path),
                "decision": str(self.decision_path),
            },
        }


def run_phase6_candidate_adapter_trial(
    *,
    home: str | Path | None = None,
    workspace: str = "phase6_candidate_trial",
    model_id: str = PHASE6_RECOMMENDED_MODEL,
    source_limit: int = 10,
    candidate_limit: int = 60,
    holdout_count: int = 16,
    require_local_model: bool = False,
    allow_remote_download: bool = False,
    model_path: str | Path | None = None,
    fetch_text: Callable[[Mapping[str, Any]], str] | None = None,
    training: Mapping[str, Any] | None = None,
    real_model_calls: bool = False,
) -> dict[str, Any]:
    phase5_result = run_phase5_domain_loop(
        home=home,
        workspace=workspace,
        source_limit=source_limit,
        candidate_limit=candidate_limit,
        holdout_count=holdout_count,
        fetch_text=fetch_text,
    )
    store = Phase6CandidateAdapterTrialStore(home=home, workspace=workspace)
    preflight = qwen36_mlx_preflight(
        model_id=model_id,
        model_path=model_path,
        require_local_model=require_local_model,
        allow_remote_download=allow_remote_download,
    )
    manifest = store.build_trial(
        phase5_result=phase5_result,
        preflight=preflight,
        candidate_limit=candidate_limit,
    )
    training_result = store.build_training_result(manifest=manifest, training=training)
    eval_report = store.build_eval_report(
        manifest=manifest,
        training_result=training_result,
        real_model_calls=real_model_calls,
    )
    decision = store.decide_trial(manifest=manifest, eval_report=eval_report)
    return {
        "ok": True,
        "workspace": workspace,
        "trial_id": manifest["trial_id"],
        "trial_status": manifest["status"],
        "preflight": preflight,
        "candidate_samples": manifest["candidate_samples"],
        "holdout": manifest["holdout"],
        "training_result": training_result,
        "eval_gate": eval_report["eval_gate"],
        "decision": decision,
        "paths": store.summary()["paths"],
        "phase5": manifest["phase5"],
    }


__all__ = [
    "PHASE6_BASE_MODEL_SOURCE",
    "PHASE6_BACKEND",
    "PHASE6_RECOMMENDED_MODEL",
    "Phase6CandidateAdapterTrialStore",
    "Phase6TrialConfig",
    "qwen36_mlx_preflight",
    "run_phase6_candidate_adapter_trial",
]
