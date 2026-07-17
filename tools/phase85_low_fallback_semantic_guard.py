#!/usr/bin/env python3
"""Run the Phase85 low-fallback semantic-guard simulated benchmark."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
SERVER_ROOT = REPO_ROOT / "pfe-server"
for import_root in (CORE_ROOT, SERVER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pfe_core.inference.engine import InferenceEngine
from pfe_core.phase75_personalization_benefit_benchmark import (
    score_phase75_transcript,
    stable_hash,
)
from pfe_core.phase77_private_value_guarded_runtime import (
    build_phase77_holdout,
    guard_phase77_messages,
    guard_phase77_output,
)
from pfe_core.phase78_persona_internalization_training import build_phase78_holdout
from pfe_core.phase79_cpu_feasible_persona_probe import build_phase79_holdout
from pfe_core.phase80_small_model_failure_taxonomy import build_phase80_holdout
from pfe_core.phase81_trainable_mid_model_selection import build_phase81_holdout
from pfe_core.phase83_persona_route_length_repair import build_phase83_holdout
from pfe_core.phase84_factual_completion_guard import (
    build_phase84_holdout,
    enforce_phase84_persona_output,
)
from pfe_core.phase85_low_fallback_semantic_guard import (
    PHASE85_CONTROL_COUNT,
    PHASE85_FALLBACK_MAXIMUM,
    PHASE85_FORMAT_ELIGIBLE_TURN_COUNT,
    PHASE85_NATIVE_FORMAT_MINIMUM,
    PHASE85_SEMANTIC_REPAIR_MAXIMUM,
    PHASE85_SESSION_COUNT,
    PHASE85_TARGET_CATEGORY_FLOOR,
    PHASE85_TARGET_COUNT,
    PHASE85_TARGET_GAIN_MINIMUM,
    PHASE85_TARGET_SCORE_MINIMUM,
    PHASE85_VARIANTS,
    audit_phase85_isolation,
    audit_phase85_routes,
    build_phase85_decision,
    build_phase85_guard_calibration,
    build_phase85_holdout,
    enforce_phase85_persona_output,
    evaluate_phase85_guard_calibration,
)
from pfe_core.pipeline import PipelineService


DEFAULT_EVIDENCE_ROOT = REPO_ROOT / "docs/demo/phase85-low-fallback-semantic-guard"
DEFAULT_MODEL_PATH = REPO_ROOT / "models/Qwen2.5-1.5B-Instruct"
DEFAULT_REVIEW_CACHE_ROOT = Path("/private/tmp/pfe-phase85-manual-review")
SYSTEM_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()
POSIX_TEMP_ROOT = Path("/tmp").resolve()
REVIEW_OUTPUT_KEY_MANIFEST = "review-output-key-manifest.json"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DRIVER_SOURCE = REPO_ROOT / "tools/phase85_low_fallback_semantic_guard.py"
CORE_SOURCE = CORE_ROOT / "pfe_core/phase85_low_fallback_semantic_guard.py"
TEST_SOURCE = REPO_ROOT / "tests/test_phase85_low_fallback_semantic_guard.py"
DRIVER_TEST_SOURCE = REPO_ROOT / "tests/test_phase85_driver_safety.py"
SEMANTIC_HARDENING_TEST_SOURCE = REPO_ROOT / "tests/test_phase85_semantic_guard_hardening.py"
ENGINE_PRIVACY_TEST_SOURCE = REPO_ROOT / "tests/test_phase85_engine_status_privacy.py"
INFERENCE_RUNTIME_TEST_SOURCE = REPO_ROOT / "tests/test_inference_runtime.py"
SCORER_SOURCE = CORE_ROOT / "pfe_core/phase75_personalization_benefit_benchmark.py"
CONTRACT_SOURCE = CORE_ROOT / "pfe_core/inference/contracts.py"
ENGINE_SOURCE = CORE_ROOT / "pfe_core/inference/engine.py"
BACKENDS_SOURCE = CORE_ROOT / "pfe_core/inference/backends.py"
PIPELINE_SOURCE = CORE_ROOT / "pfe_core/pipeline.py"
PHASE84_SOURCE = CORE_ROOT / "pfe_core/phase84_factual_completion_guard.py"
PHASE77_SOURCE = CORE_ROOT / "pfe_core/phase77_private_value_guarded_runtime.py"
PHASE78_SOURCE = CORE_ROOT / "pfe_core/phase78_persona_internalization_training.py"
PHASE79_SOURCE = CORE_ROOT / "pfe_core/phase79_cpu_feasible_persona_probe.py"
PHASE80_SOURCE = CORE_ROOT / "pfe_core/phase80_small_model_failure_taxonomy.py"
PHASE81_SOURCE = CORE_ROOT / "pfe_core/phase81_trainable_mid_model_selection.py"
PHASE83_SOURCE = CORE_ROOT / "pfe_core/phase83_persona_route_length_repair.py"

VARIANT_CONTRACTS: dict[str, str | None] = {
    PHASE85_VARIANTS[0]: None,
    PHASE85_VARIANTS[1]: "contract_persona_guarded_v3",
    PHASE85_VARIANTS[2]: "contract_persona_guarded_v4",
}
FORMAT_OUTCOMES = (
    "native",
    "semantic_repair",
    "format_fallback",
    "safety_fallback",
)
FALLBACK_OUTCOMES = frozenset({"format_fallback", "safety_fallback"})
GENERATION_PROTOCOL_BASE = {
    "kind": "phase85_frozen_three_arm_low_fallback_generation_protocol",
    "benchmark_type": "simulated benchmark",
    "api_surface": "PipelineService.chat_completion",
    "base_model": "Qwen2.5-1.5B-Instruct",
    "model_revision": MODEL_REVISION,
    "variants": {
        name: {"response_contract": contract} for name, contract in VARIANT_CONTRACTS.items()
    },
    "max_tokens": 160,
    "temperature": 0.0,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 4,
    "sessions_per_variant": PHASE85_SESSION_COUNT,
    "turns_per_session": 3,
    "api_calls_per_variant": PHASE85_SESSION_COUNT * 3,
    "exactly_one_api_call_per_turn": True,
    "exactly_one_backend_attempt_per_turn": True,
    "same_model_and_decoding_all_variants": True,
    "ordinary_route_prompt_guard_off_required": True,
    "raw_model_text_persistence_allowed": False,
    "private_source_persistence_allowed": False,
    "automatic_promotion_allowed": False,
}
FROZEN_THRESHOLDS = {
    "required_session_count_per_variant": PHASE85_SESSION_COUNT,
    "required_api_call_count_per_variant": PHASE85_SESSION_COUNT * 3,
    "required_backend_attempt_count_per_turn": 1,
    "required_model_call_count_per_variant": PHASE85_SESSION_COUNT * 3,
    "required_persona_target_count": PHASE85_TARGET_COUNT,
    "required_ordinary_control_count": PHASE85_CONTROL_COUNT,
    "required_route_accuracy": 1.0,
    "required_holdout_isolation_pass_rate": 1.0,
    "required_guard_calibration_pass_rate": 1.0,
    "required_guard_calibration_block_recall": 1.0,
    "maximum_guard_calibration_false_block_rate": 0.0,
    "required_format_partition_identity_rate": 1.0,
    "required_format_eligible_turn_count_per_variant": PHASE85_FORMAT_ELIGIBLE_TURN_COUNT,
    "minimum_v4_native_format_turn_rate": PHASE85_NATIVE_FORMAT_MINIMUM,
    "maximum_v4_semantic_repair_turn_rate": PHASE85_SEMANTIC_REPAIR_MAXIMUM,
    "maximum_v4_fallback_turn_rate": PHASE85_FALLBACK_MAXIMUM,
    "minimum_v4_target_score": PHASE85_TARGET_SCORE_MINIMUM,
    "minimum_v4_target_category_floor": PHASE85_TARGET_CATEGORY_FLOOR,
    "minimum_v4_target_gain_vs_base": PHASE85_TARGET_GAIN_MINIMUM,
    "maximum_post_guard_unsupported_completion_rate": 0.0,
    "minimum_unsupported_completion_block_recall": 1.0,
    "maximum_false_block_rate": 0.0,
    "required_ordinary_three_arm_byte_identity_rate": 1.0,
    "required_ordinary_route_off_rate": 1.0,
    "required_ordinary_system_prompt_off_rate": 1.0,
    "required_ordinary_guard_off_rate": 1.0,
    "maximum_privacy_echo_rate": 0.0,
    "maximum_think_leak_rate": 0.0,
    "manual_review_may_upgrade_failure": False,
    "automatic_promotion_allowed": False,
    "score_or_gate_relaxation_allowed": False,
}
DYNAMIC_FILES = {
    "evidence_manifest.json",
    "evidence_integrity.json",
    "finalization_state.json",
    "validation_gate.txt",
    "validation_summary.json",
}
FORBIDDEN_EVIDENCE_KEYS = {
    "command",
    "content",
    "messages",
    "raw_model_text",
    "raw_output",
    "raw_text",
    "response_text",
    "stderr",
    "turns",
    "user_goal",
    "user_correction",
    "continuation_request",
    "acceptance_request",
    "output",
    "declared_private_values",
}


@dataclass(frozen=True)
class DriverConfig:
    evidence_root: Path
    model_path: Path
    mode: str

    @property
    def preparation_root(self) -> Path:
        return self.evidence_root / "evidence-preparation"

    @property
    def generation_root(self) -> Path:
        return self.evidence_root / "evidence-generation"

    @property
    def failure_root(self) -> Path:
        return self.evidence_root / "evidence-failures"


class _ChatCompletionFailure(RuntimeError):
    def __init__(
        self,
        cause: Exception,
        *,
        api_call_count: int,
        turn_metadata: Sequence[Mapping[str, Any]],
    ) -> None:
        super().__init__(
            f"chat_completion failed after {api_call_count} API call attempt(s)"
        )
        self.api_call_count = api_call_count
        self.turn_metadata = [dict(row) for row in turn_metadata]
        self.original_error_type = cause.__class__.__name__
        self.error_fingerprint = _text_sha256(
            f"{cause.__class__.__name__}:{str(cause)}"
        )


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        dict(value)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in (json.loads(line),)
        if isinstance(value, Mapping)
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(
                "".join(
                    json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
                    for row in rows
                )
            )
        temporary.chmod(0o600)
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _resolve_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    return candidate.resolve()


def _lexical_absolute(path: Path) -> Path:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    return Path(os.path.abspath(candidate))


def _path_has_symlink(path: Path, root: Path, *, allow_root_symlink: bool = False) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    if root.is_symlink() and not allow_root_symlink:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _assert_safe_child_path(path: Path, root: Path) -> tuple[Path, Path]:
    lexical = _lexical_absolute(path)
    lexical_root = _lexical_absolute(root)
    if (
        lexical == lexical_root
        or not lexical.is_relative_to(lexical_root)
        or _path_has_symlink(lexical, lexical_root)
    ):
        raise ValueError(f"refusing unsafe child path: {lexical}")
    return lexical, lexical_root


def _safe_unlink_under(path: Path, root: Path) -> None:
    lexical, _ = _assert_safe_child_path(path, root)
    if lexical.exists() or lexical.is_symlink():
        if lexical.is_dir() and not lexical.is_symlink():
            raise ValueError(f"refusing to unlink directory as file: {lexical}")
        lexical.unlink()
    if lexical.exists() or lexical.is_symlink():
        raise RuntimeError(f"failed to remove file: {lexical}")


def _safe_clean_directory(path: Path) -> None:
    lexical = _lexical_absolute(path)
    resolved = lexical.resolve()
    repo_root = REPO_ROOT.resolve()
    docs_root_lexical = _lexical_absolute(REPO_ROOT / "docs")
    docs_demo_lexical = _lexical_absolute(REPO_ROOT / "docs/demo")
    docs_demo_root = docs_demo_lexical.resolve()
    temp_roots = {SYSTEM_TEMP_ROOT.resolve(), POSIX_TEMP_ROOT.resolve()}
    unsafe_roots = {
        Path("/").resolve(),
        repo_root,
        repo_root.parent,
        docs_demo_root,
        Path.home().resolve(),
        *temp_roots,
    }
    docs_allowed = resolved != docs_demo_root and resolved.is_relative_to(docs_demo_root)
    if docs_allowed and (
        docs_root_lexical.is_symlink()
        or docs_demo_lexical.is_symlink()
        or not lexical.is_relative_to(docs_demo_lexical)
        or _path_has_symlink(lexical, docs_demo_lexical)
    ):
        raise ValueError(f"refusing to remove unsafe evidence directory: {resolved}")

    temp_allowed = any(
        resolved != root and resolved.is_relative_to(root) for root in temp_roots
    )
    if temp_allowed:
        lexical_temp_roots = [
            (_lexical_absolute(SYSTEM_TEMP_ROOT), False),
            (_lexical_absolute(POSIX_TEMP_ROOT), False),
        ]
        macos_tmp = Path("/tmp")
        if macos_tmp.resolve() == POSIX_TEMP_ROOT.resolve():
            lexical_temp_roots.append((macos_tmp, True))
        matching_roots = [
            (root, allow_root_symlink)
            for root, allow_root_symlink in lexical_temp_roots
            if lexical.is_relative_to(root)
        ]
        if not matching_roots or all(
            _path_has_symlink(
                lexical,
                root,
                allow_root_symlink=allow_root_symlink,
            )
            for root, allow_root_symlink in matching_roots
        ):
            raise ValueError(f"refusing to remove unsafe evidence directory: {resolved}")

    allowed = docs_allowed or temp_allowed
    if resolved in unsafe_roots or not allowed:
        raise ValueError(f"refusing to remove unsafe evidence directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    if resolved.exists():
        raise RuntimeError(f"failed to clean evidence directory: {resolved}")


def _source_hashes() -> dict[str, str | None]:
    sources = {
        "phase85_driver": DRIVER_SOURCE,
        "phase85_core": CORE_SOURCE,
        "phase85_tests": TEST_SOURCE,
        "phase85_driver_tests": DRIVER_TEST_SOURCE,
        "phase85_semantic_hardening_tests": SEMANTIC_HARDENING_TEST_SOURCE,
        "phase85_engine_privacy_tests": ENGINE_PRIVACY_TEST_SOURCE,
        "inference_runtime_tests": INFERENCE_RUNTIME_TEST_SOURCE,
        "phase75_scorer": SCORER_SOURCE,
        "phase77_private_guard": PHASE77_SOURCE,
        "phase84_factual_guard": PHASE84_SOURCE,
        "runtime_contracts": CONTRACT_SOURCE,
        "runtime_engine": ENGINE_SOURCE,
        "runtime_backends": BACKENDS_SOURCE,
        "runtime_pipeline": PIPELINE_SOURCE,
        "phase78_holdout": PHASE78_SOURCE,
        "phase79_holdout": PHASE79_SOURCE,
        "phase80_holdout": PHASE80_SOURCE,
        "phase81_holdout": PHASE81_SOURCE,
        "phase83_holdout": PHASE83_SOURCE,
    }
    return {name: _sha256(path) if path.is_file() else None for name, path in sources.items()}


def _model_artifact_paths(model_path: Path) -> list[Path]:
    if not model_path.is_dir():
        return []
    fixed_names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    paths = [path for path in model_path.iterdir() if path.is_file() and path.name in fixed_names]
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        paths.extend(path for path in model_path.glob(pattern) if path.is_file())
    return sorted(set(paths), key=lambda path: path.name)


def _model_manifest(config: DriverConfig) -> dict[str, Any]:
    if config.mode == "mock":
        descriptor = {
            "mode": "mock",
            "model_path": str(config.model_path),
            "protocol": "PipelineService deterministic mock path",
        }
        return {
            "kind": "phase85_model_manifest",
            "mode": "mock",
            "model_path": str(config.model_path),
            "complete": True,
            "files": [],
            "manifest_sha256": stable_hash(descriptor),
        }
    artifacts = _model_artifact_paths(config.model_path)
    files = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in artifacts
    ]
    names = {row["name"] for row in files}
    has_weights = any(
        name.endswith(".safetensors") or re.fullmatch(r"pytorch_model.*\.bin", name)
        for name in names
    )
    has_tokenizer = bool(
        names & {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
    )
    complete = "config.json" in names and has_weights and has_tokenizer
    return {
        "kind": "phase85_model_manifest",
        "mode": "real",
        "model_path": str(config.model_path),
        "revision": MODEL_REVISION,
        "complete": complete,
        "files": files,
        "file_count": len(files),
        "manifest_sha256": stable_hash(files),
    }


def _generation_protocol(config: DriverConfig) -> dict[str, Any]:
    return {
        **GENERATION_PROTOCOL_BASE,
        "execution_mode": config.mode,
        "resolved_model_path": str(config.model_path),
    }


def _previous_holdouts() -> list[dict[str, Any]]:
    return (
        build_phase77_holdout()["sessions"]
        + build_phase78_holdout()["sessions"]
        + build_phase79_holdout()["sessions"]
        + build_phase80_holdout()["sessions"]
        + build_phase81_holdout()["sessions"]
        + build_phase83_holdout()["sessions"]
        + build_phase84_holdout()["sessions"]
    )


def _redacted_holdout_manifest(holdout: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for source in holdout.get("sessions") or []:
        session = dict(source)
        expected = dict(session.get("expected") or {})
        private_values = [value for value in session.get("declared_private_values") or [] if value]
        rows.append(
            {
                "session_id": session.get("session_id"),
                "category": session.get("category"),
                "task_type": session.get("task_type"),
                "turn_count": 3,
                "expected_routes": [bool(value) for value in session.get("expected_routes") or []],
                "required_group_count": len(expected.get("required_groups") or []),
                "required_label_count": len(expected.get("required_labels") or []),
                "has_private_source": bool(private_values),
                "private_source_count": len(private_values),
                "not_for_training": session.get("not_for_training") is True,
                "simulated_usage": session.get("simulated_usage") is True,
                "source_sha256": stable_hash(session),
            }
        )
    return {
        "kind": "phase85_redacted_holdout_manifest",
        "benchmark_type": "simulated benchmark",
        "session_count": len(rows),
        "persona_target_count": sum(row["task_type"] == "persona_target" for row in rows),
        "ordinary_control_count": sum(row["task_type"] == "ordinary_control" for row in rows),
        "format_eligible_turn_count": sum(
            sum(row["expected_routes"]) for row in rows
        ),
        "raw_source_persisted": False,
        "private_source_persisted": False,
        "holdout_manifest_sha256": stable_hash(holdout.get("sessions") or []),
        "sessions": rows,
    }


def _sanitize_isolation_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    training_overlap = list(audit.get("training_text_overlap") or [])
    previous_overlap = list(audit.get("previous_holdout_text_overlap") or [])
    near_duplicates = [dict(row) for row in audit.get("near_duplicate_overlaps") or []]
    phase85_near_duplicates = [
        dict(row) for row in audit.get("phase85_near_duplicate_overlaps") or []
    ]
    calibration_near_duplicates = [
        dict(row)
        for row in audit.get("phase85_guard_calibration_near_duplicate_overlaps")
        or []
    ]
    return {
        "kind": "phase85_redacted_training_holdout_isolation_audit",
        "passed": audit.get("passed") is True,
        "checks": dict(audit.get("checks") or {}),
        "training_text_overlap_count": len(training_overlap),
        "training_text_overlap_sha256": stable_hash(training_overlap),
        "previous_holdout_text_overlap_count": len(previous_overlap),
        "previous_holdout_text_overlap_sha256": stable_hash(previous_overlap),
        "near_duplicate_threshold": audit.get("near_duplicate_threshold"),
        "near_duplicate_overlaps": near_duplicates,
        "phase85_near_duplicate_threshold": audit.get(
            "phase85_near_duplicate_threshold"
        ),
        "phase85_near_duplicate_overlap_count": len(phase85_near_duplicates),
        "phase85_near_duplicate_overlaps": phase85_near_duplicates,
        "guard_calibration_near_duplicate_overlap_count": len(
            calibration_near_duplicates
        ),
        "guard_calibration_near_duplicate_overlaps": calibration_near_duplicates,
        "raw_source_persisted": False,
    }


def _redacted_calibration_manifest(calibration: Mapping[str, Any]) -> dict[str, Any]:
    cases = [dict(row) for row in calibration.get("cases") or []]
    return {
        "kind": "phase85_redacted_independent_guard_calibration_manifest",
        "sealed_before_evaluation": calibration.get("sealed_before_evaluation") is True,
        "not_for_training": calibration.get("not_for_training") is True,
        "case_count": len(cases),
        "expected_label_sha256": calibration.get("expected_label_sha256"),
        "calibration_source_sha256": stable_hash(calibration),
        "cases": [
            {
                "case_id": row.get("case_id"),
                "expected_action": row.get("expected_action"),
                "case_source_sha256": stable_hash(row),
            }
            for row in cases
        ],
        "raw_source_persisted": False,
        "raw_model_text_persisted": False,
    }


def _format_outcome(evaluation: Mapping[str, Any]) -> str | None:
    if evaluation.get("format_eligible") is not True:
        return None
    buckets = {
        "native": evaluation.get("native_format") is True,
        "semantic_repair": evaluation.get("semantic_repair_used") is True,
        "format_fallback": evaluation.get("format_fallback_used") is True,
        "safety_fallback": evaluation.get("safety_fallback_used") is True,
    }
    selected = [name for name, value in buckets.items() if value]
    if len(selected) != 1:
        raise AssertionError(f"non-exclusive Phase85 format accounting: {buckets}")
    return selected[0]


def _v3_evaluation(contract_output: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    format_eligible = route.get("routed") is True
    fallback = format_eligible and contract_output.get("fallback_used") is True
    safety = fallback and str(contract_output.get("fallback_reason") or "") == (
        "unsupported_completion_claim"
    )
    cleaned_structure = (
        format_eligible
        and not fallback
        and (
            contract_output.get("preamble_removed") is True
            or contract_output.get("extra_text_removed") is True
        )
    )
    return {
        **dict(contract_output),
        "route": dict(route),
        "format_eligible": format_eligible,
        "native_format": format_eligible and not fallback and not cleaned_structure,
        "semantic_repair_used": cleaned_structure,
        "format_fallback_used": fallback and not safety,
        "safety_fallback_used": safety,
        "fallback_used": fallback,
        "privacy_transform_applied": dict(contract_output.get("output_guard") or {}).get(
            "raw_model_private_echo_detected"
        )
        is True,
    }


def _prepare(config: DriverConfig, clean_evidence: bool) -> int:
    review_cache_path = _review_cache_path(config, PHASE85_VARIANTS[2])
    review_cache_temporary = review_cache_path.with_suffix(
        review_cache_path.suffix + ".tmp"
    )
    if clean_evidence:
        _safe_clean_directory(config.evidence_root)
        _safe_unlink_under(review_cache_path, DEFAULT_REVIEW_CACHE_ROOT)
        _safe_unlink_under(review_cache_temporary, DEFAULT_REVIEW_CACHE_ROOT)
    else:
        stale_generation_paths = [
            path
            for variant in PHASE85_VARIANTS
            for path in (
                *_variant_paths(config, variant),
                config.generation_root / f"freeze_check_{variant}.json",
            )
            if path.exists() or path.is_symlink()
        ]
        if stale_generation_paths or review_cache_path.exists() or review_cache_temporary.exists():
            raise SystemExit(
                "refusing to replace the Phase85 freeze while generation or review artifacts exist; "
                "use prepare --clean-evidence"
            )
    config.preparation_root.mkdir(parents=True, exist_ok=True)
    holdout = build_phase85_holdout()
    redacted_holdout = _redacted_holdout_manifest(holdout)
    calibration_source = build_phase85_guard_calibration()
    isolation = _sanitize_isolation_audit(
        audit_phase85_isolation(
            holdout["sessions"],
            _previous_holdouts(),
            calibration_source,
        )
    )
    routes = audit_phase85_routes(holdout["sessions"])
    calibration_manifest = _redacted_calibration_manifest(calibration_source)
    calibration = evaluate_phase85_guard_calibration()
    model_manifest = _model_manifest(config)
    source_hashes = _source_hashes()
    protocol = _generation_protocol(config)
    checks = {
        "model_manifest_complete": model_manifest.get("complete") is True,
        "all_source_hashes_present": all(source_hashes.values()),
        "fresh_holdout_isolated": isolation.get("passed") is True,
        "pre_call_route_audit_exact": routes.get("passed") is True
        and float(routes.get("accuracy") or 0.0) == 1.0,
        "holdout_count_30": holdout.get("session_count") == PHASE85_SESSION_COUNT,
        "persona_target_count_24": holdout.get("persona_target_count") == PHASE85_TARGET_COUNT,
        "ordinary_control_count_6": holdout.get("ordinary_control_count")
        == PHASE85_CONTROL_COUNT,
        "route_detail_count_90": routes.get("detail_count") == PHASE85_SESSION_COUNT * 3,
        "format_eligible_turn_count_68": redacted_holdout[
            "format_eligible_turn_count"
        ]
        == PHASE85_FORMAT_ELIGIBLE_TURN_COUNT,
        "independent_guard_calibration_passed": calibration.get("passed") is True,
        "base_has_no_contract": VARIANT_CONTRACTS[PHASE85_VARIANTS[0]] is None,
        "v3_contract_exact": VARIANT_CONTRACTS[PHASE85_VARIANTS[1]]
        == "contract_persona_guarded_v3",
        "v4_contract_exact": VARIANT_CONTRACTS[PHASE85_VARIANTS[2]]
        == "contract_persona_guarded_v4",
        "no_training_run_planned": True,
        "product_default_unchanged": True,
    }
    freeze = {
        "kind": "phase85_pre_experiment_freeze",
        "created_at": _utcnow(),
        "benchmark_type": "simulated benchmark",
        "frozen_before_generation": True,
        "passed": all(checks.values()),
        "checks": checks,
        "execution_mode": config.mode,
        "source_sha256": source_hashes,
        "scorer_source_sha256": source_hashes["phase75_scorer"],
        "holdout_manifest_sha256": stable_hash(holdout["sessions"]),
        "redacted_holdout_manifest_sha256": stable_hash(redacted_holdout),
        "thresholds_sha256": stable_hash(FROZEN_THRESHOLDS),
        "generation_protocol_sha256": stable_hash(protocol),
        "guard_calibration_manifest_sha256": stable_hash(calibration_source),
        "guard_calibration_result_sha256": stable_hash(calibration),
        "model_manifest_sha256": model_manifest.get("manifest_sha256"),
        "expected_format_eligible_turn_count": redacted_holdout[
            "format_eligible_turn_count"
        ],
        "score_or_gate_relaxation_allowed": False,
        "automatic_promotion_allowed": False,
        "actual_user_feedback_count": 0,
        "actual_feedback_claim_allowed": False,
        "training_benefit_claim_allowed": False,
    }
    _write_json(config.preparation_root / "holdout_manifest.json", redacted_holdout)
    _write_json(config.preparation_root / "isolation_audit.json", isolation)
    _write_json(config.preparation_root / "route_audit.json", routes)
    _write_json(config.preparation_root / "model_manifest.json", model_manifest)
    _write_json(
        config.preparation_root / "guard_calibration_manifest.json",
        calibration_manifest,
    )
    _write_json(config.preparation_root / "guard_calibration.json", calibration)
    _write_json(config.evidence_root / "generation_protocol.json", protocol)
    _write_json(config.evidence_root / "frozen_thresholds.json", FROZEN_THRESHOLDS)
    _write_json(config.evidence_root / "pre_experiment_freeze.json", freeze)
    status = "ready_for_generation" if freeze["passed"] else "blocked_before_generation"
    _write_json(
        config.evidence_root / "preparation_decision.json",
        {
            "kind": "phase85_preparation_decision",
            "benchmark_type": "simulated benchmark",
            "status": status,
            "checks": checks,
            "actual_user_feedback_count": 0,
            "actual_feedback_claim_allowed": False,
            "training_benefit_claim_allowed": False,
            "automatic_promotion_allowed": False,
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "mode": config.mode,
                "session_count": holdout["session_count"],
                "route_accuracy": routes["accuracy"],
                "format_eligible_turn_count": redacted_holdout[
                    "format_eligible_turn_count"
                ],
                "guard_calibration_passed": calibration["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if freeze["passed"] else 1


def _freeze_check(config: DriverConfig) -> dict[str, Any]:
    freeze = _read_json(config.evidence_root / "pre_experiment_freeze.json")
    holdout = build_phase85_holdout()
    redacted_holdout = _redacted_holdout_manifest(holdout)
    calibration_source = build_phase85_guard_calibration()
    calibration = evaluate_phase85_guard_calibration()
    model_manifest = _model_manifest(config)
    checks = {
        "pre_experiment_freeze_passed": freeze.get("passed") is True,
        "execution_mode_unchanged": freeze.get("execution_mode") == config.mode,
        "source_files_unchanged": freeze.get("source_sha256") == _source_hashes(),
        "scorer_source_unchanged": freeze.get("scorer_source_sha256")
        == (_sha256(SCORER_SOURCE) if SCORER_SOURCE.is_file() else None),
        "holdout_unchanged": freeze.get("holdout_manifest_sha256")
        == stable_hash(holdout["sessions"]),
        "redacted_holdout_manifest_unchanged": freeze.get(
            "redacted_holdout_manifest_sha256"
        )
        == stable_hash(redacted_holdout),
        "thresholds_unchanged": freeze.get("thresholds_sha256")
        == stable_hash(FROZEN_THRESHOLDS),
        "generation_protocol_unchanged": freeze.get("generation_protocol_sha256")
        == stable_hash(_generation_protocol(config)),
        "guard_calibration_cases_unchanged": freeze.get(
            "guard_calibration_manifest_sha256"
        )
        == stable_hash(calibration_source),
        "guard_calibration_result_unchanged": freeze.get(
            "guard_calibration_result_sha256"
        )
        == stable_hash(calibration),
        "model_manifest_unchanged": freeze.get("model_manifest_sha256")
        == model_manifest.get("manifest_sha256"),
    }
    return {
        "kind": "phase85_generation_freeze_check",
        "passed": all(checks.values()),
        "checks": checks,
        "raw_source_persisted": False,
    }


def _generation_info(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    inference = dict(metadata.get("inference") or {})
    return dict(inference.get("generation") or {})


def _runtime_attempt_count(generation: Mapping[str, Any]) -> int:
    value = generation.get("runtime_attempt_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 0
    return value


def _safe_token_budget(value: Any) -> dict[str, int | float | bool | None]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, (int, float, bool)) or item is None
    }


def _safe_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "required_group_coverage": score.get("required_group_coverage"),
        "required_group_hits": [bool(value) for value in score.get("required_group_hits") or []],
        "required_labels_hit": score.get("required_labels_hit"),
        "concise": score.get("concise"),
        "unsupported_claim": score.get("unsupported_claim"),
        "forbidden_term_hit": score.get("forbidden_term_hit"),
        "forbidden_term_hit_count": len(score.get("forbidden_term_hits") or []),
        "privacy_canary_echo": score.get("privacy_canary_echo"),
        "hard_gate_passed": score.get("hard_gate_passed"),
        "final_char_count": score.get("final_char_count"),
        "composite_personalization_score": score.get("composite_personalization_score"),
    }


def _redacted_output_summary(text: str, *, format_eligible: bool) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    label_count = sum(
        any(
            line.strip().startswith(prefix)
            for line in lines
            for prefix in (f"{label}：", f"{label}:")
        )
        for label in ("结论", "依据", "下一步")
    )
    surface = "format_eligible" if format_eligible else "ordinary_or_base"
    return (
        f"{surface}[bytes={len(text.encode('utf-8'))},chars={len(text)},"
        f"lines={len(lines)},labels={label_count}]"
    )


def _run_session(
    *,
    service: PipelineService,
    session: Mapping[str, Any],
    variant: str,
    config: DriverConfig,
) -> dict[str, Any]:
    contract = VARIANT_CONTRACTS[variant]
    private_values = [
        str(value) for value in session.get("declared_private_values") or [] if str(value)
    ]
    expected_routes = [bool(value) for value in session.get("expected_routes") or []]
    api_history: list[dict[str, str]] = []
    transient_score_turns: list[dict[str, str]] = []
    turn_metadata = []
    ephemeral_review_outputs = []
    raw_private_echo = False
    invocation_succeeded = True
    for turn, raw_user in enumerate(
        (
            str(session.get("user_goal") or ""),
            str(session.get("user_correction") or ""),
            str(session.get("continuation_request") or ""),
        ),
        start=1,
    ):
        guarded_rows, driver_input_guard = guard_phase77_messages(
            [{"role": "user", "content": raw_user}], private_values
        )
        guarded_user = {
            "role": "user",
            "content": str(guarded_rows[0].get("content") or ""),
        }
        api_history.append(guarded_user)
        transient_score_turns.append(dict(guarded_user))
        metadata: dict[str, Any] = {
            "enable_real_local": config.mode == "real",
            "repetition_penalty": GENERATION_PROTOCOL_BASE["repetition_penalty"],
            "no_repeat_ngram_size": GENERATION_PROTOCOL_BASE["no_repeat_ngram_size"],
            "phase85_simulated_benchmark": True,
            "memory_consent": False,
        }
        if contract is not None:
            metadata.update(
                {
                    "response_contract": contract,
                    "declared_private_values": private_values,
                }
            )
        started = time.perf_counter()
        api_call_count = 1
        try:
            payload = service.chat_completion(
                messages=[dict(row) for row in api_history],
                model="base",
                adapter_version="latest",
                temperature=float(GENERATION_PROTOCOL_BASE["temperature"]),
                max_tokens=int(GENERATION_PROTOCOL_BASE["max_tokens"]),
                metadata=metadata,
                request_id=f"phase85-{variant}-{session['session_id']}-t{turn}",
                session_id=f"phase85-{variant}-{session['session_id']}",
            )
        except Exception as exc:
            failed_turn_metadata = [
                *turn_metadata,
                {
                    "turn": turn,
                    "api_call_count": api_call_count,
                    "api_call_succeeded": False,
                    "runtime_attempt_count": 0,
                    "model_call_count": 0,
                    "model_call_succeeded": False,
                    "response_contract": contract,
                    "raw_model_text_persisted": False,
                    "private_source_persisted": False,
                },
            ]
            raise _ChatCompletionFailure(
                exc,
                api_call_count=sum(
                    int(row.get("api_call_count") or 0)
                    for row in failed_turn_metadata
                ),
                turn_metadata=failed_turn_metadata,
            ) from exc
        latency = time.perf_counter() - started
        generation = _generation_info(payload)
        runtime_attempt_count = _runtime_attempt_count(generation)
        returned = str(payload["choices"][0]["message"]["content"])
        safe_output, driver_output_guard = guard_phase77_output(returned, private_values)
        contract_info = dict(generation.get("response_contract") or {})
        contract_output = dict(generation.get("contract_output") or {})
        contract_input_guard = dict(contract_info.get("input_guard") or {})
        contract_output_guard = dict(contract_output.get("output_guard") or {})
        runtime_route = dict(contract_info.get("route") or {})
        if contract is None:
            runtime_route = {"routed": False, "reason": "base_no_contract"}
            _evaluated_output, evaluation = enforce_phase85_persona_output(
                returned,
                messages=[dict(row) for row in api_history],
                declared_private_values=private_values,
            )
        elif contract == "contract_persona_guarded_v3":
            evaluation = _v3_evaluation(contract_output, runtime_route)
        else:
            evaluation = dict(contract_output)
        evaluation_route = dict(evaluation.get("route") or runtime_route)
        privacy_transformed = (
            driver_output_guard.get("raw_model_private_echo_detected") is True
            or contract_output_guard.get("raw_model_private_echo_detected") is True
            or evaluation.get("privacy_transform_applied") is True
        )
        outcome = _format_outcome(evaluation)
        format_eligible = evaluation.get("format_eligible") is True
        expected_route = expected_routes[turn - 1]
        route_match = evaluation_route.get("routed") == expected_route
        api_history.append({"role": "assistant", "content": safe_output})
        transient_score_turns.append({"role": "assistant", "content": safe_output})
        raw_private_echo = raw_private_echo or privacy_transformed
        finish_reason = payload["choices"][0].get("finish_reason")
        served_by = str(payload.get("served_by") or generation.get("served_by") or "")
        turn_metadata.append(
            {
                "turn": turn,
                "api_call_count": api_call_count,
                "api_call_succeeded": True,
                "runtime_attempt_count": runtime_attempt_count,
                "model_call_count": runtime_attempt_count,
                "model_call_succeeded": True,
                "served_by": served_by,
                "latency_seconds": round(latency, 6),
                "finish_reason": finish_reason,
                "truncated": finish_reason == "length",
                "token_budget": _safe_token_budget(generation.get("token_budget")),
                "response_contract": contract,
                "contract_applied": contract_info.get("applied") is True
                if contract is not None
                else False,
                "routed": evaluation_route.get("routed") is True,
                "route_reason": evaluation_route.get("reason"),
                "expected_route": expected_route,
                "route_matches_expected": route_match,
                "runtime_routed": runtime_route.get("routed") is True,
                "runtime_route_reason": runtime_route.get("reason"),
                "system_prompt_applied": contract_info.get("system_prompt_applied") is True,
                "guard_applied": contract_output.get("guard_applied") is True,
                "factual_guard_evaluated": contract_output.get(
                    "factual_guard_evaluated"
                )
                is True,
                "ordinary_passthrough": contract_output.get("ordinary_passthrough") is True,
                "format_eligible": format_eligible,
                "format_outcome": outcome,
                "format_outcome_measurement": (
                    "counterfactual_common_v4_evaluator"
                    if contract is None
                    else "runtime_contract_output"
                ),
                "format_outcome_observed_in_runtime": contract is not None,
                "native_format": outcome == "native",
                "semantic_repair_used": outcome == "semantic_repair",
                "format_fallback_used": outcome == "format_fallback",
                "safety_fallback_used": outcome == "safety_fallback",
                "privacy_transform_used": privacy_transformed,
                "guard_reported_fallback_used": evaluation.get("fallback_used") is True,
                "fallback_reason": evaluation.get("fallback_reason"),
                "pre_guard_unsupported_completion_detected": evaluation.get(
                    "unsupported_completion_detected"
                )
                is True,
                "blocked_unsupported_completion": evaluation.get(
                    "blocked_unsupported_completion"
                )
                is True,
                "post_guard_unsupported_completion_detected": evaluation.get(
                    "post_guard_unsupported_completion_detected"
                )
                is True,
                "false_block_detected": evaluation.get("false_block_detected") is True,
                "think_leak_detected": evaluation.get("think_leak_detected") is True
                or bool(re.search(r"</?think>", safe_output, flags=re.IGNORECASE)),
                "driver_input_guard_passed": driver_input_guard.get("passed") is True,
                "contract_input_guard_passed": contract_input_guard.get("passed") is True
                if contract is not None
                else None,
                "model_input_contains_declared_private_source": driver_input_guard.get(
                    "model_input_contains_declared_private_value"
                )
                is True
                or contract_input_guard.get("model_input_contains_declared_private_value")
                is True,
                "raw_model_private_echo_detected": privacy_transformed,
                "returned_private_source_detected": any(
                    value in safe_output for value in private_values
                ),
                "output_sha256": _text_sha256(safe_output),
                "output_byte_count": len(safe_output.encode("utf-8")),
                "output_char_count": len(safe_output),
                "redacted_summary": _redacted_output_summary(
                    safe_output, format_eligible=format_eligible
                ),
                "raw_model_text_persisted": False,
                "private_source_persisted": False,
            }
        )
        if variant == PHASE85_VARIANTS[2] and format_eligible:
            ephemeral_review_outputs.append(
                {
                    "session_id": session.get("session_id"),
                    "category": session.get("category"),
                    "task_type": session.get("task_type"),
                    "turn": turn,
                    "expected_route": expected_route,
                    "format_eligible": format_eligible,
                    "format_outcome": outcome,
                    "fallback_reason": evaluation.get("fallback_reason"),
                    "completion_evidence_state": evaluation.get(
                        "completion_evidence_state"
                    ),
                    "guarded_context": [dict(item) for item in api_history],
                    "returned_output": safe_output,
                    "output_sha256": _text_sha256(safe_output),
                    "declared_private_source_absent": not any(
                        value in json.dumps(api_history, ensure_ascii=False)
                        or value in safe_output
                        for value in private_values
                    ),
                }
            )
    real_local = all(row.get("served_by") == "local" for row in turn_metadata)
    score = score_phase75_transcript(
        {
            "turns": transient_score_turns,
            "status": "completed",
            "actual_model_call": invocation_succeeded,
            "privacy_canary_echo_detected": raw_private_echo,
        },
        session,
    )
    return {
        "kind": "phase85_structural_session_evidence",
        "benchmark_type": "simulated benchmark",
        "session_id": session.get("session_id"),
        "category": session.get("category"),
        "task_type": session.get("task_type"),
        "variant": variant,
        "response_contract": contract,
        "execution_mode": config.mode,
        "status": "completed",
        "api_invocation_count": sum(row["api_call_count"] for row in turn_metadata),
        "exactly_one_api_call_per_turn": all(
            row["api_call_count"] == 1 for row in turn_metadata
        ),
        "backend_runtime_attempt_count": sum(
            row["runtime_attempt_count"] for row in turn_metadata
        ),
        "exactly_one_backend_attempt_per_turn": all(
            row["runtime_attempt_count"] == 1 for row in turn_metadata
        ),
        "model_invocation_count": sum(row["model_call_count"] for row in turn_metadata),
        "exactly_one_model_call_per_turn": all(
            row["model_call_count"] == 1 for row in turn_metadata
        ),
        "actual_model_call": config.mode == "real" and real_local,
        "api_path_invoked": True,
        "turn_metadata": turn_metadata,
        "score": _safe_score(score),
        "truncated_response": any(row["truncated"] for row in turn_metadata),
        "think_leak_detected": any(row["think_leak_detected"] for row in turn_metadata),
        "privacy_canary_echo_detected": raw_private_echo,
        "returned_private_source_detected": any(
            row["returned_private_source_detected"] for row in turn_metadata
        ),
        "raw_model_text_persisted": False,
        "private_source_persisted": False,
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_feedback_claim_allowed": False,
        "training_benefit_claim_allowed": False,
        "_ephemeral_review_outputs": ephemeral_review_outputs,
        "created_at": _utcnow(),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _aggregate_variant(
    rows: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    config: DriverConfig,
) -> dict[str, Any]:
    session_by_id = {str(row.get("session_id")): dict(row) for row in sessions}
    details = []
    for source in rows:
        row = dict(source)
        score = dict(row.get("score") or {})
        session = session_by_id.get(str(row.get("session_id") or ""), {})
        details.append(
            {
                "session_id": row.get("session_id"),
                "category": session.get("category"),
                "scores": score,
            }
        )
    metric_names = (
        "required_group_coverage",
        "required_labels_hit",
        "concise",
        "unsupported_claim",
        "forbidden_term_hit",
        "privacy_canary_echo",
        "hard_gate_passed",
        "composite_personalization_score",
    )
    count = len(details)
    score_metrics = {
        name: round(
            sum(float(row["scores"].get(name) or 0.0) for row in details) / count,
            4,
        )
        if count
        else 0.0
        for name in metric_names
    }
    category_metrics: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in details}):
        selected = [row for row in details if row["category"] == category]
        category_metrics[category] = {
            "session_count": len(selected),
            "composite_personalization_score": round(
                sum(
                    float(row["scores"].get("composite_personalization_score") or 0.0)
                    for row in selected
                )
                / len(selected),
                4,
            ),
            "hard_gate_pass_rate": round(
                sum(float(row["scores"].get("hard_gate_passed") or 0.0) for row in selected)
                / len(selected),
                4,
            ),
        }
    turns = [
        dict(turn)
        for row in rows
        for turn in row.get("turn_metadata") or []
        if isinstance(turn, Mapping)
    ]
    format_turns = [turn for turn in turns if turn.get("format_eligible") is True]
    outcome_counts = Counter(str(turn.get("format_outcome")) for turn in format_turns)
    native_count = outcome_counts["native"]
    repair_count = outcome_counts["semantic_repair"]
    format_fallback_count = outcome_counts["format_fallback"]
    safety_fallback_count = outcome_counts["safety_fallback"]
    fallback_count = format_fallback_count + safety_fallback_count
    eligible_count = len(format_turns)
    accounting_passed = (
        set(outcome_counts) <= set(FORMAT_OUTCOMES)
        and native_count + repair_count + fallback_count == eligible_count
        and format_fallback_count + safety_fallback_count == fallback_count
        and all(
            sum(turn.get("format_outcome") == name for name in FORMAT_OUTCOMES) == 1
            for turn in format_turns
        )
    )
    route_turns = [turn for turn in turns if turn.get("expected_route") is not None]
    guard_turns = [turn for turn in turns if turn.get("format_eligible") is True]
    unsupported_count = sum(
        turn.get("pre_guard_unsupported_completion_detected") is True for turn in guard_turns
    )
    blocked_count = sum(
        turn.get("blocked_unsupported_completion") is True for turn in guard_turns
    )
    post_guard_count = sum(
        turn.get("post_guard_unsupported_completion_detected") is True for turn in guard_turns
    )
    false_block_count = sum(turn.get("false_block_detected") is True for turn in guard_turns)
    latencies = [float(turn.get("latency_seconds") or 0.0) for turn in turns]
    completed = [row for row in rows if row.get("status") == "completed"]
    api_call_count = sum(int(turn.get("api_call_count") or 0) for turn in turns)
    runtime_attempt_count = sum(
        int(turn.get("runtime_attempt_count") or 0) for turn in turns
    )
    expected_api_calls = PHASE85_SESSION_COUNT * 3
    exactly_one_api_call = (
        len(turns) == expected_api_calls
        and all(int(turn.get("api_call_count") or 0) == 1 for turn in turns)
    )
    exactly_one_backend_attempt = (
        len(turns) == expected_api_calls
        and all(int(turn.get("runtime_attempt_count") or 0) == 1 for turn in turns)
    )
    return {
        "kind": "phase85_variant_metrics",
        "benchmark_type": "simulated benchmark",
        "variant": variant,
        "response_contract": VARIANT_CONTRACTS[variant],
        "execution_mode": config.mode,
        "session_count": count,
        "completed_session_count": len(completed),
        "all_sessions_completed": len(completed) == PHASE85_SESSION_COUNT,
        "actual_model_calls": config.mode == "real"
        and len(rows) == PHASE85_SESSION_COUNT
        and all(row.get("actual_model_call") is True for row in rows),
        "api_call_count": api_call_count,
        "one_api_call_per_turn": exactly_one_api_call,
        "extra_api_call_count": sum(
            max(0, int(turn.get("api_call_count") or 0) - 1) for turn in turns
        ),
        "runtime_attempt_count": runtime_attempt_count,
        "one_backend_attempt_per_turn": exactly_one_backend_attempt,
        "extra_backend_attempt_count": sum(
            max(0, int(turn.get("runtime_attempt_count") or 0) - 1)
            for turn in turns
        ),
        "model_call_count": runtime_attempt_count,
        "one_model_call_per_turn": exactly_one_backend_attempt,
        "extra_model_call_count": sum(
            max(0, int(turn.get("runtime_attempt_count") or 0) - 1)
            for turn in turns
        ),
        "personalization_score": score_metrics["composite_personalization_score"],
        "required_group_coverage": score_metrics["required_group_coverage"],
        "required_labels_hit_rate": score_metrics["required_labels_hit"],
        "concise_rate": score_metrics["concise"],
        "unsupported_claim_rate": score_metrics["unsupported_claim"],
        "forbidden_term_rate": score_metrics["forbidden_term_hit"],
        "privacy_canary_echo_rate": score_metrics["privacy_canary_echo"],
        "hard_gate_pass_rate": score_metrics["hard_gate_passed"],
        "category_metrics": category_metrics,
        "truncated_session_rate": round(
            sum(bool(row.get("truncated_response")) for row in rows) / count, 4
        )
        if count
        else 0.0,
        "think_leak_rate": round(
            sum(bool(row.get("think_leak_detected")) for row in rows) / count, 4
        )
        if count
        else 0.0,
        "route_accuracy": round(
            sum(turn.get("route_matches_expected") is True for turn in route_turns)
            / len(route_turns),
            4,
        )
        if route_turns
        else 0.0,
        "route_evaluated_turn_count": len(route_turns),
        "format_eligible_turn_count": eligible_count,
        "native_format_turn_count": native_count,
        "semantic_repair_turn_count": repair_count,
        "format_fallback_turn_count": format_fallback_count,
        "safety_fallback_turn_count": safety_fallback_count,
        "fallback_turn_count": fallback_count,
        "native_format_turn_rate": native_count / eligible_count if eligible_count else 0.0,
        "semantic_repair_turn_rate": repair_count / eligible_count
        if eligible_count
        else 0.0,
        "format_fallback_turn_rate": format_fallback_count / eligible_count
        if eligible_count
        else 0.0,
        "safety_fallback_turn_rate": safety_fallback_count / eligible_count
        if eligible_count
        else 0.0,
        "fallback_turn_rate": fallback_count / eligible_count if eligible_count else 0.0,
        "fallback_turn_rate_is_counterfactual": variant == PHASE85_VARIANTS[0],
        "format_metrics_basis": (
            "counterfactual_common_v4_evaluator"
            if variant == PHASE85_VARIANTS[0]
            else "runtime_contract_output"
        ),
        "observed_runtime_fallback_turn_rate": (
            fallback_count / eligible_count
            if eligible_count and variant != PHASE85_VARIANTS[0]
            else None
        ),
        "counterfactual_guard_fallback_turn_rate": (
            fallback_count / eligible_count
            if eligible_count and variant == PHASE85_VARIANTS[0]
            else None
        ),
        "factual_guard_fallback_turn_rate": (
            fallback_count / eligible_count
            if eligible_count and variant != PHASE85_VARIANTS[0]
            else None
        ),
        "format_accounting_passed": accounting_passed,
        "privacy_transform_turn_count": sum(
            turn.get("privacy_transform_used") is True for turn in format_turns
        ),
        "pre_guard_unsupported_completion_count": unsupported_count,
        "blocked_unsupported_completion_count": blocked_count,
        "unsupported_completion_block_recall": blocked_count / unsupported_count
        if unsupported_count
        else 1.0,
        "post_guard_unsupported_completion_count": post_guard_count,
        "post_guard_unsupported_completion_rate": post_guard_count / eligible_count
        if eligible_count
        else 0.0,
        "false_block_count": false_block_count,
        "false_block_rate": false_block_count / eligible_count if eligible_count else 0.0,
        "latency_seconds": {
            "count": len(latencies),
            "p50": round(_percentile(latencies, 0.50), 6),
            "p95": round(_percentile(latencies, 0.95), 6),
            "max": round(max(latencies), 6) if latencies else 0.0,
            "method": "nearest_rank",
        },
        "output_digest_count": len(
            {
                str(turn.get("output_sha256") or "") for turn in turns if turn.get("output_sha256")
            }
        ),
        "raw_model_output_persisted": False,
        "private_source_persisted": False,
        "actual_user_feedback_count": 0,
        "actual_feedback_claim_allowed": False,
        "training_benefit_claim_allowed": False,
        "details": details,
    }


def _set_runtime_environment(config: DriverConfig) -> dict[str, str | None]:
    previous = {
        "PFE_BASE_MODEL": os.environ.get("PFE_BASE_MODEL"),
        "PFE_ENABLE_REAL_LOCAL_INFERENCE": os.environ.get(
            "PFE_ENABLE_REAL_LOCAL_INFERENCE"
        ),
    }
    os.environ["PFE_BASE_MODEL"] = str(config.model_path)
    if config.mode == "mock":
        os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
    return previous


def _restore_runtime_environment(previous: Mapping[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    InferenceEngine._runtime_cache.clear()


def _variant_paths(config: DriverConfig, variant: str) -> tuple[Path, Path]:
    return (
        config.generation_root / f"structural_sessions_{variant}.jsonl",
        config.generation_root / f"metrics_{variant}.json",
    )


def _review_cache_path(config: DriverConfig, variant: str) -> Path:
    evidence_fingerprint = _text_sha256(str(config.evidence_root.resolve()))[:12]
    path = DEFAULT_REVIEW_CACHE_ROOT / (
        f"{evidence_fingerprint}_{config.mode}_{variant}.jsonl"
    )
    _assert_safe_child_path(path, DEFAULT_REVIEW_CACHE_ROOT)
    return path


def _generate(config: DriverConfig, variant: str, clean_evidence: bool) -> int:
    freeze = _freeze_check(config)
    if not freeze["passed"]:
        raise SystemExit(f"Phase85 generation freeze failed: {freeze}")
    output_path, metrics_path = _variant_paths(config, variant)
    review_cache_path = _review_cache_path(config, variant)
    if clean_evidence:
        _safe_unlink_under(output_path, config.generation_root)
        _safe_unlink_under(metrics_path, config.generation_root)
        _safe_unlink_under(
            config.generation_root / f"freeze_check_{variant}.json",
            config.generation_root,
        )
        if config.failure_root.is_dir():
            for path in config.failure_root.glob(f"{variant}_*.json"):
                _safe_unlink_under(path, config.failure_root)
        _safe_unlink_under(review_cache_path, DEFAULT_REVIEW_CACHE_ROOT)
        _safe_unlink_under(
            review_cache_path.with_suffix(review_cache_path.suffix + ".tmp"),
            DEFAULT_REVIEW_CACHE_ROOT,
        )
    rows = _read_jsonl(output_path)
    completed = {
        str(row.get("session_id")) for row in rows if row.get("status") == "completed"
    }
    sessions = [dict(row) for row in build_phase85_holdout()["sessions"]]
    previous_environment = _set_runtime_environment(config)
    service = PipelineService()
    try:
        for index, session in enumerate(sessions, start=1):
            session_id = str(session["session_id"])
            if session_id in completed:
                print(f"[{variant}] {index}/{len(sessions)} {session_id} resumed", flush=True)
                continue
            completed_row: dict[str, Any] | None = None
            try:
                row = _run_session(
                    service=service,
                    session=session,
                    variant=variant,
                    config=config,
                )
                completed_row = row
                review_rows = [
                    dict(item) for item in row.pop("_ephemeral_review_outputs", [])
                ]
                if review_rows:
                    cached = [
                        item
                        for item in _read_jsonl(review_cache_path)
                        if item.get("session_id") != session_id
                    ]
                    cached.extend(review_rows)
                    cached.sort(
                        key=lambda item: (
                            str(item.get("session_id") or ""),
                            int(item.get("turn") or 0),
                        )
                    )
                    _write_jsonl(review_cache_path, cached)
            except Exception as exc:
                failed_turn_metadata = (
                    [dict(row) for row in exc.turn_metadata]
                    if isinstance(exc, _ChatCompletionFailure)
                    else [
                        dict(item)
                        for item in (completed_row or {}).get("turn_metadata") or []
                        if isinstance(item, Mapping)
                    ]
                )
                api_invocation_count = (
                    exc.api_call_count
                    if isinstance(exc, _ChatCompletionFailure)
                    else (completed_row or {}).get("api_invocation_count")
                )
                attempt_counts_known = isinstance(api_invocation_count, int) and all(
                    isinstance(item.get("runtime_attempt_count"), int)
                    for item in failed_turn_metadata
                )
                backend_runtime_attempt_count = (
                    sum(
                        int(item.get("runtime_attempt_count") or 0)
                        for item in failed_turn_metadata
                    )
                    if attempt_counts_known
                    else None
                )
                row = {
                    "kind": "phase85_structural_session_evidence",
                    "benchmark_type": "simulated benchmark",
                    "session_id": session_id,
                    "category": session.get("category"),
                    "task_type": session.get("task_type"),
                    "variant": variant,
                    "response_contract": VARIANT_CONTRACTS[variant],
                    "execution_mode": config.mode,
                    "status": "failed",
                    "error_type": exc.original_error_type
                    if isinstance(exc, _ChatCompletionFailure)
                    else exc.__class__.__name__,
                    "error_fingerprint": exc.error_fingerprint
                    if isinstance(exc, _ChatCompletionFailure)
                    else _text_sha256(f"{exc.__class__.__name__}:{str(exc)}"),
                    "redacted_summary": "model invocation or structural accounting failed",
                    "api_invocation_count": api_invocation_count,
                    "attempt_counts_known": attempt_counts_known,
                    "exactly_one_api_call_per_turn": False,
                    "backend_runtime_attempt_count": backend_runtime_attempt_count,
                    "exactly_one_backend_attempt_per_turn": False,
                    "model_invocation_count": backend_runtime_attempt_count,
                    "turn_metadata": failed_turn_metadata,
                    "actual_model_call": False,
                    "api_path_invoked": api_invocation_count > 0
                    if isinstance(api_invocation_count, int)
                    else None,
                    "raw_model_text_persisted": False,
                    "private_source_persisted": False,
                    "actual_user_feedback": False,
                    "created_at": _utcnow(),
                }
                _write_json(config.failure_root / f"{variant}_{session_id}.json", row)
            rows = [item for item in rows if item.get("session_id") != session_id]
            rows.append(row)
            rows.sort(key=lambda item: str(item.get("session_id") or ""))
            _write_jsonl(output_path, rows)
            print(f"[{variant}] {index}/{len(sessions)} {session_id} {row['status']}", flush=True)
    finally:
        _restore_runtime_environment(previous_environment)
    metrics = _aggregate_variant(rows, sessions, variant=variant, config=config)
    _write_json(config.generation_root / f"freeze_check_{variant}.json", freeze)
    _write_json(metrics_path, metrics)
    print(
        json.dumps(
            {
                key: metrics.get(key)
                for key in (
                    "variant",
                    "session_count",
                    "api_call_count",
                    "runtime_attempt_count",
                    "model_call_count",
                    "format_eligible_turn_count",
                    "native_format_turn_rate",
                    "semantic_repair_turn_rate",
                    "fallback_turn_rate",
                    "format_accounting_passed",
                    "latency_seconds",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    complete = (
        metrics["all_sessions_completed"]
        and metrics["api_call_count"] == PHASE85_SESSION_COUNT * 3
        and metrics["one_api_call_per_turn"]
        and metrics["runtime_attempt_count"] == PHASE85_SESSION_COUNT * 3
        and metrics["one_backend_attempt_per_turn"]
    )
    return 0 if complete else 1


def _api_smoke(config: DriverConfig, clean_evidence: bool) -> int:
    output_path = config.evidence_root / "api_smoke.json"
    if clean_evidence:
        _safe_unlink_under(output_path, config.evidence_root)
    freeze = _freeze_check(config)
    private_value = "SYNTHETIC_PHASE85_API_SMOKE_PRIVATE_DO_NOT_ECHO"
    previous_environment = _set_runtime_environment(config)
    api_call_started = False
    try:
        service = PipelineService()
        started = time.perf_counter()
        api_call_started = True
        payload = service.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": f"核验导出状态，目标目录仍少两个文件。临时值 {private_value}",
                }
            ],
            model="base",
            adapter_version="latest",
            temperature=0.0,
            max_tokens=160,
            metadata={
                "enable_real_local": config.mode == "real",
                "response_contract": "contract_persona_guarded_v4",
                "declared_private_values": [private_value],
                "memory_consent": False,
                "phase85_simulated_benchmark": True,
            },
            request_id="phase85-api-smoke-request",
            session_id="phase85-api-smoke-session",
        )
        latency = time.perf_counter() - started
        generation = _generation_info(payload)
        runtime_attempt_count = _runtime_attempt_count(generation)
        contract = dict(generation.get("response_contract") or {})
        contract_output = dict(generation.get("contract_output") or {})
        returned = str(payload["choices"][0]["message"]["content"])
        served_by = str(payload.get("served_by") or generation.get("served_by") or "")
        expected_server = "local" if config.mode == "real" else "mock"
        checks = {
            "freeze_passed": freeze.get("passed") is True,
            "one_api_call": True,
            "one_backend_attempt": runtime_attempt_count == 1,
            "one_model_call": runtime_attempt_count == 1,
            "served_by_expected_mode": served_by == expected_server,
            "v4_contract_applied": contract.get("contract")
            == "contract_persona_guarded_v4"
            and contract.get("applied") is True,
            "persona_route_selected": dict(contract.get("route") or {}).get("routed")
            is True,
            "system_prompt_applied": contract.get("system_prompt_applied") is True,
            "private_input_replaced": int(
                dict(contract.get("input_guard") or {}).get("replacement_count") or 0
            )
            >= 1,
            "private_output_absent": private_value not in returned,
            "raw_output_not_persisted": contract_output.get("raw_output_persisted")
            is False,
        }
        result = {
            "kind": "phase85_api_smoke",
            "benchmark_type": "simulated benchmark",
            "created_at": _utcnow(),
            "execution_mode": config.mode,
            "passed": all(checks.values()),
            "checks": checks,
            "api_call_count": 1,
            "runtime_attempt_count": runtime_attempt_count,
            "model_call_count": runtime_attempt_count,
            "latency_seconds": round(latency, 6),
            "served_by": served_by,
            "response_contract": contract.get("contract"),
            "output_sha256": _text_sha256(returned),
            "output_byte_count": len(returned.encode("utf-8")),
            "raw_model_output_persisted": False,
            "private_source_persisted": False,
            "actual_user_feedback_count": 0,
        }
    except Exception as exc:
        result = {
            "kind": "phase85_api_smoke",
            "benchmark_type": "simulated benchmark",
            "created_at": _utcnow(),
            "execution_mode": config.mode,
            "passed": False,
            "error_type": exc.__class__.__name__,
            "error_fingerprint": _text_sha256(f"{exc.__class__.__name__}:{str(exc)}"),
            "api_call_count": 1 if api_call_started else 0,
            "runtime_attempt_count": None,
            "model_call_count": None,
            "runtime_attempt_count_known": False,
            "raw_model_output_persisted": False,
            "private_source_persisted": False,
        }
    finally:
        _restore_runtime_environment(previous_environment)
    _write_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def _walk_forbidden_keys(value: Any, *, location: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_location = f"{location}.{key}"
            if str(key) in FORBIDDEN_EVIDENCE_KEYS:
                findings.append(next_location)
            findings.extend(_walk_forbidden_keys(item, location=next_location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_walk_forbidden_keys(item, location=f"{location}[{index}]"))
    return findings


def _public_private_audit(config: DriverConfig) -> dict[str, Any]:
    forbidden_key_locations = []
    private_marker_locations = []
    parse_failures = []
    for path in sorted(config.evidence_root.rglob("*")):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        relative = str(path.relative_to(config.evidence_root))
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "SYNTHETIC_PHASE85_PRIVATE_" in text or (
            "SYNTHETIC_PHASE85_API_SMOKE_PRIVATE" in text
        ):
            private_marker_locations.append(relative)
        if path.suffix not in {".json", ".jsonl"}:
            continue
        try:
            values = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if path.suffix == ".jsonl"
                else [json.loads(text)]
            )
        except json.JSONDecodeError:
            parse_failures.append(relative)
            continue
        for index, value in enumerate(values):
            forbidden_key_locations.extend(
                f"{relative}:{location}"
                for location in _walk_forbidden_keys(value, location=f"$[{index}]")
            )
    structural_rows = [
        row
        for variant in PHASE85_VARIANTS
        for row in _read_jsonl(_variant_paths(config, variant)[0])
    ]
    checks = {
        "no_forbidden_evidence_keys": not forbidden_key_locations,
        "no_private_marker_in_evidence": not private_marker_locations,
        "all_evidence_json_parseable": not parse_failures,
        "structural_rows_do_not_persist_raw_model_output": bool(structural_rows)
        and all(row.get("raw_model_text_persisted") is False for row in structural_rows),
        "structural_rows_do_not_persist_private_source": bool(structural_rows)
        and all(row.get("private_source_persisted") is False for row in structural_rows),
        "no_actual_user_feedback_claim": all(
            row.get("actual_user_feedback") is False for row in structural_rows
        ),
    }
    return {
        "kind": "phase85_public_private_evidence_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "structural_session_count": len(structural_rows),
        "forbidden_key_locations": forbidden_key_locations,
        "private_marker_locations": private_marker_locations,
        "parse_failures": parse_failures,
        "review_cache_outside_evidence_root": True,
        "raw_model_output_persisted": False,
        "private_source_persisted": False,
    }


def _ordinary_identity(config: DriverConfig) -> dict[str, Any]:
    rows_by_variant = {
        variant: {
            str(row.get("session_id") or ""): dict(row)
            for row in _read_jsonl(_variant_paths(config, variant)[0])
            if row.get("task_type") == "ordinary_control"
        }
        for variant in PHASE85_VARIANTS
    }
    session_ids = sorted(rows_by_variant[PHASE85_VARIANTS[0]])
    details = []
    for session_id in session_ids:
        hashes = {}
        byte_counts = {}
        for variant in PHASE85_VARIANTS:
            row = rows_by_variant[variant].get(session_id, {})
            turns = [dict(item) for item in row.get("turn_metadata") or []]
            hashes[variant] = [str(item.get("output_sha256") or "") for item in turns]
            byte_counts[variant] = [int(item.get("output_byte_count") or 0) for item in turns]
        base_hashes = hashes[PHASE85_VARIANTS[0]]
        base_bytes = byte_counts[PHASE85_VARIANTS[0]]
        details.append(
            {
                "session_id": session_id,
                "turn_count": len(base_hashes),
                "v3_byte_identical": hashes[PHASE85_VARIANTS[1]] == base_hashes
                and byte_counts[PHASE85_VARIANTS[1]] == base_bytes,
                "v4_byte_identical": hashes[PHASE85_VARIANTS[2]] == base_hashes
                and byte_counts[PHASE85_VARIANTS[2]] == base_bytes,
                "transcript_sha256": {
                    variant: stable_hash(
                        {"hashes": hashes[variant], "byte_counts": byte_counts[variant]}
                    )
                    for variant in PHASE85_VARIANTS
                },
            }
        )
    v4_turns = [
        dict(turn)
        for row in rows_by_variant[PHASE85_VARIANTS[2]].values()
        for turn in row.get("turn_metadata") or []
    ]
    session_count = len(details)
    turn_count = len(v4_turns)
    return {
        "kind": "phase85_ordinary_three_arm_identity",
        "passed": session_count == PHASE85_CONTROL_COUNT
        and turn_count == PHASE85_CONTROL_COUNT * 3
        and all(row["v3_byte_identical"] and row["v4_byte_identical"] for row in details)
        and all(turn.get("routed") is False for turn in v4_turns)
        and all(turn.get("system_prompt_applied") is False for turn in v4_turns)
        and all(turn.get("guard_applied") is False for turn in v4_turns),
        "session_count": session_count,
        "turn_count": turn_count,
        "v3_identity_rate": round(
            sum(row["v3_byte_identical"] for row in details) / session_count, 4
        )
        if session_count
        else 0.0,
        "v4_identity_rate": round(
            sum(row["v4_byte_identical"] for row in details) / session_count, 4
        )
        if session_count
        else 0.0,
        "v4_route_off_rate": round(
            sum(turn.get("routed") is False for turn in v4_turns) / turn_count, 4
        )
        if turn_count
        else 0.0,
        "v4_system_prompt_off_rate": round(
            sum(turn.get("system_prompt_applied") is False for turn in v4_turns)
            / turn_count,
            4,
        )
        if turn_count
        else 0.0,
        "v4_guard_off_rate": round(
            sum(turn.get("guard_applied") is False for turn in v4_turns) / turn_count,
            4,
        )
        if turn_count
        else 0.0,
        "raw_output_persisted": False,
        "details": details,
    }


def _generation_audit(config: DriverConfig) -> dict[str, Any]:
    metrics = {
        variant: _read_json(_variant_paths(config, variant)[1])
        for variant in PHASE85_VARIANTS
    }
    rows_by_variant = {
        variant: _read_jsonl(_variant_paths(config, variant)[0])
        for variant in PHASE85_VARIANTS
    }
    structural_rows = [row for rows in rows_by_variant.values() for row in rows]
    turns_by_variant = {
        variant: [
            dict(turn)
            for row in rows
            for turn in row.get("turn_metadata") or []
            if isinstance(turn, Mapping)
        ]
        for variant, rows in rows_by_variant.items()
    }
    expected_api_calls_per_variant = PHASE85_SESSION_COUNT * 3
    expected_total_attempts = len(PHASE85_VARIANTS) * expected_api_calls_per_variant
    api_calls_by_variant = {
        variant: sum(int(turn.get("api_call_count") or 0) for turn in turns)
        for variant, turns in turns_by_variant.items()
    }
    runtime_attempts_by_variant = {
        variant: sum(int(turn.get("runtime_attempt_count") or 0) for turn in turns)
        for variant, turns in turns_by_variant.items()
    }
    all_turns = [turn for turns in turns_by_variant.values() for turn in turns]
    api_call_count = sum(api_calls_by_variant.values())
    runtime_attempt_count = sum(runtime_attempts_by_variant.values())
    one_api_call_per_turn = all(
        len(turns) == expected_api_calls_per_variant
        and all(int(turn.get("api_call_count") or 0) == 1 for turn in turns)
        for turns in turns_by_variant.values()
    )
    one_backend_attempt_per_turn = all(
        len(turns) == expected_api_calls_per_variant
        and all(int(turn.get("runtime_attempt_count") or 0) == 1 for turn in turns)
        for turns in turns_by_variant.values()
    )
    checks = {
        "all_three_variants_present": all(metrics.values())
        and all(turns_by_variant.values()),
        "all_variants_exactly_90_api_calls": all(
            count == expected_api_calls_per_variant
            for count in api_calls_by_variant.values()
        ),
        "all_variants_one_api_call_per_turn": one_api_call_per_turn,
        "all_turns_exactly_one_backend_attempt": one_backend_attempt_per_turn,
        "no_extra_backend_attempts": all(
            int(turn.get("runtime_attempt_count") or 0) <= 1 for turn in all_turns
        ),
        "exact_total_backend_attempt_count": runtime_attempt_count
        == expected_total_attempts,
        "all_variants_one_call_per_turn": one_backend_attempt_per_turn,
        "no_extra_model_calls": all(
            int(turn.get("runtime_attempt_count") or 0) <= 1 for turn in all_turns
        ),
        "exact_total_model_call_count": runtime_attempt_count == expected_total_attempts,
        "raw_model_output_not_persisted": all(
            row.get("raw_model_text_persisted") is False for row in structural_rows
        ),
    }
    return {
        "kind": "phase85_generation_call_and_persistence_audit",
        "passed": bool(structural_rows) and all(checks.values()),
        "checks": checks,
        "expected_api_call_count_per_variant": expected_api_calls_per_variant,
        "observed_api_call_count_by_variant": api_calls_by_variant,
        "expected_total_api_call_count": expected_total_attempts,
        "observed_total_api_call_count": api_call_count,
        "one_api_call_per_turn": one_api_call_per_turn,
        "expected_backend_attempt_count_per_turn": 1,
        "observed_backend_attempt_count_by_variant": runtime_attempts_by_variant,
        "observed_backend_runtime_attempt_count": runtime_attempt_count,
        "one_backend_attempt_per_turn": one_backend_attempt_per_turn,
        "extra_backend_attempt_count": sum(
            max(0, int(turn.get("runtime_attempt_count") or 0) - 1)
            for turn in all_turns
        ),
        "expected_model_call_count": expected_total_attempts,
        "observed_model_call_count": runtime_attempt_count,
        "one_model_call_per_turn": one_backend_attempt_per_turn,
        "extra_model_call_count": sum(
            max(0, int(turn.get("runtime_attempt_count") or 0) - 1)
            for turn in all_turns
        ),
        "raw_model_output_persisted": False,
    }


def _structural_review_keys(config: DriverConfig) -> list[dict[str, Any]]:
    structural_path, _ = _variant_paths(config, PHASE85_VARIANTS[2])
    keys: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for session in _read_jsonl(structural_path):
        session_id = str(session.get("session_id") or "")
        for turn_row in session.get("turn_metadata") or []:
            if not isinstance(turn_row, Mapping) or turn_row.get("format_eligible") is not True:
                continue
            turn = int(turn_row.get("turn") or 0)
            identity = (session_id, turn)
            output_sha256 = turn_row.get("output_sha256")
            if (
                not session_id
                or turn < 1
                or identity in seen
                or not isinstance(output_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", output_sha256) is None
            ):
                raise RuntimeError("invalid or duplicate Phase85 structural review key")
            seen.add(identity)
            keys.append(
                {
                    "session_id": session_id,
                    "turn": turn,
                    "output_sha256": output_sha256,
                }
            )
    return sorted(keys, key=lambda item: (item["session_id"], item["turn"]))


def _review_output_keys(config: DriverConfig) -> list[dict[str, Any]]:
    cache_path = _review_cache_path(config, PHASE85_VARIANTS[2])
    rows = [
        row for row in _read_jsonl(cache_path) if row.get("format_eligible") is True
    ]
    structural_keys = _structural_review_keys(config)
    structural_by_identity = {
        (str(row["session_id"]), int(row["turn"])): str(row["output_sha256"])
        for row in structural_keys
    }
    keys: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        session_id = str(row.get("session_id") or "")
        turn = int(row.get("turn") or 0)
        identity = (session_id, turn)
        returned_output = row.get("returned_output")
        declared_sha256 = row.get("output_sha256")
        if not isinstance(returned_output, str):
            raise RuntimeError("Phase85 review cache is missing returned_output")
        computed_sha256 = _text_sha256(returned_output)
        if (
            not session_id
            or turn < 1
            or identity in seen
            or declared_sha256 != computed_sha256
            or structural_by_identity.get(identity) != computed_sha256
        ):
            raise RuntimeError("Phase85 review cache does not bind to structural output")
        seen.add(identity)
        keys.append(
            {
                "session_id": session_id,
                "turn": turn,
                "output_sha256": computed_sha256,
            }
        )
    keys.sort(key=lambda item: (item["session_id"], item["turn"]))
    if keys != structural_keys:
        raise RuntimeError("Phase85 review cache and structural output sets differ")
    return keys


def _review_output_key_manifest(keys: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    key_hashes = [
        _text_sha256(
            json.dumps(dict(key), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        for key in keys
    ]
    aggregate_hash = stable_hash(key_hashes)
    return {
        "kind": "phase85_review_output_key_manifest",
        "created_at": _utcnow(),
        "output_count": len(keys),
        "output_keys_sha256": aggregate_hash,
        "output_key_hashes": key_hashes,
        "output_key_hashes_sha256": aggregate_hash,
        "raw_review_output_persisted": False,
    }


def _review_template(config: DriverConfig, clean_evidence: bool) -> int:
    path = config.evidence_root / "manual-semantic-review.json"
    manifest_path = config.evidence_root / REVIEW_OUTPUT_KEY_MANIFEST
    if clean_evidence:
        _safe_unlink_under(path, config.evidence_root)
        _safe_unlink_under(manifest_path, config.evidence_root)
    keys = _review_output_keys(config)
    existing = _read_json(path)
    if existing.get("complete") is True:
        raise SystemExit("refusing to overwrite completed Phase85 manual review")
    manifest = _review_output_key_manifest(keys)
    expected_output_keys_sha256 = manifest["output_keys_sha256"]
    template = {
        "kind": "phase85_manual_semantic_review",
        "benchmark_type": "simulated benchmark",
        "created_at": _utcnow(),
        "complete": False,
        "can_only_tighten": True,
        "passed": False,
        "review_scope": "all routed V4 returned outputs",
        "expected_reviewed_output_count": PHASE85_FORMAT_ELIGIBLE_TURN_COUNT,
        "reviewed_output_count": 0,
        "expected_output_keys_sha256": expected_output_keys_sha256,
        "reviewed_output_keys_sha256": None,
        "reviewer_ids": [],
        "residual_unsupported_claim_count": None,
        "false_block_count": None,
        "other_semantic_failure_count": None,
        "findings": [],
        "raw_output_persisted_in_evidence": False,
        "review_cache_outside_evidence_root": True,
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
    }
    _write_json(manifest_path, manifest)
    _write_json(path, template)
    print(
        json.dumps(
            {
                "status": "manual_review_required",
                "review_cache_path": str(
                    _review_cache_path(config, PHASE85_VARIANTS[2])
                ),
                "review_output_key_manifest": str(manifest_path),
                "output_count": len(keys),
                "expected_output_keys_sha256": expected_output_keys_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if len(keys) == PHASE85_FORMAT_ELIGIBLE_TURN_COUNT else 1


def _manual_review(config: DriverConfig) -> dict[str, Any]:
    review = _read_json(config.evidence_root / "manual-semantic-review.json")
    manifest = _read_json(config.evidence_root / REVIEW_OUTPUT_KEY_MANIFEST)
    key_hashes = manifest.get("output_key_hashes") or []
    manifest_count = int(manifest.get("output_count") or 0)
    expected_keys_sha256 = manifest.get("output_keys_sha256")
    aggregate_hash = stable_hash(key_hashes) if isinstance(key_hashes, list) else None
    try:
        structural_keys = _structural_review_keys(config)
        structural_manifest = _review_output_key_manifest(structural_keys)
        structural_binding = (
            manifest_count == structural_manifest.get("output_count")
            and expected_keys_sha256
            == structural_manifest.get("output_keys_sha256")
            and key_hashes == structural_manifest.get("output_key_hashes")
        )
        cache_path = _review_cache_path(config, PHASE85_VARIANTS[2])
        cache_binding = not cache_path.is_file() or _review_output_keys(config) == structural_keys
    except (OSError, TypeError, ValueError, RuntimeError):
        structural_binding = False
        cache_binding = False
    reviewer_ids = review.get("reviewer_ids")
    manifest_integrity = (
        manifest.get("kind") == "phase85_review_output_key_manifest"
        and manifest_count == PHASE85_FORMAT_ELIGIBLE_TURN_COUNT
        and isinstance(key_hashes, list)
        and len(key_hashes) == manifest_count
        and len(set(key_hashes)) == len(key_hashes)
        and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in key_hashes
        )
        and expected_keys_sha256 == aggregate_hash
        and manifest.get("output_key_hashes_sha256") == aggregate_hash
        and isinstance(expected_keys_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", expected_keys_sha256) is not None
        and manifest.get("raw_review_output_persisted") is False
        and structural_binding
        and cache_binding
    )
    integrity = (
        manifest_integrity
        and review.get("expected_output_keys_sha256") == expected_keys_sha256
        and review.get("reviewed_output_keys_sha256") == expected_keys_sha256
        and int(review.get("reviewed_output_count") or 0) == manifest_count
        and isinstance(reviewer_ids, list)
        and bool(reviewer_ids)
        and all(isinstance(value, str) and value.strip() for value in reviewer_ids)
    )
    residual_count = review.get("residual_unsupported_claim_count")
    false_block_count = review.get("false_block_count")
    other_failure_count = review.get("other_semantic_failure_count")
    passed = (
        review.get("passed") is True
        and review.get("complete") is True
        and review.get("can_only_tighten") is True
        and integrity
        and isinstance(residual_count, int)
        and residual_count == 0
        and isinstance(false_block_count, int)
        and false_block_count == 0
        and isinstance(other_failure_count, int)
        and other_failure_count == 0
    )
    return {
        **review,
        "complete": review.get("complete") is True and integrity,
        "passed": passed,
        "can_only_tighten": review.get("can_only_tighten") is True,
        "integrity_passed": integrity,
        "review_output_manifest_count": manifest_count,
        "review_cache_present": _review_cache_path(
            config, PHASE85_VARIANTS[2]
        ).is_file(),
    }


def _structural_output_summary(
    config: DriverConfig,
    metrics: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = [
        "# Phase85 Structural Output Evidence",
        "",
        "This file intentionally contains no raw model output or private source text.",
        "The benchmark is simulated_usage and does not prove actual user benefit or training benefit.",
        "",
        "| Variant | Target score | Native | Repair | Format fallback | Safety fallback | Total fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in PHASE85_VARIANTS:
        row = dict(metrics.get(variant) or {})
        target_categories = [
            dict(value)
            for name, value in dict(row.get("category_metrics") or {}).items()
            if name != "ordinary_direct"
        ]
        target_score = (
            sum(float(item.get("composite_personalization_score") or 0.0) for item in target_categories)
            / len(target_categories)
            if target_categories
            else 0.0
        )
        lines.append(
            "| {variant} | {score:.4f} | {native} | {repair} | {format_fb} | "
            "{safety_fb} | {fallback} |".format(
                variant=variant,
                score=target_score,
                native=int(row.get("native_format_turn_count") or 0),
                repair=int(row.get("semantic_repair_turn_count") or 0),
                format_fb=int(row.get("format_fallback_turn_count") or 0),
                safety_fb=int(row.get("safety_fallback_turn_count") or 0),
                fallback=int(row.get("fallback_turn_count") or 0),
            )
        )
    lines.extend(
        [
            "",
            "The temporary V4 review cache lives outside the evidence root and is deleted after review.",
            "Only hashes, counts, review findings, and structural metadata are retained in the repository.",
        ]
    )
    return "\n".join(lines)


def _decision_markdown(
    decision: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
) -> str:
    failed = list(decision.get("failed_benefit_checks") or [])
    lines = [
        "# Phase85 Final Decision",
        "",
        f"- Status: `{decision.get('status')}`",
        f"- Recommendation: `{decision.get('recommendation')}`",
        "- Evidence type: simulated benchmark with real local model calls",
        "- Actual user feedback: 0",
        "- Training or adapter benefit claim: not allowed",
        "- Automatic promotion/deployment: not allowed",
        "",
        "## Three-arm result",
        "",
    ]
    for variant in PHASE85_VARIANTS:
        row = dict(metrics.get(variant) or {})
        lines.append(
            f"- `{variant}`: native={float(row.get('native_format_turn_rate') or 0.0):.4f}, "
            f"repair={float(row.get('semantic_repair_turn_rate') or 0.0):.4f}, "
            f"fallback={float(row.get('fallback_turn_rate') or 0.0):.4f}, "
            f"latency_p95={float(dict(row.get('latency_seconds') or {}).get('p95') or 0.0):.4f}s"
        )
    lines.extend(
        [
            "",
            f"- V4 target score: `{dict(decision.get('target_scores') or {}).get(PHASE85_VARIANTS[2])}`",
            f"- V4 gain vs base: `{decision.get('v4_gain_vs_base')}`",
            f"- V4 target category floor: `{decision.get('v4_target_category_floor')}`",
            "",
            "## Failed strict gates",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in failed)
    if not failed:
        lines.append("- None. This still permits manual review only, not promotion.")
    return "\n".join(lines)


def _runbook(config: DriverConfig) -> str:
    relative_evidence = (
        str(config.evidence_root.relative_to(REPO_ROOT))
        if config.evidence_root.is_relative_to(REPO_ROOT)
        else str(config.evidence_root)
    )
    return f"""# Phase85 Low-Fallback Semantic Guard Runbook

Phase85 is a simulated benchmark. It does not use actual user feedback, does not train an adapter,
and cannot prove training benefit. Raw pre-guard model text and private source text are not retained.

## Frozen run

```bash
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real prepare --clean-evidence
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real api-smoke --clean-evidence
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real generate --variant {PHASE85_VARIANTS[0]} --clean-evidence
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real generate --variant {PHASE85_VARIANTS[1]} --clean-evidence
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real generate --variant {PHASE85_VARIANTS[2]} --clean-evidence
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real review-template --clean-evidence
```

Review every routed V4 returned output from the temporary review cache. Record only hashes and
findings in `{relative_evidence}/manual-semantic-review.json`. Manual review may turn an automated
pass into failure; it may never upgrade a failed deterministic gate.

```bash
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real full-regression
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real finalize
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real validate
```

## Strict boundaries

- Expected format denominator: {PHASE85_FORMAT_ELIGIBLE_TURN_COUNT} routed turns per variant.
- V4 native rate >= {PHASE85_NATIVE_FORMAT_MINIMUM:.2f}; repair <= {PHASE85_SEMANTIC_REPAIR_MAXIMUM:.2f}; fallback <= {PHASE85_FALLBACK_MAXIMUM:.2f}.
- V4 target score >= {PHASE85_TARGET_SCORE_MINIMUM:.2f}; every target category >= {PHASE85_TARGET_CATEGORY_FLOOR:.2f}; gain vs base >= {PHASE85_TARGET_GAIN_MINIMUM:.2f}.
- Independent pre-labeled block recall must be 1.0 and false-block rate 0.0.
- No automatic promote, deployment, Hermes attachment, or product-default change.
"""


def _next_goal(decision: Mapping[str, Any]) -> str:
    recommendation = str(decision.get("recommendation") or "")
    if recommendation == "phase86_opt_in_manual_runtime_trial":
        objective = (
            "Run an opt-in, manually reviewed local runtime trial against non-private daily tasks; "
            "keep V4 off by default and collect explicit human accept/edit/reject evidence."
        )
    else:
        objective = (
            "Archive Phase85 as non-qualified, inspect the failed native/fallback/category gates, "
            "and choose one falsifiable parser, prompt, or training-objective repair without relaxing thresholds."
        )
    return f"""# Next Pursuit Goal

Current Phase85 recommendation: `{recommendation}`.

Objective: {objective}

Constraints: simulated evidence remains simulated_usage; no automatic promotion, deployment,
Hermes attachment, private-source export, or claim of adapter/training benefit.
"""


def _manifest(config: DriverConfig) -> dict[str, Any]:
    files = []
    for path in sorted(config.evidence_root.rglob("*")):
        if not path.is_file() or path.name in DYNAMIC_FILES:
            continue
        files.append(
            {
                "path": str(path.relative_to(REPO_ROOT))
                if path.is_relative_to(REPO_ROOT)
                else str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "kind": "phase85_evidence_manifest",
        "created_at": _utcnow(),
        "file_count": len(files),
        "files": files,
        "manifest_sha256": stable_hash(files),
    }


def _finalize(config: DriverConfig) -> int:
    stored_metrics = {
        variant: _read_json(_variant_paths(config, variant)[1])
        for variant in PHASE85_VARIANTS
    }
    sessions = [dict(row) for row in build_phase85_holdout()["sessions"]]
    metrics = {
        variant: _aggregate_variant(
            _read_jsonl(_variant_paths(config, variant)[0]),
            sessions,
            variant=variant,
            config=config,
        )
        for variant in PHASE85_VARIANTS
    }
    if stored_metrics != metrics:
        raise RuntimeError("Phase85 stored metrics do not match structural evidence")
    current_freeze = _freeze_check(config)
    if current_freeze.get("passed") is not True:
        raise RuntimeError("Phase85 source or protocol freeze changed before finalize")
    for variant in PHASE85_VARIANTS:
        saved_freeze = _read_json(
            config.generation_root / f"freeze_check_{variant}.json"
        )
        if saved_freeze != current_freeze:
            raise RuntimeError(
                f"Phase85 generation freeze binding mismatch for {variant}"
            )
    isolation = _read_json(config.preparation_root / "isolation_audit.json")
    routes = _read_json(config.preparation_root / "route_audit.json")
    calibration = _read_json(config.preparation_root / "guard_calibration.json")
    api_smoke = _read_json(config.evidence_root / "api_smoke.json")
    ordinary = _ordinary_identity(config)
    generation = _generation_audit(config)
    public_private = _public_private_audit(config)
    manual = _manual_review(config)
    if manual.get("complete") is not True or manual.get("integrity_passed") is not True:
        raise RuntimeError("Phase85 manual review must be complete and integral before finalize")
    review_cache_path = _review_cache_path(config, PHASE85_VARIANTS[2])
    review_cache_temporary = review_cache_path.with_suffix(
        review_cache_path.suffix + ".tmp"
    )
    review_cache_paths = (review_cache_path, review_cache_temporary)
    review_cache_was_present = any(
        path.exists() or path.is_symlink() for path in review_cache_paths
    )
    try:
        for path in review_cache_paths:
            _safe_unlink_under(path, DEFAULT_REVIEW_CACHE_ROOT)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("failed to delete Phase85 temporary V4 review cache") from exc
    if any(path.exists() or path.is_symlink() for path in review_cache_paths):
        raise RuntimeError("Phase85 temporary V4 review cache is still present")
    review_cache_deleted = review_cache_was_present and not any(
        path.exists() or path.is_symlink() for path in review_cache_paths
    )
    decision = build_phase85_decision(
        metrics=metrics,
        isolation_audit=isolation,
        route_audit=routes,
        api_smoke=api_smoke,
        public_private_audit=public_private,
        ordinary_identity=ordinary,
        guard_calibration=calibration,
        generation_audit=generation,
        manual_review=manual,
    )
    comparison = {
        "kind": "phase85_three_arm_comparison_summary",
        "created_at": _utcnow(),
        "benchmark_type": "simulated benchmark",
        "execution_mode": config.mode,
        "model": "Qwen2.5-1.5B-Instruct",
        "variants": {
            variant: {
                key: metrics[variant].get(key)
                for key in (
                    "actual_model_calls",
                    "session_count",
                    "api_call_count",
                    "one_api_call_per_turn",
                    "runtime_attempt_count",
                    "one_backend_attempt_per_turn",
                    "model_call_count",
                    "hard_gate_pass_rate",
                    "unsupported_claim_rate",
                    "route_accuracy",
                    "format_eligible_turn_count",
                    "native_format_turn_rate",
                    "semantic_repair_turn_rate",
                    "format_fallback_turn_rate",
                    "safety_fallback_turn_rate",
                    "fallback_turn_rate",
                    "format_accounting_passed",
                    "post_guard_unsupported_completion_rate",
                    "latency_seconds",
                )
            }
            for variant in PHASE85_VARIANTS
        },
        "target_scores": decision.get("target_scores"),
        "v4_gain_vs_base": decision.get("v4_gain_vs_base"),
        "v4_delta_vs_v3": decision.get("v4_delta_vs_v3"),
        "v4_target_category_floor": decision.get("v4_target_category_floor"),
        "ordinary_identity_passed": ordinary.get("passed") is True,
        "independent_guard_calibration_passed": calibration.get("passed") is True,
        "manual_review_passed": manual.get("passed") is True,
        "status": decision.get("status"),
        "recommendation": decision.get("recommendation"),
        "actual_user_feedback_count": 0,
        "training_benefit_claim_allowed": False,
        "automatic_promotion_allowed": False,
    }
    freeze_checks = {variant: True for variant in PHASE85_VARIANTS}
    full_regression = _read_json(config.evidence_root / "full_regression_summary.json")
    integrity_checks = {
        "pre_experiment_freeze_passed": _read_json(
            config.evidence_root / "pre_experiment_freeze.json"
        ).get("passed")
        is True,
        "all_generation_freeze_checks_passed": all(freeze_checks.values()),
        "current_source_and_protocol_freeze_passed": current_freeze.get("passed")
        is True,
        "all_variant_metrics_present": all(metrics.values()),
        "real_mode_has_actual_model_calls": config.mode != "real"
        or all(row.get("actual_model_calls") is True for row in metrics.values()),
        "format_denominator_68_all_variants": all(
            int(row.get("format_eligible_turn_count") or 0)
            == PHASE85_FORMAT_ELIGIBLE_TURN_COUNT
            for row in metrics.values()
        ),
        "format_accounting_passed_all_variants": all(
            row.get("format_accounting_passed") is True for row in metrics.values()
        ),
        "ordinary_identity_passed": ordinary.get("passed") is True,
        "generation_audit_passed": generation.get("passed") is True,
        "public_private_audit_passed": public_private.get("passed") is True,
        "manual_review_complete_and_integral": manual.get("complete") is True
        and manual.get("integrity_passed") is True,
        "temporary_v4_review_cache_absent": not any(
            path.exists() or path.is_symlink() for path in review_cache_paths
        ),
        "api_smoke_passed": api_smoke.get("passed") is True,
        "full_regression_passed": full_regression.get("passed") is True,
    }
    integrity = {
        "kind": "phase85_evidence_integrity",
        "created_at": _utcnow(),
        "passed": all(integrity_checks.values()),
        "checks": integrity_checks,
        "generation_freeze_checks": freeze_checks,
        "decision_status": decision.get("status"),
        "product_gate_qualified": decision.get("simulated_lab_runtime_benefit") is True,
        "validation_pass_does_not_imply_product_pass": True,
    }
    _write_json(config.evidence_root / "ordinary_three_arm_identity.json", ordinary)
    _write_json(config.evidence_root / "generation_audit.json", generation)
    _write_json(config.evidence_root / "public_private_audit.json", public_private)
    _write_json(config.evidence_root / "comparison_summary.json", comparison)
    _write_json(config.evidence_root / "phase85-final-decision.json", decision)
    _write_text(
        config.evidence_root / "phase85-final-decision.md",
        _decision_markdown(decision, metrics),
    )
    _write_text(
        config.evidence_root / "output_examples.md",
        _structural_output_summary(config, metrics),
    )
    _write_text(config.evidence_root / "phase85-runbook.md", _runbook(config))
    _write_text(config.evidence_root / "next-pursuit-goal.md", _next_goal(decision))
    _write_json(config.evidence_root / "evidence_integrity.json", integrity)
    _write_json(config.evidence_root / "evidence_manifest.json", _manifest(config))
    _write_json(
        config.evidence_root / "finalization_state.json",
        {
            "kind": "phase85_finalization_state",
            "created_at": _utcnow(),
            "complete": integrity["passed"],
            "decision_status": decision.get("status"),
            "recommendation": decision.get("recommendation"),
            "manual_review_can_only_tighten": True,
            "review_cache_was_present": review_cache_was_present,
            "review_cache_deleted": review_cache_deleted,
            "temporary_v4_review_cache_absent": not any(
                path.exists() or path.is_symlink() for path in review_cache_paths
            ),
            "actual_user_feedback_count": 0,
            "automatic_promotion_allowed": False,
        },
    )
    print(
        json.dumps(
            {
                "integrity_passed": integrity["passed"],
                "status": decision.get("status"),
                "recommendation": decision.get("recommendation"),
                "failed_benefit_checks": decision.get("failed_benefit_checks"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if integrity["passed"] else 1


def _run_logged(command: list[str], *, command_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    output_digest = hashlib.sha256()
    output_line_count = 0
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_digest.update(line.encode("utf-8"))
        output_line_count += 1
    exit_code = process.wait()
    return {
        "command_id": command_id,
        "exit_code": exit_code,
        "duration_seconds": round(time.perf_counter() - started, 4),
        "output_sha256": output_digest.hexdigest(),
        "output_line_count": output_line_count,
        "raw_process_output_persisted": False,
    }


def _full_regression(config: DriverConfig) -> int:
    commands = (
        ("phase85_focused_pytest", [
            str(REPO_ROOT / ".venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            "tests/test_phase85_driver_safety.py",
            "tests/test_phase85_engine_status_privacy.py",
            "tests/test_phase85_low_fallback_semantic_guard.py",
            "tests/test_phase85_semantic_guard_hardening.py",
            "tests/test_phase84_factual_completion_guard.py",
            "tests/test_phase83_persona_route_length_repair.py",
            "tests/test_phase77_private_value_guarded_runtime.py",
            "tests/test_phase13_boundary_contract.py",
            "tests/test_inference_runtime.py",
        ]),
        (
            "repository_regression_targets",
            ["make", "test-unit", "test-surface", "test-e2e-mock", "smoke-beta"],
        ),
    )
    results = []
    for command_id, command in commands:
        result = _run_logged(command, command_id=command_id)
        results.append(result)
        if result["exit_code"] != 0:
            break
    summary = {
        "kind": "phase85_full_regression_summary",
        "created_at": _utcnow(),
        "passed": len(results) == len(commands)
        and all(row["exit_code"] == 0 for row in results),
        "results": results,
    }
    _write_json(config.evidence_root / "full_regression_summary.json", summary)
    return 0 if summary["passed"] else 1


def _validate(config: DriverConfig) -> int:
    manifest = _read_json(config.evidence_root / "evidence_manifest.json")
    integrity = _read_json(config.evidence_root / "evidence_integrity.json")
    decision = _read_json(config.evidence_root / "phase85-final-decision.json")
    regression = _read_json(config.evidence_root / "full_regression_summary.json")
    api_smoke = _read_json(config.evidence_root / "api_smoke.json")
    manual = _manual_review(config)
    manifest_failures = []
    for row in manifest.get("files") or []:
        path = Path(str(row.get("path") or ""))
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            manifest_failures.append(str(row.get("path") or ""))
    current_private_audit = _public_private_audit(config)
    current_freeze = _freeze_check(config)
    review_cache_path = _review_cache_path(config, PHASE85_VARIANTS[2])
    review_cache_temporary = review_cache_path.with_suffix(
        review_cache_path.suffix + ".tmp"
    )
    checks = {
        "manifest_files_unchanged": bool(manifest.get("files")) and not manifest_failures,
        "evidence_integrity_passed": integrity.get("passed") is True,
        "full_regression_passed": regression.get("passed") is True,
        "api_smoke_passed": api_smoke.get("passed") is True,
        "public_private_audit_still_passes": current_private_audit.get("passed") is True,
        "source_and_protocol_freeze_still_passes": current_freeze.get("passed") is True,
        "manual_review_complete_and_integral": manual.get("complete") is True
        and manual.get("integrity_passed") is True,
        "temporary_v4_review_cache_absent": not any(
            path.exists() or path.is_symlink()
            for path in (review_cache_path, review_cache_temporary)
        ),
        "actual_user_feedback_count_zero": isinstance(
            decision.get("actual_user_feedback_count"), int
        )
        and not isinstance(decision.get("actual_user_feedback_count"), bool)
        and int(decision["actual_user_feedback_count"]) == 0,
        "no_actual_product_benefit_claim": decision.get(
            "actual_product_benefit_claim_allowed"
        )
        is False,
        "no_auto_promotion": decision.get("auto_promotion_allowed") is False,
        "no_auto_deployment": decision.get("automatic_deployment_allowed") is False,
        "no_hermes_attachment": decision.get("hermes_attachment_allowed") is False,
        "product_default_unchanged": decision.get("product_default_changed") is False,
    }
    summary = {
        "kind": "phase85_validation_summary",
        "created_at": _utcnow(),
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_failures": manifest_failures,
        "decision_status": decision.get("status"),
        "product_gate_qualified": decision.get("simulated_lab_runtime_benefit") is True,
        "validation_pass_does_not_imply_product_pass": True,
    }
    _write_json(config.evidence_root / "validation_summary.json", summary)
    _write_text(
        config.evidence_root / "validation_gate.txt",
        "PASS" if summary["passed"] else "FAIL",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


def _run_until_manual_review(config: DriverConfig, clean_evidence: bool) -> int:
    if _prepare(config, clean_evidence) != 0:
        return 1
    if _api_smoke(config, clean_evidence) != 0:
        return 1
    for variant in PHASE85_VARIANTS:
        if _generate(config, variant, clean_evidence) != 0:
            return 1
    return _review_template(config, clean_evidence)


def _clean_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clean-evidence", "--clean", action="store_true")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--mode", choices=("real", "mock"), default="real")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    _clean_flag(prepare)
    smoke = subparsers.add_parser("api-smoke")
    _clean_flag(smoke)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--variant", choices=PHASE85_VARIANTS, required=True)
    _clean_flag(generate)
    review = subparsers.add_parser("review-template")
    _clean_flag(review)
    subparsers.add_parser("full-regression")
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    run = subparsers.add_parser("run")
    _clean_flag(run)
    args = parser.parse_args()
    config = DriverConfig(
        evidence_root=_resolve_path(args.evidence_dir),
        model_path=_resolve_path(args.model_path),
        mode=str(args.mode),
    )
    clean_evidence = bool(getattr(args, "clean_evidence", False))
    if args.command == "prepare":
        return _prepare(config, clean_evidence)
    if args.command == "api-smoke":
        return _api_smoke(config, clean_evidence)
    if args.command == "generate":
        return _generate(config, str(args.variant), clean_evidence)
    if args.command == "review-template":
        return _review_template(config, clean_evidence)
    if args.command == "full-regression":
        return _full_regression(config)
    if args.command == "finalize":
        return _finalize(config)
    if args.command == "validate":
        return _validate(config)
    if args.command == "run":
        return _run_until_manual_review(config, clean_evidence)
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
