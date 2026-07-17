#!/usr/bin/env python3
"""Create and validate the read-only Phase85 metric schema v2 overlay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


SOURCE_REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_REPO_ROOT = SOURCE_REPO_ROOT
SOURCE_EVIDENCE_ROOT = (
    SOURCE_REPO_ROOT / "docs/demo/phase85-low-fallback-semantic-guard"
)
OVERLAY_ROOT_RELATIVE = Path("docs/demo/phase85-metric-schema-v2-overlay")
OVERLAY_FILENAME = "phase85-metric-schema-v2-overlay.json"
VALIDATION_FILENAME = "validation_summary.json"
ALLOWED_ARTIFACTS = frozenset({OVERLAY_FILENAME, VALIDATION_FILENAME})

FROZEN_DRIVER_SOURCE = (
    SOURCE_REPO_ROOT / "tools/phase85_low_fallback_semantic_guard.py"
)
FROZEN_PHASE85_SOURCE_PATHS = {
    "phase85_core": (
        SOURCE_REPO_ROOT / "pfe-core/pfe_core/phase85_low_fallback_semantic_guard.py"
    ),
    "phase85_driver": FROZEN_DRIVER_SOURCE,
    "phase85_driver_tests": SOURCE_REPO_ROOT / "tests/test_phase85_driver_safety.py",
    "phase85_engine_privacy_tests": (
        SOURCE_REPO_ROOT / "tests/test_phase85_engine_status_privacy.py"
    ),
    "phase85_semantic_hardening_tests": (
        SOURCE_REPO_ROOT / "tests/test_phase85_semantic_guard_hardening.py"
    ),
    "phase85_tests": (
        SOURCE_REPO_ROOT / "tests/test_phase85_low_fallback_semantic_guard.py"
    ),
}
EXPECTED_VARIANTS = (
    "base_api_length_control_160",
    "persona_api_contract_v3_fresh",
    "persona_api_contract_v4",
)
EXPECTED_COUNTS = {
    "session_count": 30,
    "completed_session_count": 30,
    "api_call_count": 90,
    "runtime_attempt_count": 90,
    "model_call_count": 90,
    "format_eligible_turn_count": 68,
}
EXPECTED_CANONICAL_STATUS = "archive_incomplete_phase85_evidence"
OVERLAY_DECISION_STATUS = "archive_low_fallback_runtime_not_qualified"
EXPECTED_BENEFIT_FAILURES = (
    "v4_target_score_at_least_0_80",
    "v4_target_not_below_v3",
    "v4_each_target_category_at_least_0_75",
    "v4_native_format_rate_at_least_0_75",
    "v4_fallback_rate_at_most_0_10",
    "v4_fallback_below_v3",
    "manual_review_found_no_semantic_failures",
)
CLAIM_BOUNDARIES = {
    "promotion_allowed": False,
    "deployment_allowed": False,
    "hermes_attachment_allowed": False,
    "training_benefit_claim_allowed": False,
    "product_benefit_claim_allowed": False,
}


class OverlayError(RuntimeError):
    """Raised when frozen Phase85 inputs cannot support the overlay."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OverlayError(f"expected a JSON object at {_source_path(path)}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _source_path(path: Path) -> str:
    try:
        return path.relative_to(SOURCE_REPO_ROOT).as_posix()
    except ValueError as exc:
        raise OverlayError("Phase85 source path escaped the repository") from exc


def _verify_frozen_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze_path = SOURCE_EVIDENCE_ROOT / "pre_experiment_freeze.json"
    freeze = _read_json(freeze_path)
    if freeze.get("passed") is not True:
        raise OverlayError("Phase85 pre_experiment_freeze is not passed")
    expected_hashes = freeze.get("source_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise OverlayError("Phase85 pre_experiment_freeze has no source hash map")

    bindings: dict[str, Any] = {}
    for key, path in FROZEN_PHASE85_SOURCE_PATHS.items():
        expected = expected_hashes.get(key)
        actual = _sha256_file(path) if path.is_file() else None
        if not isinstance(expected, str) or actual != expected:
            raise OverlayError(f"frozen source hash mismatch: {key}")
        bindings[key] = {
            "path": _source_path(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": True,
        }
    return freeze, bindings


def _load_frozen_driver() -> ModuleType:
    module_name = "_phase85_frozen_driver_for_metric_schema_v2_overlay"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, FROZEN_DRIVER_SOURCE)
    if spec is None or spec.loader is None:
        raise OverlayError("could not load the frozen Phase85 driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _variant_view(
    variant: str,
    stored: Mapping[str, Any],
    structural_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    count_checks = {
        key: stored.get(key) == expected for key, expected in EXPECTED_COUNTS.items()
    }
    accounting_exact = (
        stored.get("format_accounting_passed") is True
        and stored.get("native_format_turn_count")
        + stored.get("semantic_repair_turn_count")
        + stored.get("fallback_turn_count")
        == EXPECTED_COUNTS["format_eligible_turn_count"]
        and stored.get("format_fallback_turn_count")
        + stored.get("safety_fallback_turn_count")
        == stored.get("fallback_turn_count")
        and stored.get("one_api_call_per_turn") is True
        and stored.get("one_backend_attempt_per_turn") is True
        and stored.get("one_model_call_per_turn") is True
        and stored.get("extra_api_call_count") == 0
        and stored.get("extra_backend_attempt_count") == 0
        and stored.get("extra_model_call_count") == 0
    )
    if not all(count_checks.values()) or not accounting_exact:
        raise OverlayError(f"Phase85 count or format accounting mismatch: {variant}")

    fallback_rate = stored.get("fallback_turn_rate")
    factual_rate = stored.get("factual_guard_fallback_turn_rate")
    observed_rate = stored.get("observed_runtime_fallback_turn_rate")
    counterfactual_rate = stored.get("counterfactual_guard_fallback_turn_rate")
    if variant == EXPECTED_VARIANTS[0]:
        variant_rules_passed = (
            stored.get("format_metrics_basis")
            == "counterfactual_common_v4_evaluator"
            and stored.get("fallback_turn_rate_is_counterfactual") is True
            and factual_rate is None
            and observed_rate is None
            and _finite_number(counterfactual_rate)
            and counterfactual_rate == fallback_rate
        )
    else:
        variant_rules_passed = (
            stored.get("format_metrics_basis") == "runtime_contract_output"
            and stored.get("fallback_turn_rate_is_counterfactual") is False
            and counterfactual_rate is None
            and _finite_number(factual_rate)
            and _finite_number(observed_rate)
            and factual_rate == fallback_rate
            and observed_rate == fallback_rate
        )
    if not variant_rules_passed:
        raise OverlayError(f"Phase85 metric schema v2 rule mismatch: {variant}")

    return {
        "structural_jsonl": {
            "path": _source_path(structural_path),
            "sha256": _sha256_file(structural_path),
        },
        "stored_metrics": {
            "path": _source_path(metrics_path),
            "sha256": _sha256_file(metrics_path),
            "exactly_reaggregated_with_frozen_driver": True,
        },
        "session_count": stored.get("session_count"),
        "completed_session_count": stored.get("completed_session_count"),
        "api_call_count": stored.get("api_call_count"),
        "runtime_attempt_count": stored.get("runtime_attempt_count"),
        "model_call_count": stored.get("model_call_count"),
        "format_eligible_turn_count": stored.get("format_eligible_turn_count"),
        "native_format_turn_count": stored.get("native_format_turn_count"),
        "semantic_repair_turn_count": stored.get("semantic_repair_turn_count"),
        "format_fallback_turn_count": stored.get("format_fallback_turn_count"),
        "safety_fallback_turn_count": stored.get("safety_fallback_turn_count"),
        "fallback_turn_count": stored.get("fallback_turn_count"),
        "fallback_turn_rate": fallback_rate,
        "fallback_turn_rate_is_counterfactual": stored.get(
            "fallback_turn_rate_is_counterfactual"
        ),
        "format_metrics_basis": stored.get("format_metrics_basis"),
        "factual_guard_fallback_turn_rate": factual_rate,
        "observed_runtime_fallback_turn_rate": observed_rate,
        "counterfactual_guard_fallback_turn_rate": counterfactual_rate,
        "format_accounting_exact": accounting_exact,
        "metric_schema_v2_rules_passed": variant_rules_passed,
    }


def _canonical_decision() -> tuple[dict[str, Any], dict[str, Any]]:
    path = SOURCE_EVIDENCE_ROOT / "phase85-final-decision.json"
    decision = _read_json(path)
    failures = decision.get("failed_benefit_checks")
    false_benefit_checks = {
        key
        for key, value in dict(decision.get("benefit_checks") or {}).items()
        if value is False
    }
    if decision.get("status") != EXPECTED_CANONICAL_STATUS:
        raise OverlayError("canonical Phase85 archive_incomplete status changed")
    if failures != list(EXPECTED_BENEFIT_FAILURES):
        raise OverlayError("canonical Phase85 benefit failure list changed")
    if false_benefit_checks != set(EXPECTED_BENEFIT_FAILURES):
        raise OverlayError("canonical Phase85 failed benefit checks changed")
    denied_claims = (
        "promotion_allowed",
        "auto_promotion_allowed",
        "automatic_deployment_allowed",
        "hermes_attachment_allowed",
        "actual_product_benefit_claim_allowed",
        "simulated_lab_runtime_benefit",
        "product_default_changed",
    )
    if any(decision.get(key) is not False for key in denied_claims):
        raise OverlayError("canonical Phase85 claim boundary changed")
    return decision, {
        "path": _source_path(path),
        "sha256": _sha256_file(path),
        "status": decision["status"],
        "failed_benefit_checks": list(failures),
    }


def _build_overlay() -> dict[str, Any]:
    freeze, frozen_sources = _verify_frozen_sources()
    driver = _load_frozen_driver()
    variants = tuple(driver.PHASE85_VARIANTS)
    if variants != EXPECTED_VARIANTS:
        raise OverlayError("frozen Phase85 variant order changed")

    config = driver.DriverConfig(
        evidence_root=SOURCE_EVIDENCE_ROOT,
        model_path=driver.DEFAULT_MODEL_PATH,
        mode=str(freeze.get("execution_mode") or ""),
    )
    sessions = [dict(row) for row in driver.build_phase85_holdout()["sessions"]]
    variant_views: dict[str, Any] = {}
    for variant in variants:
        structural_path, metrics_path = driver._variant_paths(config, variant)
        stored = driver._read_json(metrics_path)
        reaggregated = driver._aggregate_variant(
            driver._read_jsonl(structural_path),
            sessions,
            variant=variant,
            config=config,
        )
        if stored != reaggregated:
            raise OverlayError(
                f"stored metrics do not exactly reaggregate with frozen driver: {variant}"
            )
        variant_views[variant] = _variant_view(
            variant, stored, structural_path, metrics_path
        )

    _, canonical_binding = _canonical_decision()
    checks = {
        "frozen_phase85_driver_core_test_hashes_match": all(
            binding["matched"] for binding in frozen_sources.values()
        ),
        "stored_metrics_exactly_reaggregate_with_frozen_driver": all(
            view["stored_metrics"]["exactly_reaggregated_with_frozen_driver"]
            for view in variant_views.values()
        ),
        "all_variants_have_30_sessions_90_calls_68_eligible": all(
            view["session_count"] == 30
            and view["completed_session_count"] == 30
            and view["api_call_count"] == 90
            and view["format_eligible_turn_count"] == 68
            for view in variant_views.values()
        ),
        "all_variant_accounting_exact": all(
            view["format_accounting_exact"] for view in variant_views.values()
        ),
        "variant_aware_metric_schema_v2_rules_pass": all(
            view["metric_schema_v2_rules_passed"] for view in variant_views.values()
        ),
        "canonical_decision_hash_bound": bool(canonical_binding["sha256"]),
        "canonical_archive_incomplete_status_preserved_separately": (
            canonical_binding["status"] == EXPECTED_CANONICAL_STATUS
        ),
        "all_seven_benefit_failures_unchanged": (
            canonical_binding["failed_benefit_checks"]
            == list(EXPECTED_BENEFIT_FAILURES)
        ),
        "promotion_deployment_hermes_training_product_claims_denied": not any(
            CLAIM_BOUNDARIES.values()
        ),
    }
    payload: dict[str, Any] = {
        "kind": "phase85_metric_schema_v2_overlay",
        "schema_version": 2,
        "read_only_source_overlay": True,
        "source_bindings": {
            "pre_experiment_freeze": {
                "path": _source_path(
                    SOURCE_EVIDENCE_ROOT / "pre_experiment_freeze.json"
                ),
                "sha256": _sha256_file(
                    SOURCE_EVIDENCE_ROOT / "pre_experiment_freeze.json"
                ),
                "passed": True,
            },
            "frozen_phase85_sources": frozen_sources,
            "canonical_decision": canonical_binding,
        },
        "expected_counts_per_variant": dict(EXPECTED_COUNTS),
        "variants": variant_views,
        "checks": checks,
        "schema_validation_passed": all(checks.values()),
        "overlay_decision": {
            "status": OVERLAY_DECISION_STATUS,
            "canonical_archive_status": EXPECTED_CANONICAL_STATUS,
            "product_gate_qualified": False,
            "failed_benefit_checks": list(EXPECTED_BENEFIT_FAILURES),
            "failed_benefit_check_count": len(EXPECTED_BENEFIT_FAILURES),
            "claim_boundaries": dict(CLAIM_BOUNDARIES),
            "validation_pass_does_not_imply_product_pass": True,
        },
    }
    payload["payload_sha256"] = _stable_hash(payload)
    return payload


def _output_root() -> Path:
    current = OUTPUT_REPO_ROOT
    if current.is_symlink():
        raise OverlayError("output repository root must not be a symlink")
    for part in OVERLAY_ROOT_RELATIVE.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise OverlayError("overlay output directory chain contains a symlink")
    current.mkdir(parents=True, exist_ok=True)
    return current


def _artifact_path(filename: str) -> Path:
    if filename not in ALLOWED_ARTIFACTS:
        raise OverlayError(f"unsupported overlay artifact: {filename}")
    return _output_root() / filename


def _write_artifact(filename: str, value: Any) -> Path:
    path = _artifact_path(filename)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)
    return path


def _finalize() -> int:
    overlay = _build_overlay()
    _write_artifact(OVERLAY_FILENAME, overlay)
    result = {
        "artifact": (OVERLAY_ROOT_RELATIVE / OVERLAY_FILENAME).as_posix(),
        "schema_validation_passed": overlay["schema_validation_passed"],
        "overlay_decision_status": overlay["overlay_decision"]["status"],
        "product_gate_qualified": False,
    }
    print(_json_bytes(result).decode("utf-8"), end="")
    return 0


def _payload_hash_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    expected = value.get("payload_sha256")
    unhashed = dict(value)
    unhashed.pop("payload_sha256", None)
    return isinstance(expected, str) and expected == _stable_hash(unhashed)


def _validate() -> int:
    expected = _build_overlay()
    overlay_path = _artifact_path(OVERLAY_FILENAME)
    actual_bytes = overlay_path.read_bytes() if overlay_path.is_file() else None
    actual: Any = None
    if actual_bytes is not None:
        try:
            actual = json.loads(actual_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            actual = None
    checks = {
        "overlay_artifact_exists": actual_bytes is not None,
        "overlay_exactly_matches_deterministic_rerender": (
            actual_bytes == _json_bytes(expected)
        ),
        "overlay_payload_hash_valid": _payload_hash_valid(actual),
        "overlay_decision_status_preserved": isinstance(actual, dict)
        and dict(actual.get("overlay_decision") or {}).get("status")
        == OVERLAY_DECISION_STATUS,
        "product_gate_remains_not_qualified": isinstance(actual, dict)
        and dict(actual.get("overlay_decision") or {}).get("product_gate_qualified")
        is False,
        "all_seven_benefit_failures_unchanged": isinstance(actual, dict)
        and dict(actual.get("overlay_decision") or {}).get("failed_benefit_checks")
        == list(EXPECTED_BENEFIT_FAILURES),
        "all_claim_boundaries_remain_denied": isinstance(actual, dict)
        and dict(actual.get("overlay_decision") or {}).get("claim_boundaries")
        == CLAIM_BOUNDARIES,
    }
    passed = all(checks.values())
    summary = {
        "kind": "phase85_metric_schema_v2_overlay_validation",
        "passed": passed,
        "checks": checks,
        "overlay_artifact": (
            OVERLAY_ROOT_RELATIVE / OVERLAY_FILENAME
        ).as_posix(),
        "expected_overlay_payload_sha256": expected["payload_sha256"],
        "overlay_decision_status": OVERLAY_DECISION_STATUS,
        "product_gate_qualified": False,
        "validation_pass_does_not_imply_product_pass": True,
        "validation_scope": (
            "A validation pass verifies overlay integrity only and does not imply "
            "product pass."
        ),
    }
    _write_artifact(VALIDATION_FILENAME, summary)
    print(_json_bytes(summary).decode("utf-8"), end="")
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("finalize")
    subparsers.add_parser("validate")
    args = parser.parse_args(argv)
    try:
        if args.command == "finalize":
            return _finalize()
        if args.command == "validate":
            return _validate()
    except (OverlayError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            _json_bytes(
                {
                    "passed": False,
                    "error": str(exc),
                    "product_gate_qualified": False,
                    "validation_pass_does_not_imply_product_pass": True,
                }
            ).decode("utf-8"),
            end="",
            file=sys.stderr,
        )
        return 1
    raise OverlayError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
