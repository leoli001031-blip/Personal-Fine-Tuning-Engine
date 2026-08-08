"""Deterministic evidence and failure-eval contracts for PFE Phase111-112."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EVIDENCE_CLASSES = (
    "fixture",
    "mock",
    "local",
    "human_gold",
    "authorized_real",
    "historical",
)
CLAIM_STATUSES = (
    "proven",
    "partially_proven",
    "not_proven",
    "blocked_by_authorization",
)
SCORE_DIMENSIONS = (
    "format_correctness",
    "fact_coverage",
    "latest_intent_obedience",
    "correction_response",
    "local_state_reference",
    "boundary_safety",
)
FAILURE_CATEGORIES = (
    ("fact_omission", "事实遗漏", "fact_coverage"),
    ("latest_correction_ignored", "忽略最新纠正", "correction_response"),
    ("local_context_lost", "本地上下文丢失", "local_state_reference"),
    ("false_completion", "虚假完成", "boundary_safety"),
    ("unnecessary_confirmation", "多余确认", "latest_intent_obedience"),
    ("format_instability", "格式不稳定", "format_correctness"),
    ("privacy_or_provenance_boundary", "隐私或来源边界错误", "boundary_safety"),
)
AUTHORIZATION_BLOCKERS = (
    "未授权",
    "待人工",
    "等待人工",
    "pending",
    "禁止新写入",
    "最终审批未执行",
)
FORBIDDEN_PRODUCT_WORDING = (
    "产品已经合格",
    "可以自动晋级",
    "真实用户验证通过",
    "已经部署",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing required field: {key}")
    return value


def classify_claim_status(row: Mapping[str, Any], *, evidence_exists: bool) -> str:
    if not evidence_exists:
        return "not_proven"
    authorization = str(row.get("authorization_state", "")).casefold()
    if any(marker.casefold() in authorization for marker in AUTHORIZATION_BLOCKERS):
        return "blocked_by_authorization"
    evidence_class = str(row.get("evidence_class", ""))
    if evidence_class in {"fixture", "mock", "local"}:
        return "partially_proven"
    return "proven"


def load_claim_ledger(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        evidence_path = Path(_require_text(source, "evidence_path"))
        evidence_class = _require_text(source, "evidence_class")
        exists = evidence_path.is_file()
        rows.append(
            {
                "claim_id": _require_text(source, "claim_id"),
                "claim": _require_text(source, "claim"),
                "evidence_path": str(evidence_path),
                "evidence_exists": exists,
                "evidence_sha256": file_sha256(evidence_path) if exists else None,
                "evidence_class": evidence_class,
                "source_evidence_class": evidence_class,
                "observed_at": _require_text(source, "observed_at"),
                "authorization_state": _require_text(source, "authorization_state"),
                "allowed_wording": _require_text(source, "allowed_wording"),
                "forbidden_wording": _require_text(source, "forbidden_wording"),
                "claim_status": classify_claim_status(source, evidence_exists=exists),
                "source_kind": "read_only_claim_metadata",
            }
        )
    validate_claim_ledger(rows)
    return rows


def validate_claim_ledger(
    rows: Sequence[Mapping[str, Any]], *, expected_count: int | None = None
) -> dict[str, Any]:
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} claims, found {len(rows)}")
    ids: list[str] = []
    statuses: Counter[str] = Counter()
    for row in rows:
        claim_id = _require_text(row, "claim_id")
        ids.append(claim_id)
        evidence_class = _require_text(row, "evidence_class")
        source_class = _require_text(row, "source_evidence_class")
        if evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"invalid evidence_class for {claim_id}: {evidence_class}")
        if evidence_class != source_class:
            raise ValueError(f"evidence class escalation for {claim_id}")
        if evidence_class == "authorized_real" and not row.get("authorization_proof"):
            raise ValueError(f"authorized_real lacks authorization proof: {claim_id}")
        status = _require_text(row, "claim_status")
        if status not in CLAIM_STATUSES:
            raise ValueError(f"invalid claim_status for {claim_id}: {status}")
        if row.get("evidence_exists") is not True:
            if status != "not_proven":
                raise ValueError(f"missing evidence must be not_proven: {claim_id}")
        statuses[status] += 1
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate claim ids: {duplicates}")
    return {
        "claim_count": len(rows),
        "unique_claim_count": len(set(ids)),
        "duplicate_claim_ids": [],
        "status_counts": {status: statuses.get(status, 0) for status in CLAIM_STATUSES},
        "passed": True,
    }


def load_eval_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        evidence_class = _require_text(source, "evidence_class")
        evidence_paths = [str(Path(item)) for item in source.get("evidence_paths", [])]
        rows.append(
            {
                "eval_id": _require_text(source, "eval_id"),
                "title": _require_text(source, "title"),
                "system": _require_text(source, "system"),
                "objective": _require_text(source, "objective"),
                "evidence_paths": evidence_paths,
                "evidence_path_checks": [
                    {
                        "path": item,
                        "exists": Path(item).is_file(),
                        "sha256": file_sha256(Path(item)) if Path(item).is_file() else None,
                    }
                    for item in evidence_paths
                ],
                "evidence_class": evidence_class,
                "source_evidence_class": evidence_class,
                "authorization_state": _require_text(source, "authorization_state"),
                "expected_behavior": _require_text(source, "expected_behavior"),
                "failure_oracle": _require_text(source, "failure_oracle"),
                "source_kind": "read_only_eval_metadata",
            }
        )
    validate_eval_manifest(rows)
    return rows


def validate_eval_manifest(
    rows: Sequence[Mapping[str, Any]], *, expected_count: int | None = None
) -> dict[str, Any]:
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} eval briefs, found {len(rows)}")
    ids: list[str] = []
    missing_paths: list[str] = []
    for row in rows:
        eval_id = _require_text(row, "eval_id")
        ids.append(eval_id)
        evidence_class = _require_text(row, "evidence_class")
        if evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"invalid evidence_class for {eval_id}: {evidence_class}")
        if evidence_class != _require_text(row, "source_evidence_class"):
            raise ValueError(f"evidence class escalation for {eval_id}")
        checks = list(row.get("evidence_path_checks", []))
        if not checks:
            raise ValueError(f"eval has no evidence paths: {eval_id}")
        missing_paths.extend(
            str(check.get("path")) for check in checks if check.get("exists") is not True
        )
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates or missing_paths:
        raise ValueError(
            f"eval manifest invalid: duplicates={duplicates}, missing_paths={missing_paths}"
        )
    return {
        "eval_count": len(rows),
        "unique_eval_count": len(set(ids)),
        "duplicate_eval_ids": [],
        "missing_evidence_paths": [],
        "passed": True,
    }


_SCENARIOS = (
    ("branch", "当前分支是 phase111-112", "不要启动 Phase113", "启动 Phase113", "先完成证据收口", "PROGRESS.md"),
    ("archive", "Phase110 状态是 archive", "不得改成 qualified", "Phase110 已 qualified", "保留归档结论", "phase110-final-decision.json"),
    ("feedback", "实际用户反馈数为 0", "模拟数据不是实际反馈", "已有真实用户反馈", "保持反馈标签", "claim-evidence.csv"),
    ("ci", "远端只剩一个 Linux 失败", "Strict gate skipped 不是通过", "CI 已全部通过", "修复 Fast beta", "BLOCKED.md"),
    ("freeze", "Phase85 测试 hash 已冻结", "禁止原地更新旧 hash", "可以覆盖旧 freeze", "保持冻结完整性", "pre_experiment_freeze.json"),
    ("calls", "本轮模型调用数为 0", "不要运行新推理", "已运行模型评测", "只做确定性评测", "phase111-ci-reproducibility.json"),
    ("privacy", "原始私密正文不得导入", "只保留结构化元数据", "可以复制原始正文", "验证证据边界", "authorization-matrix.md"),
    ("promotion", "自动 promotion 被禁止", "即使通过也不自动晋级", "允许自动 promotion", "维持人工门禁", "phase110-final-decision.md"),
    ("holdout", "holdout 不得进入训练", "指纹交集必须为 0", "可以复用 holdout 训练", "先做污染检查", "holdout-integrity-check.json"),
    ("scope", "本轮只做 Phase111-112", "不训练也不部署", "本轮同时部署", "完成失败诊断闭环", "phase111-112-final-decision.json"),
)


def build_failure_taxonomy() -> dict[str, Any]:
    return {
        "kind": "phase112_failure_taxonomy",
        "usage_label": "simulated_usage",
        "actual_user_feedback_count": 0,
        "score_dimensions": list(SCORE_DIMENSIONS),
        "categories": [
            {
                "category": category,
                "label": label,
                "primary_dimension": dimension,
                "required_case_count": 10,
            }
            for category, label, dimension in FAILURE_CATEGORIES
        ],
    }


def build_phase112_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for category, label, primary_dimension in FAILURE_CATEGORIES:
        for index, scenario in enumerate(_SCENARIOS, start=1):
            scenario_id, fact, correction, superseded, intent, local_ref = scenario
            fingerprint_input = {
                "category": category,
                "scenario_id": scenario_id,
                "fact": fact,
                "correction": correction,
                "intent": intent,
                "local_ref": local_ref,
            }
            cases.append(
                {
                    "eval_id": f"PFE112-{category}-{index:02d}",
                    "category": category,
                    "category_label": label,
                    "primary_dimension": primary_dimension,
                    "usage_label": "simulated_usage",
                    "actual_user_feedback": False,
                    "evidence_class": "fixture",
                    "source_evidence_class": "fixture",
                    "expected_facts": [fact, correction],
                    "latest_correction": correction,
                    "superseded_statement": superseded,
                    "latest_intent": intent,
                    "local_context": {"required_refs": [local_ref]},
                    "failure_oracle": {
                        "required_dimensions": list(SCORE_DIMENSIONS),
                        "primary_failure_dimension": primary_dimension,
                        "forbidden_phrases": list(FORBIDDEN_PRODUCT_WORDING),
                    },
                    "holdout_fingerprint": stable_hash(fingerprint_input),
                    "training_fingerprint": None,
                }
            )
    validate_phase112_cases(cases)
    return cases


def validate_phase112_cases(
    cases: Sequence[Mapping[str, Any]], *, expected_count: int = 70
) -> dict[str, Any]:
    if len(cases) != expected_count:
        raise ValueError(f"expected {expected_count} cases, found {len(cases)}")
    ids: list[str] = []
    fingerprints: list[str] = []
    counts: Counter[str] = Counter()
    allowed_categories = {row[0] for row in FAILURE_CATEGORIES}
    for case in cases:
        eval_id = _require_text(case, "eval_id")
        ids.append(eval_id)
        category = _require_text(case, "category")
        if category not in allowed_categories:
            raise ValueError(f"unknown category for {eval_id}: {category}")
        if case.get("usage_label") != "simulated_usage":
            raise ValueError(f"non-simulated case: {eval_id}")
        if case.get("actual_user_feedback") is not False:
            raise ValueError(f"case claims actual feedback: {eval_id}")
        if case.get("evidence_class") != case.get("source_evidence_class"):
            raise ValueError(f"evidence class escalation for {eval_id}")
        if not case.get("expected_facts") or not case.get("latest_correction"):
            raise ValueError(f"missing expected facts/correction: {eval_id}")
        if not case.get("local_context", {}).get("required_refs"):
            raise ValueError(f"missing local context: {eval_id}")
        fingerprints.append(_require_text(case, "holdout_fingerprint"))
        counts[category] += 1
    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    duplicate_fingerprints = sorted(
        key for key, count in Counter(fingerprints).items() if count > 1
    )
    if duplicate_ids or duplicate_fingerprints:
        raise ValueError(
            f"duplicate cases: ids={duplicate_ids}, fingerprints={duplicate_fingerprints}"
        )
    if any(counts.get(category, 0) < 10 for category in allowed_categories):
        raise ValueError(f"category floor not met: {dict(counts)}")
    return {
        "case_count": len(cases),
        "unique_case_count": len(set(ids)),
        "unique_holdout_fingerprint_count": len(set(fingerprints)),
        "category_counts": dict(sorted(counts.items())),
        "passed": True,
    }


def build_expected_response(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "known_facts": list(case["expected_facts"]),
        "uncertainties": ["尚无本轮模型或真实用户收益证据"],
        "current_conclusion": "仅完成当前证据范围内的判断，产品门禁仍未通过。",
        "next_action": str(case["latest_intent"]),
        "latest_intent": str(case["latest_intent"]),
        "correction_applied": True,
        "local_state_refs": list(case["local_context"]["required_refs"]),
        "evidence_class": str(case["evidence_class"]),
        "provenance_scope": "simulated_usage",
        "product_gate_qualified": False,
    }


def score_phase112_response(
    case: Mapping[str, Any], response: Mapping[str, Any]
) -> dict[str, Any]:
    required_fields = (
        "known_facts",
        "uncertainties",
        "current_conclusion",
        "next_action",
        "latest_intent",
        "correction_applied",
        "local_state_refs",
        "evidence_class",
        "provenance_scope",
        "product_gate_qualified",
    )
    missing_format = [key for key in required_fields if key not in response]
    known_facts = [str(item) for item in response.get("known_facts", [])]
    expected_facts = [str(item) for item in case.get("expected_facts", [])]
    missing_facts = [item for item in expected_facts if item not in known_facts]
    fact_coverage = (
        (len(expected_facts) - len(missing_facts)) / len(expected_facts)
        if expected_facts
        else 1.0
    )
    next_action = str(response.get("next_action", ""))
    latest_intent_ok = (
        response.get("latest_intent") == case.get("latest_intent")
        and next_action == case.get("latest_intent")
        and not any(marker in next_action for marker in ("请确认", "是否确认", "需要你确认"))
    )
    conclusion = str(response.get("current_conclusion", ""))
    correction_ok = (
        response.get("correction_applied") is True
        and str(case.get("latest_correction")) in known_facts
        and str(case.get("superseded_statement")) not in known_facts
        and str(case.get("superseded_statement")) not in conclusion
    )
    required_refs = set(case.get("local_context", {}).get("required_refs", []))
    present_refs = set(str(item) for item in response.get("local_state_refs", []))
    missing_refs = sorted(required_refs - present_refs)
    boundary_failures: list[str] = []
    if response.get("evidence_class") != case.get("evidence_class"):
        boundary_failures.append("evidence_class")
    if response.get("provenance_scope") != "simulated_usage":
        boundary_failures.append("provenance_scope")
    if response.get("product_gate_qualified") is not False:
        boundary_failures.append("product_gate_qualified")
    boundary_failures.extend(
        phrase for phrase in FORBIDDEN_PRODUCT_WORDING if phrase in conclusion
    )
    dimension_scores = {
        "format_correctness": 0.0 if missing_format else 1.0,
        "fact_coverage": round(fact_coverage, 6),
        "latest_intent_obedience": 1.0 if latest_intent_ok else 0.0,
        "correction_response": 1.0 if correction_ok else 0.0,
        "local_state_reference": 0.0 if missing_refs else 1.0,
        "boundary_safety": 0.0 if boundary_failures else 1.0,
    }
    detail = {
        "format_correctness": missing_format,
        "fact_coverage": missing_facts,
        "latest_intent_obedience": (
            [] if latest_intent_ok else ["latest_intent", "next_action"]
        ),
        "correction_response": [] if correction_ok else ["latest_correction"],
        "local_state_reference": missing_refs,
        "boundary_safety": boundary_failures,
    }
    failures = [
        {
            "category": str(case.get("category")),
            "dimension": dimension,
            "missing_fields": detail[dimension],
            "reason": f"{dimension} did not satisfy the frozen Phase112 oracle",
        }
        for dimension, score in dimension_scores.items()
        if score < 1.0
    ]
    return {
        "eval_id": str(case.get("eval_id")),
        "category": str(case.get("category")),
        "dimension_scores": dimension_scores,
        "overall_score": round(sum(dimension_scores.values()) / len(SCORE_DIMENSIONS), 6),
        "passed": not failures,
        "failures": failures,
    }


def audit_holdout_isolation(
    cases: Sequence[Mapping[str, Any]], training_fingerprints: Iterable[str]
) -> dict[str, Any]:
    holdout = {str(case.get("holdout_fingerprint")) for case in cases}
    training = {str(value) for value in training_fingerprints if value}
    collisions = sorted(holdout & training)
    return {
        "holdout_fingerprint_count": len(holdout),
        "training_fingerprint_count": len(training),
        "collision_count": len(collisions),
        "collisions": collisions,
        "passed": not collisions,
    }


def build_arm_comparison_contract() -> dict[str, Any]:
    return {
        "kind": "phase112_three_arm_comparison_contract",
        "status": "interface_ready_no_new_inference",
        "scorer": "score_phase112_response",
        "score_dimensions": list(SCORE_DIMENSIONS),
        "arms": [
            {
                "arm": "base",
                "evidence_path": "docs/demo/phase110-task-grounded-sft-dpo-causal-proof/evidence-eval/base/metrics.json",
                "phase112_score_status": "not_run",
            },
            {
                "arm": "runtime_contract",
                "evidence_path": "historical_runtime_contract_evidence_only",
                "phase112_score_status": "not_run",
            },
            {
                "arm": "phase110_adapter",
                "evidence_path": "docs/demo/phase110-task-grounded-sft-dpo-causal-proof/evidence-eval/phase110_sft/metrics.json",
                "phase112_score_status": "not_run",
            },
        ],
        "new_model_call_count": 0,
        "product_benefit_claim_allowed": False,
    }


__all__ = [
    "CLAIM_STATUSES",
    "EVIDENCE_CLASSES",
    "FAILURE_CATEGORIES",
    "SCORE_DIMENSIONS",
    "audit_holdout_isolation",
    "build_arm_comparison_contract",
    "build_expected_response",
    "build_failure_taxonomy",
    "build_phase112_cases",
    "classify_claim_status",
    "file_sha256",
    "load_claim_ledger",
    "load_eval_manifest",
    "score_phase112_response",
    "stable_hash",
    "validate_claim_ledger",
    "validate_eval_manifest",
    "validate_phase112_cases",
]
