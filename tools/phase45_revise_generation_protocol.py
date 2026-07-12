#!/usr/bin/env python3
"""Archive failed Phase45 diagnostic generation and freeze the fair v2 protocol."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "docs" / "demo" / "phase45-privacy-structural-multiturn-preference"
RUNS_ROOT = EVIDENCE_ROOT / "evidence-diagnostic" / "runs"
ARCHIVE_ROOT = EVIDENCE_ROOT / "evidence-diagnostic" / "protocol-v1-failed"
PROTOCOL_PATH = EVIDENCE_ROOT / "evidence-holdout" / "generation_protocol.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    metrics_path = RUNS_ROOT / "metrics_base_privacy.json"
    transcript_path = RUNS_ROOT / "transcripts_base_privacy.jsonl"
    freeze_path = RUNS_ROOT / "freeze_check_base_privacy.json"
    if not metrics_path.exists() or not transcript_path.exists():
        raise SystemExit("completed protocol-v1 base_privacy diagnostic evidence is required")
    metrics = _read_json(metrics_path)
    if int(metrics.get("session_count") or 0) != 18 or metrics.get("all_transcripts_completed") is not True:
        raise SystemExit("protocol-v1 diagnostic must be complete before revision")
    if float(metrics.get("truncated_response_rate") or 0.0) <= 0.05:
        raise SystemExit("protocol-v1 already meets the truncation gate; revision is not justified")

    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    for source in (metrics_path, transcript_path, freeze_path):
        if source.exists():
            shutil.copy2(source, ARCHIVE_ROOT / source.name)
    old_protocol = _read_json(PROTOCOL_PATH)
    _write_json(ARCHIVE_ROOT / "generation_protocol_v1.json", old_protocol)
    failure = {
        "kind": "phase45_generation_protocol_v1_failure",
        "status": "failed_preflight",
        "created_at": _utcnow(),
        "variant": "base_privacy",
        "session_count": metrics.get("session_count"),
        "model_call_count": metrics.get("model_call_count"),
        "truncated_response_rate": metrics.get("truncated_response_rate"),
        "required_maximum": 0.05,
        "observed_failure": "verbose heading/table/background expansion continued to the 384-token cap",
        "scorer_changed": False,
        "holdout_changed": False,
        "unfavorable_outputs_preserved": True,
        "formal_holdout_started": False,
    }
    _write_json(ARCHIVE_ROOT / "failure_summary.json", failure)

    protocol = {
        "kind": "phase45_fair_generation_protocol",
        "protocol_version": 2,
        "revised_at": _utcnow(),
        "revision_basis": str(ARCHIVE_ROOT / "failure_summary.json"),
        "revised_before_formal_holdout": True,
        "scorer_changed": False,
        "holdout_changed": False,
        "max_new_tokens": 384,
        "input_max_length": 4096,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "eos_token_from_tokenizer": True,
        "think": False,
        "uniform_system_contract": "每次回答最多三个短句，总长度不超过120个汉字。只回答当前请求，不使用标题、表格、代码块或背景扩写；给出结论后立即结束。",
        "uniform_system_contract_applied_to_all_arms": True,
        "required_all_arm_truncated_response_rate_max": 0.05,
        "formal_eval_requires_preflight": True,
    }
    protocol["protocol_sha256"] = _stable_hash(protocol)
    _write_json(PROTOCOL_PATH, protocol)
    _write_json(EVIDENCE_ROOT / "evidence-diagnostic" / "generation_protocol_revision.json", {
        "kind": "phase45_generation_protocol_revision",
        "status": "v2_frozen_for_diagnostic_rerun",
        "v1_failure": failure,
        "v2_protocol": protocol,
        "next_action": "rerun all diagnostic arms under the identical v2 protocol",
    })
    print(json.dumps({
        "status": "v2_frozen_for_diagnostic_rerun",
        "v1_truncated_response_rate": metrics.get("truncated_response_rate"),
        "v2_protocol_sha256": protocol["protocol_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
