from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from pfe_core.db.sqlite import list_samples
from pfe_core.phase6_candidate_adapter_trial import (
    PHASE6_RECOMMENDED_MODEL,
    Phase6CandidateAdapterTrialStore,
    qwen36_mlx_preflight,
    run_phase6_candidate_adapter_trial,
)


def _fake_contract_text(source: Mapping[str, object]) -> str:
    title = str(source["title"])
    return (
        f"# {title}\n\n"
        "1. Service terms. The provider supplies a cloud service for internal business use. "
        "2. Data processing. Customer content may be processed only to provide and improve "
        "the service, and personal data handling requires a data processing addendum. "
        "3. Fees and termination. Fees, renewal, suspension, cure periods, and termination "
        "rights must be checked against the complete agreement. "
        "4. Boundary. This material supports contract summarization and risk flagging only; "
        "it does not support a final legal conclusion and requires human confirmation. "
    ) * 3


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_phase6_qwen_preflight_reports_missing_local_model(tmp_path: Path) -> None:
    preflight = qwen36_mlx_preflight(
        model_path=tmp_path / "missing-qwen36-27b-4bit",
        require_local_model=True,
    )

    assert preflight["kind"] == "phase6_qwen36_mlx_preflight"
    assert preflight["model_id"] == PHASE6_RECOMMENDED_MODEL
    assert preflight["backend"] == "mlx"
    assert preflight["status"] == "blocked"
    assert "local_model_missing" in preflight["blocked_by"]
    assert preflight["ready_for_real_training"] is False
    assert preflight["recommended_training"]["lora_rank"] == 8


def test_phase6_trial_materializes_signal_provenance_samples_and_blocks_promotion(tmp_path: Path) -> None:
    result = run_phase6_candidate_adapter_trial(
        home=tmp_path,
        workspace="demo",
        source_limit=4,
        candidate_limit=20,
        holdout_count=12,
        require_local_model=True,
        model_path=tmp_path / "missing-qwen36-27b-4bit",
        fetch_text=_fake_contract_text,
    )
    samples = _read_jsonl(result["paths"]["candidate_samples"])
    db_samples = list_samples(home=tmp_path)

    assert result["ok"] is True
    assert result["trial_status"] == "preflight_blocked"
    assert result["candidate_samples"]["count"] >= 10
    assert result["holdout"]["count"] == 12
    assert result["eval_gate"]["status"] == "blocked"
    assert result["eval_gate"]["promotion_allowed"] is False
    assert result["decision"]["action"] == "archive"
    assert all("phase6-holdout" not in sample["sample_id"] for sample in db_samples)
    assert any(sample["metadata"].get("phase") == "phase6" for sample in db_samples)
    for sample in samples[:5]:
        metadata = sample["metadata"]
        assert metadata["phase"] == "phase6"
        assert metadata["trial_id"] == result["trial_id"]
        assert metadata["signal_id"]
        assert metadata["source_ids"]
        assert metadata["chunk_ids"]
        assert metadata["provenance"]["source_url"].startswith("https://raw.githubusercontent.com/CommonPaper/")
        assert metadata["not_holdout"] is True


def test_phase6_decision_can_promote_only_when_eval_gate_allows_it(tmp_path: Path) -> None:
    store = Phase6CandidateAdapterTrialStore(home=tmp_path, workspace="demo")
    manifest = {"trial_id": "p6trial-test", "workspace": "demo"}
    decision = store.decide_trial(
        manifest=manifest,
        eval_report={
            "eval_gate": {
                "status": "pass",
                "promotion_allowed": True,
                "reasons": ["real model eval passed"],
            }
        },
    )

    assert decision["status"] == "promoted"
    assert decision["action"] == "promote"
    assert decision["promotion_allowed"] is True
