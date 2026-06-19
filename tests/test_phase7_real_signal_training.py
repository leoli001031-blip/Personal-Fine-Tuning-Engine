from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from pfe_core.db.sqlite import list_samples
from pfe_core.phase7_real_signal_training import (
    PHASE7_RECOMMENDED_MODEL,
    Phase7RealSignalTrainingStore,
    finalize_phase7_real_signal_trial,
    phase7_default_sources,
    prepare_phase7_real_signal_trial,
)


def _fake_contract_text(source: Mapping[str, object]) -> str:
    title = str(source.get("title") or "Contract")
    source_id = str(source.get("source_id") or "")
    if source_id == "cp-csa":
        return (
            f"# {title}\n\n"
            "This public template mentions a financial account number 4111 1111 1111 1111, "
            "so Phase7 should route it to review-only instead of training. "
        )
    return (
        f"# {title}\n\n"
        "1. The provider supplies services for internal business use only. "
        "2. Confidential information, data processing, payment, renewal, suspension, and "
        "termination terms must be summarized with citations. "
        "3. This text is a public template and contains no party names, signatures, emails, "
        "phone numbers, or private addresses. "
        "4. The assistant must avoid final legal conclusions and require human confirmation. "
    ) * 5


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_phase7_source_manifest_hashes_and_filters_review_only_pii(tmp_path: Path) -> None:
    store = Phase7RealSignalTrainingStore(home=tmp_path, workspace="demo")
    manifest, _texts = store.collect_source_manifest(source_limit=11, fetch_text=_fake_contract_text)

    assert manifest["kind"] == "phase7_real_source_manifest"
    assert manifest["source_count"] == 11
    assert manifest["training_allowed_count"] == 10
    assert manifest["review_only_count"] == 1
    assert manifest["meets_source_goal"] is True
    first = manifest["sources"][0]
    assert first["source_id"] == "cp-csa"
    assert first["route"] == "review_only"
    assert first["content_sha256"]
    assert first["pii_audit"]["severity"] in {"high", "critical"}
    assert any(item["source_id"] == "cp-partnership" for item in phase7_default_sources(limit=11))


def test_phase7_prepare_creates_signal_gated_samples_and_holdout_boundary(tmp_path: Path) -> None:
    prepared = prepare_phase7_real_signal_trial(
        home=tmp_path,
        workspace="demo",
        source_limit=11,
        candidate_limit=24,
        holdout_count=8,
        require_local_model=True,
        model_path=tmp_path / "missing-qwen36",
        fetch_text=_fake_contract_text,
    )
    samples = _read_jsonl(prepared["paths"]["candidate_samples"])
    db_samples = list_samples(home=tmp_path)

    assert prepared["ok"] is True
    assert prepared["source_manifest"]["training_allowed_count"] == 10
    assert prepared["source_ingest"]["ingested_count"] == 10
    assert prepared["candidate_samples"]["count"] > 0
    assert prepared["candidate_samples"]["split_counts"]["test"] == 0
    assert prepared["holdout"]["count"] == 8
    assert prepared["holdout"]["not_for_training"] is True
    assert prepared["preflight"]["model_id"] == PHASE7_RECOMMENDED_MODEL
    assert "local_model_missing" in prepared["preflight"]["blocked_by"]

    signal_types = set(prepared["signal_evidence"]["signal_types"])
    assert {"accept", "reject", "edit", "correction", "preference", "safety_block"}.issubset(signal_types)
    assert prepared["signal_evidence"]["eligible_count"] >= 4
    assert prepared["signal_evidence"]["route_summary"]["training_candidate"]
    assert any(item["signal_type"] == "safety_block" for item in prepared["signal_evidence"]["route_summary"]["excluded"])

    assert all("phase7-holdout" not in sample["sample_id"] for sample in db_samples)
    assert any(sample["metadata"].get("phase") == "phase7" for sample in db_samples)
    for sample in samples[:5]:
        metadata = sample["metadata"]
        assert metadata["phase"] == "phase7"
        assert metadata["trial_id"] == prepared["trial_id"]
        assert metadata["signal_id"]
        assert metadata["source_ids"]
        assert metadata["chunk_ids"]
        assert metadata["provenance"]["source_url"].startswith("https://raw.githubusercontent.com/CommonPaper/")
        assert metadata["not_holdout"] is True
        assert metadata["eligible_for_training"] is True


def test_phase7_finalize_blocks_promotion_without_real_training_and_real_eval(tmp_path: Path) -> None:
    prepare_phase7_real_signal_trial(
        home=tmp_path,
        workspace="demo",
        source_limit=11,
        candidate_limit=12,
        holdout_count=6,
        fetch_text=_fake_contract_text,
    )
    finalized = finalize_phase7_real_signal_trial(
        home=tmp_path,
        workspace="demo",
        training={"real_training": "not_started", "mock_fallback": False},
        generations={"real_model_calls": False},
        real_model_calls=False,
    )

    assert finalized["training_result"]["status"] == "created"
    assert finalized["eval_report"]["eval_gate"]["status"] == "blocked"
    assert finalized["eval_report"]["eval_gate"]["promotion_allowed"] is False
    assert finalized["decision"]["action"] == "archive"
    assert Path(finalized["paths"]["summary"]).is_file()
