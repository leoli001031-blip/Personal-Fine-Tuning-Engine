from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from pfe_core.db.sqlite import list_samples
from pfe_core.phase4_real_corpus import Phase4CorpusStore
from pfe_core.phase5_real_domain_loop import (
    COMMON_PAPER_CONTRACT_SOURCES,
    COMMON_PAPER_LICENSE_NOTE,
    Phase5RealDomainLoopStore,
    run_phase5_domain_loop,
)


def _fake_contract_text(source: Mapping[str, object]) -> str:
    title = str(source["title"])
    return (
        f"# {title}\n\n"
        "1. Access and use. Customer may use the service during the subscription period "
        "for internal business purposes, subject to the agreement and use limitations. "
        "2. Data use. Provider may process customer content only as needed to provide, "
        "maintain, and improve the product, and personal data requires a data processing "
        "agreement. 3. Payment. Fees are non-refundable except where the agreement states "
        "otherwise, and disputed amounts require timely notice. 4. Termination. Either "
        "party may terminate for uncured material breach after notice and cure period. "
        "5. Human review. This sample is for contract summarization and risk flagging; "
        "it does not provide a legal conclusion and needs professional confirmation. "
    ) * 3


def test_phase5_curated_sources_keep_training_license_metadata(tmp_path: Path) -> None:
    store = Phase5RealDomainLoopStore(home=tmp_path, workspace="demo")
    manifest = store.write_source_manifest()

    assert manifest["source_count"] == len(COMMON_PAPER_CONTRACT_SOURCES)
    assert manifest["training_allowed_count"] == manifest["source_count"]
    assert manifest["license_note"] == COMMON_PAPER_LICENSE_NOTE
    first = manifest["sources"][0]
    assert first["source_url"].startswith("https://raw.githubusercontent.com/CommonPaper/")
    assert first["license_status"] == "cc_by_4_0_training_allowed"
    assert first["training_allowed"] is True
    assert "risk_labels" in first


def test_phase5_ingest_preserves_real_source_metadata_and_phase4_provenance(tmp_path: Path) -> None:
    phase4 = Phase4CorpusStore(home=tmp_path, workspace="demo")
    phase5 = Phase5RealDomainLoopStore(home=tmp_path, workspace="demo")

    ingest = phase5.ingest_sources(phase4, limit=3, fetch_text=_fake_contract_text)
    candidates = phase4.generate_training_candidates(limit=12)

    assert ingest["ingested_count"] == 3
    source = phase4.list_sources(limit=1)[0]
    assert source["source_type"] == "url"
    assert source["source_url"].startswith("https://raw.githubusercontent.com/CommonPaper/")
    assert source["metadata"]["license_note"] == COMMON_PAPER_LICENSE_NOTE
    assert source["metadata"]["training_allowed"] is True
    chunk = phase4.list_chunks(limit=1)[0]
    assert chunk["provenance"]["source_url"].startswith("https://raw.githubusercontent.com/CommonPaper/")
    assert candidates["eligible_count"] >= 8
    assert candidates["candidates"][0]["provenance"]["source_url"].startswith("https://raw.githubusercontent.com/CommonPaper/")


def test_phase5_holdout_prompts_stay_out_of_training_samples(tmp_path: Path) -> None:
    result = run_phase5_domain_loop(
        home=tmp_path,
        workspace="demo",
        source_limit=4,
        candidate_limit=20,
        holdout_count=12,
        fetch_text=_fake_contract_text,
    )
    samples = list_samples(home=tmp_path)
    holdouts = json.loads(Path(result["holdout_path"]).read_text(encoding="utf-8"))

    assert result["holdout_count"] == 12
    assert samples
    assert all("phase5-holdout" not in sample["sample_id"] for sample in samples)
    assert all(sample["metadata"].get("phase") == "phase4" for sample in samples)
    assert all(item["metadata"]["not_for_training"] is True for item in holdouts)
    assert {item["safety_case"] for item in holdouts} >= {"insufficient_evidence", "legal_conclusion_boundary"}


def test_phase5_eval_report_and_loop_evidence_schema(tmp_path: Path) -> None:
    result = run_phase5_domain_loop(
        home=tmp_path,
        workspace="demo",
        source_limit=4,
        candidate_limit=20,
        holdout_count=12,
        fetch_text=_fake_contract_text,
    )
    report = json.loads(Path(result["eval_report_path"]).read_text(encoding="utf-8"))
    loop = json.loads(Path(result["loop_evidence_path"]).read_text(encoding="utf-8"))

    assert report["kind"] == "phase5_real_domain_eval_report"
    assert report["holdout_count"] == 12
    assert report["scores"]["citation_hit_rate"] >= 0.85
    assert report["scores"]["structure_hit_rate"] >= 0.85
    assert report["scores"]["safety_boundary_rate"] >= 0.85
    assert report["eval_gate"]["status"] in {"pass", "review"}
    assert loop["kind"] == "phase5_loop_evidence"
    assert loop["route_summary"]["memory"]
    assert loop["route_summary"]["profile"]
    assert loop["route_summary"]["training_candidate"]
    assert any(item["reason"] == "safety_block" for item in loop["route_summary"]["excluded"])
