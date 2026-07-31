from __future__ import annotations

from pathlib import Path

from pfe_core.db.sqlite import list_samples
from pfe_core.phase4_real_corpus import (
    DEFAULT_RESEARCH_PERSONA,
    DEFAULT_RESEARCH_SCENARIO,
    CorpusChunk,
    CorpusSource,
    Phase4CorpusStore,
)


def _write_source(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_phase4_persona_scenario_and_source_schema_validate(tmp_path: Path) -> None:
    store = Phase4CorpusStore(home=tmp_path, workspace="demo")
    personas = store.personas()
    scenarios = store.scenarios()

    assert personas[0]["persona_id"] == DEFAULT_RESEARCH_PERSONA.persona_id
    assert scenarios[0]["scenario_id"] == DEFAULT_RESEARCH_SCENARIO.scenario_id
    assert "无来源结论" in scenarios[0]["task"]

    source = CorpusSource(
        source_id="src-test",
        title="Research note",
        source_path="/tmp/research.md",
        source_type="md",
        content_hash="hash",
        license_status="local_user_provided",
    )
    chunk = CorpusChunk(
        chunk_id="src-test-chunk-001",
        source_id="src-test",
        text="This is a research note with enough context to be chunked and cited.",
        char_count=64,
        token_count=12,
        provenance={"source_id": "src-test", "chunk_index": 0},
    )

    assert CorpusSource.from_dict(source.to_dict()).source_id == "src-test"
    assert CorpusChunk.from_dict(chunk.to_dict()).provenance["source_id"] == "src-test"


def test_corpus_ingestion_chunks_and_preserves_provenance(tmp_path: Path) -> None:
    store = Phase4CorpusStore(home=tmp_path, workspace="demo")
    path = _write_source(
        tmp_path / "notes.md",
        (
            "# Research notes\n\n"
            "PFE Phase4 collects interview notes, product docs, and field observations. "
            "The system must preserve source identifiers and chunk identifiers so later "
            "training samples can be audited. The assistant should mark open questions "
            "when material is incomplete."
        ),
    )

    result = store.ingest_path(path, title="Phase4 notes", license_status="local_user_provided")

    assert result["source"]["source_type"] == "md"
    assert result["source"]["content_hash"]
    assert result["chunk_count"] >= 1
    chunk = result["chunks"][0]
    assert chunk["source_id"] == result["source"]["source_id"]
    assert chunk["provenance"]["source_path"] == str(path)
    assert chunk["char_count"] >= 60


def test_candidate_generation_keeps_provenance_and_safety_metadata(tmp_path: Path) -> None:
    store = Phase4CorpusStore(home=tmp_path, workspace="demo")
    path = _write_source(
        tmp_path / "research.txt",
        (
            "The research team observed that users wanted shorter summaries, explicit "
            "source citations, and a list of open questions. The material does not "
            "support a final decision yet, so the assistant should ask for more data."
        ),
    )
    store.ingest_path(path, title="Research observations")

    result = store.generate_training_candidates(limit=4)

    assert result["count"] >= 4
    assert result["eligible_count"] >= 4
    sample = result["candidates"][0]
    assert sample["source_ids"]
    assert sample["chunk_ids"]
    assert sample["provenance"]["source_path"] == str(path)
    assert sample["safety_metadata"]["provenance_complete"] is True
    assert sample["eligible_for_training"] is True
    assert "需补充资料" in result["candidates"][-1]["output"]


def test_pii_high_risk_and_low_quality_candidates_are_excluded(tmp_path: Path) -> None:
    store = Phase4CorpusStore(home=tmp_path, workspace="demo")
    pii_path = _write_source(
        tmp_path / "pii.md",
        (
            "User interview includes contact data: buyer@example.com and phone 13800000000. "
            "This material should not become training data even if it also mentions "
            "research summaries and citations."
        ),
    )
    legal_path = _write_source(
        tmp_path / "legal.md",
        (
            "合同资料显示乙方需要在七日内交付，逾期条款需要人工确认。"
            "该资料只能用于摘要、风险提示和待确认问题整理，不能提供法律结论。"
        ),
    )
    store.ingest_path(pii_path, title="PII interview")
    store.ingest_path(legal_path, title="Contract research note")

    result = store.generate_training_candidates(limit=8)
    excluded = [item for item in result["candidates"] if not item["eligible_for_training"]]
    eligible = [item for item in result["candidates"] if item["eligible_for_training"]]

    assert excluded
    assert any(item["excluded_reason"] == "pii_audit_blocked" for item in excluded)
    assert eligible
    assert any("legal" in item["safety_metadata"]["high_risk_domains"] for item in eligible)
    assert all("法律结论" in item["output"] or "依据" in item["output"] for item in eligible)


def test_candidate_export_writes_jsonl_and_samples_db(tmp_path: Path) -> None:
    store = Phase4CorpusStore(home=tmp_path, workspace="demo")
    path = _write_source(
        tmp_path / "notes.txt",
        (
            "Research notes say users compare base and local responses by checking citation "
            "coverage, summary quality, and unsupported assertions. Holdout prompts should "
            "use the same source material for base and local runs."
        ),
    )
    store.ingest_path(path, title="Eval notes")
    store.generate_training_candidates(limit=5)

    export = store.export_training_candidates(format="jsonl")
    sample_export = store.export_to_training_samples()
    train_samples = list_samples(home=tmp_path, dataset_split="train")

    assert Path(export["path"]).exists()
    assert sample_export["saved_samples"] >= 1
    assert sample_export["split_counts"]["train"] >= 1
    assert train_samples
    assert train_samples[0]["source"] == "signal"
    assert train_samples[0]["metadata"]["phase"] == "phase4"


def test_phase4_eval_report_compares_base_and_local(tmp_path: Path) -> None:
    store = Phase4CorpusStore(home=tmp_path, workspace="demo")
    path = _write_source(
        tmp_path / "eval.txt",
        (
            "The source states that Phase4 should preserve citations, reduce unsupported "
            "assertions, and ask for human confirmation when the corpus is insufficient. "
            "It does not support legal, medical, or financial conclusions."
        ),
    )
    store.ingest_path(path, title="Phase4 eval source")
    store.generate_training_candidates(limit=5)

    report = store.build_eval_report(adapter_version="phase4-test-adapter")

    assert report["kind"] == "phase4_eval_report"
    assert report["adapter_version"] == "phase4-test-adapter"
    assert report["holdout_count"] >= 1
    assert report["scores"]["local_delta"]["citation_hit_rate"] > 0
    assert report["scores"]["unsupported_assertions"] <= report["base_metrics"]["unsupported_assertions"]
    assert report["eval_gate"]["status"] in {"pass", "review"}
    assert report["details"][0]["expected_citation"] in report["details"][0]["local_output"]
