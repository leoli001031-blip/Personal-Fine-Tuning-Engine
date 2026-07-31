from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from pfe_core.db.sqlite import list_samples
from pfe_core.phase4_real_corpus import Phase4CorpusStore
from pfe_core.phase8_signal_quality_training import (
    PHASE8_RECOMMENDED_MODEL,
    PHASE8_EXPECTED_SECTIONS,
    Phase8SignalQualityTrainingStore,
    finalize_phase8_signal_quality_trial,
    prepare_phase8_signal_quality_trial,
)


def _fake_contract_text(source: Mapping[str, object]) -> str:
    title = str(source.get("title") or "Contract")
    source_id = str(source.get("source_id") or "")
    if source_id == "cp-csa":
        return (
            f"# {title}\n\n"
            "This public template mentions a financial account number 4111 1111 1111 1111, "
            "so Phase8 should route it to review-only instead of training. "
        )
    return (
        f"# {title}\n\n"
        "1. The provider supplies services for internal business use only and must follow the written order form. "
        "2. Confidential information, personal data processing, training data, payment, renewal, suspension, "
        "termination, service credits, liability limits, and usage restrictions must be summarized with citations. "
        "3. This public template contains no party names, signatures, emails, phone numbers, or private addresses. "
        "4. The assistant must avoid final legal conclusions and require human confirmation where evidence is incomplete. "
        "5. The text supports risk spotting and material organization only, not professional legal advice. "
    ) * 8


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_phase8_prepare_builds_quality_gated_signal_samples(tmp_path: Path) -> None:
    prepared = prepare_phase8_signal_quality_trial(
        home=tmp_path,
        workspace="demo",
        source_limit=11,
        signal_count=36,
        candidate_limit=34,
        holdout_count=8,
        require_local_model=True,
        model_path=tmp_path / "missing-qwen",
        fetch_text=_fake_contract_text,
    )
    samples = _read_jsonl(prepared["paths"]["candidate_samples"])
    db_samples = list_samples(home=tmp_path)
    quality = prepared["quality_report"]

    assert prepared["ok"] is True
    assert prepared["source_manifest"]["training_allowed_count"] == 10
    assert prepared["signal_dataset"]["quality_signal_count"] == 36
    assert prepared["signal_dataset"]["eligible_count"] == 36
    assert prepared["candidate_samples"]["count"] >= 30
    assert prepared["candidate_samples"]["split_counts"]["test"] == 0
    assert quality["passed_signal_count"] >= 30
    assert quality["candidate_passed_count"] >= 30
    assert quality["meets_quality_goal"] is True
    assert quality["rejection_reasons"]["reject_excluded"] == 1
    assert quality["rejection_reasons"]["safety_block_excluded"] == 1
    assert prepared["preflight"]["model_id"] == PHASE8_RECOMMENDED_MODEL
    assert "local_model_missing" in prepared["preflight"]["blocked_by"]

    holdout_chunk_ids = set(prepared["quality_report"]["holdout_chunk_ids"])
    assert holdout_chunk_ids
    assert all("phase8-holdout" not in sample["sample_id"] for sample in db_samples)
    assert any(sample["metadata"].get("phase") == "phase8" for sample in db_samples)
    for sample in samples[:10]:
        metadata = sample["metadata"]
        chosen = str(sample["chosen"])
        assert metadata["phase"] == "phase8"
        assert metadata["trial_id"] == prepared["trial_id"]
        assert metadata["quality_gate_passed"] is True
        assert metadata["not_holdout"] is True
        assert not (set(metadata["chunk_ids"]) & holdout_chunk_ids)
        assert metadata["expected_citation"] in chosen
        for section in PHASE8_EXPECTED_SECTIONS:
            assert section in chosen


def test_phase8_quality_gate_rejects_holdout_contamination_and_missing_citation(tmp_path: Path) -> None:
    prepared = prepare_phase8_signal_quality_trial(
        home=tmp_path,
        workspace="demo",
        source_limit=11,
        signal_count=30,
        candidate_limit=30,
        holdout_count=6,
        fetch_text=_fake_contract_text,
    )
    store = Phase8SignalQualityTrainingStore(home=tmp_path, workspace="demo")
    phase4_store = Phase4CorpusStore(home=tmp_path, workspace="demo")
    holdout = store.read_holdouts()[0]
    chunk_lookup = store._chunk_lookup(phase4_store=phase4_store)  # noqa: SLF001 - targeted quality gate unit test.
    source_id = str(holdout["source_id"])
    chunk_id = str(holdout["chunk_id"])
    bad_signal = {
        "signal_id": "bad-holdout",
        "signal_type": "correction",
        "signal_strength": "strong_correction",
        "source_id": source_id,
        "chunk_id": chunk_id,
        "expected_citation": f"[{source_id}:wrong-chunk]",
        "target_output": "摘要：太短。\n风险提示：太短。\n引用依据：[missing]\n人工确认：人工确认。",
        "eligible_for_training": True,
        "quality_score": 0.9,
    }
    check = store._evaluate_signal(  # noqa: SLF001 - direct regression for rejection reasons.
        signal=bad_signal,
        chunk_lookup=chunk_lookup,
        holdout_chunk_ids=set(prepared["quality_report"]["holdout_chunk_ids"]),
        seen_hashes=set(),
    )

    assert check["passed"] is False
    assert "holdout_contamination" in check["reasons"]
    assert "citation_does_not_match_source_chunk" in check["reasons"]
    assert "target_missing_expected_citation" in check["reasons"]
    assert "low_information_target" in check["reasons"]


def test_phase8_quality_gate_requires_complete_preference_pair(tmp_path: Path) -> None:
    prepare_phase8_signal_quality_trial(
        home=tmp_path,
        workspace="demo",
        source_limit=11,
        signal_count=30,
        candidate_limit=30,
        holdout_count=6,
        fetch_text=_fake_contract_text,
    )
    store = Phase8SignalQualityTrainingStore(home=tmp_path, workspace="demo")
    phase4_store = Phase4CorpusStore(home=tmp_path, workspace="demo")
    chunk = phase4_store.list_chunks(limit=1)[0]
    source_id = str(chunk["source_id"])
    chunk_id = str(chunk["chunk_id"])
    citation = f"[{source_id}:{chunk_id}]"
    target = store._target_output(  # noqa: SLF001 - targeted preference gate unit test.
        chunk=chunk,
        citation=citation,
        focus="终止条款",
        risk_boundary="只做资料整理。",
    )
    signal = {
        "signal_id": "bad-preference",
        "signal_type": "preference",
        "signal_strength": "preference_pair",
        "source_id": source_id,
        "chunk_id": chunk_id,
        "expected_citation": citation,
        "target_output": target,
        "chosen": target,
        "rejected": "",
        "eligible_for_training": True,
        "quality_score": 0.9,
    }
    check = store._evaluate_signal(  # noqa: SLF001 - direct regression for preference pair completeness.
        signal=signal,
        chunk_lookup=store._chunk_lookup(phase4_store=phase4_store),  # noqa: SLF001
        holdout_chunk_ids=set(store._read_json(store.quality_report_path)["holdout_chunk_ids"]),  # noqa: SLF001
        seen_hashes=set(),
    )

    assert check["passed"] is False
    assert "preference_pair_incomplete" in check["reasons"]


def test_phase8_eval_decision_recommends_manual_review_without_auto_promote(tmp_path: Path) -> None:
    prepare_phase8_signal_quality_trial(
        home=tmp_path,
        workspace="demo",
        source_limit=11,
        signal_count=30,
        candidate_limit=30,
        holdout_count=6,
        fetch_text=_fake_contract_text,
    )
    store = Phase8SignalQualityTrainingStore(home=tmp_path, workspace="demo")
    details = []
    for holdout in store.read_holdouts():
        citation = str(holdout["expected_citation"])
        details.append(
            {
                "prompt_id": holdout["prompt_id"],
                "base_output": "This looks risky. Ask a lawyer.",
                "adapter_output": (
                    "摘要：仅基于给定片段整理条款内容。\n"
                    "风险提示：关注资料中出现的义务、限制和证据不足，不判断合法/违法，不输出法律结论。\n"
                    f"引用依据：{citation}\n"
                    "人工确认：适用法、真实交易背景和最终法律判断必须人工确认。"
                ),
            }
        )
    finalized = finalize_phase8_signal_quality_trial(
        home=tmp_path,
        workspace="demo",
        training={"real_training": "completed", "mock_fallback": False, "adapter_path": str(tmp_path / "adapter")},
        generations={"real_model_calls": True, "details": details},
        real_model_calls=True,
    )

    assert finalized["eval_report"]["eval_gate"]["status"] == "pass"
    assert finalized["eval_report"]["recommendation"] == "promote_after_manual_review"
    assert finalized["eval_report"]["eval_gate"]["promotion_allowed"] is False
    assert finalized["eval_report"]["eval_gate"]["auto_promotion_allowed"] is False
    assert finalized["decision"]["action"] == "promote_after_manual_review"
    assert finalized["decision"]["promotion_allowed"] is False
    assert finalized["decision"]["manual_review_required"] is True


def test_phase8_finalize_archives_without_real_training_and_eval(tmp_path: Path) -> None:
    prepare_phase8_signal_quality_trial(
        home=tmp_path,
        workspace="demo",
        source_limit=11,
        signal_count=30,
        candidate_limit=30,
        holdout_count=6,
        fetch_text=_fake_contract_text,
    )
    finalized = finalize_phase8_signal_quality_trial(
        home=tmp_path,
        workspace="demo",
        training={"real_training": "not_started", "mock_fallback": False},
        generations={"real_model_calls": False},
        real_model_calls=False,
    )

    assert finalized["training_result"]["status"] == "created"
    assert finalized["eval_report"]["eval_gate"]["status"] == "blocked"
    assert finalized["eval_report"]["recommendation"] == "archive"
    assert finalized["decision"]["action"] == "archive"
    assert Path(finalized["paths"]["summary"]).is_file()
