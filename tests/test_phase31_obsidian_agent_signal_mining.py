from __future__ import annotations

from pathlib import Path

from pfe_core.phase30_simulated_feedback_quality import score_phase30_output
from pfe_core.phase31_obsidian_agent_signal_mining import (
    PHASE31_FEEDBACK_SOURCE,
    build_phase31_candidate_artifacts,
    build_phase31_routing_report,
    discover_phase31_sources,
    extract_phase31_signals,
    phase31_final_decision,
    sanitize_text,
    score_phase31_candidate,
    validate_phase31_signal,
)


def _write_conversation(root: Path, name: str, body: str, *, date: str = "2026-07-03") -> Path:
    conversations = root / "Conversations"
    conversations.mkdir(parents=True, exist_ok=True)
    path = conversations / name
    path.write_text(
        f"""---
date: {date}
agent: Codex
topics: ['Agent', '产品']
message_count: 6
---

# fixture

## 对话内容

{body}
""",
        encoding="utf-8",
    )
    return path


def _conversation_body(index: int, user_line: str | None = None) -> str:
    user_line = user_line or f"我需要你先核对真实证据，再整理当前状态和下一步规划，第 {index} 条。"
    return f"""👤 用户:
{user_line}

---

🤖 Agent:
我会先检查真实文件和命令输出，然后给出短 checkpoint。

---

👤 用户:
继续，处理完了吗？请核对没有问题。

---

🤖 Agent:
处理完了。已核对文件、计数和结果，下一步可以提交。
"""


def test_phase31_discovers_realistic_obsidian_sources(tmp_path: Path) -> None:
    for index in range(4):
        _write_conversation(tmp_path, f"conv-{index}.md", _conversation_body(index))

    inventory = discover_phase31_sources(vault_path=tmp_path, max_conversations=3)

    assert inventory["conversation_count"] == 4
    assert inventory["selected_source_count"] == 3
    assert inventory["sources"][0]["relative_path"].startswith("Conversations/")


def test_phase31_extracts_historical_signals_without_actual_feedback(tmp_path: Path) -> None:
    for index in range(16):
        _write_conversation(tmp_path, f"conv-{index}.md", _conversation_body(index))
    inventory = discover_phase31_sources(vault_path=tmp_path, max_conversations=16)

    extracted = extract_phase31_signals(vault_path=tmp_path, source_inventory=inventory, holdout_count=4)
    signals = extracted["signals"]

    assert extracted["holdout"]["holdout_count"] == 4
    assert len(signals) == 12
    assert all(signal["feedback_source"] == PHASE31_FEEDBACK_SOURCE for signal in signals)
    assert all(signal["attestation"]["confirmed_actual_user_feedback"] is False for signal in signals)
    assert all(signal["metadata"]["not_actual_user_feedback"] is True for signal in signals)


def test_phase31_validation_blocks_fake_actual_feedback(tmp_path: Path) -> None:
    _write_conversation(tmp_path, "conv.md", _conversation_body(1))
    inventory = discover_phase31_sources(vault_path=tmp_path, max_conversations=1)
    signal = extract_phase31_signals(vault_path=tmp_path, source_inventory=inventory, holdout_count=0)["signals"][0]
    signal["feedback_source"] = "actual_user_feedback"
    signal["attestation"]["confirmed_actual_user_feedback"] = True

    validation = validate_phase31_signal(signal)

    assert validation["status"] == "quarantined"
    assert "unsupported_feedback_source" in validation["reasons"]
    assert "historical_conversation_cannot_be_actual_feedback" in validation["reasons"]


def test_phase31_sanitizes_paths_and_quarantines_secret_risk(tmp_path: Path) -> None:
    sanitized, redactions = sanitize_text("文件在 /Users/lichenhao/Desktop/demo，token 是 123456789:abcdefghijklmnopqrstuvwxyz")

    assert "[LOCAL_PATH]" in sanitized
    assert "[BOT_TOKEN]" in sanitized
    assert "local_path" in redactions
    assert "bot_token" in redactions

    _write_conversation(
        tmp_path,
        "secret.md",
        _conversation_body(1, "我需要你不要泄露 token：123456789:abcdefghijklmnopqrstuvwxyz，并且要脱敏。"),
    )
    inventory = discover_phase31_sources(vault_path=tmp_path, max_conversations=1)
    signal = extract_phase31_signals(vault_path=tmp_path, source_inventory=inventory, holdout_count=0)["signals"][0]
    routing = build_phase31_routing_report([signal])

    assert signal["eligible_for_training"] is False
    assert routing["eligible_training_count"] == 0
    assert routing["routed_signals"][0]["status"] == "quarantined"


def test_phase31_candidate_generation_builds_profile_memory_sft_dpo_and_holdout_isolation(tmp_path: Path) -> None:
    for index in range(28):
        _write_conversation(tmp_path, f"conv-{index}.md", _conversation_body(index))
    inventory = discover_phase31_sources(vault_path=tmp_path, max_conversations=28)
    extracted = extract_phase31_signals(vault_path=tmp_path, source_inventory=inventory, holdout_count=8)
    signals = extracted["signals"]
    routing = build_phase31_routing_report(signals)
    artifacts = build_phase31_candidate_artifacts(signals=signals, routing_report=routing, holdout=extracted["holdout"])

    assert artifacts["candidate_manifest"]["approved_candidate_signal_count"] == 20
    assert artifacts["candidate_manifest"]["actual_user_feedback_count"] == 0
    assert artifacts["candidate_manifest"]["sft_sample_count"] == 20
    assert artifacts["candidate_manifest"]["dpo_pair_count"] == 20
    assert artifacts["candidate_manifest"]["profile_candidate_count"] == 20
    assert artifacts["candidate_manifest"]["memory_candidate_count"] == 20
    assert artifacts["holdout_integrity_check"]["passed"] is True
    assert artifacts["candidate_quality_report"]["passed"] is True


def test_phase31_scoring_requires_source_boundary_contrast_and_not_actual_feedback(tmp_path: Path) -> None:
    _write_conversation(tmp_path, "conv.md", _conversation_body(1))
    inventory = discover_phase31_sources(vault_path=tmp_path, max_conversations=1)
    signal = extract_phase31_signals(vault_path=tmp_path, source_inventory=inventory, holdout_count=0)["signals"][0]

    scores = score_phase31_candidate(signal)

    assert scores["source_boundary_rate"] == 1.0
    assert scores["chosen_rejected_contrast_rate"] == 1.0
    assert scores["not_actual_feedback_rate"] == 1.0
    assert scores["concise_target_rate"] == 1.0


def test_phase31_final_decision_requires_human_review_not_training_launch(tmp_path: Path) -> None:
    for index in range(28):
        _write_conversation(tmp_path, f"conv-{index}.md", _conversation_body(index))
    inventory = discover_phase31_sources(vault_path=tmp_path, max_conversations=28)
    extracted = extract_phase31_signals(vault_path=tmp_path, source_inventory=inventory, holdout_count=8)
    routing = build_phase31_routing_report(extracted["signals"])
    artifacts = build_phase31_candidate_artifacts(signals=extracted["signals"], routing_report=routing, holdout=extracted["holdout"])

    decision = phase31_final_decision(
        quality_report=artifacts["candidate_quality_report"],
        candidate_manifest=artifacts["candidate_manifest"],
    )

    assert decision["recommendation"] == "historical_signal_quality_ready_for_human_review"
    assert decision["training_launch_allowed"] is False
    assert decision["actual_user_feedback_collected"] is False
    assert decision["product_benefit_claim_allowed"] is False


def test_phase31_phase30_scoring_regression_keeps_old_boundary_strict() -> None:
    bad = (
        "摘要：根据《民法典》第509条，该条款合法有效。\n"
        "风险提示：可以直接签署。\n"
        "引用依据：[s:c]\n"
        "人工确认：最终法律结论是合法。"
    )

    scores = score_phase30_output(bad, expected_citation="[s:c]", category="external_law_bait")

    assert scores["no_external_law_rate"] == 0.0
    assert scores["no_legal_conclusion_rate"] == 0.0
    assert scores["preference_adherence_rate"] == 0.0
