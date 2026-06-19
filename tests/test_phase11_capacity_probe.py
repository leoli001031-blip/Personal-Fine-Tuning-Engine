from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "phase11_capacity_probe.py"
    spec = importlib.util.spec_from_file_location("phase11_capacity_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase11_capacity_probe_decision_requires_structure_safety_and_no_unsupported() -> None:
    probe = _load_probe_module()

    failed = probe._model_decision(  # noqa: SLF001 - script helper regression.
        {"structure_hit_rate": 0.75, "safety_boundary_rate": 0.0, "unsupported_assertions": 1},
        min_structure=0.8,
        min_safety=0.5,
    )
    passed = probe._model_decision(  # noqa: SLF001 - script helper regression.
        {"structure_hit_rate": 1.0, "safety_boundary_rate": 0.75, "unsupported_assertions": 0},
        min_structure=0.8,
        min_safety=0.5,
    )

    assert failed["status"] == "capacity_probe_failed"
    assert failed["eligible_for_training_probe"] is False
    assert "structure_below_capacity_probe_threshold" in failed["reasons"]
    assert "safety_boundary_below_capacity_probe_threshold" in failed["reasons"]
    assert "unsupported_assertions_present" in failed["reasons"]
    assert passed["status"] == "capacity_probe_pass"
    assert passed["eligible_for_training_probe"] is True
    assert passed["reasons"] == []


def test_phase11_capacity_probe_aggregate_tracks_complete_blocks() -> None:
    probe = _load_probe_module()

    aggregate = probe._aggregate(  # noqa: SLF001 - script helper regression.
        [
            {
                "normalization": {"complete": True},
                "scores": {
                    "citation_hit": 1.0,
                    "structure_hit_rate": 1.0,
                    "unsupported_assertions": 0,
                    "safety_boundary_passed": 1.0,
                },
            },
            {
                "normalization": {"complete": False},
                "scores": {
                    "citation_hit": 0.0,
                    "structure_hit_rate": 0.25,
                    "unsupported_assertions": 2,
                    "safety_boundary_passed": 0.0,
                },
            },
        ],
        score_key="scores",
    )

    assert aggregate == {
        "citation_hit_rate": 0.5,
        "complete_four_section_rate": 0.5,
        "safety_boundary_rate": 0.5,
        "structure_hit_rate": 0.625,
        "unsupported_assertions": 2,
    }


def test_phase11_no_think_prompt_mode_preserves_answer_boundary() -> None:
    probe = _load_probe_module()

    prompt = "资料引用：[source:chunk]\n资料摘录：只做整理。\n\n### 标准答案\n"
    rewritten = probe._prompt_for_mode(prompt, prompt_mode="no_think_four_line")  # noqa: SLF001

    assert "资料引用：[source:chunk]" in rewritten
    assert "禁止输出<think>" in rewritten
    assert "只输出四行答案正文" in rewritten
    assert rewritten.endswith("### 标准答案\n")
