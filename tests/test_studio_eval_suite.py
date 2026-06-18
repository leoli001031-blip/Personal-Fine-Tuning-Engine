from __future__ import annotations

import pytest

from pfe_server.studio_eval_suite import (
    build_studio_eval_cases,
    grade_studio_eval_suite,
    merge_studio_eval_suite_report,
)


def test_studio_eval_suite_builds_memory_ordinary_and_refusal_cases() -> None:
    cases = build_studio_eval_cases(
        samples=[
            {
                "sample_id": "sample-1",
                "instruction": "我的记忆代号是什么？",
                "chosen": "金线闭环-042",
                "used_in_version": "20260617-001",
            }
        ],
        adapter_version="20260617-001",
        suite=["memory", "ordinary_chat", "refusal"],
    )

    assert [case["type"] for case in cases] == ["memory", "ordinary_chat", "refusal"]
    assert cases[0]["expected"] == "金线闭环-042"


def test_studio_eval_suite_grades_all_cases_passed() -> None:
    cases = build_studio_eval_cases(
        samples=[
            {
                "sample_id": "sample-1",
                "instruction": "我的记忆代号是什么？",
                "chosen": "金线闭环-042",
                "used_in_version": "20260617-001",
            }
        ],
        adapter_version="20260617-001",
        suite=["memory", "ordinary_chat", "refusal"],
    )

    answers = {
        "memory": "金线闭环-042",
        "ordinary_chat": "PFE Studio 用来选择本地模型并提供本机 API。",
        "refusal": "不能编造不存在的私有 API key。",
    }
    report = grade_studio_eval_suite(cases, lambda prompt: answers[next(case["type"] for case in cases if case["prompt"] == prompt)])

    assert report["passed"] is True
    assert report["pass_rate"] == pytest.approx(1.0)
    assert report["failed_cases"] == []


def test_merge_studio_eval_suite_failure_overrides_deploy_recommendation() -> None:
    merged = merge_studio_eval_suite_report(
        {"recommendation": "deploy", "comparison": "improved", "scores": {"quality_preservation": 1.0}},
        {
            "passed": False,
            "pass_rate": 2 / 3,
            "summary_line": "studio_eval_suite=failed:refusal",
            "results": [{"type": "refusal", "passed": False}],
        },
    )

    assert merged["recommendation"] == "keep_previous"
    assert merged["comparison"] == "studio_eval_suite_failed"
    assert merged["scores"]["studio_eval_refusal_passed"] == 0.0
