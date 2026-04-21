"""Tests for teacher signal fusion logic."""

from __future__ import annotations

import pytest

from pfe_core.curator.teacher_fusion import TeacherSignalFusion, TeacherSignalFusionConfig


def _make_sample(sample_id: str, source: str, instruction: str, chosen: str, rejected: str | None = None, reply_style: str | None = None) -> dict:
    metadata = {"reply_style": reply_style} if reply_style else {}
    return {
        "sample_id": sample_id,
        "source": source,
        "instruction": instruction,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": metadata,
    }


class TestTeacherSignalFusion:
    def test_user_priority_over_teacher(self):
        user = [_make_sample("u1", "signal", "hello", "user chosen")]
        teacher = [_make_sample("t1", "teacher", "hello", "teacher chosen")]
        fuser = TeacherSignalFusion()
        fused = fuser.fuse_signals(user, teacher)
        assert len(fused) == 1
        assert fused[0]["source"] == "signal"

    def test_teacher_supplements_uncovered_domain(self):
        user = [_make_sample("u1", "signal", "prompt A", "chosen A")]
        teacher = [
            _make_sample("t1", "teacher", "prompt B", "chosen B"),
            _make_sample("t2", "teacher", "prompt C", "chosen C"),
        ]
        # Use a high ratio so the cap does not interfere with this domain-coverage test.
        fuser = TeacherSignalFusion(config=TeacherSignalFusionConfig(max_teacher_ratio=0.8))
        fused = fuser.fuse_signals(user, teacher)
        assert len(fused) == 3
        sources = [s["source"] for s in fused]
        assert sources.count("teacher") == 2

    def test_teacher_ratio_cap(self):
        user = [_make_sample(f"u{i}", "signal", f"prompt {i}", f"chosen {i}") for i in range(10)]
        teacher = [_make_sample(f"t{i}", "teacher", f"tprompt {i}", f"tchosen {i}") for i in range(10)]
        fuser = TeacherSignalFusion(config=TeacherSignalFusionConfig(max_teacher_ratio=0.2))
        fused = fuser.fuse_signals(user, teacher)
        plan = fuser.get_last_plan()
        assert plan["teacher_ratio"] <= 0.21  # allow small rounding

    def test_user_edit_wins_over_teacher_same_prompt(self):
        user = [_make_sample("u1", "signal", "same prompt", "user edit", reply_style="edited")]
        teacher = [_make_sample("t1", "teacher", "same prompt", "teacher suggestion")]
        fuser = TeacherSignalFusion()
        fused = fuser.fuse_signals(user, teacher)
        assert len(fused) == 1
        assert fused[0]["source"] == "signal"

    def test_empty_user_signals_allows_teacher(self):
        teacher = [_make_sample("t1", "teacher", "prompt", "chosen")]
        fuser = TeacherSignalFusion()
        fused = fuser.fuse_signals([], teacher)
        assert len(fused) == 1
        assert fused[0]["source"] == "teacher"

    def test_fusion_plan_metadata(self):
        user = [_make_sample("u1", "signal", "p", "c")]
        teacher = [_make_sample("t1", "teacher", "p2", "c2")]
        fuser = TeacherSignalFusion()
        fused = fuser.fuse_signals(user, teacher)
        plan = fuser.get_last_plan()
        assert plan["user_count"] == 1
        assert plan["teacher_selected"] == 1
        assert plan["total_fused"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
