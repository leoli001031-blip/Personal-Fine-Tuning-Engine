from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "memory_golden_smoke.py"
SPEC = importlib.util.spec_from_file_location("memory_golden_smoke", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
memory_golden_smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory_golden_smoke)


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "base_model": None,
        "strict": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_resolve_base_model_defaults_to_repo_0_5b_when_present(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path
    model = repo_root / "models" / "Qwen2.5-0.5B-Instruct"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("PFE_GOLDEN_SMOKE_MODEL", raising=False)
    monkeypatch.delenv("PFE_REAL_LOCAL_MODEL", raising=False)

    assert memory_golden_smoke.resolve_base_model(_args(), repo_root) == model.resolve()


def test_resolve_base_model_skips_when_default_model_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PFE_GOLDEN_SMOKE_MODEL", raising=False)
    monkeypatch.delenv("PFE_REAL_LOCAL_MODEL", raising=False)

    assert memory_golden_smoke.resolve_base_model(_args(), tmp_path) is None


def test_resolve_base_model_rejects_4bit_training_path(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "models" / "Qwen2.5-0.5B-Instruct-4bit"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PFE_GOLDEN_SMOKE_MODEL", str(model))

    try:
        memory_golden_smoke.resolve_base_model(_args(), tmp_path)
    except AssertionError as exc:
        assert "unquantized" in str(exc)
        assert "4bit" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("4bit model path should be rejected")


def test_normalize_answer_strips_outer_quotes_and_whitespace() -> None:
    assert memory_golden_smoke.normalize_answer("  “金线闭环-042”  ") == "金线闭环-042"
    assert memory_golden_smoke.normalize_answer("`金线闭环-042`") == "金线闭环-042"


def test_chat_answer_extracts_first_choice_message() -> None:
    assert (
        memory_golden_smoke.chat_answer(
            {"choices": [{"message": {"role": "assistant", "content": "金线闭环-042"}}]}
        )
        == "金线闭环-042"
    )
    assert memory_golden_smoke.chat_answer({"choices": []}) == ""
