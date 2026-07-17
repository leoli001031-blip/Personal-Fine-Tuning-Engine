from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tools/phase96_98_qwen3_4b_capacity_ladder.py"
SPEC = importlib.util.spec_from_file_location("phase96_driver", DRIVER)
assert SPEC and SPEC.loader
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_phase96_uses_only_two_local_model_variants() -> None:
    assert driver.MODEL_VARIANTS == {
        "qwen25_1_5b": ROOT / "models/Qwen2.5-1.5B-Instruct",
        "qwen3_4b": ROOT / "models/Qwen3-4B",
    }
    assert all(path.is_absolute() for path in driver.MODEL_VARIANTS.values())


def test_phase96_call_budget_is_exactly_48() -> None:
    holdout = driver.build_phase96_capacity_holdout()
    assert holdout["session_count"] * 3 * len(driver.MODEL_VARIANTS) == 48


def test_phase96_private_cache_stays_outside_repo() -> None:
    assert str(driver.PRIVATE_REVIEW_ROOT).startswith("/private/tmp/")
    assert driver.REPO_ROOT not in driver.PRIVATE_REVIEW_ROOT.parents
