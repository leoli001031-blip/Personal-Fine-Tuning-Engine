from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "tools/phase91_controlled_dpo_preference_diagnostic.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("phase91_driver", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase91_job_spec_freezes_parent_lineage_and_exact_steps() -> None:
    driver = _load_driver()
    rows = [{
        "sample_id": "p1",
        "instruction": "prompt",
        "chosen": "chosen",
        "rejected": "rejected",
        "sample_type": "dpo",
    }]

    spec = driver._job_spec(rows, ROOT / "trainer_job_outputs/test-phase91", 12)

    training = spec["recipe"]["training"]
    assert training["max_steps"] == 12
    assert training["incremental_context"]["parent_adapter_path"] == str(
        driver.PARENT_ADAPTER_ROOT
    )
    assert spec["phase91"]["parent_merge_required"] is True
    assert spec["phase91"]["automatic_promotion_allowed"] is False


def test_phase91_sanity_requires_strict_core_improvement() -> None:
    driver = _load_driver()
    phase89 = {
        "native_format_rate": 0.5,
        "false_block_rate": 0.1,
        "provenance_correct_rate": 0.9,
    }
    assert driver._strict_core_improvement(phase89, dict(phase89)) is False
    improved = {**phase89, "false_block_rate": 0.0}
    assert driver._strict_core_improvement(phase89, improved) is True


def test_phase91_candidate_runtime_rebuilds_parent_before_dpo() -> None:
    source = DRIVER_PATH.read_text(encoding="utf-8")

    parent_load = 'PeftModel.from_pretrained(model, str(PARENT_ADAPTER_ROOT)'
    merge = "model = model.merge_and_unload()"
    dpo_load = 'PeftModel.from_pretrained(model, str(adapter_path)'
    assert source.index(parent_load) < source.index(merge) < source.index(dpo_load)


def test_phase91_parser_rejects_external_or_unregistered_variants() -> None:
    driver = _load_driver()
    parser = driver._parser()

    try:
        parser.parse_args(
            ["generate", "--scope", "full", "--variant", "external_provider"]
        )
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("external variant should be rejected")
