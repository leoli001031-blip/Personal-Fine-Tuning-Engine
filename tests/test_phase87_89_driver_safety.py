from __future__ import annotations

from pathlib import Path

import pytest

from pfe_core.phase87_failure_driven_training import build_phase87_training_candidates
import tools.phase87_89_failure_driven_adapter_loop as driver


def test_safe_clean_is_limited_to_strict_descendants(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    child = root / "probe"
    child.mkdir(parents=True)
    (child / "sentinel").write_text("delete", encoding="utf-8")

    driver._safe_clean(child, root)

    assert not child.exists()
    for unsafe in (root, root.parent, tmp_path / "outside"):
        with pytest.raises(ValueError, match="refusing unsafe clean"):
            driver._safe_clean(unsafe, root)


def test_safe_clean_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    external = tmp_path / "external"
    root.mkdir()
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    link = root / "probe"
    link.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="refusing unsafe clean"):
        driver._safe_clean(link, root)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_job_spec_keeps_completion_only_boundary_and_local_model(tmp_path: Path) -> None:
    samples = build_phase87_training_candidates()["samples"]
    probe_samples = driver._select_probe_samples(samples)
    spec = driver._job_spec(probe_samples, tmp_path / "output", 5)
    boundary = driver._completion_boundary_report(spec)

    assert spec["recipe"]["training"]["base_model"] == str(driver.MODEL_PATH)
    assert len(probe_samples) == 25
    assert spec["recipe"]["training"]["max_steps"] == 5
    assert spec["recipe"]["training"]["max_length"] == 224
    assert spec["phase87"]["completion_only_loss_required"] is True
    assert spec["phase87"]["actual_user_feedback"] is False
    assert spec["phase87"]["auto_promotion_allowed"] is False
    assert boundary["passed"] is True


def test_cli_has_no_generation_deploy_or_promotion_commands() -> None:
    parser = driver._build_parser()
    help_text = parser.format_help()

    assert "prepare" in help_text
    assert "train" in help_text
    assert "deploy" not in help_text
    assert "promote" not in help_text
