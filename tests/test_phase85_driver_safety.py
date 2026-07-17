from __future__ import annotations

from pathlib import Path

import pytest

import tools.phase85_low_fallback_semantic_guard as driver


def _config(root: Path) -> driver.DriverConfig:
    return driver.DriverConfig(
        evidence_root=root / "evidence",
        model_path=root / "model",
        mode="mock",
    )


def _complete_review(
    config: driver.DriverConfig,
    *,
    output_count: int,
) -> Path:
    cache_path = driver._review_cache_path(config, driver.PHASE85_VARIANTS[2])
    rows = [
        {
            "session_id": f"session-{index}",
            "turn": index,
            "format_eligible": True,
            "returned_output": f"temporary raw output {index}",
            "output_sha256": driver._text_sha256(f"temporary raw output {index}"),
        }
        for index in range(1, output_count + 1)
    ]
    driver._write_jsonl(cache_path, rows)
    structural_path, _ = driver._variant_paths(config, driver.PHASE85_VARIANTS[2])
    driver._write_jsonl(
        structural_path,
        [
            {
                "session_id": row["session_id"],
                "turn_metadata": [
                    {
                        "turn": row["turn"],
                        "format_eligible": True,
                        "output_sha256": row["output_sha256"],
                    }
                ],
            }
            for row in rows
        ],
    )
    assert driver._review_template(config, clean_evidence=False) == 0

    review_path = config.evidence_root / "manual-semantic-review.json"
    review = driver._read_json(review_path)
    review.update(
        {
            "complete": True,
            "passed": True,
            "reviewed_output_count": output_count,
            "reviewed_output_keys_sha256": review["expected_output_keys_sha256"],
            "reviewer_ids": ["test-reviewer"],
            "residual_unsupported_claim_count": 0,
            "false_block_count": 0,
            "other_semantic_failure_count": 0,
        }
    )
    driver._write_json(review_path, review)
    return cache_path


def test_safe_clean_directory_allows_only_strict_allowlisted_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    docs_demo_root = repo_root / "docs/demo"
    temp_root = tmp_path / "system-temp"
    docs_child = docs_demo_root / "phase85-test" / "nested"
    temp_child = temp_root / "phase85-test" / "nested"
    external = tmp_path / "external"
    for path in (docs_child, temp_child, external):
        path.mkdir(parents=True)
        (path / "sentinel.txt").write_text("keep-or-delete", encoding="utf-8")

    monkeypatch.setattr(driver, "REPO_ROOT", repo_root)
    monkeypatch.setattr(driver, "SYSTEM_TEMP_ROOT", temp_root)

    driver._safe_clean_directory(docs_demo_root / "phase85-test")
    driver._safe_clean_directory(temp_root / "phase85-test")

    assert not docs_child.exists()
    assert not temp_child.exists()
    for unsafe in (
        Path("/"),
        repo_root,
        repo_root.parent,
        docs_demo_root,
        temp_root,
        Path.home(),
        external,
    ):
        with pytest.raises(ValueError, match="refusing to remove unsafe evidence directory"):
            driver._safe_clean_directory(unsafe)
    assert (external / "sentinel.txt").is_file()

    escape = docs_demo_root / "escape"
    escape.parent.mkdir(parents=True, exist_ok=True)
    escape.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="refusing to remove unsafe evidence directory"):
        driver._safe_clean_directory(escape)
    assert (external / "sentinel.txt").is_file()


@pytest.mark.parametrize("symlink_root", ["docs", "docs/demo"])
def test_safe_clean_directory_rejects_symlinked_repo_docs_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_root: str,
) -> None:
    repo_root = tmp_path / "repo"
    external = tmp_path / "external-docs"
    repo_root.mkdir()
    if symlink_root == "docs":
        target = external / "demo/phase85-test"
        target.mkdir(parents=True)
        (repo_root / "docs").symlink_to(external, target_is_directory=True)
        clean_target = repo_root / "docs/demo/phase85-test"
    else:
        target = external / "phase85-test"
        target.mkdir(parents=True)
        (repo_root / "docs").mkdir()
        (repo_root / "docs/demo").symlink_to(external, target_is_directory=True)
        clean_target = repo_root / "docs/demo/phase85-test"
    (target / "sentinel.txt").write_text("must survive", encoding="utf-8")

    monkeypatch.setattr(driver, "REPO_ROOT", repo_root)

    with pytest.raises(ValueError, match="refusing to remove unsafe evidence directory"):
        driver._safe_clean_directory(clean_target)
    assert (target / "sentinel.txt").is_file()


def test_safe_clean_directory_allows_standard_tmp_alias(tmp_path: Path) -> None:
    target = Path("/tmp") / f"phase85-driver-safety-{tmp_path.name}"
    (target / "nested").mkdir(parents=True)
    (target / "nested/sentinel.txt").write_text("delete", encoding="utf-8")

    driver._safe_clean_directory(target)

    assert not target.exists()


def test_safe_clean_directory_fails_closed_when_removal_does_not_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    target = repo_root / "docs/demo/phase85-test"
    target.mkdir(parents=True)
    monkeypatch.setattr(driver, "REPO_ROOT", repo_root)
    monkeypatch.setattr(driver.shutil, "rmtree", lambda _path: None)

    with pytest.raises(RuntimeError, match="failed to clean evidence directory"):
        driver._safe_clean_directory(target)

    assert target.is_dir()


def test_safe_unlink_under_rejects_symlinked_child_directory(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    external = tmp_path / "external"
    root.mkdir()
    external.mkdir()
    sentinel = external / "metrics.json"
    sentinel.write_text("must survive", encoding="utf-8")
    (root / "evidence-generation").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="refusing unsafe child path"):
        driver._safe_unlink_under(
            root / "evidence-generation/metrics.json",
            root / "evidence-generation",
        )

    assert sentinel.read_text(encoding="utf-8") == "must survive"


def test_write_jsonl_is_private_and_cleans_failed_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "review-cache/review.jsonl"
    driver._write_jsonl(path, [{"returned_output": "private"}])
    assert path.stat().st_mode & 0o777 == 0o600

    path.unlink()
    temporary = path.with_suffix(path.suffix + ".tmp")
    real_replace = Path.replace

    def fail_replace(source: Path, target: Path) -> Path:
        if source == temporary:
            raise OSError("injected replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        driver._write_jsonl(path, [{"returned_output": "private"}])
    assert not path.exists()
    assert not temporary.exists()


def test_prepare_refuses_to_rebind_freeze_over_existing_generation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    structural_path, _ = driver._variant_paths(config, driver.PHASE85_VARIANTS[0])
    driver._write_jsonl(structural_path, [{"session_id": "old-output"}])

    with pytest.raises(SystemExit, match="refusing to replace the Phase85 freeze"):
        driver._prepare(config, clean_evidence=False)


class _FakePipelineService:
    def __init__(self, runtime_attempt_count: int | None) -> None:
        self.runtime_attempt_count = runtime_attempt_count
        self.calls = 0

    def chat_completion(self, **_kwargs: object) -> dict[str, object]:
        self.calls += 1
        generation: dict[str, object] = {"served_by": "mock"}
        if self.runtime_attempt_count is not None:
            generation["runtime_attempt_count"] = self.runtime_attempt_count
        return {
            "served_by": "mock",
            "choices": [
                {
                    "message": {
                        "content": (
                            "结论：状态未验证。\n"
                            "依据：对象清单仍缺失。\n"
                            "下一步：继续核验。"
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "metadata": {"inference": {"generation": generation}},
        }


class _FailingSecondCallPipelineService:
    def __init__(self) -> None:
        self.calls = 0

    def chat_completion(self, **_kwargs: object) -> dict[str, object]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("private-marker-must-not-be-persisted")
        return _FakePipelineService(runtime_attempt_count=1).chat_completion()


def test_run_session_separates_api_calls_from_backend_attempts(tmp_path: Path) -> None:
    service = _FakePipelineService(runtime_attempt_count=2)
    session = dict(driver.build_phase85_holdout()["sessions"][0])

    row = driver._run_session(
        service=service,
        session=session,
        variant=driver.PHASE85_VARIANTS[0],
        config=_config(tmp_path),
    )

    assert service.calls == 3
    assert row["api_invocation_count"] == 3
    assert row["backend_runtime_attempt_count"] == 6
    assert row["exactly_one_api_call_per_turn"] is True
    assert row["exactly_one_backend_attempt_per_turn"] is False
    assert all(turn["api_call_count"] == 1 for turn in row["turn_metadata"])
    assert all(turn["runtime_attempt_count"] == 2 for turn in row["turn_metadata"])
    assert all(turn["model_call_count"] == 2 for turn in row["turn_metadata"])
    assert driver._runtime_attempt_count({}) == 0
    assert driver._runtime_attempt_count({"runtime_attempt_count": None}) == 0
    assert driver._runtime_attempt_count({"runtime_attempt_count": "1"}) == 0
    assert driver._runtime_attempt_count({"runtime_attempt_count": -1}) == 0


def test_generate_records_attempted_api_calls_when_chat_completion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    service = _FailingSecondCallPipelineService()
    session = dict(driver.build_phase85_holdout()["sessions"][0])
    monkeypatch.setattr(driver, "PipelineService", lambda: service)
    monkeypatch.setattr(driver, "build_phase85_holdout", lambda: {"sessions": [session]})
    monkeypatch.setattr(driver, "_freeze_check", lambda _config: {"passed": True})
    monkeypatch.setattr(driver, "_set_runtime_environment", lambda _config: {})
    monkeypatch.setattr(driver, "_restore_runtime_environment", lambda _previous: None)

    def aggregate(
        rows: list[dict[str, object]],
        _sessions: list[dict[str, object]],
        *,
        variant: str,
        config: driver.DriverConfig,
    ) -> dict[str, object]:
        del config
        turn_metadata = rows[0]["turn_metadata"]
        assert isinstance(turn_metadata, list)
        return {
            "variant": variant,
            "session_count": len(rows),
            "api_call_count": sum(
                int(turn.get("api_call_count") or 0) for turn in turn_metadata
            ),
            "runtime_attempt_count": sum(
                int(turn.get("runtime_attempt_count") or 0) for turn in turn_metadata
            ),
            "model_call_count": sum(
                int(turn.get("model_call_count") or 0) for turn in turn_metadata
            ),
            "format_eligible_turn_count": 0,
            "all_sessions_completed": False,
            "one_api_call_per_turn": False,
            "one_backend_attempt_per_turn": False,
        }

    monkeypatch.setattr(driver, "_aggregate_variant", aggregate)

    assert driver._generate(config, driver.PHASE85_VARIANTS[0], False) == 1

    structural_path, _ = driver._variant_paths(config, driver.PHASE85_VARIANTS[0])
    row = driver._read_jsonl(structural_path)[0]
    assert service.calls == 2
    assert row["status"] == "failed"
    assert row["error_type"] == "RuntimeError"
    assert row["api_invocation_count"] == 2
    assert row["api_path_invoked"] is True
    assert [turn["api_call_count"] for turn in row["turn_metadata"]] == [1, 1]
    assert [turn["api_call_succeeded"] for turn in row["turn_metadata"]] == [True, False]
    assert [turn["runtime_attempt_count"] for turn in row["turn_metadata"]] == [1, 0]
    assert "private-marker-must-not-be-persisted" not in structural_path.read_text(
        encoding="utf-8"
    )


def test_generate_preserves_known_attempt_counts_when_review_cache_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    session = dict(driver.build_phase85_holdout()["sessions"][0])
    monkeypatch.setattr(driver, "build_phase85_holdout", lambda: {"sessions": [session]})
    monkeypatch.setattr(driver, "_freeze_check", lambda _config: {"passed": True})
    monkeypatch.setattr(driver, "_set_runtime_environment", lambda _config: {})
    monkeypatch.setattr(driver, "_restore_runtime_environment", lambda _previous: None)
    monkeypatch.setattr(driver, "PipelineService", lambda: object())
    monkeypatch.setattr(
        driver,
        "_run_session",
        lambda **_kwargs: {
            "session_id": session["session_id"],
            "status": "completed",
            "api_invocation_count": 3,
            "turn_metadata": [
                {"turn": turn, "api_call_count": 1, "runtime_attempt_count": 1}
                for turn in range(1, 4)
            ],
            "_ephemeral_review_outputs": [
                {
                    "session_id": session["session_id"],
                    "turn": 1,
                    "format_eligible": True,
                    "returned_output": "temporary",
                    "output_sha256": driver._text_sha256("temporary"),
                }
            ],
        },
    )
    cache_path = driver._review_cache_path(config, driver.PHASE85_VARIANTS[2])
    real_write_jsonl = driver._write_jsonl

    def fail_cache_write(path: Path, rows: object) -> None:
        if path == cache_path:
            raise OSError("injected cache failure")
        real_write_jsonl(path, rows)

    monkeypatch.setattr(driver, "_write_jsonl", fail_cache_write)

    observed: dict[str, object] = {}

    def aggregate(
        rows: list[dict[str, object]],
        _sessions: list[dict[str, object]],
        *,
        variant: str,
        config: driver.DriverConfig,
    ) -> dict[str, object]:
        del config
        observed.update(rows[0])
        return {
            "variant": variant,
            "session_count": 1,
            "api_call_count": 3,
            "runtime_attempt_count": 3,
            "model_call_count": 3,
            "format_eligible_turn_count": 0,
            "all_sessions_completed": False,
            "one_api_call_per_turn": False,
            "one_backend_attempt_per_turn": False,
        }

    monkeypatch.setattr(driver, "_aggregate_variant", aggregate)

    assert driver._generate(config, driver.PHASE85_VARIANTS[2], False) == 1
    assert observed["status"] == "failed"
    assert observed["attempt_counts_known"] is True
    assert observed["api_invocation_count"] == 3
    assert observed["backend_runtime_attempt_count"] == 3


def test_generation_audit_requires_90_api_calls_and_one_backend_attempt_per_turn(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    for variant in driver.PHASE85_VARIANTS:
        rows = [
            {
                "session_id": f"{variant}-{session_index}",
                "raw_model_text_persisted": False,
                "turn_metadata": [
                    {
                        "turn": turn,
                        "api_call_count": 1,
                        "runtime_attempt_count": 1,
                    }
                    for turn in range(1, 4)
                ],
            }
            for session_index in range(driver.PHASE85_SESSION_COUNT)
        ]
        structural_path, metrics_path = driver._variant_paths(config, variant)
        driver._write_jsonl(structural_path, rows)
        driver._write_json(metrics_path, {"variant": variant})

    audit = driver._generation_audit(config)

    assert audit["passed"] is True
    assert audit["expected_api_call_count_per_variant"] == 90
    assert set(audit["observed_api_call_count_by_variant"].values()) == {90}
    assert audit["one_backend_attempt_per_turn"] is True

    structural_path, _ = driver._variant_paths(config, driver.PHASE85_VARIANTS[0])
    rows = driver._read_jsonl(structural_path)
    rows[0]["turn_metadata"][0]["runtime_attempt_count"] = 2
    driver._write_jsonl(structural_path, rows)

    retried = driver._generation_audit(config)
    assert retried["passed"] is False
    assert retried["checks"]["all_variants_exactly_90_api_calls"] is True
    assert retried["checks"]["all_turns_exactly_one_backend_attempt"] is False
    assert retried["observed_total_api_call_count"] == 270
    assert retried["observed_backend_runtime_attempt_count"] == 271

    rows[0]["turn_metadata"][0].pop("runtime_attempt_count")
    driver._write_jsonl(structural_path, rows)
    missing_telemetry = driver._generation_audit(config)
    assert missing_telemetry["passed"] is False
    assert missing_telemetry["checks"]["all_turns_exactly_one_backend_attempt"] is False
    assert missing_telemetry["observed_backend_runtime_attempt_count"] == 269


def test_api_smoke_failure_does_not_invent_backend_attempt_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    class FailingService:
        def chat_completion(self, **_kwargs: object) -> dict[str, object]:
            raise RuntimeError("private failure detail")

    monkeypatch.setattr(driver, "PipelineService", FailingService)
    monkeypatch.setattr(driver, "_freeze_check", lambda _config: {"passed": True})
    monkeypatch.setattr(driver, "_set_runtime_environment", lambda _config: {})
    monkeypatch.setattr(driver, "_restore_runtime_environment", lambda _previous: None)

    assert driver._api_smoke(config, clean_evidence=False) == 1
    result = driver._read_json(config.evidence_root / "api_smoke.json")
    assert result["api_call_count"] == 1
    assert result["runtime_attempt_count"] is None
    assert result["model_call_count"] is None
    assert result["runtime_attempt_count_known"] is False
    assert "private failure detail" not in str(result)


def test_manual_review_uses_hash_manifest_after_raw_cache_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(driver, "DEFAULT_REVIEW_CACHE_ROOT", tmp_path / "review-cache")
    monkeypatch.setattr(driver, "PHASE85_FORMAT_ELIGIBLE_TURN_COUNT", 2)
    cache_path = _complete_review(config, output_count=2)
    assert cache_path.stat().st_mode & 0o777 == 0o600

    manifest_path = config.evidence_root / driver.REVIEW_OUTPUT_KEY_MANIFEST
    manifest = driver._read_json(manifest_path)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert manifest["output_count"] == 2
    assert len(manifest["output_key_hashes"]) == 2
    assert manifest["output_keys_sha256"] == driver.stable_hash(
        manifest["output_key_hashes"]
    )
    assert manifest["output_key_hashes_sha256"] == manifest["output_keys_sha256"]
    assert "temporary raw output" not in manifest_text
    assert "session_id" not in manifest_text
    assert "returned_output" not in manifest_text

    cache_path.unlink()
    review = driver._manual_review(config)

    assert review["complete"] is True
    assert review["passed"] is True
    assert review["integrity_passed"] is True
    assert review["review_output_manifest_count"] == 2
    assert review["review_cache_present"] is False

    tampered_hash = "f" * 64
    manifest["output_keys_sha256"] = tampered_hash
    manifest["output_key_hashes_sha256"] = tampered_hash
    driver._write_json(manifest_path, manifest)
    raw_review = driver._read_json(config.evidence_root / "manual-semantic-review.json")
    raw_review["expected_output_keys_sha256"] = tampered_hash
    raw_review["reviewed_output_keys_sha256"] = tampered_hash
    driver._write_json(config.evidence_root / "manual-semantic-review.json", raw_review)

    tampered = driver._manual_review(config)
    assert tampered["integrity_passed"] is False
    assert tampered["complete"] is False
    assert tampered["passed"] is False


def test_review_template_rejects_cache_hash_or_structural_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(driver, "DEFAULT_REVIEW_CACHE_ROOT", tmp_path / "review-cache")
    monkeypatch.setattr(driver, "PHASE85_FORMAT_ELIGIBLE_TURN_COUNT", 1)
    cache_path = driver._review_cache_path(config, driver.PHASE85_VARIANTS[2])
    returned_output = "temporary returned output"
    driver._write_jsonl(
        cache_path,
        [
            {
                "session_id": "session-1",
                "turn": 1,
                "format_eligible": True,
                "returned_output": returned_output,
                "output_sha256": "f" * 64,
            }
        ],
    )
    structural_path, _ = driver._variant_paths(config, driver.PHASE85_VARIANTS[2])
    driver._write_jsonl(
        structural_path,
        [
            {
                "session_id": "session-1",
                "turn_metadata": [
                    {
                        "turn": 1,
                        "format_eligible": True,
                        "output_sha256": driver._text_sha256(returned_output),
                    }
                ],
            }
        ],
    )

    with pytest.raises(RuntimeError, match="does not bind to structural output"):
        driver._review_template(config, clean_evidence=False)


def test_run_logged_persists_only_digest_not_process_output(tmp_path: Path) -> None:
    del tmp_path
    private_marker = "PHASE85_PRIVATE_PROCESS_OUTPUT"
    result = driver._run_logged(
        ["/usr/bin/printf", f"{private_marker}\\n"],
        command_id="privacy-test",
    )

    assert result["exit_code"] == 0
    assert result["raw_process_output_persisted"] is False
    assert private_marker not in str(result)
    assert "command" not in result
    assert "output" not in result
    assert driver._walk_forbidden_keys(result) == []


def test_finalize_rejects_recomputed_metric_or_live_freeze_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    for variant in driver.PHASE85_VARIANTS:
        _, metrics_path = driver._variant_paths(config, variant)
        driver._write_json(metrics_path, {"variant": variant, "value": 1})

    monkeypatch.setattr(
        driver,
        "_aggregate_variant",
        lambda _rows, _sessions, *, variant, config: {
            "variant": variant,
            "value": 2,
        },
    )
    with pytest.raises(RuntimeError, match="stored metrics do not match"):
        driver._finalize(config)

    monkeypatch.setattr(
        driver,
        "_aggregate_variant",
        lambda _rows, _sessions, *, variant, config: {
            "variant": variant,
            "value": 1,
        },
    )
    monkeypatch.setattr(driver, "_freeze_check", lambda _config: {"passed": False})
    with pytest.raises(RuntimeError, match="freeze changed before finalize"):
        driver._finalize(config)


def test_finalize_deletes_review_cache_and_validate_requires_it_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(driver, "DEFAULT_REVIEW_CACHE_ROOT", tmp_path / "review-cache")
    monkeypatch.setattr(driver, "PHASE85_FORMAT_ELIGIBLE_TURN_COUNT", 2)
    cache_path = _complete_review(config, output_count=2)

    for variant in driver.PHASE85_VARIANTS:
        _, metrics_path = driver._variant_paths(config, variant)
        driver._write_json(
            metrics_path,
            {
                "variant": variant,
                "format_eligible_turn_count": 2,
                "format_accounting_passed": True,
            },
        )
        driver._write_json(
            config.generation_root / f"freeze_check_{variant}.json",
            {"passed": True},
        )
    driver._write_json(config.evidence_root / "pre_experiment_freeze.json", {"passed": True})
    driver._write_json(config.evidence_root / "api_smoke.json", {"passed": True})
    driver._write_json(
        config.evidence_root / "full_regression_summary.json", {"passed": True}
    )

    decision = {
        "status": "archive_test",
        "recommendation": "archive_test",
        "actual_user_feedback_count": 0,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
        "automatic_deployment_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_changed": False,
        "simulated_lab_runtime_benefit": False,
    }
    monkeypatch.setattr(driver, "_ordinary_identity", lambda _config: {"passed": True})
    monkeypatch.setattr(driver, "_generation_audit", lambda _config: {"passed": True})
    monkeypatch.setattr(driver, "_public_private_audit", lambda _config: {"passed": True})
    monkeypatch.setattr(driver, "_freeze_check", lambda _config: {"passed": True})
    monkeypatch.setattr(
        driver,
        "_aggregate_variant",
        lambda _rows, _sessions, *, variant, config: driver._read_json(
            driver._variant_paths(config, variant)[1]
        ),
    )
    monkeypatch.setattr(driver, "build_phase85_decision", lambda **_kwargs: dict(decision))
    monkeypatch.setattr(driver, "_decision_markdown", lambda *_args: "decision")
    monkeypatch.setattr(driver, "_structural_output_summary", lambda *_args: "summary")
    monkeypatch.setattr(driver, "_runbook", lambda _config: "runbook")
    monkeypatch.setattr(driver, "_next_goal", lambda _decision: "next")

    real_unlink = Path.unlink

    def fail_review_cache_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == cache_path:
            raise OSError("injected cache deletion failure")
        real_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as deletion_failure:
        deletion_failure.setattr(Path, "unlink", fail_review_cache_unlink)
        with pytest.raises(
            RuntimeError,
            match="failed to delete Phase85 temporary V4 review cache",
        ):
            driver._finalize(config)

    assert cache_path.is_file()
    for name in (
        "phase85-final-decision.json",
        "evidence_manifest.json",
        "finalization_state.json",
    ):
        assert not (config.evidence_root / name).exists()

    real_manual_review = driver._manual_review
    cache_states: list[bool] = []

    def observed_manual_review(review_config: driver.DriverConfig) -> dict[str, object]:
        cache_states.append(cache_path.is_file())
        return real_manual_review(review_config)

    monkeypatch.setattr(driver, "_manual_review", observed_manual_review)

    assert driver._finalize(config) == 0
    assert cache_states == [True]
    assert not cache_path.exists()
    assert (config.evidence_root / "phase85-final-decision.json").is_file()
    finalization_state = driver._read_json(
        config.evidence_root / "finalization_state.json"
    )
    assert finalization_state["review_cache_was_present"] is True
    assert finalization_state["review_cache_deleted"] is True
    assert finalization_state["temporary_v4_review_cache_absent"] is True
    integrity = driver._read_json(config.evidence_root / "evidence_integrity.json")
    assert integrity["checks"]["temporary_v4_review_cache_absent"] is True
    assert driver._validate(config) == 0

    driver._write_jsonl(cache_path, [{"returned_output": "temporary"}])
    assert driver._validate(config) == 1
    validation = driver._read_json(config.evidence_root / "validation_summary.json")
    assert validation["checks"]["temporary_v4_review_cache_absent"] is False

    cache_path.unlink()
    temporary_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary_cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache_path.write_text("temporary", encoding="utf-8")
    assert driver._validate(config) == 1
    validation = driver._read_json(config.evidence_root / "validation_summary.json")
    assert validation["checks"]["temporary_v4_review_cache_absent"] is False

    temporary_cache_path.unlink()
    decision_path = config.evidence_root / "phase85-final-decision.json"
    missing_feedback_count = driver._read_json(decision_path)
    missing_feedback_count.pop("actual_user_feedback_count")
    driver._write_json(decision_path, missing_feedback_count)
    assert driver._validate(config) == 1
    validation = driver._read_json(config.evidence_root / "validation_summary.json")
    assert validation["checks"]["actual_user_feedback_count_zero"] is False
