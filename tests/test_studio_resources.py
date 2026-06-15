from pathlib import Path

from pfe_server.studio_resources import (
    build_models_payload,
    build_workspaces_payload,
    model_candidate,
    validate_model_reference,
    workspace_record,
    workspace_slug_issues,
)


def test_workspace_slug_issues_reject_invalid_workspace_names() -> None:
    assert workspace_slug_issues("")[0]["code"] == "workspace_required"
    assert workspace_slug_issues("-client")[0]["code"] == "workspace_invalid_name"
    assert workspace_slug_issues("client/a")[0]["code"] == "workspace_invalid_chars"
    assert workspace_slug_issues("a" * 65)[0]["code"] == "workspace_too_long"
    assert workspace_slug_issues("client-a") == []


def test_workspaces_payload_discovers_current_env_and_existing_workspace_dirs(tmp_path: Path) -> None:
    (tmp_path / "adapters" / "client-a" / "20260615-001").mkdir(parents=True)
    (tmp_path / "workspaces" / "client-b").mkdir(parents=True)
    (tmp_path / "workspaces" / "bad/name").mkdir(parents=True)

    payload = build_workspaces_payload(tmp_path, "client-a", env_workspace="env-client")

    ids = [item["id"] for item in payload["items"]]
    assert ids[0] == "client-a"
    assert set(ids) == {"client-a", "client-b", "env-client", "bad"}
    assert payload["current"] == "client-a"
    assert payload["home"] == str(tmp_path)
    assert payload["count"] == 4
    client_a = payload["items"][0]
    assert client_a["version_count"] == 1
    assert client_a["exists"] is True
    env_client = next(item for item in payload["items"] if item["id"] == "env-client")
    assert env_client["exists"] is False
    assert env_client["switchable"] is True


def test_workspace_record_marks_invalid_workspace_unswitchable(tmp_path: Path) -> None:
    record = workspace_record(tmp_path, "-bad", current="client-a")
    assert record["switchable"] is False
    assert record["validation"]["valid"] is False
    assert record["adapters_path"] is None


def test_models_payload_dedupes_selected_env_and_repo_models(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    repo_models = tmp_path / "models"
    (repo_models / "repo-a").mkdir(parents=True)
    (repo_models / "repo-b").mkdir()

    payload = build_models_payload(str(selected), env_model=str(selected), repo_models_dir=repo_models)

    ids = [item["id"] for item in payload["candidates"]]
    assert ids == [str(selected), str(repo_models / "repo-a"), str(repo_models / "repo-b")]
    assert payload["selected"] == str(selected)
    assert payload["selected_label"] == "selected"
    assert payload["count"] == 3


def test_model_candidate_and_validation_surface_local_path_state(tmp_path: Path) -> None:
    existing = tmp_path / "model"
    existing.mkdir()

    candidate = model_candidate(str(existing), source="request", selected=str(existing))
    assert candidate["label"] == "model"
    assert candidate["local_path"] == str(existing)
    assert candidate["exists"] is True
    assert candidate["selected"] is True

    missing = validate_model_reference(str(tmp_path / "missing"))
    assert missing["valid"] is False
    assert missing["issues"][0]["code"] == "model_path_not_found"

    empty = validate_model_reference("")
    assert empty["valid"] is False
    assert empty["issues"][0]["code"] == "model_required"
