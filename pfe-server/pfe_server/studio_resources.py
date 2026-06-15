from __future__ import annotations

from pathlib import Path
from typing import Any


def workspace_slug_issues(name: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not name:
        issues.append({"code": "workspace_required", "detail": "workspace name is required"})
        return issues
    if len(name) > 64:
        issues.append({"code": "workspace_too_long", "detail": "workspace name is too long"})
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    if any(char not in allowed for char in name):
        issues.append({"code": "workspace_invalid_chars", "detail": "use letters, numbers, dot, dash, or underscore"})
    if name in {".", ".."} or name[0] in {".", "-"}:
        issues.append({"code": "workspace_invalid_name", "detail": "workspace name must not start with dot or dash"})
    return issues


def workspace_paths(home: Path, name: str) -> dict[str, Path]:
    return {
        "home": home,
        "adapters": home / "adapters" / name,
        "state": home / "workspaces" / name,
    }


def workspace_record(home: Path, name: str, *, current: str) -> dict[str, Any]:
    validation_issues = workspace_slug_issues(name)
    if validation_issues:
        return {
            "id": name,
            "label": Path(name).name or name,
            "current": name == current,
            "adapters_path": None,
            "state_path": None,
            "exists": False,
            "version_count": 0,
            "switchable": False,
            "validation": {"valid": False, "issues": validation_issues},
        }
    paths = workspace_paths(home, name)
    adapters_path = paths["adapters"]
    state_path = paths["state"]
    versions = [
        child.name
        for child in adapters_path.iterdir()
        if child.is_dir()
    ] if adapters_path.is_dir() else []
    return {
        "id": name,
        "label": name,
        "current": name == current,
        "adapters_path": str(adapters_path),
        "state_path": str(state_path),
        "exists": adapters_path.exists() or state_path.exists(),
        "version_count": len(versions),
        "switchable": True,
        "validation": {"valid": True, "issues": []},
    }


def discover_workspaces(home: Path, current: str, *, env_workspace: str | None = None) -> list[dict[str, Any]]:
    names: set[str] = {current or "user_default"}
    if env_workspace:
        names.add(env_workspace)
    for root in (home / "adapters", home / "workspaces"):
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and not workspace_slug_issues(child.name):
                names.add(child.name)
    return [
        workspace_record(home, name, current=current)
        for name in sorted(names, key=lambda item: (item != current, item.lower()))
    ]


def build_workspaces_payload(home: Path, current: str, *, env_workspace: str | None = None) -> dict[str, Any]:
    current = str(current or "user_default")
    items = discover_workspaces(home, current, env_workspace=env_workspace)
    return {
        "current": current,
        "home": str(home),
        "items": items,
        "count": len(items),
        "create_api": "POST /pfe/workspaces",
        "switch_api": "POST /pfe/workspaces",
        "effective_scope": "current_process_next_request",
    }


def model_candidate(model_id: str, *, source: str, selected: str | None = None) -> dict[str, Any]:
    path = Path(model_id).expanduser()
    is_local_path = path.is_absolute() or model_id.startswith(".") or model_id.startswith("~")
    return {
        "id": model_id,
        "label": path.name if is_local_path else model_id,
        "source": source,
        "selected": bool(selected and model_id == selected),
        "local_path": str(path) if is_local_path else None,
        "exists": path.exists() if is_local_path else None,
    }


def discover_model_candidates(
    selected: str,
    *,
    env_model: str | None = None,
    repo_models_dir: Path | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [model_candidate(selected, source="config", selected=selected)]
    if env_model:
        candidates.append(model_candidate(env_model, source="env:PFE_REAL_LOCAL_MODEL", selected=selected))
    if repo_models_dir and repo_models_dir.exists():
        for child in sorted(repo_models_dir.iterdir(), key=lambda item: item.name.lower()):
            if child.is_dir():
                candidates.append(model_candidate(str(child), source="repo:models", selected=selected))
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        model_id = str(candidate.get("id") or "")
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        unique.append(candidate)
    return unique


def build_models_payload(
    selected: str,
    *,
    env_model: str | None = None,
    repo_models_dir: Path | None = None,
) -> dict[str, Any]:
    candidates = discover_model_candidates(selected, env_model=env_model, repo_models_dir=repo_models_dir)
    selected_candidate = model_candidate(selected, source="config", selected=selected)
    return {
        "selected": selected,
        "selected_label": selected_candidate["label"],
        "candidates": candidates,
        "count": len(candidates),
        "mode": "configurable",
        "update_api": "PUT /pfe/config/model",
    }


def validate_model_reference(model_id: str) -> dict[str, Any]:
    candidate = model_candidate(model_id, source="request", selected=model_id)
    issues: list[dict[str, str]] = []
    if not model_id:
        issues.append({"code": "model_required", "detail": "base_model is required"})
    if len(model_id) > 512:
        issues.append({"code": "model_too_long", "detail": "base_model is too long"})
    if candidate["local_path"] and not candidate["exists"]:
        issues.append({"code": "model_path_not_found", "detail": "local model path does not exist"})
    return {
        "valid": not issues,
        "candidate": candidate,
        "issues": issues,
    }


__all__ = [
    "build_models_payload",
    "build_workspaces_payload",
    "discover_model_candidates",
    "discover_workspaces",
    "model_candidate",
    "validate_model_reference",
    "workspace_paths",
    "workspace_record",
    "workspace_slug_issues",
]
