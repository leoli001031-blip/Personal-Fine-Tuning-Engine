"""Filesystem-backed adapter lifecycle management."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .lifecycle import AdapterArtifactFormat, can_promote_from
from .manifest import AdapterManifest
from ..errors import AdapterError
from ..storage import adapter_rows, ensure_runtime_dirs, upsert_adapter_row, utcnow_iso, write_json


class AdapterStore:
    """Manage adapter versions and the latest symlink."""

    allowed_states = {"training", "pending_eval", "promoted", "failed_eval", "archived"}

    def __init__(self, home: str | Path | None = None, workspace: str = "user_default"):
        self.home = ensure_runtime_dirs(home)
        self.workspace = workspace or "user_default"
        self.root = self.home / "adapters" / self.workspace
        self.root.mkdir(parents=True, exist_ok=True)

    def _rows(self) -> list[dict[str, Any]]:
        return adapter_rows(self.home, self.workspace)

    def _version_path(self, version: str) -> Path:
        return self.root / version

    def _manifest_path(self, version: str) -> Path:
        return self._version_path(version) / "adapter_manifest.json"

    def _latest_path(self) -> Path:
        return self.root / "latest"

    def _read_manifest(self, version: str) -> dict[str, Any]:
        manifest_path = self._manifest_path(version)
        if not manifest_path.exists():
            raise AdapterError(f"adapter manifest not found for version {version}")
        return AdapterManifest.from_path(manifest_path).to_dict()

    def _write_manifest(self, version: str, payload: dict[str, Any]) -> None:
        write_json(self._manifest_path(version), payload)

    def merge_manifest(self, version: str, updates: dict[str, Any]) -> dict[str, Any]:
        manifest = self._read_manifest(version)
        merged = dict(manifest)
        for key, value in updates.items():
            if key == "metadata" and isinstance(value, dict):
                existing_metadata = dict(merged.get("metadata") or {})
                existing_metadata.update(value)
                merged["metadata"] = existing_metadata
            elif key == "export" and isinstance(value, dict):
                existing_metadata = dict(merged.get("metadata") or {})
                existing_export = dict(existing_metadata.get("export") or {})
                existing_export.update(value)
                existing_metadata["export"] = existing_export
                merged["metadata"] = existing_metadata
            else:
                merged[key] = value
        self._write_manifest(version, merged)
        return merged

    def _update_row(
        self,
        version: str,
        *,
        base_model: str,
        state: str,
        artifact_format: str,
        adapter_dir: str,
        training_config: dict[str, Any],
        created_at: str,
        num_samples: int = 0,
        eval_report: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifact_name: str = "adapter_model.safetensors",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if state not in self.allowed_states:
            raise AdapterError(f"invalid adapter state: {state}")
        upsert_adapter_row(
            {
                "version": version,
                "workspace": self.workspace,
                "base_model": base_model,
                "created_at": created_at,
                "updated_at": utcnow_iso(),
                "num_samples": num_samples,
                "state": state,
                "artifact_format": artifact_format,
                "adapter_dir": adapter_dir,
                "manifest_path": str(self._manifest_path(version)),
                "artifact_path": str(self._version_path(version) / artifact_name),
                "training_config": training_config,
                "eval_report": eval_report,
                "metrics": metrics,
                "promoted_at": utcnow_iso() if state == "promoted" else None,
                "archived_at": utcnow_iso() if state == "archived" else None,
                "metadata": metadata or {},
            },
            self.home,
        )

    def _next_version(self) -> str:
        prefix = datetime.now().strftime("%Y%m%d")
        existing = [row["version"] for row in self._rows() if str(row["version"]).startswith(prefix)]
        next_number = len(existing) + 1
        return f"{prefix}-{next_number:03d}"

    def _resolve_relative(self, version: str) -> str:
        if version == "latest":
            latest = self.current_latest_version()
            if latest is None:
                raise AdapterError("no promoted adapter is available")
            return latest
        if version.startswith("-"):
            try:
                offset = abs(int(version))
            except ValueError as exc:
                raise AdapterError(f"invalid rollback version: {version}") from exc
            rows = self._rows()
            if offset >= len(rows):
                raise AdapterError(f"cannot resolve {version}; only {len(rows)} version(s) exist")
            return rows[offset]["version"]
        return version

    def current_latest_version(self) -> str | None:
        latest_link = self._latest_path()
        if not latest_link.is_symlink():
            return None
        target = os.readlink(latest_link)
        return Path(target).name

    def create_training_version(
        self,
        *,
        base_model: str,
        training_config: dict[str, Any],
        artifact_format: str = "peft_lora",
    ) -> dict[str, Any]:
        version = self._next_version()
        version_dir = self._version_path(version)
        version_dir.mkdir(parents=True, exist_ok=False)
        created_at = utcnow_iso()
        normalized_artifact_format = artifact_format
        if not isinstance(normalized_artifact_format, AdapterArtifactFormat):
            normalized_artifact_format = AdapterArtifactFormat(str(artifact_format))
        backend_plan = dict(training_config.get("backend_plan") or {})
        runtime = dict(training_config.get("runtime") or {})
        training_summary = {
            "backend": training_config.get("backend", "mock_local"),
            "train_type": training_config.get("train_type", "sft"),
            "runtime_device": training_config.get("runtime_device") or runtime.get("runtime_device"),
            "requires_export_step": bool(training_config.get("requires_export_step", False)),
            "export_backend": training_config.get("export_backend"),
            "export_format": training_config.get("export_format"),
        }
        manifest_obj = AdapterManifest(
            version=version,
            workspace=self.workspace,
            base_model=base_model,
            created_at=datetime.fromisoformat(created_at),
            state="training",
            num_samples=0,
            adapter_dir=str(version_dir),
            artifact_format=normalized_artifact_format,
            training_backend=training_summary["backend"],
            inference_backend="transformers",
            requires_export=training_summary["requires_export_step"],
            metadata={
                "supported_inference_backends": ["transformers", "mlx", "llama_cpp"],
                "export": {},
                "training": training_summary,
                "backend_plan": backend_plan,
                "runtime": runtime,
            },
        )
        manifest = manifest_obj.to_dict()
        self._write_manifest(version, manifest)
        self._update_row(
            version,
            base_model=base_model,
            state="training",
            artifact_format=normalized_artifact_format.value,
            adapter_dir=str(version_dir),
            training_config=training_config,
            created_at=created_at,
            artifact_name=manifest.get("artifact_name", "adapter_model.safetensors"),
            metadata=manifest.get("metadata", {}),
        )
        return {"version": version, "path": str(version_dir), "manifest": manifest}

    def mark_pending_eval(self, version: str, *, num_samples: int, metrics: dict[str, Any] | None = None) -> None:
        manifest = self._read_manifest(version)
        manifest["state"] = "pending_eval"
        manifest["num_samples"] = num_samples
        manifest["updated_at"] = utcnow_iso()
        if metrics:
            manifest["training_metrics"] = metrics
        self._write_manifest(version, manifest)
        self._update_row(
            version,
            base_model=manifest["base_model"],
            state="pending_eval",
            artifact_format=manifest["artifact_format"],
            adapter_dir=str(self._version_path(version)),
            training_config={"training_backend": manifest.get("training_backend", "mock_local")},
            created_at=manifest["created_at"],
            num_samples=num_samples,
            metrics=metrics,
            artifact_name=manifest.get("artifact_name", "adapter_model.safetensors"),
            metadata=manifest.get("metadata", {}),
        )

    def attach_eval_report(self, version: str, report: dict[str, Any]) -> None:
        manifest = self._read_manifest(version)
        state = "failed_eval" if report.get("recommendation") == "keep_previous" else manifest.get("state", "pending_eval")
        manifest["state"] = state
        manifest["eval_summary"] = {
            "comparison": report.get("comparison"),
            "recommendation": report.get("recommendation"),
            "scores": report.get("scores", {}),
        }
        self._write_manifest(version, manifest)
        write_json(self._version_path(version) / "eval_report.json", report)
        self._update_row(
            version,
            base_model=manifest["base_model"],
            state=state,
            artifact_format=manifest["artifact_format"],
            adapter_dir=str(self._version_path(version)),
            training_config={"training_backend": manifest.get("training_backend", "mock_local")},
            created_at=manifest["created_at"],
            num_samples=manifest.get("num_samples", 0),
            eval_report=report,
            metrics=manifest.get("training_metrics"),
            artifact_name=manifest.get("artifact_name", "adapter_model.safetensors"),
            metadata=manifest.get("metadata", {}),
        )

    def mark_failed_eval(self, version: str, report: dict[str, Any] | None = None) -> None:
        payload = report or {"comparison": "degraded", "recommendation": "keep_previous", "scores": {}}
        self.attach_eval_report(version, payload)

    def load(self, version: str = "latest") -> str:
        resolved = self._resolve_relative(version)
        path = self._version_path(resolved)
        if not path.exists():
            raise AdapterError(f"adapter version not found: {resolved}")
        return str(path)

    def list_version_records(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._rows()[:limit]

    def list_versions(self, limit: int = 20, workspace: str | None = None) -> str:
        if workspace and workspace != self.workspace:
            return AdapterStore(self.home, workspace).list_versions(limit=limit)
        rows = self.list_version_records(limit=limit)
        if not rows:
            return "No adapter versions found."
        current = self.current_latest_version()
        lines = []
        for row in rows:
            marker = "*" if row["version"] == current else " "
            lines.append(
                f"{marker} {row['version']}  state={row['state']}  samples={row['num_samples']}  format={row['artifact_format']}"
            )
        return "\n".join(lines)

    def promote(self, version: str, workspace: str | None = None) -> str:
        if workspace and workspace != self.workspace:
            return AdapterStore(self.home, workspace).promote(version)
        resolved = self._resolve_relative(version)
        manifest = self._read_manifest(resolved)
        state = manifest.get("state")
        if state in {"promoted", "archived"}:
            if state == "archived":
                raise AdapterError(f"cannot promote version {resolved} from state {state}")
        elif not can_promote_from(state):
            raise AdapterError(f"cannot promote version {resolved} from state {state}")

        current = self.current_latest_version()
        if current and current != resolved:
            current_manifest = self._read_manifest(current)
            current_manifest["state"] = "archived"
            self._write_manifest(current, current_manifest)
            self._update_row(
                current,
                base_model=current_manifest["base_model"],
                state="archived",
                artifact_format=current_manifest["artifact_format"],
                adapter_dir=str(self._version_path(current)),
                training_config={"training_backend": current_manifest.get("training_backend", "mock_local")},
                created_at=current_manifest["created_at"],
                num_samples=current_manifest.get("num_samples", 0),
                eval_report=current_manifest.get("eval_summary"),
                metrics=current_manifest.get("training_metrics"),
                artifact_name=current_manifest.get("artifact_name", "adapter_model.safetensors"),
                metadata=current_manifest.get("metadata", {}),
            )

        latest_link = self._latest_path()
        temp_link = self.root / ".latest.tmp"
        if temp_link.exists() or temp_link.is_symlink():
            temp_link.unlink()
        os.symlink(self._version_path(resolved), temp_link)
        os.replace(temp_link, latest_link)

        manifest["state"] = "promoted"
        manifest["promoted_at"] = utcnow_iso()
        self._write_manifest(resolved, manifest)
        self._update_row(
            resolved,
            base_model=manifest["base_model"],
            state="promoted",
            artifact_format=manifest["artifact_format"],
            adapter_dir=str(self._version_path(resolved)),
            training_config={"training_backend": manifest.get("training_backend", "mock_local")},
            created_at=manifest["created_at"],
            num_samples=manifest.get("num_samples", 0),
            eval_report=manifest.get("eval_summary"),
            metrics=manifest.get("training_metrics"),
            artifact_name=manifest.get("artifact_name", "adapter_model.safetensors"),
            metadata=manifest.get("metadata", {}),
        )
        return f"Promoted {resolved} to latest."

    def archive(self, version: str, workspace: str | None = None) -> str:
        if workspace and workspace != self.workspace:
            return AdapterStore(self.home, workspace).archive(version)
        resolved = self._resolve_relative(version)
        manifest = self._read_manifest(resolved)
        state = manifest.get("state")
        current = self.current_latest_version()
        if state == "archived":
            return f"Archived {resolved}."
        if current and current == resolved and state == "promoted":
            raise AdapterError(f"cannot archive latest promoted version {resolved}")
        if state not in self.allowed_states:
            raise AdapterError(f"cannot archive version {resolved} from state {state}")

        manifest["state"] = "archived"
        manifest["archived_at"] = utcnow_iso()
        self._write_manifest(resolved, manifest)
        self._update_row(
            resolved,
            base_model=manifest["base_model"],
            state="archived",
            artifact_format=manifest["artifact_format"],
            adapter_dir=str(self._version_path(resolved)),
            training_config={"training_backend": manifest.get("training_backend", "mock_local")},
            created_at=manifest["created_at"],
            num_samples=manifest.get("num_samples", 0),
            eval_report=manifest.get("eval_summary"),
            metrics=manifest.get("training_metrics"),
            artifact_name=manifest.get("artifact_name", "adapter_model.safetensors"),
            metadata=manifest.get("metadata", {}),
        )
        return f"Archived {resolved}."

    def rollback(self, version: str, workspace: str | None = None) -> str:
        if workspace and workspace != self.workspace:
            return AdapterStore(self.home, workspace).rollback(version)
        resolved = self._resolve_relative(version)
        return self.promote(resolved)


def create_adapter_store(workspace: str | None = None) -> AdapterStore:
    return AdapterStore(workspace=workspace or "user_default")


def get_adapter_store(workspace: str | None = None) -> AdapterStore:
    return create_adapter_store(workspace=workspace)
