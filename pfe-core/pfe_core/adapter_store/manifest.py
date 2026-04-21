"""Canonical adapter manifest helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .lifecycle import AdapterArtifactFormat, llama_cpp_requires_merged_gguf

ADAPTER_MODEL_FILENAME = "adapter_model.safetensors"
ADAPTER_CONFIG_FILENAME = "adapter_config.json"
ADAPTER_MANIFEST_FILENAME = "adapter_manifest.json"
TRAINING_META_FILENAME = "training_meta.json"
EVAL_REPORT_FILENAME = "eval_report.json"
LATEST_LINK_NAME = "latest"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AdapterManifest:
    version: str = ""
    workspace: str = "user_default"
    base_model: str = ""
    created_at: datetime = field(default_factory=utc_now)
    state: str = "training"
    num_samples: int = 0
    adapter_dir: Optional[str] = None
    artifact_format: AdapterArtifactFormat = AdapterArtifactFormat.PEFT_LORA
    training_backend: str = "peft"
    inference_backend: str = "transformers"
    requires_export: bool = False
    source_adapter_version: Optional[str] = None
    source_model: Optional[str] = None
    training_run_id: Optional[str] = None
    manifest_version: int = 1
    artifact_name: str = ADAPTER_MODEL_FILENAME
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def standard_paths(self) -> dict[str, str]:
        return {
            "model": ADAPTER_MODEL_FILENAME,
            "config": ADAPTER_CONFIG_FILENAME,
            "manifest": ADAPTER_MANIFEST_FILENAME,
            "training_meta": TRAINING_META_FILENAME,
            "eval_report": EVAL_REPORT_FILENAME,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_format"] = self.artifact_format.value
        payload["created_at"] = self.created_at.isoformat()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterManifest":
        artifact_format = data.get("artifact_format", AdapterArtifactFormat.PEFT_LORA)
        if not isinstance(artifact_format, AdapterArtifactFormat):
            artifact_format = AdapterArtifactFormat(artifact_format)
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            version=data.get("version", ""),
            workspace=data.get("workspace", "user_default"),
            base_model=data.get("base_model", ""),
            created_at=created_at or utc_now(),
            state=data.get("state", "training"),
            num_samples=int(data.get("num_samples", 0)),
            adapter_dir=data.get("adapter_dir"),
            artifact_format=artifact_format,
            training_backend=data.get("training_backend", "peft"),
            inference_backend=data.get("inference_backend", "transformers"),
            requires_export=bool(data.get("requires_export", False)),
            source_adapter_version=data.get("source_adapter_version"),
            source_model=data.get("source_model"),
            training_run_id=data.get("training_run_id"),
            manifest_version=int(data.get("manifest_version", 1)),
            artifact_name=data.get("artifact_name", ADAPTER_MODEL_FILENAME),
            metadata=data.get("metadata", {}) or {},
        )

    @classmethod
    def from_json(cls, raw: str) -> "AdapterManifest":
        return cls.from_dict(json.loads(raw))

    @classmethod
    def from_path(cls, path: str | Path) -> "AdapterManifest":
        return cls.from_json(Path(path).expanduser().read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> Path:
        manifest_path = Path(path).expanduser()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(self.to_json(), encoding="utf-8")
        return manifest_path

    def requires_gguf_merged(self, target_backend: str | None = None) -> bool:
        return llama_cpp_requires_merged_gguf(target_backend)

    def validate_artifact_for_backend(self, target_backend: str | None = None) -> None:
        if target_backend and self.requires_gguf_merged(target_backend) and self.artifact_format != AdapterArtifactFormat.GGUF_MERGED:
            raise ValueError("llama.cpp targets must use merged GGUF artifacts")


class AdapterManifestSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    version: str = ""
    workspace: str = "user_default"
    base_model: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    state: str = "training"
    num_samples: int = 0
    adapter_dir: Optional[str] = None
    artifact_format: AdapterArtifactFormat = AdapterArtifactFormat.PEFT_LORA
    training_backend: str = "peft"
    inference_backend: str = "transformers"
    requires_export: bool = False
    source_adapter_version: Optional[str] = None
    source_model: Optional[str] = None
    training_run_id: Optional[str] = None
    manifest_version: int = 1
    artifact_name: str = ADAPTER_MODEL_FILENAME
    metadata: dict[str, Any] = Field(default_factory=dict)

    def validate_artifact_for_backend(self, target_backend: str | None = None) -> "AdapterManifestSchema":
        if target_backend and llama_cpp_requires_merged_gguf(target_backend) and self.artifact_format != AdapterArtifactFormat.GGUF_MERGED:
            raise ValueError("llama.cpp targets must use merged GGUF artifacts")
        return self
