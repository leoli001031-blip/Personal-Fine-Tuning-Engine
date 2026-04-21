"""Minimal local inference engine for Phase 0."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import json
import os

from .backends import plan_inference_backend
from .export import plan_export
from ..errors import AdapterError, InferenceError


@dataclass
class InferenceConfig:
    base_model: str
    adapter_path: str | None = None
    backend: str = "auto"
    quantization: str = "4bit"
    max_new_tokens: int = 128
    device: str = "auto"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _auto_local_base_model_enabled() -> bool:
    flag = str(os.environ.get("PFE_DISABLE_AUTO_LOCAL_BASE_MODEL", "")).strip().lower()
    return flag not in {"1", "true", "yes", "on"}


def _real_local_inference_env_enabled() -> bool:
    flag = str(os.environ.get("PFE_ENABLE_REAL_LOCAL_INFERENCE", "")).strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _metadata_real_local_enabled(metadata: object) -> bool:
    if not isinstance(metadata, dict):
        return False
    for key in ("enable_real_local", "real_local_inference", "use_real_local"):
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _default_local_base_model() -> str:
    env_override = os.environ.get("PFE_BASE_MODEL")
    if env_override:
        return env_override
    if not _auto_local_base_model_enabled():
        return "Qwen/Qwen2.5-3B-Instruct"
    candidate = _repo_root() / "models" / "Qwen3-4B"
    if candidate.exists():
        return str(candidate)
    return "Qwen/Qwen2.5-3B-Instruct"


def resolve_base_model_reference(base_model: str | None) -> str:
    candidate = str(base_model or "local-default")
    if candidate in {"local", "local-default", "base"}:
        return _default_local_base_model()
    return candidate


def _is_placeholder_adapter_artifact(adapter_path: str | None) -> bool:
    if not adapter_path:
        return False
    artifact_path = Path(adapter_path) / "adapter_model.safetensors"
    if not artifact_path.exists():
        return False
    try:
        prefix = artifact_path.read_bytes()[:16].lstrip()
    except Exception:
        return False
    return prefix.startswith(b"{") or prefix.startswith(b"[")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.strip()) // 4) if text.strip() else 0


def _strip_thinking_output(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"<think>\s*</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _clean_llama_cpp_output(text: str) -> str:
    cleaned = str(text or "").replace("\r\n", "\n").strip()
    if not cleaned:
        return ""
    cleaned_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if stripped.startswith("common_perf_print:"):
            continue
        if stripped.startswith("llama_memory_breakdown_print:"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = _strip_thinking_output(cleaned) or cleaned
    stop_markers = ("\nUSER:", "\nSYSTEM:", "\nASSISTANT:", "\nAI", "\nHuman:")
    for marker in stop_markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
    return cleaned.strip()


def _looks_like_local_path(value: str | None) -> bool:
    if value is None:
        return False
    candidate = Path(str(value)).expanduser()
    return candidate.exists() or str(value).startswith((".", "/", "~"))


def _safe_metadata_dict(metadata: object) -> dict[str, object]:
    return dict(metadata) if isinstance(metadata, dict) else {}


def _resolve_llama_cpp_runtime_binary() -> dict[str, object]:
    env_names = (
        "PFE_LLAMA_CPP_RUNTIME_BIN",
        "LLAMA_CPP_RUNTIME_BIN",
        "LLAMA_CPP_BIN",
        "LLAMA_CPP_PATH",
    )
    checked_paths: list[str] = []
    candidates: list[str] = []
    for env_name in env_names:
        value = os.environ.get(env_name)
        if not value:
            continue
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            candidates.extend(
                [
                    str(candidate / "llama-completion"),
                    str(candidate / "llama-cli"),
                    str(candidate / "build" / "bin" / "llama-completion"),
                    str(candidate / "build" / "bin" / "llama-cli"),
                ]
            )
        else:
            candidates.append(str(candidate))
    repo_root = _repo_root()
    candidates.extend(
        [
            str(repo_root / "tools" / "llama.cpp" / "build-cpu" / "bin" / "llama-completion"),
            str(repo_root / "tools" / "llama.cpp" / "build-cpu" / "bin" / "llama-cli"),
            str(repo_root / "tools" / "llama.cpp" / "build" / "bin" / "llama-completion"),
            str(repo_root / "tools" / "llama.cpp" / "build" / "bin" / "llama-cli"),
        ]
    )
    for name in ("llama-completion", "llama-cli"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(Path(candidate).expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)
        checked_paths.append(normalized)
        path = Path(normalized)
        if path.exists() and os.access(path, os.X_OK):
            return {
                "available": True,
                "path": normalized,
                "tool": path.name,
                "checked_paths": checked_paths,
            }
    return {
        "available": False,
        "path": None,
        "tool": None,
        "checked_paths": checked_paths,
        "reason": "llama.cpp runtime binary not found; set PFE_LLAMA_CPP_RUNTIME_BIN or build llama-completion/llama-cli",
    }


def _resolve_base_gguf_path(resolved_base_model: str) -> dict[str, object]:
    env_names = (
        "PFE_LLAMA_CPP_BASE_GGUF",
        "LLAMA_CPP_BASE_GGUF",
    )
    checked_paths: list[str] = []
    candidates: list[str] = []
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            candidates.append(str(Path(value).expanduser()))
    base_path = Path(str(resolved_base_model)).expanduser()
    if base_path.exists() and base_path.is_file() and base_path.suffix == ".gguf":
        candidates.append(str(base_path))
    if base_path.exists() and base_path.is_dir():
        candidates.extend(
            [
                str(base_path / "gguf" / "ggml-model-f16.gguf"),
                str(base_path / "gguf" / "ggml-model-q8_0.gguf"),
                str(base_path / "gguf" / "model.gguf"),
                str(base_path / "ggml-model-f16.gguf"),
                str(base_path / "model.gguf"),
            ]
        )
        candidates.extend(str(path) for path in sorted(base_path.glob("*.gguf")))
        gguf_dir = base_path / "gguf"
        if gguf_dir.exists():
            candidates.extend(str(path) for path in sorted(gguf_dir.glob("*.gguf")))
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(Path(candidate).expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)
        checked_paths.append(normalized)
        path = Path(normalized)
        if path.exists() and path.is_file():
            return {
                "available": True,
                "path": normalized,
                "checked_paths": checked_paths,
            }
    return {
        "available": False,
        "path": None,
        "checked_paths": checked_paths,
        "reason": "base GGUF model not found; set PFE_LLAMA_CPP_BASE_GGUF or convert the local base model to GGUF first",
    }


class InferenceEngine:
    """Template-driven inference placeholder with artifact checks."""

    _runtime_cache: dict[tuple[str, str | None, str], dict[str, object]] = {}

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.adapter_manifest: dict | None = None
        self.last_generation_info: dict[str, object] = {
            "served_by": "mock",
            "path": "template_fallback",
        }
        self.backend_plan: dict[str, object] = plan_inference_backend(
            requested_backend=config.backend,
            artifact_format=None,
            manifest=None,
            device=config.device,
        ).to_dict()
        self.export_plan: dict[str, object] | None = None
        if config.adapter_path:
            self._load_manifest(config.adapter_path)

    def _load_manifest(self, adapter_path: str) -> None:
        manifest_path = Path(adapter_path) / "adapter_manifest.json"
        if not manifest_path.exists():
            raise AdapterError(f"adapter manifest missing at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("base_model"):
            self.config.base_model = str(manifest["base_model"])
        backend_decision = plan_inference_backend(
            requested_backend=self.config.backend,
            artifact_format=manifest.get("artifact_format"),
            manifest=manifest,
            device=self.config.device,
        )
        export_plan = plan_export(
            target_backend=backend_decision.selected_backend,
            source_artifact_format=manifest.get("artifact_format"),
            workspace=manifest.get("workspace"),
            adapter_dir=manifest.get("adapter_dir") or str(Path(adapter_path)),
            source_adapter_version=manifest.get("version"),
            source_model=manifest.get("base_model"),
            training_run_id=manifest.get("training_run_id"),
            num_samples=manifest.get("num_samples"),
        )
        self.backend_plan = backend_decision.to_dict()
        self.export_plan = export_plan.to_dict()
        if backend_decision.requires_export:
            raise AdapterError(
                "selected inference backend requires an exported artifact first; "
                f"backend={backend_decision.selected_backend}, "
                f"required_format={backend_decision.required_artifact_format}, "
                "do not load LoRA safetensors directly on the llama.cpp path"
            )
        manifest_updates = dict(export_plan.manifest_updates)
        export_metadata = manifest_updates.pop("export", None)
        metadata = dict(manifest.get("metadata") or {})
        if export_metadata is not None:
            metadata["export"] = export_metadata
        manifest.update(manifest_updates)
        if metadata:
            manifest["metadata"] = metadata
        self.adapter_manifest = manifest

    def _real_local_inference_enabled(self, metadata: object = None) -> bool:
        return _real_local_inference_env_enabled() or _metadata_real_local_enabled(metadata)

    def _select_device(self) -> str:
        if self.config.device in {"cpu", "mps", "cuda"}:
            return self.config.device
        try:
            torch = importlib.import_module("torch")
            if bool(getattr(torch.backends, "mps", None)) and bool(torch.backends.mps.is_available()):
                return "mps"
            if bool(getattr(torch, "cuda", None)) and bool(torch.cuda.is_available()):
                return "cuda"
        except Exception:
            return "cpu"
        return "cpu"

    @staticmethod
    def _build_prompt_text(messages: list[dict[str, object]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = str(message.get("role") or "user").upper()
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            parts.append(f"{role}: {content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def _build_runtime_cache_key(self, *, resolved_base_model: str, adapter_path: str | None, device: str) -> tuple[str, str | None, str]:
        effective_adapter = None if _is_placeholder_adapter_artifact(adapter_path) else adapter_path
        return (resolved_base_model, effective_adapter, device)

    def _resolve_llama_cpp_artifact_path(self) -> dict[str, object]:
        checked_paths: list[str] = []
        manifest = self.adapter_manifest or {}
        metadata = _safe_metadata_dict(manifest.get("metadata"))
        artifact_summary = _safe_metadata_dict(metadata.get("export_artifact_summary"))
        export_metadata = _safe_metadata_dict(metadata.get("export"))
        candidates: list[str] = []
        for value in (
            artifact_summary.get("path"),
            export_metadata.get("artifact", {}).get("path") if isinstance(export_metadata.get("artifact"), dict) else None,
            export_metadata.get("artifact_path"),
            manifest.get("artifact_path"),
        ):
            if value:
                candidates.append(str(value))
        if self.config.adapter_path:
            candidates.append(str(Path(self.config.adapter_path) / "gguf_merged" / "adapter_model.gguf"))
        seen: set[str] = set()
        for candidate in candidates:
            normalized = str(Path(candidate).expanduser())
            if normalized in seen:
                continue
            seen.add(normalized)
            checked_paths.append(normalized)
            path = Path(normalized)
            if path.exists() and path.is_file():
                return {
                    "available": True,
                    "path": normalized,
                    "checked_paths": checked_paths,
                }
        return {
            "available": False,
            "path": None,
            "checked_paths": checked_paths,
            "reason": "adapter GGUF artifact not found under manifest metadata or adapter directory",
        }

    def _llama_cpp_command(
        self,
        *,
        runtime_binary: str,
        model_path: str,
        prompt_text: str,
        max_tokens: int,
        temperature: float,
        adapter_gguf_path: str | None = None,
    ) -> list[str]:
        threads = max(1, int(str(os.environ.get("PFE_LLAMA_CPP_THREADS", "4")).strip() or "4"))
        ctx_size = max(256, int(str(os.environ.get("PFE_LLAMA_CPP_CTX_SIZE", "2048")).strip() or "2048"))
        command = [
            runtime_binary,
            "-m",
            model_path,
            "-no-cnv",
            "--simple-io",
            "--no-perf",
            "--no-display-prompt",
            "--no-warmup",
            "-p",
            prompt_text,
            "-n",
            str(max(1, min(max_tokens, 128))),
            "--temp",
            str(max(0.0, temperature)),
            "-s",
            "42",
            "-t",
            str(threads),
            "-c",
            str(ctx_size),
        ]
        if adapter_gguf_path:
            command.extend(["--lora", adapter_gguf_path])
        return command

    def _generate_llama_cpp_response(
        self,
        messages: list[dict],
        *,
        resolved_base_model: str | None = None,
        **kwargs,
    ) -> dict[str, object]:
        resolved_base_model = resolved_base_model or resolve_base_model_reference(self.config.base_model)
        runtime_resolution = _resolve_llama_cpp_runtime_binary()
        if not runtime_resolution.get("available"):
            raise InferenceError(str(runtime_resolution.get("reason") or "llama.cpp runtime is unavailable"))
        base_gguf = _resolve_base_gguf_path(resolved_base_model)
        if not base_gguf.get("available"):
            raise InferenceError(str(base_gguf.get("reason") or "base GGUF model is unavailable"))

        adapter_resolution = self._resolve_llama_cpp_artifact_path() if self.adapter_manifest else {
            "available": False,
            "path": None,
            "checked_paths": [],
            "reason": "no adapter manifest loaded",
        }
        adapter_gguf_path = str(adapter_resolution.get("path")) if adapter_resolution.get("available") else None
        prompt_text = self._build_prompt_text(messages)
        temperature = float(kwargs.get("temperature") or 0.7)
        max_tokens = int(kwargs.get("max_tokens") or kwargs.get("max_new_tokens") or self.config.max_new_tokens)
        command = self._llama_cpp_command(
            runtime_binary=str(runtime_resolution["path"]),
            model_path=str(base_gguf["path"]),
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            adapter_gguf_path=adapter_gguf_path,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=str(_repo_root()),
                capture_output=True,
                text=True,
                check=False,
                timeout=max(30, max_tokens * 6),
            )
        except subprocess.TimeoutExpired as exc:
            raise InferenceError(f"llama.cpp runtime timed out: {exc}") from exc
        except Exception as exc:
            raise InferenceError(f"llama.cpp runtime failed to start: {exc.__class__.__name__}: {exc}") from exc
        if completed.returncode != 0:
            stderr_lines = [line.strip() for line in (completed.stderr or "").splitlines() if line.strip()]
            stderr_summary = stderr_lines[-1] if stderr_lines else f"exit code {completed.returncode}"
            raise InferenceError(f"llama.cpp runtime failed: {stderr_summary}")
        raw_text = str(completed.stdout or "").strip()
        text = _clean_llama_cpp_output(raw_text) or raw_text
        if not text:
            raise InferenceError("llama.cpp runtime produced an empty response")
        return {
            "text": text,
            "served_by": "local",
            "runtime_path": "llama_cpp",
            "resolved_base_model": resolved_base_model,
            "base_gguf_path": base_gguf.get("path"),
            "runtime_binary": runtime_resolution.get("path"),
            "adapter_loaded": bool(adapter_gguf_path),
            "adapter_requested": bool(self.adapter_manifest),
            "adapter_reason": None if adapter_gguf_path else adapter_resolution.get("reason"),
            "adapter_gguf_path": adapter_gguf_path,
            "raw_text": raw_text,
            "thinking_stripped": text != raw_text,
            "command": command,
        }

    def _create_runtime_bundle(
        self,
        *,
        resolved_base_model: str,
        adapter_path: str | None,
        use_cache: bool = True,
    ) -> dict[str, object]:
        device = self._select_device()
        cache_key = self._build_runtime_cache_key(
            resolved_base_model=resolved_base_model,
            adapter_path=adapter_path,
            device=device,
        )
        if use_cache:
            cached = self._runtime_cache.get(cache_key)
            if cached is not None:
                return cached

        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
            raise InferenceError("torch and transformers are required for real local inference")

        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        tokenizer_cls = getattr(transformers, "AutoTokenizer", None)
        model_cls = getattr(transformers, "AutoModelForCausalLM", None)
        if tokenizer_cls is None or model_cls is None:
            raise InferenceError("transformers runtime is missing AutoTokenizer or AutoModelForCausalLM")

        local_only = Path(resolved_base_model).expanduser().exists()
        if not local_only:
            raise InferenceError(
                "real local inference requires a local base model path; "
                f"resolved_base_model={resolved_base_model}"
            )
        load_kwargs: dict[str, object] = {"local_files_only": local_only}
        if importlib.util.find_spec("accelerate") is not None:
            load_kwargs["low_cpu_mem_usage"] = True
        if device in {"mps", "cuda"}:
            load_kwargs["torch_dtype"] = torch.float16

        tokenizer = tokenizer_cls.from_pretrained(resolved_base_model, **load_kwargs)
        model = model_cls.from_pretrained(resolved_base_model, **load_kwargs)

        adapter_loaded = False
        adapter_reason = None
        placeholder_adapter = _is_placeholder_adapter_artifact(adapter_path)
        effective_adapter_path = None if placeholder_adapter else adapter_path
        if effective_adapter_path:
            if importlib.util.find_spec("peft") is None:
                adapter_reason = "peft is unavailable for adapter loading"
            else:
                peft = importlib.import_module("peft")
                peft_model_cls = getattr(peft, "PeftModel", None)
                if peft_model_cls is None:
                    adapter_reason = "peft runtime is missing PeftModel"
                else:
                    try:
                        model = peft_model_cls.from_pretrained(model, effective_adapter_path, local_files_only=True)
                        adapter_loaded = True
                    except Exception as exc:
                        adapter_reason = f"adapter load failed: {exc.__class__.__name__}: {exc}"
        elif placeholder_adapter:
            adapter_reason = "adapter artifact is still a placeholder and cannot be loaded for inference"

        if device != "cpu":
            try:
                model = model.to(device)
            except Exception:
                device = "cpu"
                model = model.to(device)
        model.eval()

        bundle = {
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "resolved_base_model": resolved_base_model,
            "adapter_loaded": adapter_loaded,
            "adapter_path": effective_adapter_path,
            "adapter_reason": adapter_reason,
            "torch": torch,
        }
        if use_cache:
            self._runtime_cache[cache_key] = bundle
        return bundle

    def _load_runtime_bundle(self, *, resolved_base_model: str, adapter_path: str | None) -> dict[str, object]:
        return self._create_runtime_bundle(
            resolved_base_model=resolved_base_model,
            adapter_path=adapter_path,
            use_cache=True,
        )

    def _load_uncached_runtime_bundle(self, *, resolved_base_model: str, adapter_path: str | None = None) -> dict[str, object]:
        return self._create_runtime_bundle(
            resolved_base_model=resolved_base_model,
            adapter_path=adapter_path,
            use_cache=False,
        )

    def _attach_adapter_to_runtime_bundle(
        self,
        runtime_bundle: dict[str, object],
        *,
        adapter_path: str | None,
    ) -> dict[str, object]:
        adapted_bundle = dict(runtime_bundle)
        placeholder_adapter = _is_placeholder_adapter_artifact(adapter_path)
        effective_adapter_path = None if placeholder_adapter else adapter_path
        adapted_bundle["adapter_loaded"] = False
        adapted_bundle["adapter_path"] = effective_adapter_path
        adapted_bundle["adapter_reason"] = None

        if not effective_adapter_path:
            if placeholder_adapter:
                adapted_bundle["adapter_reason"] = (
                    "adapter artifact is still a placeholder and cannot be loaded for inference"
                )
            return adapted_bundle

        if importlib.util.find_spec("peft") is None:
            adapted_bundle["adapter_reason"] = "peft is unavailable for adapter loading"
            return adapted_bundle

        peft = importlib.import_module("peft")
        peft_model_cls = getattr(peft, "PeftModel", None)
        if peft_model_cls is None:
            adapted_bundle["adapter_reason"] = "peft runtime is missing PeftModel"
            return adapted_bundle

        try:
            model = peft_model_cls.from_pretrained(
                runtime_bundle["model"],
                effective_adapter_path,
                local_files_only=True,
            )
            adapted_bundle["model"] = model
            adapted_bundle["adapter_loaded"] = True
        except Exception as exc:
            adapted_bundle["adapter_reason"] = f"adapter load failed: {exc.__class__.__name__}: {exc}"
        return adapted_bundle

    def _generate_real_response(
        self,
        messages: list[dict],
        *,
        runtime_bundle: dict[str, object] | None = None,
        resolved_base_model: str | None = None,
        **kwargs,
    ) -> dict[str, object]:
        resolved_base_model = resolved_base_model or resolve_base_model_reference(self.config.base_model)
        bundle = runtime_bundle or self._load_runtime_bundle(
            resolved_base_model=resolved_base_model,
            adapter_path=self.config.adapter_path,
        )
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        torch = bundle["torch"]
        device = str(bundle["device"])
        prompt_text = self._build_prompt_text(messages)

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prompt_text = self._build_prompt_text(messages)
            except Exception:
                prompt_text = self._build_prompt_text(messages)

        encoded = tokenizer(prompt_text, return_tensors="pt")
        if hasattr(encoded, "to"):
            encoded = encoded.to(device)
        elif isinstance(encoded, dict):
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
        input_ids = encoded["input_ids"]
        input_length = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else 0
        temperature = float(kwargs.get("temperature") or 0.7)
        max_new_tokens = int(kwargs.get("max_tokens") or kwargs.get("max_new_tokens") or self.config.max_new_tokens)
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
        generation_kwargs = {
            "max_new_tokens": max(1, min(max_new_tokens, 128)),
            "do_sample": temperature > 0,
            "temperature": max(0.1, temperature) if temperature > 0 else 1.0,
            "pad_token_id": pad_token_id,
        }
        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_kwargs)
        generated_ids = output_ids[0][input_length:] if input_length else output_ids[0]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        text = _strip_thinking_output(raw_text) or raw_text
        if not text:
            raise InferenceError("real local inference produced an empty response")
        return {
            "text": text,
            "served_by": "local",
            "runtime_path": "real_local",
            "device": device,
            "resolved_base_model": resolved_base_model,
            "adapter_loaded": bool(bundle.get("adapter_loaded", False)),
            "adapter_requested": bool(self.adapter_manifest or bundle.get("adapter_path")),
            "adapter_reason": bundle.get("adapter_reason"),
            "raw_text": raw_text,
            "thinking_stripped": text != raw_text,
        }

    def generate(self, messages: list[dict], **kwargs) -> str:
        if not messages:
            raise InferenceError("messages cannot be empty")
        last_user = next((message.get("content", "") for message in reversed(messages) if message.get("role") == "user"), "")
        if not last_user:
            last_user = messages[-1].get("content", "")
        metadata = kwargs.get("metadata")
        real_local_enabled = self._real_local_inference_enabled(metadata)

        if real_local_enabled:
            resolved_base_model = resolve_base_model_reference(self.config.base_model)
            primary_runtime = (
                "llama_cpp"
                if str(self.backend_plan.get("selected_backend", "")).lower() == "llama_cpp"
                else "transformers"
            )
            runtime_attempts = []
            if primary_runtime == "llama_cpp":
                runtime_attempts.append(("llama_cpp", lambda: self._generate_llama_cpp_response(messages, resolved_base_model=resolved_base_model, **kwargs)))
            if primary_runtime != "llama_cpp":
                runtime_attempts.append(("transformers", lambda: self._generate_real_response(messages, resolved_base_model=resolved_base_model, **kwargs)))
                runtime_attempts.append(("llama_cpp", lambda: self._generate_llama_cpp_response(messages, resolved_base_model=resolved_base_model, **kwargs)))

            attempted_failures: list[str] = []
            for runtime_name, runner in runtime_attempts:
                try:
                    response = runner()
                    self.last_generation_info = dict(response)
                    self.last_generation_info["real_local_enabled"] = True
                    if attempted_failures:
                        self.last_generation_info["previous_runtime_failures"] = attempted_failures
                    return str(response["text"])
                except Exception as exc:
                    attempted_failures.append(f"{runtime_name}: {exc.__class__.__name__}: {exc}")
            self.last_generation_info = {
                "served_by": "mock",
                "path": "template_fallback",
                "fallback_reason": attempted_failures[0] if attempted_failures else "real local inference failed",
                "previous_runtime_failures": attempted_failures,
                "resolved_base_model": resolved_base_model,
                "adapter_requested": bool(self.adapter_manifest),
                "adapter_placeholder": _is_placeholder_adapter_artifact(self.config.adapter_path),
                "real_local_enabled": True,
            }
        else:
            self.last_generation_info = {
                "served_by": "mock",
                "path": "template_fallback",
                "fallback_reason": (
                    "real local inference is disabled; enable it with "
                    "metadata.enable_real_local=true or PFE_ENABLE_REAL_LOCAL_INFERENCE=1"
                ),
                "resolved_base_model": resolve_base_model_reference(self.config.base_model),
                "adapter_requested": bool(self.adapter_manifest),
                "adapter_placeholder": _is_placeholder_adapter_artifact(self.config.adapter_path),
                "real_local_enabled": False,
            }

        if self.adapter_manifest:
            version = self.adapter_manifest.get("version", "latest")
            return (
                f"[adapter:{version}] 我听到你的重点是：{last_user}。"
                " 我会用更稳定、温和、偏个性化的方式回应，并优先保留上下文连续性。"
            )
        return f"[base] 关于“{last_user}”，我会先给出一个通用、稳妥的基础回答。"

    def generate_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        text = self.generate(messages, **kwargs)
        for index in range(0, len(text), 24):
            yield text[index : index + 24]

    def swap_adapter(self, adapter_path: str) -> None:
        self.config.adapter_path = adapter_path
        self._load_manifest(adapter_path)

    def unload(self) -> None:
        self.adapter_manifest = None

    def status(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "backend": str(self.backend_plan.get("selected_backend", self.config.backend)),
            "healthy": True,
            "requested_backend": self.backend_plan.get("requested_backend", self.config.backend),
            "requires_export": self.backend_plan.get("requires_export", False),
            "backend_plan": self.backend_plan,
            "served_by": self.last_generation_info.get("served_by", "mock"),
            "resolved_base_model": resolve_base_model_reference(self.config.base_model),
            "runtime_path": self.last_generation_info.get("runtime_path", self.last_generation_info.get("path")),
            "real_local_enabled": self.last_generation_info.get("real_local_enabled", _real_local_inference_env_enabled()),
        }
        if self.export_plan is not None:
            payload["export_plan"] = self.export_plan
        if self.adapter_manifest is not None:
            payload["adapter_version"] = self.adapter_manifest.get("version")
            payload["artifact_format"] = self.adapter_manifest.get("artifact_format")
            payload["adapter_requested"] = True
        if self.last_generation_info:
            payload["generation"] = dict(self.last_generation_info)
        return payload
