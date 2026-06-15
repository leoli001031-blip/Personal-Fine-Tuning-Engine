from __future__ import annotations

import time
from typing import Any, Mapping


API_MODEL_PARAMETER = "local"
API_MODEL_ALIASES = ("local", "local-default", "base")


def runtime_host_port(
    headers: Mapping[str, str],
    runtime_probe: Mapping[str, Any],
    *,
    default_host: str = "127.0.0.1",
    default_port: int = 8921,
) -> tuple[str, int]:
    host_header = str(headers.get("host", "")).strip()
    if host_header:
        host_part = host_header.rsplit(":", 1)[0]
        port_part = host_header.rsplit(":", 1)[1] if ":" in host_header else ""
        if host_part:
            try:
                return host_part, int(port_part or default_port)
            except ValueError:
                return host_part, default_port
    runner = runtime_probe.get("runner") or {}
    kwargs = runner.get("kwargs") if isinstance(runner, Mapping) else {}
    if isinstance(kwargs, Mapping):
        host = str(kwargs.get("host") or default_host)
        try:
            port = int(kwargs.get("port") or default_port)
        except (TypeError, ValueError):
            port = default_port
        return host, port
    return default_host, default_port


def api_key_required_from_auth_mode(auth_mode: str | None) -> bool:
    return str(auth_mode or "").strip().lower() in {
        "api_key_required",
        "api_key_only",
        "key_required",
        "required",
    }


def build_openai_chat_api_contract(base_url: str, *, api_key_required: bool) -> dict[str, Any]:
    chat_completions_url = f"{base_url}/v1/chat/completions"
    return {
        "kind": "openai_chat_completions",
        "method": "POST",
        "chat_completions_url": chat_completions_url,
        "model_parameter": API_MODEL_PARAMETER,
        "model_aliases": list(API_MODEL_ALIASES),
        "content_type": "application/json",
        "auth_header": "Authorization: Bearer $PFE_API_KEY" if api_key_required else None,
        "request_body": {
            "model": API_MODEL_PARAMETER,
            "messages": [{"role": "user", "content": "hello"}],
        },
    }


def build_runtime_payload(
    *,
    headers: Mapping[str, str],
    runtime_probe: Mapping[str, Any],
    workspace: str,
    provider: str,
    allow_remote_access: bool,
    privacy_mode: str,
    auth_mode: str,
    started_at: float,
    now: float | None = None,
) -> dict[str, Any]:
    host, port = runtime_host_port(headers, runtime_probe)
    base_url = f"http://{host}:{port}"
    api_key_required = api_key_required_from_auth_mode(auth_mode)
    api_contract = build_openai_chat_api_contract(base_url, api_key_required=api_key_required)
    local_only = not bool(allow_remote_access)
    current_time = time.time() if now is None else now
    return {
        "status": "passed",
        "workspace": workspace,
        "provider": provider,
        "host": host,
        "port": port,
        "base_url": base_url,
        "studio_url": f"{base_url}/studio",
        "web_url": f"{base_url}/",
        "api_url": api_contract["chat_completions_url"],
        "api": api_contract,
        "dashboard_url": f"{base_url}/dashboard",
        "privacy_mode": privacy_mode,
        "auth_mode": auth_mode,
        "api_key_required": api_key_required,
        "access_scope": "仅本机" if local_only else "允许远程",
        "local_only": local_only,
        "started_at": started_at,
        "uptime_seconds": round(max(0.0, current_time - started_at), 3),
    }


__all__ = [
    "API_MODEL_ALIASES",
    "API_MODEL_PARAMETER",
    "api_key_required_from_auth_mode",
    "build_openai_chat_api_contract",
    "build_runtime_payload",
    "runtime_host_port",
]
