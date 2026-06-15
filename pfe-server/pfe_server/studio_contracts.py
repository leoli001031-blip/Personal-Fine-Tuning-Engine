from __future__ import annotations

import json
import time
from typing import Any, Mapping


API_MODEL_PARAMETER = "local"
API_MODEL_ALIASES = ("local", "local-default", "base")
FEEDBACK_ACTIONS = ("accept", "reject", "edit", "regenerate", "delete")
RESPONSE_ID_FIELDS = ("session_id", "request_id")


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
    feedback_url = f"{base_url}/pfe/feedback"
    return {
        "kind": "openai_chat_completions",
        "method": "POST",
        "chat_completions_url": chat_completions_url,
        "feedback_url": feedback_url,
        "feedback_method": "POST",
        "model_parameter": API_MODEL_PARAMETER,
        "model_aliases": list(API_MODEL_ALIASES),
        "content_type": "application/json",
        "auth_header": "Authorization: Bearer $PFE_API_KEY" if api_key_required else None,
        "response_id_fields": list(RESPONSE_ID_FIELDS),
        "feedback_actions": list(FEEDBACK_ACTIONS),
        "request_body": {
            "model": API_MODEL_PARAMETER,
            "messages": [{"role": "user", "content": "hello"}],
        },
        "feedback_body": {
            "session_id": "<session_id from chat response>",
            "request_id": "<request_id from chat response>",
            "action": "accept",
        },
    }


def _json_snippet(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_handoff_javascript_snippet(payload: Mapping[str, Any]) -> str:
    urls = payload.get("urls") if isinstance(payload.get("urls"), Mapping) else {}
    model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
    api = payload.get("api") if isinstance(payload.get("api"), Mapping) else {}
    chat_url = str(urls.get("api") or "")
    feedback_url = str(urls.get("feedback") or api.get("feedback_url") or "")
    content_type = str(api.get("content_type") or "application/json")
    model_parameter = str(model.get("api_parameter") or api.get("model_parameter") or API_MODEL_PARAMETER)
    headers = {"content-type": content_type}
    if api.get("auth_header"):
        headers["authorization"] = "Bearer <PFE_API_KEY>"
    return "\n".join(
        [
            f"const headers = {_json_snippet(headers)};",
            f'const chat = await fetch("{chat_url}", {{',
            '  method: "POST",',
            "  headers,",
            f'  body: JSON.stringify({{"model":"{model_parameter}","messages":[{{"role":"user","content":"hello"}}]}}),',
            "}).then((response) => response.json());",
            "const { session_id, request_id } = chat;",
            f'await fetch("{feedback_url}", {{',
            '  method: "POST",',
            "  headers,",
            '  body: JSON.stringify({ session_id, request_id, action: "accept" }),',
            "});",
        ]
    )


def build_handoff_python_snippet(payload: Mapping[str, Any]) -> str:
    urls = payload.get("urls") if isinstance(payload.get("urls"), Mapping) else {}
    model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
    api = payload.get("api") if isinstance(payload.get("api"), Mapping) else {}
    chat_url = str(urls.get("api") or "")
    feedback_url = str(urls.get("feedback") or api.get("feedback_url") or "")
    content_type = str(api.get("content_type") or "application/json")
    model_parameter = str(model.get("api_parameter") or api.get("model_parameter") or API_MODEL_PARAMETER)
    headers = {"content-type": content_type}
    if api.get("auth_header"):
        headers["authorization"] = "Bearer <PFE_API_KEY>"
    return "\n".join(
        [
            "import requests",
            "",
            f"headers = {_json_snippet(headers)}",
            f'chat = requests.post("{chat_url}", headers=headers, json={{',
            f'    "model": "{model_parameter}",',
            '    "messages": [{"role": "user", "content": "hello"}],',
            "}).json()",
            'session_id = chat["session_id"]',
            'request_id = chat["request_id"]',
            f'requests.post("{feedback_url}", headers=headers, json={{',
            '    "session_id": session_id,',
            '    "request_id": request_id,',
            '    "action": "accept",',
            "})",
        ]
    )


def build_handoff_closed_loop(payload: Mapping[str, Any]) -> dict[str, Any]:
    urls = payload.get("urls") if isinstance(payload.get("urls"), Mapping) else {}
    model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
    api = payload.get("api") if isinstance(payload.get("api"), Mapping) else {}
    return {
        "summary": "Chat gives the answer; feedback keeps the personalization loop alive.",
        "chat": {
            "method": api.get("method") or "POST",
            "url": urls.get("api"),
            "model_parameter": model.get("api_parameter") or api.get("model_parameter") or API_MODEL_PARAMETER,
        },
        "feedback": {
            "method": api.get("feedback_method") or "POST",
            "url": urls.get("feedback") or api.get("feedback_url"),
            "actions": list(api.get("feedback_actions") or FEEDBACK_ACTIONS),
        },
        "required_response_fields": list(api.get("response_id_fields") or RESPONSE_ID_FIELDS),
        "client_must_store": [
            "session_id",
            "request_id",
            "assistant_message",
            "original_user_message",
        ],
        "flow": [
            "POST chat request",
            "Store session_id and request_id with the rendered answer",
            "POST user behavior to feedback using the same ids",
            "PFE turns feedback into signals for train/eval/promote",
        ],
    }


def build_handoff_copy_text(payload: Mapping[str, Any]) -> str:
    urls = payload.get("urls") if isinstance(payload.get("urls"), Mapping) else {}
    model = payload.get("model") if isinstance(payload.get("model"), Mapping) else {}
    version = payload.get("version") if isinstance(payload.get("version"), Mapping) else {}
    api = payload.get("api") if isinstance(payload.get("api"), Mapping) else {}
    closed_loop = payload.get("closed_loop") if isinstance(payload.get("closed_loop"), Mapping) else {}
    snippets = payload.get("snippets") if isinstance(payload.get("snippets"), Mapping) else {}
    feedback = closed_loop.get("feedback") if isinstance(closed_loop.get("feedback"), Mapping) else {}
    lines = [
        "PFE closed-loop handoff",
        f"Web: {urls.get('web') or '-'}",
        f"Chat API: {urls.get('api') or '-'}",
        f"Feedback API: {urls.get('feedback') or feedback.get('url') or api.get('feedback_url') or '-'}",
        f"Model parameter: {model.get('api_parameter') or api.get('model_parameter') or API_MODEL_PARAMETER}",
        f"Selected model: {model.get('selected') or '-'}",
        f"Keep per answer: {', '.join(list(closed_loop.get('required_response_fields') or RESPONSE_ID_FIELDS))}",
        f"Report actions: {', '.join(list(feedback.get('actions') or api.get('feedback_actions') or FEEDBACK_ACTIONS))}",
        f"Current version: {version.get('current') or '-'}",
    ]
    auth_header = api.get("auth_header")
    if auth_header:
        lines.append(f"Auth: {auth_header}")
    if snippets.get("javascript"):
        lines.extend(["", "JavaScript:", str(snippets.get("javascript"))])
    if snippets.get("python"):
        lines.extend(["", "Python:", str(snippets.get("python"))])
    return "\n".join(lines)


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
    "FEEDBACK_ACTIONS",
    "RESPONSE_ID_FIELDS",
    "api_key_required_from_auth_mode",
    "build_handoff_closed_loop",
    "build_handoff_copy_text",
    "build_handoff_javascript_snippet",
    "build_handoff_python_snippet",
    "build_openai_chat_api_contract",
    "build_runtime_payload",
    "runtime_host_port",
]
