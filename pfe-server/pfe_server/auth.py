from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field


class ServerSecurityConfig(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    privacy_mode: str = "strict_local"
    allow_remote_access: bool = False
    auth_mode: str = "local_optional"
    api_key_env: str = "PFE_API_KEY"
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    local_management_only: bool = True
    cors_origins: Optional[list[str]] = None
    management_paths: set[str] = Field(
        default_factory=lambda: {"/pfe/signal", "/pfe/distill/run", "/pfe/status", "/pfe/feedback"}
    )


@dataclass
class AccessContext:
    path: str
    method: str
    client_host: Optional[str]
    headers: Mapping[str, str] = field(default_factory=dict)
    endpoint_kind: str = "inference"
    cloud_requested: bool = False


@dataclass
class AccessDecision:
    allowed: bool
    status_code: int = 200
    code: str = "ok"
    detail: str = "ok"
    hint: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_headers(headers: Optional[Mapping[str, str]]) -> dict[str, str]:
    return {str(key).lower(): str(value) for key, value in (headers or {}).items()}


def is_local_client(client_host: Optional[str]) -> bool:
    if not client_host:
        return False
    normalized = client_host.strip().lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    if normalized.startswith("::ffff:"):
        normalized = normalized.removeprefix("::ffff:")
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def extract_api_key(headers: Mapping[str, str], config: ServerSecurityConfig) -> Optional[str]:
    normalized = normalize_headers(headers)
    header_value = normalized.get(config.api_key_header.lower())
    if header_value:
        prefix = config.api_key_prefix
        return header_value[len(prefix) :].strip() if header_value.startswith(prefix) else header_value.strip()
    return normalized.get("x-api-key")


def expected_api_key(config: ServerSecurityConfig) -> Optional[str]:
    return os.getenv(config.api_key_env) or None


def _auth_mode_normalized(config: ServerSecurityConfig) -> str:
    return str(config.auth_mode or "local_optional").strip().lower()


def _auth_mode_requires_api_key(config: ServerSecurityConfig) -> bool:
    return _auth_mode_normalized(config) in {
        "api_key_required",
        "api_key_only",
        "key_required",
        "required",
    }


def authorize_request(
    context: AccessContext,
    config: Optional[ServerSecurityConfig] = None,
) -> AccessDecision:
    config = config or ServerSecurityConfig()

    if config.privacy_mode == "strict_local" and context.cloud_requested:
        return AccessDecision(
            allowed=False,
            status_code=403,
            code="strict_local_blocked",
            detail="strict_local mode forbids cloud-backed management actions.",
        )

    local_client = is_local_client(context.client_host)
    api_key_required = _auth_mode_requires_api_key(config)
    if not local_client and not config.allow_remote_access:
        return AccessDecision(
            allowed=False,
            status_code=403,
            code="remote_management_disabled",
            detail="remote access is disabled; non-local requests are blocked.",
            hint="Enable allow_remote_access in server security config or connect from localhost.",
        )

    is_management = context.endpoint_kind == "management" or context.path in config.management_paths
    requires_api_key = api_key_required or (not local_client and (config.allow_remote_access or is_management))
    if requires_api_key:
        api_key = extract_api_key(context.headers, config)
        expected = expected_api_key(config)
        if not expected:
            return AccessDecision(
                allowed=False,
                status_code=503,
                code="api_key_not_configured",
                detail="API key access requires PFE_API_KEY to be configured.",
                hint="Set PFE_API_KEY environment variable and restart the server.",
            )
        if not api_key or api_key != expected:
            scope = "management" if is_management else "inference"
            return AccessDecision(
                allowed=False,
                status_code=401,
                code="unauthorized",
                detail=f"{scope} access requires a valid API key.",
                hint="Provide a valid API key in the Authorization header: 'Bearer <key>'.",
            )

    return AccessDecision(
        allowed=True,
        metadata={
            "local_client": local_client,
            "management": is_management,
            "cloud_requested": context.cloud_requested,
            "api_key_required": requires_api_key,
            "auth_mode": _auth_mode_normalized(config),
        },
    )
