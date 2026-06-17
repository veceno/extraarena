from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from typing import Any

import jwt as pyjwt

from infrastructure.config import get_settings


MCP_TOKEN_TYPE = "ea_mcp"
MCP_TOKEN_AUDIENCE = "extraarena:mcp-admin"
MCP_JWT_ALGORITHM = "HS256"


def _settings_or_default(settings: Any | None) -> Any:
    return settings if settings is not None else get_settings()


def _mcp_secret(settings: Any) -> str:
    secret = str(getattr(settings, "mcp_token_secret", "") or "").strip()
    if not secret:
        raise ValueError("MCP_TOKEN_SECRET is required")
    return secret


def _normalize_scopes(scopes: Iterable[str]) -> list[str]:
    if isinstance(scopes, str):
        scopes = (scopes,)
    normalized: list[str] = []
    seen: set[str] = set()
    for scope in scopes:
        value = str(scope or "").strip()
        if not value:
            raise ValueError("MCP token scopes must be non-empty strings")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def mint_mcp_token(
    admin_user_id: int,
    scopes: Iterable[str],
    *,
    settings: Any | None = None,
    aud: str = MCP_TOKEN_AUDIENCE,
    ttl_seconds: int | None = None,
    issued_at: int | None = None,
    jti: str | None = None,
) -> str:
    """Mint a short-lived token for the Secure Admin MCP surface only."""

    active_settings = _settings_or_default(settings)
    if not bool(getattr(active_settings, "mcp_enabled", False)):
        raise ValueError("MCP is disabled")

    subject_id = int(admin_user_id)
    if subject_id <= 0:
        raise ValueError("admin_user_id must be positive")

    audience = str(aud or "").strip()
    if not audience:
        raise ValueError("MCP token audience is required")

    ttl = int(ttl_seconds if ttl_seconds is not None else getattr(active_settings, "mcp_token_ttl_seconds", 0))
    if ttl <= 0:
        raise ValueError("MCP token TTL must be positive")

    now = int(issued_at if issued_at is not None else time.time())
    payload = {
        "typ": MCP_TOKEN_TYPE,
        "aud": audience,
        "sub": str(subject_id),
        "admin_user_id": subject_id,
        "scopes": _normalize_scopes(scopes),
        "jti": str(jti or uuid.uuid4()),
        "iat": now,
        "exp": now + ttl,
    }
    return pyjwt.encode(payload, _mcp_secret(active_settings), algorithm=MCP_JWT_ALGORITHM)


def verify_mcp_token(
    token: str,
    *,
    settings: Any | None = None,
    aud: str = MCP_TOKEN_AUDIENCE,
    required_scopes: Iterable[str] = (),
    leeway_seconds: int = 0,
) -> dict[str, Any] | None:
    active_settings = _settings_or_default(settings)
    if not bool(getattr(active_settings, "mcp_enabled", False)):
        return None

    token_value = str(token or "").strip()
    audience = str(aud or "").strip()
    if not token_value or not audience:
        return None

    try:
        payload = pyjwt.decode(
            token_value,
            _mcp_secret(active_settings),
            algorithms=[MCP_JWT_ALGORITHM],
            audience=audience,
            leeway=int(leeway_seconds),
            options={"require": ["typ", "aud", "sub", "admin_user_id", "scopes", "jti", "iat", "exp"]},
        )
    except (pyjwt.ExpiredSignatureError, pyjwt.InvalidTokenError, ValueError, TypeError):
        return None

    if payload.get("typ") != MCP_TOKEN_TYPE:
        return None
    if payload.get("aud") != audience:
        return None
    if not isinstance(payload.get("jti"), str) or not payload["jti"].strip():
        return None
    if not isinstance(payload.get("iat"), int) or not isinstance(payload.get("exp"), int):
        return None

    try:
        admin_user_id = int(payload["admin_user_id"])
    except (TypeError, ValueError):
        return None
    if admin_user_id <= 0 or str(payload.get("sub") or "") != str(admin_user_id):
        return None

    scopes = payload.get("scopes")
    if not isinstance(scopes, list) or not all(isinstance(scope, str) and scope for scope in scopes):
        return None

    try:
        required = set(_normalize_scopes(required_scopes))
    except ValueError:
        return None
    if required and not required.issubset(set(scopes)):
        return None

    payload["admin_user_id"] = admin_user_id
    return payload


__all__ = [
    "MCP_TOKEN_AUDIENCE",
    "MCP_TOKEN_TYPE",
    "mint_mcp_token",
    "verify_mcp_token",
]
