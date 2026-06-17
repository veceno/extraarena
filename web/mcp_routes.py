from __future__ import annotations

import hashlib
import json
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from aiohttp import web

from infrastructure.config import get_settings
from web.admin_capabilities import (
    AdminCapability,
    get_admin_capability,
    list_mcp_admin_tools,
)
from web.mcp_admin_tools import (
    MCPToolInputError,
    execute_admin_capability,
    json_safe,
    semantic_arguments,
)
from web.mcp_auth import mint_mcp_token, verify_mcp_token


MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_SERVER_NAME = "extraarena-admin"
MCP_RATE_LIMIT_REQUESTS = 120
MCP_RATE_LIMIT_WINDOW_SECONDS = 60
MCP_CONFIRMATION_TTL_SECONDS = 5 * 60
MCP_IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60

RequireUserId = Callable[[web.Request], Awaitable[int]]
IsAdminUser = Callable[[Any, int | None], Awaitable[bool]]


def all_mcp_scopes() -> tuple[str, ...]:
    scopes = {"mcp:admin"}
    for tool in list_mcp_admin_tools(include_mutating=True):
        required_scope = (tool.get("annotations") or {}).get("requiredScope")
        if required_scope:
            scopes.add(str(required_scope))
    return tuple(sorted(scopes))


def _canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _scope_allowed(claims: dict[str, Any], required_scope: str) -> bool:
    scopes = set(str(scope) for scope in claims.get("scopes") or [])
    return "mcp:admin" in scopes and required_scope in scopes


def _jsonrpc_result(request_id: Any, result: dict[str, Any]) -> web.Response:
    return web.json_response({"jsonrpc": "2.0", "id": request_id, "result": json_safe(result)})


def _jsonrpc_error(
    request_id: Any,
    *,
    code: int,
    message: str,
    data: dict[str, Any] | None = None,
    status: int = 200,
) -> web.Response:
    error = {"code": int(code), "message": str(message)}
    if data:
        error["data"] = json_safe(data)
    return web.json_response({"jsonrpc": "2.0", "id": request_id, "error": error}, status=status)


def _auth_error(message: str = "mcp_auth_required") -> web.Response:
    return web.json_response({"error": message}, status=401)


def _origin_allowed(request: web.Request) -> bool:
    origin = str(request.headers.get("Origin") or "")
    if not origin:
        return True
    allowed = tuple(get_settings().mcp_allowed_origins)
    return "*" in allowed or origin in allowed


def _origin_error() -> web.Response:
    return web.json_response({"error": "mcp_origin_not_allowed"}, status=403)


def _extract_mcp_bearer_token(request: web.Request) -> str | None:
    if "_auth" in request.rel_url.query:
        return None
    if request.headers.get("Cookie"):
        return None
    authorization = str(request.headers.get("Authorization") or "").strip()
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization[7:].strip()
    return token or None


def _normalize_arguments(params: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(params, dict):
        raise MCPToolInputError("invalid_params")
    name = str(params.get("name") or "").strip()
    arguments = params.get("arguments") or {}
    if not name:
        raise MCPToolInputError("tool_name_required")
    if not isinstance(arguments, dict):
        raise MCPToolInputError("arguments_must_be_object")
    return name, arguments


def _validate_json_schema_subset(schema: dict[str, Any], value: Any, path: str = "arguments") -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            raise MCPToolInputError(f"{path}_must_be_object")
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                raise MCPToolInputError(f"{key}_required")
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extra_keys = set(value) - set(properties)
            if extra_keys:
                raise MCPToolInputError("unexpected_argument")
        for key, item in value.items():
            if key in properties:
                _validate_json_schema_subset(properties[key], item, f"{path}.{key}")
        return
    if schema_type == "array":
        if not isinstance(value, list):
            raise MCPToolInputError(f"{path}_must_be_array")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            raise MCPToolInputError(f"{path}_too_short")
        if max_items is not None and len(value) > max_items:
            raise MCPToolInputError(f"{path}_too_long")
        item_schema = schema.get("items") or {}
        for index, item in enumerate(value):
            _validate_json_schema_subset(item_schema, item, f"{path}.{index}")
        return
    if schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise MCPToolInputError(f"{path}_must_be_integer")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise MCPToolInputError(f"{path}_out_of_range")
        if maximum is not None and value > maximum:
            raise MCPToolInputError(f"{path}_out_of_range")
        return
    if schema_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise MCPToolInputError(f"{path}_must_be_number")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise MCPToolInputError(f"{path}_out_of_range")
        if maximum is not None and value > maximum:
            raise MCPToolInputError(f"{path}_out_of_range")
        return
    if schema_type == "string":
        if not isinstance(value, str):
            raise MCPToolInputError(f"{path}_must_be_string")
        enum = schema.get("enum")
        if enum is not None and value not in enum:
            raise MCPToolInputError(f"{path}_unsupported")
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if min_length is not None and len(value) < min_length:
            raise MCPToolInputError(f"{path}_too_short")
        if max_length is not None and len(value) > max_length:
            raise MCPToolInputError(f"{path}_too_long")
        return
    if schema_type == "boolean" and not isinstance(value, bool):
        raise MCPToolInputError(f"{path}_must_be_boolean")


async def _check_rate_limit(app: web.Application, *, admin_user_id: int, tool_name: str, jti: str) -> None:
    db = app["db"]
    checker = getattr(db, "check_mcp_rate_limit", None)
    if not checker:
        return
    result = await checker(
        scope=f"mcp:{tool_name}",
        subject=str(admin_user_id),
        max_requests=MCP_RATE_LIMIT_REQUESTS,
        window_seconds=MCP_RATE_LIMIT_WINDOW_SECONDS,
    )
    if isinstance(result, dict) and result.get("allowed") is False:
        raise MCPToolInputError("rate_limited")


async def _record_audit(
    app: web.Application,
    *,
    admin_user_id: int,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any = None,
    status: str,
    error: Any = None,
    duration_ms: int | None = None,
    jti: str | None = None,
    confirmation_id: str | None = None,
    idempotency_key: str | None = None,
) -> None:
    recorder = getattr(app["db"], "record_mcp_tool_call", None)
    if not recorder:
        raise MCPToolInputError("mcp_audit_unavailable")
    await recorder(
        admin_user_id=admin_user_id,
        tool_name=tool_name,
        args=arguments,
        result=result,
        status=status,
        error=error,
        duration_ms=duration_ms,
        jti=jti,
        confirmation_id=confirmation_id,
        idempotency_key=idempotency_key,
    )


async def _create_confirmation(
    app: web.Application,
    *,
    admin_user_id: int,
    capability: AdminCapability,
    args_digest: str,
    jti: str,
) -> dict[str, Any]:
    token = secrets.token_urlsafe(32)
    creator = getattr(app["db"], "create_mcp_confirmation", None)
    if not creator:
        raise MCPToolInputError("mcp_persistence_unavailable")
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=MCP_CONFIRMATION_TTL_SECONDS)
    created = await creator(
        admin_user_id=admin_user_id,
        tool_name=capability.id,
        args_digest=args_digest,
        confirmation_token=token,
        expires_at=expires_at,
        jti=jti,
        metadata={"safety_level": capability.safety_level},
    )
    return {
        "confirmation_id": created.get("confirmation_id"),
        "confirmation_token": token,
        "expires_at": json_safe(created.get("expires_at", expires_at)),
        "args_digest": args_digest,
    }


async def _consume_confirmation(
    app: web.Application,
    *,
    admin_user_id: int,
    capability: AdminCapability,
    args_digest: str,
    confirmation_token: str,
) -> dict[str, Any]:
    consumer = getattr(app["db"], "consume_mcp_confirmation", None)
    if not consumer:
        raise MCPToolInputError("mcp_persistence_unavailable")
    consumed = await consumer(
        confirmation_token=confirmation_token,
        admin_user_id=admin_user_id,
        tool_name=capability.id,
        args_digest=args_digest,
    )
    if not consumed or (isinstance(consumed, dict) and consumed.get("success") is False):
        raise MCPToolInputError("confirmation_invalid")
    return consumed


async def _reserve_idempotency(
    app: web.Application,
    *,
    admin_user_id: int,
    capability: AdminCapability,
    arguments: dict[str, Any],
    args_digest: str,
    jti: str,
) -> dict[str, Any] | None:
    if not capability.idempotency_required:
        return None
    key = str(arguments.get("idempotency_key") or "").strip()
    if not key:
        raise MCPToolInputError("idempotency_key_required")
    reserver = getattr(app["db"], "reserve_mcp_idempotency_key", None)
    completer = getattr(app["db"], "complete_mcp_idempotency_key", None)
    if not reserver or not completer:
        raise MCPToolInputError("mcp_persistence_unavailable")
    reserved = await reserver(
        idempotency_key=key,
        admin_user_id=admin_user_id,
        tool_name=capability.id,
        args_digest=args_digest,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=MCP_IDEMPOTENCY_TTL_SECONDS),
        jti=jti,
    )
    if isinstance(reserved, dict) and reserved.get("args_match") is False:
        raise MCPToolInputError("idempotency_key_conflict")
    if isinstance(reserved, dict) and reserved.get("status") == "conflict":
        raise MCPToolInputError(str(reserved.get("error") or "idempotency_key_conflict"))
    return reserved


async def _complete_idempotency(
    app: web.Application,
    *,
    admin_user_id: int,
    capability: AdminCapability,
    arguments: dict[str, Any],
    status: str,
    response: Any = None,
    error: Any = None,
) -> None:
    if not capability.idempotency_required:
        return
    key = str(arguments.get("idempotency_key") or "").strip()
    completer = getattr(app["db"], "complete_mcp_idempotency_key", None)
    if key and completer:
        await completer(
            idempotency_key=key,
            admin_user_id=admin_user_id,
            tool_name=capability.id,
            status=status,
            response=response,
            error=error,
        )


async def _call_tool(
    app: web.Application,
    *,
    request_id: Any,
    claims: dict[str, Any],
    params: Any,
) -> web.Response:
    start = time.perf_counter()
    name = ""
    arguments: dict[str, Any] = {}
    idempotency_reserved = False
    try:
        name, arguments = _normalize_arguments(params)
        capability = get_admin_capability(name)
        if not _scope_allowed(claims, capability.required_scope):
            return _jsonrpc_error(request_id, code=-32001, message="insufficient_scope")
        _validate_json_schema_subset(capability.input_schema, arguments)

        admin_user_id = int(claims["admin_user_id"])
        jti = str(claims.get("jti") or "")
        if not getattr(app["db"], "record_mcp_tool_call", None):
            return _jsonrpc_error(request_id, code=-32003, message="mcp_audit_unavailable")
        await _check_rate_limit(app, admin_user_id=admin_user_id, tool_name=name, jti=jti)

        semantic_args = semantic_arguments(capability, arguments)
        args_digest = _digest(semantic_args)
        dry_run = bool(arguments.get("dry_run", False))
        confirmation_id = None

        if capability.dry_run_required and not dry_run:
            confirmation_token = str(arguments.get("confirmation_token") or "").strip()
            if not confirmation_token:
                await _record_audit(
                    app,
                    admin_user_id=admin_user_id,
                    tool_name=name,
                    arguments=arguments,
                    status="rejected",
                    error="confirmation_required",
                    jti=jti,
                    duration_ms=int((time.perf_counter() - start) * 1000),
                    idempotency_key=arguments.get("idempotency_key"),
                )
                return _jsonrpc_error(request_id, code=-32002, message="confirmation_required")

        idempotency = None
        if capability.mutating and not dry_run:
            idempotency = await _reserve_idempotency(
                app,
                admin_user_id=admin_user_id,
                capability=capability,
                arguments=arguments,
                args_digest=args_digest,
                jti=jti,
            )
            if isinstance(idempotency, dict):
                replay_response = idempotency.get("response") or idempotency.get("response_summary")
                if (idempotency.get("replayable") or idempotency.get("status") == "replay") and replay_response is not None:
                    await _record_audit(
                        app,
                        admin_user_id=admin_user_id,
                        tool_name=name,
                        arguments=arguments,
                        result=replay_response,
                        status="replayed",
                        duration_ms=int((time.perf_counter() - start) * 1000),
                        jti=jti,
                        idempotency_key=arguments.get("idempotency_key"),
                    )
                    return _jsonrpc_result(
                        request_id,
                        {
                            "content": [{"type": "text", "text": _canonical_json(replay_response)}],
                            "structuredContent": replay_response,
                            "isError": False,
                        },
                    )
                if idempotency.get("reserved") is False:
                    raise MCPToolInputError("idempotency_key_in_progress")
                idempotency_reserved = bool(idempotency.get("reserved") is True)

        if capability.dry_run_required and not dry_run:
            consumed_confirmation = await _consume_confirmation(
                app,
                admin_user_id=admin_user_id,
                capability=capability,
                args_digest=args_digest,
                confirmation_token=confirmation_token,
            )
            confirmation_id = consumed_confirmation.get("confirmation_id") if isinstance(consumed_confirmation, dict) else None

        result = await execute_admin_capability(
            app,
            capability=capability,
            admin_user_id=admin_user_id,
            arguments=arguments,
        )

        if capability.dry_run_required and dry_run:
            result = {
                **result,
                "confirmation": await _create_confirmation(
                    app,
                    admin_user_id=admin_user_id,
                    capability=capability,
                    args_digest=args_digest,
                    jti=jti,
                ),
            }

        if idempotency_reserved:
            await _complete_idempotency(
                app,
                admin_user_id=admin_user_id,
                capability=capability,
                arguments=arguments,
                status="success",
                response=result,
            )
        await _record_audit(
            app,
            admin_user_id=admin_user_id,
            tool_name=name,
            arguments=arguments,
            result=result,
            status="success",
            duration_ms=int((time.perf_counter() - start) * 1000),
            jti=jti,
            confirmation_id=(
                (result.get("confirmation") or {}).get("confirmation_id")
                if isinstance(result, dict) and result.get("confirmation")
                else confirmation_id
            ),
            idempotency_key=arguments.get("idempotency_key"),
        )
        return _jsonrpc_result(
            request_id,
            {
                "content": [{"type": "text", "text": _canonical_json(result)}],
                "structuredContent": result,
                "isError": False,
            },
        )
    except KeyError:
        return _jsonrpc_error(request_id, code=-32601, message="tool_not_found")
    except MCPToolInputError as exc:
        return _jsonrpc_error(request_id, code=-32602, message=exc.error)
    except Exception as exc:
        admin_user_id = int(claims.get("admin_user_id") or 0)
        if name and idempotency_reserved:
            try:
                await _complete_idempotency(
                    app,
                    admin_user_id=admin_user_id,
                    capability=get_admin_capability(name),
                    arguments=arguments,
                    status="error",
                    error="tool_execution_failed",
                )
            except Exception:
                pass
        if name and admin_user_id:
            await _record_audit(
                app,
                admin_user_id=admin_user_id,
                tool_name=name,
                arguments=arguments,
                status="error",
                error=str(exc),
                duration_ms=int((time.perf_counter() - start) * 1000),
                jti=str(claims.get("jti") or ""),
                idempotency_key=arguments.get("idempotency_key"),
            )
        return _jsonrpc_error(request_id, code=-32000, message="tool_execution_failed")


def _resources() -> list[dict[str, Any]]:
    return [
        {
            "uri": f"extraarena-admin://tools/{tool['name']}",
            "name": tool["name"],
            "title": tool.get("title"),
            "mimeType": "application/json",
        }
        for tool in list_mcp_admin_tools(include_mutating=True)
    ]


async def _read_resource(uri: str) -> dict[str, Any] | None:
    prefix = "extraarena-admin://tools/"
    if not uri.startswith(prefix):
        return None
    tool_name = uri[len(prefix):]
    for tool in list_mcp_admin_tools(include_mutating=True):
        if tool["name"] == tool_name:
            return tool
    return None


def register_admin_mcp_routes(
    app: web.Application,
    *,
    require_user_id: RequireUserId,
    is_admin_user: IsAdminUser,
) -> None:
    settings = get_settings()
    app["settings"] = settings

    async def mcp_session_handler(request: web.Request) -> web.Response:
        active_settings = get_settings()
        if not bool(active_settings.mcp_enabled):
            return web.json_response({"error": "mcp_disabled"}, status=404)
        if not _origin_allowed(request):
            return _origin_error()
        admin_user_id = request.get("admin_user_id")
        if not admin_user_id:
            admin_user_id = await require_user_id(request)
        if not await is_admin_user(request.app["db"], int(admin_user_id)):
            return web.json_response({"error": "admin_access_required"}, status=403)

        body = {}
        try:
            parsed = await request.json()
            body = parsed if isinstance(parsed, dict) else {}
        except Exception:
            body = {}
        allowed_scopes = set(all_mcp_scopes())
        requested_scopes = body.get("scopes") or sorted(allowed_scopes)
        if not isinstance(requested_scopes, list):
            return web.json_response({"error": "invalid_scopes"}, status=400)
        scopes = tuple(scope for scope in sorted({str(scope) for scope in requested_scopes}) if scope in allowed_scopes)
        if "mcp:admin" not in scopes:
            scopes = tuple(sorted({"mcp:admin", *scopes}))
        token = mint_mcp_token(admin_user_id=int(admin_user_id), scopes=scopes, settings=active_settings)
        return web.json_response({
            "status": "ok",
            "endpoint": active_settings.mcp_endpoint_path,
            "token": token,
            "expires_in": int(active_settings.mcp_token_ttl_seconds),
            "scopes": list(scopes),
        })

    async def mcp_options_handler(request: web.Request) -> web.Response:
        active_settings = get_settings()
        origin = str(request.headers.get("Origin") or "")
        allowed = tuple(active_settings.mcp_allowed_origins)
        response = web.Response(status=204)
        if "*" in allowed:
            response.headers["Access-Control-Allow-Origin"] = "*"
        elif origin and origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        response.headers["Access-Control-Max-Age"] = "600"
        return response

    async def mcp_handler(request: web.Request) -> web.Response:
        active_settings = get_settings()
        if not bool(active_settings.mcp_enabled):
            return web.json_response({"error": "mcp_disabled"}, status=404)
        if not _origin_allowed(request):
            return _origin_error()
        token = _extract_mcp_bearer_token(request)
        if not token:
            return _auth_error()
        claims = verify_mcp_token(token, settings=active_settings, required_scopes=("mcp:admin",))
        if not claims:
            return _auth_error()

        try:
            payload = await request.json()
        except Exception:
            return _jsonrpc_error(None, code=-32700, message="parse_error", status=400)
        if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
            return _jsonrpc_error(None, code=-32600, message="invalid_request")

        request_id = payload.get("id")
        method = str(payload.get("method") or "")
        params = payload.get("params") or {}

        if method == "initialize":
            return _jsonrpc_result(request_id, {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "serverInfo": {"name": MCP_SERVER_NAME, "version": "0.1.0"},
                "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False}},
            })
        if method == "tools/list":
            return _jsonrpc_result(request_id, {"tools": list_mcp_admin_tools(include_mutating=True)})
        if method == "tools/call":
            return await _call_tool(app, request_id=request_id, claims=claims, params=params)
        if method == "resources/list":
            return _jsonrpc_result(request_id, {"resources": _resources()})
        if method == "resources/read":
            uri = str(params.get("uri") if isinstance(params, dict) else "")
            resource = await _read_resource(uri)
            if not resource:
                return _jsonrpc_error(request_id, code=-32602, message="resource_not_found")
            return _jsonrpc_result(
                request_id,
                {"contents": [{"uri": uri, "mimeType": "application/json", "text": _canonical_json(resource)}]},
            )
        return _jsonrpc_error(request_id, code=-32601, message="method_not_found")

    app.router.add_post(settings.mcp_session_path, mcp_session_handler)
    app.router.add_options(settings.mcp_endpoint_path, mcp_options_handler)
    app.router.add_post(settings.mcp_endpoint_path, mcp_handler)
