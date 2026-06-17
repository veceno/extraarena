from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from web.mcp_admin_tools import (
    MCPToolInputError,
    _bool_arg,
    _call_db,
    _int_arg,
    _str_arg,
    json_safe,
)

JsonSchema = dict[str, Any]

PLAYER_ACCOUNT_UPDATE_FIELDS: tuple[str, ...] = (
    "username",
    "first_name",
    "last_name",
    "trophies",
    "max_trophies",
    "league",
    "status",
    "energy",
)
PLAYER_ACCOUNT_STATUSES: tuple[str, ...] = ("active", "warn", "banned")
EXISTING_PLAYER_ADMIN_CAPABILITY_IDS: tuple[str, ...] = (
    "admin.extrapass.players.entitlement.set",
    "admin.players.note.create",
    "admin.players.resource.grant",
)


def _object_schema(
    properties: Mapping[str, Any] | None = None,
    *,
    required: tuple[str, ...] = (),
    additional_properties: bool = False,
) -> JsonSchema:
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required),
        "additionalProperties": additional_properties,
    }


def _int_schema(*, minimum: int | None = None, maximum: int | None = None) -> JsonSchema:
    schema: JsonSchema = {"type": "integer"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _string_schema(
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    enum: tuple[str, ...] | None = None,
) -> JsonSchema:
    schema: JsonSchema = {"type": "string"}
    if min_length is not None:
        schema["minLength"] = min_length
    if max_length is not None:
        schema["maxLength"] = max_length
    if enum is not None:
        schema["enum"] = list(enum)
    return schema


def _boolean_schema(default: bool | None = None) -> JsonSchema:
    schema: JsonSchema = {"type": "boolean"}
    if default is not None:
        schema["default"] = default
    return schema


def _account_update_fields_schema() -> JsonSchema:
    return _object_schema(
        {
            "username": _string_schema(max_length=120),
            "first_name": _string_schema(max_length=120),
            "last_name": _string_schema(max_length=120),
            "trophies": _int_schema(minimum=0, maximum=10_000_000),
            "max_trophies": _int_schema(minimum=0, maximum=10_000_000),
            "league": _int_schema(minimum=0, maximum=10_000),
            "status": _string_schema(enum=PLAYER_ACCOUNT_STATUSES),
            "energy": _int_schema(minimum=0, maximum=10_000),
        },
    )


def _mutating_player_spec(
    *,
    capability_id: str,
    title: str,
    description: str,
    input_schema: JsonSchema,
    safety_level: str,
    audit_policy: str,
    adapter_function: str,
) -> dict[str, Any]:
    return {
        "id": capability_id,
        "title": title,
        "description": description,
        "input_schema": input_schema,
        "required_scope": "admin:players:write",
        "read_only": False,
        "mutating": True,
        "safety_level": safety_level,
        "audit_policy": audit_policy,
        "dry_run_required": True,
        "idempotency_required": True,
        "adapter_function": adapter_function,
    }


def _idempotent_mutation_fields() -> dict[str, JsonSchema]:
    return {
        "dry_run": _boolean_schema(),
        "idempotency_key": _string_schema(min_length=8, max_length=128),
        "confirmation_token": _string_schema(min_length=16, max_length=256),
    }


PLAYER_ADMIN_ACTION_CAPABILITY_SPECS: tuple[dict[str, Any], ...] = (
    _mutating_player_spec(
        capability_id="admin.players.ban",
        title="Ban Player",
        description="Ban a player through the existing audited admin account path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "reason": _string_schema(min_length=1, max_length=500),
                "until": _string_schema(max_length=80),
                "confirm_self": _boolean_schema(default=False),
                **_idempotent_mutation_fields(),
            },
            required=("user_id", "reason", "dry_run"),
        ),
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_ban_player_account",
    ),
    _mutating_player_spec(
        capability_id="admin.players.unban",
        title="Unban Player",
        description="Remove a player ban through the existing audited admin account path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "reason": _string_schema(max_length=500),
                **_idempotent_mutation_fields(),
            },
            required=("user_id", "dry_run"),
        ),
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_unban_player_account",
    ),
    _mutating_player_spec(
        capability_id="admin.players.warn",
        title="Warn Player",
        description="Issue a player warning through the existing audited admin account path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "reason": _string_schema(min_length=1, max_length=500),
                **_idempotent_mutation_fields(),
            },
            required=("user_id", "reason", "dry_run"),
        ),
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_warn_player_account",
    ),
    _mutating_player_spec(
        capability_id="admin.players.account.update",
        title="Update Player Account",
        description="Update allowlisted player account fields through the existing audited admin account path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "fields": _account_update_fields_schema(),
                "reason": _string_schema(min_length=1, max_length=500),
                "confirm_self": _boolean_schema(default=False),
                **_idempotent_mutation_fields(),
            },
            required=("user_id", "fields", "reason", "dry_run"),
        ),
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_update_player_account",
    ),
)


def player_admin_action_capability_specs() -> tuple[dict[str, Any], ...]:
    """Return deepcopy-safe capability kwargs for integration into ADMIN_CAPABILITIES."""
    return tuple(deepcopy(spec) for spec in PLAYER_ADMIN_ACTION_CAPABILITY_SPECS)


def _optional_reason_arg(args: dict[str, Any]) -> str | None:
    reason = _str_arg(args, "reason", "", max_length=500)
    return reason or None


def _required_reason_arg(args: dict[str, Any]) -> str:
    reason = _str_arg(args, "reason", "", max_length=500)
    if not reason:
        raise MCPToolInputError("reason_required")
    return reason


def _parse_optional_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MCPToolInputError("invalid_until_date") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _normalize_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _require_self_confirmation(admin_user_id: int, target_user_id: int, args: dict[str, Any]) -> None:
    if int(admin_user_id) == int(target_user_id) and not _bool_arg(args, "confirm_self", False):
        raise MCPToolInputError("self_ban_requires_confirm")


def _normalize_account_fields(fields: Any) -> dict[str, Any]:
    if not isinstance(fields, dict):
        raise MCPToolInputError("fields_required")
    unknown = set(fields) - set(PLAYER_ACCOUNT_UPDATE_FIELDS)
    if unknown:
        raise MCPToolInputError("unsupported_account_field")

    normalized: dict[str, Any] = {}
    for key in PLAYER_ACCOUNT_UPDATE_FIELDS:
        if key not in fields:
            continue
        value = fields.get(key)
        if key in {"trophies", "max_trophies", "league", "energy"}:
            try:
                value = max(0, int(value or 0))
            except (TypeError, ValueError) as exc:
                raise MCPToolInputError(f"invalid_{key}") from exc
        elif key == "status":
            value = str(value or "active").strip().lower()
            if value not in set(PLAYER_ACCOUNT_STATUSES):
                raise MCPToolInputError("invalid_status")
        elif value is not None:
            value = str(value).strip()
        normalized[key] = value

    if not normalized:
        raise MCPToolInputError("no_valid_fields")
    return normalized


def _raise_on_db_error(result: Any, fallback: str) -> None:
    if not isinstance(result, dict) or result.get("error"):
        raise MCPToolInputError(str((result or {}).get("error") or fallback))


async def adapter_ban_player_account(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    reason = _required_reason_arg(args)
    until = _parse_optional_datetime(args.get("until"))
    _require_self_confirmation(admin_user_id, user_id, args)
    payload = {
        "target_user_id": user_id,
        "reason": reason,
        "until": _normalize_datetime(until),
    }
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, **payload}
    result = await _call_db(
        app["db"],
        "admin_ban_user",
        admin_user_id,
        user_id,
        reason=reason,
        until=until,
        default={"error": "ban_unavailable"},
    )
    _raise_on_db_error(result, "ban_failed")
    return json_safe({"dry_run": False, **payload, "result": result})


async def adapter_unban_player_account(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    reason = _optional_reason_arg(args)
    payload = {"target_user_id": user_id, "reason": reason}
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, **payload}
    result = await _call_db(
        app["db"],
        "admin_unban_user",
        admin_user_id,
        user_id,
        reason=reason,
        default={"error": "unban_unavailable"},
    )
    _raise_on_db_error(result, "unban_failed")
    return json_safe({"dry_run": False, **payload, "result": result})


async def adapter_warn_player_account(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    reason = _required_reason_arg(args)
    payload = {"target_user_id": user_id, "reason": reason}
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, **payload}
    result = await _call_db(
        app["db"],
        "admin_warn_user",
        admin_user_id,
        user_id,
        reason,
        default={"error": "warn_unavailable"},
    )
    _raise_on_db_error(result, "warn_failed")
    return json_safe({"dry_run": False, **payload, "result": result})


async def adapter_update_player_account(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    fields = _normalize_account_fields(args.get("fields"))
    reason = _required_reason_arg(args)
    if str(fields.get("status") or "").lower() == "banned":
        _require_self_confirmation(admin_user_id, user_id, args)
    payload = {"target_user_id": user_id, "fields": fields, "reason": reason}
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, **payload}
    result = await _call_db(
        app["db"],
        "admin_update_user_account",
        admin_user_id,
        user_id,
        fields=fields,
        reason=reason,
        default={"error": "update_account_unavailable"},
    )
    _raise_on_db_error(result, "update_account_failed")
    return json_safe({"dry_run": False, **payload, "result": result})


PLAYER_ADMIN_ACTION_ADAPTERS = {
    "adapter_ban_player_account": adapter_ban_player_account,
    "adapter_unban_player_account": adapter_unban_player_account,
    "adapter_warn_player_account": adapter_warn_player_account,
    "adapter_update_player_account": adapter_update_player_account,
}


__all__ = [
    "EXISTING_PLAYER_ADMIN_CAPABILITY_IDS",
    "PLAYER_ACCOUNT_STATUSES",
    "PLAYER_ACCOUNT_UPDATE_FIELDS",
    "PLAYER_ADMIN_ACTION_ADAPTERS",
    "PLAYER_ADMIN_ACTION_CAPABILITY_SPECS",
    "adapter_ban_player_account",
    "adapter_unban_player_account",
    "adapter_update_player_account",
    "adapter_warn_player_account",
    "player_admin_action_capability_specs",
]
