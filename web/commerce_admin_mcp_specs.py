from __future__ import annotations

from typing import Any

JsonSchema = dict[str, Any]
CapabilitySpec = dict[str, Any]
AdapterSpec = dict[str, Any]

CURRENCIES = ("rubles", "gems", "coins")
RUBLE_PRODUCT_ITEM_TYPES = (
    "extrapass",
    "extrapass_ultra",
    "starter_boost",
    "gems_package",
    "shop_set",
)
PROMOCODE_TYPES = ("permanent", "personal", "welcome")
SHOP_SET_REWARD_TYPES = ("gems", "coins", "keys", "case", "card", "particles")

SOURCE_REFS = {
    "promocode_http": "web/server.py:5543",
    "promocode_routes": "web/server.py:10762",
    "promocode_tables": "infrastructure/database.py:3740",
    "promocode_db": "infrastructure/database.py:11063",
    "shop_sets_http": "web/server.py:16724",
    "shop_sets_routes": "web/server.py:16942",
    "shop_sets_db": "infrastructure/database.py:8999",
    "ruble_products_http": "web/server.py:16952",
    "ruble_products_routes": "web/server.py:17264",
    "ruble_products_db": "infrastructure/database.py:8878",
    "existing_capability_literal": "web/admin_capabilities.py:10",
    "existing_shop_adapters": "web/mcp_admin_tools.py:527",
    "existing_adapter_map": "web/mcp_admin_tools.py:819",
}


def _object_schema(
    properties: dict[str, JsonSchema] | None = None,
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


def _array_schema(
    items: JsonSchema,
    *,
    min_items: int | None = None,
    max_items: int | None = None,
) -> JsonSchema:
    schema: JsonSchema = {"type": "array", "items": items}
    if min_items is not None:
        schema["minItems"] = min_items
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def _boolean_schema(default: bool | None = None) -> JsonSchema:
    schema: JsonSchema = {"type": "boolean"}
    if default is not None:
        schema["default"] = default
    return schema


def _int_schema(*, minimum: int | None = None, maximum: int | None = None) -> JsonSchema:
    schema: JsonSchema = {"type": "integer"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _number_schema(*, minimum: float | None = None, maximum: float | None = None) -> JsonSchema:
    schema: JsonSchema = {"type": "number"}
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


def _freeform_object_schema() -> JsonSchema:
    return {"type": "object", "additionalProperties": True}


def _mutating_controls() -> dict[str, JsonSchema]:
    return {
        "dry_run": _boolean_schema(),
        "idempotency_key": _string_schema(min_length=8, max_length=128),
        "confirmation_token": _string_schema(min_length=16, max_length=256),
        "reason": _string_schema(max_length=500),
    }


def _shop_set_reward_schema() -> JsonSchema:
    return _object_schema(
        {
            "type": _string_schema(enum=SHOP_SET_REWARD_TYPES),
            "amount": _int_schema(minimum=0, maximum=1_000_000),
            "card_id": _int_schema(minimum=1, maximum=1_000_000),
        },
        required=("type",),
    )


def _shop_set_patch_schema() -> JsonSchema:
    return _object_schema(
        {
            "name": _string_schema(min_length=1, max_length=120),
            "description": _string_schema(max_length=1_000),
            "image_file_id": _string_schema(max_length=500),
            "price": _number_schema(minimum=0, maximum=1_000_000),
            "currency": _string_schema(enum=CURRENCIES),
            "rewards": _array_schema(_shop_set_reward_schema(), max_items=50),
            "is_active": _boolean_schema(),
        },
    )


def _ruble_product_patch_schema() -> JsonSchema:
    return _object_schema(
        {
            "code": _string_schema(min_length=1, max_length=120),
            "item_type": _string_schema(enum=RUBLE_PRODUCT_ITEM_TYPES),
            "package_type": _string_schema(max_length=120),
            "shop_set_id": _int_schema(minimum=1),
            "name": _string_schema(min_length=1, max_length=160),
            "description": _string_schema(max_length=2_000),
            "price": _number_schema(minimum=0, maximum=1_000_000),
            "currency": _string_schema(enum=CURRENCIES),
            "image_url": _string_schema(max_length=1_000),
            "badge": _string_schema(max_length=80),
            "sort_order": _int_schema(minimum=-1_000_000, maximum=1_000_000),
            "show_in_game": _boolean_schema(),
            "show_in_shop": _boolean_schema(),
            "is_active": _boolean_schema(),
            "rustore_product_id": _string_schema(max_length=200),
            "metadata": _freeform_object_schema(),
        },
    )


def _promocode_patch_schema() -> JsonSchema:
    return _object_schema(
        {
            "code": _string_schema(min_length=1, max_length=120),
            "type": _string_schema(enum=PROMOCODE_TYPES),
            "reward_gems": _int_schema(minimum=0, maximum=1_000_000),
            "reward_coins": _int_schema(minimum=0, maximum=1_000_000),
            "reward_keys": _int_schema(minimum=0, maximum=1_000_000),
            "reward_extrapass": _boolean_schema(),
            "expires_at": _string_schema(max_length=80),
        },
    )


def _capability(
    *,
    capability_id: str,
    title: str,
    description: str,
    input_schema: JsonSchema,
    required_scope: str,
    read_only: bool,
    safety_level: str,
    audit_policy: str,
    adapter_function: str,
    dry_run_required: bool = False,
    idempotency_required: bool = False,
) -> CapabilitySpec:
    return {
        "id": capability_id,
        "title": title,
        "description": description,
        "input_schema": input_schema,
        "required_scope": required_scope,
        "read_only": read_only,
        "mutating": not read_only,
        "safety_level": safety_level,
        "audit_policy": audit_policy,
        "dry_run_required": dry_run_required,
        "idempotency_required": idempotency_required,
        "adapter_function": adapter_function,
    }


COMMERCE_ADMIN_CAPABILITY_SPECS: tuple[CapabilitySpec, ...] = (
    _capability(
        capability_id="admin.shop.products.options.read",
        title="Read Ruble Product Options",
        description="Read allowed ruble product item types, gem package presets, and shop-set references.",
        input_schema=_object_schema(),
        required_scope="admin:shop:read",
        read_only=True,
        safety_level="low",
        audit_policy="metadata",
        adapter_function="adapter_read_ruble_product_options",
    ),
    _capability(
        capability_id="admin.shop.products.detail.read",
        title="Read Ruble Product Detail",
        description="Read a single DB-backed ruble product by code.",
        input_schema=_object_schema(
            {"code": _string_schema(min_length=1, max_length=120)},
            required=("code",),
        ),
        required_scope="admin:shop:read",
        read_only=True,
        safety_level="low",
        audit_policy="metadata",
        adapter_function="adapter_get_ruble_product_detail",
    ),
    _capability(
        capability_id="admin.shop.products.create",
        title="Create Ruble Product",
        description="Create a server-owned ruble product for ExtraPass, gem packages, starter boosts, or shop sets.",
        input_schema=_object_schema(
            {
                **_ruble_product_patch_schema()["properties"],
                **_mutating_controls(),
            },
            required=("code", "item_type", "name", "price", "dry_run"),
        ),
        required_scope="admin:shop:write",
        read_only=False,
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_create_ruble_product",
        dry_run_required=True,
        idempotency_required=True,
    ),
    _capability(
        capability_id="admin.shop.products.update",
        title="Update Ruble Product",
        description="Patch an existing ruble product by code or id using the same normalized field rules as the HTTP admin path.",
        input_schema=_object_schema(
            {
                "code": _string_schema(min_length=1, max_length=120),
                "id": _int_schema(minimum=1),
                "patch": _ruble_product_patch_schema(),
                **_mutating_controls(),
            },
            required=("patch", "dry_run"),
        ),
        required_scope="admin:shop:write",
        read_only=False,
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_update_ruble_product",
        dry_run_required=True,
        idempotency_required=True,
    ),
    _capability(
        capability_id="admin.shop.products.delete",
        title="Delete Ruble Product",
        description="Soft-delete a ruble product by code or id by marking it inactive.",
        input_schema=_object_schema(
            {
                "code": _string_schema(min_length=1, max_length=120),
                "id": _int_schema(minimum=1),
                **_mutating_controls(),
            },
            required=("dry_run",),
        ),
        required_scope="admin:shop:write",
        read_only=False,
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_delete_ruble_product",
        dry_run_required=True,
        idempotency_required=True,
    ),
    _capability(
        capability_id="admin.shop.sets.detail.read",
        title="Read Shop Set Detail",
        description="Read a single DB-backed shop set with reward rows.",
        input_schema=_object_schema(
            {"set_id": _int_schema(minimum=1)},
            required=("set_id",),
        ),
        required_scope="admin:shop:read",
        read_only=True,
        safety_level="low",
        audit_policy="metadata",
        adapter_function="adapter_get_shop_set_detail",
    ),
    _capability(
        capability_id="admin.shop.sets.update",
        title="Update Shop Set",
        description="Patch a DB-backed shop set name, pricing, activation state, image id, or rewards.",
        input_schema=_object_schema(
            {
                "set_id": _int_schema(minimum=1),
                "patch": _shop_set_patch_schema(),
                **_mutating_controls(),
            },
            required=("set_id", "patch", "dry_run"),
        ),
        required_scope="admin:shop:write",
        read_only=False,
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_update_shop_set",
        dry_run_required=True,
        idempotency_required=True,
    ),
    _capability(
        capability_id="admin.shop.sets.delete",
        title="Delete Shop Set",
        description="Soft-delete a shop set by marking it inactive.",
        input_schema=_object_schema(
            {
                "set_id": _int_schema(minimum=1),
                **_mutating_controls(),
            },
            required=("set_id", "dry_run"),
        ),
        required_scope="admin:shop:write",
        read_only=False,
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_delete_shop_set",
        dry_run_required=True,
        idempotency_required=True,
    ),
    _capability(
        capability_id="admin.promocodes.read",
        title="List Promocodes",
        description="List promocodes, reward fields, expiry, and usage counts.",
        input_schema=_object_schema(
            {"created_by": _int_schema(minimum=1)}
        ),
        required_scope="admin:promocodes:read",
        read_only=True,
        safety_level="medium",
        audit_policy="metadata",
        adapter_function="adapter_list_promocodes",
    ),
    _capability(
        capability_id="admin.promocodes.create",
        title="Create Promocode",
        description="Create a promocode with gem, coin, key, and reward_extrapass support.",
        input_schema=_object_schema(
            {
                "code": _string_schema(min_length=1, max_length=120),
                "type": _string_schema(enum=PROMOCODE_TYPES),
                "reward_gems": _int_schema(minimum=0, maximum=1_000_000),
                "reward_coins": _int_schema(minimum=0, maximum=1_000_000),
                "reward_keys": _int_schema(minimum=0, maximum=1_000_000),
                "reward_extrapass": _boolean_schema(default=False),
                "expires_at": _string_schema(max_length=80),
                **_mutating_controls(),
            },
            required=("code", "dry_run"),
        ),
        required_scope="admin:promocodes:write",
        read_only=False,
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_create_promocode",
        dry_run_required=True,
        idempotency_required=True,
    ),
    _capability(
        capability_id="admin.promocodes.update",
        title="Update Promocode",
        description="Patch promocode code, type, rewards, reward_extrapass, or expiry. Requires a DB update method before enabling.",
        input_schema=_object_schema(
            {
                "id": _int_schema(minimum=1),
                "patch": _promocode_patch_schema(),
                **_mutating_controls(),
            },
            required=("id", "patch", "dry_run"),
        ),
        required_scope="admin:promocodes:write",
        read_only=False,
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_update_promocode",
        dry_run_required=True,
        idempotency_required=True,
    ),
    _capability(
        capability_id="admin.promocodes.delete",
        title="Delete Promocode",
        description="Delete a promocode by id.",
        input_schema=_object_schema(
            {
                "id": _int_schema(minimum=1),
                **_mutating_controls(),
            },
            required=("id", "dry_run"),
        ),
        required_scope="admin:promocodes:write",
        read_only=False,
        safety_level="high",
        audit_policy="request_and_result",
        adapter_function="adapter_delete_promocode",
        dry_run_required=True,
        idempotency_required=True,
    ),
)


COMMERCE_ADMIN_ADAPTER_SPECS: tuple[AdapterSpec, ...] = (
    {
        "adapter_function": "adapter_read_ruble_product_options",
        "capability_id": "admin.shop.products.options.read",
        "http_handler": "ruble_products_options_handler",
        "http_route": "GET /api/admin/ruble-products/options",
        "db_methods": ("get_shop_sets",),
        "returns": ("status", "data.item_types", "data.package_types", "data.shop_sets"),
        "known_errors": {"admin_only": 403, "internal_server_error": 500},
        "notes": (
            "Mirror _admin_ruble_product_options_payload. Options are local constants plus get_shop_sets(active_only=False).",
        ),
    },
    {
        "adapter_function": "adapter_get_ruble_product_detail",
        "capability_id": "admin.shop.products.detail.read",
        "http_handler": "ruble_product_detail_handler",
        "http_route": "GET /api/admin/ruble-products/{code}",
        "db_methods": ("get_ruble_product",),
        "returns": ("product",),
        "known_errors": {"code_required": 400, "product_not_found": 404, "internal_server_error": 500},
    },
    {
        "adapter_function": "adapter_create_ruble_product",
        "capability_id": "admin.shop.products.create",
        "http_handler": "ruble_product_create_handler",
        "http_route": "POST /api/admin/ruble-products/create",
        "db_methods": ("get_ruble_product", "get_shop_set", "create_ruble_product"),
        "returns": ("dry_run", "product_id", "product"),
        "known_errors": {
            "code_itemtype_name_required": 400,
            "invalid_item_type": 400,
            "package_type_required": 400,
            "invalid_package_type": 400,
            "shop_set_id_required": 400,
            "invalid_shop_set_id": 400,
            "shop_set_not_found": 404,
            "name_required": 400,
            "invalid_price": 400,
            "invalid_currency": 400,
            "invalid_image_url": 400,
            "invalid_sort_order": 400,
            "product_code_exists": 409,
        },
    },
    {
        "adapter_function": "adapter_update_ruble_product",
        "capability_id": "admin.shop.products.update",
        "http_handler": "ruble_product_update_handler",
        "http_route": "POST /api/admin/ruble-products/update",
        "db_methods": ("get_ruble_product", "get_shop_set", "update_ruble_product"),
        "returns": ("dry_run", "product", "patch"),
        "known_errors": {
            "code_or_id_required": 400,
            "product_not_found": 404,
            "no_fields": 400,
            "invalid_item_type": 400,
            "package_type_required": 400,
            "invalid_package_type": 400,
            "shop_set_id_required": 400,
            "invalid_shop_set_id": 400,
            "shop_set_not_found": 404,
            "name_required": 400,
            "invalid_price": 400,
            "invalid_currency": 400,
            "invalid_image_url": 400,
            "invalid_sort_order": 400,
        },
    },
    {
        "adapter_function": "adapter_delete_ruble_product",
        "capability_id": "admin.shop.products.delete",
        "http_handler": "ruble_product_delete_handler",
        "http_route": "POST /api/admin/ruble-products/delete",
        "db_methods": ("get_ruble_product", "delete_ruble_product"),
        "returns": ("dry_run", "success", "product"),
        "known_errors": {"code_or_id_required": 400, "product_not_found": 404},
    },
    {
        "adapter_function": "adapter_get_shop_set_detail",
        "capability_id": "admin.shop.sets.detail.read",
        "http_handler": "shop_set_detail_handler",
        "http_route": "GET /api/admin/shop/sets/{set_id}",
        "db_methods": ("get_shop_set",),
        "returns": ("set",),
        "known_errors": {"invalid_set_id": 400, "set_not_found": 404, "internal_server_error": 500},
    },
    {
        "adapter_function": "adapter_update_shop_set",
        "capability_id": "admin.shop.sets.update",
        "http_handler": "shop_set_update_handler",
        "http_route": "POST /api/admin/shop/sets/update",
        "db_methods": ("get_shop_set", "_normalize_shop_set_rewards", "update_shop_set"),
        "returns": ("dry_run", "set", "patch"),
        "known_errors": {
            "invalid_set_id": 400,
            "set_not_found": 404,
            "name_required": 400,
            "invalid_price": 400,
            "invalid_currency": 400,
            "invalid_rewards": 400,
            "invalid_reward_amount": 400,
            "invalid_reward_card": 400,
            "invalid_reward_particles": 400,
            "unknown_reward_type": 400,
            "no_fields": 400,
        },
    },
    {
        "adapter_function": "adapter_delete_shop_set",
        "capability_id": "admin.shop.sets.delete",
        "http_handler": "shop_set_delete_handler",
        "http_route": "POST /api/admin/shop/sets/delete",
        "db_methods": ("get_shop_set", "delete_shop_set"),
        "returns": ("dry_run", "success", "set"),
        "known_errors": {"invalid_set_id": 400, "set_not_found": 404},
    },
    {
        "adapter_function": "adapter_list_promocodes",
        "capability_id": "admin.promocodes.read",
        "http_handler": "promocode_list_handler",
        "http_route": "GET /api/admin/promocodes/list",
        "db_methods": ("get_promocodes_list",),
        "returns": ("promocodes",),
        "known_errors": {"admin_access_required": 403, "internal_server_error": 500},
    },
    {
        "adapter_function": "adapter_create_promocode",
        "capability_id": "admin.promocodes.create",
        "http_handler": "promocode_create_handler",
        "http_route": "POST /api/admin/promocodes/create",
        "db_methods": ("create_promocode",),
        "returns": ("dry_run", "success", "promocode"),
        "known_errors": {
            "code_required": 400,
            "invalid_type": 400,
            "invalid_reward_gems": 400,
            "invalid_reward_coins": 400,
            "invalid_reward_keys": 400,
            "promocode_reward_required": 400,
            "invalid_expires_at": 400,
            "code_exists": 409,
        },
        "notes": ("reward_extrapass must participate in the reward-required check.",),
    },
    {
        "adapter_function": "adapter_update_promocode",
        "capability_id": "admin.promocodes.update",
        "http_handler": None,
        "http_route": None,
        "db_methods": ("update_promocode",),
        "returns": ("dry_run", "success", "promocode", "patch"),
        "known_errors": {
            "invalid_promocode_id": 400,
            "promocode_not_found": 404,
            "no_fields": 400,
            "code_exists": 409,
            "invalid_type": 400,
            "invalid_reward_gems": 400,
            "invalid_reward_coins": 400,
            "invalid_reward_keys": 400,
            "promocode_reward_required": 400,
            "invalid_expires_at": 400,
            "promocode_update_unavailable": 501,
        },
        "requires_new_db_method": True,
        "notes": (
            "No existing HTTP handler or DB update method was found. Add update_promocode before enabling this tool.",
        ),
    },
    {
        "adapter_function": "adapter_delete_promocode",
        "capability_id": "admin.promocodes.delete",
        "http_handler": "promocode_delete_handler",
        "http_route": "POST /api/admin/promocodes/delete",
        "db_methods": ("delete_promocode",),
        "returns": ("dry_run", "success"),
        "known_errors": {"invalid_promocode_id": 400, "not_found": 404},
    },
)


def commerce_capability_ids() -> tuple[str, ...]:
    return tuple(str(spec["id"]) for spec in COMMERCE_ADMIN_CAPABILITY_SPECS)


def commerce_adapter_names() -> tuple[str, ...]:
    return tuple(str(spec["adapter_function"]) for spec in COMMERCE_ADMIN_ADAPTER_SPECS)


def commerce_capability_by_id(capability_id: str) -> CapabilitySpec:
    for spec in COMMERCE_ADMIN_CAPABILITY_SPECS:
        if spec["id"] == capability_id:
            return dict(spec)
    raise KeyError(capability_id)


def commerce_adapter_by_name(adapter_function: str) -> AdapterSpec:
    for spec in COMMERCE_ADMIN_ADAPTER_SPECS:
        if spec["adapter_function"] == adapter_function:
            return dict(spec)
    raise KeyError(adapter_function)
