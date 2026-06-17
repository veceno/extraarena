from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

JsonSchema = dict[str, Any]
SafetyLevel = Literal["low", "medium", "high", "critical"]
AuditPolicy = Literal["none", "metadata", "request", "request_and_result"]
AdapterFunctionName = str

_CAPABILITY_ID_RE = re.compile(r"^admin\.[a-z0-9]+(?:[._-][a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class AdminCapability:
    """Stable allowlist entry for an admin MCP capability."""

    id: str
    title: str
    description: str
    input_schema: JsonSchema
    required_scope: str
    read_only: bool
    mutating: bool
    safety_level: SafetyLevel
    audit_policy: AuditPolicy
    dry_run_required: bool
    idempotency_required: bool
    adapter_function: AdapterFunctionName

    def __post_init__(self) -> None:
        if not _CAPABILITY_ID_RE.match(self.id):
            raise ValueError(f"invalid admin capability id: {self.id!r}")
        if self.read_only == self.mutating:
            raise ValueError(f"capability must be either read-only or mutating: {self.id}")
        if self.read_only and self.safety_level not in {"low", "medium"}:
            raise ValueError(f"read-only capability cannot be high risk: {self.id}")
        if self.mutating and self.audit_policy in {"none", "metadata"}:
            raise ValueError(f"mutating capability requires request audit policy: {self.id}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_mcp_tool(self) -> dict[str, Any]:
        """Return a minimal MCP-compatible tool descriptor."""
        return {
            "name": self.id,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {
                "readOnlyHint": self.read_only,
                "destructiveHint": self.mutating,
                "idempotencyRequired": self.idempotency_required,
                "dryRunRequired": self.dry_run_required,
                "safetyLevel": self.safety_level,
                "requiredScope": self.required_scope,
                "auditPolicy": self.audit_policy,
            },
        }


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


def _boolean_schema(default: bool | None = None) -> JsonSchema:
    schema: JsonSchema = {"type": "boolean"}
    if default is not None:
        schema["default"] = default
    return schema


def _array_schema(items: JsonSchema, *, min_items: int | None = None, max_items: int | None = None) -> JsonSchema:
    schema: JsonSchema = {"type": "array", "items": items}
    if min_items is not None:
        schema["minItems"] = min_items
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def _shop_set_reward_schema() -> JsonSchema:
    return _object_schema(
        {
            "type": _string_schema(enum=("gems", "coins", "keys", "case", "card", "particles")),
            "amount": _int_schema(minimum=0, maximum=1_000_000),
            "card_id": _int_schema(minimum=1, maximum=1_000_000),
        },
        required=("type",),
    )


def _freeform_object_schema() -> JsonSchema:
    return {
        "type": "object",
        "additionalProperties": True,
    }


def _extrapass_season_patch_schema() -> JsonSchema:
    return _object_schema(
        {
            "slug": _string_schema(min_length=1, max_length=120),
            "name": _string_schema(min_length=1, max_length=160),
            "subtitle": _string_schema(max_length=240),
            "description": _string_schema(max_length=2_000),
            "season_number": _int_schema(minimum=1, maximum=10_000),
            "status": _string_schema(enum=("draft", "scheduled", "active", "archived")),
            "auto_switch": _boolean_schema(),
            "preset_key": _string_schema(max_length=80),
            "start_date": _string_schema(max_length=80),
            "end_date": _string_schema(max_length=80),
            "is_active": _boolean_schema(),
            "max_stars": _int_schema(minimum=1, maximum=99),
            "free_track_type": _string_schema(min_length=1, max_length=80),
            "pass_track_type": _string_schema(min_length=1, max_length=80),
            "ultra_track_type": _string_schema(min_length=1, max_length=80),
            "pass_end_position": _int_schema(minimum=1, maximum=99),
            "ultra_start_position": _int_schema(minimum=1, maximum=99),
            "stage_cost_min": _int_schema(minimum=1, maximum=1_000),
            "stage_cost_growth": _number_schema(minimum=0, maximum=100),
            "stage_cost_exponent": _number_schema(minimum=0.01, maximum=100),
            "stage_cost_cap": _int_schema(minimum=1, maximum=10_000),
            "theme": _freeform_object_schema(),
        },
    )


def _extrapass_reward_row_schema() -> JsonSchema:
    return _object_schema(
        {
            "lane": _string_schema(enum=("free", "premium", "pass", "extra_pass", "ultra")),
            "track": _string_schema(enum=("free", "premium", "pass", "extra_pass", "ultra")),
            "track_type": _string_schema(min_length=1, max_length=80),
            "position": _int_schema(minimum=1, maximum=99),
            "reward_type": _string_schema(min_length=1, max_length=64),
            "reward_amount": _int_schema(minimum=0, maximum=10_000_000),
            "reward_meta": _freeform_object_schema(),
            "extra_pass_required": _boolean_schema(),
        },
    )


def _extrapass_tracks_schema() -> JsonSchema:
    return _object_schema(
        {
            "free": _array_schema(_extrapass_reward_row_schema(), max_items=200),
            "premium": _array_schema(_extrapass_reward_row_schema(), max_items=200),
            "pass": _array_schema(_extrapass_reward_row_schema(), max_items=200),
            "extra_pass": _array_schema(_extrapass_reward_row_schema(), max_items=200),
            "ultra": _array_schema(_extrapass_reward_row_schema(), max_items=200),
            "tracks": _array_schema(_extrapass_reward_row_schema(), max_items=600),
            "rewards": _array_schema(_extrapass_reward_row_schema(), max_items=600),
            "rows": _array_schema(_extrapass_reward_row_schema(), max_items=600),
        },
    )


def _runtime_config_patch_schema() -> JsonSchema:
    return _object_schema(
        {
            "maintenance_mode": _object_schema(
                {"enabled": _boolean_schema()},
                required=("enabled",),
            ),
            "feature_availability": _object_schema(
                {
                    "shop": _boolean_schema(),
                    "collection": _boolean_schema(),
                    "squads": _boolean_schema(),
                    "training": _boolean_schema(),
                    "friendly": _boolean_schema(),
                    "classic": _boolean_schema(),
                    "extra_arena": _boolean_schema(),
                    "rating": _boolean_schema(),
                },
            ),
            "disabled_card_ids": {
                "type": "array",
                "items": _int_schema(minimum=1),
                "uniqueItems": True,
            },
        }
    )


def _mutating_controls() -> dict[str, JsonSchema]:
    return {
        "dry_run": _boolean_schema(),
        "idempotency_key": _string_schema(min_length=8, max_length=128),
        "confirmation_token": _string_schema(min_length=16, max_length=256),
        "reason": _string_schema(max_length=500),
    }


def _account_update_fields_schema() -> JsonSchema:
    return _object_schema(
        {
            "username": _string_schema(max_length=120),
            "first_name": _string_schema(max_length=120),
            "last_name": _string_schema(max_length=120),
            "trophies": _int_schema(minimum=0, maximum=10_000_000),
            "max_trophies": _int_schema(minimum=0, maximum=10_000_000),
            "league": _int_schema(minimum=0, maximum=10_000),
            "status": _string_schema(enum=("active", "warn", "banned")),
            "energy": _int_schema(minimum=0, maximum=10_000),
        }
    )


def _shop_set_patch_schema() -> JsonSchema:
    return _object_schema(
        {
            "name": _string_schema(min_length=1, max_length=120),
            "description": _string_schema(max_length=1_000),
            "image_file_id": _string_schema(max_length=500),
            "price": _number_schema(minimum=0, maximum=1_000_000),
            "currency": _string_schema(enum=("rubles", "gems", "coins")),
            "rewards": _array_schema(_shop_set_reward_schema(), max_items=50),
            "is_active": _boolean_schema(),
        }
    )


def _ruble_product_fields_schema() -> dict[str, JsonSchema]:
    return {
        "code": _string_schema(min_length=1, max_length=120),
        "id": _int_schema(minimum=1),
        "item_type": _string_schema(enum=("extrapass", "extrapass_ultra", "starter_boost", "gems_package", "shop_set")),
        "package_type": _string_schema(max_length=120),
        "shop_set_id": _int_schema(minimum=1),
        "name": _string_schema(min_length=1, max_length=160),
        "description": _string_schema(max_length=2_000),
        "price": _number_schema(minimum=0, maximum=1_000_000),
        "currency": _string_schema(enum=("rubles", "gems", "coins")),
        "image_url": _string_schema(max_length=1_000),
        "badge": _string_schema(max_length=80),
        "sort_order": _int_schema(minimum=-1_000_000, maximum=1_000_000),
        "show_in_game": _boolean_schema(),
        "show_in_shop": _boolean_schema(),
        "is_active": _boolean_schema(),
        "rustore_product_id": _string_schema(max_length=200),
        "metadata": _freeform_object_schema(),
    }


def _promocode_fields_schema() -> dict[str, JsonSchema]:
    return {
        "id": _int_schema(minimum=1),
        "code": _string_schema(min_length=1, max_length=120),
        "type": _string_schema(enum=("permanent", "personal", "welcome")),
        "reward_gems": _int_schema(minimum=0, maximum=1_000_000),
        "reward_coins": _int_schema(minimum=0, maximum=1_000_000),
        "reward_keys": _int_schema(minimum=0, maximum=1_000_000),
        "reward_extrapass": _boolean_schema(),
        "expires_at": _string_schema(max_length=80),
        "is_active": _boolean_schema(),
    }


def _reward_track_patch_schema() -> JsonSchema:
    return _object_schema(
        {
            "track_type": _string_schema(min_length=1, max_length=80),
            "position": _int_schema(minimum=1, maximum=100_000),
            "reward_type": _string_schema(max_length=64),
            "reward_amount": _int_schema(minimum=0, maximum=10_000_000),
            "reward_meta": _freeform_object_schema(),
            "extra_pass_required": _boolean_schema(),
            "is_active": _boolean_schema(),
        }
    )


def _catalog_card_fields_schema() -> dict[str, JsonSchema]:
    return {
        "name": _string_schema(min_length=1, max_length=120),
        "description": _string_schema(max_length=2_000),
        "rarity": _string_schema(enum=("common", "rare", "superrare", "epic", "legendary", "mythic", "divine", "limited", "start", "unique")),
        "power": _int_schema(minimum=0, maximum=1_000_000),
        "mana_cost": _int_schema(minimum=0, maximum=100),
        "base_attack": _int_schema(minimum=0, maximum=1_000_000),
        "base_hp": _int_schema(minimum=0, maximum=1_000_000),
        "card_type": _string_schema(max_length=80),
        "image_file_id": _string_schema(max_length=500),
        "mechanics": _array_schema(_string_schema(max_length=80), max_items=50),
    }


def _catalog_item_fields_schema() -> dict[str, JsonSchema]:
    return {
        "name": _string_schema(min_length=1, max_length=120),
        "description": _string_schema(max_length=2_000),
        "rarity": _string_schema(enum=("common", "rare", "epic", "legendary", "mythic", "divine", "limited", "start")),
        "power": _int_schema(minimum=0, maximum=1_000_000),
        "image_file_id": _string_schema(max_length=500),
    }


ADMIN_CAPABILITIES: tuple[AdminCapability, ...] = (
    AdminCapability(
        id="admin.runtime.status.read",
        title="Read Runtime Status",
        description="Summarize runtime health and safety-relevant configuration state.",
        input_schema=_object_schema(
            {"include_config": _boolean_schema(default=True)}
        ),
        required_scope="admin:runtime:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_runtime_status",
    ),
    AdminCapability(
        id="admin.runtime.config.read",
        title="Read Runtime Config",
        description="Read DB-backed runtime switches such as maintenance mode, feature availability, and disabled cards.",
        input_schema=_object_schema(),
        required_scope="admin:runtime:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_runtime_config",
    ),
    AdminCapability(
        id="admin.players.search",
        title="Search Players",
        description="Search admin-visible players by id, name, status, league, and activity filters.",
        input_schema=_object_schema(
            {
                "query": _string_schema(max_length=80),
                "status": _string_schema(enum=("all", "active", "banned", "warned", "bots")),
                "league": _int_schema(minimum=1),
                "activity": _string_schema(
                    enum=("all", "active_24h", "active_7d", "dormant_7d", "dormant_30d", "paying")
                ),
                "limit": _int_schema(minimum=1, maximum=200),
                "offset": _int_schema(minimum=0, maximum=1_000_000),
            }
        ),
        required_scope="admin:players:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_search_players",
    ),
    AdminCapability(
        id="admin.players.detail.read",
        title="Read Player Detail",
        description="Read a single admin player detail payload, including account, session, payment, economy, battle, and admin-action history.",
        input_schema=_object_schema({"user_id": _int_schema(minimum=1)}, required=("user_id",)),
        required_scope="admin:players:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_get_player_detail",
    ),
    AdminCapability(
        id="admin.shop.products.read",
        title="List Shop Products",
        description="List server-owned ruble products and optionally their public shop catalog projection.",
        input_schema=_object_schema(
            {
                "active_only": _boolean_schema(default=False),
                "surface": _string_schema(enum=("shop", "game")),
                "include_catalog": _boolean_schema(default=True),
            }
        ),
        required_scope="admin:shop:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_list_shop_products",
    ),
    AdminCapability(
        id="admin.shop.sets.read",
        title="List Shop Sets",
        description="List DB-backed shop sets with rewards and activation state.",
        input_schema=_object_schema({"active_only": _boolean_schema(default=False)}),
        required_scope="admin:shop:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_list_shop_sets",
    ),
    AdminCapability(
        id="admin.shop.sets.create",
        title="Create Shop Set",
        description="Create a DB-backed shop reward set through the existing shop-set path.",
        input_schema=_object_schema(
            {
                "name": _string_schema(min_length=1, max_length=120),
                "description": _string_schema(max_length=1_000),
                "image_file_id": _string_schema(max_length=500),
                "price": _number_schema(minimum=0, maximum=1_000_000),
                "currency": _string_schema(enum=("rubles", "gems", "coins")),
                "rewards": _array_schema(_shop_set_reward_schema(), min_items=1, max_items=50),
                "dry_run": _boolean_schema(),
                "idempotency_key": _string_schema(min_length=8, max_length=128),
                "confirmation_token": _string_schema(min_length=16, max_length=256),
                "reason": _string_schema(max_length=500),
            },
            required=("name", "price", "currency", "rewards", "dry_run"),
        ),
        required_scope="admin:shop:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_create_shop_set",
    ),
    AdminCapability(
        id="admin.seasons.reward_tracks.read",
        title="Read Seasons And Reward Tracks",
        description="Read ExtraPass seasons, reset summaries, and reward track rows.",
        input_schema=_object_schema(
            {
                "include_inactive_rewards": _boolean_schema(default=True),
                "include_reset_summaries": _boolean_schema(default=True),
            }
        ),
        required_scope="admin:seasons:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_seasons_reward_tracks",
    ),
    AdminCapability(
        id="admin.extrapass.seasons.draft.create",
        title="Create ExtraPass Season Draft",
        description="Create an ExtraPass season draft from a typed preset.",
        input_schema=_object_schema(
            {
                "preset_key": _string_schema(enum=("blank", "copy_current", "balanced_45")),
                "dry_run": _boolean_schema(),
                "idempotency_key": _string_schema(min_length=8, max_length=128),
                "confirmation_token": _string_schema(min_length=16, max_length=256),
                "reason": _string_schema(max_length=500),
            },
            required=("preset_key", "dry_run"),
        ),
        required_scope="admin:seasons:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_create_extrapass_season_draft",
    ),
    AdminCapability(
        id="admin.extrapass.seasons.patch",
        title="Patch ExtraPass Season",
        description="Patch allowlisted ExtraPass season fields, including schedule, status, progression, track types, and theme.",
        input_schema=_object_schema(
            {
                "season_id": _int_schema(minimum=1),
                "patch": _extrapass_season_patch_schema(),
                "dry_run": _boolean_schema(),
                "idempotency_key": _string_schema(min_length=8, max_length=128),
                "confirmation_token": _string_schema(min_length=16, max_length=256),
                "reason": _string_schema(max_length=500),
            },
            required=("season_id", "patch", "dry_run"),
        ),
        required_scope="admin:seasons:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_patch_extrapass_season",
    ),
    AdminCapability(
        id="admin.extrapass.rewards.import",
        title="Import ExtraPass Rewards",
        description="Replace or upsert ExtraPass reward rows for a season using typed lane/row payloads.",
        input_schema=_object_schema(
            {
                "season_id": _int_schema(minimum=1),
                "tracks": _extrapass_tracks_schema(),
                "replace": _boolean_schema(default=True),
                "dry_run": _boolean_schema(),
                "idempotency_key": _string_schema(min_length=8, max_length=128),
                "confirmation_token": _string_schema(min_length=16, max_length=256),
                "reason": _string_schema(max_length=500),
            },
            required=("season_id", "tracks", "dry_run"),
        ),
        required_scope="admin:seasons:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_import_extrapass_rewards",
    ),
    AdminCapability(
        id="admin.extrapass.reset.preview",
        title="Preview ExtraPass Season Reset",
        description="Preview active-season reset impact before executing it.",
        input_schema=_object_schema(
            {
                "season_id": _int_schema(minimum=1),
                "sample_limit": _int_schema(minimum=0, maximum=1_000),
            },
            required=("season_id",),
        ),
        required_scope="admin:seasons:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_preview_extrapass_reset",
    ),
    AdminCapability(
        id="admin.extrapass.reset.execute",
        title="Execute ExtraPass Season Reset",
        description="Execute a confirmed active ExtraPass season reset.",
        input_schema=_object_schema(
            {
                "season_id": _int_schema(minimum=1),
                "confirm_season_id": _int_schema(minimum=1),
                "reason": _string_schema(max_length=500),
                "dry_run": _boolean_schema(),
                "idempotency_key": _string_schema(min_length=8, max_length=128),
                "confirmation_token": _string_schema(min_length=16, max_length=256),
            },
            required=("season_id", "confirm_season_id", "dry_run"),
        ),
        required_scope="admin:seasons:write",
        read_only=False,
        mutating=True,
        safety_level="critical",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_execute_extrapass_reset",
    ),
    AdminCapability(
        id="admin.extrapass.players.entitlement.set",
        title="Set Player ExtraPass Entitlement",
        description="Grant, upgrade, or revoke a player's ExtraPass entitlement through the audited admin account path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "mode": _string_schema(enum=("inactive", "active", "ultra")),
                "days": _int_schema(minimum=1, maximum=3_650),
                "reason": _string_schema(max_length=500),
                "dry_run": _boolean_schema(),
                "idempotency_key": _string_schema(min_length=8, max_length=128),
                "confirmation_token": _string_schema(min_length=16, max_length=256),
            },
            required=("user_id", "mode", "dry_run"),
        ),
        required_scope="admin:players:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_set_extrapass_player_entitlement",
    ),
    AdminCapability(
        id="admin.match_modes.read",
        title="List Match Modes",
        description="List fixed and ExtraArena match modes with DB availability overrides.",
        input_schema=_object_schema(),
        required_scope="admin:match_modes:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_list_match_modes",
    ),
    AdminCapability(
        id="admin.push.status.read",
        title="Read Push Status",
        description="Read Android push configuration status and registered-device count.",
        input_schema=_object_schema({"platform": _string_schema(enum=("android",))}),
        required_scope="admin:push:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_push_status",
    ),
    AdminCapability(
        id="admin.analytics.overview.read",
        title="Read Analytics Overview",
        description="Read high-level admin analytics for users, revenue, purchases, and battles.",
        input_schema=_object_schema({"days": _int_schema(minimum=1, maximum=365)}),
        required_scope="admin:analytics:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_analytics_overview",
    ),
    AdminCapability(
        id="admin.runtime.config.patch",
        title="Patch Runtime Config",
        description="Apply a typed patch to runtime config. No arbitrary settings or raw payload passthrough are accepted.",
        input_schema=_object_schema(
            {
                "patch": _runtime_config_patch_schema(),
                "dry_run": _boolean_schema(),
                "idempotency_key": _string_schema(min_length=8, max_length=128),
                "confirmation_token": _string_schema(min_length=16, max_length=256),
                "reason": _string_schema(max_length=500),
            },
            required=("patch", "dry_run"),
        ),
        required_scope="admin:runtime:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_patch_runtime_config",
    ),
    AdminCapability(
        id="admin.players.note.create",
        title="Create Player Note",
        description="Record an admin-only player note through the account-action audit log.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "note": _string_schema(min_length=1, max_length=2_000),
                "dry_run": _boolean_schema(),
                "idempotency_key": _string_schema(min_length=8, max_length=128),
                "confirmation_token": _string_schema(min_length=16, max_length=256),
            },
            required=("user_id", "note", "dry_run"),
        ),
        required_scope="admin:players:write",
        read_only=False,
        mutating=True,
        safety_level="medium",
        audit_policy="request",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_create_player_note",
    ),
    AdminCapability(
        id="admin.players.resource.grant",
        title="Grant Player Resource",
        description="Grant a positive amount of an allowlisted player resource through the existing admin economy path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "resource": _string_schema(enum=("gems", "coins", "keys", "stars")),
                "amount": _int_schema(minimum=1, maximum=1_000_000),
                "reason": _string_schema(min_length=1, max_length=500),
                "dry_run": _boolean_schema(),
                "idempotency_key": _string_schema(min_length=8, max_length=128),
                "confirmation_token": _string_schema(min_length=16, max_length=256),
            },
            required=("user_id", "resource", "amount", "reason", "dry_run"),
        ),
        required_scope="admin:economy:grant",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_grant_player_resource",
    ),
    AdminCapability(
        id="admin.configs.summary.read",
        title="Read Admin Config Summary",
        description="Read the same cross-section DB-backed configuration summary used by ExtraAdmin.",
        input_schema=_object_schema(),
        required_scope="admin:runtime:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_admin_config_summary",
    ),
    AdminCapability(
        id="admin.runtime.tps.read",
        title="Read TPS Statistics",
        description="Read live TPS monitor statistics from the running web process.",
        input_schema=_object_schema(),
        required_scope="admin:runtime:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_tps_statistics",
    ),
    AdminCapability(
        id="admin.analytics.section.read",
        title="Read Analytics Section",
        description="Read any ExtraAdmin analytics section through stable DB analytics methods.",
        input_schema=_object_schema(
            {
                "section": _string_schema(
                    enum=(
                        "overview",
                        "revenue",
                        "players",
                        "battles",
                        "economy",
                        "cards",
                        "heroes",
                        "retention",
                        "onboarding",
                        "battle_actions",
                    )
                ),
                "days": _int_schema(minimum=1, maximum=365),
            },
            required=("section",),
        ),
        required_scope="admin:analytics:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_analytics_section",
    ),
    AdminCapability(
        id="admin.analytics.dataset.export.read",
        title="Export Analytics Dataset",
        description="Export a bounded battle-action analytics dataset with TrainV3-style card context as JSON rows.",
        input_schema=_object_schema(
            {
                "days": _int_schema(minimum=1, maximum=365),
                "limit": _int_schema(minimum=1, maximum=100_000),
                "include_players": _boolean_schema(default=False),
            }
        ),
        required_scope="admin:analytics:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_export_analytics_dataset",
    ),
    AdminCapability(
        id="admin.players.analytics.read",
        title="Read Player Analytics Section",
        description="Read player admin overview, league, or activity analytics.",
        input_schema=_object_schema(
            {
                "section": _string_schema(enum=("overview", "leagues", "activity")),
                "days": _int_schema(minimum=1, maximum=365),
            },
            required=("section",),
        ),
        required_scope="admin:players:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_players_analytics",
    ),
    AdminCapability(
        id="admin.players.ban",
        title="Ban Player",
        description="Ban a player through the existing audited admin account path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "reason": _string_schema(min_length=1, max_length=500),
                "until": _string_schema(max_length=80),
                "confirm_self": _boolean_schema(default=False),
                **_mutating_controls(),
            },
            required=("user_id", "reason", "dry_run"),
        ),
        required_scope="admin:players:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_ban_player_account",
    ),
    AdminCapability(
        id="admin.players.unban",
        title="Unban Player",
        description="Remove a player ban through the existing audited admin account path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                **_mutating_controls(),
            },
            required=("user_id", "dry_run"),
        ),
        required_scope="admin:players:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_unban_player_account",
    ),
    AdminCapability(
        id="admin.players.warn",
        title="Warn Player",
        description="Issue a player warning through the existing audited admin account path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "reason": _string_schema(min_length=1, max_length=500),
                **_mutating_controls(),
            },
            required=("user_id", "reason", "dry_run"),
        ),
        required_scope="admin:players:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_warn_player_account",
    ),
    AdminCapability(
        id="admin.players.account.update",
        title="Update Player Account",
        description="Update allowlisted player account fields through the existing audited admin account path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "fields": _account_update_fields_schema(),
                "confirm_self": _boolean_schema(default=False),
                **_mutating_controls(),
            },
            required=("user_id", "fields", "reason", "dry_run"),
        ),
        required_scope="admin:players:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_update_player_account",
    ),
    AdminCapability(
        id="admin.players.delete",
        title="Delete Player",
        description="Permanently delete a player and owned game data through the existing Database.delete_user path.",
        input_schema=_object_schema(
            {
                "user_id": _int_schema(minimum=1),
                "confirm_user_id": _int_schema(minimum=1),
                **_mutating_controls(),
            },
            required=("user_id", "confirm_user_id", "reason", "dry_run"),
        ),
        required_scope="admin:players:delete",
        read_only=False,
        mutating=True,
        safety_level="critical",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_delete_player_account",
    ),
    AdminCapability(
        id="admin.shop.products.options.read",
        title="Read Ruble Product Options",
        description="Read allowed ruble product types, gem package presets, and shop-set references.",
        input_schema=_object_schema(),
        required_scope="admin:shop:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_ruble_product_options",
    ),
    AdminCapability(
        id="admin.shop.products.detail.read",
        title="Read Ruble Product Detail",
        description="Read a single DB-backed ruble product by code.",
        input_schema=_object_schema({"code": _string_schema(min_length=1, max_length=120)}, required=("code",)),
        required_scope="admin:shop:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_get_ruble_product_detail",
    ),
    AdminCapability(
        id="admin.shop.products.create",
        title="Create Ruble Product",
        description="Create a server-owned ruble product for ExtraPass, gem packages, starter boosts, or shop sets.",
        input_schema=_object_schema(
            {**_ruble_product_fields_schema(), **_mutating_controls()},
            required=("code", "item_type", "name", "price", "dry_run"),
        ),
        required_scope="admin:shop:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_create_ruble_product",
    ),
    AdminCapability(
        id="admin.shop.products.update",
        title="Update Ruble Product",
        description="Patch an existing ruble product by code or id.",
        input_schema=_object_schema(
            {**_ruble_product_fields_schema(), **_mutating_controls()},
            required=("dry_run",),
        ),
        required_scope="admin:shop:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_update_ruble_product",
    ),
    AdminCapability(
        id="admin.shop.products.delete",
        title="Delete Ruble Product",
        description="Deactivate a ruble product by code or id.",
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
        mutating=True,
        safety_level="critical",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_delete_ruble_product",
    ),
    AdminCapability(
        id="admin.shop.sets.detail.read",
        title="Read Shop Set Detail",
        description="Read one DB-backed shop set by id.",
        input_schema=_object_schema({"set_id": _int_schema(minimum=1)}, required=("set_id",)),
        required_scope="admin:shop:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_get_shop_set_detail",
    ),
    AdminCapability(
        id="admin.shop.sets.update",
        title="Update Shop Set",
        description="Patch an existing DB-backed shop set.",
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
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_update_shop_set",
    ),
    AdminCapability(
        id="admin.shop.sets.delete",
        title="Delete Shop Set",
        description="Deactivate an existing DB-backed shop set.",
        input_schema=_object_schema(
            {
                "set_id": _int_schema(minimum=1),
                **_mutating_controls(),
            },
            required=("set_id", "dry_run"),
        ),
        required_scope="admin:shop:write",
        read_only=False,
        mutating=True,
        safety_level="critical",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_delete_shop_set",
    ),
    AdminCapability(
        id="admin.promocodes.read",
        title="Read Promocodes",
        description="Read admin promocodes with usage counts.",
        input_schema=_object_schema({"created_by": _int_schema(minimum=1)}),
        required_scope="admin:promocodes:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_list_promocodes",
    ),
    AdminCapability(
        id="admin.promocodes.create",
        title="Create Promocode",
        description="Create a promocode with gems, coins, keys, or ExtraPass reward.",
        input_schema=_object_schema(
            {**_promocode_fields_schema(), **_mutating_controls()},
            required=("code", "type", "dry_run"),
        ),
        required_scope="admin:promocodes:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_create_promocode",
    ),
    AdminCapability(
        id="admin.promocodes.update",
        title="Update Promocode",
        description="Patch an existing promocode by id.",
        input_schema=_object_schema(
            {**_promocode_fields_schema(), **_mutating_controls()},
            required=("id", "dry_run"),
        ),
        required_scope="admin:promocodes:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_update_promocode",
    ),
    AdminCapability(
        id="admin.promocodes.delete",
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
        mutating=True,
        safety_level="critical",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_delete_promocode",
    ),
    AdminCapability(
        id="admin.match_modes.availability.set",
        title="Set Match Mode Availability",
        description="Toggle a known match mode through DB-backed availability overrides.",
        input_schema=_object_schema(
            {
                "mode_id": _string_schema(enum=("classic", "friendly", "training", "extra_arena:blitzkrieg", "extra_arena:spellstorm", "extra_arena:sudden_death", "extra_arena:powermax")),
                "enabled": _boolean_schema(),
                **_mutating_controls(),
            },
            required=("mode_id", "enabled", "dry_run"),
        ),
        required_scope="admin:match_modes:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_set_match_mode_availability",
    ),
    AdminCapability(
        id="admin.push.app_update.broadcast",
        title="Broadcast App Update Push",
        description="Send or preview the Android app-update-required push broadcast.",
        input_schema=_object_schema(
            {
                "title": _string_schema(max_length=120),
                "body": _string_schema(max_length=500),
                "url": _string_schema(max_length=500),
                "limit": _int_schema(minimum=1, maximum=50_000),
                **_mutating_controls(),
            },
            required=("dry_run",),
        ),
        required_scope="admin:push:write",
        read_only=False,
        mutating=True,
        safety_level="critical",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_broadcast_app_update_push",
    ),
    AdminCapability(
        id="admin.rewards.tracks.read",
        title="Read Reward Tracks",
        description="Read generic reward track rows with optional filters.",
        input_schema=_object_schema(
            {
                "track_type": _string_schema(min_length=1, max_length=80),
                "active_only": _boolean_schema(default=False),
                "limit": _int_schema(minimum=1, maximum=2_000),
            }
        ),
        required_scope="admin:seasons:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_reward_tracks",
    ),
    AdminCapability(
        id="admin.rewards.tracks.create",
        title="Create Reward Track Row",
        description="Create or reactivate one generic reward track row.",
        input_schema=_object_schema(
            {
                "track_type": _string_schema(min_length=1, max_length=80),
                "position": _int_schema(minimum=1, maximum=100_000),
                "reward_type": _string_schema(min_length=1, max_length=64),
                "reward_amount": _int_schema(minimum=0, maximum=10_000_000),
                "reward_meta": _freeform_object_schema(),
                "extra_pass_required": _boolean_schema(),
                **_mutating_controls(),
            },
            required=("track_type", "position", "reward_type", "reward_amount", "dry_run"),
        ),
        required_scope="admin:seasons:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_create_reward_track",
    ),
    AdminCapability(
        id="admin.rewards.tracks.patch",
        title="Patch Reward Track Row",
        description="Patch one generic reward track row by id.",
        input_schema=_object_schema(
            {
                "id": _int_schema(minimum=1),
                "patch": _reward_track_patch_schema(),
                **_mutating_controls(),
            },
            required=("id", "patch", "dry_run"),
        ),
        required_scope="admin:seasons:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_patch_reward_track",
    ),
    AdminCapability(
        id="admin.rewards.tracks.delete",
        title="Delete Reward Track Row",
        description="Soft-delete one generic reward track row by id.",
        input_schema=_object_schema(
            {
                "id": _int_schema(minimum=1),
                **_mutating_controls(),
            },
            required=("id", "dry_run"),
        ),
        required_scope="admin:seasons:write",
        read_only=False,
        mutating=True,
        safety_level="critical",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_delete_reward_track",
    ),
    AdminCapability(
        id="admin.catalog.cards.read",
        title="Read Card Catalog",
        description="Read admin card catalog rows.",
        input_schema=_object_schema(),
        required_scope="admin:catalog:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_list_catalog_cards",
    ),
    AdminCapability(
        id="admin.catalog.cards.create",
        title="Create Catalog Card",
        description="Create a catalog card through the existing admin card path.",
        input_schema=_object_schema(
            {**_catalog_card_fields_schema(), **_mutating_controls()},
            required=("name", "rarity", "power", "dry_run"),
        ),
        required_scope="admin:catalog:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_create_catalog_card",
    ),
    AdminCapability(
        id="admin.catalog.cards.collection.set",
        title="Set Admin Card Collection",
        description="Add all cards to, or remove all cards from, the calling admin account collection.",
        input_schema=_object_schema(
            {
                "action": _string_schema(enum=("add_all", "delete_all")),
                **_mutating_controls(),
            },
            required=("action", "dry_run"),
        ),
        required_scope="admin:catalog:write",
        read_only=False,
        mutating=True,
        safety_level="critical",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_set_admin_card_collection",
    ),
    AdminCapability(
        id="admin.catalog.items.read",
        title="Read Item Catalog",
        description="Read admin item catalog rows.",
        input_schema=_object_schema(),
        required_scope="admin:catalog:read",
        read_only=True,
        mutating=False,
        safety_level="low",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_list_catalog_items",
    ),
    AdminCapability(
        id="admin.catalog.items.create",
        title="Create Catalog Item",
        description="Create an admin item catalog row.",
        input_schema=_object_schema(
            {**_catalog_item_fields_schema(), **_mutating_controls()},
            required=("name", "rarity", "power", "dry_run"),
        ),
        required_scope="admin:catalog:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_create_catalog_item",
    ),
    AdminCapability(
        id="admin.stars_test_mode.toggle",
        title="Toggle Stars Test Mode",
        description="Toggle the process-local Stars test mode flag.",
        input_schema=_object_schema(
            {
                "enabled": _boolean_schema(),
                **_mutating_controls(),
            },
            required=("dry_run",),
        ),
        required_scope="admin:runtime:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_toggle_stars_test_mode",
    ),
    AdminCapability(
        id="admin.squads.read",
        title="Read Squads Admin Section",
        description="Read squad analytics, list, detail, or runtime config.",
        input_schema=_object_schema(
            {
                "section": _string_schema(enum=("analytics", "list", "detail", "config")),
                "clan_id": _string_schema(max_length=80),
                "query": _string_schema(max_length=80),
                "filter": _string_schema(max_length=40),
                "sort": _string_schema(max_length=40),
                "days": _int_schema(minimum=1, maximum=365),
                "limit": _int_schema(minimum=1, maximum=200),
                "offset": _int_schema(minimum=0, maximum=1_000_000),
            },
            required=("section",),
        ),
        required_scope="admin:squads:read",
        read_only=True,
        mutating=False,
        safety_level="medium",
        audit_policy="metadata",
        dry_run_required=False,
        idempotency_required=False,
        adapter_function="adapter_read_squads_section",
    ),
    AdminCapability(
        id="admin.squads.action.execute",
        title="Execute Squad Admin Action",
        description="Execute a typed squad admin action using existing squad DB methods.",
        input_schema=_object_schema(
            {
                "action": _string_schema(enum=("create", "update", "balance", "member", "request", "upgrade", "config_set", "process_weekly", "delete")),
                "clan_id": _string_schema(max_length=80),
                "owner_id": _int_schema(minimum=1),
                "name": _string_schema(max_length=120),
                "tag": _string_schema(max_length=16),
                "description": _string_schema(max_length=1_000),
                "type": _string_schema(max_length=40),
                "min_trophies": _int_schema(minimum=0, maximum=1_000_000),
                "fields": _freeform_object_schema(),
                "resource": _string_schema(max_length=40),
                "amount": _int_schema(minimum=-1_000_000, maximum=1_000_000),
                "member_action": _string_schema(max_length=40),
                "target_user_id": _int_schema(minimum=1),
                "personal_tokens": _int_schema(minimum=0, maximum=1_000_000),
                "request_id": _int_schema(minimum=1),
                "request_action": _string_schema(enum=("accept", "reject")),
                "upgrade_type": _string_schema(max_length=80),
                "mode": _string_schema(enum=("set", "buy")),
                "level": _int_schema(minimum=0, maximum=1_000),
                "key": _string_schema(max_length=120),
                "value": {},
                **_mutating_controls(),
            },
            required=("action", "dry_run"),
        ),
        required_scope="admin:squads:write",
        read_only=False,
        mutating=True,
        safety_level="critical",
        audit_policy="request_and_result",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_execute_squad_action",
    ),
    AdminCapability(
        id="admin.uploads.product_image.create",
        title="Upload Product Image",
        description="Upload a small product image from base64 content using the same type and size restrictions as ExtraAdmin.",
        input_schema=_object_schema(
            {
                "filename": _string_schema(max_length=160),
                "content_type": _string_schema(enum=("image/png", "image/jpeg", "image/webp")),
                "base64": _string_schema(min_length=1, max_length=7_000_000),
                **_mutating_controls(),
            },
            required=("content_type", "base64", "dry_run"),
        ),
        required_scope="admin:shop:write",
        read_only=False,
        mutating=True,
        safety_level="high",
        audit_policy="request",
        dry_run_required=True,
        idempotency_required=True,
        adapter_function="adapter_upload_product_image",
    ),
)


def _build_registry(capabilities: tuple[AdminCapability, ...]) -> dict[str, AdminCapability]:
    registry: dict[str, AdminCapability] = {}
    adapter_names: set[str] = set()
    for capability in capabilities:
        if capability.id in registry:
            raise ValueError(f"duplicate admin capability id: {capability.id}")
        if capability.adapter_function in adapter_names:
            raise ValueError(f"duplicate admin adapter function: {capability.adapter_function}")
        registry[capability.id] = capability
        adapter_names.add(capability.adapter_function)
    return registry


ADMIN_CAPABILITY_REGISTRY: dict[str, AdminCapability] = _build_registry(ADMIN_CAPABILITIES)


def get_admin_capability(capability_id: str) -> AdminCapability:
    try:
        return ADMIN_CAPABILITY_REGISTRY[capability_id]
    except KeyError as exc:
        raise KeyError(f"admin capability is not allowlisted: {capability_id}") from exc


def list_admin_capabilities(*, include_mutating: bool = True) -> list[dict[str, Any]]:
    return [
        capability.to_dict()
        for capability in ADMIN_CAPABILITIES
        if include_mutating or capability.read_only
    ]


def list_mcp_admin_tools(*, include_mutating: bool = True) -> list[dict[str, Any]]:
    return [
        capability.to_mcp_tool()
        for capability in ADMIN_CAPABILITIES
        if include_mutating or capability.read_only
    ]
