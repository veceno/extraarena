from __future__ import annotations

import base64
import binascii
import hashlib
import inspect
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from infrastructure.database import SQUAD_SETTINGS_DEFAULTS
from infrastructure.case_config import CASE_CONFIG_FIELDS, RARITY_ORDER
from infrastructure.push_notifications import build_android_push_payload, send_android_broadcast
from infrastructure.shop_config import GEM_PACKAGES
from web.admin_capabilities import AdminCapability

RUNTIME_FEATURE_KEYS = {
    "shop",
    "collection",
    "squads",
    "training",
    "friendly",
    "classic",
    "extra_arena",
    "rating",
    "rating_human_vs_human",
}
MATCH_MODE_CATALOG: tuple[dict[str, Any], ...] = (
    {"mode_id": "classic", "label": "Classic", "available": True},
    {"mode_id": "friendly", "label": "Friendly", "available": True},
    {"mode_id": "training", "label": "Training", "available": True},
    {"mode_id": "extra_arena:blitzkrieg", "label": "ExtraArena Blitzkrieg", "available": True},
    {"mode_id": "extra_arena:spellstorm", "label": "ExtraArena Spellstorm", "available": True},
    {"mode_id": "extra_arena:sudden_death", "label": "ExtraArena Sudden Death", "available": True},
    {"mode_id": "extra_arena:powermax", "label": "ExtraArena PowerMax", "available": True},
)
DEFAULT_EXTRA_PASS_SEASON: dict[str, Any] = {
    "id": None,
    "slug": "arena-rift",
    "name": "Разлом Арены",
    "subtitle": "45 этапов | звезды за бои",
    "description": "Забирай награды по этапам. ExtraPass открывает вторую дорожку, Ultra добавляет финал.",
    "start_date": None,
    "end_date": None,
    "is_active": True,
    "season_number": 1,
    "status": "active",
    "auto_switch": True,
    "preset_key": "default",
    "max_stars": 45,
    "stage_cost_min": 3,
    "stage_cost_growth": 0.07,
    "stage_cost_exponent": 1.5,
    "stage_cost_cap": 25,
    "free_track_type": "bp_free",
    "pass_track_type": "bp_premium",
    "ultra_track_type": "bp_ultra",
    "pass_end_position": 40,
    "ultra_start_position": 41,
    "theme": {},
}
EXTRAPASS_PRESETS = {"blank", "copy_current", "balanced_45"}
EXTRAPASS_SEASON_STATUSES = {"draft", "scheduled", "active", "archived"}
EXTRAPASS_SEASON_PATCH_KEYS = {
    "slug",
    "name",
    "subtitle",
    "description",
    "season_number",
    "status",
    "auto_switch",
    "preset_key",
    "start_date",
    "end_date",
    "is_active",
    "max_stars",
    "free_track_type",
    "pass_track_type",
    "ultra_track_type",
    "pass_end_position",
    "ultra_start_position",
    "stage_cost_min",
    "stage_cost_growth",
    "stage_cost_exponent",
    "stage_cost_cap",
    "theme",
}
EXTRAPASS_REWARD_TYPES = {"coins", "gems", "keys", "card", "specific_card", "case", "particles", "cosmetic"}
EXTRAPASS_STAGE_COST_ROW_FIELDS = {
    "stage_cost",
    "required_stars",
    "stars_required",
    "cost",
    "stage_cost_min",
    "stage_cost_growth",
    "stage_cost_exponent",
    "stage_cost_cap",
}
RUBLE_PRODUCT_ITEM_TYPES = {"extrapass", "extrapass_ultra", "starter_boost", "squad_boost", "gems_package", "shop_set", "gift_shop_set"}
RUBLE_PRODUCT_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")
RUBLE_PRODUCT_IMAGE_URL_PREFIX = "/extraShop/uploads/products/"
RUBLE_PRODUCT_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PROMOCODE_TYPES = {"permanent", "personal", "welcome"}
CATALOG_CARD_RARITIES = {"common", "rare", "superrare", "epic", "legendary", "mythic", "divine", "limited", "start", "unique"}
CATALOG_ITEM_RARITIES = {"common", "rare", "epic", "legendary", "mythic", "divine", "limited", "start"}
PLAYER_ACCOUNT_UPDATE_FIELDS = {"username", "first_name", "last_name", "trophies", "max_trophies", "league", "status", "energy"}
PLAYER_ACCOUNT_STATUSES = {"active", "warn", "banned"}
PRODUCT_IMAGE_CONTENT_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
MAX_PRODUCT_IMAGE_BYTES = 5 * 1024 * 1024
DESIGN_ASSETS_DIR = Path(__file__).resolve().parents[1] / "DesignAssets"
COSMETIC_ITEM_TYPES = {"avatar", "profile_background", "title"}
COSMETIC_IMAGE_CONTENT_TYPES = PRODUCT_IMAGE_CONTENT_TYPES
COSMETIC_IMAGE_SIZES = {"avatar": (750, 750), "profile_background": (760, 380)}
COSMETIC_UPLOAD_DIRS = {
    "avatar": ("PlayerCosmetics", "Avatars", "Admin"),
    "profile_background": ("PlayerCosmetics", "Background", "Admin"),
}
COSMETIC_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")


class MCPToolInputError(ValueError):
    def __init__(self, error: str):
        super().__init__(error)
        self.error = error


def json_safe(value: Any) -> Any:
    if value.__class__.__name__ == "Decimal":
        return int(value) if value == value.to_integral_value() else float(value)
    if value.__class__.__module__ == "datetime" and hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_db(db: Any, method_name: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    method = getattr(db, method_name, None)
    if not method:
        return default
    return await _maybe_await(method(*args, **kwargs))


def _bool_arg(args: dict[str, Any], key: str, default: bool = False) -> bool:
    value = args.get(key, default)
    return bool(value)


def _int_arg(
    args: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    try:
        value = int(args.get(key, default))
    except (TypeError, ValueError) as exc:
        raise MCPToolInputError(f"invalid_{key}") from exc
    if minimum is not None and value < minimum:
        raise MCPToolInputError(f"invalid_{key}")
    if maximum is not None and value > maximum:
        raise MCPToolInputError(f"invalid_{key}")
    return value


def _str_arg(args: dict[str, Any], key: str, default: str = "", *, max_length: int | None = None) -> str:
    value = str(args.get(key, default) or "").strip()
    if max_length is not None and len(value) > max_length:
        raise MCPToolInputError(f"invalid_{key}")
    return value


def _float_arg(
    args: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(args.get(key, default))
    except (TypeError, ValueError) as exc:
        raise MCPToolInputError(f"invalid_{key}") from exc
    if not math.isfinite(value):
        raise MCPToolInputError(f"invalid_{key}")
    if minimum is not None and value < minimum:
        raise MCPToolInputError(f"invalid_{key}")
    if maximum is not None and value > maximum:
        raise MCPToolInputError(f"invalid_{key}")
    return value


def _optional_str_arg(args: dict[str, Any], key: str, *, max_length: int | None = None) -> str | None:
    if key not in args or args.get(key) is None:
        return None
    value = _str_arg(args, key, "", max_length=max_length)
    return value or None


def _normalize_ruble_product_code(value: Any) -> str:
    code = str(value or "").strip()
    if not code:
        raise MCPToolInputError("code_required")
    if not RUBLE_PRODUCT_CODE_RE.fullmatch(code):
        raise MCPToolInputError("invalid_code")
    return code


def _normalize_shop_set_rewards(rewards: Any) -> list[dict[str, Any]]:
    if not isinstance(rewards, list):
        raise MCPToolInputError("invalid_rewards")
    if not rewards:
        raise MCPToolInputError("empty_rewards")
    normalized: list[dict[str, Any]] = []
    for reward in rewards:
        if not isinstance(reward, dict):
            raise MCPToolInputError("invalid_reward")
        reward_type = str(reward.get("type") or "").strip()
        amount = reward.get("amount", 0)
        card_id = reward.get("card_id")
        try:
            amount_int = int(amount or 0)
        except (TypeError, ValueError) as exc:
            raise MCPToolInputError("invalid_reward_amount") from exc
        if reward_type in {"gems", "coins", "keys"}:
            if amount_int <= 0:
                raise MCPToolInputError("invalid_reward_amount")
            normalized.append({"type": reward_type, "amount": amount_int})
        elif reward_type == "case":
            normalized.append({"type": reward_type, "amount": max(1, amount_int)})
        elif reward_type == "card":
            if not card_id:
                raise MCPToolInputError("invalid_reward_card")
            normalized.append({"type": reward_type, "card_id": int(card_id)})
        elif reward_type == "particles":
            if amount_int <= 0 or not card_id:
                raise MCPToolInputError("invalid_reward_particles")
            normalized.append({"type": reward_type, "amount": amount_int, "card_id": int(card_id)})
        elif reward_type == "cosmetic":
            cosmetic_slug = str(reward.get("cosmetic_slug") or "").strip()
            if not cosmetic_slug:
                raise MCPToolInputError("cosmetic_slug_required")
            normalized.append(
                {
                    "type": reward_type,
                    "cosmetic_slug": cosmetic_slug,
                    "auto_equip": bool(reward.get("auto_equip", False)),
                }
            )
        else:
            raise MCPToolInputError("unknown_reward_type")
    return normalized


async def _validated_shop_set_rewards(db: Any, rewards: Any) -> list[dict[str, Any]]:
    normalized = _normalize_shop_set_rewards(rewards)
    validator = getattr(db, "validate_shop_set_rewards", None)
    if validator:
        validated, error = await _maybe_await(validator(normalized))
        if error:
            raise MCPToolInputError(str(error))
        if isinstance(validated, list):
            return validated
    return normalized


def _shop_set_response(record: Any) -> dict[str, Any]:
    item = json_safe(dict(record) if record else {})
    rewards = item.get("rewards")
    if isinstance(rewards, str):
        try:
            parsed = json.loads(rewards)
        except Exception:
            parsed = []
        item["rewards"] = parsed if isinstance(parsed, list) else []
    elif not isinstance(rewards, list):
        item["rewards"] = []
    return item


def _parse_extrapass_datetime(value: Any) -> datetime | None:
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
        raise MCPToolInputError("invalid_season_datetime") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_extrapass_season(record: Any) -> dict[str, Any]:
    raw = dict(record) if record else {}
    season = {**DEFAULT_EXTRA_PASS_SEASON, **raw}
    return {
        **season,
        "max_stars": max(1, int(season.get("max_stars") or DEFAULT_EXTRA_PASS_SEASON["max_stars"])),
        "stage_cost_min": max(1, int(season.get("stage_cost_min") or DEFAULT_EXTRA_PASS_SEASON["stage_cost_min"])),
        "stage_cost_growth": max(0.0, float(season.get("stage_cost_growth") if season.get("stage_cost_growth") is not None else DEFAULT_EXTRA_PASS_SEASON["stage_cost_growth"])),
        "stage_cost_exponent": max(0.01, float(season.get("stage_cost_exponent") if season.get("stage_cost_exponent") is not None else DEFAULT_EXTRA_PASS_SEASON["stage_cost_exponent"])),
        "stage_cost_cap": max(1, int(season.get("stage_cost_cap") or DEFAULT_EXTRA_PASS_SEASON["stage_cost_cap"])),
        "free_track_type": str(season.get("free_track_type") or DEFAULT_EXTRA_PASS_SEASON["free_track_type"]),
        "pass_track_type": str(season.get("pass_track_type") or DEFAULT_EXTRA_PASS_SEASON["pass_track_type"]),
        "ultra_track_type": str(season.get("ultra_track_type") or DEFAULT_EXTRA_PASS_SEASON["ultra_track_type"]),
        "pass_end_position": max(1, int(season.get("pass_end_position") or DEFAULT_EXTRA_PASS_SEASON["pass_end_position"])),
        "ultra_start_position": max(1, int(season.get("ultra_start_position") or DEFAULT_EXTRA_PASS_SEASON["ultra_start_position"])),
        "theme": _json_dict(season.get("theme")),
    }


def _extrapass_track_defs(season: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized = _normalize_extrapass_season(season)
    premium = {
        "id": "premium",
        "track_type": normalized["pass_track_type"],
        "access": "extra_pass",
        "start_position": 1,
        "end_position": int(normalized["pass_end_position"]),
    }
    return {
        "free": {
            "id": "free",
            "track_type": normalized["free_track_type"],
            "access": "free",
            "start_position": 1,
            "end_position": int(normalized["max_stars"]),
        },
        "premium": premium,
        "pass": premium,
        "extra_pass": premium,
        "ultra": {
            "id": "ultra",
            "track_type": normalized["ultra_track_type"],
            "access": "ultra",
            "start_position": int(normalized["ultra_start_position"]),
            "end_position": int(normalized["max_stars"]),
        },
    }


def _extrapass_track_types(season: dict[str, Any]) -> list[str]:
    normalized = _normalize_extrapass_season(season)
    return [
        normalized["free_track_type"],
        normalized["pass_track_type"],
        normalized["ultra_track_type"],
    ]


def _extrapass_track_types_unique(season: dict[str, Any]) -> bool:
    track_types = [track_type for track_type in _extrapass_track_types(season) if track_type]
    return len(track_types) == len(set(track_types))


def _extrapass_track_type_reuse_conflict(
    candidate: dict[str, Any],
    seasons: list[dict[str, Any]],
    *,
    current_season_id: Any = None,
) -> bool:
    candidate_types = set(_extrapass_track_types(candidate))
    current_id = str(current_season_id or candidate.get("id") or "")
    for season in seasons or []:
        if current_id and str(season.get("id") or "") == current_id:
            continue
        if candidate_types.intersection(_extrapass_track_types(season)):
            return True
    return False


def _normalize_extrapass_season_patch(patch: Any, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise MCPToolInputError("patch_required")
    unknown = set(patch) - EXTRAPASS_SEASON_PATCH_KEYS
    if unknown:
        raise MCPToolInputError("unsupported_extrapass_season_field")
    fields: dict[str, Any] = {key: patch[key] for key in EXTRAPASS_SEASON_PATCH_KEYS if key in patch}
    if not fields:
        raise MCPToolInputError("empty_extrapass_season_patch")

    if "status" in fields:
        status = str(fields.get("status") or "").strip().lower()
        if status not in EXTRAPASS_SEASON_STATUSES:
            raise MCPToolInputError("invalid_season_status")
        fields["status"] = status
        if status == "active":
            fields["is_active"] = True

    for key in ("start_date", "end_date"):
        if key in fields:
            fields[key] = _parse_extrapass_datetime(fields[key])

    candidate = dict(existing or {})
    candidate.update(fields)
    start = _parse_extrapass_datetime(candidate.get("start_date"))
    end = _parse_extrapass_datetime(candidate.get("end_date"))
    if start is not None and end is not None and start >= end:
        raise MCPToolInputError("season_dates_invalid")

    try:
        max_stars = int(candidate.get("max_stars", DEFAULT_EXTRA_PASS_SEASON["max_stars"]) or DEFAULT_EXTRA_PASS_SEASON["max_stars"])
        pass_end = int(candidate.get("pass_end_position", min(40, max_stars)) or min(40, max_stars))
        ultra_start = int(candidate.get("ultra_start_position", pass_end + 1) or pass_end + 1)
    except (TypeError, ValueError) as exc:
        raise MCPToolInputError("invalid_season_progression") from exc
    if not 1 <= max_stars <= 99:
        raise MCPToolInputError("invalid_max_stars")
    if not 1 <= pass_end <= max_stars or not 1 <= ultra_start <= max_stars or ultra_start <= pass_end:
        raise MCPToolInputError("invalid_season_positions")
    if not _extrapass_track_types_unique({**DEFAULT_EXTRA_PASS_SEASON, **candidate, "max_stars": max_stars, "pass_end_position": pass_end, "ultra_start_position": ultra_start}):
        raise MCPToolInputError("duplicate_season_track_types")

    int_fields = ("season_number", "max_stars", "pass_end_position", "ultra_start_position", "stage_cost_min", "stage_cost_cap")
    for key in int_fields:
        if key in fields:
            fields[key] = int(fields[key])
    float_fields = ("stage_cost_growth", "stage_cost_exponent")
    for key in float_fields:
        if key in fields:
            fields[key] = float(fields[key])
    if "stage_cost_min" in fields:
        fields["stage_cost_min"] = max(1, int(fields["stage_cost_min"]))
    if "stage_cost_cap" in fields:
        fields["stage_cost_cap"] = max(1, int(fields["stage_cost_cap"]))
    if "stage_cost_growth" in fields:
        fields["stage_cost_growth"] = max(0.0, float(fields["stage_cost_growth"]))
    if "stage_cost_exponent" in fields:
        fields["stage_cost_exponent"] = max(0.01, float(fields["stage_cost_exponent"]))
    candidate_for_costs = {**candidate, **fields}
    min_cost = int(candidate_for_costs.get("stage_cost_min", DEFAULT_EXTRA_PASS_SEASON["stage_cost_min"]) or DEFAULT_EXTRA_PASS_SEASON["stage_cost_min"])
    cap_cost = int(candidate_for_costs.get("stage_cost_cap", DEFAULT_EXTRA_PASS_SEASON["stage_cost_cap"]) or DEFAULT_EXTRA_PASS_SEASON["stage_cost_cap"])
    if cap_cost < min_cost:
        raise MCPToolInputError("invalid_stage_cost_formula")
    if "theme" in fields and not isinstance(fields["theme"], dict):
        raise MCPToolInputError("invalid_theme")
    return fields


def _normalize_extrapass_reward_import_payload(payload: Any, season: dict[str, Any]) -> list[dict[str, Any]]:
    normalized_season = _normalize_extrapass_season(season)
    track_defs = _extrapass_track_defs(normalized_season)
    rows: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        source_rows: list[tuple[str | None, dict[str, Any]]] = []
        for lane_key in ("free", "premium", "pass", "extra_pass", "ultra"):
            lane_rows = payload.get(lane_key)
            if isinstance(lane_rows, list):
                source_rows.extend((lane_key, item) for item in lane_rows if isinstance(item, dict))
        for bucket_key in ("tracks", "rewards", "rows"):
            bucket = payload.get(bucket_key)
            if isinstance(bucket, list):
                source_rows.extend((None, item) for item in bucket if isinstance(item, dict))
    elif isinstance(payload, list):
        source_rows = [(None, item) for item in payload if isinstance(item, dict)]
    else:
        raise MCPToolInputError("extra_pass_tracks_must_be_object_or_array")

    for lane, raw in source_rows:
        if any(field in raw for field in EXTRAPASS_STAGE_COST_ROW_FIELDS):
            raise MCPToolInputError("stage_cost_formula_owned")
        lane_id = str(raw.get("lane") or raw.get("track") or lane or "").strip().lower()
        track_type = str(raw.get("track_type") or "").strip()
        track_def = track_defs.get(lane_id)
        if not track_def and track_type:
            track_def = next((item for item in track_defs.values() if item["track_type"] == track_type), None)
        if not track_def:
            raise MCPToolInputError("unknown_extra_pass_lane")
        try:
            position = int(raw.get("position") or raw.get("tier") or raw.get("stage") or 0)
            reward_amount = int(raw.get("reward_amount", raw.get("amount", 0)) or 0)
        except (TypeError, ValueError) as exc:
            raise MCPToolInputError("invalid_reward_track_row") from exc
        if position < 1 or position > int(normalized_season["max_stars"]):
            raise MCPToolInputError("invalid_reward_position")
        if position < int(track_def["start_position"]) or position > int(track_def["end_position"]):
            raise MCPToolInputError(f"position_out_of_track_scope:{track_def['id']}:{position}")
        reward_type = str(raw.get("reward_type") or raw.get("type") or "").strip()
        if not reward_type:
            raise MCPToolInputError("reward_type_required")
        if reward_amount < 0:
            raise MCPToolInputError("invalid_reward_amount")
        reward_meta = raw.get("reward_meta", raw.get("meta"))
        if reward_meta is None:
            reward_meta = {}
        if isinstance(reward_meta, str):
            try:
                reward_meta = json.loads(reward_meta) if reward_meta.strip() else {}
            except json.JSONDecodeError as exc:
                raise MCPToolInputError("invalid_reward_meta") from exc
        if not isinstance(reward_meta, dict):
            raise MCPToolInputError("invalid_reward_meta")
        reward_meta = _reward_meta_with_inline_fields(raw, reward_meta)
        reward_config = _normalize_extrapass_reward_config(reward_type, reward_amount, reward_meta)
        _validate_extrapass_reward_config(
            reward_config["reward_type"],
            reward_config["reward_amount"],
            reward_config["reward_meta"],
        )
        rows.append({
            "track_type": str(track_def["track_type"]),
            "position": position,
            "reward_type": reward_config["reward_type"],
            "reward_amount": reward_config["reward_amount"],
            "reward_meta": reward_config["reward_meta"],
            "extra_pass_required": track_def["access"] != "free",
        })
    if not rows:
        raise MCPToolInputError("empty_reward_tracks")
    return rows


def _reward_meta_with_inline_fields(raw: dict[str, Any], reward_meta: dict[str, Any]) -> dict[str, Any]:
    meta = dict(reward_meta or {})
    if "card_id" not in meta and "id" not in meta and raw.get("card_id") is not None:
        meta["card_id"] = raw.get("card_id")
    if "cosmetic_slug" not in meta and "slug" not in meta and raw.get("cosmetic_slug") is not None:
        meta["cosmetic_slug"] = raw.get("cosmetic_slug")
    if "auto_equip" not in meta and raw.get("auto_equip") is not None:
        meta["auto_equip"] = bool(raw.get("auto_equip"))
    if "case_tier" not in meta and "tier" not in meta and raw.get("case_tier") is not None:
        meta["case_tier"] = raw.get("case_tier")
    return meta


def _specific_card_id_from_reward_meta(reward_meta: dict[str, Any]) -> int | None:
    try:
        card_id = int(reward_meta.get("card_id") or reward_meta.get("id") or 0)
    except (TypeError, ValueError):
        return None
    return card_id if card_id > 0 else None


def _case_tier_from_reward_meta(reward_meta: dict[str, Any]) -> int | None:
    try:
        tier = int(reward_meta.get("case_tier") or reward_meta.get("tier") or 0)
    except (TypeError, ValueError):
        return None
    return max(1, min(5, tier)) if tier > 0 else None


def _cosmetic_slug_from_reward_meta(reward_meta: dict[str, Any]) -> str | None:
    slug = str(reward_meta.get("cosmetic_slug") or reward_meta.get("slug") or "").strip()
    return slug or None


def _normalize_extrapass_reward_config(reward_type: str, reward_amount: int, reward_meta: dict[str, Any]) -> dict[str, Any]:
    reward_type = str(reward_type or "").strip()
    reward_meta = dict(reward_meta or {})
    if reward_type == "guaranteed_card":
        reward_type = "specific_card" if _specific_card_id_from_reward_meta(reward_meta) is not None else "card"
        reward_amount = 1
    if reward_type == "case":
        reward_amount = _case_tier_from_reward_meta(reward_meta) or int(reward_amount or 0)
    if reward_type == "cosmetic":
        reward_amount = 1
    return {"reward_type": reward_type, "reward_amount": int(reward_amount or 0), "reward_meta": reward_meta}


def _validate_extrapass_reward_config(reward_type: str, reward_amount: int, reward_meta: dict[str, Any]) -> None:
    normalized = _normalize_extrapass_reward_config(reward_type, reward_amount, reward_meta)
    reward_type = normalized["reward_type"]
    reward_amount = normalized["reward_amount"]
    reward_meta = normalized["reward_meta"]
    if reward_type not in EXTRAPASS_REWARD_TYPES:
        raise MCPToolInputError("unsupported_reward_type")
    if reward_type in {"coins", "gems", "keys"} and reward_amount <= 0:
        raise MCPToolInputError("invalid_reward_amount")
    if reward_type == "case" and not 1 <= reward_amount <= 5:
        raise MCPToolInputError("invalid_case_tier")
    if reward_type in {"card", "specific_card"} and reward_amount != 1:
        raise MCPToolInputError("invalid_card_reward_amount")
    if reward_type == "specific_card" and _specific_card_id_from_reward_meta(reward_meta) is None:
        raise MCPToolInputError("specific_card_id_required")
    if reward_type == "particles":
        if reward_amount <= 0:
            raise MCPToolInputError("invalid_reward_amount")
        if _specific_card_id_from_reward_meta(reward_meta) is None:
            raise MCPToolInputError("reward_card_id_required")
    if reward_type == "cosmetic":
        if reward_amount < 0:
            raise MCPToolInputError("invalid_reward_amount")
        if _cosmetic_slug_from_reward_meta(reward_meta) is None:
            raise MCPToolInputError("cosmetic_slug_required")


async def _validate_extrapass_reward_config_with_db(db: Any, row: dict[str, Any]) -> None:
    reward_meta = row.get("reward_meta") if isinstance(row.get("reward_meta"), dict) else {}
    _validate_extrapass_reward_config(
        str(row.get("reward_type") or ""),
        int(row.get("reward_amount") or 0),
        reward_meta,
    )
    normalized = _normalize_extrapass_reward_config(
        str(row.get("reward_type") or ""),
        int(row.get("reward_amount") or 0),
        reward_meta,
    )
    if normalized["reward_type"] == "particles":
        card_id = _specific_card_id_from_reward_meta(normalized["reward_meta"])
        if card_id is None:
            raise MCPToolInputError("reward_card_id_required")
        card = await _call_db(db, "get_card_info", card_id, default=None)
        if not card:
            raise MCPToolInputError("reward_card_not_found")
        return
    if normalized["reward_type"] == "cosmetic":
        slug = _cosmetic_slug_from_reward_meta(normalized["reward_meta"])
        if not slug:
            raise MCPToolInputError("cosmetic_slug_required")
        item = await _call_db(db, "get_cosmetic_item", slug, default=None)
        if not item:
            raise MCPToolInputError("reward_cosmetic_not_found")
        return
    if normalized["reward_type"] != "specific_card":
        return
    card_id = _specific_card_id_from_reward_meta(normalized["reward_meta"])
    if card_id is None:
        raise MCPToolInputError("specific_card_id_required")
    card = await _call_db(db, "get_card_info", card_id, default=None)
    if not card:
        raise MCPToolInputError("specific_card_not_found")
    if str(card.get("card_type") or "warrior") != "warrior":
        raise MCPToolInputError("specific_card_must_be_warrior")


def _normalize_runtime_patch(patch: Any) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise MCPToolInputError("patch_required")
    normalized: dict[str, Any] = {}
    allowed = {"maintenance_mode", "feature_availability", "disabled_card_ids"}
    unknown = set(patch) - allowed
    if unknown:
        raise MCPToolInputError("unsupported_runtime_config_field")

    if "maintenance_mode" in patch:
        maintenance = patch.get("maintenance_mode")
        if not isinstance(maintenance, dict) or "enabled" not in maintenance:
            raise MCPToolInputError("invalid_maintenance_mode")
        normalized["maintenance_mode"] = {"enabled": bool(maintenance.get("enabled"))}

    if "feature_availability" in patch:
        features = patch.get("feature_availability")
        if not isinstance(features, dict):
            raise MCPToolInputError("invalid_feature_availability")
        unknown_features = {str(key) for key in features} - RUNTIME_FEATURE_KEYS
        if unknown_features:
            raise MCPToolInputError("unsupported_runtime_feature")
        normalized["feature_availability"] = {
            str(key): bool(value)
            for key, value in features.items()
            if str(key).strip()
        }

    if "disabled_card_ids" in patch:
        cards = patch.get("disabled_card_ids")
        if not isinstance(cards, list):
            raise MCPToolInputError("invalid_disabled_card_ids")
        try:
            normalized["disabled_card_ids"] = sorted({int(card_id) for card_id in cards if int(card_id) > 0})
        except (TypeError, ValueError) as exc:
            raise MCPToolInputError("invalid_disabled_card_ids") from exc

    if not normalized:
        raise MCPToolInputError("empty_runtime_config_patch")
    return normalized


def _normalize_case_config_patch(patch: Any) -> dict[str, Any]:
    """Валидировать partial-патч конфигурации кейсов.

    Коэрсит строковые ключи тиров к int, валидирует диапазоны/суммы, возвращает
    нормализованный патч (int tier keys), готовый для merge_case_config_patch.
    Поднимает MCPToolInputError с типизированным кодом при ошибке.
    """
    if not isinstance(patch, dict):
        raise MCPToolInputError("patch_required")
    unknown = set(patch) - CASE_CONFIG_FIELDS
    if unknown:
        raise MCPToolInputError("unsupported_case_config_field")
    normalized: dict[str, Any] = {}

    def _is_number(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    def _tier_dict(value: Any, lo: int, hi: int, err: str) -> dict[int, Any]:
        if not isinstance(value, dict) or not value:
            raise MCPToolInputError(err)
        out: dict[int, Any] = {}
        for k, v in value.items():
            try:
                ik = int(k)
            except (TypeError, ValueError):
                raise MCPToolInputError(err)
            if not (lo <= ik <= hi):
                raise MCPToolInputError(err)
            out[ik] = v
        return out

    def _int_pair(value: Any, err: str) -> list[int]:
        if not (isinstance(value, (list, tuple)) and len(value) == 2
                and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
                and value[0] <= value[1]):
            raise MCPToolInputError(err)
        return [int(value[0]), int(value[1])]

    if "tier_rarity_probabilities" in patch:
        trp = _tier_dict(patch["tier_rarity_probabilities"], 1, 5, "invalid_tier_rarity_probabilities")
        validated: dict[int, Any] = {}
        for tier, probs in trp.items():
            if not isinstance(probs, dict) or not probs:
                raise MCPToolInputError("invalid_tier_rarity_probabilities")
            vprobs: dict[str, float] = {}
            s = 0.0
            for rarity, p in probs.items():
                if rarity not in RARITY_ORDER:
                    raise MCPToolInputError("invalid_tier_rarity_rarity")
                if not _is_number(p) or not (0 <= p <= 1):
                    raise MCPToolInputError("invalid_tier_rarity_prob_value")
                vprobs[rarity] = float(p)
                s += p
            if abs(s - 1.0) > 0.02:
                raise MCPToolInputError("invalid_tier_rarity_sum")
            validated[tier] = vprobs
        normalized["tier_rarity_probabilities"] = validated

    if "tier_particles_multiplier" in patch:
        tpm = _tier_dict(patch["tier_particles_multiplier"], 1, 5, "invalid_tier_particles_multiplier")
        validated = {}
        for tier, m in tpm.items():
            if not _is_number(m) or m < 0:
                raise MCPToolInputError("invalid_tier_particles_multiplier")
            validated[tier] = float(m)
        normalized["tier_particles_multiplier"] = validated

    if "base_particles_by_rarity" in patch:
        bpb = patch["base_particles_by_rarity"]
        if not isinstance(bpb, dict) or not bpb:
            raise MCPToolInputError("invalid_base_particles_by_rarity")
        validated = {}
        for rarity, v in bpb.items():
            if not isinstance(rarity, str):
                raise MCPToolInputError("invalid_base_particles_by_rarity")
            if rarity not in RARITY_ORDER:
                raise MCPToolInputError("invalid_base_particles_rarity")
            if not _is_number(v) or v < 0:
                raise MCPToolInputError("invalid_base_particles_value")
            validated[rarity] = v
        normalized["base_particles_by_rarity"] = validated

    if "tier_rewards_count" in patch:
        trc = _tier_dict(patch["tier_rewards_count"], 1, 5, "invalid_tier_rewards_count")
        validated = {}
        for tier, cfg in trc.items():
            if not isinstance(cfg, dict):
                raise MCPToolInputError("invalid_tier_rewards_count")
            vcfg: dict[str, Any] = {
                "coins": _int_pair(cfg.get("coins"), "invalid_tier_rewards_coins"),
                "cards": _int_pair(cfg.get("cards"), "invalid_tier_rewards_cards"),
            }
            if "gems_chance" in cfg:
                gc = cfg["gems_chance"]
                if not _is_number(gc) or not (0 <= gc <= 1):
                    raise MCPToolInputError("invalid_tier_rewards_gems_chance")
                vcfg["gems_chance"] = float(gc)
            if "gems_amount" in cfg:
                vcfg["gems_amount"] = _int_pair(cfg["gems_amount"], "invalid_tier_rewards_gems_amount")
            validated[tier] = vcfg
        normalized["tier_rewards_count"] = validated

    if "start_rarity_replacement" in patch:
        srr = patch["start_rarity_replacement"]
        if not isinstance(srr, dict):
            raise MCPToolInputError("invalid_start_rarity_replacement")
        validated = {}
        for rarity, p in srr.items():
            if not isinstance(rarity, str):
                raise MCPToolInputError("invalid_start_rarity_replacement")
            if rarity not in RARITY_ORDER:
                raise MCPToolInputError("invalid_start_rarity_replacement_rarity")
            if not _is_number(p) or not (0 <= p <= 1):
                raise MCPToolInputError("invalid_start_rarity_replacement_value")
            validated[rarity] = float(p)
        normalized["start_rarity_replacement"] = validated

    if "max_rarity_by_tier" in patch:
        mrb = _tier_dict(patch["max_rarity_by_tier"], 1, 5, "invalid_max_rarity_by_tier")
        validated = {}
        for tier, rarity in mrb.items():
            if rarity not in RARITY_ORDER:
                raise MCPToolInputError("invalid_max_rarity_by_tier")
            validated[tier] = rarity
        normalized["max_rarity_by_tier"] = validated

    if "t5_common_jackpot_particles" in patch:
        jc = patch["t5_common_jackpot_particles"]
        if not isinstance(jc, int) or isinstance(jc, bool) or jc < 0:
            raise MCPToolInputError("invalid_t5_jackpot")
        normalized["t5_common_jackpot_particles"] = int(jc)

    if "tier_upgrade_chances" in patch:
        tuc = _tier_dict(patch["tier_upgrade_chances"], 1, 4, "invalid_tier_upgrade_chances")
        validated = {}
        for tier, p in tuc.items():
            if not _is_number(p) or not (0 <= p <= 1):
                raise MCPToolInputError("invalid_tier_upgrade_chances")
            validated[tier] = float(p)
        normalized["tier_upgrade_chances"] = validated

    if "limited_event_active" in patch:
        lea = patch["limited_event_active"]
        if not isinstance(lea, bool):
            raise MCPToolInputError("invalid_limited_event_active")
        normalized["limited_event_active"] = lea

    if "limited_event_probability" in patch:
        lep = patch["limited_event_probability"]
        if not _is_number(lep) or not (0 <= lep <= 1):
            raise MCPToolInputError("invalid_limited_event_probability")
        normalized["limited_event_probability"] = float(lep)

    if not normalized:
        raise MCPToolInputError("empty_case_config_patch")
    return normalized


def _required_reason_arg(args: dict[str, Any]) -> str:
    reason = _str_arg(args, "reason", "", max_length=500)
    if not reason:
        raise MCPToolInputError("reason_required")
    return reason


def _parse_optional_datetime_arg(value: Any, error: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return _parse_extrapass_datetime(value)
    except MCPToolInputError as exc:
        raise MCPToolInputError(error) from exc


def _require_self_ban_confirmation(admin_user_id: int, target_user_id: int, args: dict[str, Any]) -> None:
    if int(admin_user_id) == int(target_user_id) and not _bool_arg(args, "confirm_self", False):
        raise MCPToolInputError("self_ban_requires_confirm")


def _normalize_account_fields(fields: Any) -> dict[str, Any]:
    if not isinstance(fields, dict):
        raise MCPToolInputError("fields_required")
    unknown = set(fields) - PLAYER_ACCOUNT_UPDATE_FIELDS
    if unknown:
        raise MCPToolInputError("unsupported_account_field")
    normalized: dict[str, Any] = {}
    for key, value in fields.items():
        if key in {"trophies", "max_trophies", "league", "energy"}:
            try:
                normalized[key] = max(0, int(value or 0))
            except (TypeError, ValueError) as exc:
                raise MCPToolInputError(f"invalid_{key}") from exc
        elif key == "status":
            status = str(value or "active").strip().lower()
            if status not in PLAYER_ACCOUNT_STATUSES:
                raise MCPToolInputError("invalid_status")
            normalized[key] = status
        else:
            normalized[key] = str(value or "").strip() if value is not None else None
    if not normalized:
        raise MCPToolInputError("no_valid_fields")
    return normalized


def _safe_product_image_url(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        return None
    if any(ch in text for ch in ("'", '"', "`", "<", ">")):
        return None
    if text.startswith(RUBLE_PRODUCT_IMAGE_URL_PREFIX):
        filename = text[len(RUBLE_PRODUCT_IMAGE_URL_PREFIX):]
        suffix = Path(filename).suffix.lower()
        if (
            filename
            and "/" not in filename
            and "\\" not in filename
            and ".." not in filename
            and suffix in RUBLE_PRODUCT_IMAGE_EXTENSIONS
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", filename)
        ):
            return text
        return None
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if parts.scheme != "https" or not parts.netloc or "\\" in text:
        return None
    return text


async def _normalize_ruble_product_payload(
    db: Any,
    data: dict[str, Any],
    *,
    require_identity: bool,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = dict(existing or {})
    normalized: dict[str, Any] = {}

    if require_identity or "code" in data:
        normalized["code"] = _normalize_ruble_product_code(data.get("code"))

    item_type = str(data.get("item_type") if "item_type" in data else existing.get("item_type") or "").strip()
    if require_identity or "item_type" in data or "package_type" in data or "shop_set_id" in data:
        if not item_type:
            raise MCPToolInputError("item_type_required")
        if item_type not in RUBLE_PRODUCT_ITEM_TYPES:
            raise MCPToolInputError("invalid_item_type")
        if require_identity or "item_type" in data:
            normalized["item_type"] = item_type
        package_type = data.get("package_type") if "package_type" in data else existing.get("package_type")
        shop_set_id_raw = data.get("shop_set_id") if "shop_set_id" in data else existing.get("shop_set_id")
        if item_type == "gems_package":
            package_text = str(package_type or "").strip()
            if not package_text:
                raise MCPToolInputError("package_type_required")
            if package_text not in GEM_PACKAGES:
                raise MCPToolInputError("invalid_package_type")
            normalized["package_type"] = package_text
            normalized["shop_set_id"] = None
        elif item_type in {"shop_set", "gift_shop_set"}:
            if shop_set_id_raw in (None, ""):
                raise MCPToolInputError("shop_set_id_required")
            try:
                shop_set_id = int(shop_set_id_raw)
            except (TypeError, ValueError) as exc:
                raise MCPToolInputError("invalid_shop_set_id") from exc
            if shop_set_id <= 0:
                raise MCPToolInputError("invalid_shop_set_id")
            if not await _call_db(db, "get_shop_set", shop_set_id, default=None):
                raise MCPToolInputError("shop_set_not_found")
            normalized["shop_set_id"] = shop_set_id
            normalized["package_type"] = None
        else:
            if "package_type" in data:
                normalized["package_type"] = None
            if "shop_set_id" in data:
                normalized["shop_set_id"] = None

    if require_identity or "name" in data:
        name = str(data.get("name") or "").strip()
        if not name:
            raise MCPToolInputError("name_required")
        normalized["name"] = name

    if require_identity or "price" in data:
        price = _float_arg(data, "price", 0, minimum=0, maximum=1_000_000)
        if item_type == "gift_shop_set":
            if price != 0:
                raise MCPToolInputError("gift_shop_set_price_must_be_zero")
        elif price <= 0:
            raise MCPToolInputError("invalid_price")
        normalized["price"] = price

    if require_identity or "item_type" in data or "price" in data:
        try:
            effective_price = float(normalized.get("price", existing.get("price", 0)) or 0)
        except (TypeError, ValueError) as exc:
            raise MCPToolInputError("invalid_price") from exc
        if item_type == "gift_shop_set":
            if effective_price != 0:
                raise MCPToolInputError("gift_shop_set_price_must_be_zero")
        elif effective_price <= 0:
            raise MCPToolInputError("invalid_price")

    if require_identity or "item_type" in data or "currency" in data:
        effective_currency = str(normalized.get("currency", existing.get("currency", "rubles")) or "rubles")
        if item_type == "gift_shop_set" and effective_currency != "rubles":
            raise MCPToolInputError("invalid_currency")

    if require_identity or "currency" in data:
        currency = str(data.get("currency") or "rubles")
        if currency not in {"rubles", "gems", "coins"}:
            raise MCPToolInputError("invalid_currency")
        if item_type == "gift_shop_set" and currency != "rubles":
            raise MCPToolInputError("invalid_currency")
        normalized["currency"] = currency

    for field in ("description", "badge"):
        if field in data:
            raw_value = data.get(field)
            normalized[field] = (str(raw_value).strip() if raw_value is not None else None) or None

    if "image_url" in data:
        image_url = _safe_product_image_url(data.get("image_url"))
        if data.get("image_url") and not image_url:
            raise MCPToolInputError("invalid_image_url")
        normalized["image_url"] = image_url

    if "sort_order" in data:
        normalized["sort_order"] = _int_arg(data, "sort_order", 100, minimum=-1_000_000, maximum=1_000_000)

    for field in ("show_in_game", "show_in_shop", "is_active"):
        if field in data:
            normalized[field] = bool(data[field])

    if "metadata" in data or "rustore_product_id" in data:
        metadata = _json_dict(existing.get("metadata"))
        incoming_metadata = data.get("metadata")
        if isinstance(incoming_metadata, dict):
            metadata.update(incoming_metadata)
        if "rustore_product_id" in data:
            rustore_product_id = str(data.get("rustore_product_id") or "").strip()
            if rustore_product_id:
                metadata["rustore_product_id"] = rustore_product_id
            else:
                metadata.pop("rustore_product_id", None)
        normalized["metadata"] = metadata

    if not normalized:
        raise MCPToolInputError("no_valid_fields")
    return normalized


def _ruble_product_identity(args: dict[str, Any]) -> str | int:
    if args.get("id") is not None:
        return _int_arg(args, "id", 0, minimum=1)
    if not str(args.get("code") or "").strip():
        raise MCPToolInputError("code_or_id_required")
    return _normalize_ruble_product_code(args.get("code"))


async def _get_ruble_product_for_identity(db: Any, identity: str | int) -> dict[str, Any] | None:
    if isinstance(identity, str):
        return await _call_db(db, "get_ruble_product", identity, default=None)
    products = await _call_db(db, "get_ruble_products", default=[], active_only=False)
    if not isinstance(products, list):
        return None
    return next(
        (product for product in products if isinstance(product, dict) and int(product.get("id") or 0) == int(identity)),
        None,
    )


def _normalize_shop_set_patch(patch: Any) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise MCPToolInputError("patch_required")
    allowed = {"name", "description", "image_file_id", "price", "currency", "rewards", "is_active"}
    unknown = set(patch) - allowed
    if unknown:
        raise MCPToolInputError("unsupported_shop_set_field")
    normalized: dict[str, Any] = {}
    if "name" in patch:
        name = str(patch.get("name") or "").strip()
        if not name:
            raise MCPToolInputError("name_required")
        normalized["name"] = name
    for field in ("description", "image_file_id"):
        if field in patch:
            normalized[field] = (str(patch.get(field) or "").strip() or None)
    if "price" in patch:
        normalized["price"] = _float_arg(patch, "price", 0, minimum=0, maximum=1_000_000)
    if "currency" in patch:
        currency = str(patch.get("currency") or "rubles")
        if currency not in {"rubles", "gems", "coins"}:
            raise MCPToolInputError("invalid_currency")
        normalized["currency"] = currency
    if "rewards" in patch:
        normalized["rewards"] = _normalize_shop_set_rewards(patch.get("rewards"))
    if "is_active" in patch:
        normalized["is_active"] = bool(patch.get("is_active"))
    if not normalized:
        raise MCPToolInputError("empty_shop_set_patch")
    return normalized


def _normalize_promocode_payload(data: dict[str, Any], *, require_reward: bool) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if "code" in data:
        code = str(data.get("code") or "").strip().upper()
        if not code:
            raise MCPToolInputError("code_required")
        normalized["code"] = code
    if "type" in data:
        promo_type = str(data.get("type") or "permanent").strip()
        if promo_type not in PROMOCODE_TYPES:
            raise MCPToolInputError("invalid_type")
        normalized["type"] = promo_type
    for field in ("reward_gems", "reward_coins", "reward_keys"):
        if field in data:
            normalized[field] = _int_arg(data, field, 0, minimum=0, maximum=1_000_000)
    if "reward_extrapass" in data:
        normalized["reward_extrapass"] = bool(data.get("reward_extrapass"))
    if "expires_at" in data:
        normalized["expires_at"] = _parse_optional_datetime_arg(data.get("expires_at"), "invalid_expires_at")
    if "is_active" in data:
        normalized["is_active"] = bool(data.get("is_active"))
    if require_reward:
        if "code" not in normalized:
            raise MCPToolInputError("code_required")
        normalized.setdefault("type", "permanent")
        normalized.setdefault("reward_gems", 0)
        normalized.setdefault("reward_coins", 0)
        normalized.setdefault("reward_keys", 0)
        normalized.setdefault("reward_extrapass", False)
        if not any((normalized["reward_gems"], normalized["reward_coins"], normalized["reward_keys"], normalized["reward_extrapass"])):
            raise MCPToolInputError("promocode_reward_required")
    if not normalized:
        raise MCPToolInputError("no_valid_fields")
    return normalized


def _normalize_reward_track_row(args: dict[str, Any]) -> dict[str, Any]:
    track_type = _str_arg(args, "track_type", "", max_length=80)
    reward_type = _str_arg(args, "reward_type", "", max_length=64)
    if not track_type:
        raise MCPToolInputError("track_type_required")
    if not reward_type:
        raise MCPToolInputError("reward_type_required")
    position = _int_arg(args, "position", 0, minimum=1, maximum=100_000)
    reward_amount = _int_arg(args, "reward_amount", 0, minimum=0, maximum=10_000_000)
    reward_meta = args.get("reward_meta")
    if reward_meta is None:
        reward_meta = {}
    if not isinstance(reward_meta, dict):
        raise MCPToolInputError("invalid_reward_meta")
    reward_config = _normalize_extrapass_reward_config(reward_type, reward_amount, reward_meta)
    _validate_extrapass_reward_config(
        reward_config["reward_type"],
        reward_config["reward_amount"],
        reward_config["reward_meta"],
    )
    return {
        "track_type": track_type,
        "position": position,
        "reward_type": reward_config["reward_type"],
        "reward_amount": reward_config["reward_amount"],
        "reward_meta": reward_config["reward_meta"],
        "extra_pass_required": bool(args.get("extra_pass_required", False)),
    }


def _normalize_reward_track_patch(patch: Any) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise MCPToolInputError("patch_required")
    allowed = {"track_type", "position", "reward_type", "reward_amount", "reward_meta", "extra_pass_required", "is_active"}
    unknown = set(patch) - allowed
    if unknown:
        raise MCPToolInputError("unsupported_reward_track_field")
    normalized: dict[str, Any] = {}
    if "track_type" in patch:
        normalized["track_type"] = _str_arg(patch, "track_type", "", max_length=80)
        if not normalized["track_type"]:
            raise MCPToolInputError("track_type_required")
    if "position" in patch:
        normalized["position"] = _int_arg(patch, "position", 0, minimum=1, maximum=100_000)
    if "reward_type" in patch:
        normalized["reward_type"] = _str_arg(patch, "reward_type", "", max_length=64)
        if not normalized["reward_type"]:
            raise MCPToolInputError("reward_type_required")
    if "reward_amount" in patch:
        normalized["reward_amount"] = _int_arg(patch, "reward_amount", 0, minimum=0, maximum=10_000_000)
    if "reward_meta" in patch:
        if patch["reward_meta"] is not None and not isinstance(patch["reward_meta"], dict):
            raise MCPToolInputError("invalid_reward_meta")
        normalized["reward_meta"] = patch["reward_meta"] or {}
    for field in ("extra_pass_required", "is_active"):
        if field in patch:
            normalized[field] = bool(patch.get(field))
    if not normalized:
        raise MCPToolInputError("empty_reward_track_patch")
    return normalized


def _known_match_mode_ids() -> set[str]:
    return {str(mode["mode_id"]) for mode in MATCH_MODE_CATALOG}


async def _match_modes_payload(db: Any) -> list[dict[str, Any]]:
    overrides = {
        str(row.get("mode_id")): row
        for row in (await _call_db(db, "get_match_mode_overrides", default=[])) or []
        if isinstance(row, dict) and row.get("mode_id")
    }
    modes = []
    for mode in MATCH_MODE_CATALOG:
        mode_id = str(mode["mode_id"])
        modes.append({
            **mode,
            "db_enabled": bool(overrides.get(mode_id, {}).get("enabled", mode.get("available", False))),
        })
    return modes


def _image_signature_matches(data: bytes, content_type: str) -> bool:
    if content_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if content_type == "image/jpeg":
        return data.startswith(b"\xff\xd8")
    if content_type == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def _normalize_cosmetic_slug(value: Any) -> str:
    slug = str(value or "").strip()
    if not slug:
        raise MCPToolInputError("slug_required")
    if not COSMETIC_SLUG_RE.fullmatch(slug):
        raise MCPToolInputError("invalid_slug")
    return slug


def _validate_cosmetic_item_type(value: Any) -> str:
    item_type = str(value or "").strip()
    if item_type not in COSMETIC_ITEM_TYPES:
        raise MCPToolInputError("invalid_item_type")
    return item_type


def _safe_cosmetic_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value or "").strip()).strip("-._:")
    return safe[:80] or "cosmetic"


def _safe_cosmetic_asset_path(value: Any, item_type: str) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if item_type == "title":
        raise MCPToolInputError("image_not_allowed_for_title")
    parts = COSMETIC_UPLOAD_DIRS.get(item_type)
    if not parts:
        raise MCPToolInputError("invalid_item_type")
    prefix = "/DesignAssets/" + "/".join(parts[:-1]) + "/"
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise MCPToolInputError("invalid_asset_path")
    if any(ch in text for ch in ("'", '"', "`", "<", ">")):
        raise MCPToolInputError("invalid_asset_path")
    if not text.startswith(prefix):
        raise MCPToolInputError("invalid_asset_path")
    relative = text[len("/DesignAssets/"):]
    parts_tuple = Path(relative).parts
    if any(not part or part in {".", ".."} for part in parts_tuple):
        raise MCPToolInputError("invalid_asset_path")
    suffix = Path(relative).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise MCPToolInputError("invalid_asset_path")
    return text


def _cosmetic_response(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    payload = dict(item)
    if "class_name" in payload and "class" not in payload:
        payload["class"] = payload.pop("class_name")
    return json_safe(payload)


def _normalize_cosmetic_payload(args: dict[str, Any], *, require_identity: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    item_type = None
    if "item_type" in args or require_identity:
        item_type = _validate_cosmetic_item_type(args.get("item_type"))
        payload["item_type"] = item_type
    else:
        item_type = str(args.get("existing_item_type") or "")
    if "slug" in args or require_identity:
        payload["slug"] = _normalize_cosmetic_slug(args.get("slug"))
    if "name" in args or require_identity:
        name = _str_arg(args, "name", "", max_length=160)
        if not name:
            raise MCPToolInputError("name_required")
        payload["name"] = name
    if "class" in args:
        payload["class_name"] = _str_arg(args, "class", "common", max_length=80) or "common"
    elif require_identity:
        payload["class_name"] = "common"
    if "asset_path" in args:
        if not item_type:
            raise MCPToolInputError("item_type_required")
        payload["asset_path"] = _safe_cosmetic_asset_path(args.get("asset_path"), item_type)
    if "media_type" in args:
        media_type = _str_arg(args, "media_type", "", max_length=20)
        if media_type not in {"image", "text"}:
            raise MCPToolInputError("invalid_media_type")
        if item_type == "title" and media_type != "text":
            raise MCPToolInputError("invalid_media_type")
        payload["media_type"] = media_type
    elif require_identity:
        payload["media_type"] = "text" if item_type == "title" else "image"
    if item_type == "title":
        payload.setdefault("asset_path", None)
        payload.setdefault("media_type", "text")
    if "has_sound" in args:
        payload["has_sound"] = bool(args.get("has_sound"))
    elif require_identity:
        payload["has_sound"] = False
    if "sort_order" in args:
        payload["sort_order"] = _int_arg(args, "sort_order", 0, minimum=0, maximum=1_000_000)
    elif require_identity:
        payload["sort_order"] = 0
    if "is_active" in args:
        payload["is_active"] = bool(args.get("is_active"))
    elif require_identity:
        payload["is_active"] = True
    return payload


def _cosmetic_upload_path(item_type: str, slug: str | None, content_type: str) -> tuple[Path, str]:
    ext = COSMETIC_IMAGE_CONTENT_TYPES[content_type]
    base = _safe_cosmetic_filename_part(slug or "cosmetic")
    filename = f"{base}-{uuid.uuid4().hex}{ext}"
    relative = Path(*COSMETIC_UPLOAD_DIRS[item_type], filename)
    return DESIGN_ASSETS_DIR / relative, f"/DesignAssets/{relative.as_posix()}"


def semantic_arguments(capability: AdminCapability, arguments: dict[str, Any]) -> dict[str, Any]:
    excluded = {"dry_run", "confirmation_id", "confirmation_token", "idempotency_key"}
    semantic: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in excluded:
            continue
        if key == "base64":
            text = str(value or "")
            semantic["base64_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            semantic["base64_length"] = len(text)
            continue
        semantic[key] = value
    return semantic


async def adapter_read_runtime_status(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    include_config = _bool_arg(args, "include_config", True)
    payload = {
        "status": "ok",
        "mcp_enabled": bool(getattr(app.get("settings"), "mcp_enabled", False)),
        "active_matches": len(app.get("active_matches", {}) or {}),
        "online_users": len(app.get("online_users", {}) or {}),
    }
    if include_config:
        payload["runtime_config"] = await _call_db(db, "get_runtime_config", default={})
    return json_safe(payload)


async def adapter_read_runtime_config(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    return json_safe(await _call_db(app["db"], "get_runtime_config", default={}))


async def adapter_search_players(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    data = await _call_db(
        app["db"],
        "search_admin_players",
        query=_str_arg(args, "query", "", max_length=80),
        status=_str_arg(args, "status", "all", max_length=24) or "all",
        league=args.get("league"),
        activity=_str_arg(args, "activity", "all", max_length=32) or "all",
        limit=_int_arg(args, "limit", 50, minimum=1, maximum=200),
        offset=_int_arg(args, "offset", 0, minimum=0, maximum=1_000_000),
        default={"players": [], "total": 0},
    )
    return json_safe(data)


async def adapter_get_player_detail(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    return json_safe(await _call_db(app["db"], "get_admin_player_detail", user_id, default={"error": "user_not_found"}))


async def adapter_list_shop_products(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    active_only = _bool_arg(args, "active_only", False)
    surface = args.get("surface")
    products = await _call_db(app["db"], "get_ruble_products", active_only=active_only, surface=surface, default=[])
    return json_safe({"products": products})


async def adapter_list_shop_sets(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    sets = await _call_db(app["db"], "get_shop_sets", active_only=_bool_arg(args, "active_only", False), default=[])
    return json_safe({"sets": [_shop_set_response(item) for item in (sets or [])]})


async def adapter_list_cosmetics(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    item_type = args.get("item_type")
    normalized_type = _validate_cosmetic_item_type(item_type) if item_type else None
    items = await _call_db(
        app["db"],
        "list_cosmetic_items",
        active_only=_bool_arg(args, "active_only", False),
        item_type=normalized_type,
        default=[],
    )
    return json_safe({"items": [_cosmetic_response(item) for item in (items or [])]})


async def adapter_get_cosmetic_detail(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    identity: Any = args.get("id")
    if not identity:
        identity = _normalize_cosmetic_slug(args.get("slug"))
    item = await _call_db(app["db"], "get_cosmetic_item", identity, default=None)
    if not item:
        raise MCPToolInputError("cosmetic_not_found")
    return json_safe({"item": _cosmetic_response(item)})


async def adapter_create_cosmetic(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    payload = _normalize_cosmetic_payload(args, require_identity=True)
    if payload["item_type"] != "title" and not payload.get("asset_path"):
        raise MCPToolInputError("asset_path_required")
    if payload["item_type"] == "title" and payload.get("media_type") not in {None, "text"}:
        raise MCPToolInputError("invalid_media_type")
    if _bool_arg(args, "dry_run", False):
        return json_safe({"dry_run": True, "cosmetic": _cosmetic_response(payload)})
    result = await _call_db(
        app["db"],
        "create_cosmetic_item",
        default={"success": False, "error": "cosmetic_create_unavailable"},
        **payload,
    )
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "cosmetic_create_failed"))
    return json_safe({"dry_run": False, "cosmetic": _cosmetic_response(result.get("cosmetic") or payload)})


async def adapter_update_cosmetic(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    cosmetic_id = _int_arg(args, "id", 0, minimum=1)
    current = await _call_db(app["db"], "get_cosmetic_item", cosmetic_id, default=None)
    if not current:
        raise MCPToolInputError("cosmetic_not_found")
    normalized_args = dict(args)
    normalized_args["existing_item_type"] = args.get("item_type") or current.get("item_type")
    patch = _normalize_cosmetic_payload(normalized_args, require_identity=False)
    patch.pop("slug", None) if "slug" not in args else None
    patch.pop("item_type", None) if "item_type" not in args else None
    if not patch:
        raise MCPToolInputError("empty_cosmetic_patch")
    if _bool_arg(args, "dry_run", False):
        preview = {**_cosmetic_response(current), **_cosmetic_response(patch)}
        return json_safe({"dry_run": True, "cosmetic": preview})
    result = await _call_db(
        app["db"],
        "update_cosmetic_item",
        cosmetic_id,
        default={"success": False, "error": "cosmetic_update_unavailable"},
        **patch,
    )
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "cosmetic_update_failed"))
    return json_safe({"dry_run": False, "cosmetic": _cosmetic_response(result.get("cosmetic") or {})})


async def adapter_delete_cosmetic(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    cosmetic_id = _int_arg(args, "id", 0, minimum=1)
    current = await _call_db(app["db"], "get_cosmetic_item", cosmetic_id, default=None)
    if not current:
        raise MCPToolInputError("cosmetic_not_found")
    if _bool_arg(args, "dry_run", False):
        return json_safe({"dry_run": True, "cosmetic": _cosmetic_response(current)})
    result = await _call_db(
        app["db"],
        "delete_cosmetic_item",
        cosmetic_id,
        default={"success": False, "error": "cosmetic_delete_unavailable"},
    )
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "cosmetic_delete_failed"))
    return json_safe({"dry_run": False, "deleted_id": cosmetic_id})


async def adapter_create_shop_set(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    name = _str_arg(args, "name", "", max_length=120)
    if not name:
        raise MCPToolInputError("name_required")
    price = _float_arg(args, "price", 0, minimum=0, maximum=1_000_000)
    currency = _str_arg(args, "currency", "rubles", max_length=16) or "rubles"
    if currency not in {"rubles", "gems", "coins"}:
        raise MCPToolInputError("invalid_currency")
    rewards = await _validated_shop_set_rewards(db, args.get("rewards"))
    description = _optional_str_arg(args, "description", max_length=1_000)
    image_file_id = _optional_str_arg(args, "image_file_id", max_length=500)
    payload = {
        "name": name,
        "description": description,
        "image_file_id": image_file_id,
        "price": price,
        "currency": currency,
        "created_by": admin_user_id,
        "rewards": rewards,
    }
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "set": payload}
    result = await _call_db(db, "create_shop_set", default={"success": False, "error": "shop_set_create_unavailable"}, **payload)
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "shop_set_create_failed"))
    return json_safe({"dry_run": False, "set_id": result.get("set_id"), "set": payload})


async def adapter_read_seasons_reward_tracks(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    seasons = await _call_db(db, "get_seasons", default=[])
    reward_tracks = await _call_db(db, "get_all_reward_tracks", default=[])
    if not _bool_arg(args, "include_inactive_rewards", True):
        reward_tracks = [
            row for row in reward_tracks or []
            if not isinstance(row, dict) or row.get("is_active", True) is not False
        ]
    reset_summaries = {}
    if _bool_arg(args, "include_reset_summaries", True):
        reset_summaries = await _call_db(db, "get_season_reset_summaries", default={})
    return json_safe({
        "seasons": seasons,
        "reward_tracks": reward_tracks,
        "reset_summaries": reset_summaries,
    })


async def adapter_create_extrapass_season_draft(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    preset_key = _str_arg(args, "preset_key", "blank", max_length=80) or "blank"
    if preset_key not in EXTRAPASS_PRESETS:
        raise MCPToolInputError("unknown_preset_key")
    if _bool_arg(args, "dry_run", False):
        seasons = await _call_db(app["db"], "get_seasons", default=[])
        return {
            "dry_run": True,
            "preset_key": preset_key,
            "current_seasons": len(seasons or []),
        }
    season = await _call_db(app["db"], "create_season_draft", preset_key=preset_key, default={"error": "season_draft_unavailable"})
    if not isinstance(season, dict) or season.get("error"):
        raise MCPToolInputError(str((season or {}).get("error") or "season_draft_failed"))
    return json_safe({"dry_run": False, "season": season})


async def adapter_patch_extrapass_season(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    season_id = _int_arg(args, "season_id", 0, minimum=1)
    existing = await _call_db(db, "get_season_by_id", season_id, default=None)
    if not existing:
        raise MCPToolInputError("season_not_found")
    patch = _normalize_extrapass_season_patch(args.get("patch"), dict(existing))
    seasons = await _call_db(db, "get_seasons", default=[])
    candidate = {**dict(existing), **patch}
    if _extrapass_track_type_reuse_conflict(candidate, seasons or [], current_season_id=season_id):
        raise MCPToolInputError("season_track_type_reused")
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "season_id": season_id, "current": json_safe(existing), "patch": json_safe(patch), "candidate": json_safe(candidate)}
    season = await _call_db(db, "update_season", season_id, default={"error": "season_update_unavailable"}, admin_user_id=admin_user_id, **patch)
    if not isinstance(season, dict) or season.get("error"):
        raise MCPToolInputError(str((season or {}).get("error") or "season_update_failed"))
    return json_safe({"dry_run": False, "season": season})


async def adapter_import_extrapass_rewards(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    season_id = _int_arg(args, "season_id", 0, minimum=1)
    season = await _call_db(db, "get_season_by_id", season_id, default=None)
    if not season:
        raise MCPToolInputError("season_not_found")
    rows = _normalize_extrapass_reward_import_payload(args.get("tracks"), dict(season))
    for row in rows:
        await _validate_extrapass_reward_config_with_db(db, row)
    replace = _bool_arg(args, "replace", True)
    track_types = _extrapass_track_types(dict(season))
    if _bool_arg(args, "dry_run", False):
        counts: dict[str, int] = {}
        for row in rows:
            track_type = str(row["track_type"])
            counts[track_type] = counts.get(track_type, 0) + 1
        return {"dry_run": True, "season_id": season_id, "replace": replace, "track_types": track_types, "rows": len(rows), "counts": counts}
    if replace:
        created = await _call_db(db, "replace_reward_tracks", track_types, rows, default=None)
        if created is None:
            await _call_db(db, "clear_reward_tracks", track_types, default=0)
            created = []
            for row in rows:
                result = await _call_db(db, "create_reward_track", default={"error": "reward_track_create_unavailable"}, **row)
                if isinstance(result, dict) and result.get("error"):
                    raise MCPToolInputError(str(result["error"]))
                created.append(result)
    else:
        created = []
        for row in rows:
            result = await _call_db(db, "create_reward_track", default={"error": "reward_track_create_unavailable"}, **row)
            if isinstance(result, dict) and result.get("error"):
                raise MCPToolInputError(str(result["error"]))
            created.append(result)
    return json_safe({"dry_run": False, "season_id": season_id, "replace": replace, "imported": len(created or []), "tiers": created or []})


async def adapter_preview_extrapass_reset(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    season_id = _int_arg(args, "season_id", 0, minimum=1)
    sample_limit = _int_arg(args, "sample_limit", 200, minimum=0, maximum=1_000)
    season = await _call_db(db, "get_season_by_id", season_id, default=None)
    if not season:
        raise MCPToolInputError("season_not_found")
    if not bool(_normalize_extrapass_season(season).get("is_active")):
        raise MCPToolInputError("season_reset_requires_active_season")
    preview = await _call_db(db, "preview_season_reset", season_id, sample_limit=sample_limit, default={"error": "season_reset_preview_unavailable"})
    if isinstance(preview, dict) and preview.get("error"):
        raise MCPToolInputError(str(preview["error"]))
    return json_safe(preview)


async def adapter_execute_extrapass_reset(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    season_id = _int_arg(args, "season_id", 0, minimum=1)
    confirm_season_id = _int_arg(args, "confirm_season_id", 0, minimum=1)
    if confirm_season_id != season_id:
        raise MCPToolInputError("confirm_season_mismatch")
    reason = _str_arg(args, "reason", "", max_length=500) or None
    season = await _call_db(db, "get_season_by_id", season_id, default=None)
    if not season:
        raise MCPToolInputError("season_not_found")
    if not bool(_normalize_extrapass_season(season).get("is_active")):
        raise MCPToolInputError("season_reset_requires_active_season")
    preview = await _call_db(db, "preview_season_reset", season_id, sample_limit=200, default={})
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "season_id": season_id, "reason": reason, "preview": json_safe(preview)}
    reset = await _call_db(
        db,
        "execute_season_reset",
        default={"error": "season_reset_unavailable"},
        season_id=season_id,
        previous_season_id=None,
        trigger="admin",
        admin_user_id=admin_user_id,
        reason=reason,
        require_active=True,
    )
    if not isinstance(reset, dict) or reset.get("error"):
        raise MCPToolInputError(str((reset or {}).get("error") or "season_reset_failed"))
    return json_safe({"dry_run": False, "reset": reset})


async def adapter_set_extrapass_player_entitlement(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    mode = _str_arg(args, "mode", "", max_length=16)
    if mode not in {"inactive", "active", "ultra"}:
        raise MCPToolInputError("invalid_mode")
    reason = _str_arg(args, "reason", "", max_length=500) or None
    payload = {"target_user_id": user_id, "mode": mode, "reason": reason}
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, **payload}
    result = await _call_db(
        app["db"],
        "admin_set_extra_pass",
        admin_user_id,
        user_id,
        mode,
        days=None,
        reason=reason,
        default={"error": "extra_pass_set_unavailable"},
    )
    if not isinstance(result, dict) or result.get("error"):
        raise MCPToolInputError(str((result or {}).get("error") or "extra_pass_set_failed"))
    return json_safe({"dry_run": False, **payload, "result": result})


async def adapter_list_match_modes(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    overrides = {
        str(row.get("mode_id")): row
        for row in (await _call_db(db, "get_match_mode_overrides", default=[])) or []
        if isinstance(row, dict) and row.get("mode_id")
    }
    modes_by_id = {str(mode["mode_id"]): dict(mode) for mode in MATCH_MODE_CATALOG}
    for mode_id in overrides:
        modes_by_id.setdefault(mode_id, {"mode_id": mode_id, "label": mode_id, "available": False})
    modes = []
    for mode_id in sorted(modes_by_id):
        mode = modes_by_id[mode_id]
        override = overrides.get(mode_id, {})
        modes.append({
            "mode_id": mode_id,
            "label": mode.get("label", mode_id),
            "available": bool(mode.get("available", False)),
            "db_enabled": bool(override.get("enabled", mode.get("available", False))),
        })
    return {"modes": modes}


async def adapter_read_push_status(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    sender = app.get("push_sender")
    configured = bool(getattr(sender, "configured", False)) if sender is not None else False
    init_error = getattr(sender, "init_error", None) if sender is not None else "push_sender_unavailable"
    devices = await _call_db(app["db"], "count_push_devices", platform="android", default=0)
    return {
        "configured": configured,
        "init_error": None if configured else init_error,
        "android_devices": int(devices or 0),
    }


async def adapter_read_analytics_overview(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    days = _int_arg(args, "days", 30, minimum=1, maximum=365)
    return json_safe(await _call_db(app["db"], "get_admin_analytics_overview", days=days, default={"days": days}))


async def adapter_patch_runtime_config(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    patch = _normalize_runtime_patch(args.get("patch"))
    dry_run = _bool_arg(args, "dry_run", False)
    if dry_run:
        current = await _call_db(app["db"], "get_runtime_config", default={})
        return {"dry_run": True, "current": json_safe(current), "patch": patch}
    applied = await _call_db(app["db"], "set_runtime_config", default={}, **patch)
    return {"dry_run": False, "applied": json_safe(applied), "patch": patch}


async def adapter_read_case_config(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    """Читать live-конфигурацию кейсов (game_settings case_config) с заполнением дефолтов."""
    return json_safe(await _call_db(app["db"], "get_case_config", default={}))


async def adapter_patch_case_config(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    """Применить partial-патч конфигурации кейсов со структурным deep-merge.

    dry_run=True возвращает текущую конфигурацию + нормализованный патч без записи.
    Применение идёт через Database.set_case_config (merge + validate + persist).
    """
    patch = _normalize_case_config_patch(args.get("patch"))
    if _bool_arg(args, "dry_run", False):
        current = await _call_db(app["db"], "get_case_config", default={})
        return {"dry_run": True, "current": json_safe(current), "patch": json_safe(patch)}
    applied = await _call_db(app["db"], "set_case_config", default={}, patch=patch)
    return {"dry_run": False, "applied": json_safe(applied), "patch": json_safe(patch)}


async def adapter_create_player_note(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    note = _str_arg(args, "note", "", max_length=2_000)
    if not note:
        raise MCPToolInputError("note_required")
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "target_user_id": user_id, "note_length": len(note)}
    result = await _call_db(app["db"], "admin_note_user", admin_user_id, user_id, note, default={"error": "note_unavailable"})
    return json_safe(result)


async def adapter_grant_player_resource(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    resource = _str_arg(args, "resource", "", max_length=24)
    amount = _int_arg(args, "amount", 0, minimum=1, maximum=1_000_000)
    reason = _str_arg(args, "reason", "", max_length=500)
    if resource not in {"gems", "coins", "keys", "stars"}:
        raise MCPToolInputError("invalid_resource")
    if not reason:
        raise MCPToolInputError("reason_required")
    if _bool_arg(args, "dry_run", False):
        return {
            "dry_run": True,
            "target_user_id": user_id,
            "resource": resource,
            "amount": amount,
            "reason": reason,
        }
    result = await _call_db(
        app["db"],
        "admin_adjust_resource",
        admin_user_id,
        user_id,
        resource,
        amount,
        reason=reason,
        default={"error": "resource_grant_unavailable"},
    )
    return json_safe({"dry_run": False, "result": result})


async def adapter_read_admin_config_summary(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    errors: dict[str, str] = {}

    async def section(name: str, default: Any, method_name: str, *method_args: Any, **kwargs: Any) -> Any:
        try:
            return await _call_db(db, method_name, *method_args, default=default, **kwargs)
        except Exception as exc:
            errors[name] = str(exc)
            return default

    cards = await section("cards", [], "get_cards_list")
    return json_safe({
        "match_modes": await _match_modes_payload(db),
        "promocodes_count": len(await section("promocodes", [], "get_promocodes_list") or []),
        "reward_tracks": await section("reward_tracks", [], "get_all_reward_tracks"),
        "season": await section("season", None, "get_active_season"),
        "shop_sets": await section("shop_sets", [], "get_shop_sets", active_only=False),
        "ruble_products": await section("ruble_products", [], "get_ruble_products", active_only=False),
        "runtime_config": await section("runtime_config", {}, "get_runtime_config"),
        "cards": [
            {
                "id": card.get("id"),
                "name": card.get("name") or str(card.get("id") or ""),
                "card_type": card.get("card_type", "unit"),
                "rarity": card.get("rarity", ""),
            }
            for card in (cards or [])
            if isinstance(card, dict)
        ],
        "squads": await section("squads", {}, "get_admin_squads_analytics", days=30),
        "errors": errors,
    })


async def adapter_read_tps_statistics(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    try:
        from tps_monitor import get_tps_monitor
        return json_safe(get_tps_monitor().get_statistics())
    except Exception as exc:
        return {"error": "tps_unavailable", "message": str(exc)}


async def adapter_read_analytics_section(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    section = _str_arg(args, "section", "", max_length=40)
    days = _int_arg(args, "days", 30, minimum=1, maximum=365)
    method_by_section = {
        "overview": "get_admin_analytics_overview",
        "revenue": "get_admin_revenue_analytics",
        "players": "get_admin_players_analytics",
        "battles": "get_admin_battle_analytics",
        "cards": "get_admin_cards_analytics",
        "heroes": "get_admin_heroes_analytics",
        "retention": "get_admin_retention_analytics",
        "onboarding": "get_admin_onboarding_analytics",
        "battle_actions": "get_admin_battle_actions_analytics",
    }
    if section == "economy":
        rows = await _call_db(
            db,
            "fetch",
            """
            SELECT event_type, resource, COUNT(*) AS cnt, COALESCE(SUM(amount), 0) AS total_amount
            FROM economy_events
            WHERE created_at >= NOW() - make_interval(days => $1::int)
            GROUP BY event_type, resource
            ORDER BY cnt DESC
            """,
            days,
            default=[],
        )
        events = [
            {
                "event_type": row["event_type"],
                "resource": row["resource"],
                "count": row["cnt"],
                "total_amount": float(row["total_amount"]),
            }
            for row in (rows or [])
        ]
        return json_safe({"section": section, "days": days, "data": {"events": events, "total_events": sum(e["count"] for e in events)}})
    method_name = method_by_section.get(section)
    if not method_name:
        raise MCPToolInputError("unsupported_analytics_section")
    data = await _call_db(db, method_name, days=days, default={})
    return json_safe({"section": section, "days": days, "data": data})


async def adapter_export_analytics_dataset(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    days = _int_arg(args, "days", 30, minimum=1, maximum=365)
    limit = _int_arg(args, "limit", 5_000, minimum=1, maximum=100_000)
    include_players = _bool_arg(args, "include_players", False)
    rows = await _call_db(
        app["db"],
        "export_train_v2_battle_dataset",
        days=days,
        limit=limit,
        include_players=include_players,
        default=[],
    )
    return json_safe({
        "format": "train_v2_admin_battle_action_jsonl_v2",
        "format_version": 2,
        "dataset_schema": "train_v3_battle_action_context_v1",
        "compatible_with": ["train_v2_admin_battle_action_jsonl_v1"],
        "days": days,
        "limit": limit,
        "include_players": include_players,
        "rows": rows or [],
        "row_count": len(rows or []),
    })


async def adapter_read_players_analytics(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    section = _str_arg(args, "section", "", max_length=40)
    days = _int_arg(args, "days", 30, minimum=1, maximum=365)
    db = app["db"]
    if section == "overview":
        data = await _call_db(db, "get_admin_players_overview", days=days, default={})
    elif section == "leagues":
        data = await _call_db(db, "get_admin_players_leagues", default={})
    elif section == "activity":
        data = await _call_db(db, "get_admin_players_activity", days=days, default={})
    else:
        raise MCPToolInputError("unsupported_players_section")
    return json_safe({"section": section, "days": days, "data": data})


async def adapter_ban_player_account(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    _require_self_ban_confirmation(admin_user_id, user_id, args)
    reason = _required_reason_arg(args)
    until = _parse_optional_datetime_arg(args.get("until"), "invalid_until_date")
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "target_user_id": user_id, "reason": reason, "until": json_safe(until)}
    result = await _call_db(app["db"], "admin_ban_user", admin_user_id, user_id, reason=reason, until=until, default={"error": "ban_unavailable"})
    if isinstance(result, dict) and result.get("error"):
        raise MCPToolInputError(str(result["error"]))
    return json_safe({"dry_run": False, "result": result})


async def adapter_unban_player_account(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    reason = _str_arg(args, "reason", "", max_length=500) or None
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "target_user_id": user_id, "reason": reason}
    result = await _call_db(app["db"], "admin_unban_user", admin_user_id, user_id, reason=reason, default={"error": "unban_unavailable"})
    if isinstance(result, dict) and result.get("error"):
        raise MCPToolInputError(str(result["error"]))
    return json_safe({"dry_run": False, "result": result})


async def adapter_warn_player_account(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    reason = _required_reason_arg(args)
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "target_user_id": user_id, "reason": reason}
    result = await _call_db(app["db"], "admin_warn_user", admin_user_id, user_id, reason=reason, default={"error": "warn_unavailable"})
    if isinstance(result, dict) and result.get("error"):
        raise MCPToolInputError(str(result["error"]))
    return json_safe({"dry_run": False, "result": result})


async def adapter_update_player_account(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    fields = _normalize_account_fields(args.get("fields"))
    if fields.get("status") == "banned":
        _require_self_ban_confirmation(admin_user_id, user_id, args)
    reason = _required_reason_arg(args)
    if _bool_arg(args, "dry_run", False):
        current = await _call_db(app["db"], "get_admin_player_detail", user_id, default=None)
        return {"dry_run": True, "target_user_id": user_id, "fields": fields, "reason": reason, "current": json_safe(current)}
    result = await _call_db(app["db"], "admin_update_user_account", admin_user_id, user_id, fields=fields, reason=reason, default={"error": "account_update_unavailable"})
    if isinstance(result, dict) and result.get("error"):
        raise MCPToolInputError(str(result["error"]))
    return json_safe({"dry_run": False, "result": result})


async def adapter_delete_player_account(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    user_id = _int_arg(args, "user_id", 0, minimum=1)
    confirm_user_id = _int_arg(args, "confirm_user_id", 0, minimum=1)
    if confirm_user_id != user_id:
        raise MCPToolInputError("confirm_user_mismatch")
    if int(admin_user_id) == int(user_id):
        raise MCPToolInputError("self_delete_not_allowed")
    reason = _required_reason_arg(args)
    current = await _call_db(app["db"], "get_admin_player_detail", user_id, default=None)
    if not current or (isinstance(current, dict) and current.get("error")):
        raise MCPToolInputError("user_not_found")
    summary = {
        "user_id": user_id,
        "username": current.get("account", {}).get("username") if isinstance(current.get("account"), dict) else current.get("username"),
        "status": current.get("account", {}).get("status") if isinstance(current.get("account"), dict) else current.get("status"),
        "reason": reason,
    }
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "target": json_safe(summary)}
    deleted = await _call_db(app["db"], "delete_user", user_id, default=False)
    if not deleted:
        raise MCPToolInputError("player_delete_failed")
    return {"dry_run": False, "deleted": True, "target": json_safe(summary)}


async def adapter_read_ruble_product_options(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    sets = await _call_db(app["db"], "get_shop_sets", active_only=False, default=[])
    return json_safe({
        "item_types": sorted(RUBLE_PRODUCT_ITEM_TYPES),
        "currencies": ["rubles", "gems", "coins"],
        "package_types": {
            "gems_package": [
                {
                    "value": package_id,
                    "price": package.get("price"),
                    "gems": package.get("gems"),
                    "one_time": bool(package.get("one_time")),
                }
                for package_id, package in GEM_PACKAGES.items()
            ]
        },
        "shop_sets": [
            {
                "id": row.get("id"),
                "name": row.get("name") or f"Set #{row.get('id')}",
                "price": row.get("price"),
                "currency": row.get("currency"),
                "is_active": row.get("is_active", True),
            }
            for row in (sets or [])
            if isinstance(row, dict)
        ],
    })


async def adapter_get_ruble_product_detail(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    code = _normalize_ruble_product_code(args.get("code"))
    product = await _call_db(app["db"], "get_ruble_product", code, default=None)
    if not product:
        raise MCPToolInputError("product_not_found")
    return json_safe({"product": product})


async def adapter_create_ruble_product(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    payload = await _normalize_ruble_product_payload(db, args, require_identity=True)
    if await _call_db(db, "get_ruble_product", str(payload["code"]), default=None):
        raise MCPToolInputError("product_code_exists")
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "product": json_safe(payload)}
    result = await _call_db(db, "create_ruble_product", default={"success": False, "error": "product_create_unavailable"}, **payload)
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "product_create_failed"))
    return json_safe({"dry_run": False, "result": result, "product": payload})


async def adapter_update_ruble_product(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    identity = _ruble_product_identity(args)
    existing = await _get_ruble_product_for_identity(db, identity)
    if not existing:
        raise MCPToolInputError("product_not_found")
    payload = await _normalize_ruble_product_payload(db, args, require_identity=False, existing=existing)
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "identity": identity, "current": json_safe(existing), "patch": json_safe(payload)}
    result = await _call_db(db, "update_ruble_product", identity, default={"success": False, "error": "product_update_unavailable"}, **payload)
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "product_update_failed"))
    return json_safe({"dry_run": False, "identity": identity, "result": result})


async def adapter_delete_ruble_product(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    identity = _ruble_product_identity(args)
    current = await _call_db(app["db"], "get_ruble_product", identity, default=None) if isinstance(identity, str) else None
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "identity": identity, "current": json_safe(current)}
    result = await _call_db(app["db"], "delete_ruble_product", identity, default={"success": False, "error": "product_delete_unavailable"})
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "product_delete_failed"))
    return json_safe({"dry_run": False, "identity": identity, "result": result})


async def adapter_get_shop_set_detail(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    set_id = _int_arg(args, "set_id", 0, minimum=1)
    shop_set = await _call_db(app["db"], "get_shop_set", set_id, default=None)
    if not shop_set:
        raise MCPToolInputError("shop_set_not_found")
    return json_safe({"set": _shop_set_response(shop_set)})


async def adapter_update_shop_set(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    set_id = _int_arg(args, "set_id", 0, minimum=1)
    current = await _call_db(db, "get_shop_set", set_id, default=None)
    if not current:
        raise MCPToolInputError("shop_set_not_found")
    patch = _normalize_shop_set_patch(args.get("patch"))
    if "rewards" in patch:
        patch["rewards"] = await _validated_shop_set_rewards(db, patch["rewards"])
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "set_id": set_id, "current": json_safe(current), "patch": json_safe(patch)}
    result = await _call_db(db, "update_shop_set", set_id, default={"success": False, "error": "shop_set_update_unavailable"}, **patch)
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "shop_set_update_failed"))
    return json_safe({"dry_run": False, "set_id": set_id, "result": result})


async def adapter_delete_shop_set(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    set_id = _int_arg(args, "set_id", 0, minimum=1)
    current = await _call_db(db, "get_shop_set", set_id, default=None)
    if not current:
        raise MCPToolInputError("shop_set_not_found")
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "set_id": set_id, "current": json_safe(current)}
    result = await _call_db(db, "delete_shop_set", set_id, default={"success": False, "error": "shop_set_delete_unavailable"})
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "shop_set_delete_failed"))
    return json_safe({"dry_run": False, "set_id": set_id, "result": result})


async def adapter_list_promocodes(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    created_by = args.get("created_by")
    if created_by is not None:
        created_by = _int_arg(args, "created_by", 0, minimum=1)
    promocodes = await _call_db(app["db"], "get_promocodes_list", created_by=created_by, default=[])
    return json_safe({"promocodes": promocodes})


async def adapter_create_promocode(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    payload = _normalize_promocode_payload(args, require_reward=True)
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "promocode": json_safe(payload)}
    result = await _call_db(
        app["db"],
        "create_promocode",
        default={"success": False, "error": "promocode_create_unavailable"},
        created_by=admin_user_id,
        **payload,
    )
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "promocode_create_failed"))
    return json_safe({"dry_run": False, "result": result, "promocode": payload})


async def adapter_update_promocode(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    promocode_id = _int_arg(args, "id", 0, minimum=1)
    current = await _call_db(db, "get_promocode", promocode_id, default=None)
    if not current:
        raise MCPToolInputError("promocode_not_found")
    payload = _normalize_promocode_payload({k: v for k, v in args.items() if k != "id"}, require_reward=False)
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "id": promocode_id, "current": json_safe(current), "patch": json_safe(payload)}
    result = await _call_db(db, "update_promocode", promocode_id, default={"success": False, "error": "promocode_update_unavailable"}, **payload)
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "promocode_update_failed"))
    return json_safe({"dry_run": False, "id": promocode_id, "result": result})


async def adapter_delete_promocode(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    promocode_id = _int_arg(args, "id", 0, minimum=1)
    current = await _call_db(app["db"], "get_promocode", promocode_id, default=None)
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "id": promocode_id, "current": json_safe(current)}
    result = await _call_db(app["db"], "delete_promocode", promocode_id, default={"success": False, "error": "promocode_delete_unavailable"})
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "promocode_delete_failed"))
    return json_safe({"dry_run": False, "id": promocode_id, "result": result})


async def adapter_set_match_mode_availability(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    mode_id = _str_arg(args, "mode_id", "", max_length=120)
    if mode_id not in _known_match_mode_ids():
        raise MCPToolInputError("unknown_mode_id")
    enabled = _bool_arg(args, "enabled", True)
    current_modes = await _match_modes_payload(db)
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "mode_id": mode_id, "enabled": enabled, "current_modes": current_modes}
    await _call_db(db, "set_match_mode_enabled", mode_id, enabled, default=None)
    return json_safe({"dry_run": False, "mode_id": mode_id, "enabled": enabled, "modes": await _match_modes_payload(db)})


async def adapter_broadcast_app_update_push(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    sender = app.get("push_sender")
    limit = _int_arg(args, "limit", 10_000, minimum=1, maximum=50_000)
    payload = build_android_push_payload(
        "app_update",
        "app_update_required",
        {
            "title": _optional_str_arg(args, "title", max_length=120),
            "body": _optional_str_arg(args, "body", max_length=500),
            "url": _optional_str_arg(args, "url", max_length=500),
        },
    )
    devices = await _call_db(app["db"], "count_push_devices", platform="android", default=0)
    payload_preview = {"title": payload.title, "body": payload.body, "data": payload.data}
    if _bool_arg(args, "dry_run", False):
        return {
            "dry_run": True,
            "configured": bool(getattr(sender, "configured", False)) if sender is not None else False,
            "init_error": getattr(sender, "init_error", None) if sender is not None else "push_sender_unavailable",
            "devices": int(devices or 0),
            "limit": limit,
            "payload": payload_preview,
        }
    if sender is None:
        raise MCPToolInputError("push_sender_unavailable")
    if not getattr(sender, "configured", False):
        raise MCPToolInputError("push_sender_not_configured")
    result = await send_android_broadcast(db=app["db"], push_sender=sender, payload=payload, platform="android", limit=limit)
    return json_safe({"dry_run": False, "devices": result.total, "sent": result.sent, "failed": result.failed})


async def adapter_read_reward_tracks(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    track_type = _str_arg(args, "track_type", "", max_length=80)
    active_only = _bool_arg(args, "active_only", False)
    limit = _int_arg(args, "limit", 1_000, minimum=1, maximum=2_000)
    tracks = await _call_db(app["db"], "get_all_reward_tracks", default=[])
    filtered = [
        track for track in (tracks or [])
        if isinstance(track, dict)
        and (not track_type or str(track.get("track_type") or "") == track_type)
        and (not active_only or bool(track.get("is_active", True)))
    ]
    return json_safe({"tracks": filtered[:limit], "total": len(filtered), "truncated": len(filtered) > limit})


async def adapter_create_reward_track(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    row = _normalize_reward_track_row(args)
    await _validate_extrapass_reward_config_with_db(app["db"], row)
    if _bool_arg(args, "dry_run", False):
        existing = await _call_db(app["db"], "get_reward_track_entries", row["track_type"], row["position"], default=[])
        return {"dry_run": True, "row": row, "existing": json_safe(existing)}
    result = await _call_db(app["db"], "create_reward_track", default={"error": "reward_track_create_unavailable"}, **row)
    if isinstance(result, dict) and result.get("error"):
        raise MCPToolInputError(str(result["error"]))
    return json_safe({"dry_run": False, "tier": result})


async def adapter_patch_reward_track(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    reward_id = _int_arg(args, "id", 0, minimum=1)
    current = await _call_db(app["db"], "get_reward_track_by_id", reward_id, default=None)
    if not current:
        raise MCPToolInputError("reward_track_not_found")
    patch = _normalize_reward_track_patch(args.get("patch"))
    merged = {**dict(current), **patch}
    await _validate_extrapass_reward_config_with_db(app["db"], merged)
    if all(key in merged and merged.get(key) not in (None, "") for key in ("reward_type", "reward_amount")):
        reward_meta = merged.get("reward_meta") if isinstance(merged.get("reward_meta"), dict) else {}
        reward_config = _normalize_extrapass_reward_config(
            str(merged.get("reward_type") or ""),
            int(merged.get("reward_amount") or 0),
            reward_meta,
        )
        patch["reward_type"] = reward_config["reward_type"]
        patch["reward_amount"] = reward_config["reward_amount"]
        # Only propagate the normalized reward_meta when the caller actually supplied one
        # (either via patch or pre-existing on the row). Avoids silently overwriting an
        # existing meta when the patch was meant to change only type/amount.
        if "reward_meta" in patch or "reward_meta" in current:
            patch["reward_meta"] = reward_config["reward_meta"]
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "id": reward_id, "current": json_safe(current), "patch": json_safe(patch)}
    result = await _call_db(app["db"], "update_reward_track", reward_id, default={"error": "reward_track_update_unavailable"}, **patch)
    if isinstance(result, dict) and result.get("error"):
        raise MCPToolInputError(str(result["error"]))
    return json_safe({"dry_run": False, "tier": result})


async def adapter_delete_reward_track(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    reward_id = _int_arg(args, "id", 0, minimum=1)
    current = await _call_db(app["db"], "get_reward_track_by_id", reward_id, default=None)
    if not current:
        raise MCPToolInputError("reward_track_not_found")
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "id": reward_id, "current": json_safe(current)}
    result = await _call_db(app["db"], "delete_reward_track", reward_id, default=False)
    if result is False or (isinstance(result, dict) and result.get("error")):
        raise MCPToolInputError(str(result.get("error") if isinstance(result, dict) else "reward_track_delete_failed"))
    return {"dry_run": False, "id": reward_id, "success": True}


async def adapter_list_catalog_cards(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    cards = await _call_db(app["db"], "get_cards_list", default=[])
    return json_safe({"cards": cards})


async def adapter_create_catalog_card(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    name = _str_arg(args, "name", "", max_length=120)
    if not name:
        raise MCPToolInputError("name_required")
    rarity = _str_arg(args, "rarity", "common", max_length=40)
    if rarity not in CATALOG_CARD_RARITIES:
        raise MCPToolInputError("invalid_rarity")
    mechanics = args.get("mechanics") or []
    if not isinstance(mechanics, list):
        raise MCPToolInputError("invalid_mechanics")
    payload = {
        "name": name,
        "description": _str_arg(args, "description", "", max_length=2_000),
        "rarity": rarity,
        "power": _int_arg(args, "power", 0, minimum=0, maximum=1_000_000),
        "image_file_id": _optional_str_arg(args, "image_file_id", max_length=500),
        "created_by": admin_user_id,
        "mana_cost": _int_arg(args, "mana_cost", 3, minimum=0, maximum=100),
        "base_attack": _int_arg(args, "base_attack", 100, minimum=0, maximum=1_000_000),
        "base_hp": _int_arg(args, "base_hp", 100, minimum=0, maximum=1_000_000),
        "mechanics": [str(item) for item in mechanics],
        "card_type": _str_arg(args, "card_type", "warrior", max_length=80) or "warrior",
    }
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "card": payload}
    result = await _call_db(app["db"], "create_card", default={"success": False, "error": "card_create_unavailable"}, **payload)
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "card_create_failed"))
    return json_safe({"dry_run": False, "result": result, "card": payload})


async def adapter_set_admin_card_collection(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    action = _str_arg(args, "action", "", max_length=40)
    if action not in {"add_all", "delete_all"}:
        raise MCPToolInputError("invalid_action")
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "admin_user_id": admin_user_id, "action": action}
    method_name = "add_all_cards_to_user" if action == "add_all" else "delete_all_user_cards"
    result = await _call_db(app["db"], method_name, admin_user_id, default={"success": False, "error": "card_collection_action_unavailable"})
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "card_collection_action_failed"))
    return json_safe({"dry_run": False, "action": action, "result": result})


async def adapter_list_catalog_items(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    items = await _call_db(app["db"], "get_items_list", default=[])
    return json_safe({"items": items})


async def adapter_create_catalog_item(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    name = _str_arg(args, "name", "", max_length=120)
    if not name:
        raise MCPToolInputError("name_required")
    rarity = _str_arg(args, "rarity", "common", max_length=40)
    if rarity not in CATALOG_ITEM_RARITIES:
        raise MCPToolInputError("invalid_rarity")
    payload = {
        "name": name,
        "description": _str_arg(args, "description", "", max_length=2_000),
        "rarity": rarity,
        "power": _int_arg(args, "power", 0, minimum=0, maximum=1_000_000),
        "image_file_id": _optional_str_arg(args, "image_file_id", max_length=500),
        "created_by": admin_user_id,
    }
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "item": payload}
    result = await _call_db(app["db"], "create_item", default={"success": False, "error": "item_create_unavailable"}, **payload)
    if not isinstance(result, dict) or not result.get("success"):
        raise MCPToolInputError(str((result or {}).get("error") or "item_create_failed"))
    return json_safe({"dry_run": False, "result": result, "item": payload})


async def adapter_toggle_stars_test_mode(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    current = bool(app.get("stars_test_mode", False))
    enabled = bool(args["enabled"]) if "enabled" in args else not current
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "current": current, "candidate": enabled}
    app["stars_test_mode"] = enabled
    return {"dry_run": False, "previous": current, "stars_test_mode": enabled}


async def adapter_read_squads_section(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    section = _str_arg(args, "section", "", max_length=40)
    if section == "analytics":
        days = _int_arg(args, "days", 30, minimum=1, maximum=365)
        return json_safe({"section": section, "data": await _call_db(db, "get_admin_squads_analytics", days=days, default={})})
    if section == "list":
        data = await _call_db(
            db,
            "search_admin_squads",
            query=_str_arg(args, "query", "", max_length=80) or None,
            filter_type=_str_arg(args, "filter", "all", max_length=40) or "all",
            sort=_str_arg(args, "sort", "cbrp", max_length=40) or "cbrp",
            limit=_int_arg(args, "limit", 50, minimum=1, maximum=200),
            offset=_int_arg(args, "offset", 0, minimum=0, maximum=1_000_000),
            default={"squads": [], "total": 0},
        )
        return json_safe({"section": section, "data": data})
    if section == "detail":
        clan_id = _str_arg(args, "clan_id", "", max_length=80)
        if not clan_id:
            raise MCPToolInputError("clan_id_required")
        clan = await _call_db(db, "resolve_clan_identifier", clan_id, default=None)
        if not clan:
            raise MCPToolInputError("clan_not_found")
        data = await _call_db(db, "get_admin_squad_detail", int(clan["id"]), default={"error": "squad_detail_unavailable"})
        return json_safe({"section": section, "clan": clan, "data": data})
    if section == "config":
        return json_safe({"section": section, "data": await _call_db(db, "get_squad_runtime_config", default={})})
    raise MCPToolInputError("unsupported_squads_section")


async def _resolve_squad_or_raise(db: Any, clan_id: Any) -> dict[str, Any]:
    text = str(clan_id or "").strip()
    if not text:
        raise MCPToolInputError("clan_id_required")
    clan = await _call_db(db, "resolve_clan_identifier", text, default=None)
    if not clan:
        raise MCPToolInputError("clan_not_found")
    return dict(clan)


async def adapter_execute_squad_action(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    db = app["db"]
    action = _str_arg(args, "action", "", max_length=40)
    reason = _str_arg(args, "reason", "", max_length=500) or None
    dry_run = _bool_arg(args, "dry_run", False)

    if action == "create":
        owner_id = _int_arg(args, "owner_id", 0, minimum=1)
        name = _str_arg(args, "name", "", max_length=120)
        tag = _str_arg(args, "tag", "", max_length=16).upper()
        if not owner_id or not name or not tag:
            raise MCPToolInputError("owner_name_tag_required")
        payload = {
            "owner_id": owner_id,
            "name": name,
            "tag": tag,
            "description": _str_arg(args, "description", "", max_length=1_000),
            "clan_type": _str_arg(args, "type", "open", max_length=40) or "open",
            "min_trophies": _int_arg(args, "min_trophies", 0, minimum=0, maximum=1_000_000),
        }
        if dry_run:
            return {"dry_run": True, "action": action, "payload": payload}
        clan = await _call_db(db, "create_clan", default={"error": "squad_create_unavailable"}, **payload)
        if isinstance(clan, dict) and clan.get("error"):
            raise MCPToolInputError(str(clan["error"]))
        if isinstance(clan, dict) and clan.get("id"):
            await _call_db(db, "_log_clan_activity", int(clan["id"]), "admin_create", "Squad created by MCP admin", user_id=admin_user_id, default=None)
        return json_safe({"dry_run": False, "action": action, "clan": clan})

    if action == "config_set":
        key = _str_arg(args, "key", "", max_length=120)
        if key not in SQUAD_SETTINGS_DEFAULTS:
            raise MCPToolInputError("invalid_setting_key")
        if "value" not in args:
            raise MCPToolInputError("value_required")
        current = await _call_db(db, "get_squad_runtime_config", default={})
        if dry_run:
            return {"dry_run": True, "action": action, "key": key, "value": json_safe(args.get("value")), "current": json_safe(current)}
        await _call_db(db, "set_game_setting", key, args.get("value"), "Updated from MCP admin", default=None)
        return json_safe({"dry_run": False, "action": action, "data": await _call_db(db, "get_squad_runtime_config", default={})})

    if action == "process_weekly":
        if dry_run:
            return {"dry_run": True, "action": action, "preview": await _call_db(db, "get_admin_squads_analytics", days=7, default={})}
        result = await _call_db(db, "process_weekly_squad_cbrp", default={"error": "weekly_process_unavailable"})
        if isinstance(result, dict) and result.get("error"):
            raise MCPToolInputError(str(result["error"]))
        return json_safe({"dry_run": False, "action": action, "result": result})

    if action == "request":
        request_id = _int_arg(args, "request_id", 0, minimum=1)
        request_action = _str_arg(args, "request_action", "", max_length=16)
        if request_action not in {"accept", "reject"}:
            raise MCPToolInputError("invalid_request_action")
        if dry_run:
            return {"dry_run": True, "action": action, "request_id": request_id, "request_action": request_action}
        method_name = "accept_join_request" if request_action == "accept" else "reject_join_request"
        result = await _call_db(db, method_name, request_id, admin_user_id, default={"error": "request_action_unavailable"})
        if isinstance(result, dict) and result.get("error"):
            raise MCPToolInputError(str(result["error"]))
        return json_safe({"dry_run": False, "action": action, "request": result})

    clan = await _resolve_squad_or_raise(db, args.get("clan_id"))

    if action == "update":
        fields = args.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise MCPToolInputError("fields_required")
        if dry_run:
            return {"dry_run": True, "action": action, "clan": json_safe(clan), "fields": json_safe(fields), "reason": reason}
        result = await _call_db(db, "admin_update_squad", admin_user_id, int(clan["id"]), fields=fields, reason=reason, default={"error": "squad_update_unavailable"})
    elif action == "balance":
        resource = _str_arg(args, "resource", "", max_length=40)
        amount = _int_arg(args, "amount", 0, minimum=-1_000_000, maximum=1_000_000)
        if not resource or amount == 0:
            raise MCPToolInputError("resource_amount_required")
        if not reason:
            raise MCPToolInputError("reason_required")
        if dry_run:
            return {"dry_run": True, "action": action, "clan": json_safe(clan), "resource": resource, "amount": amount, "reason": reason}
        result = await _call_db(db, "admin_adjust_squad_balance", admin_user_id, int(clan["id"]), resource=resource, amount=amount, reason=reason, default={"error": "squad_balance_unavailable"})
    elif action == "member":
        member_action = _str_arg(args, "member_action", "", max_length=40)
        target_user_id = _int_arg(args, "target_user_id", 0, minimum=1)
        payload = {
            "action": member_action,
            "target_user_id": target_user_id,
            "personal_tokens": args.get("personal_tokens"),
        }
        if dry_run:
            return {"dry_run": True, "action": action, "clan": json_safe(clan), **payload}
        result = await _call_db(db, "admin_squad_member_action", admin_user_id, int(clan["id"]), default={"error": "squad_member_action_unavailable"}, **payload)
    elif action == "upgrade":
        upgrade_type = _str_arg(args, "upgrade_type", "", max_length=80)
        mode = _str_arg(args, "mode", "set", max_length=16) or "set"
        if not upgrade_type:
            raise MCPToolInputError("upgrade_type_required")
        squad_cfg = await _call_db(db, "get_squad_runtime_config", default={})
        valid_upgrades = (squad_cfg.get("squad_upgrades") if isinstance(squad_cfg, dict) else None) or SQUAD_SETTINGS_DEFAULTS.get("squad_upgrades", {})
        if upgrade_type not in valid_upgrades:
            raise MCPToolInputError("invalid_upgrade_type")
        level = _int_arg(args, "level", 0, minimum=0, maximum=1_000)
        if dry_run:
            return {"dry_run": True, "action": action, "clan": json_safe(clan), "upgrade_type": upgrade_type, "mode": mode, "level": level}
        if mode == "buy":
            result = await _call_db(db, "buy_clan_upgrade", int(clan["id"]), int(clan["owner_id"]), upgrade_type, default={"error": "squad_upgrade_buy_unavailable"})
        else:
            await _call_db(
                db,
                "execute",
                """
                INSERT INTO clan_upgrades (clan_id, upgrade_type, level)
                VALUES ($1, $2, $3)
                ON CONFLICT (clan_id, upgrade_type) DO UPDATE SET level = EXCLUDED.level
                """,
                int(clan["id"]),
                upgrade_type,
                level,
                default=None,
            )
            await _call_db(db, "_log_clan_activity", int(clan["id"]), "admin_upgrade", f"MCP admin set {upgrade_type} = {level}", user_id=admin_user_id, default=None)
            result = {"upgrade_type": upgrade_type, "level": level}
    elif action == "delete":
        if dry_run:
            return {"dry_run": True, "action": action, "clan": json_safe(clan)}
        await _call_db(db, "_log_clan_activity", int(clan["id"]), "admin_delete", "Squad deleted by MCP admin", user_id=admin_user_id, default=None)
        deleted = await _call_db(db, "delete_clan", int(clan["id"]), default=False)
        result = {"deleted": deleted}
    else:
        raise MCPToolInputError("unsupported_squad_action")

    if isinstance(result, dict) and result.get("error"):
        raise MCPToolInputError(str(result["error"]))
    return json_safe({"dry_run": False, "action": action, "clan_id": clan.get("id"), "result": result})


async def adapter_upload_product_image(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    content_type = _str_arg(args, "content_type", "", max_length=80)
    if content_type not in PRODUCT_IMAGE_CONTENT_TYPES:
        raise MCPToolInputError("invalid_image_type")
    try:
        data = base64.b64decode(str(args.get("base64") or ""), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MCPToolInputError("invalid_base64") from exc
    if not data or len(data) > MAX_PRODUCT_IMAGE_BYTES:
        raise MCPToolInputError("file_too_large")
    if not _image_signature_matches(data, content_type):
        raise MCPToolInputError("invalid_image_signature")
    ext = PRODUCT_IMAGE_CONTENT_TYPES[content_type]
    filename = f"{uuid.uuid4()}{ext}"
    image_url = f"{RUBLE_PRODUCT_IMAGE_URL_PREFIX}{filename}"
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, "bytes": len(data), "content_type": content_type, "image_url": image_url}
    uploads_dir = Path(__file__).resolve().parents[1] / "extraShop" / "uploads" / "products"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    path = uploads_dir / filename
    path.write_bytes(data)
    return {"dry_run": False, "bytes": len(data), "content_type": content_type, "image_url": image_url}


async def adapter_upload_cosmetic_image(app: Any, admin_user_id: int, args: dict[str, Any]) -> dict[str, Any]:
    item_type = _validate_cosmetic_item_type(args.get("item_type"))
    if item_type == "title":
        raise MCPToolInputError("title_image_not_supported")
    content_type = _str_arg(args, "content_type", "", max_length=80)
    if content_type not in COSMETIC_IMAGE_CONTENT_TYPES:
        raise MCPToolInputError("invalid_image_type")
    try:
        data = base64.b64decode(str(args.get("base64") or ""), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MCPToolInputError("invalid_base64") from exc
    if not data or len(data) > MAX_PRODUCT_IMAGE_BYTES:
        raise MCPToolInputError("file_too_large")
    if not _image_signature_matches(data, content_type):
        raise MCPToolInputError("invalid_image_signature")
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    except Exception as exc:  # pragma: no cover - Pillow reports several image-specific exceptions.
        raise MCPToolInputError("invalid_image") from exc
    expected = COSMETIC_IMAGE_SIZES[item_type]
    if (width, height) != expected:
        raise MCPToolInputError(f"invalid_dimensions:{width}x{height}:expected:{expected[0]}x{expected[1]}")
    slug = args.get("slug")
    normalized_slug = _normalize_cosmetic_slug(slug) if slug else None
    path, asset_path = _cosmetic_upload_path(item_type, normalized_slug, content_type)
    payload = {
        "bytes": len(data),
        "content_type": content_type,
        "dimensions": {"width": width, "height": height},
        "asset_path": asset_path,
    }
    if _bool_arg(args, "dry_run", False):
        return {"dry_run": True, **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"dry_run": False, **payload}


ADAPTERS = {
    "adapter_read_runtime_status": adapter_read_runtime_status,
    "adapter_read_runtime_config": adapter_read_runtime_config,
    "adapter_search_players": adapter_search_players,
    "adapter_get_player_detail": adapter_get_player_detail,
    "adapter_list_shop_products": adapter_list_shop_products,
    "adapter_list_shop_sets": adapter_list_shop_sets,
    "adapter_list_cosmetics": adapter_list_cosmetics,
    "adapter_get_cosmetic_detail": adapter_get_cosmetic_detail,
    "adapter_create_cosmetic": adapter_create_cosmetic,
    "adapter_update_cosmetic": adapter_update_cosmetic,
    "adapter_delete_cosmetic": adapter_delete_cosmetic,
    "adapter_create_shop_set": adapter_create_shop_set,
    "adapter_read_seasons_reward_tracks": adapter_read_seasons_reward_tracks,
    "adapter_create_extrapass_season_draft": adapter_create_extrapass_season_draft,
    "adapter_patch_extrapass_season": adapter_patch_extrapass_season,
    "adapter_import_extrapass_rewards": adapter_import_extrapass_rewards,
    "adapter_preview_extrapass_reset": adapter_preview_extrapass_reset,
    "adapter_execute_extrapass_reset": adapter_execute_extrapass_reset,
    "adapter_set_extrapass_player_entitlement": adapter_set_extrapass_player_entitlement,
    "adapter_list_match_modes": adapter_list_match_modes,
    "adapter_read_push_status": adapter_read_push_status,
    "adapter_read_analytics_overview": adapter_read_analytics_overview,
    "adapter_patch_runtime_config": adapter_patch_runtime_config,
    "adapter_read_case_config": adapter_read_case_config,
    "adapter_patch_case_config": adapter_patch_case_config,
    "adapter_create_player_note": adapter_create_player_note,
    "adapter_grant_player_resource": adapter_grant_player_resource,
    "adapter_read_admin_config_summary": adapter_read_admin_config_summary,
    "adapter_read_tps_statistics": adapter_read_tps_statistics,
    "adapter_read_analytics_section": adapter_read_analytics_section,
    "adapter_export_analytics_dataset": adapter_export_analytics_dataset,
    "adapter_read_players_analytics": adapter_read_players_analytics,
    "adapter_ban_player_account": adapter_ban_player_account,
    "adapter_unban_player_account": adapter_unban_player_account,
    "adapter_warn_player_account": adapter_warn_player_account,
    "adapter_update_player_account": adapter_update_player_account,
    "adapter_delete_player_account": adapter_delete_player_account,
    "adapter_read_ruble_product_options": adapter_read_ruble_product_options,
    "adapter_get_ruble_product_detail": adapter_get_ruble_product_detail,
    "adapter_create_ruble_product": adapter_create_ruble_product,
    "adapter_update_ruble_product": adapter_update_ruble_product,
    "adapter_delete_ruble_product": adapter_delete_ruble_product,
    "adapter_get_shop_set_detail": adapter_get_shop_set_detail,
    "adapter_update_shop_set": adapter_update_shop_set,
    "adapter_delete_shop_set": adapter_delete_shop_set,
    "adapter_list_promocodes": adapter_list_promocodes,
    "adapter_create_promocode": adapter_create_promocode,
    "adapter_update_promocode": adapter_update_promocode,
    "adapter_delete_promocode": adapter_delete_promocode,
    "adapter_set_match_mode_availability": adapter_set_match_mode_availability,
    "adapter_broadcast_app_update_push": adapter_broadcast_app_update_push,
    "adapter_read_reward_tracks": adapter_read_reward_tracks,
    "adapter_create_reward_track": adapter_create_reward_track,
    "adapter_patch_reward_track": adapter_patch_reward_track,
    "adapter_delete_reward_track": adapter_delete_reward_track,
    "adapter_list_catalog_cards": adapter_list_catalog_cards,
    "adapter_create_catalog_card": adapter_create_catalog_card,
    "adapter_set_admin_card_collection": adapter_set_admin_card_collection,
    "adapter_list_catalog_items": adapter_list_catalog_items,
    "adapter_create_catalog_item": adapter_create_catalog_item,
    "adapter_toggle_stars_test_mode": adapter_toggle_stars_test_mode,
    "adapter_read_squads_section": adapter_read_squads_section,
    "adapter_execute_squad_action": adapter_execute_squad_action,
    "adapter_upload_product_image": adapter_upload_product_image,
    "adapter_upload_cosmetic_image": adapter_upload_cosmetic_image,
}
ADMIN_TOOL_ADAPTERS = ADAPTERS


async def execute_admin_capability(
    app: Any,
    *,
    capability: AdminCapability,
    admin_user_id: int,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    adapter = ADAPTERS.get(capability.adapter_function)
    if adapter is None:
        raise MCPToolInputError("adapter_not_available")
    result = await adapter(app, admin_user_id, dict(arguments or {}))
    return json_safe(result)
