from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json as _stdlib_json
import logging
import math
import os
import random
import string
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import bcrypt
import jwt as pyjwt
from aiohttp import web
import socketio

from bot.constants import ADMIN_ID
from infrastructure.config import DECK_SIZE, LEAGUE_CONFIG, get_settings
from infrastructure.database import Card, Database, RUNTIME_FEATURE_DEFAULTS, SQUAD_SETTINGS_DEFAULTS
from ai.bot_factory import BotGenerator
from ai.bot_ai import BotAI
from ai.bot_brain import BerserkInference
from battle_engine import BattleEngine, BattleEventEmitter
from core.state import ReplacementStatus
from infrastructure.matchmaking import Matchmaker
from infrastructure.match_modes import (
    EXTRA_ARENA_ROTATING_IDS,
    ROTATION_ANCHOR_EPOCH_SECONDS,
    ROTATION_ENABLED,
    ROTATION_INTERVAL_SECONDS,
    build_extra_arena_widget_payload,
    get_current_extra_arena_mode,
    get_extra_arena_mode_list,
    mode_unavailable_payload,
    resolve_canonical_mode_id,
    resolve_mode_config,
    serialize_mode_config,
)
from infrastructure.push_notifications import build_android_push_payload, send_android_broadcast
from infrastructure.case_system import (
    roll_tier_upgrade,
    process_case_opening,
    generate_case_rewards,
    get_user_case_pass_status,
    simulate_case_tap_results,
)
from infrastructure.payments_logic import process_successful_payment
from infrastructure.rustore_payments import resolve_rustore_product_id
from infrastructure.shop_config import (
    CASE_PACKS,
    SHOP_PRICES,
    GEM_PACKAGES,
    PARTICLES_COSTS,
    build_shop_catalog,
    order_particles_for_shop,
)
from web.extraid_handlers import register_handlers as register_extraid_handlers
from infrastructure.extraid_database import ExtraIDDatabase

WEBAPP_DIR = Path(__file__).resolve().parents[1] / "webapp"
DESIGN_ASSETS_DIR = Path(__file__).resolve().parents[1] / "DesignAssets"
EXTRA_SHOP_DIR = Path(__file__).resolve().parents[1] / "extraShop"
STATIC_ASSET_CACHE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}
COMMUNITY_UPLOADS_DIR = Path(__file__).resolve().parents[1] / "uploads" / "community"
COMMUNITY_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SQUAD_UPLOADS_DIR = Path(__file__).resolve().parents[1] / "uploads" / "squads"
SQUAD_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
# Единый URL-путь к изображениям карт, чтобы фронт и боевая логика не зависели
# от устаревших image_file_id.
CARD_IMAGE_URL_PREFIX = "/DesignAssets/Cards"
TRAINING_BOT_NAME = "🤖 Тренер"
TRAINING_BOT_TITLE = "extra-lr series"
TRAINING_BOT_TITLE_CLASS = "starter"
TRAINING_BOT_AVATAR_URL = "/DesignAssets/Arena/TrainingModeBotCosmetics/avatar.png"
TRAINING_BOT_BACKGROUND_URL = "/DesignAssets/Arena/TrainingModeBotCosmetics/ProfileBackground.png"

# Глобальный кеш активных боёв: match_id -> экземпляр движка
ACTIVE_MATCHES: dict[str, BattleEngine] = {}

# Глобальный эмиттер событий для всех боёв
BATTLE_EVENT_EMITTER = BattleEventEmitter()

# Словарь для отслеживания связей sid -> {match_id, user_id}
SID_TO_MATCH: dict[str, dict[str, Any]] = {}
DISCONNECT_TAKEOVER_MAX_WINDOW_SECONDS = 5.0
DISCONNECT_TAKEOVER_MIN_WINDOW_SECONDS = 1.0
DISCONNECT_TAKEOVER_FRACTION = 0.2
MATCH_DISCONNECT_STATES: dict[tuple[str, int], dict[str, Any]] = {}
MATCH_DISCONNECT_TASKS: dict[tuple[str, int], asyncio.Task] = {}
BOT_TASKS: dict[str, asyncio.Task] = {}
BOT_TASK_KEYS: dict[str, tuple[str, int]] = {}
BOT_VS_BOT_MARKERS: dict[str, int] = {}
ENDED_MATCH_IDS: set[str] = set()
ENDED_MATCH_TIMES: dict[str, float] = {}
FINISHED_MATCH_TTL_SECONDS = 600
ONLINE_USER_TTL_SECONDS = 45
CASE_KEY_ROLL_TTL_SECONDS = 300
CASE_KEY_ROLLS: dict[str, dict[str, Any]] = {}
ACTION_RESULT_TTL_SECONDS = 120
ACTION_RESULT_CACHE: dict[tuple[str, int, str], dict[str, Any]] = {}


def _mark_match_ended(match_id: Any) -> None:
    match_id_str = str(match_id or "")
    if not match_id_str:
        return
    ENDED_MATCH_IDS.add(match_id_str)
    ENDED_MATCH_TIMES[match_id_str] = time.time()


async def _mark_matchmaker_finished(app: web.Application, match_id: Any, winner_id: Any = None) -> None:
    matchmaker = app.get("matchmaker") if app else None
    marker = getattr(matchmaker, "mark_match_finished", None)
    if not callable(marker):
        return
    try:
        maybe_result = marker(str(match_id), winner_id=winner_id)
        if asyncio.iscoroutine(maybe_result):
            await maybe_result
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to mark matchmaker match as finished: match_id=%s",
            match_id,
            exc_info=True,
        )


def _prune_finished_match_runtime(now: float | None = None) -> None:
    now = now or time.time()
    expired = [
        match_id for match_id, ended_at in list(ENDED_MATCH_TIMES.items())
        if now - ended_at > FINISHED_MATCH_TTL_SECONDS
    ]
    for match_id in expired:
        engine = ACTIVE_MATCHES.get(match_id)
        if (
            engine is not None
            and not getattr(engine, "is_ended", False)
            and not getattr(engine, "rewards_granted", False)
            and not getattr(engine, "battle_end_processed", False)
        ):
            continue
        ACTIVE_MATCHES.pop(match_id, None)
        MATCH_SESSIONS.pop(match_id, None)
        MATCH_LOCKS.pop(match_id, None)
        ENDED_MATCH_IDS.discard(match_id)
        ENDED_MATCH_TIMES.pop(match_id, None)
        for sid, session_data in list(SID_TO_MATCH.items()):
            if str(session_data.get("match_id")) == match_id:
                SID_TO_MATCH.pop(sid, None)
        for key, task in list(MATCH_DISCONNECT_TASKS.items()):
            if key[0] == match_id:
                MATCH_DISCONNECT_TASKS.pop(key, None)
                if task and not task.done() and task is not asyncio.current_task():
                    task.cancel()
        for key in list(MATCH_DISCONNECT_STATES.keys()):
            if key[0] == match_id:
                MATCH_DISCONNECT_STATES.pop(key, None)
        bot_task = BOT_TASKS.pop(match_id, None)
        if bot_task and not bot_task.done() and bot_task is not asyncio.current_task():
            bot_task.cancel()
        BOT_TASK_KEYS.pop(match_id, None)
        BOT_VS_BOT_MARKERS.pop(match_id, None)


def _training_bot_profile_payload() -> dict[str, Any]:
    return {
        "name": TRAINING_BOT_NAME,
        "avatar_url": TRAINING_BOT_AVATAR_URL,
        "title": TRAINING_BOT_TITLE,
        "title_class": TRAINING_BOT_TITLE_CLASS,
        "background_url": TRAINING_BOT_BACKGROUND_URL,
        "trophies": 0,
        "clan": "",
        "extra_pass": None,
    }


def _decorate_training_bot_info(bot_info: dict[str, Any] | None) -> dict[str, Any]:
    decorated = dict(bot_info or {})
    profile = _training_bot_profile_payload()
    cosmetics = dict(decorated.get("cosmetics") or {})

    cosmetics["avatar"] = {
        **(cosmetics.get("avatar") or {}),
        "name": "Training Bot Avatar",
        "item_type": "avatar",
        "class": profile["title_class"],
        "asset_path": profile["avatar_url"],
    }
    cosmetics["title"] = {
        **(cosmetics.get("title") or {}),
        "item_type": "title",
        "class": profile["title_class"],
        "name": profile["title"],
    }
    cosmetics["profile_background"] = {
        **(cosmetics.get("profile_background") or {}),
        "name": "Training Bot Background",
        "item_type": "profile_background",
        "class": profile["title_class"],
        "asset_path": profile["background_url"],
    }

    decorated.update({
        "name": profile["name"],
        "avatar_url": profile["avatar_url"],
        "trophies": profile["trophies"],
        "clan": profile["clan"],
        "extra_pass": profile["extra_pass"],
        "cosmetics": cosmetics,
    })
    return decorated


async def _resolve_battle_profile(
    db_instance: Any,
    user_id: int,
    *,
    fallback_name: str | None = None,
) -> dict[str, Any]:
    """Resolve the visual/profile payload used by battle and prebattle screens."""
    profile: dict[str, Any] = {}
    try:
        loaded = await db_instance.get_user_profile(user_id)
        if loaded:
            profile = dict(loaded)
    except Exception:
        logging.getLogger(__name__).debug(
            "Failed to load battle profile for user_id=%s",
            user_id,
            exc_info=True,
        )

    name = (
        profile.get("display_name")
        or profile.get("custom_nickname")
        or profile.get("nickname")
        or profile.get("name")
        or profile.get("first_name")
        or profile.get("username")
        or fallback_name
        or f"Игрок {user_id}"
    )
    avatar_url = (
        profile.get("equipped_avatar_url")
        or profile.get("img")
        or profile.get("photo_url")
        or profile.get("avatar_file_id")
        or profile.get("avatar_url")
    )
    trophies = int(profile.get("trophies", 0) or 0)
    clan = profile.get("clan") or profile.get("clan_name") or ""

    title = ""
    title_class = ""
    try:
        equipped_title = await db_instance.get_equipped_title(user_id)
        if equipped_title:
            title = equipped_title.get("name", "") or ""
            title_class = equipped_title.get("class", "starter") or "starter"
    except Exception:
        logging.getLogger(__name__).debug(
            "Failed to load equipped title for user_id=%s",
            user_id,
            exc_info=True,
        )
    if not title:
        legacy_title = str(profile.get("title", "") or "").strip()
        if legacy_title and legacy_title not in {"Игрок", "Новичок"}:
            title = legacy_title
            title_class = "starter"

    background_url = None
    try:
        cosmetics = await db_instance.get_user_cosmetics(user_id)
        equipped = (cosmetics or {}).get("equipped", {})
        background = equipped.get("profile_background") or {}
        background_url = background.get("asset_path")
    except Exception:
        logging.getLogger(__name__).debug(
            "Failed to load equipped background for user_id=%s",
            user_id,
            exc_info=True,
        )

    extra_pass = profile.get("extra_pass")
    if extra_pass is None:
        try:
            extra_pass = await db_instance.fetchval(
                "SELECT extra_pass FROM users WHERE user_id = $1",
                user_id,
            )
        except Exception:
            extra_pass = None

    return {
        "name": name,
        "avatar_url": avatar_url,
        "background_url": background_url,
        "title": title,
        "title_class": title_class,
        "extra_pass": extra_pass,
        "trophies": trophies,
        "clan": clan,
        "raw_profile": profile,
    }


def _extra_pass_access(extra_pass: Any) -> dict[str, bool | str]:
    """Normalize ExtraPass tier checks for reward tracks and claims."""
    mode = str(extra_pass or "inactive").lower()
    has_ultra = mode == "ultra"
    has_extra_pass = mode in {"active", "ultra"}
    if not has_extra_pass:
        mode = "inactive"
    return {
        "mode": mode,
        "has_extra_pass": has_extra_pass,
        "has_ultra": has_ultra,
    }


def _reward_track_pass_unlocked(track_type: str, access: dict[str, bool | str]) -> bool:
    if track_type == "bp_ultra":
        return bool(access.get("has_ultra"))
    return bool(access.get("has_extra_pass"))


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
    "free_track_type": "bp_free",
    "pass_track_type": "bp_premium",
    "ultra_track_type": "bp_ultra",
    "pass_end_position": 40,
    "ultra_start_position": 41,
    "theme": {},
}


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = _stdlib_json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_extra_pass_season(record: Any) -> dict[str, Any]:
    raw = dict(record) if record else {}
    season = {**DEFAULT_EXTRA_PASS_SEASON, **raw}

    max_stars = int(season.get("max_stars") or DEFAULT_EXTRA_PASS_SEASON["max_stars"])
    pass_end = int(season.get("pass_end_position") or min(40, max_stars))
    ultra_start = int(season.get("ultra_start_position") or pass_end + 1)

    return {
        "id": season.get("id"),
        "slug": str(season.get("slug") or DEFAULT_EXTRA_PASS_SEASON["slug"]),
        "name": str(season.get("name") or DEFAULT_EXTRA_PASS_SEASON["name"]),
        "subtitle": str(season.get("subtitle") or DEFAULT_EXTRA_PASS_SEASON["subtitle"]),
        "description": str(season.get("description") or DEFAULT_EXTRA_PASS_SEASON["description"]),
        "start_date": _iso_or_none(season.get("start_date")),
        "end_date": _iso_or_none(season.get("end_date")),
        "is_active": bool(season.get("is_active", True)),
        "season_number": int(season.get("season_number") or DEFAULT_EXTRA_PASS_SEASON["season_number"]),
        "status": str(season.get("status") or DEFAULT_EXTRA_PASS_SEASON["status"]),
        "auto_switch": bool(season.get("auto_switch", DEFAULT_EXTRA_PASS_SEASON["auto_switch"])),
        "preset_key": season.get("preset_key") or DEFAULT_EXTRA_PASS_SEASON["preset_key"],
        "max_stars": max(1, max_stars),
        "free_track_type": str(season.get("free_track_type") or "bp_free"),
        "pass_track_type": str(season.get("pass_track_type") or "bp_premium"),
        "ultra_track_type": str(season.get("ultra_track_type") or "bp_ultra"),
        "pass_end_position": max(1, pass_end),
        "ultra_start_position": max(1, ultra_start),
        "theme": _json_dict(season.get("theme")),
    }


def _extra_pass_track_defs(season: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "free",
            "label": "Free",
            "track_type": season["free_track_type"],
            "access": "free",
            "start_position": 1,
            "end_position": season["max_stars"],
        },
        {
            "id": "premium",
            "label": "ExtraPass",
            "track_type": season["pass_track_type"],
            "access": "extra_pass",
            "start_position": 1,
            "end_position": season["pass_end_position"],
        },
        {
            "id": "ultra",
            "label": "ExtraPass Ultra",
            "track_type": season["ultra_track_type"],
            "access": "ultra",
            "start_position": season["ultra_start_position"],
            "end_position": season["max_stars"],
        },
    ]


def _parse_schedule_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _season_track_types(season: dict[str, Any]) -> list[str]:
    normalized = _normalize_extra_pass_season(season)
    return [
        normalized["free_track_type"],
        normalized["pass_track_type"],
        normalized["ultra_track_type"],
    ]


def _season_schedule_relation(current: dict[str, Any], next_season: dict[str, Any] | None) -> dict[str, Any]:
    if not next_season:
        return {
            "status": "tail",
            "label": "Следующий сезон не задан",
            "days": None,
            "seconds": None,
        }

    current_end = _parse_schedule_datetime(current.get("end_date"))
    next_start = _parse_schedule_datetime(next_season.get("start_date"))
    if not current_end or not next_start:
        return {
            "status": "unknown",
            "label": "Не хватает дат",
            "days": None,
            "seconds": None,
        }

    delta_seconds = int((next_start - current_end).total_seconds())
    if delta_seconds == 0:
        return {
            "status": "aligned",
            "label": "Дата окончания/начала совпадают",
            "days": 0,
            "seconds": 0,
        }
    if delta_seconds > 0:
        return {
            "status": "gap",
            "label": "Есть пауза между сезонами",
            "days": delta_seconds // 86400,
            "seconds": delta_seconds,
        }
    return {
        "status": "overlap",
        "label": "Сезоны пересекаются",
        "days": abs(delta_seconds) // 86400,
        "seconds": delta_seconds,
    }


def _build_season_schedule_overview(
    seasons: list[dict[str, Any]],
    reward_counts_by_type: dict[str, int],
) -> list[dict[str, Any]]:
    def sort_key(season: dict[str, Any]) -> tuple[datetime, int, int]:
        start = _parse_schedule_datetime(season.get("start_date")) or datetime.max.replace(tzinfo=timezone.utc)
        return (
            start,
            int(season.get("season_number") or 0),
            int(season.get("id") or 0),
        )

    ordered = sorted((_normalize_extra_pass_season(season) for season in seasons), key=sort_key)
    overview: list[dict[str, Any]] = []
    for index, season in enumerate(ordered):
        track_counts = {
            track_type: int(reward_counts_by_type.get(track_type, 0) or 0)
            for track_type in _season_track_types(season)
        }
        overview.append({
            "season": season,
            "track_counts": track_counts,
            "has_reward_tracks": all(count > 0 for count in track_counts.values()),
            "relation_to_next": _season_schedule_relation(
                season,
                ordered[index + 1] if index + 1 < len(ordered) else None,
            ),
        })
    return overview


def _normalize_reward_track_import_payload(payload: Any, season: dict[str, Any]) -> list[dict[str, Any]]:
    normalized_season = _normalize_extra_pass_season(season)
    track_defs = {track_def["id"]: track_def for track_def in _extra_pass_track_defs(normalized_season)}
    track_defs["pass"] = track_defs["premium"]
    track_defs["extra_pass"] = track_defs["premium"]

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
        raise ValueError("extra_pass_tracks_json_must_be_object_or_array")

    for lane, raw in source_rows:
        lane_id = str(raw.get("lane") or raw.get("track") or lane or "").strip().lower()
        track_type = str(raw.get("track_type") or "").strip()
        track_def = track_defs.get(lane_id)
        if not track_def and track_type:
            for candidate in _extra_pass_track_defs(normalized_season):
                if candidate["track_type"] == track_type:
                    track_def = candidate
                    break
        if not track_def:
            raise ValueError(f"unknown_extra_pass_lane:{lane_id or track_type or 'empty'}")

        position = int(raw.get("position") or raw.get("tier") or raw.get("stage") or 0)
        if position < int(track_def["start_position"]) or position > int(track_def["end_position"]):
            raise ValueError(f"position_out_of_track_scope:{track_def['id']}:{position}")

        reward_type = str(raw.get("reward_type") or raw.get("type") or "").strip()
        if not reward_type:
            raise ValueError("reward_type_required")
        reward_amount = int(raw.get("reward_amount") if raw.get("reward_amount") is not None else raw.get("amount", 0))
        if reward_amount < 0:
            raise ValueError("reward_amount_must_be_non_negative")

        reward_meta = raw.get("reward_meta", raw.get("meta"))
        if isinstance(reward_meta, str) and reward_meta.strip():
            reward_meta = _stdlib_json.loads(reward_meta)
        elif reward_meta in ("", None):
            reward_meta = None

        rows.append({
            "track_type": track_def["track_type"],
            "position": position,
            "reward_type": reward_type,
            "reward_amount": reward_amount,
            "reward_meta": reward_meta,
            "extra_pass_required": track_def["access"] != "free",
        })

    return rows


def _track_def_unlocked(track_def: dict[str, Any], access: dict[str, bool | str]) -> bool:
    required = track_def.get("access")
    if required == "free":
        return True
    if required == "ultra":
        return bool(access.get("has_ultra"))
    return bool(access.get("has_extra_pass"))


def _reward_track_def_for_type(track_type: str, season: Any) -> dict[str, Any] | None:
    normalized = _normalize_extra_pass_season(season)
    for track_def in _extra_pass_track_defs(normalized):
        if track_def["track_type"] == track_type:
            return track_def
    return None


def _reward_track_allowed(track_type: str, season: Any) -> bool:
    return track_type == "glory" or _reward_track_def_for_type(track_type, season) is not None


def _reward_track_unlocked_for_type(track_type: str, access: dict[str, bool | str], season: Any) -> bool:
    if track_type == "glory":
        return True
    track_def = _reward_track_def_for_type(track_type, season)
    if track_def:
        return _track_def_unlocked(track_def, access)
    return _reward_track_pass_unlocked(track_type, access)


def _entry_in_track_scope(entry: dict[str, Any], track_def: dict[str, Any]) -> bool:
    position = int(entry.get("position") or 0)
    return int(track_def["start_position"]) <= position <= int(track_def["end_position"])


def _serialize_reward_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": entry.get("id"),
        "reward_type": entry.get("reward_type"),
        "reward_amount": entry.get("reward_amount"),
        "reward_meta": _json_dict(entry.get("reward_meta")) if entry.get("reward_meta") is not None else None,
    }


def _build_extra_pass_payload(
    *,
    profile: dict[str, Any],
    season: Any,
    tracks_by_type: dict[str, list[dict[str, Any]]],
    claimed_by_type: dict[str, set[int]],
) -> dict[str, Any]:
    normalized_season = _normalize_extra_pass_season(season)
    access = _extra_pass_access(profile.get("extra_pass", "inactive"))
    progress_value = int(profile.get("stars") or 0)
    capped_progress = min(progress_value, normalized_season["max_stars"])
    track_defs = _extra_pass_track_defs(normalized_season)

    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    all_positions: set[int] = set()
    for track_def in track_defs:
        track_type = track_def["track_type"]
        for entry in tracks_by_type.get(track_type, []):
            if not _entry_in_track_scope(entry, track_def):
                continue
            position = int(entry.get("position") or 0)
            grouped.setdefault(track_type, {}).setdefault(position, []).append(entry)
            all_positions.add(position)

    summary = {
        "available_now": 0,
        "claimable_with_extra_pass": 0,
        "claimable_with_ultra": 0,
    }
    tiers = []

    for position in sorted(all_positions):
        tier_tracks: dict[str, Any] = {}
        for track_def in track_defs:
            track_type = track_def["track_type"]
            entries = grouped.get(track_type, {}).get(position, [])
            if not entries:
                tier_tracks[track_def["id"]] = None
                continue

            claimed = position in claimed_by_type.get(track_type, set())
            progress_locked = progress_value < position
            access_locked = not _track_def_unlocked(track_def, access)
            extra_pass_required = track_def["access"] != "free" or any(bool(e.get("extra_pass_required")) for e in entries)
            locked = progress_locked or access_locked
            available = not claimed and not locked
            progress_unlocked = not progress_locked and not claimed

            if available:
                summary["available_now"] += 1
            elif progress_unlocked and access_locked:
                if track_def["access"] == "extra_pass" and not bool(access.get("has_extra_pass")):
                    summary["claimable_with_extra_pass"] += 1
                elif track_def["access"] == "ultra" and not bool(access.get("has_ultra")):
                    summary["claimable_with_ultra"] += 1

            tier_tracks[track_def["id"]] = {
                "track_id": track_def["id"],
                "track_type": track_type,
                "label": track_def["label"],
                "access": track_def["access"],
                "position": position,
                "rewards": [_serialize_reward_entry(e) for e in entries],
                "claimed": claimed,
                "available": available,
                "locked": locked,
                "progress_locked": progress_locked,
                "access_locked": access_locked,
                "extra_pass_required": extra_pass_required,
            }

        tiers.append({"position": position, "tracks": tier_tracks})

    now = datetime.now(timezone.utc)
    end_date_raw = (dict(season) if season else {}).get("end_date") if season else None
    days_left = max(0, (end_date_raw - now).days) if hasattr(end_date_raw, "__sub__") else 0

    return {
        "season": {**normalized_season, "days_left": days_left},
        "progress": {
            "value": progress_value,
            "capped_value": capped_progress,
            "max": normalized_season["max_stars"],
            "percent": min(100, (progress_value / normalized_season["max_stars"]) * 100),
        },
        "access": access,
        "tracks": track_defs,
        "tiers": tiers,
        "summary": summary,
    }


# ONNX-мозг Берсерка для всех bot-match ботов с is_bot=True
# Модели: ai/models/*.onnx
# Активируется для любого бота, если модель загружена.
# Fallback на rule-based BotAI если модель не загрузилась.
BERSERK_BRAIN: Optional[BerserkInference] = None

# Socket.io сервер для WebSocket подключений
sio = socketio.AsyncServer(
    async_mode='aiohttp',
    cors_allowed_origins='*',
    logger=False,
    engineio_logger=False
)


def initialize_game_services(db: Database, battle_engine: BattleEngine | None = None) -> dict[str, Any]:
    """
    Единая точка инициализации игровых сервисов.

    Создает бота-генератор, матчмейкер и, при наличии, боевой движок.
    Возвращает словарь для DI в web.Application.
    """
    # Генератор ботов всегда требуется матчмейкеру
    bot_generator = BotGenerator(db)

    # Боевой движок можем принять извне или создать здесь.
    engine = battle_engine or BattleEngine(db=db, active_matches=ACTIVE_MATCHES)

    # Матчмейкер получает ссылки на БД, генератор ботов и опциональный движок.
    matchmaker = Matchmaker(db, bot_generator, engine)

    return {
        "bot_generator": bot_generator,
        "battle_engine": engine,
        "matchmaker": matchmaker,
        "active_matches": ACTIVE_MATCHES,
        "event_emitter": BATTLE_EVENT_EMITTER,
    }


# ---------------------------------------------------------------------- #
# Обработчики Socket.io для реального времени
# ---------------------------------------------------------------------- #
@sio.event
async def connect(sid: str, environ: dict[str, Any]) -> None:
    """
    Обработчик подключения клиента к Socket.io.
    Логируем подключение и сохраняем сессию.
    """
    match_id = environ.get("HTTP_MATCH_ID") or environ.get("match_id") or "unknown"
    logging.info("[SOCKET] connect sid=%s match_id=%s", sid, match_id)


@sio.event
async def disconnect(sid: str) -> None:
    """
    Multi-session: track disconnect only when last session for that user in that match leaves.
    """
    session_data = SID_TO_MATCH.get(sid)

    if session_data:
        match_id = session_data.get("match_id")
        user_id = session_data.get("user_id")

        del SID_TO_MATCH[sid]

        is_last_session = _unregister_session(match_id, user_id, sid)

        engine = ACTIVE_MATCHES.get(match_id)

        if engine and user_id and is_last_session:
            _mark_player_disconnected(str(match_id), int(user_id), engine)
        elif engine and user_id and not is_last_session:
            remaining = len((MATCH_SESSIONS.get(match_id) or {}).get(user_id, set()))
            logging.info(
                "[SOCKET] Session closed for player %s in match %s. %d session(s) remain.",
                user_id, match_id, remaining,
            )
    else:
        logging.info("[SOCKET] disconnect sid=%s match_id=unknown", sid)


@sio.event
async def join_match(sid: str, data: dict[str, Any]) -> None:
    """
    Клиент присоединяется к комнате матча.
    Требует _auth (Telegram initData) для верификации.
    """
    try:
        match_id = str(data.get("match_id", ""))
        if not match_id:
            await sio.emit("error", {"message": "match_id required"}, to=sid)
            return

        auth_token = data.get("_auth") or data.get("auth")
        app = getattr(sio, "app", None)

        if not auth_token or not app:
            await sio.emit("error", {"message": "authentication required"}, to=sid)
            return

        user_id = await _require_user_id_from_auth_token_str(str(auth_token), app)
        if user_id is None:
            await sio.emit("error", {"message": "invalid_auth"}, to=sid)
            return

        engine = ACTIVE_MATCHES.get(match_id)
        if not engine:
            await sio.emit("error", {"message": "match_not_found"}, to=sid)
            return

        p1_uid = getattr(engine.p1_state, "user_id", None)
        p2_uid = getattr(engine.p2_state, "user_id", None)
        if user_id not in (p1_uid, p2_uid):
            await sio.emit("error", {"message": "not_participant"}, to=sid)
            return

        room_name = str(match_id)
        await sio.enter_room(sid, room_name)

        SID_TO_MATCH[sid] = {"match_id": match_id, "user_id": int(user_id)}

        _register_session(match_id, user_id, sid)

        logging.info(
            "[SOCKET] join_match sid=%s match_id=%s user_id=%s",
            sid, match_id, user_id,
        )

        await sio.emit("joined_match", {"match_id": match_id, "user_id": str(user_id)}, to=sid)

    except Exception as exc:
        logging.error("join_match error: %s", exc, exc_info=True)
        await sio.emit("error", {"message": str(exc)}, to=sid)


@sio.event
async def leave_match(sid: str, data: dict[str, Any]) -> None:
    """
    Клиент покидает комнату матча.
    """
    try:
        match_id = str(data.get("match_id", ""))
        if not match_id:
            return

        room_name = str(match_id)
        await sio.leave_room(sid, room_name)
        session_data = SID_TO_MATCH.pop(sid, None)
        if session_data:
            _unregister_session(
                str(session_data.get("match_id") or match_id),
                session_data.get("user_id"),
                sid,
            )

        logging.info("[SOCKET] leave_match sid=%s match_id=%s", sid, match_id)

    except Exception as exc:
        logging.error(f"Ошибка при выходе из матча: {exc}")


@sio.event
async def client_ready(sid: str, data: dict[str, Any]) -> None:
    """
    Клиент сигнализирует о загрузке боя. Использует sid-bound user_id.
    """
    logger = logging.getLogger(__name__)
    try:
        session = SID_TO_MATCH.get(sid)
        if not session:
            await sio.emit("error", {"message": "not authenticated"}, to=sid)
            return

        match_id = str(session.get("match_id", ""))
        user_id = session.get("user_id")

        logger.info(
            "[SOCKET] client_ready sid=%s match_id=%s user_id=%s",
            sid, match_id or "unknown", user_id,
        )

        engine = ACTIVE_MATCHES.get(match_id)

        if not engine:
            logger.warning("client_ready: engine not found for match_id=%s", match_id)
            await sio.emit("error", {"message": "Match not found"}, to=sid)
            return

        ready_info: dict[str, Any] = {"all_ready": True}
        if hasattr(engine, "mark_client_ready"):
            try:
                ready_info = engine.mark_client_ready(user_id)
            except TypeError:
                engine.mark_client_ready()
                ready_info = {"all_ready": bool(getattr(engine, "client_ready", True))}
            logger.info(
                "client_ready: client ready for match_id=%s, user_id=%s readiness=%s",
                match_id,
                user_id,
                ready_info,
            )
        else:
            logger.warning("client_ready: engine has no mark_client_ready for match_id=%s", match_id)

        if ready_info.get("all_ready"):
            try:
                await check_and_run_bot(match_id, ACTIVE_MATCHES)
            except Exception as exc:
                logger.error("client_ready: error in check_and_run_bot: %s", exc, exc_info=True)

        readiness_event = "match_ready" if ready_info.get("all_ready") else "match_waiting"
        try:
            await _emit_personalized_match_state(
                sio,
                match_id,
                readiness_event,
                {"readiness": ready_info},
                engine=engine,
            )
        except Exception as exc:
            logger.warning("client_ready: failed readiness emit for match_id=%s: %s", match_id, exc)

        await sio.emit("client_ready_ack", {"match_id": match_id, **ready_info}, to=sid)

    except Exception as exc:
        logger.error("client_ready: ошибка обработки: %s", exc, exc_info=True)
        await sio.emit("error", {"message": str(exc)}, to=sid)


def calculate_trophy_delta(
    current_trophies: int,
    is_winner: bool,
    status: "ReplacementStatus"
) -> tuple[int, str, dict]:
    """
    Динамический расчёт изменения трофеев на основе тира прогрессии.

    Args:
        current_trophies: Текущее количество трофеев игрока
        is_winner: True если игрок победил, False если проиграл
        status: Статус замены (ACTIVE/AFK/SURRENDERED)

    Returns:
        Кортеж (trophy_delta, tier_name, tier_data)
    """
    from infrastructure.config import TROPHY_TIERS
    from core.state import ReplacementStatus
    import random

    # Определяем тир игрока
    tier_name = None
    tier_data = None
    for name, data in TROPHY_TIERS.items():
        if current_trophies in data["range"]:
            tier_name = name
            tier_data = data
            break

    # Fallback на последний тир если выше максимума
    if tier_data is None:
        tier_name = "master"
        tier_data = TROPHY_TIERS["master"]

    # Если победитель AFK/SURRENDERED -> обнуляем награды
    if is_winner and status in (ReplacementStatus.AFK, ReplacementStatus.SURRENDERED):
        return 0, tier_name, tier_data

    # Рассчитываем дельту трофеев
    if is_winner:
        # Победа: случайное значение из диапазона win
        win_min, win_max = tier_data["win"]
        delta = random.randint(win_min, win_max)
    else:
        # Поражение: случайное значение из диапазона loss (отрицательное)
        loss_min, loss_max = tier_data["loss"]

        # Если игрок SURRENDERED -> максимальный штраф
        if status == ReplacementStatus.SURRENDERED:
            delta = -loss_max
        else:
            delta = -random.randint(loss_min, loss_max)

    return delta, tier_name, tier_data


def calculate_coins_reward(
    tier_data: dict,
    is_winner: bool,
    status: "ReplacementStatus"
) -> int:
    """
    Динамический расчёт награды монетами на основе тира.

    Args:
        tier_data: Данные тира из TROPHY_TIERS
        is_winner: True если игрок победил
        status: Статус замены (ACTIVE/AFK/SURRENDERED)

    Returns:
        Количество монет (0 если проиграл или AFK/SURRENDERED)
    """
    from core.state import ReplacementStatus
    import random

    # Монеты только победителю с активным статусом
    if not is_winner:
        return 0

    if status in (ReplacementStatus.AFK, ReplacementStatus.SURRENDERED):
        return 0

    # Случайное значение из диапазона coin_range
    coin_min, coin_max = tier_data["coin_range"]
    return random.randint(coin_min, coin_max)


async def _track_economy_safe(
    db: Any,
    *,
    user_id: int,
    event_type: str,
    resource: str,
    amount: Any,
    source: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    try:
        await db.track_economy_event(
            user_id=user_id,
            event_type=event_type,
            resource=resource,
            amount=amount,
            source=source,
            metadata=metadata,
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "_track_economy_safe failed: user=%s type=%s resource=%s",
            user_id, event_type, resource, exc_info=True,
        )


def _resolve_match_game_mode(app: web.Application, match_id: str, engine: Any) -> str:
    """Резолв режима игры: match_game_modes > engine.game_mode > classic."""
    return resolve_mode_config(
        app.get("match_game_modes", {}).get(match_id, "")
        or getattr(engine, "game_mode", "")
        or "classic"
    ).mode_id


async def _process_battle_end(
    app: web.Application,
    match_id: str,
    engine: Any,
    winner_id: Optional[int]
) -> None:
    """Finalize a battle once, using battle_summary as the reward idempotency gate."""
    logger = logging.getLogger(__name__)
    db = app.get("db")

    if not db:
        logger.error("Database not available for processing battle end")
        return

    if getattr(engine, 'battle_end_processed', False) or getattr(engine, 'rewards_granted', False):
        _mark_match_ended(match_id)
        await _mark_matchmaker_finished(app, match_id, winner_id)
        logger.info("Battle end already processed for match %s", match_id)
        return

    p1_raw = getattr(engine.p1_state, "user_id", None)
    p2_raw = getattr(engine.p2_state, "user_id", None)
    try:
        p1_id_int = int(p1_raw) if p1_raw else 0
    except (ValueError, TypeError):
        p1_id_int = 0
    try:
        p2_id_int = int(p2_raw) if p2_raw else 0
    except (ValueError, TypeError):
        p2_id_int = 0
    try:
        winner_id_int = int(winner_id) if winner_id is not None else None
    except (ValueError, TypeError):
        winner_id_int = None

    if not p1_id_int or not p2_id_int:
        logger.error(
            "Empty player id in match=%s p1_raw=%s p2_raw=%s p1_int=%s p2_int=%s winner=%s player_ids=%s",
            match_id, p1_raw, p2_raw, p1_id_int, p2_id_int, winner_id_int,
            getattr(engine, "player_ids", ()),
        )
        engine.is_ended = True
        engine.battle_end_processed = True
        engine.rewards_granted = False
        _mark_match_ended(match_id)
        await _mark_matchmaker_finished(app, match_id, winner_id_int)
        app.get("match_game_modes", {}).pop(match_id, None)
        if match_id in ACTIVE_MATCHES:
            del ACTIVE_MATCHES[match_id]
        return

    p1_is_bot = False
    p2_is_bot = False
    if hasattr(engine, "is_bot"):
        p1_is_bot = engine.is_bot(p1_id_int) if p1_id_int else False
        p2_is_bot = engine.is_bot(p2_id_int) if p2_id_int else False
    p1_is_bot = p1_is_bot or bool(getattr(engine.p1_state, "is_bot", False))
    p2_is_bot = p2_is_bot or bool(getattr(engine.p2_state, "is_bot", False))
    is_bot_match = bool(getattr(engine, "is_bot_match", False)) or p1_is_bot or p2_is_bot

    game_mode = _resolve_match_game_mode(app, match_id, engine)
    mode_config = resolve_mode_config(game_mode)
    rewards = mode_config.rewards

    loser_id = None
    if winner_id_int is not None:
        loser_id = p2_id_int if winner_id_int == p1_id_int else p1_id_int

    from core.state import ReplacementStatus
    p1_status = getattr(engine.p1_state, "replacement_status", ReplacementStatus.ACTIVE)
    p2_status = getattr(engine.p2_state, "replacement_status", ReplacementStatus.ACTIVE)
    did_surrender = p1_status == ReplacementStatus.SURRENDERED or p2_status == ReplacementStatus.SURRENDERED
    did_afk = p1_status == ReplacementStatus.AFK or p2_status == ReplacementStatus.AFK

    winner_trophy_delta = 0
    loser_trophy_delta = 0
    winner_coins_delta = 0
    winner_is_bot = False
    loser_is_bot = False
    winner_is_active_human = False
    loser_is_active_human = False
    reward_plans: dict[int, dict[str, Any]] = {}
    economy_events: list[dict[str, Any]] = []
    winner_info: dict[str, Any] | None = None
    loser_info: dict[str, Any] | None = None

    if rewards.enabled and winner_id_int is not None:
        winner_is_bot = p1_is_bot if winner_id_int == p1_id_int else p2_is_bot
        loser_is_bot = p2_is_bot if winner_id_int == p1_id_int else p1_is_bot
        winning_side_is_p1 = (winner_id_int == p1_id_int)
        winner_status = p1_status if winning_side_is_p1 else p2_status
        loser_status = p2_status if winning_side_is_p1 else p1_status
        winner_is_active_human = (not winner_is_bot) and winner_status == ReplacementStatus.ACTIVE
        loser_is_active_human = (not loser_is_bot) and loser_status == ReplacementStatus.ACTIVE
        winner_state = engine.p1_state if winning_side_is_p1 else engine.p2_state
        loser_state = engine.p2_state if winning_side_is_p1 else engine.p1_state
        winner_surrender_processed = getattr(winner_state, "surrender_processed", False)
        loser_surrender_processed = getattr(loser_state, "surrender_processed", False)
        winner_current_trophies = 0
        loser_current_trophies = 0

        if winner_is_active_human:
            try:
                winner_info = await db.get_user_info(winner_id_int)
                winner_current_trophies = winner_info.get("trophies", 0) if winner_info else 0
            except Exception as exc:
                logger.error("Failed to get winner trophies: %s", exc)

        if not loser_is_bot:
            try:
                loser_info = await db.get_user_info(loser_id)
                loser_current_trophies = loser_info.get("trophies", 0) if loser_info else 0
            except Exception as exc:
                logger.error("Failed to get loser trophies: %s", exc)

        if winner_is_active_human and not winner_surrender_processed and rewards.trophies:
            winner_trophy_delta, winner_tier, winner_tier_data = calculate_trophy_delta(
                winner_current_trophies, is_winner=True, status=winner_status
            )
        else:
            winner_trophy_delta, winner_tier, winner_tier_data = 0, "bot", {}

        if (not loser_is_bot) and (not loser_surrender_processed) and rewards.trophies:
            loser_trophy_delta, loser_tier, loser_tier_data = calculate_trophy_delta(
                loser_current_trophies, is_winner=False, status=loser_status
            )
        else:
            loser_trophy_delta, loser_tier, loser_tier_data = 0, "bot", {}

        if rewards.coins and winner_is_active_human:
            winner_coins_delta = calculate_coins_reward(winner_tier_data, is_winner=True, status=winner_status)

        if winner_is_active_human:
            plan = reward_plans.setdefault(winner_id_int, {"old_league": (winner_info or {}).get("league", 1)})
            if winner_trophy_delta > 0:
                plan["trophies"] = winner_trophy_delta
            if winner_coins_delta > 0:
                plan["coins"] = winner_coins_delta
            if rewards.stars:
                plan["stars"] = int(plan.get("stars", 0) or 0) + 3
            if rewards.win_counter:
                winner_extra_pass = (winner_info or {}).get("extra_pass", "inactive")
                plan["wins_for_case"] = 3 if winner_extra_pass == "ultra" else (4 if winner_extra_pass == "active" else 5)

        if not loser_is_bot:
            plan = reward_plans.setdefault(int(loser_id), {"old_league": (loser_info or {}).get("league", 1) if loser_info else 1})
            if loser_trophy_delta < 0:
                plan["trophies"] = loser_trophy_delta
            if rewards.stars and loser_is_active_human:
                plan["stars"] = int(plan.get("stars", 0) or 0) + 1

        eco_meta = {
            "match_id": match_id,
            "game_mode": game_mode,
            "p1_user_id": p1_id_int,
            "p2_user_id": p2_id_int,
            "is_bot_match": is_bot_match,
        }
        if winner_is_active_human:
            if winner_trophy_delta > 0:
                economy_events.append({"user_id": winner_id_int, "event_type": "earn", "resource": "trophies", "amount": winner_trophy_delta, "source": "battle", "metadata": {**eco_meta, "result": "win"}})
            if winner_coins_delta > 0:
                economy_events.append({"user_id": winner_id_int, "event_type": "earn", "resource": "coins", "amount": winner_coins_delta, "source": "battle", "metadata": {**eco_meta, "result": "win"}})
            if rewards.stars:
                economy_events.append({"user_id": winner_id_int, "event_type": "earn", "resource": "stars", "amount": 3, "source": "battle", "metadata": {**eco_meta, "result": "win"}})
        if loser_trophy_delta < 0 and not loser_is_bot:
            economy_events.append({"user_id": int(loser_id), "event_type": "spend", "resource": "trophies", "amount": abs(loser_trophy_delta), "source": "battle", "metadata": {**eco_meta, "result": "loss"}})
        if loser_is_active_human and rewards.stars:
            economy_events.append({"user_id": int(loser_id), "event_type": "earn", "resource": "stars", "amount": 1, "source": "battle", "metadata": {**eco_meta, "result": "loss"}})

    try:
        if winner_id_int is not None:
            if winner_id_int == p1_id_int:
                p1_trophy_change = winner_trophy_delta
                p2_trophy_change = loser_trophy_delta
            else:
                p1_trophy_change = loser_trophy_delta
                p2_trophy_change = winner_trophy_delta
        else:
            p1_trophy_change = 0
            p2_trophy_change = 0

        match_duration_seconds = int(time.time() - engine.match_start_time) if hasattr(engine, 'match_start_time') and engine.match_start_time else 0
        p1_hero_id = None
        p2_hero_id = None
        try:
            if hasattr(engine, "_arena") and engine._arena:
                p1_hero_id = getattr(engine._arena.state.p1.hero, "card_id", None)
                p2_hero_id = getattr(engine._arena.state.p2.hero, "card_id", None)
        except Exception:
            pass

        p1_cards_played = 0
        p2_cards_played = 0
        for a in getattr(engine, "_analytics_actions", []) or []:
            if a.get("action_json", {}).get("type") == "play_card":
                if a.get("acting_player") == 1:
                    p1_cards_played += 1
                elif a.get("acting_player") == 2:
                    p2_cards_played += 1

        turns = getattr(engine, "turn", 0)
        tx_result = await db.apply_battle_end_rewards_transaction(
            match_id=match_id,
            p1_user_id=p1_id_int,
            p2_user_id=p2_id_int,
            winner_user_id=winner_id_int if winner_id is not None else None,
            loser_user_id=loser_id,
            p1_hero_id=p1_hero_id,
            p2_hero_id=p2_hero_id,
            p1_deck=getattr(engine, "_p1_initial_deck_ids", []) or [],
            p2_deck=getattr(engine, "_p2_initial_deck_ids", []) or [],
            surrender=did_surrender,
            afk=did_afk,
            match_type=game_mode,
            game_mode=game_mode,
            duration_seconds=match_duration_seconds,
            turns_count=turns,
            p1_trophy_change=p1_trophy_change,
            p2_trophy_change=p2_trophy_change,
            p1_coins_earned=winner_coins_delta if winner_id_int == p1_id_int else 0,
            p2_coins_earned=winner_coins_delta if winner_id_int == p2_id_int else 0,
            p1_cards_played=p1_cards_played,
            p2_cards_played=p2_cards_played,
            metadata={
                "is_bot_match": is_bot_match,
                "p1_is_bot": p1_is_bot,
                "p2_is_bot": p2_is_bot,
            },
            battle_result={
                "winner_score": 0,
                "loser_score": 0,
                "match_duration": match_duration_seconds,
                "match_type": game_mode,
            },
            rewards=reward_plans,
            economy_events=economy_events,
        )
    except Exception as exc:
        logger.error("Battle end transaction failed for match %s: %s", match_id, exc, exc_info=True)
        return

    def _merge_engine_map(attr: str, values: Any) -> None:
        if not isinstance(values, dict):
            return
        current = getattr(engine, attr, None)
        if not isinstance(current, dict):
            current = {}
        current.update(values)
        setattr(engine, attr, current)

    _merge_engine_map("_trophy_changes", tx_result.get("trophy_changes"))
    _merge_engine_map("_trophy_totals", tx_result.get("trophy_totals"))
    _merge_engine_map("_coins_changes", tx_result.get("coins_changes"))
    _merge_engine_map("_coins_totals", tx_result.get("coins_totals"))
    _merge_engine_map("_stars_changes", tx_result.get("stars_changes"))
    _merge_engine_map("_stars_totals", tx_result.get("stars_totals"))
    _merge_engine_map("_keys_changes", tx_result.get("keys_changes"))
    _merge_engine_map("_keys_totals", tx_result.get("keys_totals"))
    _merge_engine_map("_league_up", tx_result.get("league_up"))

    engine.is_ended = True
    engine.battle_end_processed = True
    engine.rewards_granted = bool(tx_result.get("applied") or tx_result.get("reason") == "duplicate_summary")
    _mark_match_ended(match_id)
    await _mark_matchmaker_finished(app, match_id, winner_id_int)
    app.get("match_game_modes", {}).pop(match_id, None)

    try:
        try:
            eligible_mode = str(game_mode or "").lower() not in ("training", "friendly")
            if tx_result.get("applied") and eligible_mode and not did_surrender and not did_afk:
                battle_meta = {
                    "match_id": match_id,
                    "game_mode": game_mode,
                    "is_bot_match": is_bot_match,
                    "p1_trophy_change": p1_trophy_change,
                    "p2_trophy_change": p2_trophy_change,
                }
                if not p1_is_bot and p1_status == ReplacementStatus.ACTIVE:
                    await db.award_squad_cbrp(
                        p1_id_int,
                        "battle_win" if winner_id_int == p1_id_int else "battle_loss",
                        source_id=f"battle:{match_id}:p1",
                        metadata={**battle_meta, "result": "win" if winner_id_int == p1_id_int else "loss"},
                    )
                if not p2_is_bot and p2_status == ReplacementStatus.ACTIVE:
                    await db.award_squad_cbrp(
                        p2_id_int,
                        "battle_win" if winner_id_int == p2_id_int else "battle_loss",
                        source_id=f"battle:{match_id}:p2",
                        metadata={**battle_meta, "result": "win" if winner_id_int == p2_id_int else "loss"},
                    )
        except Exception:
            logger.warning("Failed to award squad CBRP for battle match_id=%s", match_id, exc_info=True)

        if not getattr(engine, "_analytics_flushed", False):
            actions = getattr(engine, "_analytics_actions", []) or []
            if actions:
                count = await db.record_battle_actions(match_id, actions)
                logger.info("Flushed %d battle_actions for match %s", count, match_id)
            engine._analytics_flushed = True
    except Exception as exc:
        logger.error("Analytics flush failed for match %s: %s", match_id, exc, exc_info=True)


def _engine_economy_value(values: Any, user_id: Any) -> Any:
    if not isinstance(values, dict):
        return None
    keys = [user_id, str(user_id)]
    try:
        keys.append(int(user_id))
    except (TypeError, ValueError):
        pass
    for key in keys:
        if key in values:
            return values[key]
    return None


def _build_game_over_payload(engine: Any, winner_id: Any, *, reason: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "game_over": True,
        "winner_id": winner_id,
        "p1_hp": getattr(engine.p1_state, "hero_hp", None),
        "p2_hp": getattr(engine.p2_state, "hero_hp", None),
        "reason": reason,
        "players": {},
    }

    economy_fields = (
        ("_trophy_changes", "trophy_delta"),
        ("_trophy_totals", "trophy_total"),
        ("_coins_changes", "coins_delta"),
        ("_coins_totals", "coins_total"),
        ("_stars_changes", "stars_delta"),
        ("_stars_totals", "stars_total"),
        ("_keys_changes", "keys_delta"),
        ("_keys_totals", "keys_total"),
        ("_league_up", "league_up"),
    )

    for player_state in (getattr(engine, "p1_state", None), getattr(engine, "p2_state", None)):
        user_id = getattr(player_state, "user_id", None)
        if user_id is None:
            continue
        player_payload: dict[str, Any] = {}
        for source_attr, payload_key in economy_fields:
            value = _engine_economy_value(getattr(engine, source_attr, {}), user_id)
            if value is not None:
                player_payload[payload_key] = value
        payload["players"][str(user_id)] = player_payload

    return payload


def _is_finished_match(match_id: Any, engine: Any = None) -> bool:
    match_id_str = str(match_id or "")
    return bool(
        (match_id_str and match_id_str in ENDED_MATCH_IDS)
        or getattr(engine, "is_ended", False)
        or getattr(engine, "rewards_granted", False)
        or getattr(engine, "battle_end_processed", False)
    )


def _client_action_id(payload: dict[str, Any]) -> str | None:
    raw = payload.get("client_action_id") or payload.get("action_id") or payload.get("nonce")
    value = str(raw or "").strip()
    return value[:128] if value else None


def _prune_action_result_cache(now: float | None = None) -> None:
    now = now or time.time()
    for key, cached in list(ACTION_RESULT_CACHE.items()):
        if now - float(cached.get("created_at", 0) or 0) > ACTION_RESULT_TTL_SECONDS:
            ACTION_RESULT_CACHE.pop(key, None)


def _action_cache_get(match_id: Any, user_id: Any, client_action_id: str | None) -> dict[str, Any] | None:
    if not client_action_id:
        return None
    _prune_action_result_cache()
    try:
        key = (str(match_id), int(user_id), str(client_action_id))
    except (TypeError, ValueError):
        return None
    cached = ACTION_RESULT_CACHE.get(key)
    if not cached:
        return None
    return {
        "payload": cached.get("payload", {}),
        "status": int(cached.get("status", 200) or 200),
    }


def _action_cache_set(
    match_id: Any,
    user_id: Any,
    client_action_id: str | None,
    payload: dict[str, Any],
    *,
    status: int = 200,
) -> None:
    if not client_action_id:
        return
    try:
        key = (str(match_id), int(user_id), str(client_action_id))
    except (TypeError, ValueError):
        return
    ACTION_RESULT_CACHE[key] = {
        "payload": payload,
        "status": int(status),
        "created_at": time.time(),
    }


def _action_failure_status(result: dict[str, Any]) -> int:
    error = str((result or {}).get("error") or "")
    if error in {"not_your_turn", "turn_expired", "game_already_ended", "match_not_ready"}:
        return 409
    if error in {"not_participant", "unauthorized"}:
        return 403
    return 400


def _is_match_waiting_for_players(engine: Any) -> bool:
    if not engine or not hasattr(engine, "is_waiting_for_players"):
        return False
    try:
        return bool(engine.is_waiting_for_players())
    except Exception:
        logging.getLogger(__name__).debug("Failed to read match readiness", exc_info=True)
        return False


def _match_not_ready_response(match_id: Any, engine: Any, viewer_id: Any) -> web.Response | None:
    if not _is_match_waiting_for_players(engine):
        return None
    state = engine.get_full_state(viewer_id=viewer_id) if hasattr(engine, "get_full_state") else {}
    return web.json_response(
        {
            "error": "match_not_ready",
            "result": {"success": False, "error": "match_not_ready"},
            "match_id": str(match_id),
            "state": state,
        },
        status=409,
    )


async def _auto_end_expired_turn_response(
    app: web.Application,
    match_id: str,
    engine: Any,
    viewer_id: int,
    client_action_id: str | None,
) -> web.Response | None:
    if _is_match_waiting_for_players(engine):
        return None
    if not hasattr(engine, "is_turn_expired") or not engine.is_turn_expired():
        return None
    handled = await _handle_natural_turn_timeout(app, str(match_id), engine)
    if not handled:
        return None
    state = engine.get_full_state(viewer_id=viewer_id) if hasattr(engine, "get_full_state") else {}
    payload = {
        "error": "turn_expired",
        "result": {"success": False, "error": "turn_expired"},
        "state": state,
    }
    _action_cache_set(match_id, viewer_id, client_action_id, payload, status=409)
    return web.json_response(payload, status=409)


def _build_finished_match_action_payload(
    match_id: Any,
    engine: Any = None,
    viewer_id: Any = None,
) -> dict[str, Any]:
    match_id_str = str(match_id or "")
    payload: dict[str, Any] = {
        "match_id": match_id_str,
        "success": True,
        "game_over": True,
        "already_ended": True,
        "error": "game_already_ended",
    }
    if match_id_str:
        _mark_match_ended(match_id_str)
    if engine is not None and viewer_id is not None and hasattr(engine, "get_full_state"):
        try:
            payload["state"] = engine.get_full_state(viewer_id=viewer_id)
        except Exception:
            logging.getLogger(__name__).debug(
                "Failed to attach final state for already-ended match %s",
                match_id_str,
                exc_info=True,
            )
    return payload


async def _emit_personalized_match_state(
    sio_inst: Any,
    match_id: str,
    event_type: str,
    event_data: dict[str, Any] | None = None,
    *,
    engine: Any = None,
) -> None:
    """Emit a viewer-specific battle state to every connected sid in a match."""
    logger = logging.getLogger(__name__)
    match_id_str = str(match_id)
    engine = engine or ACTIVE_MATCHES.get(match_id_str)
    if not sio_inst or not engine or not hasattr(engine, "get_full_state"):
        return

    participants: list[tuple[str, Any]] = []
    for sid, session_data in list(SID_TO_MATCH.items()):
        if str(session_data.get("match_id")) == match_id_str and session_data.get("user_id") is not None:
            participants.append((sid, session_data.get("user_id")))

    for sid, user_id in participants:
        try:
            personalized_state = engine.get_full_state(viewer_id=user_id)
            event_extra = {
                key: value
                for key, value in (event_data or {}).items()
                if key not in {"event_type", "match_id", "state", "state_p1", "data"}
            }
            await sio_inst.emit(
                event_type,
                {
                    "match_id": match_id_str,
                    "state": personalized_state,
                    "data": (event_data or {}).get("data", {}),
                    **event_extra,
                },
                to=sid,
            )
        except Exception as exc:
            logger.error(
                "[SOCKET_EMIT] Failed personalized emit event=%s match=%s sid=%s user=%s: %s",
                event_type,
                match_id_str,
                sid,
                user_id,
                exc,
                exc_info=True,
            )


async def _apply_surrender_penalty_once(
    app: web.Application,
    match_id: str,
    engine: Any,
    user_id_int: int,
) -> dict[str, Any]:
    """Mark surrender as processed once and apply the immediate trophy penalty if enabled."""
    logger = logging.getLogger(__name__)
    from core.state import ReplacementStatus

    player_state = engine.get_player_state(user_id_int)
    existing_changes = getattr(engine, "_trophy_changes", {}) or {}
    existing_totals = getattr(engine, "_trophy_totals", {}) or {}

    db = app.get("db")
    if not db:
        return {"success": False, "error": "database_unavailable", "status": 500}

    try:
        user_info = await db.get_user_info(user_id_int)
        current_trophies = user_info.get("trophies", 0) if user_info else 0
    except Exception as exc:
        logger.error("Failed to get user trophies for surrender: %s", exc, exc_info=True)
        current_trophies = int(existing_totals.get(user_id_int, 0) or 0)

    if getattr(player_state, "surrender_processed", False):
        return {
            "success": True,
            "already_processed": True,
            "surrender_processed": True,
            "trophy_penalty": int(existing_changes.get(user_id_int, 0) or 0),
            "new_trophies": int(existing_totals.get(user_id_int, current_trophies) or 0),
        }

    game_mode = _resolve_match_game_mode(app, match_id, engine)
    if not resolve_mode_config(game_mode).rewards.trophies:
        player_state.surrender_processed = True
        return {
            "success": True,
            "already_processed": False,
            "surrender_processed": True,
            "trophy_penalty": 0,
            "new_trophies": current_trophies,
        }

    penalty_delta, _tier_name, _tier_data = calculate_trophy_delta(
        current_trophies,
        is_winner=False,
        status=ReplacementStatus.SURRENDERED,
    )

    try:
        result = await db.update_user_trophies(user_id_int, penalty_delta)
    except Exception as exc:
        logger.error("Failed to update trophies for surrendered player: %s", exc, exc_info=True)
        return {"success": False, "error": "trophy_update_failed", "status": 500}

    new_trophies = int((result or {}).get("trophies", 0) or 0)
    player_state.surrender_processed = True
    engine._trophy_changes = getattr(engine, "_trophy_changes", {}) or {}
    engine._trophy_totals = getattr(engine, "_trophy_totals", {}) or {}
    engine._trophy_changes[user_id_int] = penalty_delta
    engine._trophy_totals[user_id_int] = new_trophies
    logger.warning(
        "[SURRENDER_IMMEDIATE] Player %s lost %d trophies instantly (%d -> %d).",
        user_id_int,
        abs(penalty_delta),
        current_trophies,
        new_trophies,
    )
    return {
        "success": True,
        "already_processed": False,
        "surrender_processed": True,
        "trophy_penalty": penalty_delta,
        "new_trophies": new_trophies,
    }


@sio.event
async def surrender(sid: str, data: dict[str, Any]) -> None:
    """
    Сдача игрока. Использует sid-bound user_id, игнорирует client-sent user_id.
    """
    logger = logging.getLogger(__name__)
    try:
        session = SID_TO_MATCH.get(sid)
        if not session:
            logger.warning("surrender: no session for sid=%s", sid)
            await sio.emit("error", {"message": "not_authenticated"}, to=sid)
            return

        match_id = str(session.get("match_id", ""))
        user_id = session.get("user_id")

        logger.info(
            "[SOCKET] surrender sid=%s match_id=%s user_id=%s",
            sid, match_id or "unknown", user_id,
        )

        if not match_id or user_id is None:
            logger.warning("surrender: missing match_id/user_id for sid=%s", sid)
            await sio.emit("error", {"message": "match_id and user_id required"}, to=sid)
            return

        engine = ACTIVE_MATCHES.get(match_id)
        if not engine:
            logger.warning("surrender: engine not found for match_id=%s", match_id)
            await sio.emit("error", {"message": "Match not found"}, to=sid)
            return

        user_id_int = int(user_id)

        if str(engine.p1_state.user_id) != str(user_id_int) and str(engine.p2_state.user_id) != str(user_id_int):
            logger.error("surrender: player %s not in match %s", user_id_int, match_id)
            await sio.emit("error", {"message": "User not in match"}, to=sid)
            return

        engine.mark_surrender(user_id_int)

        app = getattr(sio, 'app', None)
        if not app:
            logger.error("surrender: не удалось получить app для начисления трофеев")
            await sio.emit("error", {"message": "Database unavailable"}, to=sid)
            return

        penalty_result = await _apply_surrender_penalty_once(app, match_id, engine, user_id_int)
        if not penalty_result.get("success"):
            await sio.emit("error", {"message": penalty_result.get("error", "surrender_failed")}, to=sid)
            return

        await sio.emit(
            "surrender_ack",
            {
                "match_id": match_id,
                "user_id": user_id_int,
                "trophy_penalty": penalty_result.get("trophy_penalty", 0),
                "new_trophies": penalty_result.get("new_trophies", 0),
                "already_processed": bool(penalty_result.get("already_processed")),
            },
            to=sid
        )

        # СРАЗУ вызываем проверку окончания игры (согласно правилам)
        game_over_result = engine.check_game_over()

        # Явно отправляем game_over текущему сокету, если матч завершился
        if game_over_result.get("game_over"):
            logger.info("surrender: матч %s завершен после сдачи (game_over=True)", match_id)
            winner_id = game_over_result.get("winner_id")

            # Начисляем награды победителю и сохраняем результат
            await _process_battle_end(app, match_id, engine, winner_id)

            # Отправляем game_over именно сдавшемуся игроку (sid) перед тем как он уйдет
            await sio.emit(
                "game_over",
                _build_game_over_payload(engine, winner_id, reason="surrender"),
                to=sid
            )

            # Также уведомляем комнату, если есть другие участники
            await sio.emit(
                "game_over",
                _build_game_over_payload(engine, winner_id, reason="surrender"),
                room=match_id
            )

        # Запускаем бота, если сейчас ход сдавшегося игрока и игра НЕ закончена
        if not game_over_result.get("game_over") and engine.current_player_id == user_id_int:
            logger.info("surrender: запускаем бота для сдавшегося игрока %s", user_id_int)
            await check_and_run_bot(match_id, ACTIVE_MATCHES)

    except Exception as exc:
        logger.error("surrender: ошибка обработки: %s", exc, exc_info=True)
        await sio.emit("error", {"message": str(exc)}, to=sid)


def setup_battle_events() -> None:
    """
    Регистрирует обработчики событий движка для отправки через Socket.io.
    Вызывается один раз при старте сервера.
    """
    logging.getLogger(__name__).debug("setup_battle_events() called")

    def emit_to_match(match_id: str, event_data: dict[str, Any]) -> None:
        """
        Персонализированный broadcast: отправляет каждому участнику состояние с его legal_actions.
        Для каждого клиента вызывается engine.get_full_state(viewer_id=user_id).
        """
        try:
            event_type = event_data.get("event_type", "state_changed")
            asyncio.create_task(
                _emit_personalized_match_state(
                    sio,
                    str(match_id),
                    event_type,
                    event_data,
                )
            )

        except Exception as exc:
            logging.error(f"[SOCKET_EMIT] Ошибка при персональной рассылке: {exc}", exc_info=True)

    # Регистрируем обработчики для всех типов событий
    BATTLE_EVENT_EMITTER.on("turn_start", emit_to_match)
    BATTLE_EVENT_EMITTER.on("card_played", emit_to_match)
    BATTLE_EVENT_EMITTER.on("attack", emit_to_match)
    BATTLE_EVENT_EMITTER.on("turn_end", emit_to_match)
    BATTLE_EVENT_EMITTER.on("turn_switched", emit_to_match)  # КРИТИЧНО: Смена хода после бота
    BATTLE_EVENT_EMITTER.on("state_changed", emit_to_match)
    BATTLE_EVENT_EMITTER.on("potion_used", emit_to_match)  # НОВОЕ: Поддержка зелий

    # КРИТИЧНО: Событийный триггер бота - запускается при каждом начале хода
    # Это гарантирует, что бот "проснется" всегда: и после хода игрока, и после авто-смены по таймеру
    def bot_trigger_listener(match_id: str, event_data: dict[str, Any]) -> None:
        """
        Автоматически запускает бота при начале его хода.
        Вызывается при событии turn_start.
        """
        if event_data.get("event_type") == "turn_start":
            logging.info(f"[BOT_TRIGGER] 🤖 Событие turn_start получено для матча {match_id}")
            # Создаем задачу запуска бота (не блокируем эмиттер)
            asyncio.create_task(check_and_run_bot(match_id, ACTIVE_MATCHES))

    BATTLE_EVENT_EMITTER.on("turn_start", bot_trigger_listener)

    logging.info("Socket.io: обработчики событий боя зарегистрированы (включая bot_trigger)")



def _serialize_datetime(obj: Any) -> Any:
    """Рекурсивно преобразовать все datetime объекты в строки для JSON."""
    if obj is None:
        return None
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_serialize_datetime(item) for item in obj)
    elif obj.__class__.__name__ == "Decimal":
        return float(obj)
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    # Для других типов (например, asyncpg.Record) пытаемся преобразовать в dict
    elif hasattr(obj, '__dict__'):
        return _serialize_datetime(obj.__dict__)
    else:
        return obj


def _verify_init_data(init_data: str, bot_token: str) -> dict[str, str] | None:
    """Проверить подпись initData от Telegram и вернуть параметры."""
    try:
        data_dict = dict(parse_qsl(init_data))
        received_hash = data_dict.pop("hash", "")
        if not received_hash:
            return None

        data_check_string = "\n".join(
            f"{k}={v}" for k, v in sorted(data_dict.items())
        )

        secret_key = hmac.new(
            key=b"WebAppData",
            msg=bot_token.encode(),
            digestmod=hashlib.sha256,
        ).digest()

        calculated_hash = hmac.new(
            key=secret_key,
            msg=data_check_string.encode(),
            digestmod=hashlib.sha256,
        ).hexdigest()

        if calculated_hash != received_hash:
            return None

        return data_dict
    except Exception:
        return None


def _extract_user_id_from_init_data(data_dict: dict[str, str]) -> int | None:
    """Извлечь user_id из проверенного initData."""
    try:
        user_str = data_dict.get("user", "")
        if not user_str:
            return None
        import json

        user_data = json.loads(user_str)
        return int(user_data.get("id"))
    except Exception:
        return None



# ============================================================================
# Центральный модуль аутентификации
# ============================================================================

AUTH_MAX_AGE_SECONDS = 86400
AUTH_CLOCK_SKEW_SECONDS = 300
ADMIN_SESSION_COOKIE_NAME = "ea_admin_session"
ADMIN_SESSION_MAX_AGE_SECONDS = 4 * 60 * 60

MATCH_SESSIONS: dict[str, dict[int, set[str]]] = {}
MATCH_LOCKS: dict[str, asyncio.Lock] = {}

# Кеш IP-геолокации: {ip: (country, expiry_timestamp)}
_ip_geo_cache: dict[str, tuple[str, float]] = {}


def _display_id_generator(check_exists_fn) -> str:
    while True:
        digits = "".join(random.choices(string.digits, k=4))
        letters = "".join(random.choices(string.ascii_uppercase, k=3))
        display_id = f"{digits}-{letters}"
        if not check_exists_fn(display_id):
            return display_id


def _mask_email(email: str) -> str:
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) <= 1:
        return f"{local[0]}***@{domain}"
    return f"{local[0]}***@{domain}"


def _nickname_valid(nick: str) -> bool:
    import re
    return bool(re.fullmatch(r"[a-zA-Z0-9_-]{3,20}", nick))


async def _get_ip_country(ip: str, settings) -> str | None:
    import logging
    logger = logging.getLogger(__name__)
    now = time.time()
    cached = _ip_geo_cache.get(ip)
    if cached and cached[1] > now:
        return cached[0]

    api_key = settings.ip_geo_api_key
    try:
        if api_key:
            url = f"http://pro.ip-api.com/json/{ip}?key={api_key}&fields=countryCode"
        else:
            url = f"http://ip-api.com/json/{ip}?fields=countryCode"
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    country = data.get("countryCode")
                    if country:
                        _ip_geo_cache[ip] = (country, now + 3600)
                        return country
    except Exception:
        logger.debug("IP geo lookup failed", exc_info=True)
    return None


async def _verify_jwt_token_async(token: str, db, settings) -> tuple[int, str] | None:
    try:
        payload = pyjwt.decode(
            token, settings.jwt_secret, algorithms=["HS256"],
            options={"require": ["user_id", "session_id", "exp", "iat"]}
        )
    except (pyjwt.ExpiredSignatureError, pyjwt.InvalidTokenError, pyjwt.DecodeError):
        return None

    session_id = payload.get("session_id")
    if not session_id:
        return None

    try:
        session_uuid = uuid.UUID(session_id)
    except (ValueError, TypeError):
        return None

    session = await db.verify_session(session_uuid, token)
    if not session:
        return None
    return (int(payload["user_id"]), session_id)


def _make_admin_session_token(user_id: int) -> str:
    now = int(time.time())
    settings = get_settings()
    return pyjwt.encode(
        {
            "typ": "admin_session",
            "user_id": int(user_id),
            "iat": now,
            "exp": now + ADMIN_SESSION_MAX_AGE_SECONDS,
        },
        settings.jwt_secret,
        algorithm="HS256",
    )


def _verify_admin_session_token(token: str) -> int | None:
    try:
        settings = get_settings()
        payload = pyjwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            options={"require": ["typ", "user_id", "exp", "iat"]},
        )
        if payload.get("typ") != "admin_session":
            return None
        return int(payload["user_id"])
    except (pyjwt.ExpiredSignatureError, pyjwt.InvalidTokenError, pyjwt.DecodeError, ValueError, TypeError):
        return None


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _is_dev_local(request) -> bool:
    settings = get_settings()
    return (
        settings.environment == "development"
        and request.remote in {"127.0.0.1", "::1", "localhost"}
    )


def _is_admin_surface_request(request: web.Request) -> bool:
    path = request.path
    return path.startswith("/api/admin/") or path in {"/extraShop/admin", "/extraShop/admin/"}


def _allow_dev_user_id_auth(request: web.Request) -> bool:
    return _is_dev_local(request) and not _is_admin_surface_request(request)


def _validate_auth_date(data_dict: dict[str, str]) -> bool:
    try:
        auth_date_str = data_dict.get("auth_date", "")
        if not auth_date_str:
            return False
        auth_date = int(auth_date_str)
        now = int(time.time())
        if auth_date < now - AUTH_MAX_AGE_SECONDS - AUTH_CLOCK_SKEW_SECONDS:
            return False
        if auth_date > now + AUTH_CLOCK_SKEW_SECONDS:
            return False
        return True
    except (ValueError, TypeError):
        return False


def _request_auth_token(request: web.Request) -> str:
    auth_param = str(request.rel_url.query.get("_auth") or "").strip()
    if auth_param:
        return auth_param
    authorization = str(request.headers.get("Authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


async def require_user_id(request) -> int:
    auth_param = _request_auth_token(request)
    if auth_param:
        # 1. Попытаться верифицировать как JWT
        settings = get_settings()
        jwt_result = await _verify_jwt_token_async(auth_param, request.app["extraid_db"], settings)
        if jwt_result:
            return jwt_result[0]

        # 2. Попытаться верифицировать как Telegram initData
        verified_data = _verify_init_data(auth_param, request.app["bot_token"])
        if verified_data:
            if not _validate_auth_date(verified_data):
                raise web.HTTPUnauthorized(
                    reason="auth_expired",
                    text='{"error":"auth_expired"}',
                    content_type="application/json",
                )
            uid = _extract_user_id_from_init_data(verified_data)
            if uid:
                return uid

        if _allow_dev_user_id_auth(request):
            if auth_param.isdigit():
                return int(auth_param)
            raise web.HTTPUnauthorized(
                reason="invalid_init_data",
                text='{"error":"invalid_auth"}',
                content_type="application/json",
            )
        raise web.HTTPUnauthorized(
            reason="invalid_init_data",
            text='{"error":"invalid_auth"}',
            content_type="application/json",
        )

    if _is_admin_surface_request(request):
        cookie_user_id = _verify_admin_session_token(
            str(request.cookies.get(ADMIN_SESSION_COOKIE_NAME) or "")
        )
        if cookie_user_id:
            return cookie_user_id

    if _allow_dev_user_id_auth(request):
        user_id_param = request.rel_url.query.get("user_id")
        if user_id_param:
            try:
                return int(user_id_param)
            except ValueError:
                pass

    raise web.HTTPUnauthorized(
        reason="authentication_required",
        text='{"error":"authentication_required"}',
        content_type="application/json",
    )


async def require_user_id_from_payload(request, payload: dict) -> int:
    auth_token = payload.get("_auth") or payload.get("auth")
    if auth_token:
        # 1. Попытаться верифицировать как JWT
        settings = get_settings()
        jwt_result = await _verify_jwt_token_async(str(auth_token), request.app["extraid_db"], settings)
        if jwt_result:
            return jwt_result[0]

        # 2. Попытаться верифицировать как Telegram initData
        verified_data = _verify_init_data(str(auth_token), request.app["bot_token"])
        if verified_data:
            if not _validate_auth_date(verified_data):
                raise web.HTTPUnauthorized(
                    reason="auth_expired",
                    text='{"error":"auth_expired"}',
                    content_type="application/json",
                )
            uid = _extract_user_id_from_init_data(verified_data)
            if uid:
                return uid

    if _allow_dev_user_id_auth(request):
        if "user_id" in payload:
            try:
                return int(payload["user_id"])
            except (ValueError, TypeError):
                pass

    raise web.HTTPUnauthorized(
        reason="authentication_required",
        text='{"error":"authentication_required"}',
        content_type="application/json",
    )


def _require_user_id_from_init_data_str(init_data_str: str, bot_token: str) -> int | None:
    verified_data = _verify_init_data(init_data_str, bot_token)
    if not verified_data:
        return None
    if not _validate_auth_date(verified_data):
        return None
    return _extract_user_id_from_init_data(verified_data)


async def _require_user_id_from_auth_token_str(auth_token: str, app: web.Application) -> int | None:
    """Socket-friendly equivalent of HTTP auth: JWT/ExtraID first, Telegram initData second."""
    settings = get_settings()
    extraid_db = app.get("extraid_db")
    if extraid_db is not None:
        jwt_result = await _verify_jwt_token_async(auth_token, extraid_db, settings)
        if jwt_result:
            return jwt_result[0]

    bot_token = str(app.get("bot_token") or "")
    if bot_token:
        return _require_user_id_from_init_data_str(auth_token, bot_token)
    return None


def _verify_participant(engine, user_id: int) -> None:
    p1_uid = getattr(engine.p1_state, "user_id", None)
    p2_uid = getattr(engine.p2_state, "user_id", None)
    if user_id not in (p1_uid, p2_uid):
        raise web.HTTPForbidden(
            reason="not_participant",
            text='{"error":"not_participant"}',
            content_type="application/json",
        )


def _get_match_lock(match_id: str) -> asyncio.Lock:
    if match_id not in MATCH_LOCKS:
        MATCH_LOCKS[match_id] = asyncio.Lock()
    return MATCH_LOCKS[match_id]


def _register_session(match_id: str, user_id: int, sid: str) -> None:
    pending_key = (str(match_id), int(user_id))
    pending_task = MATCH_DISCONNECT_TASKS.pop(pending_key, None)
    if pending_task and not pending_task.done():
        pending_task.cancel()

    MATCH_DISCONNECT_STATES.pop(pending_key, None)
    _cancel_replacement_bot_task(str(match_id), int(user_id))
    engine = ACTIVE_MATCHES.get(str(match_id))
    if engine and hasattr(engine, "restore_player_control"):
        restored = engine.restore_player_control(int(user_id))
        if restored:
            logging.getLogger(__name__).info(
                "[SOCKET] Player %s restored control in match %s on reconnect.",
                user_id, match_id,
            )

    if match_id not in MATCH_SESSIONS:
        MATCH_SESSIONS[match_id] = {}
    if user_id not in MATCH_SESSIONS[match_id]:
        MATCH_SESSIONS[match_id][user_id] = set()
    MATCH_SESSIONS[match_id][user_id].add(sid)


def _unregister_session(match_id: str, user_id: int, sid: str) -> bool:
    user_sessions = (MATCH_SESSIONS.get(match_id) or {}).get(user_id)
    if not user_sessions:
        return True
    user_sessions.discard(sid)
    remaining = len(user_sessions)
    if remaining == 0:
        MATCH_SESSIONS[match_id].pop(user_id, None)
        if not MATCH_SESSIONS[match_id]:
            MATCH_SESSIONS.pop(match_id, None)
    return remaining == 0


def _mark_player_disconnected(match_id: str, user_id: int, engine: BattleEngine) -> None:
    """Track a fully disconnected human without immediately replacing them."""
    key = (str(match_id), int(user_id))
    state = MATCH_DISCONNECT_STATES.setdefault(
        key,
        {
            "disconnected_at": time.time(),
            "timed_out_turns": 0,
            "takeover_started": False,
        },
    )
    state["disconnected"] = True
    state["last_seen_turn"] = int(getattr(engine, "turn", 0) or 0)
    logging.getLogger(__name__).info(
        "[SOCKET] Last session for player %s in match %s disconnected. Waiting for timed turns before bot takeover.",
        user_id, match_id,
    )


def _is_player_disconnected(match_id: str, user_id: int) -> bool:
    key = (str(match_id), int(user_id))
    return bool(MATCH_DISCONNECT_STATES.get(key)) and not bool(
        (MATCH_SESSIONS.get(str(match_id)) or {}).get(int(user_id))
    )


def _record_disconnected_turn_timeout(match_id: str, user_id: int) -> int:
    key = (str(match_id), int(user_id))
    state = MATCH_DISCONNECT_STATES.setdefault(
        key,
        {"disconnected_at": time.time(), "timed_out_turns": 0, "takeover_started": False},
    )
    state["timed_out_turns"] = int(state.get("timed_out_turns", 0)) + 1
    state["last_timeout_at"] = time.time()
    return int(state["timed_out_turns"])


def _disconnect_takeover_window(engine: BattleEngine) -> float:
    turn_duration = float(getattr(engine, "turn_duration", 25) or 25)
    return min(
        DISCONNECT_TAKEOVER_MAX_WINDOW_SECONDS,
        max(DISCONNECT_TAKEOVER_MIN_WINDOW_SECONDS, turn_duration * DISCONNECT_TAKEOVER_FRACTION),
    )


def _cancel_replacement_bot_task(match_id: str, user_id: int) -> None:
    task = BOT_TASKS.get(str(match_id))
    key = BOT_TASK_KEYS.get(str(match_id))
    if task and not task.done() and key and key[0] == str(user_id):
        task.cancel()
        BOT_TASKS.pop(str(match_id), None)
        BOT_TASK_KEYS.pop(str(match_id), None)


def _mark_user_activity_for_match(match_id: str, user_id: int, engine: BattleEngine) -> None:
    key = (str(match_id), int(user_id))
    task = MATCH_DISCONNECT_TASKS.pop(key, None)
    if task and not task.done():
        task.cancel()
    MATCH_DISCONNECT_STATES.pop(key, None)
    _cancel_replacement_bot_task(str(match_id), int(user_id))
    if hasattr(engine, "mark_player_activity"):
        engine.mark_player_activity(int(user_id))


def _schedule_disconnect_takeover(match_id: str, user_id: int, engine: BattleEngine) -> None:
    """Start replacement bot only in the final proportional window of the second disconnected turn."""
    key = (str(match_id), int(user_id))
    state = MATCH_DISCONNECT_STATES.setdefault(
        key,
        {"disconnected_at": time.time(), "timed_out_turns": 1, "takeover_started": False},
    )
    existing = MATCH_DISCONNECT_TASKS.get(key)
    if existing and not existing.done():
        return

    window = _disconnect_takeover_window(engine)
    remaining = engine.get_turn_time_remaining() if hasattr(engine, "get_turn_time_remaining") else float(getattr(engine, "turn_duration", 25) or 25)
    delay = max(0.0, float(remaining) - window)
    state["takeover_scheduled_turn"] = int(getattr(engine, "turn", 0) or 0)

    async def _takeover_if_still_gone() -> None:
        try:
            await asyncio.sleep(delay)
            if not _is_player_disconnected(match_id, user_id):
                return
            current_engine = ACTIVE_MATCHES.get(str(match_id))
            if not current_engine or getattr(current_engine, "is_ended", False):
                return
            current_player = current_engine.get_current_player_id() if hasattr(current_engine, "get_current_player_id") else None
            if current_player != user_id:
                return

            from core.state import ReplacementStatus

            status = (
                current_engine.get_player_replacement_status(user_id)
                if hasattr(current_engine, "get_player_replacement_status")
                else ReplacementStatus.ACTIVE
            )
            if status == ReplacementStatus.SURRENDERED:
                return
            if status == ReplacementStatus.ACTIVE:
                if not await _replacement_bot_allowed(str(match_id), current_engine):
                    await _terminate_match_without_rewards(
                        str(match_id),
                        ACTIVE_MATCHES,
                        reason="opponent_disconnected",
                        message="Противник отключился",
                    )
                    return
                current_engine.set_player_replacement_status(user_id, ReplacementStatus.AFK)
                state["takeover_started"] = True
                logging.getLogger(__name__).warning(
                    "[SOCKET] Player %s in match %s stayed disconnected into takeover window. Replacement bot starts.",
                    user_id, match_id,
                )
            await check_and_run_bot(str(match_id), ACTIVE_MATCHES)
        except asyncio.CancelledError:
            return
        finally:
            if MATCH_DISCONNECT_TASKS.get(key) is asyncio.current_task():
                MATCH_DISCONNECT_TASKS.pop(key, None)

    MATCH_DISCONNECT_TASKS[key] = asyncio.create_task(_takeover_if_still_gone())


def _schedule_disconnected_takeover_if_needed(match_id: str, engine: BattleEngine) -> bool:
    """Handle disconnected-current-player waiting/takeover even while client_ready is false."""
    from core.state import ReplacementStatus

    current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else getattr(engine, "current_player_id", None)
    if current_player is None:
        return False
    try:
        current_player_int = int(current_player)
    except (TypeError, ValueError):
        return False

    is_bot = engine.is_bot(current_player_int) if hasattr(engine, "is_bot") else False
    player_status = _player_replacement_status(engine, current_player_int)
    if (
        is_bot
        or player_status != ReplacementStatus.ACTIVE
        or not _is_player_disconnected(str(match_id), current_player_int)
    ):
        return False

    disconnect_state = MATCH_DISCONNECT_STATES.get((str(match_id), current_player_int), {})
    logger = logging.getLogger(__name__)
    if int(disconnect_state.get("timed_out_turns", 0)) >= 1:
        _schedule_disconnect_takeover(str(match_id), current_player_int, engine)
        logger.info(
            "check_and_run_bot: disconnected player %s is on second missed turn; takeover scheduled.",
            current_player_int,
        )
    else:
        logger.info(
            "check_and_run_bot: disconnected player %s is on first missed turn; waiting full timer.",
            current_player_int,
        )
    return True


def _player_replacement_status(engine: BattleEngine, user_id: int) -> Any:
    from core.state import ReplacementStatus

    if hasattr(engine, "get_player_replacement_status"):
        return engine.get_player_replacement_status(user_id)
    if hasattr(engine, "_arena") and engine._arena:
        state = engine._arena.state
        if state.p1.user_id == user_id:
            return getattr(state.p1, "replacement_status", ReplacementStatus.ACTIVE)
        if state.p2.user_id == user_id:
            return getattr(state.p2, "replacement_status", ReplacementStatus.ACTIVE)
    return ReplacementStatus.ACTIVE


async def _get_user_squad_id(db: Any, user_id: int) -> int:
    try:
        if hasattr(db, "get_user_profile"):
            profile = await db.get_user_profile(user_id)
            if profile:
                return int(profile.get("squad_id") or 0)
        if hasattr(db, "fetchval"):
            return int(await db.fetchval("SELECT COALESCE(squad_id, 0) FROM users WHERE user_id = $1", user_id) or 0)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to load squad_id for %s: %s", user_id, exc)
    return 0


async def _replacement_bot_allowed(match_id: str, engine: BattleEngine) -> bool:
    app_ref = getattr(sio, "app", None)
    raw_mode = (app_ref or {}).get("match_game_modes", {}).get(match_id, "") or getattr(engine, "game_mode", "") or "classic"
    engine_mode_config = getattr(engine, "mode_config", None)
    if engine_mode_config is not None and getattr(engine_mode_config, "mode_id", None) == raw_mode:
        mode_config = engine_mode_config
    else:
        mode_config = resolve_mode_config(raw_mode)
    if mode_config.classic.bots_allowed:
        return True

    if not app_ref:
        return False
    db = app_ref.get("db")
    if not db:
        return False

    try:
        p1_id = int(getattr(engine.p1_state, "user_id", 0) or 0)
        p2_id = int(getattr(engine.p2_state, "user_id", 0) or 0)
    except (TypeError, ValueError):
        return False
    if not p1_id or not p2_id:
        return False

    try:
        if hasattr(db, "are_friends") and await db.are_friends(p1_id, p2_id):
            return False
    except Exception as exc:
        logging.getLogger(__name__).warning("Friendship anti-fraud check failed for match %s: %s", match_id, exc)
        return False

    p1_squad = await _get_user_squad_id(db, p1_id)
    p2_squad = await _get_user_squad_id(db, p2_id)
    if p1_squad and p1_squad == p2_squad:
        return False
    return True


def _is_bot_controlled_player(engine: BattleEngine, player: Any) -> bool:
    from core.state import ReplacementStatus

    return bool(getattr(player, "is_bot", False)) or getattr(
        player, "replacement_status", ReplacementStatus.ACTIVE
    ) in (ReplacementStatus.AFK, ReplacementStatus.SURRENDERED)


def _both_players_bot_controlled(engine: BattleEngine) -> bool:
    if not hasattr(engine, "_arena") or not engine._arena:
        return False
    state = engine._arena.state
    return _is_bot_controlled_player(engine, state.p1) and _is_bot_controlled_player(engine, state.p2)


def _replacement_human_is_active_again(engine: BattleEngine, user_id: int | str) -> bool:
    from core.state import ReplacementStatus

    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return False
    is_real_bot = engine.is_bot(uid) if hasattr(engine, "is_bot") else False
    if is_real_bot:
        return False
    return _player_replacement_status(engine, uid) == ReplacementStatus.ACTIVE


async def _terminate_match_without_rewards(
    match_id: str,
    active_matches: dict[str, BattleEngine],
    *,
    reason: str,
    message: str,
) -> None:
    logger = logging.getLogger(__name__)
    engine = active_matches.get(str(match_id))
    if engine:
        engine.is_ended = True
        engine.rewards_granted = False
    _mark_match_ended(match_id)
    active_matches.pop(str(match_id), None)
    for key, task in list(MATCH_DISCONNECT_TASKS.items()):
        if key[0] != str(match_id):
            continue
        MATCH_DISCONNECT_TASKS.pop(key, None)
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
    for key in list(MATCH_DISCONNECT_STATES.keys()):
        if key[0] == str(match_id):
            MATCH_DISCONNECT_STATES.pop(key, None)
    MATCH_SESSIONS.pop(str(match_id), None)
    MATCH_LOCKS.pop(str(match_id), None)
    for sid, session_data in list(SID_TO_MATCH.items()):
        if str(session_data.get("match_id")) == str(match_id):
            SID_TO_MATCH.pop(sid, None)
    bot_task = BOT_TASKS.pop(str(match_id), None)
    if bot_task and not bot_task.done() and bot_task is not asyncio.current_task():
        bot_task.cancel()
    BOT_TASK_KEYS.pop(str(match_id), None)
    BOT_VS_BOT_MARKERS.pop(str(match_id), None)
    app_ref = getattr(sio, "app", None)
    if app_ref:
        app_ref.get("match_game_modes", {}).pop(str(match_id), None)
    try:
        await sio.emit(
            "match_terminated",
            {"match_id": str(match_id), "reason": reason, "message": message},
            room=str(match_id),
        )
    except Exception as emit_exc:
        logger.error("Failed to emit match_terminated event: %s", emit_exc)


async def _handle_bot_vs_bot_policy(match_id: str, engine: BattleEngine, active_matches: dict[str, BattleEngine]) -> bool:
    """Return True when bot-vs-bot policy consumed the check."""
    if not _both_players_bot_controlled(engine):
        BOT_VS_BOT_MARKERS.pop(str(match_id), None)
        return False

    current_turn = int(getattr(engine, "turn", 0) or 0)
    marker = BOT_VS_BOT_MARKERS.get(str(match_id))
    if marker is None:
        BOT_VS_BOT_MARKERS[str(match_id)] = current_turn
        logging.getLogger(__name__).warning(
            "[BOT_VS_BOT] Match %s entered bot-controlled state at turn %s. Allowing one full turn.",
            match_id, current_turn,
        )
        return False
    if current_turn > marker:
        await _terminate_match_without_rewards(
            str(match_id),
            active_matches,
            reason="bot_vs_bot_after_takeover",
            message="Матч автоматически завершён: оба игрока покинули игру",
        )
        return True
    return False


def _start_guarded_bot_task(match_id: str, engine: BattleEngine, bot_id: int | str) -> None:
    """Start one bot routine per match/current-turn pair."""
    turn_key = (str(bot_id), int(getattr(engine, "turn", 0) or 0))
    existing = BOT_TASKS.get(match_id)
    if existing and not existing.done() and BOT_TASK_KEYS.get(match_id) == turn_key:
        logging.getLogger(__name__).info(
            "Bot routine already running for match_id=%s bot_id=%s turn=%s",
            match_id, bot_id, turn_key[1],
        )
        return
    if existing and existing.done():
        BOT_TASKS.pop(match_id, None)
        BOT_TASK_KEYS.pop(match_id, None)

    current_task: asyncio.Task | None = None

    async def _guarded() -> None:
        try:
            await run_bot_routine(engine, bot_id)
            if not getattr(engine, "is_ended", False):
                if _both_players_bot_controlled(engine):
                    if await _handle_bot_vs_bot_policy(match_id, engine, ACTIVE_MATCHES):
                        return

                next_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else getattr(engine, "current_player_id", None)
                next_key = (str(next_player), int(getattr(engine, "turn", 0) or 0))
                if next_player is not None and next_key != turn_key:
                    await check_and_run_bot(match_id, ACTIVE_MATCHES)
        except asyncio.CancelledError:
            return
        finally:
            if BOT_TASKS.get(match_id) is current_task and BOT_TASK_KEYS.get(match_id) == turn_key:
                BOT_TASK_KEYS.pop(match_id, None)
                BOT_TASKS.pop(match_id, None)

    BOT_TASK_KEYS[match_id] = turn_key
    current_task = asyncio.create_task(_guarded())
    BOT_TASKS[match_id] = current_task

def _create_ssl_disabled_session():
    """Создать aiohttp сессию с отключенной проверкой SSL для локальной разработки."""
    import aiohttp
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    return aiohttp.ClientSession(connector=connector)


def _create_ssl_disabled_session():
    """Создать aiohttp сессию с отключенной проверкой SSL для локальной разработки."""
    import aiohttp
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    return aiohttp.ClientSession(connector=connector)


# ============================================================================
# Модульные функции управления ботами (вынесены на уровень модуля)
# ============================================================================

async def check_and_run_bot(match_id: str, active_matches: dict[str, BattleEngine]) -> None:
    """
    Функция-триггер для проверки и запуска бота.
    Проверяет: если текущий игрок == бот, запускает run_bot_routine.
    Вызывается:
    1. После создания матча (если бот ходит первым)
    2. После передачи хода игроком (если ход перешел к боту)
    3. После получения сигнала client_ready от фронтенда

    Args:
        match_id: идентификатор матча
        active_matches: словарь активных матчей
    """
    logger = logging.getLogger(__name__)

    # Получаем движок из активных матчей
    engine = active_matches.get(match_id)
    if not engine:
        logger.warning("check_and_run_bot: engine not found for match_id=%s", match_id)
        return

    if _schedule_disconnected_takeover_if_needed(str(match_id), engine):
        return

    # КРИТИЧНО: Проверяем, готов ли клиент к началу боя
    # Это предотвращает преждевременный ход бота до того, как игрок загрузит состояние
    if hasattr(engine, 'client_ready') and not engine.client_ready:
        logger.info("check_and_run_bot: client not ready for match_id=%s, bot waiting", match_id)
        return

    # Проверяем, является ли текущий игрок ботом
    try:
        # ДОБАВЛЕНО: Проверяем, не окончена ли игра
        current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
        state = engine.get_full_state(viewer_id=current_player) if hasattr(engine, "get_full_state") else {}
        if state.get("is_ended") or state.get("game_over"):
            logger.info("check_and_run_bot: game already ended for match_id=%s, skipping bot", match_id)
            return
        is_bot = engine.is_bot(current_player) if hasattr(engine, "is_bot") else False

        # Проверяем статус замены игрока (AFK/SURRENDERED)
        from core.state import ReplacementStatus
        player_status = ReplacementStatus.ACTIVE
        if hasattr(engine, "_arena") and engine._arena:
            arena_state = engine._arena.state
            if arena_state.p1.user_id == current_player:
                player_status = getattr(arena_state.p1, "replacement_status", ReplacementStatus.ACTIVE)
            elif arena_state.p2.user_id == current_player:
                player_status = getattr(arena_state.p2, "replacement_status", ReplacementStatus.ACTIVE)

        if await _handle_bot_vs_bot_policy(match_id, engine, active_matches):
            return

        if (
            not is_bot
            and player_status == ReplacementStatus.ACTIVE
            and current_player is not None
            and _is_player_disconnected(match_id, int(current_player))
        ):
            disconnect_state = MATCH_DISCONNECT_STATES.get((str(match_id), int(current_player)), {})
            if int(disconnect_state.get("timed_out_turns", 0)) >= 1:
                _schedule_disconnect_takeover(match_id, int(current_player), engine)
                logger.info(
                    "check_and_run_bot: disconnected player %s is on second missed turn; takeover scheduled.",
                    current_player,
                )
            else:
                logger.info(
                    "check_and_run_bot: disconnected player %s is on first missed turn; waiting full timer.",
                    current_player,
                )
            return

        # Бот играет, если: 1) игрок - бот ИЛИ 2) игрок AFK/SURRENDERED
        should_run_bot = is_bot or (player_status != ReplacementStatus.ACTIVE)

        logger.debug(
            "check_and_run_bot: match_id=%s current_player=%s is_bot=%s status=%s should_run_bot=%s",
            match_id, current_player, is_bot, player_status.value, should_run_bot,
        )

        if should_run_bot:
            if player_status != ReplacementStatus.ACTIVE:
                if not is_bot and not await _replacement_bot_allowed(match_id, engine):
                    await _terminate_match_without_rewards(
                        match_id,
                        active_matches,
                        reason="opponent_disconnected",
                        message="Противник отключился",
                    )
                    return
                logger.info("check_and_run_bot: player %s replaced by bot (status=%s)", current_player, player_status.value)
            else:
                logger.info("check_and_run_bot: starting bot routine for match_id=%s, bot_id=%s", match_id, current_player)

            # Запускаем ход бота асинхронно, с защитой от дублей на один ход
            _start_guarded_bot_task(match_id, engine, current_player)
        else:
            logger.debug("check_and_run_bot: current player is active match_id=%s current_player=%s", match_id, current_player)
    except Exception as exc:
        logger.error("check_and_run_bot: ошибка проверки бота для match_id=%s: %s", match_id, exc, exc_info=True)


async def run_bot_routine(engine: BattleEngine, bot_id: int | str) -> None:
    """
    Асинхронный сценарий хода бота:
    - короткая \"задержка обдумывания\",
    - принятие решения через BotAI,
    - последовательное выполнение всех действий,
    - завершение хода.
    """
    logger = logging.getLogger(__name__)
    logger.info("BOT ROUTINE STARTED for %s", bot_id)

    if _replacement_human_is_active_again(engine, bot_id):
        logger.info("run_bot_routine: player %s restored control before bot action", bot_id)
        return

    # КРИТИЧНО: ПЕРВАЯ ПРОВЕРКА - это вообще ход бота?
    if engine.current_player_id != bot_id:
        logger.warning("run_bot_routine called but not bot's turn (current=%s, bot=%s)", engine.current_player_id, bot_id)
        return

    # КРИТИЧНО: Проверяем флаг is_ended СРАЗУ - если игра завершена, бот МГНОВЕННО останавливается
    if hasattr(engine, 'is_ended') and engine.is_ended:
        logger.info("run_bot_routine: engine.is_ended=True, bot aborting immediately")
        return

    # ДОБАВЛЕНО: Проверяем, не окончена ли игра перед действиями бота
    try:
        state = engine.get_full_state(viewer_id=bot_id) if hasattr(engine, "get_full_state") else {}
        if state.get("is_ended") or state.get("game_over"):
            logger.info("run_bot_routine: game already ended (via state), bot skipping turn")
            return
    except Exception as exc:
        logger.debug("run_bot_routine: failed to check game state: %s", exc, exc_info=True)

    # Логируем состояние маны бота перед принятием решений
    try:
        bot_state = engine.get_player_state(bot_id)
        bot_mana = getattr(bot_state, "mana", 0)
        bot_max_mana = getattr(bot_state, "max_mana", 0)
        logger.info("BOT STATE: bot_id=%s mana=%s/%s turn=%s", bot_id, bot_mana, bot_max_mana, getattr(engine, 'turn', 'unknown'))
    except Exception as exc:
        logger.error("Failed to get bot state: %s", exc, exc_info=True)

    try:
        # Получаем сложность бота для расчёта задержек
        difficulty = getattr(engine, "bot_difficulty", "medium")
        difficulty_label = getattr(engine, "bot_difficulty_label", difficulty)
        match_id = getattr(engine, "match_id", "unknown")
        mode_config = getattr(engine, "mode_config", resolve_mode_config(getattr(engine, "game_mode", "classic")))
        classic_params = mode_config.classic

        # Единая задержка хода (Turn Delay): бот «раздумывает» перед серией действий
        delay_range = (
            classic_params.bot_hard_turn_delay_range
            if str(difficulty_label).startswith(("hard", "max"))
            else classic_params.bot_turn_delay_range
        )
        turn_delay = random.uniform(*delay_range)

        logger.info(
            "[BOT_THINKING] Match: %s | Difficulty: %s | Turn delay: %.2fs",
            match_id, difficulty_label, turn_delay
        )
        await asyncio.sleep(turn_delay)

        if _replacement_human_is_active_again(engine, bot_id):
            logger.info("run_bot_routine: player %s restored control during bot delay", bot_id)
            return

        # КРИТИЧНО: Еще раз проверяем is_ended перед тем как бот начнет думать
        if hasattr(engine, 'is_ended') and engine.is_ended:
            logger.info("run_bot_routine: engine.is_ended=True before actions, bot aborting")
            return

        # Определяем тип бота: ONNX Берсерк для любого bot-match бота с is_bot=True
        is_bot_player = engine.is_bot(bot_id) if hasattr(engine, "is_bot") else True
        use_berserk = is_bot_player and BERSERK_BRAIN is not None

        if use_berserk:
            logger.info("[SERVER] ONNX Берсерк для bot_id=%s (difficulty=%s)", bot_id, difficulty)
        else:
            reason = "no ONNX session" if BERSERK_BRAIN is None else "not bot player"
            logger.info("[SERVER] Rule-based BotAI для bot_id=%s (reason=%s)", bot_id, reason)

        # Пошаговое выполнение действий
        max_actions = 20
        action_count = 0
        total_action_delays = 0.0  # Суммарное время на технические паузы
        bot_aborted_for_reconnect = False

        for step in range(max_actions):
            if _replacement_human_is_active_again(engine, bot_id):
                logger.info("run_bot_routine: player %s restored control during bot step %d", bot_id, step)
                bot_aborted_for_reconnect = True
                break

            # Проверка завершения игры
            if hasattr(engine, 'is_ended') and engine.is_ended:
                logger.info("run_bot_routine: game ended at step %d", step)
                break

            # Проверка хода
            if engine.current_player_id != bot_id:
                logger.info("run_bot_routine: not bot's turn at step %d", step)
                break

            # Безопасность по времени: пропускаем технические паузы у края таймера.
            time_remaining = engine.get_turn_time_remaining() if hasattr(engine, "get_turn_time_remaining") else 25
            emergency_mode = time_remaining < classic_params.bot_emergency_threshold_seconds

            # Получаем легальные действия из core/engine
            try:
                if not hasattr(engine, '_arena') or engine._arena is None:
                    logger.error("[SERVER] engine._arena не инициализирован")
                    break

                legal_actions_obj = engine._arena.get_legal_actions(bot_id)
                legal_actions_dict = [engine._serialize_action(a) for a in legal_actions_obj]

                if not legal_actions_dict:
                    logger.info("[SERVER] Нет легальных действий, завершаем ход")
                    if engine.current_player_id == bot_id:
                        try:
                            engine.record_analytics_action(bot_id, {
                                "type": "end_turn", "forced": True, "reason": "no_legal_actions"
                            })
                        except Exception:
                            pass
                        engine.end_turn(bot_id)

                        # Отправляем обновленное состояние (для нового текущего игрока)
                        try:
                            new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                            full_state = engine.get_full_state(viewer_id=new_current_player)
                            event_data = {
                                "event_type": "turn_switched",
                                "match_id": match_id,
                                "state_p1": full_state,
                            }
                            BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                        except Exception as emit_exc:
                            logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)
                    break

                logger.debug("[SERVER] Доступно %d действий на шаге %d", len(legal_actions_dict), step)

                # Выбор действия
                action_id = 0
                if use_berserk:
                    # ONNX-инференс с учетом difficulty
                    try:
                        if hasattr(BERSERK_BRAIN, "get_action_async"):
                            action_id = await BERSERK_BRAIN.get_action_async(
                                engine._arena.state,
                                bot_id,
                                legal_actions_obj,
                                difficulty=difficulty,
                            )
                        else:
                            action_id = await asyncio.to_thread(
                                BERSERK_BRAIN.get_action,
                                engine._arena.state,
                                bot_id,
                                legal_actions_obj,
                                difficulty,
                            )
                        logger.info(
                            "[BERSERK] Выбрано действие ID=%d из %d (difficulty=%s)",
                            action_id, len(legal_actions_obj), difficulty
                        )
                    except Exception as exc:
                        logger.error("[BERSERK] Ошибка инференса: %s, fallback на rule-based", exc, exc_info=True)
                        # Fallback на rule-based
                        chosen_action = BotAI.decide_action(legal_actions_dict)
                        if chosen_action:
                            action_id = legal_actions_dict.index(chosen_action)
                else:
                    # Rule-based выбор
                    chosen_action = BotAI.decide_action(legal_actions_dict)
                    if chosen_action:
                        action_id = legal_actions_dict.index(chosen_action)

                # Проверка валидности
                if action_id < 0 or action_id >= len(legal_actions_dict):
                    logger.warning("[SERVER] Невалидный action_id=%d, принудительный end_turn", action_id)
                    if engine.current_player_id == bot_id:
                        try:
                            engine.record_analytics_action(bot_id, {
                                "type": "end_turn", "forced": True, "reason": "invalid_action_id"
                            })
                        except Exception:
                            pass
                        engine.end_turn(bot_id)

                        # Отправляем обновленное состояние (для нового текущего игрока)
                        try:
                            new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                            full_state = engine.get_full_state(viewer_id=new_current_player)
                            event_data = {
                                "event_type": "turn_switched",
                                "match_id": match_id,
                                "state_p1": full_state,
                            }
                            BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                        except Exception as emit_exc:
                            logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)
                    break

                action_dict = legal_actions_dict[action_id]
                action_type = action_dict.get("type")

                logger.info("[SERVER] Executing action: %s", action_dict)

                try:
                    engine.record_analytics_action(bot_id, action_dict)
                except Exception:
                    pass

                # Выполнение действия через execute_bot_action
                result = engine.execute_bot_action(action_dict)
                action_count += 1

                # Минимальная техническая пауза для анимаций UI.
                if not emergency_mode:
                    action_gap = random.uniform(*classic_params.bot_action_gap_range)
                    total_action_delays += action_gap
                    await asyncio.sleep(action_gap)
                else:
                    logger.warning(
                        "[BOT_EMERGENCY] Match: %s | Time remaining: %.1fs | Skipping delays",
                        match_id, time_remaining
                    )

                if not result.get("success", True):
                    logger.warning("[SERVER] Действие не выполнено: %s", result.get("error"))
                    # Пытаемся завершить ход
                    if engine.current_player_id == bot_id:
                        try:
                            engine.record_analytics_action(bot_id, {
                                "type": "end_turn", "forced": True, "reason": "invalid_action"
                            })
                        except Exception:
                            pass
                        engine.end_turn(bot_id)

                        # Отправляем обновленное состояние (для нового текущего игрока)
                        try:
                            new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                            full_state = engine.get_full_state(viewer_id=new_current_player)
                            event_data = {
                                "event_type": "turn_switched",
                                "match_id": match_id,
                                "state_p1": full_state,
                            }
                            BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                        except Exception as emit_exc:
                            logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)
                    break

                # Проверка game_over
                if result.get("game_over"):
                    logger.info("[SERVER] Игра завершена после действия")
                    break

                # Если это был end_turn, выходим
                if action_type == "end_turn":
                    logger.info("[SERVER] Ход бота завершен (end_turn), выполнено действий: %d", action_count)

                    # КРИТИЧНО: Отправляем обновленное состояние после смены хода (для нового текущего игрока)
                    try:
                        new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                        full_state = engine.get_full_state(viewer_id=new_current_player)
                        event_data = {
                            "event_type": "turn_switched",
                            "match_id": match_id,
                            "state_p1": full_state,
                        }
                        BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                        logger.info("[SERVER] Отправлено событие turn_switched после end_turn бота")
                    except Exception as emit_exc:
                        logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)

                    break

            except Exception as exc:
                logger.error("[SERVER] Ошибка на шаге %d: %s", step, exc, exc_info=True)
                try:
                    if engine.current_player_id == bot_id:
                        try:
                            engine.record_analytics_action(bot_id, {
                                "type": "end_turn", "forced": True, "reason": "exception"
                            })
                        except Exception:
                            pass
                        engine.end_turn(bot_id)

                        # Отправляем обновленное состояние (для нового текущего игрока)
                        try:
                            new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                            full_state = engine.get_full_state(viewer_id=new_current_player)
                            event_data = {
                                "event_type": "turn_switched",
                                "match_id": match_id,
                                "state_p1": full_state,
                            }
                            BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                        except Exception as emit_exc:
                            logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)
                except:
                    pass
                break

        # КРИТИЧНО: Если бот убил героя оппонента — вызываем _process_battle_end.
        # Внутри цикла мы только break, но не вызываем обработчик завершения.
        if (
            hasattr(engine, 'is_ended')
            and engine.is_ended
            and not getattr(engine, 'rewards_granted', False)
            and not getattr(engine, 'battle_end_processed', False)
        ):
            app = getattr(sio, 'app', None)
            if app:
                game_over_result = engine.check_game_over()
                logger.info(
                    "[BOT_GAME_OVER] Bot triggered game over in match=%s winner=%s — calling _process_battle_end",
                    match_id, game_over_result.get("winner_id"),
                )
                try:
                    await _process_battle_end(app, match_id, engine, game_over_result.get("winner_id"))
                    try:
                        await sio.emit(
                            "game_over",
                            _build_game_over_payload(engine, game_over_result.get("winner_id"), reason="hero_death"),
                            room=match_id,
                        )
                    except Exception:
                        pass
                except Exception as exc:
                    logger.error("[BOT_GAME_OVER] _process_battle_end failed: %s", exc, exc_info=True)
            else:
                logger.error("[BOT_GAME_OVER] No app available for bot game_over processing match=%s", match_id)

        # Итоговая статистика хода бота
        total_turn_time = turn_delay + total_action_delays
        logger.info(
            "[BOT_TURN_SUMMARY] Match: %s | Difficulty: %s | Total thinking time: %.2fs "
            "(turn_delay=%.2fs + action_gaps=%.2fs) | Actions executed: %d",
            match_id, difficulty, total_turn_time, turn_delay, total_action_delays, action_count
        )

        # Финальная проверка: если бот все еще владеет ходом, завершаем принудительно
        if (
            engine.current_player_id == bot_id
            and not bot_aborted_for_reconnect
            and not (hasattr(engine, 'is_ended') and engine.is_ended)
        ):
            logger.warning("[SERVER] Бот не завершил ход явно, принудительный end_turn")
            try:
                try:
                    engine.record_analytics_action(bot_id, {
                        "type": "end_turn", "forced": True, "reason": "max_actions_or_fallback"
                    })
                except Exception:
                    pass
                engine.end_turn(bot_id)

                # КРИТИЧНО: Отправляем обновленное состояние после принудительного end_turn (для нового текущего игрока)
                try:
                    new_current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else engine.current_player_id
                    full_state = engine.get_full_state(viewer_id=new_current_player)
                    event_data = {
                        "event_type": "turn_switched",
                        "match_id": match_id,
                        "state_p1": full_state,
                    }
                    BATTLE_EVENT_EMITTER.emit("turn_switched", match_id, event_data)
                    logger.info("[SERVER] Отправлено событие turn_switched после принудительного end_turn")
                except Exception as emit_exc:
                    logger.error("[SERVER] Ошибка отправки turn_switched: %s", emit_exc)

            except Exception as exc:
                logger.error("[SERVER] Ошибка принудительного end_turn: %s", exc)

    except Exception as exc:
        logger.error("run_bot_routine fatal error: %s", exc, exc_info=True)


async def trigger_bot_move(match_id: str) -> None:
    """
    Триггер для запуска хода бота после действия игрока.
    Проверяет, перешел ли ход к боту, и если да - запускает run_bot_routine.

    Args:
        match_id: ID матча
    """
    logger = logging.getLogger(__name__)

    engine = ACTIVE_MATCHES.get(match_id)
    if not engine:
        logger.warning("[trigger_bot_move] Движок не найден для match_id=%s", match_id)
        return

    # Проверяем, что игра еще идет
    if hasattr(engine, 'is_ended') and engine.is_ended:
        logger.debug("[trigger_bot_move] Игра завершена, бот не запускается")
        return

    # Проверяем, является ли текущий игрок ботом
    current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else None
    if current_player is None:
        logger.warning("[trigger_bot_move] Не удалось определить текущего игрока")
        return

    is_bot = engine.is_bot(current_player) if hasattr(engine, "is_bot") else False

    if is_bot:
        logger.info("[trigger_bot_move] Ход перешел к боту %s, запускаем run_bot_routine", current_player)

        # Запускаем ход бота асинхронно, с защитой от дублей на один ход
        _start_guarded_bot_task(match_id, engine, current_player)
    else:
        logger.debug("[trigger_bot_move] Текущий игрок %s не является ботом", current_player)


async def _handle_natural_turn_timeout(app: web.Application, match_id: str, engine: BattleEngine) -> bool:
    """Auto-end one naturally expired turn and update AFK/disconnect accounting."""
    logger = logging.getLogger(__name__)
    if _is_match_waiting_for_players(engine):
        return False
    if not hasattr(engine, "is_turn_expired") or not engine.is_turn_expired() or getattr(engine, "is_ended", False):
        return False

    current_player = engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else getattr(engine, "current_player_id", None)
    if current_player is None:
        return False

    try:
        current_player_int = int(current_player)
    except (TypeError, ValueError):
        return False

    lock = _get_match_lock(str(match_id))
    auto_ended = False
    async with lock:
        if not engine.is_turn_expired() or getattr(engine, "is_ended", False):
            return False

        is_bot_turn = engine.is_current_player_bot() if hasattr(engine, "is_current_player_bot") else False
        status = _player_replacement_status(engine, current_player_int)

        if _is_player_disconnected(str(match_id), current_player_int) and not is_bot_turn:
            missed = _record_disconnected_turn_timeout(str(match_id), current_player_int)
            logger.warning(
                "⏰ Disconnected player %s timed out naturally in match %s (missed_turns=%s)",
                current_player_int, match_id, missed,
            )
            if (
                missed >= 2
                and getattr(status, "value", "active") == "active"
                and hasattr(engine, "set_player_replacement_status")
            ):
                from core.state import ReplacementStatus

                if not await _replacement_bot_allowed(str(match_id), engine):
                    await _terminate_match_without_rewards(
                        str(match_id),
                        ACTIVE_MATCHES,
                        reason="opponent_disconnected",
                        message="Противник отключился",
                    )
                    return True

                engine.set_player_replacement_status(current_player_int, ReplacementStatus.AFK)
                disconnect_state = MATCH_DISCONNECT_STATES.get((str(match_id), current_player_int))
                if disconnect_state is not None:
                    disconnect_state["takeover_started"] = True
                logger.warning(
                    "⏰ Disconnected player %s in match %s reached AFK takeover after %s missed turns.",
                    current_player_int, match_id, missed,
                )
            else:
                try:
                    engine.end_turn(current_player_int)
                    auto_ended = True
                except Exception as exc:
                    logger.warning("Failed to auto-end turn on timer expiry: %s", exc)
                    return False
        elif not is_bot_turn and getattr(status, "value", "active") == "active" and hasattr(engine, "mark_timeout"):
            engine.mark_timeout(current_player_int)
            try:
                engine.end_turn(current_player_int)
                auto_ended = True
            except Exception as exc:
                logger.warning("Failed to auto-end turn on timer expiry: %s", exc)
                return False
        else:
            try:
                engine.end_turn(current_player_int)
                auto_ended = True
            except Exception as exc:
                logger.warning("Failed to auto-end turn on timer expiry: %s", exc)
                return False

    sio_inst = app.get("socketio")
    if sio_inst and auto_ended:
        try:
            event_name = "turn_end"
            await _emit_personalized_match_state(
                sio_inst,
                str(match_id),
                event_name,
                {
                    "match_id": str(match_id),
                    "auto_ended": True,
                    "reason": "time_expired",
                },
                engine=engine,
            )
        except Exception as emit_exc:
            logger.warning("Failed to emit auto turn_end for match %s: %s", match_id, emit_exc)

    await check_and_run_bot(str(match_id), ACTIVE_MATCHES)
    return True


def create_web_app(
    db: Database,
    bot_token: str,
    extraid_db: ExtraIDDatabase | None = None,
    payment_service=None,
    rustore_payment_service=None,
    rustore_console_app_id: str = "",
    rustore_app_url: str = "https://www.rustore.ru/catalog/app/ru.extraarena.app",
    payment_provider_order: str = "yookassa,rustore,stars",
    webapp_url: str | None = None,
    extra_shop_url: str | None = None,
    stars_rate_rub: float = 1.5,
    stars_markup: float = 1.2,
    stars_test_mode: bool = False,
    android_latest_version_code: int = 2,
    android_latest_version_name: str = "0.2.0",
    android_min_supported_version_code: int | None = None,
    android_update_channel_url: str = "https://t.me/extraarenamobile",
    android_apk_url: str = "https://apk.laveqox.ru",
    battle_engine=None,
) -> web.Application:
    logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
    logging.getLogger('engineio').setLevel(logging.WARNING)
    logging.getLogger('socketio').setLevel(logging.WARNING)

    app = web.Application()
    app["db"] = db
    app["extraid_db"] = extraid_db
    app["bot_token"] = bot_token
    app["payment_service"] = payment_service
    app["rustore_payment_service"] = rustore_payment_service
    app["rustore_console_app_id"] = str(rustore_console_app_id or "")
    app["rustore_app_url"] = str(rustore_app_url or "https://www.rustore.ru/catalog/app/ru.extraarena.app")
    app["payment_provider_order"] = str(payment_provider_order or "yookassa,rustore,stars")
    app["webapp_url"] = webapp_url or "https://t.me/your_bot"
    app["extra_shop_url"] = extra_shop_url or webapp_url or "https://t.me/your_bot"
    app["stars_rate_rub"] = stars_rate_rub
    app["stars_markup"] = stars_markup
    app["stars_test_mode"] = stars_test_mode
    app["android_latest_version_code"] = int(android_latest_version_code or 0)
    app["android_latest_version_name"] = str(android_latest_version_name or "")
    app["android_min_supported_version_code"] = int(
        android_min_supported_version_code
        if android_min_supported_version_code is not None
        else android_latest_version_code
    )
    app["android_update_channel_url"] = str(android_update_channel_url or "https://t.me/extraarenamobile")
    app["android_apk_url"] = str(android_apk_url or "https://apk.laveqox.ru")
    app["admin_ids"] = {ADMIN_ID}
    # Инициализируем игровые сервисы и прокладываем их в контекст приложения
    services = initialize_game_services(db, battle_engine=battle_engine)
    app["bot_generator"] = services["bot_generator"]
    app["battle_engine"] = services["battle_engine"]
    app["matchmaker"] = services["matchmaker"]
    app["active_matches"] = services["active_matches"]
    app["event_emitter"] = services["event_emitter"]
    app["socketio"] = sio
    app["match_game_modes"] = {}
    app["online_users"] = {}
    logging.getLogger(__name__).warning(
        "PvP matchmaking and active battle state are process-local. Use one web process "
        "unless a shared queue/state backend is added."
    )

    # Инициализация ONNX-мозга Берсерка с профилями сложности
    global BERSERK_BRAIN
    try:
        from infrastructure.config import BOT_DIFFICULTY_PROFILES
        BERSERK_BRAIN = BerserkInference(profiles=BOT_DIFFICULTY_PROFILES)
        loaded_profiles = list(BERSERK_BRAIN.sessions.keys())
        if loaded_profiles:
            logging.getLogger(__name__).info(
                f"✅ ONNX Берсерк загружен: {len(loaded_profiles)} профилей ({', '.join(loaded_profiles)})"
            )
        else:
            logging.getLogger(__name__).warning(
                "⚠️ ONNX Берсерк не загрузил ни одного TrainV2 v4 профиля; боты будут использовать rule-based AI"
            )
            BERSERK_BRAIN = None
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "⚠️ Не удалось загрузить ONNX Берсерк: %s (боты будут использовать rule-based AI)",
            exc,
        )
        BERSERK_BRAIN = None

    # Настраиваем обработчики событий для Socket.io
    setup_battle_events()

    # Прикрепляем Socket.io к приложению aiohttp
    sio.attach(app)

    # КРИТИЧНО: Сохраняем ссылку на app в sio для доступа из обработчиков Socket.io
    sio.app = app

    MODE_TEMP_UNAVAILABLE_MESSAGE = "Этот режим временно недоступен. Следи за новостями, скоро все починем!"
    MAINTENANCE_MESSAGE = "Ведутся технические работы"

    def _mode_feature_key(mode_id: str | None) -> str:
        mode = str(mode_id or "classic")
        if mode == "training":
            return "training"
        if mode == "friendly":
            return "friendly"
        if mode == "extra_arena" or mode.startswith("extra_arena"):
            return "extra_arena"
        return "classic"

    def _runtime_unavailable_payload(feature: str) -> dict[str, Any]:
        return {
            "error": "feature_unavailable",
            "feature": feature,
            "message": MODE_TEMP_UNAVAILABLE_MESSAGE,
        }

    async def _is_admin_user(db_instance: Database, user_id: int | None) -> bool:
        if not user_id:
            return False
        if int(user_id) in app.get("admin_ids", set()):
            return True
        try:
            return bool(await db_instance.is_admin(int(user_id)))
        except Exception:
            return False

    async def _runtime_config_safe(db_instance: Database) -> dict[str, Any]:
        try:
            return await db_instance.get_runtime_config()
        except Exception as exc:
            logging.getLogger(__name__).warning("runtime config fallback: %s", exc)
            return {
                "maintenance_mode": {"enabled": False},
                "feature_availability": dict(RUNTIME_FEATURE_DEFAULTS),
                "disabled_card_ids": [],
            }

    async def _is_runtime_feature_enabled(db_instance: Database, feature: str) -> bool:
        config = await _runtime_config_safe(db_instance)
        availability = config.get("feature_availability") or {}
        return bool(availability.get(feature, True))

    @web.middleware
    async def admin_auth_middleware(request: web.Request, handler):
        if not request.path.startswith("/api/admin/"):
            return await handler(request)
        user_id = await require_user_id(request)
        if not await _is_admin_user(request.app["db"], user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        request["admin_user_id"] = user_id
        return await handler(request)

    @web.middleware
    async def runtime_gate_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            return await handler(request)

        path = request.path or ""
        if not path.startswith("/api/"):
            return await handler(request)
        if path.startswith("/api/admin/"):
            return await handler(request)

        db_instance: Database = request.app["db"]
        config = await _runtime_config_safe(db_instance)
        user_id: int | None = None
        try:
            user_id = await require_user_id(request)
        except web.HTTPException:
            if request.method == "POST":
                try:
                    payload = await request.json()
                    user_id = await require_user_id_from_payload(request, payload if isinstance(payload, dict) else {})
                except Exception:
                    user_id = None
            else:
                user_id = None
        is_admin = await _is_admin_user(db_instance, user_id)

        maintenance_allowed = (
            "/api/runtime/status",
            "/api/mobile/client-version",
            "/api/community/news",
        )
        maintenance_on = bool((config.get("maintenance_mode") or {}).get("enabled", False))
        if maintenance_on and not is_admin and not path.startswith(maintenance_allowed):
            return web.json_response(
                {"error": "maintenance_mode", "message": MAINTENANCE_MESSAGE},
                status=503,
            )

        feature_prefixes = (
            ("shop", ("/api/shop/", "/api/payments/", "/api/cases/")),
            ("collection", ("/api/cards", "/api/deck/")),
            ("squads", ("/api/squads",)),
        )
        availability = config.get("feature_availability") or {}
        if not is_admin:
            for feature, prefixes in feature_prefixes:
                if path.startswith(prefixes) and not bool(availability.get(feature, True)):
                    return web.json_response(_runtime_unavailable_payload(feature), status=503)

        return await handler(request)

    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        """
        CORS middleware с защитой от None response.
        Если handler возвращает None, сразу возвращаем None без обработки headers.
        """
        response = await handler(request)

        # КРИТИЧНО: Проверяем, что response не None перед добавлением headers
        if response is None:
            return response

        if _is_admin_surface_request(request):
            if int(getattr(response, "status", 200)) >= 500:
                return web.json_response(
                    {"error": "internal_server_error", "message": "Internal server error"},
                    status=response.status,
                )
            return response

        # Добавляем CORS headers только если response существует
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    app.middlewares.append(admin_auth_middleware)
    app.middlewares.append(runtime_gate_middleware)
    app.middlewares.append(cors_middleware)

    async def health_check(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "service": "extracards-webapp"})

    async def mobile_client_version_handler(request: web.Request) -> web.Response:
        def as_int(value: Any, fallback: int = 0) -> int:
            try:
                return int(str(value or "").strip())
            except (TypeError, ValueError):
                return fallback

        platform = str(request.rel_url.query.get("platform") or "android").lower()
        current_code = as_int(request.rel_url.query.get("version_code"), 0)
        current_name = str(request.rel_url.query.get("version_name") or "")
        latest_code = int(app.get("android_latest_version_code") or 0)
        min_supported_code = int(app.get("android_min_supported_version_code") or latest_code)
        latest_name = str(app.get("android_latest_version_name") or "")
        required = platform == "android" and current_code < max(latest_code, min_supported_code)

        return web.json_response({
            "platform": platform,
            "current_version_code": current_code,
            "current_version_name": current_name,
            "latest_version_code": latest_code,
            "latest_version_name": latest_name,
            "min_supported_version_code": min_supported_code,
            "required": required,
            "update_required": required,
            "update_url": app.get("android_update_channel_url"),
            "telegram_url": app.get("android_update_channel_url"),
            "apk_url": app.get("android_apk_url"),
            "rustore_url": app.get("rustore_app_url"),
            "message": "Вышло обновление ExtraArena. Скачай новую версию, чтобы продолжить игру.",
        })

    async def runtime_status_handler(request: web.Request) -> web.Response:
        db_instance: Database = request.app["db"]
        user_id: int | None = None
        try:
            user_id = await require_user_id(request)
        except web.HTTPException:
            user_id = None
        config = await _runtime_config_safe(db_instance)
        return web.json_response({
            **config,
            "is_admin": await _is_admin_user(db_instance, user_id),
        })

    async def index(_: web.Request) -> web.FileResponse:
        index_path = WEBAPP_DIR / "index.html"
        if not index_path.exists():
            raise web.HTTPInternalServerError(text="index.html not found")
        return web.FileResponse(index_path, headers={"Cache-Control": "no-store, no-cache, must-revalidate"})

    async def battle_page_handler(_: web.Request) -> web.Response:
        """
        Отдает выделенную страницу боя, чтобы фронтенд мог открывать бой по отдельному URL.
        Читаем HTML с диска и возвращаем его целиком, так как здесь нет шаблонизации.
        """
        arena_path = WEBAPP_DIR / "arena.html"
        if not arena_path.exists():
            raise web.HTTPInternalServerError(text="arena.html not found")

        return web.Response(
            text=arena_path.read_text(encoding="utf-8"),
            content_type="text/html",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    async def static_handler(request: web.Request) -> web.FileResponse:
        relative_path = request.match_info["path"]
        if ".." in relative_path:
            raise web.HTTPForbidden()

        # Если путь начинается с DesignAssets, ищем в DesignAssets
        if relative_path.startswith("DesignAssets/"):
            file_path = DESIGN_ASSETS_DIR / relative_path.replace("DesignAssets/", "", 1)
        else:
            # Иначе ищем в webapp
            file_path = WEBAPP_DIR / relative_path

        if not file_path.exists() or not file_path.is_file():
            raise web.HTTPNotFound()

        cache_headers = STATIC_ASSET_CACHE_HEADERS
        if relative_path.endswith(".css"):
            cache_headers = {"Cache-Control": "no-store, no-cache, must-revalidate"}
        return web.FileResponse(file_path, headers=cache_headers)

    def _serialize_settings_record(settings_record) -> dict[str, Any]:
        settings_dict = dict(settings_record)
        for key, value in list(settings_dict.items()):
            if hasattr(value, "isoformat"):
                settings_dict[key] = value.isoformat()
        return settings_dict

    async def profile_handler(request: web.Request) -> web.Response:
        photo_url = None
        first_name = None
        username = None
        first_name_from_data = None
        last_name = None

        user_id = await require_user_id(request)

        init_data = request.rel_url.query.get("_auth")
        if init_data and not init_data.isdigit():
            verified_data = _verify_init_data(init_data, request.app["bot_token"])
            if verified_data:
                user_str = verified_data.get("user", "")
                if user_str:
                    import json
                    try:
                        user_data = json.loads(user_str)
                        photo_url = user_data.get("photo_url")
                        first_name = user_data.get("first_name")
                        first_name_from_data = first_name
                        username = user_data.get("username")
                        last_name = user_data.get("last_name")
                    except Exception:
                        pass

        if user_id and not photo_url:
            try:
                async with _create_ssl_disabled_session() as session:
                    url = f"https://api.telegram.org/bot{request.app['bot_token']}/getUserProfilePhotos"
                    async with session.get(url, params={"user_id": user_id, "limit": 1}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("ok") and data.get("result", {}).get("total_count", 0) > 0:
                                file_id = data["result"]["photos"][0][0]["file_id"]
                                file_url = f"https://api.telegram.org/bot{request.app['bot_token']}/getFile"
                                async with session.get(file_url, params={"file_id": file_id}) as file_resp:
                                    if file_resp.status == 200:
                                        file_data = await file_resp.json()
                                        if file_data.get("ok"):
                                            file_path = file_data["result"]["file_path"]
                                            photo_url = f"https://api.telegram.org/file/bot{request.app['bot_token']}/{file_path}"
            except Exception:
                pass

        # Проверяем, есть ли пользователь
        record = await db.get_user_profile(user_id)
        welcome_should_show = False

        if not record:
            # Пользователя нет - возвращаем специальный ответ для показа приветствия
            # НЕ создаем пользователя здесь - это будет сделано после завершения приветствия
            return web.json_response({
                "error": "user_not_found",
                "should_show_welcome": True,
                "need_registration": True
            }, status=404)

        settings_record = await db.get_user_settings(user_id)
        settings_data = {}
        if settings_record:
            settings_data = {
                "notif_cases": settings_record["notif_cases"],
                "notif_daily_rewards": settings_record["notif_daily_rewards"],
                "notif_game_invites": settings_record["notif_game_invites"],
                "notif_friend_requests": settings_record["notif_friend_requests"],
                "notif_events": settings_record["notif_events"],
                "notif_news": settings_record["notif_news"],
                "notif_generator": settings_record.get("notif_generator", True),
                "notif_shop": settings_record.get("notif_shop", False),
                "notif_reminders": settings_record.get("notif_reminders", True),
                "notif_squad_member_role": settings_record.get("notif_squad_member_role", True),
                "notif_squad_new_member": settings_record.get("notif_squad_new_member", True),
                "notif_squad_disbanded": settings_record.get("notif_squad_disbanded", True),
                "notif_squad_boost": settings_record.get("notif_squad_boost", True),
                "notif_extra_arena_modifiers": settings_record.get("notif_extra_arena_modifiers", True),
                "notification_delivery_mode": settings_record.get("notification_delivery_mode", "app_then_telegram"),
                "ads_enabled": settings_record["ads_enabled"],
                "sound_music": settings_record["sound_music"],
                "sound_sfx": settings_record["sound_sfx"],
                "social_block_friend_requests": settings_record["social_block_friend_requests"],
                "wins_since_last_case": settings_record.get("wins_since_last_case", 0),
            }

        # Убеждаемся, что title есть (по умолчанию "Игрок")
        title = record.get("title") or "Игрок"

        # Получить asset_path надетой аватарки
        equipped_avatar_row = await db.fetchrow("""
            SELECT ci.asset_path
            FROM user_equipped_cosmetics uec
            JOIN cosmetic_items ci ON ci.id = uec.cosmetic_id
            WHERE uec.user_id = $1 AND uec.item_type = 'avatar'
        """, user_id)
        equipped_avatar_url = equipped_avatar_row["asset_path"] if equipped_avatar_row else None

        # Вычисляем уровни для показа в UI (информационные, не источник истины)
        avg_level = await db.get_player_deck_avg_level(user_id)
        max_level = await db.get_player_deck_max_level(user_id)
        extraid_linked_telegram = False
        if record.get("extra_account_id"):
            try:
                extra = await request.app["extraid_db"].get_extra_account_by_user_id(user_id)
                extraid_linked_telegram = bool(
                    extra and extra.get("user_id") is not None and int(extra.get("user_id")) < 9000000000000
                )
            except Exception:
                extraid_linked_telegram = False

        payload: dict[str, Any] = {
            "user_id": record["user_id"],
            "is_admin": await _is_admin_user(db, user_id),
            "username": record.get("username"),
            "first_name": first_name or record.get("first_name"),
            "photo_url": photo_url,
            "extra_pass": record.get("extra_pass", "inactive"),
            "equipped_avatar_url": equipped_avatar_url,
            "trophies": record.get("trophies", 0),
            "max_trophies": record.get("max_trophies", 0),
            "league": record.get("league", 1),
            "keys": record.get("keys", 0),
            "gems": record.get("gems", 0),
            "coins": record.get("coins", 0),
            "squad_id": record.get("squad_id"),
            "status": record.get("status", "active"),
            "reg_date": record["reg_date"].isoformat() if record.get("reg_date") else None,
            "stars": record.get("stars", 0),
            "energy": record.get("energy", 5),
            "energy_cd": record.get("energy_cd").isoformat() if record.get("energy_cd") else None,
            "season": record.get("season", 0),
            "title": title,
            "img": record.get("img", ""),
            "selected_hero_id": record.get("selected_hero_id", 0),
            "custom_nickname": record.get("custom_nickname"),
            "nickname_changed": record.get("nickname_changed", False),
            "extra_account_id": str(record.get("extra_account_id")) if record.get("extra_account_id") else None,
            "extraid_linked_telegram": extraid_linked_telegram,
            "settings": settings_data,
            "should_show_welcome": welcome_should_show,
            "avg_level": avg_level,
            "max_level": max_level,
        }

        return web.json_response(payload)

    async def settings_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)

        if request.method == "GET":
            try:
                # Проверяем, существует ли пользователь
                user_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", user_id)
                if not user_exists:
                    # Пользователя нет - возвращаем настройки по умолчанию
                    return web.json_response({
                        "notif_cases": False,
                        "notif_daily_rewards": False,
                        "notif_game_invites": False,
                        "notif_friend_requests": False,
                        "notif_events": False,
                        "notif_news": False,
                        "notif_generator": True,
                        "notif_shop": False,
                        "notif_reminders": True,
                        "notif_squad_member_role": True,
                        "notif_squad_new_member": True,
                        "notif_squad_disbanded": True,
                        "notif_squad_boost": True,
                        "notif_extra_arena_modifiers": True,
                        "notification_delivery_mode": "app_then_telegram",
                        "ads_enabled": False,
                        "sound_music": True,
                        "sound_sfx": True,
                        "social_block_friend_requests": False,
                    })

                settings_record = await db.get_user_settings(user_id)
                if not settings_record:
                    # Создаем настройки по умолчанию, если их нет
                    db_instance = request.app["db"]
                    await db_instance.execute(
                        """
                        INSERT INTO user_settings (user_id)
                        VALUES ($1)
                        ON CONFLICT (user_id) DO NOTHING
                        """,
                        user_id,
                    )
                    settings_record = await db.get_user_settings(user_id)

                settings_dict = _serialize_settings_record(settings_record)

                return web.json_response(settings_dict)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    "Ошибка получения настроек для user_id %s: %s", user_id, e, exc_info=True
                )
                return web.json_response(
                    {"error": "internal_server_error"}, status=500
                )

        elif request.method == "POST":
            try:
                data = await request.json()
                import logging
                logging.getLogger(__name__).info(
                    "Сохранение настроек для user_id %s: %s", user_id, data
                )
                await db.update_user_settings(user_id, **data)

                # Проверяем, что настройки сохранились
                updated_settings = await db.get_user_settings(user_id)
                logging.getLogger(__name__).info(
                    "Настройки сохранены для user_id %s: %s", user_id, dict(updated_settings) if updated_settings else "не найдены"
                )

                settings_dict = _serialize_settings_record(updated_settings) if updated_settings else {}

                return web.json_response({"status": "ok", "settings": settings_dict})
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    "Ошибка сохранения настроек для user_id %s: %s", user_id, e, exc_info=True
                )
                return web.json_response(
                    {"error": "internal_server_error", "message": str(e)}, status=500
                )

    async def admin_players_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method == "GET":
            # Получаем список всех игроков
            try:
                players = await db.fetch(
                    """
                    SELECT u.user_id, u.username, u.first_name, u.extra_pass,
                           u.trophies, u.status, u.reg_date
                    FROM users u
                    ORDER BY u.trophies DESC
                    LIMIT 100
                    """
                )
                total_count = await db.fetchval("SELECT COUNT(*) FROM users")

                # Преобразуем в словари и сериализуем datetime
                from datetime import datetime
                players_list = []
                for p in players:
                    player_dict = dict(p)
                    if "reg_date" in player_dict and player_dict["reg_date"]:
                        if isinstance(player_dict["reg_date"], datetime):
                            player_dict["reg_date"] = player_dict["reg_date"].isoformat()
                    players_list.append(player_dict)

                return web.json_response({
                    "players": players_list,
                    "total": total_count or len(players_list)
                })
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    "Ошибка получения списка игроков: %s", e, exc_info=True
                )
                return web.json_response(
                    {"error": "internal_server_error", "message": str(e)}, status=500
                )

    async def admin_stats_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        stats = await db.get_statistics()
        return web.json_response(stats)

    async def change_nickname_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            new_nickname = data.get("nickname", "").strip()

            if not new_nickname:
                return web.json_response({"error": "nickname_required"}, status=400)

            if len(new_nickname) > 20:
                return web.json_response({"error": "nickname_too_long"}, status=400)

            # Проверяем, первая ли это смена
            profile = await db.fetchrow(
                "SELECT nickname_changed FROM profiles WHERE user_id = $1", user_id
            )
            nickname_changed = profile["nickname_changed"] if profile else False
            cost_gems = 0 if not nickname_changed else 500

            result = await db.change_nickname(user_id, new_nickname, cost_gems)

            if not result["success"]:
                return web.json_response(result, status=400)

            return web.json_response({
                "success": True,
                "nickname": new_nickname,
                "cost": cost_gems,
                "is_first_change": result["is_first_change"]
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка смены никнейма для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def cosmetics_owned_handler(request: web.Request) -> web.Response:
        """GET /api/cosmetics/owned — предметы пользователя + equipped."""
        user_id = await require_user_id(request)
        data = await db.get_user_cosmetics(user_id)
        return web.json_response(data)

    async def cosmetics_equip_handler(request: web.Request) -> web.Response:
        """POST /api/cosmetics/equip — надеть предмет."""
        user_id = await require_user_id(request)
        try:
            payload = await request.json()
            cosmetic_id = int(payload.get("cosmetic_id", 0))
            if not cosmetic_id:
                return web.json_response({"error": "cosmetic_id_required"}, status=400)
            result = await db.equip_cosmetic(user_id, cosmetic_id)
            if not result["success"]:
                return web.json_response(result, status=400)
            if result.get("success") and isinstance(result.get("rewards"), dict):
                result = {
                    **result,
                    "gems": result["rewards"].get("gems", 0),
                    "coins": result["rewards"].get("coins", 0),
                    "keys": result["rewards"].get("keys", 0),
                    "extrapass": result["rewards"].get("extrapass", False),
                }
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("equip_cosmetic error user=%s: %s", user_id, e)
            return web.json_response({"error": "internal_error"}, status=500)

    async def promocode_use_handler(request: web.Request) -> web.Response:
        """Обработчик использования промокода."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            code = data.get("code", "").strip()

            if not code:
                return web.json_response({"error": "code_required"}, status=400)

            result = await db.use_promocode(user_id, code)

            if not result["success"]:
                error_messages = {
                    "not_found": "Промокод не найден",
                    "expired": "Промокод истек",
                    "already_used": "Вы уже использовали этот промокод",
                    "not_eligible": "Этот промокод доступен только новым игрокам"
                }
                return web.json_response({
                    "success": False,
                    "error": result["error"],
                    "message": error_messages.get(result["error"], "Ошибка использования промокода")
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка использования промокода для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def promocode_create_handler(request: web.Request) -> web.Response:
        """Обработчик создания промокода (только для админа)."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            code = data.get("code", "").strip().upper()
            type = data.get("type", "permanent")
            reward_gems = data.get("reward_gems", 0)
            reward_coins = data.get("reward_coins", 0)
            reward_keys = data.get("reward_keys", 0)
            reward_extrapass = data.get("reward_extrapass", False)
            expires_at = data.get("expires_at")

            if not code:
                return web.json_response({"error": "code_required"}, status=400)

            if type not in ["permanent", "personal", "welcome"]:
                return web.json_response({"error": "invalid_type"}, status=400)

            expires_datetime = None
            if expires_at:
                from datetime import datetime
                try:
                    expires_datetime = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                except Exception:
                    return web.json_response({"error": "invalid_expires_at"}, status=400)

            result = await db.create_promocode(
                code=code,
                type=type,
                reward_gems=reward_gems,
                reward_coins=reward_coins,
                reward_keys=reward_keys,
                reward_extrapass=reward_extrapass,
                created_by=user_id,
                expires_at=expires_datetime
            )

            if not result["success"]:
                error_messages = {
                    "code_exists": "Промокод с таким кодом уже существует"
                }
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": error_messages.get(result.get("error"), "Ошибка создания промокода")
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания промокода для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def promocode_list_handler(request: web.Request) -> web.Response:
        """Обработчик получения списка промокодов (только для админа)."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        try:
            promocodes = await db.get_promocodes_list()
            # Преобразуем datetime в строки для JSON
            for p in promocodes:
                if p.get("created_at") and hasattr(p["created_at"], "isoformat"):
                    p["created_at"] = p["created_at"].isoformat()
                if p.get("expires_at") and hasattr(p["expires_at"], "isoformat"):
                    p["expires_at"] = p["expires_at"].isoformat()
            return web.json_response({"promocodes": promocodes})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения списка промокодов для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def promocode_delete_handler(request: web.Request) -> web.Response:
        """Удалить промокод (только для админа)."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            data = await request.json()
            promocode_id = int(data.get("id") or 0)
            if promocode_id <= 0:
                return web.json_response({"error": "invalid_promocode_id"}, status=400)
            result = await db.delete_promocode(promocode_id)
            return web.json_response(result, status=200 if result.get("success") else 400)
        except Exception as e:
            logging.getLogger(__name__).error("promocode_delete_handler error: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)


    async def admin_cards_create_handler(request: web.Request) -> web.Response:
        """
        Обработчик создания карты (только для админа).

        Принимает POST запрос с JSON данными:
        - name: название карты (обязательно)
        - description: описание карты (опционально)
        - rarity: редкость карты (обязательно, должна быть из списка допустимых)
        - power: сила карты (обязательно, целое число >= 0)

        Возвращает JSON с результатом создания карты и card_id.
        """
        # Извлекаем user_id из параметров запроса
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        # Проверяем метод запроса
        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            # Получаем данные из запроса
            data = await request.json()
            name = data.get("name", "").strip()
            description = data.get("description", "").strip()
            rarity = data.get("rarity", "common")
            power = int(data.get("power", 0))

            # Валидация названия карты
            if not name:
                return web.json_response({"error": "name_required"}, status=400)

            # Валидация редкости карты
            valid_rarities = ["common", "rare", "superrare", "epic", "legendary", "mythic", "divine", "limited", "start"]
            if rarity not in valid_rarities:
                return web.json_response({"error": "invalid_rarity"}, status=400)

            # Валидация силы карты
            if power < 0:
                return web.json_response({"error": "invalid_power"}, status=400)

            # Создаем карту в базе данных
            # image_file_id больше не используется - изображения берутся из DesignAssets/Cards/<card_id>.png
            result = await db.create_card(
                name=name,
                description=description,
                rarity=rarity,
                power=power,
                image_file_id=None,  # Больше не используется
                created_by=user_id
            )

            # Проверяем результат создания
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка создания карты"
                }, status=400)

            # Возвращаем успешный результат с ID созданной карты
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания карты для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def admin_cards_list_handler(request: web.Request) -> web.Response:
        """Обработчик получения списка карт (только для админа)."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        try:
            cards = await db.get_cards_list()
            # Преобразуем все datetime объекты в строки для JSON
            cards = _serialize_datetime(cards)
            return web.json_response({"cards": cards})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения списка карт для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def admin_items_create_handler(request: web.Request) -> web.Response:
        """Обработчик создания предмета (только для админа)."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            name = data.get("name", "").strip()
            description = data.get("description", "").strip()
            rarity = data.get("rarity", "common")
            power = int(data.get("power", 0))
            image_file_id = data.get("image_file_id")
            image_file_id = image_file_id.strip() if image_file_id else None

            if not name:
                return web.json_response({"error": "name_required"}, status=400)

            if rarity not in ["common", "rare", "epic", "legendary", "mythic", "divine", "limited", "start"]:
                return web.json_response({"error": "invalid_rarity"}, status=400)

            result = await db.create_item(
                name=name,
                description=description,
                rarity=rarity,
                power=power,
                image_file_id=image_file_id,
                created_by=user_id
            )

            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка создания предмета"
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания предмета для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def admin_items_list_handler(request: web.Request) -> web.Response:
        """Обработчик получения списка предметов (только для админа)."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        try:
            items = await db.get_items_list()
            # Преобразуем datetime в строки для JSON
            for item in items:
                if item.get("created_at"):
                    item["created_at"] = item["created_at"].isoformat()
            return web.json_response({"items": items})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения списка предметов для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_presets_list_handler(request: web.Request) -> web.Response:
        """Обработчик получения списка пресетов колод пользователя."""
        user_id = await require_user_id(request)

        try:
            presets = await db.get_user_deck_presets(user_id)
            primary = await db.fetchval("SELECT primary_deck FROM users WHERE user_id=$1", user_id)
            for preset in presets:
                if preset.get("updated_at"):
                    preset["updated_at"] = preset["updated_at"].isoformat()
                preset["is_primary"] = (preset.get("preset_number") == primary)
            return web.json_response({"presets": presets})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения пресетов для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_preset_save_handler(request: web.Request) -> web.Response:
        """Обработчик сохранения пресета колоды (9 карт, герой внутри колоды)."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            preset_number = int(data.get("preset_number", 1))
            preset_name = data.get("preset_name", "Колода").strip()
            card_slots = data.get("card_slots", [])

            if len(card_slots) != DECK_SIZE:
                return web.json_response({"error": "invalid_slots_count"}, status=400)

            card_slots_processed = []
            for slot in card_slots:
                if slot is None or slot == "":
                    card_slots_processed.append(None)
                else:
                    try:
                        card_slots_processed.append(int(slot))
                    except (ValueError, TypeError):
                        card_slots_processed.append(None)

            result = await db.save_deck_preset(
                user_id=user_id,
                preset_number=preset_number,
                preset_name=preset_name,
                card_slots=card_slots_processed,
            )

            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка сохранения пресета"
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка сохранения пресета для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_preset_create_handler(request: web.Request) -> web.Response:
        """Обработчик создания нового пресета колоды."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            preset_name = data.get("preset_name", "Новая колода").strip()

            if not preset_name:
                preset_name = "Новая колода"

            result = await db.create_deck_preset(
                user_id=user_id,
                preset_name=preset_name
            )

            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка создания пресета"
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания пресета для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_preset_delete_handler(request: web.Request) -> web.Response:
        """Обработчик удаления пресета колоды."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            preset_number = int(data.get("preset_number"))

            result = await db.delete_deck_preset(
                user_id=user_id,
                preset_number=preset_number
            )

            if not result["success"]:
                error_messages = {
                    "min_presets_required": "Нельзя удалить пресет. Минимум 2 пресета должно остаться."
                }
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": result.get("message") or error_messages.get(result.get("error"), "Ошибка удаления пресета")
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка удаления пресета для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_preset_rename_handler(request: web.Request) -> web.Response:
        """Обработчик переименования пресета колоды."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            preset_number = int(data.get("preset_number"))
            new_name = data.get("new_name", "").strip()

            if not new_name:
                return web.json_response({"error": "empty_name"}, status=400)

            result = await db.rename_deck_preset(
                user_id=user_id,
                preset_number=preset_number,
                new_name=new_name
            )

            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка переименования пресета"
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка переименования пресета для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def deck_preset_set_primary_handler(request: web.Request) -> web.Response:
        """Установить/снять основную колоду."""
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            preset_number = data.get("preset_number")
            clear = data.get("clear", False)
            if clear:
                result = await db.set_primary_deck(user_id, None)
            else:
                try: preset_number = int(preset_number)
                except (TypeError, ValueError): return web.json_response({"error": "invalid_preset_number"}, status=400)
                result = await db.set_primary_deck(user_id, preset_number)
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("set_primary_deck error uid=%s: %s", user_id, e, exc_info=True)
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def cards_catalog_handler(_: web.Request) -> web.Response:
        """Публичный список карт с текущими статами (уровень 1)."""
        try:
            cards = await db.get_cards_list()
            cards = _serialize_datetime(cards)
            return web.json_response({"cards": cards})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения каталога карт: %s", e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def user_cards_handler(request: web.Request) -> web.Response:
        """Обработчик получения карт пользователя."""
        user_id = await require_user_id(request)

        try:
            cards = await db.get_user_cards(user_id)
            # Преобразуем все datetime объекты в строки для JSON
            cards = _serialize_datetime(cards)
            return web.json_response({"cards": cards})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения карт для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def collection_with_status_handler(request: web.Request) -> web.Response:
        """Все карты каталога с признаком locked/owned для экрана коллекции."""
        user_id = await require_user_id(request)
        try:
            cards = await db.get_collection_with_status(user_id)
            cards = _serialize_datetime(cards)
            return web.json_response({"cards": cards})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения коллекции для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def admin_get_all_cards_handler(request: web.Request) -> web.Response:
        """Обработчик получения всех карт в коллекцию админа (только для админа)."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            result = await db.add_all_cards_to_user(user_id)
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка добавления карт"
                }, status=400)
            # Преобразуем все datetime объекты в строки для JSON
            result = _serialize_datetime(result)
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения всех карт для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def admin_delete_all_cards_handler(request: web.Request) -> web.Response:
        """Обработчик удаления всех карт из коллекции админа (только для админа)."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            result = await db.delete_all_user_cards(user_id)
            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка удаления карт"
                }, status=400)
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка удаления всех карт для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def card_upgrade_handler(request: web.Request) -> web.Response:
        """Обработчик улучшения карты."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            card_id = int(data.get("card_id"))

            result = await db.upgrade_card(user_id, card_id)

            if not result["success"]:
                error_messages = {
                    "card_not_found": "Карта не найдена",
                    "insufficient_particles": f"Недостаточно частиц. Нужно: {result.get('required', 0)}, имеется: {result.get('current', 0)}",
                    "insufficient_coins": f"Недостаточно монет. Нужно: {result.get('required', 0)}, имеется: {result.get('current', 0)}",
                    "max_level_reached": "Карта уже достигла максимального уровня (10)"
                }
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": error_messages.get(result.get("error"), "Ошибка улучшения карты"),
                    "required": result.get("required"),
                    "current": result.get("current")
                }, status=400)

            # Преобразуем datetime объекты в строки
            result = _serialize_datetime(result)
            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка улучшения карты для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def card_add_particles_handler(request: web.Request) -> web.Response:
        """Обработчик добавления частиц к карте."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            card_id = int(data.get("card_id"))
            particles = int(data.get("particles", 0))

            if particles <= 0:
                return web.json_response({
                    "success": False,
                    "error": "invalid_particles",
                    "message": "Количество частиц должно быть больше 0"
                }, status=400)

            result = await db.add_particles_to_card(user_id, card_id, particles)

            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка добавления частиц"
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка добавления частиц для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def community_posts_list_handler(request: web.Request) -> web.Response:
        """Обработчик получения списка постов коммьюнити."""
        user_id = await require_user_id(request)

        try:
            limit = int(request.rel_url.query.get("limit", 50))
            posts = await db.get_community_posts(limit=limit, user_id=user_id)
            # Преобразуем datetime в строки для JSON
            for post in posts:
                if post.get("created_at"):
                    post["created_at"] = post["created_at"].isoformat()
                # Убеждаемся, что likes_count есть
                if "likes_count" not in post:
                    post["likes_count"] = 0
                if "is_liked" not in post:
                    post["is_liked"] = False

            # Получаем фото авторов через Bot API используя author_id
            # Используем семафор для ограничения параллельных запросов к Telegram API
            # Telegram имеет ограничения на количество запросов в секунду (rate limiting)
            import aiohttp
            import asyncio

            # Создаем семафор для ограничения параллельных запросов (максимум 3 одновременно)
            # Это помогает избежать rate limiting со стороны Telegram API
            photo_semaphore = asyncio.Semaphore(3)

            async def get_user_photo(user_id: int, retry_count: int = 3) -> str | None:
                """
                Получить фото пользователя из Telegram API с повторными попытками.

                Args:
                    user_id: ID пользователя в Telegram
                    retry_count: Количество попыток при ошибке

                Returns:
                    URL фото или None, если не удалось получить
                """
                for attempt in range(retry_count):
                    try:
                        # Используем семафор для ограничения параллельных запросов
                        async with photo_semaphore:
                            async with _create_ssl_disabled_session() as session:
                                # Увеличиваем таймаут до 15 секунд для медленных соединений
                                timeout = aiohttp.ClientTimeout(total=15, connect=10)

                                # Получаем список фото пользователя
                                url = f"https://api.telegram.org/bot{bot_token}/getUserProfilePhotos"
                                async with session.get(
                                    url,
                                    params={"user_id": user_id, "limit": 1},
                                    timeout=timeout
                                ) as resp:
                                    if resp.status == 200:
                                        data = await resp.json()
                                        if data.get("ok") and data.get("result", {}).get("total_count", 0) > 0:
                                            file_id = data["result"]["photos"][0][0]["file_id"]

                                            # Получаем путь к файлу
                                            file_url = f"https://api.telegram.org/bot{bot_token}/getFile"
                                            async with session.get(
                                                file_url,
                                                params={"file_id": file_id},
                                                timeout=timeout
                                            ) as file_resp:
                                                if file_resp.status == 200:
                                                    file_data = await file_resp.json()
                                                    if file_data.get("ok"):
                                                        file_path = file_data["result"]["file_path"]
                                                        photo_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
                                                        import logging
                                                        logging.getLogger(__name__).info(f"Получено фото для user_id {user_id}: {photo_url}")
                                                        return photo_url
                                    elif resp.status == 429:
                                        # Rate limit - ждем перед повторной попыткой
                                        wait_time = (attempt + 1) * 2  # Экспоненциальная задержка: 2, 4, 6 секунд
                                        import logging
                                        logging.getLogger(__name__).warning(
                                            f"Rate limit при получении фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}, ждем {wait_time}с"
                                        )
                                        await asyncio.sleep(wait_time)
                                        continue
                    except asyncio.TimeoutError:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Таймаут при получении фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}"
                        )
                        if attempt < retry_count - 1:
                            # Ждем перед повторной попыткой
                            await asyncio.sleep((attempt + 1) * 1)  # 1, 2, 3 секунды
                        continue
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Ошибка получения фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}: {e}"
                        )
                        if attempt < retry_count - 1:
                            # Ждем перед повторной попыткой
                            await asyncio.sleep((attempt + 1) * 1)  # 1, 2, 3 секунды
                        continue

                return None

            # Получаем уникальные ID авторов
            author_ids = list(set(post.get("author_id") for post in posts if post.get("author_id")))
            # Получаем фото параллельно с ограничением через семафор
            photo_tasks = [get_user_photo(aid) for aid in author_ids]
            photo_results = await asyncio.gather(*photo_tasks, return_exceptions=True)
            photo_map = {aid: result for aid, result in zip(author_ids, photo_results) if not isinstance(result, Exception) and result}

            # Присваиваем фото постам
            for post in posts:
                if post.get("author_id") and post["author_id"] in photo_map:
                    post["author_photo_url"] = photo_map[post["author_id"]]
            return web.json_response({"posts": posts})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения постов коммьюнити: %s", e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def community_post_create_handler(request: web.Request) -> web.Response:
        """Обработчик создания поста коммьюнити (только для админа)."""
        user_id = await require_user_id(request)
        if not await db.is_admin(user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            title = data.get("title", "").strip()
            content = data.get("content", "").strip()
            photo_file_id = data.get("photo_file_id")
            photo_file_id = photo_file_id.strip() if photo_file_id else None

            if not title:
                return web.json_response({"error": "title_required"}, status=400)
            if not content:
                return web.json_response({"error": "content_required"}, status=400)

            result = await db.create_community_post(
                author_id=user_id,
                title=title,
                content=content,
                photo_file_id=photo_file_id
            )

            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка создания поста"
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания поста для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def global_chat_messages_handler(request: web.Request) -> web.Response:
        """Обработчик получения сообщений глобального чата."""
        try:
            limit = int(request.rel_url.query.get("limit", 100))
            messages = await db.get_chat_messages(limit=limit)
            # Преобразуем datetime в строки для JSON
            for msg in messages:
                if msg.get("created_at"):
                    msg["created_at"] = msg["created_at"].isoformat()

            # Получаем фото пользователей через Bot API используя user_id
            # Используем семафор для ограничения параллельных запросов к Telegram API
            # Telegram имеет ограничения на количество запросов в секунду (rate limiting)
            import aiohttp
            import asyncio

            # Создаем семафор для ограничения параллельных запросов (максимум 3 одновременно)
            # Это помогает избежать rate limiting со стороны Telegram API
            photo_semaphore = asyncio.Semaphore(3)

            async def get_user_photo(user_id: int, retry_count: int = 3) -> str | None:
                """
                Получить фото пользователя из Telegram API с повторными попытками.

                Args:
                    user_id: ID пользователя в Telegram
                    retry_count: Количество попыток при ошибке

                Returns:
                    URL фото или None, если не удалось получить
                """
                for attempt in range(retry_count):
                    try:
                        # Используем семафор для ограничения параллельных запросов
                        async with photo_semaphore:
                            async with _create_ssl_disabled_session() as session:
                                # Увеличиваем таймаут до 15 секунд для медленных соединений
                                timeout = aiohttp.ClientTimeout(total=15, connect=10)

                                # Получаем список фото пользователя
                                url = f"https://api.telegram.org/bot{bot_token}/getUserProfilePhotos"
                                async with session.get(
                                    url,
                                    params={"user_id": user_id, "limit": 1},
                                    timeout=timeout
                                ) as resp:
                                    if resp.status == 200:
                                        data = await resp.json()
                                        if data.get("ok") and data.get("result", {}).get("total_count", 0) > 0:
                                            file_id = data["result"]["photos"][0][0]["file_id"]

                                            # Получаем путь к файлу
                                            file_url = f"https://api.telegram.org/bot{bot_token}/getFile"
                                            async with session.get(
                                                file_url,
                                                params={"file_id": file_id},
                                                timeout=timeout
                                            ) as file_resp:
                                                if file_resp.status == 200:
                                                    file_data = await file_resp.json()
                                                    if file_data.get("ok"):
                                                        file_path = file_data["result"]["file_path"]
                                                        photo_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
                                                        import logging
                                                        logging.getLogger(__name__).info(f"Получено фото для user_id {user_id}: {photo_url}")
                                                        return photo_url
                                    elif resp.status == 429:
                                        # Rate limit - ждем перед повторной попыткой
                                        wait_time = (attempt + 1) * 2  # Экспоненциальная задержка: 2, 4, 6 секунд
                                        import logging
                                        logging.getLogger(__name__).warning(
                                            f"Rate limit при получении фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}, ждем {wait_time}с"
                                        )
                                        await asyncio.sleep(wait_time)
                                        continue
                    except asyncio.TimeoutError:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Таймаут при получении фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}"
                        )
                        if attempt < retry_count - 1:
                            # Ждем перед повторной попыткой
                            await asyncio.sleep((attempt + 1) * 1)  # 1, 2, 3 секунды
                        continue
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Ошибка получения фото для user_id {user_id}, попытка {attempt + 1}/{retry_count}: {e}"
                        )
                        if attempt < retry_count - 1:
                            # Ждем перед повторной попыткой
                            await asyncio.sleep((attempt + 1) * 1)  # 1, 2, 3 секунды
                        continue

                return None

            # Получаем уникальные ID пользователей
            user_ids = list(set(msg.get("user_id") for msg in messages if msg.get("user_id")))
            # Получаем фото параллельно с ограничением через семафор
            photo_tasks = [get_user_photo(uid) for uid in user_ids]
            photo_results = await asyncio.gather(*photo_tasks, return_exceptions=True)
            photo_map = {uid: result for uid, result in zip(user_ids, photo_results) if not isinstance(result, Exception) and result}

            # Присваиваем фото сообщениям
            for msg in messages:
                if msg.get("user_id") and msg["user_id"] in photo_map:
                    msg["user_photo_url"] = photo_map[msg["user_id"]]
            return web.json_response({"messages": messages})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения сообщений чата: %s", e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def global_chat_send_handler(request: web.Request) -> web.Response:
        """Обработчик отправки сообщения в глобальный чат."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            message = data.get("message", "").strip()

            if not message:
                return web.json_response({"error": "message_required"}, status=400)

            # Проверяем extra_pass для определения CD
            user_profile = await db.get_user_profile(user_id)
            has_extra_pass = user_profile and user_profile.get("extra_pass") == "active"
            cooldown_seconds = 3 if has_extra_pass else 15

            # Проверяем CD на отправку сообщений
            last_message = await db.fetchrow(
                """
                SELECT created_at FROM global_chat
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                user_id
            )

            if last_message and last_message.get("created_at"):
                from datetime import datetime, timezone
                last_time = last_message["created_at"]

                # Если это datetime объект из БД
                if isinstance(last_time, datetime):
                    if last_time.tzinfo is None:
                        last_time = last_time.replace(tzinfo=timezone.utc)
                elif isinstance(last_time, str):
                    # Парсим ISO формат
                    try:
                        if 'T' in last_time:
                            last_time = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
                        else:
                            last_time = datetime.fromisoformat(last_time)
                        if last_time.tzinfo is None:
                            last_time = last_time.replace(tzinfo=timezone.utc)
                    except (ValueError, AttributeError) as e:
                        import logging
                        logging.getLogger(__name__).error(f"Ошибка парсинга времени: {e}, last_time={last_time}")
                        # Если не удалось распарсить, пропускаем проверку CD
                        last_time = None

                if last_time and isinstance(last_time, datetime):
                    now = datetime.now(timezone.utc)
                    time_diff = (now - last_time).total_seconds()

                    if time_diff < cooldown_seconds:
                        remaining = int(cooldown_seconds - time_diff)
                        return web.json_response({
                            "success": False,
                            "error": "cooldown",
                            "message": f"Подождите {remaining} секунд перед отправкой следующего сообщения",
                            "cooldown_remaining": remaining
                        }, status=429)

            result = await db.create_chat_message(
                user_id=user_id,
                message=message
            )

            if not result["success"]:
                error_messages = {
                    "empty_message": "Сообщение не может быть пустым",
                    "message_too_long": "Сообщение слишком длинное (максимум 500 символов)"
                }
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": error_messages.get(result.get("error"), "Ошибка отправки сообщения")
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка отправки сообщения для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    # ========== Хэндлеры дружеских матчей ==========

    async def recent_opponents_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        opponents = await db.get_recent_opponents(user_id)
        return web.json_response({"opponents": opponents})

    def _prune_online_users() -> None:
        now = time.time()
        online_users = app.get("online_users", {})
        stale = [
            uid for uid, seen_at in list(online_users.items())
            if now - float(seen_at or 0) > ONLINE_USER_TTL_SECONDS
        ]
        for uid in stale:
            online_users.pop(uid, None)

    def _is_user_online(target_user_id: int) -> bool:
        _prune_online_users()
        seen_at = app.get("online_users", {}).get(int(target_user_id))
        return bool(seen_at and time.time() - float(seen_at) <= ONLINE_USER_TTL_SECONDS)

    def _webapp_url_with_invite(invite_id: int) -> str:
        base_url = str(app.get("webapp_url") or "").strip() or "/"
        parts = urlsplit(base_url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["invite_id"] = str(invite_id)
        query["invite_action"] = "accept"
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    async def friend_online_ping_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)
        _prune_online_users()
        app["online_users"][int(user_id)] = time.time()
        return web.json_response({"success": True, "online_ttl_seconds": ONLINE_USER_TTL_SECONDS})

    async def _friend_invite_status_payload(invite: dict[str, Any], viewer_id: int) -> dict[str, Any]:
        status = str(invite.get("status") or "")
        expires_at = invite.get("expires_at")
        if status == "pending" and expires_at:
            expires_dt = expires_at
            if isinstance(expires_dt, str):
                try:
                    expires_dt = datetime.fromisoformat(expires_dt.replace("Z", "+00:00"))
                except ValueError:
                    expires_dt = None
            if isinstance(expires_dt, datetime):
                if expires_dt.tzinfo is None:
                    expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                if expires_dt <= datetime.now(timezone.utc):
                    await db.update_invite_status(int(invite["id"]), "expired")
                    status = "expired"

        payload = {
            "success": True,
            "invite_id": invite.get("id"),
            "status": status,
            "battle_id": invite.get("battle_id"),
        }
        if status != "accepted" or not invite.get("battle_id"):
            return payload

        from_user_id = int(invite["from_user_id"])
        to_user_id = int(invite["to_user_id"])
        opponent_id = to_user_id if int(viewer_id) == from_user_id else from_user_id
        player_profile = await _resolve_battle_profile(
            db,
            int(viewer_id),
            fallback_name="Игрок",
        )
        opponent_profile = await _resolve_battle_profile(
            db,
            opponent_id,
            fallback_name="Соперник",
        )

        payload.update({
            "match_id": invite.get("battle_id"),
            "game_mode": "friendly",
            "mode_config": serialize_mode_config(resolve_mode_config("friendly")),
            "player_name": player_profile["name"],
            "player_avatar_url": player_profile["avatar_url"],
            "player_trophies": player_profile["trophies"],
            "player_clan": player_profile["clan"],
            "player_title": player_profile["title"],
            "player_title_rarity": player_profile["title_class"],
            "player_extra_pass": player_profile["extra_pass"],
            "player_background": player_profile["background_url"],
            "opponent_name": opponent_profile["name"],
            "opponent_avatar_url": opponent_profile["avatar_url"],
            "opponent_trophies": opponent_profile["trophies"],
            "opponent_clan": opponent_profile["clan"],
            "opponent_title": opponent_profile["title"],
            "opponent_title_rarity": opponent_profile["title_class"],
            "opponent_extra_pass": opponent_profile["extra_pass"],
            "opponent_background": opponent_profile["background_url"],
            "players": {
                str(from_user_id): await _resolve_battle_profile(db, from_user_id, fallback_name="Игрок 1"),
                str(to_user_id): await _resolve_battle_profile(db, to_user_id, fallback_name="Игрок 2"),
            },
        })
        return payload

    async def battle_history_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)

        user_info = await db.get_user_info(user_id)
        extra_pass = (user_info or {}).get("extra_pass", "inactive")
        is_ultra = extra_pass == "ultra"
        limit = 50 if is_ultra else 25

        all_battles = await db.get_battle_history(user_id, 100000)
        battles = all_battles[:limit]
        ranked_battles = [b for b in all_battles if b.get("mode") not in ("friendly", "training")]
        wins = sum(1 for b in all_battles if b.get("result") == "win")
        losses = sum(1 for b in all_battles if b.get("result") == "lose")
        draws = sum(1 for b in all_battles if b.get("result") == "draw")
        decided = wins + losses
        avg_turns_values = [int(b.get("turns_count") or 0) for b in all_battles if int(b.get("turns_count") or 0) > 0]
        avg_duration_values = [int(b.get("duration_seconds") or 0) for b in all_battles if int(b.get("duration_seconds") or 0) > 0]
        mode_counts: dict[str, int] = {}
        for b in all_battles:
            mode = str(b.get("mode") or "classic")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        current_streak_result = None
        current_streak_count = 0
        for b in all_battles:
            result = b.get("result")
            if result not in ("win", "lose"):
                if current_streak_count == 0:
                    continue
                break
            if current_streak_result is None:
                current_streak_result = result
                current_streak_count = 1
            elif current_streak_result == result:
                current_streak_count += 1
            else:
                break
        stats = {
            "total": len(all_battles),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round((wins / decided) * 100, 1) if decided else 0,
            "trophy_delta": sum(int(b.get("trophies_change") or 0) for b in ranked_battles),
            "avg_turns": round(sum(avg_turns_values) / len(avg_turns_values), 1) if avg_turns_values else None,
            "avg_duration_seconds": round(sum(avg_duration_values) / len(avg_duration_values)) if avg_duration_values else None,
            "favorite_mode": max(mode_counts.items(), key=lambda item: item[1])[0] if mode_counts else None,
            "current_streak_result": current_streak_result,
            "current_streak_count": current_streak_count,
        }
        # ── DEBUG ──
        logger_dbg = logging.getLogger(__name__)
        logger_dbg.info(
            "[BATTLE_HISTORY] user_id=%s limit=%s total=%s",
            user_id, limit, len(all_battles),
        )
        for b in battles[:5]:
            logger_dbg.info(
                "[BATTLE_HISTORY]   match_id=%s created_at=%s opponent_id=%s mode=%s result=%s",
                b.get("battle_id"), b.get("created_at"), b.get("opponent_id"),
                b.get("mode"), b.get("result"),
            )
        return web.json_response({
            "battles": battles,
            "total": len(all_battles),
            "stats": stats,
            "limit": limit,
            "is_limited": not is_ultra,
        })

    async def friend_invite_handler(request: web.Request) -> web.Response:
        logger = logging.getLogger(__name__)
        user_id = await require_user_id(request)
        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)
        if not await _is_admin_user(db, user_id) and not await _is_runtime_feature_enabled(db, "friendly"):
            return web.json_response(_runtime_unavailable_payload("friendly"), status=503)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        to_user_id = data.get("to_user_id")
        try:
            to_user_id = int(to_user_id)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_to_user_id"}, status=400)

        if to_user_id == user_id:
            return web.json_response({"error": "cannot_invite_self"}, status=400)

        selected_deck_id = data.get("selected_deck_id") or data.get("deck_id")
        try:
            selected_deck_id = int(selected_deck_id) if selected_deck_id is not None else None
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_deck_id"}, status=400)
        selected_deck_id = await _resolve_match_deck_id(db, user_id, selected_deck_id)

        target_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", to_user_id)
        if not target_exists:
            return web.json_response({"error": "user_not_found"}, status=404)

        if hasattr(db, "are_friends") and not await db.are_friends(user_id, to_user_id):
            return web.json_response({"error": "not_friends"}, status=403)

        has_active = await db.has_active_pending_invite(user_id, to_user_id)
        if has_active:
            return web.json_response({"error": "invite_already_sent"}, status=409)

        blocked_card = await _find_disabled_card_in_deck(db, user_id, selected_deck_id)
        if blocked_card:
            return _disabled_card_response(blocked_card)

        result = await db.create_friend_invite(user_id, to_user_id, selected_deck_id)
        if not result.get("success"):
            return web.json_response({"error": "invite_create_failed"}, status=500)

        invite_id = result["id"]
        target_online = _is_user_online(to_user_id)

        from_profile = await db.get_user_profile(user_id)
        from_name = "Игрок"
        if from_profile:
            from_name = (
                from_profile.get("display_name")
                or from_profile.get("custom_nickname")
                or from_profile.get("nickname")
                or from_profile.get("name")
                or from_profile.get("username")
                or str(user_id)
            )

        telegram_sent = False
        if not target_online:
            try:
                bot = app.get("telegram_bot") or app.get("bot")
                if bot:
                    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
                    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                        InlineKeyboardButton(
                            text="Принять бой",
                            web_app=WebAppInfo(url=_webapp_url_with_invite(invite_id)),
                        )
                    ]])
                    await bot.send_message(
                        chat_id=to_user_id,
                        text=f"⚔️ <b>{from_name}</b> вызывает тебя на дружеский бой!",
                        parse_mode="HTML",
                        reply_markup=keyboard,
                    )
                    telegram_sent = True
            except Exception as e:
                logger.warning("Failed to send Telegram invite notification: %s", e)

        return web.json_response({
            "success": True,
            "invite_id": invite_id,
            "created_at": result.get("created_at"),
            "expires_at": result.get("expires_at"),
            "delivery": "in_app" if target_online else "telegram",
            "target_online": target_online,
            "telegram_sent": telegram_sent,
        })

    async def friend_invite_status_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        invite_id = request.rel_url.query.get("id") or request.rel_url.query.get("invite_id")
        try:
            invite_id = int(invite_id)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_invite_id"}, status=400)

        invite = await db.get_friend_invite_for_user(invite_id, user_id)
        if not invite:
            return web.json_response({"error": "invite_not_found"}, status=404)

        return web.json_response(await _friend_invite_status_payload(invite, user_id))

    async def friend_invite_respond_handler(request: web.Request) -> web.Response:
        logger = logging.getLogger(__name__)
        user_id = await require_user_id(request)
        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        invite_id = data.get("invite_id")
        action = data.get("action")
        try:
            invite_id = int(invite_id)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_invite_id"}, status=400)

        if action not in ("accept", "decline"):
            return web.json_response({"error": "invalid_action"}, status=400)

        invite = await db.get_friend_invite_by_id(invite_id)
        if not invite:
            return web.json_response({"error": "invite_not_found"}, status=404)

        if invite["to_user_id"] != user_id:
            return web.json_response({"error": "not_your_invite"}, status=403)

        expires_at = invite.get("expires_at")
        if expires_at:
            expires_dt = expires_at
            if isinstance(expires_dt, str):
                try:
                    expires_dt = datetime.fromisoformat(expires_dt.replace("Z", "+00:00"))
                except ValueError:
                    expires_dt = None
            if isinstance(expires_dt, datetime):
                if expires_dt.tzinfo is None:
                    expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                if expires_dt <= datetime.now(timezone.utc):
                    await db.update_invite_status(invite_id, "expired")
                    return web.json_response({"error": "invite_expired"}, status=409)

        if action == "accept" and invite["status"] == "accepted" and invite.get("battle_id"):
            return web.json_response(await _friend_invite_status_payload(invite, user_id))

        if invite["status"] != "pending":
            return web.json_response({"error": "invite_already_responded"}, status=409)

        if action == "decline":
            await db.update_invite_status(invite_id, "declined")
            return web.json_response({"success": True, "status": "declined"})

        async def _fail_friend_invite_accept(
            error: str,
            *,
            status: int = 500,
            message: str | None = None,
            details: Any = None,
            to_selected_deck_id: int | None = None,
        ) -> web.Response:
            try:
                await db.update_invite_status(invite_id, "failed", to_selected_deck_id=to_selected_deck_id)
            except Exception as exc:
                logger.warning(
                    "Failed to mark friend invite %s as failed after accept error %s: %s",
                    invite_id,
                    error,
                    exc,
                    exc_info=True,
                )
            payload: dict[str, Any] = {
                "success": False,
                "status": "failed",
                "error": error,
            }
            if message:
                payload["message"] = message
            if details is not None:
                payload["details"] = details
            return web.json_response(payload, status=status)

        from_user_id = int(invite["from_user_id"])
        to_selected_deck_id_raw = data.get("selected_deck_id") or data.get("deck_id")
        try:
            to_selected_deck_id = int(to_selected_deck_id_raw) if to_selected_deck_id_raw is not None else None
        except (TypeError, ValueError):
            return await _fail_friend_invite_accept(
                "invalid_deck_id",
                status=400,
                message="Не удалось принять вызов: выбери корректную колоду и попробуй заново.",
            )
        try:
            to_selected_deck_id = await _resolve_match_deck_id(db, user_id, to_selected_deck_id)
        except web.HTTPException as exc:
            error = exc.reason or "deck_unavailable"
            try:
                if exc.text:
                    parsed_error = _stdlib_json.loads(exc.text).get("error")
                    if parsed_error:
                        error = parsed_error
            except Exception:
                pass
            return await _fail_friend_invite_accept(
                str(error),
                status=getattr(exc, "status", 400) or 400,
                message="Не удалось принять вызов: проверь выбранную колоду.",
            )

        from_selected_deck_id_raw = invite.get("from_selected_deck_id")
        try:
            from_selected_deck_id = int(from_selected_deck_id_raw) if from_selected_deck_id_raw is not None else None
        except (TypeError, ValueError):
            from_selected_deck_id = None
        try:
            from_selected_deck_id = await _resolve_match_deck_id(db, from_user_id, from_selected_deck_id)
        except web.HTTPException as exc:
            error = exc.reason or "opponent_deck_unavailable"
            try:
                if exc.text:
                    parsed_error = _stdlib_json.loads(exc.text).get("error")
                    if parsed_error:
                        error = parsed_error
            except Exception:
                pass
            return await _fail_friend_invite_accept(
                str(error),
                status=getattr(exc, "status", 400) or 400,
                message="Не удалось принять вызов: у соперника не готова колода.",
                to_selected_deck_id=to_selected_deck_id,
            )

        admin_bypass = await _is_admin_user(db, user_id) or await _is_admin_user(db, from_user_id)
        if not admin_bypass and not await _is_runtime_feature_enabled(db, "friendly"):
            unavailable = _runtime_unavailable_payload("friendly")
            return await _fail_friend_invite_accept(
                "feature_unavailable",
                status=503,
                message=unavailable.get("message"),
                details={"feature": "friendly"},
                to_selected_deck_id=to_selected_deck_id,
            )
        if not admin_bypass:
            blocked_card = await _find_disabled_card_in_deck(db, user_id, to_selected_deck_id)
            if blocked_card:
                return await _fail_friend_invite_accept(
                    "card_unavailable",
                    status=200,
                    message=blocked_card.get("message"),
                    details=blocked_card,
                    to_selected_deck_id=to_selected_deck_id,
                )
            blocked_card = await _find_disabled_card_in_deck(db, from_user_id, from_selected_deck_id)
            if blocked_card:
                return await _fail_friend_invite_accept(
                    "card_unavailable",
                    status=200,
                    message=blocked_card.get("message"),
                    details=blocked_card,
                    to_selected_deck_id=to_selected_deck_id,
                )
        match_id = str(uuid.uuid4())

        try:
            (p1_raw_deck, _), (p2_raw_deck, _) = await asyncio.wait_for(
                asyncio.gather(
                    _load_player_deck_and_hero(user_id, to_selected_deck_id),
                    _load_player_deck_and_hero(from_user_id, from_selected_deck_id),
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            return await _fail_friend_invite_accept(
                "deck_load_timeout",
                status=504,
                message="Не удалось подготовить бой: колоды загружались слишком долго.",
                to_selected_deck_id=to_selected_deck_id,
            )
        except Exception as e:
            logger.error("Deck load failed: %s", e, exc_info=True)
            return await _fail_friend_invite_accept(
                "deck_load_failed",
                status=500,
                message="Не удалось подготовить бой: ошибка загрузки колод.",
                to_selected_deck_id=to_selected_deck_id,
            )

        p1_profile = await _resolve_battle_profile(db, user_id, fallback_name="Игрок 1")
        p2_profile = await _resolve_battle_profile(db, from_user_id, fallback_name="Игрок 2")
        p1_name = p1_profile["name"]
        p1_avatar = p1_profile["avatar_url"]
        p2_name = p2_profile["name"]
        p2_avatar = p2_profile["avatar_url"]

        try:
            card_cache = await asyncio.wait_for(_load_card_cache(), timeout=3.0)
        except asyncio.TimeoutError:
            card_cache = {}
        except Exception:
            card_cache = {}

        p1_deck_ids = _normalize_deck_with_cache(p1_raw_deck or [], card_cache)
        p2_deck_ids = _normalize_deck_with_cache(p2_raw_deck or [], card_cache)

        p1_deck_int = [int(str(d).split(":")[0]) for d in p1_deck_ids if d]
        p2_deck_int = [int(str(d).split(":")[0]) for d in p2_deck_ids if d]

        engine = BattleEngine(
            db=db,
            match_id=match_id,
            player_ids=[user_id, from_user_id],
            is_bot_match=False,
            card_cache=card_cache,
            active_matches=request.app["active_matches"],
            event_emitter=request.app.get("event_emitter"),
            game_mode="friendly",
        )
        logger.info("game_mode in engine (friendly): %s", engine.game_mode)

        try:
            create_result = await engine.create_match(
                match_id=match_id,
                p1_data={
                    "user_id": user_id,
                    "deck_ids": p1_deck_int,
                    "name": p1_name,
                    "avatar_url": p1_avatar,
                    "is_bot": False,
                    "trophies": p1_profile["trophies"],
                    "clan": p1_profile["clan"],
                    "title": p1_profile["title"],
                    "rarity": p1_profile["title_class"],
                    "extra_pass": p1_profile["extra_pass"],
                    "background_url": p1_profile["background_url"],
                },
                p2_data={
                    "user_id": from_user_id,
                    "deck_ids": p2_deck_int,
                    "name": p2_name,
                    "avatar_url": p2_avatar,
                    "is_bot": False,
                    "trophies": p2_profile["trophies"],
                    "clan": p2_profile["clan"],
                    "title": p2_profile["title"],
                    "rarity": p2_profile["title_class"],
                    "extra_pass": p2_profile["extra_pass"],
                    "background_url": p2_profile["background_url"],
                },
            )
        except Exception as e:
            logger.error("Friendly match create failed: %s", e, exc_info=True)
            return await _fail_friend_invite_accept(
                "match_create_failed",
                status=500,
                message="Не удалось создать бой. Отправь вызов еще раз.",
                details=str(e),
                to_selected_deck_id=to_selected_deck_id,
            )

        if not create_result.get("success"):
            return await _fail_friend_invite_accept(
                "match_create_failed",
                status=500,
                message="Не удалось создать бой. Отправь вызов еще раз.",
                details=create_result.get("error"),
                to_selected_deck_id=to_selected_deck_id,
            )

        request.app["active_matches"][match_id] = engine
        request.app["match_game_modes"][match_id] = "friendly"
        await db.update_invite_status(
            invite_id,
            "accepted",
            battle_id=match_id,
            to_selected_deck_id=to_selected_deck_id,
        )

        return web.json_response({
            "success": True,
            "status": "accepted",
            "match_id": match_id,
            "game_mode": "friendly",
            "mode_config": serialize_mode_config(resolve_mode_config("friendly")),
            "opponent_name": p2_name,
            "opponent_avatar_url": p2_avatar,
            "opponent_trophies": p2_profile["trophies"],
            "opponent_clan": p2_profile["clan"],
            "opponent_title": p2_profile["title"],
            "opponent_title_rarity": p2_profile["title_class"],
            "opponent_extra_pass": p2_profile["extra_pass"],
            "opponent_background": p2_profile["background_url"],
            "player_name": p1_name,
            "player_avatar_url": p1_avatar,
            "player_trophies": p1_profile["trophies"],
            "player_clan": p1_profile["clan"],
            "player_title": p1_profile["title"],
            "player_title_rarity": p1_profile["title_class"],
            "player_extra_pass": p1_profile["extra_pass"],
            "player_background": p1_profile["background_url"],
            "players": {
                str(from_user_id): p2_profile,
                str(user_id): p1_profile,
            },
        })

    async def friend_invite_pending_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)
        invite = await db.get_pending_invite(user_id)
        return web.json_response({"invite": invite})

    async def friend_invite_cancel_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not user_id:
            return web.json_response({"error": "authentication required"}, status=401)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        invite_id = data.get("invite_id")
        try:
            invite_id = int(invite_id)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_invite_id"}, status=400)

        invite = await db.get_friend_invite_by_id(invite_id)
        if not invite:
            return web.json_response({"error": "invite_not_found"}, status=404)

        if invite["from_user_id"] != user_id:
            return web.json_response({"error": "not_your_invite"}, status=403)

        if invite["status"] != "pending":
            return web.json_response({"error": "invite_not_pending"}, status=409)

        await db.update_invite_status(invite_id, "cancelled")
        return web.json_response({"success": True, "status": "cancelled"})

    # ========== Хэндлеры раздела "Друзья" (friend requests) ==========

    async def friend_request_send_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        addressee_id = data.get("addressee_id")
        try:
            addressee_id = int(addressee_id)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_addressee_id"}, status=400)

        if addressee_id == user_id:
            return web.json_response({"success": False, "error": "cannot_add_self"}, status=400)

        target_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", addressee_id)
        if not target_exists:
            return web.json_response({"success": False, "error": "user_not_found"}, status=404)

        has_pending = await db.has_pending_friend_request_pair(user_id, addressee_id)
        if has_pending:
            return web.json_response({"success": False, "error": "request_already_exists"}, status=409)

        result = await db.create_friend_request(user_id, addressee_id)
        if not result.get("success"):
            return web.json_response({"success": False, "error": result.get("error", "request_create_failed")}, status=500)

        return web.json_response({"success": True, "request_id": result["id"]})

    async def friend_requests_list_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)

        try:
            incoming = await db.get_incoming_friend_requests(user_id)
            outgoing = await db.get_outgoing_friend_requests(user_id)
            return web.json_response({"incoming": incoming, "outgoing": outgoing})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения заявок в друзья для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def friend_request_respond_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        request_id = data.get("request_id")
        action = data.get("action")
        try:
            request_id = int(request_id)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_request_id"}, status=400)

        if action not in ("accept", "decline"):
            return web.json_response({"error": "invalid_action"}, status=400)

        fr = await db.get_friend_request_by_id(request_id)
        if not fr:
            return web.json_response({"success": False, "error": "request_not_found"}, status=404)

        if fr["addressee_id"] != user_id:
            return web.json_response({"success": False, "error": "not_your_request"}, status=403)

        if fr["status"] != "pending":
            return web.json_response({"success": False, "error": "request_already_processed"}, status=409)

        new_status = "accepted" if action == "accept" else "declined"
        await db.update_friend_request_status(request_id, new_status)
        return web.json_response({"success": True, "status": action})

    async def friend_request_cancel_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        request_id = data.get("request_id")
        try:
            request_id = int(request_id)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_request_id"}, status=400)

        fr = await db.get_friend_request_by_id(request_id)
        if not fr:
            return web.json_response({"success": False, "error": "request_not_found"}, status=404)

        if fr["requester_id"] != user_id:
            return web.json_response({"success": False, "error": "not_your_request"}, status=403)

        if fr["status"] != "pending":
            return web.json_response({"success": False, "error": "request_not_pending"}, status=409)

        await db.update_friend_request_status(request_id, "cancelled")
        return web.json_response({"success": True})

    async def friend_list_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)

        try:
            friends = await db.get_friend_list(user_id)
            return web.json_response({"friends": friends})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения списка друзей для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def friend_remove_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        friend_id = data.get("friend_id")
        try:
            friend_id = int(friend_id)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_friend_id"}, status=400)

        if friend_id == user_id:
            return web.json_response({"success": False, "error": "cannot_remove_self"}, status=400)

        removed = await db.remove_friendship(user_id, friend_id)
        if not removed:
            return web.json_response({"success": False, "error": "friendship_not_found"}, status=404)

        return web.json_response({"success": True})

    # ========== Хэндлеры для работы с кейсами ==========

    async def user_cases_handler(request: web.Request) -> web.Response:
        """Получить список неоткрытых кейсов пользователя."""
        user_id = await require_user_id(request)

        try:
            # Система кейсов удалена, возвращаем пустой список
            # await db.sync_user_key_cases(user_id)
            # cases = await db.get_user_cases(user_id)
            return web.json_response({"success": True, "cases": []})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения кейсов для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def user_case_detail_handler(request: web.Request) -> web.Response:
        """Получить информацию о конкретном кейсе пользователя."""
        user_id = await require_user_id(request)

        user_case_id = request.match_info.get("user_case_id")
        if not user_case_id:
            return web.json_response({"error": "user_case_id_required"}, status=400)

        try:
            await db.sync_user_key_cases(user_id)
            user_case_id = int(user_case_id)
            user_case = await db.get_user_case(user_case_id, user_id)
            if not user_case:
                return web.json_response({"error": "case_not_found"}, status=404)
            return web.json_response({"success": True, "case": user_case})
        except ValueError:
            return web.json_response({"error": "invalid_user_case_id"}, status=400)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения кейса %s для user_id %s: %s", user_case_id, user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def case_tap_handler(request: web.Request) -> web.Response:
        """Обработать тап по кейсу (один из 4 тапов с проверкой апгрейда тира)."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            user_case_id = int(data.get("user_case_id"))
            current_tier = int(data.get("current_tier", 1))
            tap_number = int(data.get("tap_number"))  # 1-4

            if not user_case_id or not (1 <= tap_number <= 4):
                return web.json_response({"error": "invalid_parameters"}, status=400)

            # Проверяем, что кейс принадлежит пользователю
            await db.sync_user_key_cases(user_id)
            user_case = await db.get_user_case(user_case_id, user_id)
            if not user_case:
                return web.json_response({"error": "case_not_found"}, status=404)

            # Используем актуальный тир из БД, а не переданный параметр
            actual_tier = user_case["tier"]

            extra_pass = await get_user_case_pass_status(db, user_id)

            # Проверяем апгрейд тира
            new_tier = roll_tier_upgrade(actual_tier, tap_number, extra_pass)
            upgraded = new_tier > actual_tier

            # Обновляем тир в БД, если произошел апгрейд
            if upgraded:
                await db.update_case_tier(user_case_id, user_id, new_tier)
                actual_tier = new_tier

            return web.json_response({
                "success": True,
                "upgraded": upgraded,
                "old_tier": user_case["tier"],
                "new_tier": actual_tier,
                "tap_number": tap_number,
                "extra_pass_bonus": {"tier": "ultra"} if extra_pass == "ultra" else None,
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка обработки тапа для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def case_open_handler(request: web.Request) -> web.Response:
        """Открыть кейс и получить награды."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            user_case_id = int(data.get("user_case_id"))
            tap_results = data.get("tap_results", [])  # Список из 4 тиров после каждого тапа

            if not user_case_id or len(tap_results) != 4:
                return web.json_response({"error": "invalid_parameters"}, status=400)

            # Обрабатываем открытие кейса
            await db.sync_user_key_cases(user_id)
            result = await process_case_opening(db, user_id, user_case_id, tap_results)

            if not result.get("success"):
                return web.json_response(result, status=400)

            # Economy tracking for case open rewards
            _track_case_rewards(db, user_id, result.get("rewards") or {}, result)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка открытия кейса для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def case_skip_handler(request: web.Request) -> web.Response:
        """Пропустить анимацию и сразу открыть кейс."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            user_case_id = int(data.get("user_case_id"))

            if not user_case_id:
                return web.json_response({"error": "invalid_parameters"}, status=400)

            # Получаем текущий тир кейса
            await db.sync_user_key_cases(user_id)
            user_case = await db.get_user_case(user_case_id, user_id)
            if not user_case:
                return web.json_response({"error": "case_not_found"}, status=404)

            # Симулируем все 4 тапа с апгрейдом тира
            extra_pass = await get_user_case_pass_status(db, user_id)
            current_tier = user_case["tier"]
            tap_results = []
            for tap_num in range(1, 5):  # Тапы 1-4
                new_tier = roll_tier_upgrade(current_tier, tap_num, extra_pass)
                if new_tier > current_tier:
                    current_tier = new_tier
                    # Обновляем тир в БД после каждого апгрейда
                    await db.update_case_tier(user_case_id, user_id, new_tier)
                tap_results.append(current_tier)

            # Открываем кейс
            result = await process_case_opening(db, user_id, user_case_id, tap_results)

            if not result.get("success"):
                return web.json_response(result, status=400)

            _track_case_rewards(db, user_id, result.get("rewards") or {}, result)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка пропуска анимации для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response({"error": "internal_server_error"}, status=500)

    def _cleanup_expired_case_key_rolls() -> None:
        now = datetime.now(timezone.utc)
        expired_tokens = [
            token
            for token, roll in CASE_KEY_ROLLS.items()
            if roll.get("expires_at") and roll["expires_at"] < now
        ]
        for token in expired_tokens:
            CASE_KEY_ROLLS.pop(token, None)

    async def case_roll_from_keys_handler(request: web.Request) -> web.Response:
        """Сгенерировать серверную последовательность тапов для открытия кейса из ключа."""
        user_id = await require_user_id(request)
        try:
            _cleanup_expired_case_key_rolls()
            keys = await db.fetchval("SELECT COALESCE(keys,0) FROM users WHERE user_id=$1", user_id)
            if not keys or keys < 1:
                return web.json_response({"success": False, "error": "no_keys", "message": "Нет ключей"}, status=400)

            extra_pass = await get_user_case_pass_status(db, user_id)
            tap_results = simulate_case_tap_results(1, extra_pass)
            final_tier = tap_results[-1] if tap_results else 1
            roll_token = uuid.uuid4().hex
            CASE_KEY_ROLLS[roll_token] = {
                "user_id": user_id,
                "tap_results": tap_results,
                "final_tier": final_tier,
                "extra_pass": extra_pass,
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=CASE_KEY_ROLL_TTL_SECONDS),
            }
            return web.json_response({
                "success": True,
                "roll_token": roll_token,
                "tap_results": tap_results,
                "final_tier": final_tier,
                "extra_pass_bonus": {"tier": "ultra"} if extra_pass == "ultra" else None,
            })
        except Exception as e:
            logging.getLogger(__name__).error("case_roll_from_keys error uid=%s: %s", user_id, e, exc_info=True)
            return web.json_response({"success": False, "error": "internal_server_error"}, status=500)

    async def case_open_from_keys_handler(request: web.Request) -> web.Response:
        """Открыть кейс напрямую из ключей (keys), без user_case_id."""
        user_id = await require_user_id(request)
        try:
            data = {}
            if request.can_read_body:
                try:
                    data = await request.json()
                except Exception:
                    data = {}

            keys = await db.fetchval("SELECT COALESCE(keys,0) FROM users WHERE user_id=$1", user_id)
            if not keys or keys < 1:
                return web.json_response({"success": False, "error": "no_keys", "message": "Нет ключей"}, status=400)

            _cleanup_expired_case_key_rolls()
            roll_token = str(data.get("roll_token") or "").strip()
            stored_roll = CASE_KEY_ROLLS.get(roll_token) if roll_token else None
            if stored_roll and stored_roll.get("user_id") == user_id:
                tap_results = list(stored_roll.get("tap_results") or [])
                final_tier = int(stored_roll.get("final_tier") or (tap_results[-1] if tap_results else 1))
                extra_pass = str(stored_roll.get("extra_pass") or "inactive")
                CASE_KEY_ROLLS.pop(roll_token, None)
            else:
                extra_pass = await get_user_case_pass_status(db, user_id)
                tap_results = simulate_case_tap_results(1, extra_pass)
                final_tier = tap_results[-1] if tap_results else 1

            final_tier = max(1, min(final_tier, 5))

            # Получаем карты пользователя для проверки дубликатов
            user_cards = await db.get_user_cards(user_id)
            user_card_ids = {card["id"] for card in (user_cards or [])}

            # Генерируем награды
            rewards = await generate_case_rewards(db, final_tier, user_id, user_card_ids, extra_pass)

            key_row = await db.fetchrow(
                """
                UPDATE users
                SET keys = GREATEST(0, COALESCE(keys, 0) - 1)
                WHERE user_id = $1 AND COALESCE(keys, 0) > 0
                RETURNING keys
                """,
                user_id,
            )
            if not key_row:
                return web.json_response({"success": False, "error": "no_keys", "message": "Нет ключей"}, status=400)
            new_keys = key_row["keys"]

            # Выдаём
            if rewards["coins"] > 0:
                await db.update_user_coins(user_id, rewards["coins"])
            for c in rewards.get("cards", []):
                await db.add_card_to_user(user_id, c["card_id"])
            for p2 in rewards.get("particles", []):
                await db.add_particles_to_card(user_id, p2["card_id"], p2["particles"])
            if rewards.get("gems", 0) > 0:
                await db.add_gems(user_id, rewards["gems"])

            await _track_economy_safe(db, user_id=user_id, event_type="spend",
                resource="keys", amount=1, source="case_open",
                metadata={"final_tier": final_tier})

            _track_case_rewards(db, user_id, rewards, {"final_tier": final_tier, "tap_results": tap_results})

            try:
                event_source = f"case_open_key:{user_id}:{uuid.uuid4().hex}"
                squad_award = await db.award_squad_cbrp(
                    user_id,
                    "case_open",
                    source_id=event_source,
                    metadata={"tier": final_tier, "source": "keys"},
                )
                for card_reward in rewards.get("cards", []):
                    card_id = card_reward.get("card_id")
                    rarity = str(card_reward.get("rarity") or "").lower()
                    await db.award_squad_cbrp(
                        user_id,
                        "new_card",
                        source_id=f"{event_source}:new_card:{card_id}",
                        metadata={
                            "tier": final_tier,
                            "card_id": card_id,
                            "rarity": rarity,
                            "source": "keys",
                        },
                    )
                    if rarity in ("epic", "legendary"):
                        await db.award_squad_cbrp(
                            user_id,
                            "new_epic_plus_card_bonus",
                            source_id=f"{event_source}:epic_plus:{card_id}",
                            metadata={
                                "tier": final_tier,
                                "card_id": card_id,
                                "rarity": rarity,
                                "source": "keys",
                            },
                        )
            except Exception:
                logging.getLogger(__name__).warning(
                    "Failed to award squad CBRP for key case opening: user_id=%s",
                    user_id,
                    exc_info=True,
                )
                squad_award = {"awarded": False, "reason": "award_failed"}

            return web.json_response({
                "success": True,
                "final_tier": final_tier,
                "tap_results": tap_results,
                "rewards": rewards,
                "extra_pass_bonus": rewards.get("extra_pass_bonus"),
                "remaining_keys": new_keys,
                "squad_cbrp": squad_award,
            })
        except Exception as e:
            logging.getLogger(__name__).error("case_open_from_keys error uid=%s: %s", user_id, e, exc_info=True)
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def debug_add_key_handler(request: web.Request) -> web.Response:
        """Admin-only: добавить +1 ключ пользователю."""
        user_id = await require_user_id(request)
        try:
            await db.execute("UPDATE users SET keys = COALESCE(keys,0)+1 WHERE user_id=$1", user_id)
            new_keys = await db.fetchval("SELECT COALESCE(keys,0) FROM users WHERE user_id=$1", user_id)
            return web.json_response({"success": True, "keys": new_keys})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _get_enabled_extra_arena_mode_ids(db_instance: Database) -> list[str]:
        try:
            overrides = await db_instance.get_match_mode_overrides()
        except Exception as exc:
            logging.getLogger(__name__).warning("get_match_mode_overrides failed: %s", exc)
            overrides = []
        enabled_map: dict[str, bool] = {o["mode_id"]: o["enabled"] for o in overrides}
        return [
            mode_id for mode_id in EXTRA_ARENA_ROTATING_IDS
            if enabled_map.get(mode_id, True) and resolve_mode_config(mode_id).available
        ]

    async def _resolve_db_aware_mode(
        db_instance: Database,
        raw_game_mode: str,
        user_id: int | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        is_admin = await _is_admin_user(db_instance, user_id)
        enabled_extra_arena = list(EXTRA_ARENA_ROTATING_IDS) if is_admin else await _get_enabled_extra_arena_mode_ids(db_instance)
        canonical_mode = resolve_canonical_mode_id(raw_game_mode, enabled_mode_ids=enabled_extra_arena)
        if canonical_mode == "extra_arena":
            return None, {
                "error": "mode_unavailable",
                "mode": "extra_arena",
                "message": "ExtraArena сейчас недоступна: нет включенных модификаторов.",
            }

        mode_config = resolve_mode_config(canonical_mode)
        if not mode_config.available:
            return None, mode_unavailable_payload(canonical_mode)
        if not is_admin:
            feature = _mode_feature_key(mode_config.mode_id)
            if not await _is_runtime_feature_enabled(db_instance, feature):
                return None, _runtime_unavailable_payload(feature)
            try:
                db_enabled = await db_instance.is_match_mode_enabled(mode_config.mode_id)
            except Exception:
                db_enabled = True
            if not db_enabled:
                return None, _runtime_unavailable_payload(feature)
        return mode_config.mode_id, None

    async def _card_name_by_id(db_instance: Database, card_id: int) -> str:
        try:
            cards = await db_instance.get_cards_list()
            for item in cards:
                cid = getattr(item, "id", None) if not isinstance(item, dict) else item.get("id")
                if int(cid or 0) == int(card_id):
                    return (
                        getattr(item, "name", None)
                        if not isinstance(item, dict)
                        else item.get("name")
                    ) or str(card_id)
        except Exception:
            pass
        return str(card_id)

    async def _find_disabled_card_in_deck(
        db_instance: Database,
        user_id: int,
        selected_deck_id: int | None = None,
    ) -> dict[str, Any] | None:
        disabled_ids = set(await db_instance.get_disabled_card_ids())
        if not disabled_ids:
            return None
        try:
            presets = await db_instance.get_user_deck_presets(user_id)
        except Exception:
            presets = []
        if not presets:
            return None

        preset = None
        if selected_deck_id is not None:
            preset = next((p for p in presets if int(p.get("preset_number") or 0) == int(selected_deck_id)), None)
        if not preset:
            try:
                primary = await db_instance.fetchval("SELECT primary_deck FROM users WHERE user_id=$1", user_id)
            except Exception:
                primary = None
            if primary is not None:
                preset = next((p for p in presets if int(p.get("preset_number") or 0) == int(primary)), None)
        if not preset:
            preset = next((p for p in presets if p.get("card_ids")), presets[0])

        for raw_card_id in preset.get("card_ids") or []:
            try:
                card_id = int(raw_card_id)
            except (TypeError, ValueError):
                continue
            if card_id in disabled_ids:
                card_name = await _card_name_by_id(db_instance, card_id)
                return {
                    "card_id": card_id,
                    "card_name": card_name,
                    "message": f"Карта {card_name} сейчас недоступна: выбери любую другую!",
                }
        return None

    def _disabled_card_response(blocked: dict[str, Any]) -> web.Response:
        return web.json_response({
            "status": "canceled",
            "error": "card_unavailable",
            **blocked,
        }, status=200)

    async def _prepare_and_cache_engine(
        request: web.Request,
        match_id: str,
        player_ids: list[int | str],
        is_bot: bool,
        bot_info: dict[str, Any] | None = None,
        player_decks: dict[str, int | None] | None = None,
        starting_player_id: int | str | None = None,
        game_mode: str = "classic",
    ) -> bool:
        """
        Собирает и кеширует BattleEngine для match_id.
        Возвращает True, если движок создан и помещен в ACTIVE_MATCHES.
        """
        logger = logging.getLogger(__name__)
        db_instance: Database = request.app["db"]
        mode_config = resolve_mode_config(game_mode)
        if not mode_config.available:
            logger.warning("Battle init rejected unavailable mode=%s", mode_config.mode_id)
            return False
        game_mode = mode_config.mode_id

        # ------------------------------------------------------------------
        # Нормализация идентификаторов игроков (совместимость int <-> str)
        # ------------------------------------------------------------------
        # В разных местах пайплайна (JS → API → матчмейкер → движок) user_id может
        # оказаться строкой ("6803...") или числом (6803...). Если не нормализовать,
        # возникают тонкие баги:
        # - кеш колод в Database пишется под ключом "6803...", а движок читает по ключу 6803
        # - сравнения current_player_id/user_id дают ложные запреты хода
        #
        # Поэтому здесь централизованно приводим player_ids к «стабильному» виду:
        # - числовые строки -> int
        # - нечисловые значения -> str (или 0 как фолбэк)
        def _normalize_player_id(raw: Any) -> int | str:
            if raw is None:
                return 0
            if isinstance(raw, int):
                return raw
            try:
                return int(raw)  # "6803..." -> 6803...
            except Exception:
                return str(raw)

        normalized_player_ids: list[int | str] = []
        for raw in (player_ids or []):
            normalized_player_ids.append(_normalize_player_id(raw))
        # Гарантируем минимум двух участников, чтобы не падать на IndexError.
        while len(normalized_player_ids) < 2:
            normalized_player_ids.append(0)

        p1_id = normalized_player_ids[0]
        p2_id = normalized_player_ids[1]

        def _as_int(raw: int | str) -> int:
            try:
                return int(raw)
            except Exception:
                return 0

        p1_id_int = _as_int(p1_id)
        p2_id_int = _as_int(p2_id)
        starting_player_id_int = _as_int(starting_player_id) if starting_player_id is not None else p1_id_int
        if starting_player_id_int not in (p1_id_int, p2_id_int):
            starting_player_id_int = p1_id_int

        admin_bypass = await _is_admin_user(db_instance, p1_id_int) or await _is_admin_user(db_instance, p2_id_int)
        if not admin_bypass:
            feature = _mode_feature_key(game_mode)
            try:
                db_enabled = await db_instance.is_match_mode_enabled(game_mode)
            except Exception:
                db_enabled = True
            if not db_enabled or not await _is_runtime_feature_enabled(db_instance, feature):
                logger.warning("Battle init rejected runtime-disabled mode=%s feature=%s", game_mode, feature)
                return False

        # Загружаем профили игроков для получения имен и аватаров
        p1_name, p1_avatar_url = "Игрок 1", None
        p2_name, p2_avatar_url = ("Бот" if is_bot else "Игрок 2"), None
        p1_trophies, p2_trophies = 0, 0
        p1_clan, p2_clan = "", ""
        p1_title, p1_title_class = "", ""
        p2_title, p2_title_class = "", ""
        p1_background_url: Optional[str] = None
        p1_extra_pass: Optional[str] = None
        p2_background_url: Optional[str] = None
        p2_extra_pass: Optional[str] = None

        try:
            # Загружаем профиль первого игрока
            if p1_id_int > 0:
                p1_profile = await db_instance.get_user_profile(p1_id_int)
                logger.debug("Battle init profile p1 id=%s profile=%s", p1_id_int, p1_profile)
                if p1_profile:
                    # Приоритет: display_name > nickname > name > username > fallback
                    p1_name = (
                        p1_profile.get("display_name") or
                        p1_profile.get("nickname") or
                        p1_profile.get("name") or
                        p1_profile.get("username") or
                        f"Игрок {p1_id_int}"
                    )
                    # Для аватара: ПРИОРИТЕТ — equipped cosmetic → Telegram img → fallback
                    p1_avatar_url = (
                        p1_profile.get("equipped_avatar_url") or
                        p1_profile.get("img") or
                        p1_profile.get("photo_url") or
                        p1_profile.get("avatar_file_id") or
                        p1_profile.get("avatar_url")
                    )
                    p1_trophies = int(p1_profile.get("trophies", 0) or 0)
                    p1_clan = p1_profile.get("clan") or p1_profile.get("clan_name") or ""

                    # Титул: equipped cosmetic → legacy profiles.title → дефолт
                    p1_title = ""
                    p1_title_class = ""
                    try:
                        eq_title = await db_instance.get_equipped_title(p1_id_int)
                        if eq_title:
                            p1_title = eq_title.get("name", "")
                            p1_title_class = eq_title.get("class", "starter")
                    except Exception:
                        pass
                    if not p1_title:
                        legacy_title = p1_profile.get("title", "").strip()
                        if legacy_title and legacy_title != "Игрок" and legacy_title != "Новичок":
                            p1_title = legacy_title
                            p1_title_class = "starter"
                    logger.debug("Battle init p1 title=%s class=%s", p1_title, p1_title_class)
                    logger.debug("Battle init p1 name=%s avatar=%s trophies=%s clan=%s", p1_name, p1_avatar_url, p1_trophies, p1_clan)

            # Загружаем профиль второго игрока
            if is_bot:
                logger.debug("Battle init resolving bot profile id=%s", p2_id_int)

                # Для бота ПРИОРИТЕТ: bot_info -> профиль из БД -> дефолт "Бот"
                if bot_info:
                    # Сначала проверяем bot_info (данные из bot_factory)
                    p2_name = bot_info.get("name") or p2_name
                    bot_cosmetics = bot_info.get("cosmetics", {}) or {}
                    bot_avatar_cosmetic = bot_cosmetics.get("avatar", {}) or {}
                    p2_avatar_url = bot_avatar_cosmetic.get("asset_path") or bot_info.get("avatar_url") or p2_avatar_url
                    logger.info("Battle init: bot name from bot_info: %s, avatar: %s", p2_name, p2_avatar_url)
                else:
                    logger.debug("Battle init bot_info is empty")

                # ПРИНУДИТЕЛЬНО загружаем профиль из БД, если имя еще не определено или это дефолт
                if p2_id_int > 0 and (p2_name == "Бот" or not p2_name or not bot_info or not bot_info.get("name")):
                    logger.debug("Battle init loading fallback DB profile for bot id=%s", p2_id_int)
                    p2_profile = await db_instance.get_user_profile(p2_id_int)
                    logger.debug("Battle init profile p2 bot id=%s profile=%s", p2_id_int, p2_profile)
                    if p2_profile:
                        # Берем имя из БД
                        db_name = (
                            p2_profile.get("display_name") or
                            p2_profile.get("nickname") or
                            p2_profile.get("name") or
                            p2_profile.get("username")
                        )
                        if db_name:
                            p2_name = db_name
                            logger.debug("Battle init bot name loaded from DB: %s", p2_name)

                        # Берем аватар из БД только если bot_info не предоставил аватар
                        if not p2_avatar_url:
                            p2_avatar_url = (
                                p2_profile.get("equipped_avatar_url") or
                                p2_profile.get("img") or
                                p2_profile.get("photo_url") or
                                p2_profile.get("avatar_file_id") or
                                p2_profile.get("avatar_url")
                            )
                            if p2_avatar_url:
                                logger.debug("Battle init bot avatar loaded from DB: %s", p2_avatar_url)
                        p2_trophies = int(p2_profile.get("trophies", 0) or 0) if p2_trophies == 0 else p2_trophies
                        p2_clan = p2_profile.get("clan") or p2_profile.get("clan_name") or ""
                    else:
                        logger.warning("Battle init: no DB profile found for bot id=%s", p2_id_int)

                logger.debug("Battle init final bot name=%s avatar=%s", p2_name, p2_avatar_url)
                if bot_info:
                    p2_trophies = int(bot_info.get("trophies", 0) or 0) if p2_trophies == 0 else p2_trophies

                # Fallback: если трофеи всё ещё 0 — загрузить из БД
                if p2_id_int > 0 and p2_trophies == 0:
                    try:
                        p2_db = await db_instance.get_user_profile(p2_id_int)
                        if p2_db:
                            p2_trophies = int(p2_db.get("trophies", 0) or 0)
                            if p2_trophies:
                                logger.debug("Battle init bot trophies loaded from DB: %s", p2_trophies)
                    except Exception:
                        pass
            else:
                # Для обычного игрока (не бота) загружаем профиль из БД
                if p2_id_int > 0:
                    p2_profile = await db_instance.get_user_profile(p2_id_int)
                    logger.debug("Battle init profile p2 id=%s profile=%s", p2_id_int, p2_profile)
                    if p2_profile:
                        p2_name = (
                            p2_profile.get("display_name") or
                            p2_profile.get("nickname") or
                            p2_profile.get("name") or
                            p2_profile.get("username") or
                            f"Игрок {p2_id_int}"
                        )
                        p2_avatar_url = (
                            p2_profile.get("equipped_avatar_url") or
                            p2_profile.get("img") or
                            p2_profile.get("photo_url") or
                            p2_profile.get("avatar_file_id") or
                            p2_profile.get("avatar_url")
                        )
                        p2_trophies = int(p2_profile.get("trophies", 0) or 0) if p2_trophies == 0 else p2_trophies
                        p2_clan = p2_profile.get("clan") or p2_profile.get("clan_name") or ""
                        logger.debug("Battle init p2 name=%s avatar=%s", p2_name, p2_avatar_url)

            logger.info("Battle init: loaded player names p1=%s, p2=%s", p1_name, p2_name)

            # Загружаем титул P2 (общий путь для бота и не-бота)
            if p2_id_int > 0:
                try:
                    eq_title_p2 = await db_instance.get_equipped_title(p2_id_int)
                    if eq_title_p2:
                        p2_title = eq_title_p2.get("name", "")
                        p2_title_class = eq_title_p2.get("class", "starter")
                except Exception:
                    pass
                if not p2_title and not is_bot:
                    p2_profile_title = await db_instance.get_user_profile(p2_id_int)
                    if p2_profile_title:
                        legacy_title = p2_profile_title.get("title", "").strip()
                        if legacy_title and legacy_title != "Игрок" and legacy_title != "Новичок":
                            p2_title = legacy_title
                            p2_title_class = "starter"
                elif not p2_title and is_bot:
                    p2_title = "Бот-чемпион"
                    p2_title_class = "starter"
                logger.debug("Battle init p2 title=%s class=%s", p2_title, p2_title_class)

            # Загружаем косметику бота (фон, extra_pass)
            if is_bot and p2_id_int > 0:
                # Primary source: bot_info from matchmaker
                if bot_info:
                    p2_extra_pass = p2_extra_pass or bot_info.get("extra_pass")
                    cosmetics = bot_info.get("cosmetics", {})
                    bg = cosmetics.get("profile_background", {})
                    if bg and bg.get("asset_path"):
                        p2_background_url = p2_background_url or bg["asset_path"]
                # Fallback: query DB for already-persisted cosmetics
                try:
                    bot_cosmetics = await db_instance.get_user_cosmetics(p2_id_int)
                    eq = bot_cosmetics.get("equipped", {})
                    if eq.get("profile_background") and not p2_background_url:
                        p2_background_url = eq["profile_background"].get("asset_path")
                except Exception:
                    pass
                try:
                    if not p2_extra_pass:
                        p2_extra_pass = await db_instance.fetchval(
                            "SELECT extra_pass FROM users WHERE user_id = $1", p2_id_int
                        )
                except Exception:
                    pass

            if is_bot and game_mode == "training":
                training_profile = _training_bot_profile_payload()
                p2_name = training_profile["name"]
                p2_avatar_url = training_profile["avatar_url"]
                p2_trophies = training_profile["trophies"]
                p2_clan = training_profile["clan"]
                p2_title = training_profile["title"]
                p2_title_class = training_profile["title_class"]
                p2_background_url = training_profile["background_url"]
                p2_extra_pass = training_profile["extra_pass"]
                bot_info = _decorate_training_bot_info(bot_info)

            if p1_id_int > 0:
                p1_battle_profile = await _resolve_battle_profile(
                    db_instance,
                    p1_id_int,
                    fallback_name=f"Игрок {p1_id_int}",
                )
                p1_name = p1_battle_profile["name"]
                p1_avatar_url = p1_battle_profile["avatar_url"]
                p1_trophies = p1_battle_profile["trophies"]
                p1_clan = p1_battle_profile["clan"]
                p1_title = p1_battle_profile["title"]
                p1_title_class = p1_battle_profile["title_class"]
                p1_background_url = p1_battle_profile["background_url"]
                p1_extra_pass = p1_battle_profile["extra_pass"]

            if not is_bot and p2_id_int > 0:
                p2_battle_profile = await _resolve_battle_profile(
                    db_instance,
                    p2_id_int,
                    fallback_name=f"Игрок {p2_id_int}",
                )
                p2_name = p2_battle_profile["name"]
                p2_avatar_url = p2_battle_profile["avatar_url"]
                p2_trophies = p2_battle_profile["trophies"]
                p2_clan = p2_battle_profile["clan"]
                p2_title = p2_battle_profile["title"]
                p2_title_class = p2_battle_profile["title_class"]
                p2_background_url = p2_battle_profile["background_url"]
                p2_extra_pass = p2_battle_profile["extra_pass"]
        except Exception as profile_exc:
            logger.warning("Battle init: failed to load player profiles: %s", profile_exc)

        try:
            p1_deck_id = player_decks.get(str(p1_id_int)) if player_decks else None
            p2_deck_id = player_decks.get(str(p2_id_int)) if player_decks else None

            if is_bot and bot_info and bot_info.get("deck_ids"):
                (p1_raw_deck, p1_hero_hp) = await asyncio.wait_for(
                    _load_player_deck_and_hero(p1_id_int, p1_deck_id),
                    timeout=3.0,
                )
                p2_raw_deck = [str(card_id) for card_id in (bot_info.get("deck_ids") or [])]
                p2_hero_hp = None
            else:
                (p1_raw_deck, p1_hero_hp), (p2_raw_deck, p2_hero_hp) = await asyncio.wait_for(
                    asyncio.gather(
                        _load_player_deck_and_hero(p1_id_int, p1_deck_id),
                        _load_player_deck_and_hero(p2_id_int, p2_deck_id),
                    ),
                    timeout=3.0,
                )
            logger.info(
                "Battle init: decks loaded (p1=%s cards, p2=%s cards)",
                len(p1_raw_deck or []),
                len(p2_raw_deck or []),
            )
            if not admin_bypass:
                disabled_ids = set(await db_instance.get_disabled_card_ids())
                if disabled_ids:
                    def _raw_deck_has_disabled(raw_deck: list[Any]) -> bool:
                        for raw_card in raw_deck or []:
                            try:
                                card_id = int(str(raw_card).split(":", 1)[0])
                            except (TypeError, ValueError):
                                continue
                            if card_id in disabled_ids:
                                return True
                        return False
                    if _raw_deck_has_disabled(p1_raw_deck) or (not is_bot and _raw_deck_has_disabled(p2_raw_deck)):
                        logger.warning("Battle init rejected deck with disabled card match_id=%s", match_id)
                        return False
        except asyncio.TimeoutError:
            logger.warning(
                "Battle init: deck load timed out, falling back to empty seeds (p1=%s, p2=%s)",
                p1_id,
                p2_id,
            )
            p1_raw_deck, p2_raw_deck, p1_hero_hp, p2_hero_hp = [], [], None, None
        except Exception as deck_exc:  # noqa: BLE001 - не блокируем бой при сбоях загрузки
            logger.warning(
                "Battle init: deck load failed (%s / %s): %s",
                p1_id,
                p2_id,
                deck_exc,
                exc_info=True,
            )
            p1_raw_deck, p2_raw_deck, p1_hero_hp, p2_hero_hp = [], [], None, None

        try:
            card_cache = await asyncio.wait_for(_load_card_cache(), timeout=3.0)
            logger.info("Battle init: card cache loaded, items=%s", len(card_cache))
        except asyncio.TimeoutError:
            logger.warning("Battle init: card cache load timed out, using empty cache")
            card_cache = {}
        except Exception as cache_exc:  # noqa: BLE001
            logger.warning("Battle init: card cache load failed: %s", cache_exc, exc_info=True)
            card_cache = {}

        p1_deck_ids = _normalize_deck_with_cache(p1_raw_deck, card_cache)
        p2_deck_ids = _normalize_deck_with_cache(p2_raw_deck, card_cache)

        try:
            deck_cache = getattr(db_instance, "deck_presets_cache", None)
            if not isinstance(deck_cache, dict):
                deck_cache = {}
            # Важно: кладем под int-ключи, потому что BattleEngine читает кеш по int(user_id).
            deck_cache[p1_id_int] = {"cards": list(p1_deck_ids)}
            deck_cache[p2_id_int] = {"cards": list(p2_deck_ids)}
            db_instance.deck_presets_cache = deck_cache
        except Exception as cache_exc:  # noqa: BLE001 - кеш не критичен для старта боя
            logger.debug("Не удалось записать deck_presets_cache: %s", cache_exc)

        try:
            engine = BattleEngine(
                db=db_instance,
                match_id=match_id,
                player_ids=normalized_player_ids,
                is_bot_match=is_bot,
                card_cache=card_cache,
                active_matches=request.app["active_matches"],
                event_emitter=request.app.get("event_emitter"),
                game_mode=game_mode,
            )
            logger.info("game_mode in engine: %s", engine.game_mode)

            # Вызываем create_match для инициализации core/engine.ArenaEnvironment
            # Конвертируем deck_ids в int для совместимости с converter.deck_from_card_ids
            p1_deck_int_ids = [int(str(d).split(":")[0]) for d in p1_deck_ids if d]
            p2_deck_int_ids = [int(str(d).split(":")[0]) for d in p2_deck_ids if d]

            # КРИТИЧНО: Извлекаем difficulty и card_levels из bot_info для передачи в движок
            bot_difficulty = "lite"
            bot_card_levels = None
            bot_difficulty_label = "lite"
            bot_strength_tier = "lite"
            bot_brain_profile = None
            bot_selection = None
            bot_temperature = None
            bot_card_level_policy = None
            bot_deck_policy = None
            if is_bot and bot_info:
                bot_difficulty = bot_info.get("difficulty", "lite")
                bot_card_levels = bot_info.get("card_levels")
                bot_difficulty_label = bot_info.get("difficulty_label", bot_difficulty)
                bot_strength_tier = bot_info.get("strength_tier", bot_difficulty)
                bot_brain_profile = bot_info.get("brain_profile")
                bot_selection = bot_info.get("selection")
                bot_temperature = bot_info.get("temperature")
                bot_card_level_policy = bot_info.get("card_level_policy")
                bot_deck_policy = bot_info.get("deck_policy")

            create_result = await engine.create_match(
                match_id=match_id,
                starting_player_id=starting_player_id_int,
                p1_data={
                    "user_id": p1_id_int,
                    "deck_ids": p1_deck_int_ids,
                    "name": p1_name,
                    "avatar_url": p1_avatar_url,
                    "is_bot": False,
                    "trophies": p1_trophies,
                    "clan": p1_clan,
                    "title": p1_title,
                    "rarity": p1_title_class,
                    "extra_pass": p1_extra_pass,
                    "background_url": p1_background_url,
                },
                p2_data={
                    "user_id": p2_id_int,
                    "deck_ids": p2_deck_int_ids,
                    "name": p2_name,
                    "avatar_url": p2_avatar_url,
                    "is_bot": is_bot,
                    "trophies": p2_trophies,
                    "clan": p2_clan,
                    "difficulty": bot_difficulty,
                    "difficulty_label": bot_difficulty_label,
                    "strength_tier": bot_strength_tier,
                    "brain_profile": bot_brain_profile,
                    "selection": bot_selection,
                    "temperature": bot_temperature,
                    "card_level_policy": bot_card_level_policy,
                    "deck_policy": bot_deck_policy,
                    "card_levels": bot_card_levels,
                    "title": p2_title,
                    "rarity": p2_title_class,
                    "extra_pass": p2_extra_pass,
                    "background_url": p2_background_url,
                },
            )

            if not create_result.get("success"):
                logger.error("Battle init: create_match failed: %s", create_result.get("error"))
                return False
            logger.info(
                "Battle init: starting_player_id=%s current_player_id=%s match_id=%s",
                starting_player_id_int,
                create_result.get("current_player_id"),
                match_id,
            )

        except Exception as engine_exc:  # noqa: BLE001
            logger.error(
                "Battle init: engine creation failed for match_id=%s players=%s: %s",
                match_id,
                player_ids,
                engine_exc,
                exc_info=True,
            )
            return False

        request.app["active_matches"][match_id] = engine
        logger.info("Battle init: engine cached for match_id=%s", match_id)

        # УДАЛЕНО: Больше не запускаем бота сразу после создания движка
        # Теперь бот запускается только после получения сигнала 'client_ready' от фронтенда
        # Это предотвращает преждевременный ход бота до того, как игрок загрузит состояние боя
        logger.info("Battle init: engine ready, waiting for client_ready signal for match_id=%s", match_id)

        return True

    def _attach_match_preview_fields(result: dict[str, Any], engine: BattleEngine, viewer_id: int | None, fallback_mode: str) -> None:
        """Attach pre-battle names/cosmetics from the perspective of the polling player."""
        p1_id = getattr(engine, "_p1_id", None)
        viewer_is_p1 = viewer_id is None or int(viewer_id) == int(p1_id or 0)
        if viewer_is_p1:
            player_prefix, opponent_prefix = "_p1", "_p2"
        else:
            player_prefix, opponent_prefix = "_p2", "_p1"

        result["game_mode"] = getattr(engine, "game_mode", fallback_mode)
        result["mode_config"] = serialize_mode_config(resolve_mode_config(result["game_mode"]))
        result["player_name"] = getattr(engine, f"{player_prefix}_name", "Игрок")
        result["player_avatar_url"] = getattr(engine, f"{player_prefix}_avatar_url", None)
        result["opponent_name"] = getattr(engine, f"{opponent_prefix}_name", "Соперник")
        result["opponent_avatar_url"] = getattr(engine, f"{opponent_prefix}_avatar_url", None)
        result["opponent_trophies"] = getattr(engine, f"{opponent_prefix}_trophies", 0)
        result["opponent_clan"] = getattr(engine, f"{opponent_prefix}_clan", "")
        result["opponent_title"] = getattr(engine, f"{opponent_prefix}_title", "")
        result["opponent_title_rarity"] = getattr(engine, f"{opponent_prefix}_rarity", "")
        result["opponent_extra_pass"] = getattr(engine, f"{opponent_prefix}_extra_pass", None)
        result["opponent_background"] = getattr(engine, f"{opponent_prefix}_background_url", None)

    def _match_status_participant_ids(status: dict[str, Any]) -> set[int]:
        participant_ids: set[int] = set()
        for raw_id in status.get("player_ids") or ():
            try:
                participant_ids.add(int(raw_id))
            except (TypeError, ValueError):
                continue
        for key in ("user_id", "opponent_id"):
            try:
                raw_id = status.get(key)
                if raw_id is not None:
                    participant_ids.add(int(raw_id))
            except (TypeError, ValueError):
                continue
        return participant_ids

    def _verify_match_status_viewer(status: dict[str, Any], viewer_id: int) -> None:
        participant_ids = _match_status_participant_ids(status)
        if participant_ids and int(viewer_id) in participant_ids:
            return
        raise web.HTTPForbidden(
            reason="not_participant",
            text='{"error":"not_participant"}',
            content_type="application/json",
        )

    async def _load_authoritative_match_profile(db_instance: Database, user_id: int) -> dict[str, Any]:
        try:
            profile = await db_instance.get_user_profile(user_id)
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Failed to load authoritative profile for matchmaking user_id=%s: %s",
                user_id,
                exc,
                exc_info=True,
            )
            raise web.HTTPServiceUnavailable(
                reason="profile_unavailable",
                text='{"error":"profile_unavailable"}',
                content_type="application/json",
            )
        if not profile:
            raise web.HTTPNotFound(
                reason="profile_not_found",
                text='{"error":"profile_not_found"}',
                content_type="application/json",
            )
        return profile

    async def _resolve_match_deck_id(
        db_instance: Database,
        user_id: int,
        selected_deck_id: int | None,
    ) -> int | None:
        try:
            presets = await db_instance.get_user_deck_presets(user_id)
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Failed to load deck presets for matchmaking user_id=%s: %s",
                user_id,
                exc,
                exc_info=True,
            )
            raise web.HTTPServiceUnavailable(
                reason="deck_unavailable",
                text='{"error":"deck_unavailable"}',
                content_type="application/json",
            )

        if not presets:
            raise web.HTTPBadRequest(
                reason="deck_required",
                text='{"error":"deck_required"}',
                content_type="application/json",
            )

        preset = None
        if selected_deck_id is not None:
            preset = next((p for p in presets if p.get("preset_number") == selected_deck_id), None)
            if not preset:
                raise web.HTTPForbidden(
                    reason="deck_not_owned",
                    text='{"error":"deck_not_owned"}',
                    content_type="application/json",
                )
        else:
            try:
                primary = await db_instance.fetchval(
                    "SELECT primary_deck FROM users WHERE user_id = $1",
                    user_id,
                )
            except Exception:
                primary = None
            if primary is not None:
                try:
                    primary_int = int(primary)
                    preset = next((p for p in presets if p.get("preset_number") == primary_int), None)
                except (TypeError, ValueError):
                    preset = None
            if not preset:
                preset = next(
                    (p for p in presets if len(p.get("card_ids") or []) >= DECK_SIZE),
                    presets[0],
                )

        card_ids = preset.get("card_ids") or []
        if len(card_ids) < DECK_SIZE:
            raise web.HTTPBadRequest(
                reason="deck_incomplete",
                text='{"error":"deck_incomplete"}',
                content_type="application/json",
            )
        try:
            return int(preset.get("preset_number"))
        except (TypeError, ValueError):
            return selected_deck_id

    async def match_find_handler(request: web.Request) -> web.Response:
        """Найти матч: Soft Start <300 трофеев и очередь для остальных."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        user_id = await require_user_id_from_payload(request, data)
        selected_deck_id = data.get("selected_deck_id") or data.get("deck_id")
        raw_game_mode = data.get("game_mode") or data.get("mode") or "classic"
        db_instance: Database = request.app["db"]

        try:
            user_id = int(user_id)
            if selected_deck_id is not None:
                selected_deck_id = int(selected_deck_id)
        except Exception:
            return web.json_response({"error": "invalid_parameters"}, status=400)

        selected_deck_id = await _resolve_match_deck_id(db_instance, user_id, selected_deck_id)
        profile = await _load_authoritative_match_profile(db_instance, user_id)
        trophies = int(profile.get("trophies", 0) or 0)

        canonical_mode, unavailable_payload = await _resolve_db_aware_mode(db_instance, raw_game_mode, user_id=user_id)
        if unavailable_payload:
            return web.json_response({"status": "canceled", **unavailable_payload}, status=200)
        assert canonical_mode is not None
        game_mode = canonical_mode

        if not await _is_admin_user(db_instance, user_id):
            blocked_card = await _find_disabled_card_in_deck(db_instance, user_id, selected_deck_id)
            if blocked_card:
                return _disabled_card_response(blocked_card)

        # Backend-computed level: источник истины для bot difficulty scaling
        user_max_level = await db_instance.get_player_deck_max_level(user_id, selected_deck_id)

        matchmaker: Matchmaker = request.app["matchmaker"]
        logger = logging.getLogger(__name__)
        try:
            logger.info(
                "match_find_handler: user_id=%s trophies=%s max_level=%s deck_id=%s game_mode=%s canonical=%s",
                user_id, trophies, user_max_level, selected_deck_id, raw_game_mode, canonical_mode
            )
            result = await matchmaker.find_match(
                user_id, trophies, user_max_level, selected_deck_id,
                game_mode=raw_game_mode,
                canonical_mode=canonical_mode,
            )

            # Если матч найден сразу - прогреваем движок до ответа, чтобы арена
            # по прямому редиректу не упала с 404.
            if result.get("status") == "found":
                match_id = str(result.get("match_id"))
                # Совместимость с несколькими форматами матчмейкера:
                # - старый: user_id + opponent_id
                # - новый: player_ids=[p1, p2]
                raw_player_ids = result.get("player_ids")
                player_ids = (
                    list(raw_player_ids)
                    if isinstance(raw_player_ids, (list, tuple)) and len(raw_player_ids) >= 2
                    else [result.get("user_id"), result.get("opponent_id")]
                )
                if match_id and match_id not in request.app["active_matches"]:
                    try:
                        # Извлекаем информацию о боте (если есть)
                        bot_info = result.get("bot_info") if result.get("is_bot") else None
                        logger.info(
                            "match_find_handler: preparing engine for match_id=%s, is_bot=%s, bot_info=%s",
                            match_id,
                            result.get("is_bot"),
                            bot_info
                        )
                        ok = await asyncio.wait_for(
                            _prepare_and_cache_engine(
                                request,
                                match_id=match_id,
                                player_ids=player_ids,
                                is_bot=bool(result.get("is_bot")),
                                bot_info=bot_info,
                                player_decks=result.get("player_decks"),
                                starting_player_id=result.get("starting_player_id"),
                                game_mode=game_mode,
                            ),
                            timeout=6.0,
                        )
                        logger.info("match_find_handler battle init result: %s (match_id=%s)", ok, match_id)
                        if not ok:
                            cancel_payload = await matchmaker.cancel_match(
                                match_id,
                                game_mode=game_mode,
                                message="Выбранный режим сейчас недоступен. Попробуйте позже.",
                            )
                            return web.json_response(cancel_payload, status=200)
                    except asyncio.TimeoutError:
                        logging.getLogger(__name__).error(
                            "match_find_handler: battle init timeout (match_id=%s)", match_id, exc_info=True
                        )
                        return web.json_response({"status": "error", "message": "battle_init_timeout"}, status=504)
                    except Exception as exc:  # noqa: BLE001
                        logging.getLogger(__name__).error(
                            "match_find_handler: battle init failed (match_id=%s): %s", match_id, exc, exc_info=True
                        )
                        return web.json_response({"status": "error", "message": "battle_init_failed"}, status=500)

            if result.get("status") == "found":
                match_id = str(result.get("match_id"))
                if match_id:
                    current_state = None
                    engine = request.app["active_matches"].get(match_id)
                    if engine:
                        try:
                            current_state = engine.get_full_state() if hasattr(engine, "get_full_state") else engine.get_state()
                        except Exception:
                            current_state = None
                    logger.info(
                        "match_find_handler battle snapshot: match_id=%s state=%s",
                        match_id,
                        current_state,
                    )
                    if engine:
                        _attach_match_preview_fields(result, engine, user_id, game_mode)
                        request.app["match_game_modes"][match_id] = getattr(engine, "game_mode", game_mode)

            logger.info("match_find_handler response: %s", result)
            return web.json_response(result)
        except Exception as e:
            logging.getLogger(__name__).error("Ошибка матчмейкинга: %s", e, exc_info=True)
            return web.json_response({"error": "matchmaking_failed"}, status=500)

    async def match_vs_bot_handler(request: web.Request) -> web.Response:
        """Немедленный бой против бота с заданной сложностью."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        user_id = await require_user_id_from_payload(request, data)
        difficulty = data.get("difficulty", "medium")
        selected_deck_id = data.get("selected_deck_id") or data.get("deck_id")
        raw_game_mode = data.get("game_mode") or data.get("mode") or "classic"
        db_instance: Database = request.app["db"]

        valid_difficulties = ("lite", "easy", "medium", "hard", "max")
        if difficulty not in valid_difficulties:
            difficulty = "medium"

        try:
            user_id = int(user_id)
            if selected_deck_id is not None:
                selected_deck_id = int(selected_deck_id)
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_user_id"}, status=400)

        selected_deck_id = await _resolve_match_deck_id(db_instance, user_id, selected_deck_id)
        profile = await _load_authoritative_match_profile(db_instance, user_id)
        trophies = int(profile.get("trophies", 0) or 0)

        canonical_mode, unavailable_payload = await _resolve_db_aware_mode(db_instance, raw_game_mode, user_id=user_id)
        if unavailable_payload:
            return web.json_response({"status": "canceled", **unavailable_payload}, status=200)
        assert canonical_mode is not None
        game_mode = canonical_mode

        if not await _is_admin_user(db_instance, user_id):
            blocked_card = await _find_disabled_card_in_deck(db_instance, user_id, selected_deck_id)
            if blocked_card:
                return _disabled_card_response(blocked_card)

        # Backend-источник истины: max уровень колоды и трофеи из БД
        user_max_level = await db_instance.get_player_deck_max_level(user_id, selected_deck_id)

        # Per-card уровни по выбранной difficulty
        from ai.bot_factory import BotGenerator as _BG

        logger = logging.getLogger(__name__)
        logger.info(
            "match_vs_bot_handler: user_id=%s trophies=%s max_level=%s difficulty=%s",
            user_id, trophies, user_max_level, difficulty
        )

        matchmaker: Matchmaker = request.app["matchmaker"]
        try:
            result = await matchmaker._create_bot_match(
                user_id=user_id,
                trophies=trophies,
                user_max_level=user_max_level,
                selected_deck_id=selected_deck_id,
                game_mode=game_mode,
            )
            # Override difficulty + per-card levels
            bot_deck_ids = result.get("bot_info", {}).get("deck_ids", [])
            card_levels = _BG._build_bot_card_levels(difficulty, user_max_level, len(bot_deck_ids))
            difficulty_meta = _BG._difficulty_metadata(difficulty)
            if result.get("bot_info"):
                result["bot_info"]["difficulty"] = difficulty
                result["bot_info"].update(difficulty_meta)
                result["bot_info"]["card_levels"] = card_levels
            else:
                result["bot_info"] = {"difficulty": difficulty, **difficulty_meta, "card_levels": card_levels}
            if game_mode == "training":
                result["bot_info"] = _decorate_training_bot_info(result.get("bot_info"))

            match_id = str(result["match_id"])
            player_ids = result.get("player_ids", [user_id, result.get("opponent_id", -1)])
            bot_info = result.get("bot_info")

            ok = await asyncio.wait_for(
                _prepare_and_cache_engine(
                    request,
                    match_id=match_id,
                    player_ids=player_ids,
                    is_bot=True,
                    bot_info=bot_info,
                    player_decks=result.get("player_decks"),
                    starting_player_id=result.get("starting_player_id"),
                    game_mode=game_mode,
                ),
                timeout=10.0,
            )
            if not ok:
                return web.json_response({"error": "battle_init_failed"}, status=500)

            engine = request.app["active_matches"].get(match_id)
            opponent_name = engine._p2_name if engine else "Бот"
            opponent_avatar_url = engine._p2_avatar_url if engine else None
            opponent_trophies = getattr(engine, "_p2_trophies", 0) if engine else 0
            opponent_clan = getattr(engine, "_p2_clan", "") if engine else ""
            opponent_title = getattr(engine, "_p2_title", "") if engine else ""
            opponent_title_rarity = getattr(engine, "_p2_rarity", "") if engine else ""
            opponent_extra_pass = getattr(engine, "_p2_extra_pass", None) if engine else None
            opponent_background = getattr(engine, "_p2_background_url", None) if engine else None
            request.app["match_game_modes"][match_id] = getattr(engine, "game_mode", game_mode) if engine else game_mode

            if game_mode == "training":
                training_profile = _training_bot_profile_payload()
                opponent_name = training_profile["name"]
                opponent_avatar_url = training_profile["avatar_url"]
                opponent_trophies = training_profile["trophies"]
                opponent_clan = training_profile["clan"]
                opponent_title = training_profile["title"]
                opponent_title_rarity = training_profile["title_class"]
                opponent_extra_pass = training_profile["extra_pass"]
                opponent_background = training_profile["background_url"]

            return web.json_response({
                "status": "found",
                "match_id": match_id,
                "starting_player_id": result.get("starting_player_id"),
                "game_mode": game_mode,
                "mode_config": serialize_mode_config(resolve_mode_config(game_mode)),
                "opponent_name": opponent_name,
                "opponent_avatar_url": opponent_avatar_url,
                "opponent_trophies": opponent_trophies,
                "opponent_clan": opponent_clan,
                "opponent_title": opponent_title,
                "opponent_title_rarity": opponent_title_rarity,
                "opponent_extra_pass": opponent_extra_pass,
                "opponent_background": opponent_background,
            })
        except asyncio.TimeoutError:
            return web.json_response({"error": "timeout"}, status=504)
        except Exception as e:
            logging.getLogger(__name__).error("vs_bot error: %s", e, exc_info=True)
            return web.json_response({"error": "internal_error", "message": str(e)}, status=500)

    async def match_modes_handler(request: web.Request) -> web.Response:
        """Вернуть текущие режимы и состояние ротации ExtraArena."""
        db_instance: Database = request.app["db"]
        logger = logging.getLogger(__name__)
        try:
            user_id = await require_user_id(request)
        except web.HTTPException:
            user_id = None
        is_admin = await _is_admin_user(db_instance, user_id)
        runtime_config = await _runtime_config_safe(db_instance)
        availability = runtime_config.get("feature_availability") or {}
        try:
            overrides = await db_instance.get_match_mode_overrides()
        except Exception as exc:
            logger.warning("get_match_mode_overrides failed: %s", exc)
            overrides = []

        enabled_map: dict[str, bool] = {o["mode_id"]: o["enabled"] for o in overrides}
        enabled_rotating = [
            m for m in EXTRA_ARENA_ROTATING_IDS
            if (is_admin or enabled_map.get(m, True)) and resolve_mode_config(m).available
        ]
        extra_arena_enabled = is_admin or bool(availability.get("extra_arena", True))

        now = time.time()
        rot = get_current_extra_arena_mode(now, enabled_rotating) if extra_arena_enabled else None
        modifiers = []
        for m in get_extra_arena_mode_list():
            mid = m["mode_id"]
            modifiers.append({
                "mode_id": mid,
                "label": m["label"],
                "description": m.get("description", ""),
                "enabled": (is_admin or enabled_map.get(mid, True)) and extra_arena_enabled,
                "is_current": rot is not None and rot.mode_id == mid,
            })

        return web.json_response({
            "classic": {
                "mode_id": "classic",
                "label": "Classic",
                "enabled": is_admin or (bool(availability.get("classic", True)) and enabled_map.get("classic", True)),
            },
            "training": {"mode_id": "training", "label": "Training", "enabled": is_admin or (bool(availability.get("training", True)) and enabled_map.get("training", True))},
            "friendly": {"mode_id": "friendly", "label": "Friendly", "enabled": is_admin or (bool(availability.get("friendly", True)) and enabled_map.get("friendly", True))},
            "extra_arena": {
                "current_mode_id": rot.mode_id if rot else None,
                "current_label": rot.label if rot else None,
                "current_description": rot.description if rot else None,
                "next_rotation_at": rot.next_rotation_at if rot else None,
                "seconds_to_rotation": rot.seconds_to_rotation if rot else None,
                "rotation_interval_seconds": ROTATION_INTERVAL_SECONDS,
                "rotation_enabled": ROTATION_ENABLED,
                "enabled": extra_arena_enabled,
                "modifiers": modifiers,
            },
        })

    async def mobile_extra_arena_widget_handler(request: web.Request) -> web.Response:
        """Compact public payload for the native Android ExtraArena widget."""
        db_instance: Database = request.app["db"]
        logger = logging.getLogger(__name__)
        runtime_config = await _runtime_config_safe(db_instance)
        availability = runtime_config.get("feature_availability") or {}
        try:
            overrides = await db_instance.get_match_mode_overrides()
        except Exception as exc:
            logger.warning("get_match_mode_overrides failed for widget: %s", exc)
            overrides = []

        enabled_map: dict[str, bool] = {o["mode_id"]: o["enabled"] for o in overrides}
        enabled_rotating = [
            mode_id for mode_id in EXTRA_ARENA_ROTATING_IDS
            if enabled_map.get(mode_id, True) and resolve_mode_config(mode_id).available
        ]
        payload = build_extra_arena_widget_payload(
            time.time(),
            enabled_rotating,
            extra_arena_enabled=bool(availability.get("extra_arena", True)),
        )
        return web.json_response(payload)

    async def match_status_handler(request: web.Request) -> web.Response:
        """Статус матча по match_id для периодического поллинга фронта."""
        match_id = request.rel_url.query.get("id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)

        viewer_id = await require_user_id(request)
        matchmaker: Matchmaker = request.app["matchmaker"]
        try:
            result = await matchmaker.get_status(match_id)
        except Exception as e:
            logging.getLogger(__name__).error("Ошибка статуса матчмейкинга: %s", e, exc_info=True)
            return web.json_response({"error": "matchmaking_failed"}, status=500)

        if result.get("status") == "not_found":
            return web.json_response(result, status=404)

        _verify_match_status_viewer(result, int(viewer_id))

        # Если матч найден, но движок еще не создан (например, матчмейкер вернул статус
        # в поллинге), создаем и кешируем BattleEngine здесь же.
        if result.get("status") == "found":
            match_id = str(result.get("match_id"))
            if match_id and match_id not in request.app["active_matches"]:
                try:
                    # Совместимость с несколькими форматами матчмейкера:
                    # - старый: user_id + opponent_id
                    # - новый: player_ids=[p1, p2]
                    raw_player_ids = result.get("player_ids")
                    player_ids = (
                        list(raw_player_ids)
                        if isinstance(raw_player_ids, (list, tuple)) and len(raw_player_ids) >= 2
                        else [result.get("user_id"), result.get("opponent_id")]
                    )
                    logging.getLogger(__name__).info(
                        "match_status_handler: preparing engine match_id=%s players=%s", match_id, player_ids
                    )
                    # Извлекаем информацию о боте (если есть)
                    bot_info = result.get("bot_info") if result.get("is_bot") else None
                    lazy_game_mode = (
                        result.get("game_mode")
                        or request.app.get("match_game_modes", {}).get(match_id, "")
                        or "classic"
                    )
                    ok = await asyncio.wait_for(
                        _prepare_and_cache_engine(
                            request,
                            match_id=match_id,
                            player_ids=player_ids,
                            is_bot=bool(result.get("is_bot")),
                            bot_info=bot_info,
                            player_decks=result.get("player_decks"),
                            starting_player_id=result.get("starting_player_id"),
                            game_mode=lazy_game_mode,
                        ),
                        timeout=5.0,
                    )
                    if not ok:
                        cancel_payload = await matchmaker.cancel_match(
                            match_id,
                            game_mode=lazy_game_mode,
                            message="Выбранный режим сейчас недоступен. Попробуйте позже.",
                        )
                        return web.json_response(cancel_payload, status=200)
                    else:
                        engine = request.app["active_matches"].get(match_id)
                        if engine:
                            request.app["match_game_modes"][match_id] = getattr(engine, "game_mode", lazy_game_mode)
                            _attach_match_preview_fields(result, engine, int(viewer_id), lazy_game_mode)
                        try:
                            snapshot = engine.get_full_state() if engine and hasattr(engine, "get_full_state") else None
                        except Exception:
                            snapshot = None
                        logging.getLogger(__name__).info(
                            "match_status_handler battle snapshot: match_id=%s state=%s",
                            match_id,
                            snapshot,
                        )
                except asyncio.TimeoutError:
                    logging.getLogger(__name__).error(
                        "match_status_handler: battle init timeout (match_id=%s)", match_id, exc_info=True
                    )
                    return web.json_response({"error": "battle_init_timeout"}, status=504)
                except Exception as exc:  # noqa: BLE001
                    logging.getLogger(__name__).error(
                        "Не удалось инициализировать бой при поллинге статуса (match_id=%s): %s",
                        match_id,
                        exc,
                        exc_info=True,
                    )
                    return web.json_response({"error": "battle_init_failed"}, status=500)

        return web.json_response(result)

    def _get_match_engine(match_id: str) -> BattleEngine | None:
        return ACTIVE_MATCHES.get(str(match_id))


    async def _track_economy_safe(
        db: Database,
        *,
        user_id: int,
        event_type: str,
        resource: str,
        amount: Any,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            await db.track_economy_event(
                user_id=user_id,
                event_type=event_type,
                resource=resource,
                amount=amount,
                source=source,
                metadata=metadata,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "_track_economy_safe failed: user=%s type=%s resource=%s",
                user_id, event_type, resource, exc_info=True,
            )


    def _track_case_rewards(
        db: Database,
        user_id: int,
        rewards: Dict[str, Any],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = dict(extra_meta or {})
        import asyncio as _asyncio
        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            return
        if rewards.get("coins", 0) > 0:
            loop.create_task(
                _track_economy_safe(db, user_id=user_id, event_type="earn",
                    resource="coins", amount=rewards["coins"],
                    source="case_open", metadata=meta))
        if rewards.get("gems", 0) > 0:
            loop.create_task(
                _track_economy_safe(db, user_id=user_id, event_type="earn",
                    resource="gems", amount=rewards["gems"],
                    source="case_open", metadata=meta))
        for c in rewards.get("cards", []):
            loop.create_task(
                _track_economy_safe(db, user_id=user_id, event_type="earn",
                    resource="card", amount=1,
                    source="case_open",
                    metadata={**meta, "card_id": c.get("card_id"), "card_name": c.get("card_name"), "rarity": c.get("rarity")}))
        for p in rewards.get("particles", []):
            loop.create_task(
                _track_economy_safe(db, user_id=user_id, event_type="earn",
                    resource="particles", amount=p.get("particles", 0),
                    source="case_open",
                    metadata={**meta, "card_id": p.get("card_id"), "rarity": p.get("rarity")}))

    def _extract_engine_state(engine: BattleEngine) -> Any:
        """
        Унифицируем получение текущего состояния боя, чтобы не зависеть
        от конкретной версии BattleEngine (state attr или get_state()).
        """
        if hasattr(engine, "get_state"):
            return engine.get_state()
        if hasattr(engine, "state"):
            return getattr(engine, "state")
        # Фолбэк - минимальный срез, чтобы фронт хотя бы видел чей ход.
        return {
            "current_player": getattr(engine, "current_player_id", None),
            "turn": getattr(engine, "turn", 0),
        }

    def _decorate_card_image(card_obj: Card) -> Card:
        """
        Проставляем путь к изображению в DesignAssets/Cards/<id>.png, чтобы
        движок и фронт тянули файлы напрямую из статики, независимо от БД.
        """
        try:
            card_obj.image_url = f"{CARD_IMAGE_URL_PREFIX}/{card_obj.id}.png"
        except Exception:
            # Если по какой-то причине объект не допускает атрибут, тихо продолжаем.
            pass
        return card_obj

    async def _load_card_cache() -> dict[str, Card]:
        """
        Загружаем полный каталог карт и приводим к dict[str, Card].
        Нужен для BattleEngine, чтобы сразу возвращать названия и статы.
        """
        cache: dict[str, Card] = {}
        try:
            cards = await db.get_cards_list()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Не удалось загрузить каталог карт: %s", exc, exc_info=True)
            return cache

        for entry in cards:
            try:
                card_obj = entry if isinstance(entry, Card) else Card.from_row(entry)
                # Прописываем url картинки, чтобы дальше в бою приходила статика из DesignAssets.
                card_obj = _decorate_card_image(card_obj)
                cache[str(card_obj.id)] = card_obj
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).debug("Пропущена карта при построении кеша: %s", exc)
        return cache

    def _normalize_deck_with_cache(deck_ids: list[Any], card_cache: dict[str, Card]) -> list[str]:
        """
        Оставляем только валидные card_id, совпадающие с кешом карт.
        Если валидных карт нет, подбираем первые доступные из кеша.
        """
        normalized: list[str] = []
        available_ids = set(card_cache.keys())

        for raw in deck_ids or []:
            try:
                if isinstance(raw, dict):
                    cid = raw.get("id")
                    level = raw.get("level", 1)
                    candidate = f"{cid}:{level}" if cid is not None else None
                elif isinstance(raw, str) and ":" in raw:
                    candidate = raw
                    cid = raw.split(":", 1)[0]
                else:
                    cid = raw
                    candidate = str(raw) if raw is not None else None
                if cid is None or candidate is None:
                    continue
                if str(cid) in available_ids:
                    normalized.append(candidate)
            except Exception:
                continue

        if not normalized and available_ids:
            # Подстраховка: выдаем первые 9 карт из кеша, чтобы бой мог стартовать.
            normalized = list(list(available_ids)[:9])

        return normalized

    async def _load_player_deck_and_hero(user_id: int, selected_deck_id: int | None = None) -> tuple[list[str], int | None]:
        """
        Асинхронно загружаем выбранный пресет игрока (список из 9 карт, включая героя).

        НОВАЯ ЛОГИКА (Герой внутри колоды):
        - Загружаем все 9 слотов колоды
        - Герой находится внутри колоды как карта с card_type == 'hero'
        - BattleEngine сам найдет героя и отфильтрует его из игровой колоды
        - Возвращаем hero_hp=None, чтобы BattleEngine использовал героя из колоды

        Возвращает: (deck_ids: list[str], hero_hp: None)
        """
        is_bot_user = False
        try:
            # Определяем, бот ли пользователь, чтобы приоритетно брать «бот-колоду».
            bot_flag = await db.fetchval("SELECT COALESCE(is_bot, FALSE) FROM users WHERE user_id = $1", user_id)
            is_bot_user = bool(bot_flag)
        except Exception as exc:  # noqa: BLE001 - не прерываем бой, даже если проверка флагов упала
            logging.getLogger(__name__).warning(
                "Не удалось проверить флаг is_bot для %s: %s", user_id, exc, exc_info=True
            )
            # Боты генерируются в выделенном диапазоне id, поэтому используем его как эвристику.
            is_bot_user = user_id >= 900_000_000

        try:
            presets = await db.get_user_deck_presets(user_id)
        except Exception as exc:  # noqa: BLE001 - логируем любые сбои БД
            logging.getLogger(__name__).warning(
                "Не удалось получить колоду пользователя %s: %s", user_id, exc, exc_info=True
            )
            return [], None

        if not presets:
            return [], None

        # Пытаемся найти запрошенный пресет
        preset = None
        if selected_deck_id is not None:
            preset = next((p for p in presets if p.get("preset_number") == selected_deck_id), None)
            if preset:
                logging.getLogger(__name__).info("Using selected deck preset %s for user %s", selected_deck_id, user_id)

        if not preset:
            try:
                primary = await db.fetchval("SELECT primary_deck FROM users WHERE user_id = $1", user_id)
            except Exception:
                primary = None
            if primary is not None:
                try:
                    primary_int = int(primary)
                    preset = next((p for p in presets if p.get("preset_number") == primary_int), None)
                except (TypeError, ValueError):
                    preset = None

        if not preset:
            # Для ботов отдаем приоритет пресетам, помеченным used_by_bot, чтобы не использовать случайные пользовательские данные.
            preset_candidates = presets
            if is_bot_user:
                bot_presets = [p for p in presets if p.get("used_by_bot")]
                if bot_presets:
                    preset_candidates = bot_presets

            # Берем первый непустой пресет среди кандидатов, иначе первый по порядку.
            preset = next(
                (p for p in preset_candidates if any(p.get(f"card_slot_{idx}") for idx in range(1, 10))),
                preset_candidates[0],
            )

        # Загружаем все 9 слотов колоды (включая героя)
        deck_ids = [
            str(preset.get(f"card_slot_{idx}"))
            for idx in range(1, 10)
            if preset.get(f"card_slot_{idx}") is not None
        ]

        if not deck_ids:
            logging.getLogger(__name__).warning(
                "Колода пользователя %s не найдена в БД (preset_number=%s, used_by_bot=%s)",
                user_id,
                preset.get("preset_number"),
                preset.get("used_by_bot"),
            )

        # ВАЖНО: Возвращаем hero_hp=None, чтобы BattleEngine использовал героя из колоды
        # Старое поле "hero" игнорируется - герой теперь часть колоды
        return deck_ids, None

    async def battle_state_handler(request: web.Request) -> web.Response:
        """Вернуть актуальное состояние боя из кешированного движка."""
        match_id = request.rel_url.query.get("match_id") or request.rel_url.query.get("id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)

        viewer_id = await require_user_id(request)
        engine = _get_match_engine(match_id)
        if not engine:
            # Попытка ленивой инициализации боя, если матч уже найден матчмейкером.
            try:
                matchmaker: Matchmaker = request.app["matchmaker"]
                status = await matchmaker.get_status(match_id)
                if status.get("status") == "found":
                    _verify_match_status_viewer(status, int(viewer_id))
                    # Совместимость с несколькими форматами матчмейкера:
                    # - старый: user_id + opponent_id
                    # - новый: player_ids=[p1, p2]
                    raw_player_ids = status.get("player_ids")
                    player_ids = (
                        list(raw_player_ids)
                        if isinstance(raw_player_ids, (list, tuple)) and len(raw_player_ids) >= 2
                        else [status.get("user_id"), status.get("opponent_id")]
                    )
                    logging.getLogger(__name__).info(
                        "battle_state_handler: engine missing, preparing on-demand match_id=%s players=%s",
                        match_id,
                        player_ids,
                    )
                    # Извлекаем информацию о боте (если есть)
                    bot_info = status.get("bot_info") if status.get("is_bot") else None
                    lazy_game_mode = (
                        status.get("game_mode")
                        or request.app.get("match_game_modes", {}).get(match_id, "")
                        or "classic"
                    )
                    ok = await asyncio.wait_for(
                        _prepare_and_cache_engine(
                            request,
                            match_id=match_id,
                            player_ids=player_ids,
                            is_bot=bool(status.get("is_bot")),
                            bot_info=bot_info,
                            player_decks=status.get("player_decks"),
                            starting_player_id=status.get("starting_player_id"),
                            game_mode=lazy_game_mode,
                        ),
                        timeout=5.0,
                    )
                    if ok:
                        engine = _get_match_engine(match_id)
                    else:
                        cancel_payload = await matchmaker.cancel_match(
                            match_id,
                            game_mode=lazy_game_mode,
                            message="Выбранный режим сейчас недоступен. Попробуйте позже.",
                        )
                        return web.json_response(cancel_payload, status=200)
            except asyncio.TimeoutError:
                return web.json_response({"error": "battle_init_timeout"}, status=504)
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).error(
                    "battle_state_handler: failed to init engine on-demand (match_id=%s): %s",
                    match_id,
                    exc,
                    exc_info=True,
                )

        if not engine:
            return web.json_response({"error": "match_not_found"}, status=404)

        try:
            _verify_participant(engine, viewer_id)
            await _handle_natural_turn_timeout(request.app, str(match_id), engine)

            if hasattr(engine, "get_full_state"):
                state = engine.get_full_state(viewer_id=viewer_id)
            else:
                state = _extract_engine_state(engine)
                state["match_id"] = match_id

            # Загружаем ExtraPass статус игрока для премиального визуала
            if viewer_id is not None:
                try:
                    viewer_id_int = int(viewer_id) if not isinstance(viewer_id, int) else viewer_id
                    db_inst = request.app.get("db")
                    if db_inst:
                        user_data = await db_inst.get_user_info(viewer_id_int)
                        if user_data:
                            state["extra_pass"] = user_data.get("extra_pass", "inactive")
                except Exception as extra_pass_exc:
                    logging.getLogger(__name__).debug("Failed to load ExtraPass status: %s", extra_pass_exc)

            return web.json_response(state)
        except web.HTTPException:
            raise
        except Exception as exc:
            logging.getLogger(__name__).error(
                "Не удалось получить состояние боя %s: %s", match_id, exc, exc_info=True
            )
            return web.json_response({"error": "battle_state_failed"}, status=500)

    async def battle_active_handler(request: web.Request) -> web.Response:
        """Return the current in-memory active battle for this user, if any."""
        try:
            user_id = await require_user_id(request)
        except web.HTTPException:
            raise
        except Exception as exc:
            logging.getLogger(__name__).warning("battle_active_handler auth failed: %s", exc)
            return web.json_response({"active": False})

        user_id_int = int(user_id)
        for active_match_id, engine in list(request.app["active_matches"].items()):
            if getattr(engine, "is_ended", False):
                continue
            p1_uid = getattr(engine.p1_state, "user_id", None)
            p2_uid = getattr(engine.p2_state, "user_id", None)
            if user_id_int not in (p1_uid, p2_uid):
                continue

            status = _player_replacement_status(engine, user_id_int)
            game_mode = _resolve_match_game_mode(request.app, str(active_match_id), engine)
            return web.json_response({
                "active": True,
                "match_id": str(active_match_id),
                "game_mode": game_mode,
                "current_player_id": engine.get_current_player_id() if hasattr(engine, "get_current_player_id") else None,
                "replacement_status": getattr(status, "value", str(status)),
                "redirect_url": f"/arena?id={active_match_id}",
            })

        matchmaker = request.app.get("matchmaker")
        if matchmaker and hasattr(matchmaker, "get_active_match_for_user"):
            try:
                status = await matchmaker.get_active_match_for_user(user_id_int)
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).warning(
                    "battle_active_handler matchmaker lookup failed for user_id=%s: %s",
                    user_id_int,
                    exc,
                )
                status = {}
            if status.get("status") == "found" and status.get("match_id"):
                match_id = str(status["match_id"])
                if match_id in ENDED_MATCH_IDS:
                    return web.json_response({"active": False})
                existing_engine = request.app["active_matches"].get(match_id)
                if existing_engine is not None and getattr(existing_engine, "is_ended", False):
                    _mark_match_ended(match_id)
                    return web.json_response({"active": False})
                return web.json_response({
                    "active": True,
                    "match_id": match_id,
                    "game_mode": status.get("game_mode") or "classic",
                    "current_player_id": None,
                    "replacement_status": "active",
                    "redirect_url": f"/arena?id={match_id}",
                })

        return web.json_response({"active": False})

    async def battle_play_card_handler(request: web.Request) -> web.Response:
        """
        Розыгрыш карты игрока через core/actions.PlayCardAction.
        """
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        match_id = payload.get("match_id") or payload.get("id")
        # Поддержка: hand_index/card_id_from_hand/card_id. Индекс из руки
        # приоритетнее card_id, иначе дубликаты одной карты разыгрывают первую копию.
        raw_card_id = payload.get("card_id_from_hand")
        if raw_card_id is None:
            raw_card_id = payload.get("hand_index")
        if raw_card_id is None:
            raw_card_id = payload.get("card_id")
        board_position = payload.get("target_position") or payload.get("board_position") or payload.get("position") or 0
        user_id_int = await require_user_id_from_payload(request, payload)
        client_action_id = _client_action_id(payload)

        # Параметры для зелий / battlecry
        target_id = payload.get("target_id")
        target_is_hero = payload.get("target_is_hero", False)

        logger = logging.getLogger(__name__)
        logger.info(
            "play_card_handler: match_id=%s card=%s pos=%s user=%s target=%s hero=%s",
            match_id, raw_card_id, board_position, user_id_int, target_id, target_is_hero
        )

        if not match_id or raw_card_id is None or user_id_int is None:
            return web.json_response({"error": "invalid_parameters"}, status=400)
        cached_action = _action_cache_get(match_id, user_id_int, client_action_id)
        if cached_action:
            return web.json_response(cached_action["payload"], status=cached_action["status"])

        try:
            board_position = int(board_position)
        except (ValueError, TypeError):
            board_position = 0

        engine = _get_match_engine(match_id)
        if not engine:
            if _is_finished_match(match_id):
                return web.json_response(_build_finished_match_action_payload(match_id))
            return web.json_response({"error": "match_not_found"}, status=404)

        _verify_participant(engine, user_id_int)
        if _is_finished_match(match_id, engine):
            payload_out = _build_finished_match_action_payload(match_id, engine, user_id_int)
            _action_cache_set(match_id, user_id_int, client_action_id, payload_out)
            return web.json_response(payload_out)

        lock = _get_match_lock(match_id)
        async with lock:
            try:
                not_ready_response = _match_not_ready_response(match_id, engine, int(user_id_int))
                if not_ready_response is not None:
                    return not_ready_response

                expired_response = await _auto_end_expired_turn_response(
                    request.app,
                    str(match_id),
                    engine,
                    int(user_id_int),
                    client_action_id,
                )
                if expired_response is not None:
                    return expired_response

                _mark_user_activity_for_match(str(match_id), int(user_id_int), engine)
                try:
                    engine.record_analytics_action(user_id_int, {
                        "type": "play_card",
                        "card_ref": raw_card_id,
                        "board_position": board_position,
                        "target_id": target_id,
                        "target_is_hero": target_is_hero,
                    })
                except Exception:
                    pass

                result = engine.play_card(
                    user_id_int,
                    raw_card_id,
                    board_position,
                    target_id=target_id,
                    target_is_hero=target_is_hero
                )
                if result.get("success") is False:
                    state = engine.get_full_state(viewer_id=user_id_int)
                    payload_out = {"result": result, "state": state, "error": result.get("error")}
                    status = _action_failure_status(result)
                    _action_cache_set(match_id, user_id_int, client_action_id, payload_out, status=status)
                    return web.json_response(payload_out, status=status)

                # Обработка завершения игры
                if result.get("game_over"):
                    logger.info("🏁 Game Over after play_card! Winner: %s", result.get("winner"))
                    await _process_battle_end(request.app, match_id, engine, result.get("winner"))

                    sio_inst = request.app.get("socketio")
                    if sio_inst:
                        await sio_inst.emit(
                            "game_over",
                            _build_game_over_payload(engine, result.get("winner"), reason="hero_death"),
                            room=match_id,
                        )

                # Получаем состояние с legal_actions
                state = engine.get_full_state(viewer_id=user_id_int)

                # Добавляем данные о трофеях/монетах/звёздах если game_over
                if result.get("game_over"):
                    trophy_changes = getattr(engine, "_trophy_changes", {})
                    trophy_totals = getattr(engine, "_trophy_totals", {})
                    coins_changes = getattr(engine, "_coins_changes", {})
                    coins_totals = getattr(engine, "_coins_totals", {})
                    stars_changes = getattr(engine, "_stars_changes", {})
                    stars_totals = getattr(engine, "_stars_totals", {})
                    keys_changes = getattr(engine, "_keys_changes", {})
                    keys_totals = getattr(engine, "_keys_totals", {})

                    if user_id_int in trophy_changes:
                        state["trophy_delta"] = trophy_changes[user_id_int]
                        state["trophy_total"] = trophy_totals.get(user_id_int, 0)
                    if user_id_int in coins_changes:
                        state["coins_delta"] = coins_changes[user_id_int]
                        state["coins_total"] = coins_totals.get(user_id_int, 0)
                    if user_id_int in stars_changes:
                        state["stars_delta"] = stars_changes[user_id_int]
                        state["stars_total"] = stars_totals.get(user_id_int, 0)
                    if user_id_int in keys_changes:
                        state["keys_delta"] = keys_changes[user_id_int]
                        state["keys_total"] = keys_totals.get(user_id_int, 0)
                    league_up = getattr(engine, "_league_up", {})
                    if user_id_int in league_up:
                        state["league_up"] = league_up[user_id_int]

                # Проверяем, перешел ли ход к боту
                if not result.get("game_over"):
                    await trigger_bot_move(match_id)

                payload_out = {"result": result, "state": state}
                _action_cache_set(match_id, user_id_int, client_action_id, payload_out)
                return web.json_response(payload_out)
            except Exception as exc:
                logger.warning("Ошибка розыгрыша карты в матче %s: %s", match_id, exc, exc_info=True)
                return web.json_response({"error": "play_card_failed"}, status=400)

    async def battle_attack_handler(request: web.Request) -> web.Response:
        """
        Атака существом через core/actions.AttackAction.
        """
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        match_id = payload.get("match_id") or payload.get("id")
        attacker_id = payload.get("attacker_id")
        target_id = payload.get("target_id")
        target_is_hero = bool(payload.get("target_is_hero"))
        user_id_int = await require_user_id_from_payload(request, payload)
        client_action_id = _client_action_id(payload)

        if not match_id or attacker_id is None or user_id_int is None:
            return web.json_response({"error": "invalid_parameters"}, status=400)
        cached_action = _action_cache_get(match_id, user_id_int, client_action_id)
        if cached_action:
            return web.json_response(cached_action["payload"], status=cached_action["status"])

        engine = _get_match_engine(match_id)
        if not engine:
            if _is_finished_match(match_id):
                return web.json_response(_build_finished_match_action_payload(match_id))
            return web.json_response({"error": "match_not_found"}, status=404)

        _verify_participant(engine, user_id_int)
        if _is_finished_match(match_id, engine):
            payload_out = _build_finished_match_action_payload(match_id, engine, user_id_int)
            _action_cache_set(match_id, user_id_int, client_action_id, payload_out)
            return web.json_response(payload_out)

        logger = logging.getLogger(__name__)
        logger.info("attack_handler: match=%s attacker=%s target=%s hero=%s user=%s",
                    match_id, attacker_id, target_id, target_is_hero, user_id_int)

        lock = _get_match_lock(match_id)
        async with lock:
            try:
                not_ready_response = _match_not_ready_response(match_id, engine, int(user_id_int))
                if not_ready_response is not None:
                    return not_ready_response

                expired_response = await _auto_end_expired_turn_response(
                    request.app,
                    str(match_id),
                    engine,
                    int(user_id_int),
                    client_action_id,
                )
                if expired_response is not None:
                    return expired_response

                _mark_user_activity_for_match(str(match_id), int(user_id_int), engine)
                try:
                    engine.record_analytics_action(user_id_int, {
                        "type": "attack",
                        "attacker_id": attacker_id,
                        "target_id": target_id,
                        "target_is_hero": target_is_hero,
                    })
                except Exception:
                    pass

                result = engine.attack_target(
                    user_id_int,
                    attacker_id,
                    target_id,
                    target_is_hero=target_is_hero,
                )
                if result.get("success") is False:
                    state = engine.get_full_state(viewer_id=user_id_int)
                    payload_out = {"result": result, "state": state, "error": result.get("error")}
                    status = _action_failure_status(result)
                    _action_cache_set(match_id, user_id_int, client_action_id, payload_out, status=status)
                    return web.json_response(payload_out, status=status)

                # Обработка завершения игры
                if result.get("game_over"):
                    logger.info("🏁 Game Over after attack! Winner: %s", result.get("winner"))
                    await _process_battle_end(request.app, match_id, engine, result.get("winner"))

                    sio_inst = request.app.get("socketio")
                    if sio_inst:
                        await sio_inst.emit(
                            "game_over",
                            _build_game_over_payload(engine, result.get("winner"), reason="hero_death"),
                            room=match_id,
                        )

                # Получаем состояние с legal_actions
                state = engine.get_full_state(viewer_id=user_id_int)

                # Добавляем данные о трофеях/монетах/звёздах если game_over
                if result.get("game_over"):
                    trophy_changes = getattr(engine, "_trophy_changes", {})
                    trophy_totals = getattr(engine, "_trophy_totals", {})
                    coins_changes = getattr(engine, "_coins_changes", {})
                    coins_totals = getattr(engine, "_coins_totals", {})
                    stars_changes = getattr(engine, "_stars_changes", {})
                    stars_totals = getattr(engine, "_stars_totals", {})
                    keys_changes = getattr(engine, "_keys_changes", {})
                    keys_totals = getattr(engine, "_keys_totals", {})

                    if user_id_int in trophy_changes:
                        state["trophy_delta"] = trophy_changes[user_id_int]
                        state["trophy_total"] = trophy_totals.get(user_id_int, 0)
                    if user_id_int in coins_changes:
                        state["coins_delta"] = coins_changes[user_id_int]
                        state["coins_total"] = coins_totals.get(user_id_int, 0)
                    if user_id_int in stars_changes:
                        state["stars_delta"] = stars_changes[user_id_int]
                        state["stars_total"] = stars_totals.get(user_id_int, 0)
                    if user_id_int in keys_changes:
                        state["keys_delta"] = keys_changes[user_id_int]
                        state["keys_total"] = keys_totals.get(user_id_int, 0)
                    league_up = getattr(engine, "_league_up", {})
                    if user_id_int in league_up:
                        state["league_up"] = league_up[user_id_int]

                # Проверяем, перешел ли ход к боту
                if not result.get("game_over"):
                    await trigger_bot_move(match_id)

                payload_out = {"result": result, "state": state}
                _action_cache_set(match_id, user_id_int, client_action_id, payload_out)
                return web.json_response(payload_out)
            except Exception as exc:
                logger.warning("Ошибка атаки в матче %s: %s", match_id, exc, exc_info=True)
                return web.json_response({"error": "attack_failed", "details": str(exc)}, status=400)

    # calculate_trophy_delta и calculate_coins_reward вынесены на уровень модуля

    # [MOVED TO MODULE LEVEL]

    # УДАЛЕНО: check_and_run_bot и run_bot_routine перенесены на уровень модуля
    # См. определения функций перед create_web_app()

    async def battle_surrender_handler(request: web.Request) -> web.Response:
        """
        Обработчик сдачи игрока (surrender).
        Трофеи списываются немедленно, но бой продолжается под управлением бота.
        """
        logger = logging.getLogger(__name__)
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        match_id = request.match_info.get("match_id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)

        user_id_int = await require_user_id_from_payload(request, payload)
        if user_id_int is None:
            return web.json_response({"error": "user_id_required"}, status=400)
        client_action_id = _client_action_id(payload)
        cached_action = _action_cache_get(match_id, user_id_int, client_action_id)
        if cached_action:
            return web.json_response(cached_action["payload"], status=cached_action["status"])

        engine = _get_match_engine(match_id)
        if not engine:
            return web.json_response({"error": "match_not_found"}, status=404)

        _verify_participant(engine, user_id_int)

        lock = _get_match_lock(match_id)
        async with lock:
            engine.mark_surrender(user_id_int)

        penalty_result = await _apply_surrender_penalty_once(request.app, match_id, engine, user_id_int)
        if not penalty_result.get("success"):
            payload_out = {"error": penalty_result.get("error", "surrender_failed")}
            status = int(penalty_result.get("status", 500) or 500)
            _action_cache_set(match_id, user_id_int, client_action_id, payload_out, status=status)
            return web.json_response(payload_out, status=status)

        game_over_result = engine.check_game_over()

        if game_over_result.get("game_over"):
            winner_id = game_over_result.get("winner_id")
            await _process_battle_end(request.app, match_id, engine, winner_id)
            sio_inst = request.app.get("socketio")
            if sio_inst:
                await sio_inst.emit(
                    "game_over",
                    _build_game_over_payload(engine, winner_id, reason="surrender"),
                    room=match_id,
                )

        if not game_over_result.get("game_over") and engine.current_player_id == user_id_int:
            await check_and_run_bot(match_id, ACTIVE_MATCHES)

        current_state = engine.get_full_state(viewer_id=user_id_int)

        payload_out = {
            "success": True,
            "message": "surrender_processed",
            "already_processed": bool(penalty_result.get("already_processed")),
            "trophy_penalty": penalty_result.get("trophy_penalty", 0),
            "new_trophies": penalty_result.get("new_trophies", 0),
            "state": current_state,
            "game_over": game_over_result.get("game_over", False)
        }
        _action_cache_set(match_id, user_id_int, client_action_id, payload_out)
        return web.json_response(payload_out)

    async def battle_turn_end_handler(request: web.Request) -> web.Response:
        """Завершение хода игрока через core/actions.EndTurnAction."""
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        match_id = payload.get("match_id") or payload.get("id")
        if not match_id:
            return web.json_response({"error": "match_id_required"}, status=400)

        user_id_int = await require_user_id_from_payload(request, payload)
        if user_id_int is None:
            return web.json_response({"error": "user_id_required"}, status=400)
        client_action_id = _client_action_id(payload)
        cached_action = _action_cache_get(match_id, user_id_int, client_action_id)
        if cached_action:
            return web.json_response(cached_action["payload"], status=cached_action["status"])

        engine = _get_match_engine(match_id)
        if not engine:
            if _is_finished_match(match_id):
                return web.json_response(_build_finished_match_action_payload(match_id))
            return web.json_response({"error": "match_not_found"}, status=404)

        _verify_participant(engine, user_id_int)
        if _is_finished_match(match_id, engine):
            payload_out = _build_finished_match_action_payload(match_id, engine, user_id_int)
            _action_cache_set(match_id, user_id_int, client_action_id, payload_out)
            return web.json_response(payload_out)

        logger = logging.getLogger(__name__)
        logger.info("turn_end_handler: match=%s user=%s", match_id, user_id_int)

        lock = _get_match_lock(match_id)
        async with lock:
            try:
                not_ready_response = _match_not_ready_response(match_id, engine, int(user_id_int))
                if not_ready_response is not None:
                    return not_ready_response

                expired_response = await _auto_end_expired_turn_response(
                    request.app,
                    str(match_id),
                    engine,
                    int(user_id_int),
                    client_action_id,
                )
                if expired_response is not None:
                    return expired_response

                _mark_user_activity_for_match(str(match_id), int(user_id_int), engine)
                try:
                    engine.record_analytics_action(user_id_int, {"type": "end_turn"})
                except Exception:
                    pass
                result = engine.end_turn(user_id_int)
                if result.get("success") is False:
                    state = engine.get_full_state(viewer_id=user_id_int)
                    payload_out = {"result": result, "state": state, "error": result.get("error")}
                    status = _action_failure_status(result)
                    _action_cache_set(match_id, user_id_int, client_action_id, payload_out, status=status)
                    return web.json_response(payload_out, status=status)
                logger.info("turn_end_handler: success, current_player=%s", engine.get_current_player_id())
            except Exception as exc:
                logger.warning("Ошибка завершения хода для матча %s: %s", match_id, exc, exc_info=True)
                return web.json_response({"error": "turn_end_failed", "details": str(exc)}, status=400)

        try:
            await check_and_run_bot(match_id, ACTIVE_MATCHES)
        except Exception as exc:
            logger.warning("Не удалось запустить проверку бота: %s", exc, exc_info=True)

        state = engine.get_full_state(viewer_id=user_id_int)

        payload_out = {"match_id": match_id, "result": result, "state": state}
        _action_cache_set(match_id, user_id_int, client_action_id, payload_out)
        return web.json_response(payload_out)

    async def battle_preview_handler(request: web.Request) -> web.Response:
        """Предпросмотр урона для действия без его выполнения."""
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)

        match_id = payload.get("match_id")
        action_data = payload.get("action")

        if not match_id or not action_data:
            return web.json_response({"error": "match_id_and_action_required"}, status=400)

        user_id_int = await require_user_id_from_payload(request, payload)
        if user_id_int is None:
            return web.json_response({"error": "user_id_required"}, status=400)

        engine = _get_match_engine(match_id)
        if not engine:
            return web.json_response({"error": "match_not_found"}, status=404)

        _verify_participant(engine, user_id_int)
        if hasattr(engine, "get_current_player_id"):
            try:
                current_player_id = int(engine.get_current_player_id())
            except (TypeError, ValueError):
                current_player_id = None
            if current_player_id != int(user_id_int):
                return web.json_response({"error": "not_your_turn"}, status=403)

        logger = logging.getLogger(__name__)
        logger.info("preview_handler: match=%s user=%s action=%s", match_id, user_id_int, action_data)

        try:
            from core.actions import PlayCardAction, AttackAction, EndTurnAction
            action_type = action_data.get("type")
            if action_type == "play_card":
                action = PlayCardAction(
                    hand_index=action_data.get("hand_index", 0),
                    target_id=action_data.get("target_id"),
                    position=action_data.get("position"),
                )
            elif action_type == "attack":
                action = AttackAction(
                    attacker_id=str(action_data.get("attacker_id", "")),
                    target_id=action_data.get("target_id"),
                    target_is_hero=action_data.get("target_is_hero", False),
                )
            elif action_type == "end_turn":
                action = EndTurnAction()
            else:
                return web.json_response({"error": "unknown_action_type"}, status=400)
        except Exception as exc:
            logger.warning("Ошибка парсинга действия: %s", exc, exc_info=True)
            return web.json_response({"error": "invalid_action_format", "details": str(exc)}, status=400)

        try:
            preview_delta = engine.get_preview_delta(action)
            return web.json_response({"success": True, "preview_data": preview_delta})
        except Exception as exc:
            logger.warning("Ошибка получения предпросмотра: %s", exc, exc_info=True)
            return web.json_response({"error": "preview_failed", "details": str(exc)}, status=400)

    # ========== Регистрация роутов ==========

    app.router.add_get("/health", health_check)
    app.router.add_get("/", index)

    # === СТАТИКА: РАЗДАЕМ РЕСУРСЫ КАРТ (ДОЛЖНО БЫТЬ В НАЧАЛЕ) ===
    # Добавляем статическую раздачу ресурсов перед всеми остальными роутами
    import os
    # Путь от web/server.py -> .. -> DesignAssets
    design_assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DesignAssets"))
    if os.path.exists(design_assets_path):
        app.router.add_static("/DesignAssets/", path=design_assets_path, name="design_assets")
        logging.getLogger(__name__).info("✅ Static route added: /DesignAssets/ -> %s", design_assets_path)
    else:
        logging.getLogger(__name__).error("❌ DesignAssets directory NOT FOUND at: %s", design_assets_path)

    # Отдельная HTML-страница боя; /arena - основной путь редиректа из фронтенда
    app.router.add_get("/arena", battle_page_handler)   # основной маршрут для UI арены
    app.router.add_get("/battle", battle_page_handler)  # обратная совместимость
    app.router.add_get("/api/mobile/client-version", mobile_client_version_handler)
    app.router.add_get("/api/mobile/extra-arena-widget", mobile_extra_arena_widget_handler)
    app.router.add_get("/api/runtime/status", runtime_status_handler)
    app.router.add_get("/api/profile", profile_handler)
    app.router.add_get("/api/match/modes", match_modes_handler)
    app.router.add_post("/api/match/find", match_find_handler)
    app.router.add_post("/api/match/vs-bot", match_vs_bot_handler)
    app.router.add_get("/api/match/status", match_status_handler)
    app.router.add_get("/api/battle/active", battle_active_handler)
    app.router.add_get("/api/battle/state", battle_state_handler)
    # Новые роуты с дефисами (используются фронтендом)
    app.router.add_post("/api/battle/play-card", battle_play_card_handler)
    app.router.add_post("/api/battle/attack", battle_attack_handler)
    app.router.add_post("/api/battle/end-turn", battle_turn_end_handler)
    app.router.add_post("/api/battle/preview", battle_preview_handler)
    # Старые роуты для обратной совместимости
    app.router.add_post("/api/battle/action/play_card", battle_play_card_handler)
    app.router.add_post("/api/battle/action/attack_target", battle_attack_handler)
    app.router.add_post("/api/battle/turn_end", battle_turn_end_handler)
    app.router.add_post("/api/matches/{match_id}/surrender", battle_surrender_handler)
    app.router.add_get("/api/settings", settings_handler)
    app.router.add_post("/api/settings", settings_handler)
    app.router.add_post("/api/change-nickname", change_nickname_handler)
    app.router.add_get("/api/cosmetics/owned",  cosmetics_owned_handler)
    app.router.add_post("/api/cosmetics/equip", cosmetics_equip_handler)
    app.router.add_post("/api/promocode/use", promocode_use_handler)
    app.router.add_post("/api/admin/promocodes/create", promocode_create_handler)
    app.router.add_get("/api/admin/promocodes/list", promocode_list_handler)
    app.router.add_post("/api/admin/promocodes/delete", promocode_delete_handler)
    app.router.add_post("/api/admin/cards/create", admin_cards_create_handler)
    app.router.add_get("/api/admin/cards/list", admin_cards_list_handler)
    app.router.add_post("/api/admin/items/create", admin_items_create_handler)
    app.router.add_get("/api/admin/items/list", admin_items_list_handler)
    app.router.add_get("/api/deck/presets", deck_presets_list_handler)
    app.router.add_post("/api/deck/presets/save", deck_preset_save_handler)
    app.router.add_post("/api/deck/presets/create", deck_preset_create_handler)
    app.router.add_post("/api/deck/presets/delete", deck_preset_delete_handler)
    app.router.add_post("/api/deck/presets/rename", deck_preset_rename_handler)
    app.router.add_post("/api/deck/presets/set-primary", deck_preset_set_primary_handler)
    app.router.add_get("/api/cards", cards_catalog_handler)
    app.router.add_get("/api/cards/user", user_cards_handler)
    app.router.add_get("/api/cards/collection", collection_with_status_handler)
    app.router.add_get("/api/cases/user", user_cases_handler)
    app.router.add_get("/api/cases/{user_case_id}", user_case_detail_handler)
    app.router.add_post("/api/cases/tap", case_tap_handler)
    app.router.add_post("/api/cases/open", case_open_handler)
    app.router.add_post("/api/cases/skip", case_skip_handler)
    app.router.add_post("/api/cases/roll-from-keys", case_roll_from_keys_handler)
    app.router.add_post("/api/cases/open-from-keys", case_open_from_keys_handler)
    app.router.add_post("/api/debug/add-key", debug_add_key_handler)
    app.router.add_post("/api/admin/cards/get-all", admin_get_all_cards_handler)
    app.router.add_post("/api/admin/cards/delete-all", admin_delete_all_cards_handler)
    app.router.add_post("/api/cards/upgrade", card_upgrade_handler)
    app.router.add_post("/api/cards/add-particles", card_add_particles_handler)
    app.router.add_get("/api/admin/players", admin_players_handler)
    app.router.add_get("/api/admin/stats", admin_stats_handler)

    # ── Analytics endpoints (public: session + onboarding tracking) ──

    def _analytics_json_safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(k): _analytics_json_safe(v)
                for k, v in value.items()
                if str(k) not in {"_auth", "initData", "hash", "signature"}
            }
        if isinstance(value, list):
            return [_analytics_json_safe(v) for v in value]
        return str(value)

    def _safe_json_list(value: Any, max_items: int = 200) -> list:
        if not isinstance(value, list):
            return []
        return [_analytics_json_safe(v) for v in value[:max_items]]

    def _safe_json_dict(value: Any) -> dict:
        if not isinstance(value, dict):
            return {}
        return _analytics_json_safe(value)

    async def analytics_session_start_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        session_id = str(data.get("session_id", "")).strip()
        if not session_id or len(session_id) > 128:
            return web.json_response({"error": "invalid_session_id"}, status=400)
        source = str(data.get("source", "telegram_webapp"))
        await db.start_user_session(user_id, session_id, source)
        return web.json_response({"success": True})

    async def analytics_session_update_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        session_id = str(data.get("session_id", "")).strip()
        if not session_id:
            return web.json_response({"error": "invalid_session_id"}, status=400)
        screens = _safe_json_list(data.get("screens_visited"), max_items=200)
        battles = int(data.get("battles_played") or 0)
        cases = int(data.get("cases_opened") or 0)
        await db.update_user_session(session_id, screens_visited=screens, battles_played=battles, cases_opened=cases)
        return web.json_response({"success": True})

    async def analytics_session_end_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        session_id = str(data.get("session_id", "")).strip()
        if not session_id:
            return web.json_response({"error": "invalid_session_id"}, status=400)
        screens = _safe_json_list(data.get("screens_visited"), max_items=200)
        battles = int(data.get("battles_played") or 0)
        cases = int(data.get("cases_opened") or 0)
        await db.finish_user_session(session_id, screens_visited=screens, battles_played=battles, cases_opened=cases)
        return web.json_response({"success": True})

    async def analytics_onboarding_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid_json"}, status=400)
        step = str(data.get("step", "")).strip()
        if not step or len(step) > 64:
            return web.json_response({"error": "invalid_step"}, status=400)
        completed = bool(data.get("completed", False))
        time_spent = data.get("time_spent_seconds")
        if time_spent is not None:
            try:
                time_spent = float(time_spent)
            except (ValueError, TypeError):
                time_spent = None
        meta = _safe_json_dict(data.get("metadata"))
        await db.track_onboarding_event(user_id, step, completed, time_spent, meta)
        return web.json_response({"success": True})

    app.router.add_post("/api/analytics/session/start", analytics_session_start_handler)
    app.router.add_post("/api/analytics/session/update", analytics_session_update_handler)
    app.router.add_post("/api/analytics/session/end", analytics_session_end_handler)
    app.router.add_post("/api/analytics/onboarding", analytics_onboarding_handler)

    # ── Analytics endpoints (admin only) ──

    async def admin_analytics_overview_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "7"))
            data = await db.get_admin_analytics_overview(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("analytics_overview error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_analytics_revenue_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            data = await db.get_admin_revenue_analytics(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("analytics_revenue error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_analytics_players_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            data = await db.get_admin_players_analytics(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("analytics_players error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_analytics_battles_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            data = await db.get_admin_battle_analytics(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("analytics_battles error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_analytics_economy_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            rows = await db.fetch(
                """
                SELECT event_type, resource, COUNT(*) AS cnt, COALESCE(SUM(amount), 0) AS total_amount
                FROM economy_events
                WHERE created_at >= NOW() - make_interval(days => $1::int)
                GROUP BY event_type, resource
                ORDER BY cnt DESC
                """,
                max(days, 1),
            )
            events = [{"event_type": r["event_type"], "resource": r["resource"], "count": r["cnt"], "total_amount": float(r["total_amount"])} for r in rows]
            return web.json_response({"status": "ok", "data": {"events": events, "total_events": sum(e["count"] for e in events)}})
        except Exception as e:
            logging.getLogger(__name__).error("analytics_economy error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_get("/api/admin/analytics/overview", admin_analytics_overview_handler)
    app.router.add_get("/api/admin/analytics/revenue", admin_analytics_revenue_handler)
    app.router.add_get("/api/admin/analytics/players", admin_analytics_players_handler)
    app.router.add_get("/api/admin/analytics/battles", admin_analytics_battles_handler)
    app.router.add_get("/api/admin/analytics/economy", admin_analytics_economy_handler)

    # ── Analytics: cards / heroes / retention / onboarding / battle-actions ──

    async def admin_analytics_cards_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            data = await db.get_admin_cards_analytics(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("analytics_cards error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_analytics_heroes_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            data = await db.get_admin_heroes_analytics(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("analytics_heroes error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_analytics_retention_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            data = await db.get_admin_retention_analytics(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("analytics_retention error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_analytics_onboarding_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            data = await db.get_admin_onboarding_analytics(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("analytics_onboarding error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_analytics_battle_actions_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "7"))
            data = await db.get_admin_battle_actions_analytics(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("analytics_battle_actions error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_analytics_dataset_export_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            limit = int(request.rel_url.query.get("limit", "5000"))
            include_players = request.rel_url.query.get("include_players", "0") in ("1", "true", "yes")
            rows = await db.export_train_v2_battle_dataset(
                days=days,
                limit=limit,
                include_players=include_players,
            )
            header = {
                "format": "train_v2_admin_battle_action_jsonl_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "days": max(days, 1),
                "rows": len(rows),
                "notes": "Each following line is a battle action sample with raw state_json/action_json captured from production analytics.",
            }
            lines = [_stdlib_json.dumps(header, ensure_ascii=False)]
            lines.extend(_stdlib_json.dumps(row, ensure_ascii=False) for row in rows)
            filename = f"extraarena_trainv2_dataset_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
            return web.Response(
                text="\n".join(lines) + "\n",
                content_type="application/x-ndjson",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as e:
            logging.getLogger(__name__).error("analytics_dataset_export error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_get("/api/admin/analytics/cards", admin_analytics_cards_handler)
    app.router.add_get("/api/admin/analytics/heroes", admin_analytics_heroes_handler)
    app.router.add_get("/api/admin/analytics/retention", admin_analytics_retention_handler)
    app.router.add_get("/api/admin/analytics/onboarding", admin_analytics_onboarding_handler)
    app.router.add_get("/api/admin/analytics/battle-actions", admin_analytics_battle_actions_handler)
    app.router.add_get("/api/admin/analytics/dataset/export", admin_analytics_dataset_export_handler)

    # ── Players Admin: analytics ──

    async def admin_players_overview_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            data = await db.get_admin_players_overview(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("admin_players_overview_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_players_leagues_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            data = await db.get_admin_players_leagues()
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("admin_players_leagues_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_players_activity_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            data = await db.get_admin_players_activity(days=days)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("admin_players_activity_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    # ── Players Admin: list / detail ──

    async def admin_players_list_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            q = request.rel_url.query.get("q", "")
            status = request.rel_url.query.get("status", "all")
            league_raw = request.rel_url.query.get("league")
            league = int(league_raw) if league_raw and league_raw.isdigit() else None
            activity = request.rel_url.query.get("activity", "all")
            limit = min(int(request.rel_url.query.get("limit", "50")), 200)
            offset = max(int(request.rel_url.query.get("offset", "0")), 0)
            data = await db.search_admin_players(
                query=q, status=status, league=league, activity=activity,
                limit=limit, offset=offset,
            )
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("admin_players_list_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_player_detail_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            target_user_id = int(request.match_info["user_id"])
            data = await db.get_admin_player_detail(target_user_id)
            return web.json_response({"status": "ok", "data": data})
        except Exception as e:
            logging.getLogger(__name__).error("admin_player_detail_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    # ── Players Admin: actions ──

    async def admin_player_ban_handler(request: web.Request) -> web.Response:
        admin_id = await require_user_id(request)
        if not await _is_admin_user(db, admin_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            target_user_id = int(request.match_info["user_id"])
            body = await request.json()
            if target_user_id == admin_id:
                if not body.get("confirm_self"):
                    return web.json_response(
                        {"error": "self_ban_requires_confirm", "message": "Set confirm_self=true to ban yourself"},
                        status=400,
                    )
            reason = body.get("reason")
            until_raw = body.get("until")
            until = None
            if until_raw:
                try:
                    until = datetime.fromisoformat(until_raw)
                except Exception:
                    return web.json_response({"error": "invalid_until_date"}, status=400)
            result = await db.admin_ban_user(admin_id, target_user_id, reason=reason, until=until)
            return web.json_response({"status": "ok", "data": result})
        except Exception as e:
            logging.getLogger(__name__).error("admin_player_ban_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_player_unban_handler(request: web.Request) -> web.Response:
        admin_id = await require_user_id(request)
        if not await _is_admin_user(db, admin_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            target_user_id = int(request.match_info["user_id"])
            try:
                body = await request.json()
            except Exception:
                body = {}
            reason = body.get("reason")
            result = await db.admin_unban_user(admin_id, target_user_id, reason=reason)
            return web.json_response({"status": "ok", "data": result})
        except Exception as e:
            logging.getLogger(__name__).error("admin_player_unban_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_player_warn_handler(request: web.Request) -> web.Response:
        admin_id = await require_user_id(request)
        if not await _is_admin_user(db, admin_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            target_user_id = int(request.match_info["user_id"])
            body = await request.json()
            reason = body.get("reason", "")
            result = await db.admin_warn_user(admin_id, target_user_id, reason=str(reason))
            return web.json_response({"status": "ok", "data": result})
        except Exception as e:
            logging.getLogger(__name__).error("admin_player_warn_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_player_note_handler(request: web.Request) -> web.Response:
        admin_id = await require_user_id(request)
        if not await _is_admin_user(db, admin_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            target_user_id = int(request.match_info["user_id"])
            body = await request.json()
            note = body.get("note", "")
            result = await db.admin_note_user(admin_id, target_user_id, note=str(note))
            return web.json_response({"status": "ok", "data": result})
        except Exception as e:
            logging.getLogger(__name__).error("admin_player_note_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_player_update_handler(request: web.Request) -> web.Response:
        admin_id = await require_user_id(request)
        if not await _is_admin_user(db, admin_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            target_user_id = int(request.match_info["user_id"])
            body = await request.json()
            fields = body.get("fields") if isinstance(body.get("fields"), dict) else body
            reason = body.get("reason") if isinstance(body, dict) else None
            result = await db.admin_update_user_account(
                admin_id,
                target_user_id,
                fields=fields,
                reason=reason,
            )
            status = 400 if result.get("error") else 200
            return web.json_response({"status": "ok" if status == 200 else "error", "data": result}, status=status)
        except Exception as e:
            logging.getLogger(__name__).error("admin_player_update_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_player_resource_handler(request: web.Request) -> web.Response:
        admin_id = await require_user_id(request)
        if not await _is_admin_user(db, admin_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            target_user_id = int(request.match_info["user_id"])
            body = await request.json()
            resource = str(body.get("resource", ""))
            amount = float(body.get("amount", 0))
            reason = body.get("reason")
            result = await db.admin_adjust_resource(
                admin_id, target_user_id, resource, amount, reason=reason,
            )
            return web.json_response({"status": "ok", "data": result})
        except Exception as e:
            logging.getLogger(__name__).error("admin_player_resource_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_player_extra_pass_handler(request: web.Request) -> web.Response:
        admin_id = await require_user_id(request)
        if not await _is_admin_user(db, admin_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            target_user_id = int(request.match_info["user_id"])
            body = await request.json()
            mode = str(body.get("mode", ""))
            days_raw = body.get("days")
            days = int(days_raw) if days_raw is not None else None
            reason = body.get("reason")
            result = await db.admin_set_extra_pass(
                admin_id, target_user_id, mode, days=days, reason=reason,
            )
            return web.json_response({"status": "ok", "data": result})
        except Exception as e:
            logging.getLogger(__name__).error("admin_player_extra_pass_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_get("/api/admin/players/overview", admin_players_overview_handler)
    app.router.add_get("/api/admin/players/leagues", admin_players_leagues_handler)
    app.router.add_get("/api/admin/players/activity", admin_players_activity_handler)
    app.router.add_get("/api/admin/players/list", admin_players_list_handler)
    app.router.add_get("/api/admin/players/{user_id}", admin_player_detail_handler)
    app.router.add_post("/api/admin/players/{user_id}/ban", admin_player_ban_handler)
    app.router.add_post("/api/admin/players/{user_id}/unban", admin_player_unban_handler)
    app.router.add_post("/api/admin/players/{user_id}/warn", admin_player_warn_handler)
    app.router.add_post("/api/admin/players/{user_id}/note", admin_player_note_handler)
    app.router.add_post("/api/admin/players/{user_id}/update", admin_player_update_handler)
    app.router.add_post("/api/admin/players/{user_id}/resource", admin_player_resource_handler)
    app.router.add_post("/api/admin/players/{user_id}/extra-pass", admin_player_extra_pass_handler)

    # ── Admin configs: DB-backed game configuration ──

    def _admin_match_mode_catalog() -> list[dict[str, Any]]:
        fixed = [
            {
                "mode_id": "classic",
                "label": "Classic",
                "description": "Бой на трофеи без ExtraArena-модификаторов.",
                "bot_allowed": False,
                "bot_available": False,
            },
            {
                "mode_id": "training",
                "label": "Training",
                "description": "Тренировка против бота.",
                "bot_allowed": True,
                "bot_available": True,
            },
            {
                "mode_id": "friendly",
                "label": "Friendly",
                "description": "Дружеский бой по приглашению.",
                "bot_allowed": False,
                "bot_available": False,
            },
        ]
        return fixed + get_extra_arena_mode_list()

    def _admin_card_payload(card: Any) -> dict[str, Any]:
        def _get(name: str, default: Any = None) -> Any:
            return card.get(name, default) if isinstance(card, dict) else getattr(card, name, default)

        return {
            "id": _get("id"),
            "name": _get("name") or str(_get("id", "")),
            "card_type": _get("card_type", "unit"),
            "rarity": _get("rarity", ""),
        }

    def _empty_admin_squads_analytics(error: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "summary": {
                "total_squads": 0,
                "boosted_squads": 0,
                "total_members": 0,
                "bot_members": 0,
                "total_slots": 0,
                "total_cbrp": 0,
                "total_treasury": 0,
                "new_squads": 0,
            },
            "requests": [],
            "growth": [],
            "top_cbrp": [],
            "top_treasury": [],
            "member_roles": [],
            "cbrp_events": [],
            "activity": [],
            "upgrades": [],
            "purchases": [],
            "snapshots": [],
            "config": dict(SQUAD_SETTINGS_DEFAULTS),
            "errors": {},
        }
        if error:
            payload["errors"]["analytics"] = error
        return payload

    async def _admin_match_modes_payload() -> list[dict[str, Any]]:
        overrides = {r["mode_id"]: r for r in await db.get_match_mode_overrides()}
        modes = []
        for mode in _admin_match_mode_catalog():
            mode_id = str(mode.get("mode_id", ""))
            resolved = resolve_mode_config(mode_id)
            modes.append({
                **mode,
                "available": resolved.available,
                "db_enabled": overrides.get(mode_id, {}).get("enabled", resolved.available),
            })
        return modes

    async def admin_configs_summary_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        errors: dict[str, str] = {}

        async def section(name: str, fallback: Any, loader) -> Any:
            try:
                value = loader()
                if hasattr(value, "__await__"):
                    value = await value
                return value
            except Exception as exc:
                logging.getLogger(__name__).warning("admin configs %s fallback: %s", name, exc, exc_info=True)
                errors[name] = str(exc)
                return fallback

        match_modes = await section("match_modes", [], _admin_match_modes_payload)
        promocodes = await section("promocodes", [], lambda: db.get_promocodes_list())
        reward_tracks = await section("reward_tracks", [], lambda: db.get_all_reward_tracks())
        season = await section(
            "season",
            None,
            lambda: db.get_active_season() if hasattr(db, "get_active_season") else None,
        )
        shop_sets = await section("shop_sets", [], lambda: db.get_shop_sets(active_only=False))
        ruble_products = await section("ruble_products", [], lambda: db.get_ruble_products(active_only=False))
        runtime_config = await section("runtime_config", await _runtime_config_safe(db), lambda: db.get_runtime_config())
        cards = await section(
            "cards",
            [],
            lambda: db.get_cards_list(),
        )
        squads = await section("squads", _empty_admin_squads_analytics(), lambda: db.get_admin_squads_analytics(days=30))

        data = {
            "match_modes": match_modes,
            "promocodes_count": len(promocodes or []),
            "reward_tracks": reward_tracks or [],
            "season": season,
            "shop_sets": shop_sets or [],
            "ruble_products": ruble_products or [],
            "runtime_config": runtime_config,
            "cards": [_admin_card_payload(card) for card in (cards or [])],
            "squads": squads or _empty_admin_squads_analytics(),
            "excluded": [],
            "errors": errors,
        }
        return web.json_response({"status": "ok", "data": _serialize_datetime(data)})


    async def admin_match_modes_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            if request.method == "GET":
                overrides = {r["mode_id"]: r for r in await db.get_match_mode_overrides()}
                modes = []
                for mode in _admin_match_mode_catalog():
                    mode_id = str(mode.get("mode_id", ""))
                    data = dict(mode)
                    data["available"] = resolve_mode_config(mode_id).available
                    data["db_enabled"] = overrides.get(mode_id, {}).get("enabled", data["available"])
                    modes.append(data)
                return web.json_response({"status": "ok", "data": {"modes": modes}})
            body = await request.json()
            mode_id = str(body.get("mode_id", "")).strip()
            enabled = bool(body.get("enabled", True))
            known = {str(m.get("mode_id", "")) for m in _admin_match_mode_catalog()}
            if mode_id not in known:
                return web.json_response({"error": "unknown_mode_id"}, status=400)
            await db.set_match_mode_enabled(mode_id, enabled)
            return web.json_response({"status": "ok", "data": {"mode_id": mode_id, "enabled": enabled}})
        except Exception as e:
            logging.getLogger(__name__).error("admin_match_modes_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_runtime_config_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            if request.method == "GET":
                return web.json_response({"status": "ok", "data": await db.get_runtime_config()})
            body = await request.json()
            config = await db.set_runtime_config(
                maintenance_mode=body.get("maintenance_mode") if "maintenance_mode" in body else None,
                feature_availability=body.get("feature_availability") if "feature_availability" in body else None,
                disabled_card_ids=body.get("disabled_card_ids") if "disabled_card_ids" in body else None,
            )
            return web.json_response({"status": "ok", "data": config})
        except Exception as e:
            logging.getLogger(__name__).error("admin_runtime_config_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    def _parse_admin_season_datetime(value: Any) -> datetime | None:
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
            raise ValueError("invalid_season_datetime") from exc
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _normalize_admin_season_payload(body: dict[str, Any], allowed: set[str]) -> tuple[dict[str, Any] | None, str | None]:
        fields = {key: body[key] for key in allowed if key in body}
        status = fields.get("status")
        if status is not None:
            status = str(status or "").strip().lower()
            if status not in {"draft", "scheduled", "active", "archived"}:
                return None, "invalid_season_status"
            fields["status"] = status
            if status == "active":
                fields["is_active"] = True

        for key in ("start_date", "end_date"):
            if key in fields:
                try:
                    fields[key] = _parse_admin_season_datetime(fields[key])
                except ValueError as exc:
                    return None, str(exc)

        start = fields.get("start_date")
        end = fields.get("end_date")
        if start is not None and end is not None and start >= end:
            return None, "season_dates_invalid"

        try:
            max_stars = int(fields.get("max_stars", DEFAULT_EXTRA_PASS_SEASON["max_stars"]) or DEFAULT_EXTRA_PASS_SEASON["max_stars"])
            pass_end = int(fields.get("pass_end_position", min(40, max_stars)) or min(40, max_stars))
            ultra_start = int(fields.get("ultra_start_position", pass_end + 1) or pass_end + 1)
        except (TypeError, ValueError):
            return None, "invalid_season_progression"
        if not 1 <= max_stars <= 99:
            return None, "invalid_max_stars"
        if not 1 <= pass_end <= max_stars or not 1 <= ultra_start <= max_stars or ultra_start <= pass_end:
            return None, "invalid_season_positions"
        if "max_stars" in fields:
            fields["max_stars"] = max_stars
        if "pass_end_position" in fields:
            fields["pass_end_position"] = pass_end
        if "ultra_start_position" in fields:
            fields["ultra_start_position"] = ultra_start
        if "season_number" in fields:
            try:
                fields["season_number"] = max(1, int(fields["season_number"] or 1))
            except (TypeError, ValueError):
                return None, "invalid_season_number"
        return fields, None

    async def admin_season_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            if request.method == "GET":
                return web.json_response({"status": "ok", "data": _serialize_datetime(await db.get_active_season())})
            body = await request.json()
            allowed = {
                "slug", "name", "subtitle", "description", "start_date", "end_date",
                "is_active", "season_number", "status", "auto_switch", "preset_key",
                "max_stars", "free_track_type", "pass_track_type",
                "ultra_track_type", "pass_end_position", "ultra_start_position", "theme",
            }
            fields, error = _normalize_admin_season_payload(body, allowed)
            if error:
                return web.json_response({"error": error}, status=400)
            season = await db.upsert_active_season(**(fields or {}))
            return web.json_response({"status": "ok", "data": _serialize_datetime(season)})
        except Exception as e:
            logging.getLogger(__name__).error("admin_season_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    def _extra_pass_preset_catalog() -> list[dict[str, str]]:
        return [
            {
                "key": "blank",
                "name": "Пустой сезон",
                "description": "Создает сезон без наград, дорожки заполняются вручную или JSON.",
            },
            {
                "key": "copy_current",
                "name": "Копия текущего сезона",
                "description": "Создает новые track_type и копирует активные награды текущего ExtraPass.",
            },
            {
                "key": "balanced_45",
                "name": "Базовый ExtraPass 45",
                "description": "Копирует системные bp_free, bp_premium и bp_ultra в новые дорожки.",
            },
        ]

    async def _admin_seasons_payload() -> dict[str, Any]:
        seasons = await db.get_seasons()
        tracks = await db.get_all_reward_tracks()
        reward_counts: dict[str, int] = {}
        for track in tracks:
            if track.get("is_active", True) is False:
                continue
            track_type = str(track.get("track_type") or "")
            reward_counts[track_type] = reward_counts.get(track_type, 0) + 1
        normalized = [_normalize_extra_pass_season(season) for season in seasons]
        active = next((season for season in normalized if season.get("is_active")), None)
        drafts = [season for season in normalized if season.get("status") == "draft"]
        scheduled = [season for season in normalized if season.get("status") == "scheduled"]
        return {
            "seasons": normalized,
            "active": active,
            "drafts": drafts,
            "scheduled": scheduled,
            "reward_counts": reward_counts,
            "reward_tracks": tracks,
            "overview": _build_season_schedule_overview(normalized, reward_counts),
            "presets": _extra_pass_preset_catalog(),
        }

    async def admin_seasons_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            return web.json_response({"status": "ok", "data": _serialize_datetime(await _admin_seasons_payload())})
        except Exception as e:
            logging.getLogger(__name__).error("admin_seasons_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_season_presets_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        return web.json_response({"status": "ok", "data": {"presets": _extra_pass_preset_catalog()}})

    async def admin_season_create_draft_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            try:
                body = await request.json()
            except Exception:
                body = {}
            preset_key = str(body.get("preset_key") or "blank")
            if preset_key not in {preset["key"] for preset in _extra_pass_preset_catalog()}:
                return web.json_response({"error": "unknown_preset_key"}, status=400)
            season = await db.create_season_draft(preset_key=preset_key)
            payload = await _admin_seasons_payload()
            return web.json_response({
                "status": "ok",
                "data": _serialize_datetime({"season": season, **payload}),
            })
        except Exception as e:
            logging.getLogger(__name__).error("admin_season_create_draft_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_season_update_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            season_id = int(request.match_info["season_id"])
            body = await request.json()
            allowed = {
                "slug", "name", "subtitle", "description", "season_number", "status",
                "auto_switch", "preset_key", "start_date", "end_date", "is_active",
                "max_stars", "free_track_type", "pass_track_type", "ultra_track_type",
                "pass_end_position", "ultra_start_position", "theme",
            }
            fields, error = _normalize_admin_season_payload(body, allowed)
            if error:
                return web.json_response({"error": error}, status=400)
            season = await db.update_season(season_id, **(fields or {}))
            if season.get("error"):
                return web.json_response({"error": season["error"]}, status=400)
            payload = await _admin_seasons_payload()
            return web.json_response({"status": "ok", "data": _serialize_datetime({"season": season, **payload})})
        except Exception as e:
            logging.getLogger(__name__).error("admin_season_update_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_season_rewards_import_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            season_id = int(request.match_info["season_id"])
            body = await request.json()
            season = await db.get_season_by_id(season_id)
            if not season:
                return web.json_response({"error": "season_not_found"}, status=404)

            payload = body.get("tracks") if isinstance(body, dict) and "tracks" in body else body
            if isinstance(body, dict) and "tracks_json" in body:
                payload = _stdlib_json.loads(str(body.get("tracks_json") or "{}"))
            rows = _normalize_reward_track_import_payload(payload, _normalize_extra_pass_season(season))
            replace = bool(body.get("replace", True)) if isinstance(body, dict) else True
            if replace:
                await db.clear_reward_tracks(_season_track_types(season))
            created = []
            for row in rows:
                result = await db.create_reward_track(**row)
                if result.get("error"):
                    return web.json_response({"error": result["error"], "row": row}, status=400)
                created.append(result)
            payload = await _admin_seasons_payload()
            return web.json_response({
                "status": "ok",
                "data": _serialize_datetime({"imported": len(created), "tiers": created, **payload}),
            })
        except (TypeError, ValueError, _stdlib_json.JSONDecodeError) as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            logging.getLogger(__name__).error("admin_season_rewards_import_handler error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_push_status_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        sender = request.app.get("push_sender")
        configured = bool(getattr(sender, "configured", False)) if sender is not None else False
        init_error = getattr(sender, "init_error", None) if sender is not None else "push_sender_unavailable"
        devices = await db.count_push_devices(platform="android") if hasattr(db, "count_push_devices") else 0
        return web.json_response({
            "status": "ok",
            "data": {
                "configured": configured,
                "init_error": init_error if not configured else None,
                "android_devices": devices,
            },
        })

    async def admin_push_app_update_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        sender = request.app.get("push_sender")
        if sender is None:
            return web.json_response({"error": "push_sender_unavailable"}, status=503)

        try:
            body = await request.json()
        except Exception:
            body = {}

        limit = max(1, min(int(body.get("limit") or 10000), 50000))
        dry_run = bool(body.get("dry_run", False))
        payload = build_android_push_payload(
            "app_update",
            "app_update_required",
            {
                "title": body.get("title"),
                "body": body.get("body"),
                "url": body.get("url"),
            },
        )

        devices = await db.count_push_devices(platform="android") if hasattr(db, "count_push_devices") else 0
        if dry_run:
            return web.json_response({
                "status": "ok",
                "data": {
                    "dry_run": True,
                    "devices": devices,
                    "limit": limit,
                    "payload": {
                        "title": payload.title,
                        "body": payload.body,
                        "data": payload.data,
                    },
                },
            })

        if not getattr(sender, "configured", False):
            return web.json_response(
                {"error": "push_sender_not_configured", "details": getattr(sender, "init_error", None)},
                status=503,
            )

        result = await send_android_broadcast(
            db=db,
            push_sender=sender,
            payload=payload,
            platform="android",
            limit=limit,
        )
        return web.json_response({
            "status": "ok",
            "data": {
                "devices": result.total,
                "sent": result.sent,
                "failed": result.failed,
            },
        })

    app.router.add_get("/api/admin/configs", admin_configs_summary_handler)
    app.router.add_get("/api/admin/runtime-config", admin_runtime_config_handler)
    app.router.add_post("/api/admin/runtime-config", admin_runtime_config_handler)
    app.router.add_get("/api/admin/season", admin_season_handler)
    app.router.add_post("/api/admin/season", admin_season_handler)
    app.router.add_get("/api/admin/seasons", admin_seasons_handler)
    app.router.add_get("/api/admin/seasons/presets", admin_season_presets_handler)
    app.router.add_post("/api/admin/seasons/create-draft", admin_season_create_draft_handler)
    app.router.add_post("/api/admin/seasons/{season_id:\\d+}", admin_season_update_handler)
    app.router.add_post("/api/admin/seasons/{season_id:\\d+}/rewards/import", admin_season_rewards_import_handler)
    app.router.add_get("/api/admin/push/status", admin_push_status_handler)
    app.router.add_post("/api/admin/push/app-update", admin_push_app_update_handler)
    app.router.add_get("/api/admin/match-modes", admin_match_modes_handler)
    app.router.add_post("/api/admin/match-modes", admin_match_modes_handler)

    async def admin_tps_handler(request: web.Request) -> web.Response:
        """Обработчик получения TPS статистики (только для админа)."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        try:
            from tps_monitor import get_tps_monitor

            monitor = get_tps_monitor()
            stats = monitor.get_statistics()

            return web.json_response(stats)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Ошибка получения TPS: {e}", exc_info=True)
            return web.json_response(
                {"error": f"Ошибка получения TPS: {str(e)}"}, status=500
            )

    app.router.add_get("/api/admin/tps", admin_tps_handler)

    async def admin_stars_test_mode_toggle_handler(request: web.Request) -> web.Response:
        """Обработчик переключения тестового режима Stars (только для админа)."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            # Переключаем тестовый режим
            current_mode = app.get("stars_test_mode", False)
            new_mode = not current_mode
            app["stars_test_mode"] = new_mode

            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Тестовый режим Stars переключен администратором {user_id}: {current_mode} -> {new_mode}"
            )

            return web.json_response({
                "success": True,
                "stars_test_mode": new_mode,
                "message": f"Тестовый режим {'включен' if new_mode else 'выключен'}"
            })
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Ошибка переключения тестового режима Stars: {e}", exc_info=True)
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    app.router.add_post("/api/admin/stars-test-mode/toggle", admin_stars_test_mode_toggle_handler)

    async def admin_rewards_tracks_handler(request: web.Request) -> web.Response:
        """GET /api/admin/rewards/tracks — все тиры всех треков."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        try:
            tracks = await db.get_all_reward_tracks()
            return web.json_response({"tracks": tracks})
        except Exception as e:
            logging.getLogger(__name__).error("admin_rewards_tracks error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_rewards_track_create_handler(request: web.Request) -> web.Response:
        """POST /api/admin/rewards/tracks/create — создать новый тир."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            track_type = str(data.get("track_type", ""))
            position = int(data.get("position", 0))
            reward_type = str(data.get("reward_type", "coins"))
            reward_amount = int(data.get("reward_amount", 0))
            reward_meta = data.get("reward_meta")
            extra_pass_required = bool(data.get("extra_pass_required", False))
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        result = await db.create_reward_track(
            track_type=track_type,
            position=position,
            reward_type=reward_type,
            reward_amount=reward_amount,
            reward_meta=reward_meta,
            extra_pass_required=extra_pass_required,
        )
        if result.get("error"):
            return web.json_response({"error": result.get("error")}, status=400)
        return web.json_response({"success": True, "tier": result})

    async def admin_rewards_track_update_handler(request: web.Request) -> web.Response:
        """POST /api/admin/rewards/tracks/update — обновить тир."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            reward_id = int(data.get("id", 0))
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        fields = {}
        for key in ("track_type", "position", "reward_type", "reward_amount", "reward_meta", "extra_pass_required", "is_active"):
            if key in data:
                fields[key] = data[key]

        result = await db.update_reward_track(reward_id, **fields)
        if result.get("error"):
            return web.json_response({"error": result.get("error")}, status=400)
        return web.json_response({"success": True, "tier": result})

    async def admin_rewards_track_delete_handler(request: web.Request) -> web.Response:
        """POST /api/admin/rewards/tracks/delete — мягкое удаление тира."""
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            reward_id = int(data.get("id", 0))
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        await db.delete_reward_track(reward_id)
        return web.json_response({"success": True})

    app.router.add_get("/api/admin/rewards/tracks", admin_rewards_tracks_handler)
    app.router.add_post("/api/admin/rewards/tracks/create", admin_rewards_track_create_handler)
    app.router.add_post("/api/admin/rewards/tracks/update", admin_rewards_track_update_handler)
    app.router.add_post("/api/admin/rewards/tracks/delete", admin_rewards_track_delete_handler)

    async def post_delete_handler(request: web.Request) -> web.Response:
        """Обработчик удаления поста (только для админа)."""
        user_id = await require_user_id(request)
        if not await db.is_admin(user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            post_id = int(data.get("post_id"))

            if not post_id:
                return web.json_response({"error": "post_id_required"}, status=400)

            result = await db.delete_community_post(post_id, user_id)

            if not result["success"]:
                error_messages = {
                    "admin_only": "Только администратор может удалять посты",
                    "post_not_found": "Пост не найден"
                }
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": error_messages.get(result.get("error"), "Ошибка удаления поста")
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка удаления поста для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def post_like_handler(request: web.Request) -> web.Response:
        """Обработчик лайка/дизлайка поста."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            post_id = int(data.get("post_id"))

            if not post_id:
                return web.json_response({"error": "post_id_required"}, status=400)

            result = await db.toggle_post_like(post_id, user_id)

            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown"),
                    "message": "Ошибка обработки лайка"
                }, status=400)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка обработки лайка для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def rewards_track_handler(request: web.Request) -> web.Response:
        """GET /api/rewards/track/{track_type} — список тиров с claimed/available/locked."""
        import json as _json

        user_id = await require_user_id(request)

        track_type = request.match_info.get("track_type", "")

        try:
            season = await db.get_active_season() if hasattr(db, "get_active_season") else None
            if not _reward_track_allowed(track_type, season):
                return web.json_response({"error": "invalid_track_type"}, status=400)

            tracks = await db.get_reward_tracks(track_type)
            profile = await db.get_user_profile(user_id)
            if not profile:
                return web.json_response({"error": "user_not_found"}, status=404)

            extra_pass = profile.get("extra_pass", "inactive")
            pass_access = _extra_pass_access(extra_pass)
            has_extra_pass = bool(pass_access["has_extra_pass"])
            has_ultra = bool(pass_access["has_ultra"])
            user_trophies = profile.get("trophies", 0)
            user_stars = profile.get("stars", 0)
            required_value = user_trophies if track_type == "glory" else user_stars

            claimed_set = await db.get_claimed_rewards(user_id, track_type)

            tiers_by_position: dict[int, list[dict[str, Any]]] = {}
            for t in tracks:
                pos = t["position"]
                reward_info = {
                    "reward_type": t["reward_type"],
                    "reward_amount": t["reward_amount"],
                    "reward_meta": t["reward_meta"],
                }
                if pos not in tiers_by_position:
                    tiers_by_position[pos] = []
                tiers_by_position[pos].append(reward_info)

            tiers_response = []
            for pos in sorted(tiers_by_position):
                entry = tracks[0]  # use any entry at this position for meta
                for t in tracks:
                    if t["position"] == pos:
                        entry = t
                        break

                claimed = pos in claimed_set
                ep_required = entry.get("extra_pass_required", False)
                locked = required_value < pos
                if ep_required and not _reward_track_unlocked_for_type(track_type, pass_access, season):
                    locked = True
                available = not claimed and not locked

                tiers_response.append({
                    "position": pos,
                    "rewards": tiers_by_position[pos],
                    "claimed": claimed,
                    "available": available,
                    "locked": locked,
                    "extra_pass_required": ep_required,
                })

            return web.json_response({
                "track_type": track_type,
                "required_value": required_value,
                "extra_pass": extra_pass,
                "has_extra_pass": has_extra_pass,
                "has_ultra": has_ultra,
                "tiers": tiers_response,
            })
        except Exception as e:
            logging.getLogger(__name__).error("reward_track error: %s", e, exc_info=True)
            return web.json_response({"error": "internal_error", "message": str(e)}, status=500)

    async def rewards_extra_pass_handler(request: web.Request) -> web.Response:
        """GET /api/rewards/extra-pass — сезон, дорожки и статусы ExtraPass одним payload."""
        user_id = await require_user_id(request)

        try:
            profile = await db.get_user_profile(user_id)
            if not profile:
                return web.json_response({"error": "user_not_found"}, status=404)

            season = await db.get_active_season() if hasattr(db, "get_active_season") else None
            normalized_season = _normalize_extra_pass_season(season)
            track_defs = _extra_pass_track_defs(normalized_season)
            track_types = list(dict.fromkeys(track["track_type"] for track in track_defs))

            tracks_by_type = {}
            claimed_by_type = {}
            for track_type in track_types:
                tracks_by_type[track_type] = await db.get_reward_tracks(track_type)
                claimed_by_type[track_type] = await db.get_claimed_rewards(user_id, track_type)

            payload = _build_extra_pass_payload(
                profile=dict(profile),
                season=season,
                tracks_by_type=tracks_by_type,
                claimed_by_type=claimed_by_type,
            )
            return web.json_response(payload)
        except Exception as e:
            logging.getLogger(__name__).error("rewards_extra_pass error: %s", e, exc_info=True)
            return web.json_response({"error": "internal_error", "message": str(e)}, status=500)

    async def rewards_claim_handler(request: web.Request) -> web.Response:
        """POST /api/rewards/claim — запросить награду за конкретную позицию."""
        import json as _json

        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            track_type = str(data.get("track_type", ""))
            position = int(data.get("position", 0))
        except (TypeError, ValueError):
            return web.json_response({"error": "invalid_body"}, status=400)

        try:
            season = await db.get_active_season() if hasattr(db, "get_active_season") else None
            if not _reward_track_allowed(track_type, season):
                return web.json_response({"error": "invalid_track_type"}, status=400)

            entries = await db.get_reward_track_entries(track_type, position)
            if not entries:
                return web.json_response({"error": "tier_not_found"}, status=404)

            profile = await db.get_user_profile(user_id)
            if not profile:
                return web.json_response({"error": "user_not_found"}, status=404)

            extra_pass = profile.get("extra_pass", "inactive")
            pass_access = _extra_pass_access(extra_pass)
            user_trophies = profile.get("trophies", 0)
            user_stars = profile.get("stars", 0)
            required_value = user_trophies if track_type == "glory" else user_stars

            if required_value < position:
                return web.json_response({"error": "tier_locked", "message": "Недостаточно прогресса"}, status=400)

            ep_required = any(e.get("extra_pass_required", False) for e in entries)
            if ep_required and not _reward_track_unlocked_for_type(track_type, pass_access, season):
                track_def = _reward_track_def_for_type(track_type, season)
                message = "Нужен ExtraPass Ultra" if (track_def or {}).get("access") == "ultra" or track_type == "bp_ultra" else "Нужен ExtraPass"
                return web.json_response({"error": "extra_pass_required", "message": message}, status=400)

            # Mark claimed first to prevent race conditions
            await db.claim_reward(user_id, track_type, position)

            granted = []
            for entry in entries:
                rtype = entry["reward_type"]
                ramount = entry["reward_amount"]
                rmeta = entry.get("reward_meta")

                if rtype == "coins":
                    await db.update_user_coins(user_id, ramount)
                    granted.append({"reward_type": "coins", "reward_amount": ramount})
                    await _track_economy_safe(db, user_id=user_id, event_type="earn",
                        resource="coins", amount=ramount, source="reward_track",
                        metadata={"track_type": track_type, "position": position})

                elif rtype == "gems":
                    await db.add_gems(user_id, ramount)
                    granted.append({"reward_type": "gems", "reward_amount": ramount})
                    await _track_economy_safe(db, user_id=user_id, event_type="earn",
                        resource="gems", amount=ramount, source="reward_track",
                        metadata={"track_type": track_type, "position": position})

                elif rtype == "keys":
                    await db.increment_user_keys(user_id, ramount)
                    granted.append({"reward_type": "keys", "reward_amount": ramount})
                    await _track_economy_safe(db, user_id=user_id, event_type="earn",
                        resource="keys", amount=ramount, source="reward_track",
                        metadata={"track_type": track_type, "position": position})

                elif rtype == "card":
                    rarities = ["common", "rare"]
                    if isinstance(rmeta, dict) and "rarity" in rmeta:
                        rarities = rmeta["rarity"]

                    cards = await db.get_random_cards_by_rarities(rarities, limit=1)
                    if cards:
                        card = cards[0]
                        await db.add_card_to_user(user_id, card["id"])
                        granted.append({
                            "reward_type": "card",
                            "reward_amount": 1,
                            "card_id": card["id"],
                            "card_name": card.get("name", ""),
                        })
                        await _track_economy_safe(db, user_id=user_id, event_type="earn",
                            resource="card", amount=1, source="reward_track",
                            metadata={"track_type": track_type, "position": position,
                                      "card_id": card["id"], "card_name": card.get("name")})
                    else:
                        fallback_coins = 100
                        await db.update_user_coins(user_id, fallback_coins)
                        granted.append({
                            "reward_type": "coins",
                            "reward_amount": fallback_coins,
                            "fallback_for": "card",
                        })
                        await _track_economy_safe(db, user_id=user_id, event_type="earn",
                            resource="coins", amount=fallback_coins, source="reward_track",
                            metadata={"track_type": track_type, "position": position, "fallback_for": "card"})

            logging.getLogger(__name__).info(
                "Reward claimed: user=%s track=%s pos=%s granted=%s",
                user_id, track_type, position, _json.dumps(granted),
            )

            return web.json_response({
                "success": True,
                "track_type": track_type,
                "position": position,
                "granted": granted,
            })
        except Exception as e:
            logging.getLogger(__name__).error("claim_reward error: %s", e, exc_info=True)
            return web.json_response({"error": "internal_error", "message": str(e)}, status=500)

    app.router.add_get("/api/rewards/extra-pass", rewards_extra_pass_handler)
    app.router.add_get("/api/rewards/track/{track_type}", rewards_track_handler)
    app.router.add_post("/api/rewards/claim", rewards_claim_handler)

    async def yookassa_webhook_handler(request: web.Request) -> web.Response:
        """Обработчик вебхуков от YooKassa."""
        import json
        import logging
        webhook_logger = logging.getLogger(__name__)

        try:
            raw_data = await request.read()
            data = await request.json() if raw_data else {}

            webhook_logger.info("WEBHOOK raw event=%s", data.get("event", "unknown"))

            payment_service = request.app.get("payment_service")

            if not payment_service:
                webhook_logger.error("Payment service не настроен")
                return web.json_response(
                    {"error": "payment_service_not_configured",
                     "message": "Платежный сервис не настроен. Проверьте настройки YooKassa."},
                    status=503,
                )

            webhook_data = payment_service.parse_webhook(data)

            if not webhook_data:
                webhook_logger.warning("Не удалось распарсить вебхук: %s",
                                       json.dumps(data, ensure_ascii=False, default=str)[:500])
                return web.json_response(
                    {"error": "invalid_webhook"}, status=400
                )

            event = webhook_data.get("event")
            payment_id = webhook_data.get("payment_id")
            status = webhook_data.get("status")
            paid = webhook_data.get("paid")
            metadata = webhook_data.get("metadata", {})

            webhook_logger.info(
                "WEBHOOK parsed: event=%s payment_id=%s status=%s paid=%s metadata_keys=%s",
                event, payment_id, status, paid,
                list(metadata.keys())[:10] if isinstance(metadata, dict) else [],
            )

            if not payment_id or not status:
                webhook_logger.warning("Отсутствуют обязательные данные в вебхуке: %s", webhook_data)
                return web.json_response(
                    {"error": "missing_payment_data"}, status=400
                )

            payment_record = await db.get_payment_by_id(payment_id)
            record_found = bool(payment_record)

            if not payment_record:
                webhook_logger.warning(
                    "WEBHOOK rejected unknown payment_id=%s. Rewards are granted only for server-created payments.",
                    payment_id,
                )
                return web.json_response({"status": "ignored", "reason": "unknown_payment"}, status=202)

            webhook_logger.info(
                "WEBHOOK record: found=%s rewards_processed=%s",
                record_found,
                payment_record.get("rewards_processed") if payment_record else "N/A",
            )

            verified = await asyncio.to_thread(payment_service.get_payment_status, payment_id)
            if not verified.get("success"):
                webhook_logger.warning(
                    "WEBHOOK payment verification failed for %s: %s",
                    payment_id,
                    verified.get("error"),
                )
                return web.json_response({"status": "verification_failed"}, status=202)

            verified_status = verified.get("status") or status
            verified_paid = bool(verified.get("paid"))
            try:
                verified_amount = float(verified.get("amount") or 0)
                record_amount = float(payment_record.get("amount") or 0)
            except (TypeError, ValueError):
                verified_amount = record_amount = -1.0
            verified_currency = str(verified.get("currency") or "").upper()
            record_currency = str(payment_record.get("currency") or "").upper()
            if abs(verified_amount - record_amount) > 0.01 or (
                verified_currency and record_currency and verified_currency != record_currency
            ):
                webhook_logger.error(
                    "WEBHOOK amount mismatch for %s: provider %.2f %s, record %.2f %s",
                    payment_id,
                    verified_amount,
                    verified_currency,
                    record_amount,
                    record_currency,
                )
                return web.json_response({"status": "verification_failed", "reason": "amount_mismatch"}, status=202)
            if verified_status != status or verified_paid != bool(paid):
                webhook_logger.warning(
                    "WEBHOOK provider mismatch for %s: webhook status=%s paid=%s, provider status=%s paid=%s",
                    payment_id,
                    status,
                    paid,
                    verified_status,
                    verified_paid,
                )

            await db.update_payment_status(payment_id=payment_id, status=verified_status)

            payment_record = await db.get_payment_by_id(payment_id)

            if event == "payment.succeeded" and verified_status == "succeeded":
                if not verified_paid:
                    webhook_logger.warning(
                        "Платеж %s: provider status=succeeded, but paid=%s. Rewards are not granted.",
                        payment_id,
                        verified_paid,
                    )
                else:
                    processing_result = await process_successful_payment(
                        db=db,
                        payment_id=payment_id,
                        payment_record=payment_record,
                        source="yookassa_webhook",
                        logger=webhook_logger,
                    )
                    if processing_result["status"] == "already_processed":
                        webhook_logger.info("Платеж %s уже был обработан ранее, повтор не требуется", payment_id)
                    elif processing_result["status"] == "missing_payment":
                        webhook_logger.error("Платеж %s пропал во время обработки", payment_id)
                    elif processing_result.get("rewards_text"):
                        webhook_logger.info(
                            "Платеж %s обработан, награды: %s",
                            payment_id,
                            processing_result["rewards_text"],
                        )
                    else:
                        webhook_logger.info("Платеж %s обработан без дополнительных наград", payment_id)
            elif event == "payment.canceled" and verified_status == "canceled":
                webhook_logger.info("Платеж %s отменён", payment_id)

            elif event == "payment.waiting_for_capture":
                webhook_logger.info("Платеж %s ожидает подтверждения", payment_id)

            return web.json_response({"status": "ok"})
        except Exception as e:
            webhook_logger.error(
                "Ошибка обработки вебхука YooKassa: %s", e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def webhook_test_handler(request: web.Request) -> web.Response:
        """GET /api/payments/webhook/test — проверка доступности webhook URL."""
        payment_service = request.app.get("payment_service")
        return web.json_response({
            "ok": True,
            "service": "yookassa-webhook",
            "payment_service_configured": payment_service is not None,
            "webapp_url": request.app.get("webapp_url", ""),
        })

    async def webhook_debug_handler(request: web.Request) -> web.Response:
        """POST /api/payments/webhook/debug — диагностический webhook (не выдаёт награды)."""
        import json
        import logging
        debug_logger = logging.getLogger(__name__)
        try:
            raw_data = await request.read()
            try:
                data = await request.json() if raw_data else {}
            except Exception:
                data = {"raw": raw_data[:500].decode(errors="replace") if raw_data else ""}

            debug_logger.debug("webhook request method=%s headers=%s body=%s",
                              request.method,
                              dict(request.headers),
                              json.dumps(data, ensure_ascii=False, default=str)[:2000])

            payment_service = request.app.get("payment_service")
            parsed = None
            if payment_service and isinstance(data, dict):
                parsed = payment_service.parse_webhook(data)

            return web.json_response({
                "ok": True,
                "received_method": request.method,
                "received_event": data.get("event") if isinstance(data, dict) else "N/A",
                "parsed": parsed is not None,
                "parsed_event": parsed.get("event") if parsed else None,
                "parsed_payment_id": parsed.get("payment_id") if parsed else None,
                "parsed_status": parsed.get("status") if parsed else None,
                "parsed_paid": parsed.get("paid") if parsed else None,
                "payment_service_configured": payment_service is not None,
            })
        except Exception as e:
            debug_logger.error("WEBHOOK_DEBUG error: %s", e, exc_info=True)
            return web.json_response({"ok": False, "error": str(e)}, status=500)

    def _payment_metadata(payment_record: dict[str, Any] | None) -> dict[str, Any]:
        if not payment_record:
            return {}
        metadata = payment_record.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = _stdlib_json.loads(metadata)
            except Exception:
                metadata = {}
        return metadata if isinstance(metadata, dict) else {}

    async def _merge_payment_metadata(payment_id: str, patch: dict[str, Any]) -> None:
        clean_patch = {k: v for k, v in (patch or {}).items() if v is not None and v != ""}
        if not clean_patch:
            return
        await db.execute(
            """
            UPDATE payments
            SET metadata = metadata || $2::jsonb,
                updated_at = NOW()
            WHERE payment_id = $1
            """,
            payment_id,
            _stdlib_json.dumps(clean_patch, ensure_ascii=False),
        )

    async def _rustore_product_id_from_checkout(metadata: dict[str, Any]) -> str:
        product_code = str(metadata.get("product_code") or "").strip()
        if product_code:
            try:
                product = await db.get_ruble_product(product_code)
                resolved = resolve_rustore_product_id(product)
                if resolved:
                    return resolved
            except Exception:
                pass
            return product_code

        item_type = str(metadata.get("item_type") or "").strip()
        package_type = str(metadata.get("package_type") or "").strip()
        if item_type == "gems_package" and package_type:
            return "gems_starter_once" if package_type == "starter_once" else package_type
        if item_type:
            return item_type
        return ""

    async def _verify_and_process_rustore_payment(
        *,
        payment_id: str,
        payment_record: dict[str, Any],
        invoice_id: str,
        source: str,
        logger: logging.Logger,
    ) -> dict[str, Any]:
        rustore_service = app.get("rustore_payment_service")
        if not rustore_service:
            return {
                "success": False,
                "status": payment_record.get("status", "pending"),
                "reason": "rustore_public_token_missing",
                "message": "RuStore Public API token is not configured",
            }

        metadata = _payment_metadata(payment_record)
        rustore_product_id = str(metadata.get("rustore_product_id") or "").strip()
        if not rustore_product_id:
            rustore_product_id = await _rustore_product_id_from_checkout(metadata)
        if not rustore_product_id:
            return {
                "success": False,
                "status": "verification_failed",
                "reason": "rustore_product_id_missing",
                "message": "RuStore product id is missing",
            }

        verification = await asyncio.to_thread(
            rustore_service.verify_invoice,
            invoice_id=str(invoice_id),
            expected_payment_id=payment_id,
            expected_amount_rub=float(payment_record.get("amount") or 0),
            expected_currency=str(payment_record.get("currency") or "RUB"),
            expected_product_id=rustore_product_id,
        )

        if not verification.get("success"):
            logger.warning(
                "RuStore verification failed for %s: %s",
                payment_id,
                verification.get("reason"),
            )
            return verification

        new_status = str(verification.get("status") or payment_record.get("status") or "pending")
        await _merge_payment_metadata(
            payment_id,
            {
                "provider": "rustore",
                "rustore_invoice_id": verification.get("invoice_id") or invoice_id,
                "rustore_purchase_id": verification.get("purchase_id"),
                "rustore_invoice_status": verification.get("invoice_status"),
                "rustore_product_id": rustore_product_id,
                "rustore_sandbox": verification.get("sandbox"),
            },
        )
        if new_status and new_status != payment_record.get("status"):
            await db.update_payment_status(payment_id=payment_id, status=new_status)
            payment_record = await db.get_payment_by_id(payment_id) or payment_record
        else:
            payment_record = await db.get_payment_by_id(payment_id) or payment_record

        rewards_processed = bool(payment_record.get("rewards_processed", False))
        if new_status == "succeeded" and not rewards_processed:
            processing_result = await process_successful_payment(
                db=db,
                payment_id=payment_id,
                payment_record=payment_record,
                source=source,
                logger=logger,
            )
            rewards_processed = processing_result["status"] not in ("missing_payment", "no_rewards")
            if processing_result.get("rewards_text"):
                logger.info(
                    "RuStore payment %s processed, rewards: %s",
                    payment_id,
                    processing_result["rewards_text"],
                )

        return {
            "success": True,
            "payment_id": payment_id,
            "provider": "rustore",
            "status": new_status,
            "paid": new_status == "succeeded",
            "amount": float(payment_record.get("amount") or 0),
            "currency": payment_record.get("currency") or "RUB",
            "rewards_processed": rewards_processed,
            "metadata": _payment_metadata(payment_record),
            "rustore": {
                "invoice_id": verification.get("invoice_id") or invoice_id,
                "purchase_id": verification.get("purchase_id"),
                "invoice_status": verification.get("invoice_status"),
                "product_id": rustore_product_id,
                "sandbox": verification.get("sandbox"),
            },
        }

    async def create_rustore_payment_handler(request: web.Request) -> web.Response:
        """POST /api/payments/rustore/create — создать pending-платёж для Pay SDK."""
        rustore_logger = logging.getLogger(__name__)
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            resolved = await _resolve_ruble_checkout_item(
                db,
                user_id=user_id,
                data=_normalize_checkout_item_data(data),
            )
            if "error" in resolved:
                return web.json_response(
                    {"error": resolved["error"], "message": resolved.get("message", "")},
                    status=resolved.get("status", 400),
                )

            amount = float(resolved["amount_rub"])
            description = str(resolved["item_name"] or "Покупка в ExtraArena")
            metadata = dict(resolved["metadata"])
            rustore_product_id = await _rustore_product_id_from_checkout(metadata)
            if not rustore_product_id:
                return web.json_response(
                    {"error": "rustore_product_id_missing", "message": "Для товара не найден RuStore productId"},
                    status=400,
                )

            payment_id = f"rustore_{uuid.uuid4()}"
            metadata.update({
                "provider": "rustore",
                "rustore_product_id": rustore_product_id,
                "rustore_order_id": payment_id,
                "amount_rub": amount,
            })

            db_result = await db.create_payment(
                user_id=user_id,
                payment_id=payment_id,
                amount=amount,
                currency="RUB",
                description=description,
                metadata=metadata,
                status="pending",
            )
            if not db_result.get("success"):
                rustore_logger.error("Failed to save RuStore payment %s: %s", payment_id, db_result.get("error"))
                return web.json_response({
                    "success": False,
                    "error": "payment_record_not_saved",
                    "message": "Платеж не был создан. Попробуйте еще раз.",
                }, status=500)

            return web.json_response({
                "success": True,
                "provider": "rustore",
                "payment_id": payment_id,
                "order_id": payment_id,
                "product_id": rustore_product_id,
                "quantity": 1,
                "developer_payload": payment_id,
                "app_user_id": str(user_id),
                "amount": amount,
                "currency": "RUB",
                "description": description,
            })
        except Exception as e:
            rustore_logger.error("Ошибка создания RuStore payment для user_id %s: %s", user_id, e, exc_info=True)
            return web.json_response(
                {"error": "internal_server_error", "message": "Не удалось создать платеж RuStore"},
                status=500,
            )

    async def attach_rustore_payment_handler(request: web.Request) -> web.Response:
        """POST /api/payments/rustore/attach — сохранить invoiceId/purchaseId от SDK."""
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            payment_id = str(data.get("payment_id") or data.get("order_id") or "").strip()
            if not payment_id:
                return web.json_response({"error": "payment_id_required"}, status=400)
            payment_record = await db.get_payment_by_id(payment_id)
            if not payment_record:
                return web.json_response({"error": "payment_not_found"}, status=404)
            if int(payment_record["user_id"]) != int(user_id):
                return web.json_response({"error": "access_denied"}, status=403)

            await _merge_payment_metadata(
                payment_id,
                {
                    "provider": "rustore",
                    "rustore_invoice_id": data.get("invoice_id") or data.get("invoiceId"),
                    "rustore_purchase_id": data.get("purchase_id") or data.get("purchaseId"),
                    "rustore_event": data.get("event") or data.get("type"),
                },
            )
            return web.json_response({"success": True, "payment_id": payment_id})
        except Exception as e:
            logging.getLogger(__name__).error("Ошибка attach RuStore payment: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def complete_rustore_payment_handler(request: web.Request) -> web.Response:
        """POST /api/payments/rustore/complete — проверить invoice и выдать товар."""
        complete_logger = logging.getLogger(__name__)
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            payment_id = str(data.get("payment_id") or data.get("order_id") or "").strip()
            invoice_id = str(data.get("invoice_id") or data.get("invoiceId") or "").strip()
            if not payment_id:
                return web.json_response({"error": "payment_id_required"}, status=400)

            payment_record = await db.get_payment_by_id(payment_id)
            if not payment_record:
                return web.json_response({"error": "payment_not_found"}, status=404)
            if int(payment_record["user_id"]) != int(user_id):
                return web.json_response({"error": "access_denied"}, status=403)

            metadata = _payment_metadata(payment_record)
            invoice_id = invoice_id or str(metadata.get("rustore_invoice_id") or "").strip()
            if not invoice_id:
                return web.json_response({"error": "invoice_id_required"}, status=400)

            await _merge_payment_metadata(
                payment_id,
                {
                    "provider": "rustore",
                    "rustore_invoice_id": invoice_id,
                    "rustore_purchase_id": data.get("purchase_id") or data.get("purchaseId"),
                    "rustore_event": data.get("event") or data.get("type") or "complete",
                },
            )
            payment_record = await db.get_payment_by_id(payment_id) or payment_record

            result = await _verify_and_process_rustore_payment(
                payment_id=payment_id,
                payment_record=payment_record,
                invoice_id=invoice_id,
                source="rustore_complete",
                logger=complete_logger,
            )
            if not result.get("success") and result.get("reason") == "rustore_public_token_missing":
                return web.json_response(result, status=503)
            if not result.get("success"):
                return web.json_response(result, status=409)
            return web.json_response(result)
        except Exception as e:
            complete_logger.error("Ошибка complete RuStore payment: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def create_payment_handler(request: web.Request) -> web.Response:
        """Обработчик создания платежа."""
        import logging
        logger = logging.getLogger(__name__)

        user_id = await require_user_id(request)

        payment_service = request.app.get("payment_service")
        if not payment_service:
            import logging
            logger = logging.getLogger(__name__)
            logger.error("Payment service не настроен в create_payment_handler")
            return web.json_response(
                {
                    "error": "payment_service_not_configured",
                    "message": "Платежный сервис не настроен. Проверьте настройки YooKassa в .env файле."
                },
                status=503
            )

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            resolved = await _resolve_ruble_checkout_item(
                db,
                user_id=user_id,
                data=_normalize_checkout_item_data(data),
            )
            if "error" in resolved:
                return web.json_response(
                    {"error": resolved["error"], "message": resolved.get("message", "")},
                    status=resolved.get("status", 400),
                )

            amount = float(resolved["amount_rub"])
            description = str(resolved["item_name"] or "Покупка в ExtraArena")
            metadata = dict(resolved["metadata"])

            # Получаем URL для возврата после оплаты
            return_url = data.get("return_url", request.app.get("webapp_url", "https://t.me/your_bot"))

            # Создаем платеж в YooKassa
            logger.info("Создание платежа: amount=%s, currency=RUB, description=%s", amount, description)
            payment_result = await asyncio.to_thread(
                payment_service.create_payment,
                amount=amount,
                currency="RUB",
                description=description,
                return_url=return_url,
                metadata=metadata,
            )

            logger.info(f"Результат создания платежа: success={payment_result.get('success')}, error={payment_result.get('error')}")

            if not payment_result.get("success"):
                error_msg = payment_result.get("error", "unknown")
                logger.error(f"Ошибка создания платежа в YooKassa: {error_msg}")
                return web.json_response({
                    "success": False,
                    "error": error_msg,
                    "message": f"Ошибка создания платежа: {error_msg}"
                }, status=400)

            payment_id = payment_result.get("payment_id")

            # Сохраняем платеж в БД
            db_result = await db.create_payment(
                user_id=user_id,
                payment_id=payment_id,
                amount=amount,
                currency="RUB",
                description=description,
                metadata=metadata
            )

            if not db_result.get("success"):
                logger.error("Не удалось сохранить платеж %s в БД: %s", payment_id, db_result.get("error"))
                return web.json_response({
                    "success": False,
                    "error": "payment_record_not_saved",
                    "message": "Платеж не был создан. Попробуйте еще раз.",
                }, status=500)

            return web.json_response({
                "success": True,
                "payment_id": payment_id,
                "confirmation_url": payment_result.get("confirmation_url"),
                "status": payment_result.get("status"),
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка создания платежа для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": "Не удалось создать платеж"}, status=500
            )

    async def get_payment_status_handler(request: web.Request) -> web.Response:
        """Обработчик получения статуса платежа с fallback-выдачей наград."""
        import logging
        status_logger = logging.getLogger(__name__)

        user_id = await require_user_id(request)

        payment_id = request.rel_url.query.get("payment_id")
        if not payment_id:
            return web.json_response({"error": "payment_id_required"}, status=400)

        try:
            payment_record = await db.get_payment_by_id(payment_id)
            if not payment_record:
                return web.json_response({"error": "payment_not_found"}, status=404)

            if payment_record["user_id"] != user_id:
                return web.json_response({"error": "access_denied"}, status=403)

            if payment_id.startswith("stars_"):
                return web.json_response({
                    "payment_id": payment_id,
                    "provider": "stars",
                    "status": payment_record["status"],
                    "amount": float(payment_record["amount"]),
                    "currency": payment_record["currency"],
                    "rewards_processed": payment_record.get("rewards_processed", False),
                    "paid": payment_record["status"] == "succeeded",
                })

            if payment_id.startswith("rustore_"):
                metadata = _payment_metadata(payment_record)
                invoice_id = str(
                    request.rel_url.query.get("invoice_id")
                    or metadata.get("rustore_invoice_id")
                    or ""
                ).strip()
                result = {
                    "payment_id": payment_id,
                    "provider": "rustore",
                    "status": payment_record["status"],
                    "paid": payment_record["status"] == "succeeded",
                    "amount": float(payment_record["amount"]),
                    "currency": payment_record["currency"],
                    "rewards_processed": payment_record.get("rewards_processed", False),
                    "metadata": metadata,
                }
                if invoice_id and payment_record["status"] not in {"succeeded", "canceled"}:
                    verified = await _verify_and_process_rustore_payment(
                        payment_id=payment_id,
                        payment_record=payment_record,
                        invoice_id=invoice_id,
                        source="rustore_status_check",
                        logger=status_logger,
                    )
                    if verified.get("success"):
                        return web.json_response(verified)
                    result.update({
                        "verification_failed": True,
                        "reason": verified.get("reason"),
                        "message": verified.get("message"),
                    })
                return web.json_response(result)

            payment_service = request.app.get("payment_service")
            if not payment_service:
                return web.json_response({
                    "payment_id": payment_id,
                    "provider": "yookassa",
                    "status": payment_record["status"],
                    "amount": float(payment_record["amount"]),
                    "currency": payment_record["currency"],
                    "rewards_processed": payment_record.get("rewards_processed", False),
                })

            status_logger.info(
                "STATUS_CHECK: payment_id=%s db_status=%s rewards_processed=%s",
                payment_id, payment_record["status"], payment_record.get("rewards_processed"),
            )

            status_result = await asyncio.to_thread(
                payment_service.get_payment_status, payment_id
            )

            if not status_result.get("success"):
                status_logger.warning("STATUS_CHECK: не удалось получить статус из YooKassa для %s", payment_id)
                return web.json_response({
                    "payment_id": payment_id,
                    "status": payment_record["status"],
                    "amount": float(payment_record["amount"]),
                    "currency": payment_record["currency"],
                    "rewards_processed": payment_record.get("rewards_processed", False),
                })

            yookassa_status = status_result.get("status")
            yookassa_paid = status_result.get("paid")

            status_logger.info(
                "STATUS_CHECK: yookassa_status=%s yookassa_paid=%s for %s",
                yookassa_status, yookassa_paid, payment_id,
            )

            try:
                provider_amount = float(status_result.get("amount") or 0)
                record_amount = float(payment_record.get("amount") or 0)
            except (TypeError, ValueError):
                provider_amount = record_amount = -1.0
            provider_currency = str(status_result.get("currency") or "").upper()
            record_currency = str(payment_record.get("currency") or "").upper()
            if abs(provider_amount - record_amount) > 0.01 or (
                provider_currency and record_currency and provider_currency != record_currency
            ):
                status_logger.error(
                    "STATUS_CHECK amount mismatch for %s: provider %.2f %s, record %.2f %s",
                    payment_id,
                    provider_amount,
                    provider_currency,
                    record_amount,
                    record_currency,
                )
                return web.json_response({
                    "payment_id": payment_id,
                    "status": payment_record["status"],
                    "paid": False,
                    "amount": float(payment_record["amount"]),
                    "currency": payment_record["currency"],
                    "rewards_processed": payment_record.get("rewards_processed", False),
                    "verification_failed": True,
                    "reason": "amount_mismatch",
                })

            if yookassa_status and yookassa_status != payment_record["status"]:
                await db.update_payment_status(payment_id=payment_id, status=yookassa_status)
                payment_record = await db.get_payment_by_id(payment_id)

            rewards_processed = payment_record.get("rewards_processed", False)

            if not rewards_processed and yookassa_status == "succeeded" and yookassa_paid:
                status_logger.info(
                    "STATUS_CHECK: payment %s succeeded in YooKassa but not processed — выдаём награду",
                    payment_id,
                )
                processing_result = await process_successful_payment(
                    db=db,
                    payment_id=payment_id,
                    payment_record=payment_record,
                    source="yookassa_status_check",
                    logger=status_logger,
                )
                rewards_processed = processing_result["status"] not in ("missing_payment",)
                if processing_result.get("rewards_text"):
                    status_logger.info(
                        "STATUS_CHECK: награды выданы для %s: %s",
                        payment_id, processing_result["rewards_text"],
                    )

            result = {
                "payment_id": payment_id,
                "provider": "yookassa",
                "status": yookassa_status or payment_record["status"],
                "paid": yookassa_paid,
                "amount": status_result.get("amount") or float(payment_record["amount"]),
                "currency": status_result.get("currency") or payment_record["currency"],
                "rewards_processed": rewards_processed,
            }
            return web.json_response(result)

        except Exception as e:
            status_logger.error(
                "Ошибка получения статуса платежа %s: %s", payment_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def user_mail_handler(request: web.Request) -> web.Response:
        """Обработчик получения писем пользователя."""
        user_id = await require_user_id(request)

        try:
            category = request.rel_url.query.get("category")
            unread_only = request.rel_url.query.get("unread_only", "false").lower() == "true"
            try:
                limit = int(request.rel_url.query.get("limit", 50))
            except (TypeError, ValueError):
                limit = 50
            limit = max(1, min(limit, 100))

            mail_list = await db.get_user_mail(
                user_id=user_id,
                category=category,
                limit=limit,
                unread_only=unread_only
            )

            # Преобразуем datetime в строки для JSON и добавляем mail_id для совместимости
            from datetime import datetime
            for mail in mail_list:
                if mail.get("created_at"):
                    if isinstance(mail["created_at"], datetime):
                        mail["created_at"] = mail["created_at"].isoformat()
                # Добавляем mail_id для совместимости с фронтендом
                if "id" in mail and "mail_id" not in mail:
                    mail["mail_id"] = mail["id"]
                if "text" in mail:
                    mail.setdefault("content", mail["text"])
                    mail.setdefault("body", mail["text"])
                if mail.get("attachments") is None:
                    mail["attachments"] = {}

            unread_count = await db.get_unread_mail_count(user_id)

            return web.json_response({
                "mail": mail_list,
                "unread_count": unread_count
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения писем для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def mark_mail_read_handler(request: web.Request) -> web.Response:
        """Обработчик отметки письма как прочитанного."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            try:
                mail_id = int(data.get("mail_id"))
            except (TypeError, ValueError):
                return web.json_response({"error": "invalid_mail_id"}, status=400)

            result = await db.mark_mail_as_read(mail_id, user_id)

            if not result["success"]:
                return web.json_response({
                    "success": False,
                    "error": result.get("error", "unknown")
                }, status=400)

            return web.json_response({"success": True})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка отметки письма как прочитанного для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def get_unread_mail_count_handler(request: web.Request) -> web.Response:
        """Обработчик получения количества непрочитанных писем."""
        user_id = await require_user_id(request)

        try:
            unread_count = await db.get_unread_mail_count(user_id)

            return web.json_response({
                "count": unread_count
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка получения количества непрочитанных писем для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    app.router.add_get("/api/community/posts", community_posts_list_handler)
    app.router.add_post("/api/community/posts/create", community_post_create_handler)
    app.router.add_post("/api/community/posts/delete", post_delete_handler)
    app.router.add_post("/api/community/posts/like", post_like_handler)
    app.router.add_get("/api/community/chat/messages", global_chat_messages_handler)
    app.router.add_post("/api/community/chat/send", global_chat_send_handler)

    # ── Community v2 ────────────────────────────────────────────────────────

    async def community_user_role_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        role = await db.get_user_role(user_id)
        clan = await db.get_user_clan(user_id)
        return web.json_response({
            "role": role,
            "is_admin": role == "admin",
            "clan": clan,
            "is_clan_owner": clan is not None and clan.get("member_role") in ("creator", "owner"),
        })

    app.router.add_get("/api/user/role", community_user_role_handler)

    # ── Squads ────────────────────────────────────────────────────────────────

    def _squad_public_payload(clan: dict | None) -> dict | None:
        if not clan:
            return None
        d = dict(clan)
        if not d.get("tag"):
            d["tag"] = f"#{d.get('public_id') or d.get('id')}"
        d["internalId"] = d.get("id")
        d["publicId"] = d.get("public_id") or d.get("id")
        d["boostPublicId"] = d.get("boost_public_id")
        d["displayId"] = d.get("boost_public_id") if d.get("has_boost") and d.get("boost_public_id") else d["publicId"]
        d["members"] = d.get("members_count", d.get("members", 0))
        d["max"] = d.get("max_members", d.get("max", 0))
        d["maxMembers"] = d["max"]
        d["boost"] = bool(d.get("has_boost", d.get("boost", False)))
        d["minTrophies"] = int(d.get("min_trophies", d.get("minTrophies", 0)) or 0)
        d["treasury"] = int(d.get("treasury_tokens", d.get("treasury", 0)) or 0)
        d["myTokens"] = int(d.get("personal_tokens", d.get("myTokens", 0)) or 0)
        d["memberRole"] = d.get("member_role")
        return _serialize_datetime(d)

    def _squad_error_status(error: str) -> int:
        if error in ("already_in_clan", "already_in_squad", "request_already_exists", "already_purchased", "tag_taken"):
            return 409
        if error in (
            "clan_full",
            "no_suitable_squad",
            "not_enough_trophies",
            "creator_must_transfer_or_delete",
            "cannot_kick",
            "cannot_kick_officer",
            "max_officers_reached",
            "max_level_reached",
            "insufficient_treasury",
            "insufficient_personal_tokens",
            "invalid_action",
            "invalid_type",
            "invalid_name",
            "invalid_tag",
        ):
            return 400
        if error in ("clan_not_found", "request_not_found", "reward_not_found", "unknown_upgrade"):
            return 404
        if error in (
            "no_permission",
            "clan_owner_required",
            "extra_pass_required",
            "only_creator_can_promote",
            "only_creator_can_demote",
            "only_creator_can_transfer",
            "only_creator_can_delete",
            "not_in_squad",
        ):
            return 403
        return 400

    async def _user_has_extra_pass(user_id: int) -> bool:
        profile = await db.get_user_profile(user_id)
        mode = str((profile or {}).get("extra_pass") or "").lower()
        return mode in ("active", "ultra")

    async def _player_notify_name(user_id: int) -> str:
        name = await db.fetchval(
            """
            SELECT COALESCE(p.custom_nickname, u.username, u.first_name, 'Игрок')
            FROM users u
            LEFT JOIN profiles p ON p.user_id = u.user_id
            WHERE u.user_id = $1
            """,
            user_id,
        )
        return str(name or "Игрок")

    async def _squad_creator_ids(clan_id: int) -> list[int]:
        rows = await db.fetch(
            """
            SELECT user_id
            FROM clan_members
            WHERE clan_id = $1 AND role IN ('creator', 'owner')
            """,
            clan_id,
        )
        return [int(r["user_id"]) for r in rows]

    async def _enqueue_squad_creator_notification(
        clan_id: int,
        *,
        event_type: str,
        payload: dict[str, Any],
        dedupe_suffix: str,
    ) -> None:
        for creator_id in await _squad_creator_ids(clan_id):
            await db.enqueue_notification(
                creator_id,
                category={
                    "squad_member_role": "squad_member_role",
                    "squad_new_member": "squad_new_member",
                }.get(event_type, "squad_new_member"),
                event_type=event_type,
                payload={**payload, "section": "squads"},
                dedupe_key=f"squad:{event_type}:{clan_id}:{creator_id}:{dedupe_suffix}",
            )

    async def _enqueue_squad_member_notifications(
        clan_id: int,
        member_ids: list[int],
        *,
        category: str,
        event_type: str,
        payload: dict[str, Any],
        dedupe_suffix: str,
    ) -> None:
        for member_id in member_ids:
            await db.enqueue_notification(
                member_id,
                category=category,
                event_type=event_type,
                payload={**payload, "section": "squads"},
                dedupe_key=f"squad:{event_type}:{clan_id}:{member_id}:{dedupe_suffix}",
            )

    async def squads_config_handler(request: web.Request) -> web.Response:
        await require_user_id(request)
        config = await db.get_squad_runtime_config()
        public_config = {
            "creation_policy": config.get("squad_creation_policy", "beta_free"),
            "weekly_cbrp_enabled": bool(config.get("squad_weekly_cbrp_enabled", True)),
            "seasonal_cbrp_enabled": bool(config.get("squad_seasonal_cbrp_enabled", False)),
            "clan_boost_token_multiplier": config.get("squad_clan_boost_token_multiplier", 1.2),
            "creator_passive_tax_pct": config.get("squad_creator_passive_tax_pct", 0.15),
            "weekly_delta_divisor": config.get("squad_weekly_delta_divisor", 25),
            "weekly_personal_tokens_divisor": config.get("squad_weekly_personal_tokens_divisor", 100),
            "weekly_treasury_tokens_divisor": config.get("squad_weekly_treasury_tokens_divisor", 150),
            "seasonal_cbrp_divisor": config.get("squad_seasonal_cbrp_divisor", 50),
            "seasonal_personal_tokens_divisor": config.get("squad_seasonal_personal_tokens_divisor", 200),
            "seasonal_treasury_tokens_divisor": config.get("squad_seasonal_treasury_tokens_divisor", 300),
            "rewards": config.get("squad_rewards", {}),
            "upgrades": config.get("squad_upgrades", {}),
            "personal_rewards": config.get("squad_personal_rewards", []),
        }
        return web.json_response(public_config)

    async def _require_user_squad(user_id: int) -> dict:
        clan = await db.get_user_clan(user_id)
        if not clan:
            raise ValueError("not_in_squad")
        return clan

    async def _require_squad_role(user_id: int, allowed: tuple[str, ...]) -> dict:
        clan = await _require_user_squad(user_id)
        role = clan.get("member_role")
        if role == "owner":
            role = "creator"
        if role not in allowed:
            raise ValueError("no_permission")
        return clan

    async def _moderate_squad_payload(
        user_id: int,
        *,
        text: str,
        image_b64: str | None = None,
        image_mime: str = "image/jpeg",
    ) -> web.Response | None:
        from infrastructure.moderation import check_rate_limit, moderate_content

        rl = await check_rate_limit(db, user_id)
        if not rl["allowed"]:
            return web.json_response({
                "error": "rate_limit_exceeded",
                "message": f"Слишком много отправок. Попробуйте через {rl['retry_after_seconds'] // 60} мин.",
                "retry_after_seconds": rl["retry_after_seconds"],
            }, status=429)

        mod = await moderate_content(text, "SQUAD", image_b64=image_b64, image_mime=image_mime)
        await db.record_submission(user_id, "SQUAD")
        if mod["decision"] == "reject":
            return web.json_response({
                "error": "moderation_rejected",
                "reason": mod.get("reason", "Контент не прошёл модерацию"),
            }, status=422)
        return None

    async def _moderate_squad_batch(
        user_id: int,
        *,
        text: str,
        images: list[dict[str, str]] | None = None,
    ) -> web.Response | None:
        from infrastructure.moderation import check_rate_limit, moderate_content

        rl = await check_rate_limit(db, user_id)
        if not rl["allowed"]:
            return web.json_response({
                "error": "rate_limit_exceeded",
                "message": f"Слишком много отправок. Попробуйте через {rl['retry_after_seconds'] // 60} мин.",
                "retry_after_seconds": rl["retry_after_seconds"],
            }, status=429)

        checks = images or []
        if not checks:
            mod = await moderate_content(text, "SQUAD")
            await db.record_submission(user_id, "SQUAD")
            if mod["decision"] == "reject":
                return web.json_response({
                    "error": "moderation_rejected",
                    "reason": mod.get("reason", "Контент не прошёл модерацию"),
                }, status=422)
            return None

        for image in checks:
            mod = await moderate_content(
                f"{text}\nИзображение: {image.get('kind', '-')}",
                "SQUAD",
                image_b64=image.get("b64"),
                image_mime=image.get("mime", "image/jpeg"),
            )
            if mod["decision"] == "reject":
                await db.record_submission(user_id, "SQUAD")
                return web.json_response({
                    "error": "moderation_rejected",
                    "reason": mod.get("reason", "Контент не прошёл модерацию"),
                }, status=422)

        await db.record_submission(user_id, "SQUAD")
        return None

    async def squads_me_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        clan = await db.get_user_clan(user_id)
        if not clan:
            return web.json_response({"clan": None})
        clan_id = int(clan["id"])
        members = await db.get_clan_members(clan_id)
        activity = await db.get_clan_activity(clan_id, limit=20)
        my_events = await db.get_squad_cbrp_events(user_id=user_id, limit=30)
        squad_events = await db.get_squad_cbrp_events(clan_id=clan_id, limit=20)
        upgrades = await db.get_clan_upgrades(clan_id)
        upgrades["boost"] = 1 if clan.get("has_boost") else 0
        requests = []
        if clan.get("member_role") in ("creator", "owner", "officer"):
            requests = await db.get_join_requests(clan_id)
        notice_count = len(requests) if clan.get("member_role") in ("creator", "owner") else 0
        return web.json_response(_serialize_datetime({
            "clan": _squad_public_payload(clan),
            "members": members,
            "activity": activity,
            "my_events": my_events,
            "squad_events": squad_events,
            "upgrades": upgrades,
            "requests": requests,
            "notice_count": notice_count,
        }))

    async def mobile_squad_personal_cbrp_widget_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        payload = await db.get_mobile_squad_personal_cbrp_widget(user_id)
        return web.json_response(_serialize_datetime(payload))

    async def mobile_squad_owner_overview_widget_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        payload = await db.get_mobile_squad_owner_overview_widget(user_id)
        status = 403 if payload.get("error") == "owner_required" else 200
        return web.json_response(_serialize_datetime(payload), status=status)

    async def mobile_squad_owner_cbrp_widget_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        payload = await db.get_mobile_squad_owner_cbrp_widget(user_id)
        status = 403 if payload.get("error") == "owner_required" else 200
        return web.json_response(_serialize_datetime(payload), status=status)

    async def squads_search_handler(request: web.Request) -> web.Response:
        await require_user_id(request)
        query = request.rel_url.query.get("q") or None
        filter_type = request.rel_url.query.get("filter") or None
        sort = request.rel_url.query.get("sort", "cbrp")
        clans = await db.search_clans(query=query, filter_type=filter_type, sort=sort, limit=30)
        return web.json_response({"clans": [_squad_public_payload(c) for c in clans]})

    async def squads_get_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        clan = await db.resolve_clan_identifier(request.match_info["clan_id"])
        if not clan:
            return web.json_response({"error": "clan_not_found"}, status=404)
        clan_id = int(clan["id"])
        members = await db.get_clan_members(clan_id)
        pending = await db.get_user_pending_request(user_id, clan_id)
        return web.json_response(_serialize_datetime({
            "clan": _squad_public_payload(clan),
            "members": members[:10],
            "pending_request": pending,
        }))

    async def squads_create_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            from infrastructure.clan_config import (
                CLAN_DESC_MAX,
                CLAN_NAME_MAX,
                CLAN_NAME_MIN,
                CLAN_TAG_MAX,
                CLAN_TAG_MIN,
                TAG_PATTERN,
            )
            config = await db.get_squad_runtime_config()
            if config.get("squad_creation_policy") == "extrapass_only" and not await _user_has_extra_pass(user_id):
                return web.json_response({"error": "extra_pass_required"}, status=403)

            name = str(data.get("name") or "").strip()
            tag = str(data.get("tag") or "").strip().upper()
            description = str(data.get("description") or data.get("desc") or "").strip()[:CLAN_DESC_MAX]
            squad_type = str(data.get("type") or "open")
            min_trophies = max(0, int(data.get("min_trophies") or data.get("minTrophies") or 0))
            if not (CLAN_NAME_MIN <= len(name) <= CLAN_NAME_MAX):
                return web.json_response({"error": "invalid_name"}, status=400)
            if not (CLAN_TAG_MIN <= len(tag) <= CLAN_TAG_MAX) or not TAG_PATTERN.match(tag):
                return web.json_response({"error": "invalid_tag"}, status=400)
            if squad_type not in ("open", "closed"):
                return web.json_response({"error": "invalid_type"}, status=400)
            if not await db.is_tag_unique(tag):
                return web.json_response({"error": "tag_taken"}, status=409)
            moderation_error = await _moderate_squad_payload(
                user_id,
                text=(
                    f"Создание сквада\n"
                    f"Название: {name}\n"
                    f"Тег: {tag}\n"
                    f"Описание: {description or '-'}\n"
                    f"Тип: {squad_type}\n"
                    f"Мин. трофеев: {min_trophies}"
                ),
            )
            if moderation_error is not None:
                return moderation_error
            clan = await db.create_clan(
                owner_id=user_id,
                name=name,
                tag=tag,
                description=description,
                clan_type=squad_type,
                min_trophies=min_trophies,
            )
            return web.json_response({"success": True, "clan": _squad_public_payload(clan)})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("squads_create failed: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def squads_join_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            clan = await db.resolve_clan_identifier(data.get("clan_id"))
            if not clan:
                return web.json_response({"error": "clan_not_found"}, status=404)
            clan_id = int(clan["id"])
            profile = await db.get_user_profile(user_id)
            trophies = int((profile or {}).get("trophies") or 0)
            if trophies < int(clan.get("min_trophies") or 0):
                return web.json_response({"error": "not_enough_trophies"}, status=400)
            if int(clan.get("members_count") or 0) >= int(clan.get("max_members") or 0):
                return web.json_response({"error": "clan_full"}, status=400)
            if clan.get("type") == "closed":
                req = await db.create_join_request(clan_id, user_id)
                return web.json_response({"success": True, "status": "pending", "request": _serialize_datetime(req)})
            await db.clan_join(clan_id, user_id)
            joined = await db.get_user_clan(user_id)
            await _enqueue_squad_creator_notification(
                clan_id,
                event_type="squad_new_member",
                payload={
                    "nick": await _player_notify_name(user_id),
                    "squad_name": clan.get("name") or "сквад",
                },
                dedupe_suffix=f"{user_id}:{int(time.time())}",
            )
            return web.json_response({"success": True, "status": "joined", "clan": _squad_public_payload(joined)})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("squads_join failed: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def squads_random_join_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            clan = await db.random_join_squad(user_id)
            clan_id = int(clan.get("id") or 0)
            if clan_id:
                await _enqueue_squad_creator_notification(
                    clan_id,
                    event_type="squad_new_member",
                    payload={
                        "nick": await _player_notify_name(user_id),
                        "squad_name": clan.get("name") or "сквад",
                    },
                    dedupe_suffix=f"{user_id}:{int(time.time())}",
                )
            return web.json_response({"success": True, "clan": _squad_public_payload(clan)})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("squads_random_join failed: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def squads_leave_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        clan = await db.get_user_clan(user_id)
        if not clan:
            return web.json_response({"error": "not_in_squad"}, status=400)
        try:
            await db.clan_leave(int(clan["id"]), user_id)
            return web.json_response({"success": True})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))

    async def squads_settings_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            clan = await _require_squad_role(user_id, ("creator", "officer"))
            data: dict[str, Any] = {}
            pending_images: dict[str, dict[str, Any]] = {}
            if request.content_type.startswith("multipart/"):
                reader = await request.multipart()
                while True:
                    field = await reader.next()
                    if field is None:
                        break
                    if field.name in ("avatar", "banner"):
                        if clan.get("member_role") not in ("creator", "owner"):
                            raise ValueError("only_creator_can_customize")
                        if not field.filename:
                            continue
                        kind = field.name
                        content_type = field.headers.get("Content-Type", "")
                        if content_type not in ("image/png", "image/jpeg", "image/webp"):
                            return web.json_response({"error": "invalid_content_type"}, status=400)
                        raw = await field.read(decode=True)
                        max_size = 6 * 1024 * 1024 if kind == "banner" else 4 * 1024 * 1024
                        if len(raw) > max_size:
                            return web.json_response({"error": "file_too_large"}, status=400)
                        ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(content_type, "jpg")
                        pending_images[kind] = {
                            "raw": raw,
                            "mime": content_type,
                            "ext": ext,
                            "b64": base64.b64encode(raw).decode("ascii"),
                        }
                    else:
                        data[field.name] = await field.text()
            else:
                data = await request.json()
            from infrastructure.clan_config import (
                CLAN_DESC_MAX,
                CLAN_NAME_MAX,
                CLAN_NAME_MIN,
                CLAN_TAG_MAX,
                CLAN_TAG_MIN,
                TAG_PATTERN,
            )
            updates = {}
            if "name" in data:
                name = str(data.get("name") or "").strip()
                if not (CLAN_NAME_MIN <= len(name) <= CLAN_NAME_MAX):
                    raise ValueError("invalid_name")
                updates["name"] = name
            if "tag" in data:
                tag = str(data.get("tag") or "").strip().upper()
                if not (CLAN_TAG_MIN <= len(tag) <= CLAN_TAG_MAX) or not TAG_PATTERN.match(tag):
                    raise ValueError("invalid_tag")
                existing = await db.get_clan_by_tag(tag)
                if existing and int(existing["id"]) != int(clan["id"]):
                    raise ValueError("tag_taken")
                updates["tag"] = tag
            if "description" in data or "desc" in data:
                updates["description"] = str(data.get("description") or data.get("desc") or "").strip()[:CLAN_DESC_MAX]
            if "type" in data:
                squad_type = str(data.get("type") or "open")
                if squad_type not in ("open", "closed"):
                    raise ValueError("invalid_type")
                updates["type"] = squad_type
            if "min_trophies" in data or "minTrophies" in data:
                updates["min_trophies"] = max(0, int(data.get("min_trophies") or data.get("minTrophies") or 0))
            if "avatar_url" in data:
                updates["avatar_url"] = str(data.get("avatar_url") or "").strip()[:500] or None
            if "banner_url" in data:
                updates["banner_url"] = str(data.get("banner_url") or "").strip()[:500] or None
            if updates or pending_images:
                moderation_error = await _moderate_squad_batch(
                    user_id,
                    text=(
                        f"Изменение настроек сквада\n"
                        f"Текущее название: {clan.get('name')}\n"
                        f"Название: {updates.get('name', clan.get('name'))}\n"
                        f"Тег: {updates.get('tag', clan.get('tag'))}\n"
                        f"Описание: {updates.get('description', clan.get('description') or '-')}\n"
                        f"Тип: {updates.get('type', clan.get('type'))}\n"
                        f"Аватар: {'новый файл' if 'avatar' in pending_images else updates.get('avatar_url', clan.get('avatar_url') or '-')}\n"
                        f"Фон: {'новый файл' if 'banner' in pending_images else updates.get('banner_url', clan.get('banner_url') or '-')}"
                    ),
                    images=[
                        {"kind": kind, "b64": image["b64"], "mime": image["mime"]}
                        for kind, image in pending_images.items()
                    ],
                )
                if moderation_error is not None:
                    return moderation_error
            for kind, image in pending_images.items():
                filename = f"{int(clan['id'])}_{kind}_{uuid.uuid4().hex}.{image['ext']}"
                filepath = SQUAD_UPLOADS_DIR / filename
                filepath.write_bytes(image["raw"])
                updates["avatar_url" if kind == "avatar" else "banner_url"] = f"/uploads/squads/{filename}"
            await db.update_clan_settings(int(clan["id"]), **updates)
            updated = await db.get_user_clan(user_id)
            return web.json_response({"success": True, "clan": _squad_public_payload(updated)})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("squads_settings failed: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def squads_members_action_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            clan = await _require_user_squad(user_id)
            data = await request.json()
            action = str(data.get("action") or "").strip()
            target_id = int(data.get("target_user_id") or data.get("user_id") or 0)
            if not target_id:
                raise ValueError("invalid_action")
            clan_id = int(clan["id"])
            if action == "kick":
                await db.kick_member(clan_id, user_id, target_id)
            elif action == "promote":
                await db.promote_member(clan_id, user_id, target_id)
                await _enqueue_squad_creator_notification(
                    clan_id,
                    event_type="squad_member_role",
                    payload={
                        "nick": await _player_notify_name(target_id),
                        "action": "повышен до офицера",
                    },
                    dedupe_suffix=f"promote:{target_id}:{int(time.time())}",
                )
            elif action == "demote":
                await db.demote_member(clan_id, user_id, target_id)
                await _enqueue_squad_creator_notification(
                    clan_id,
                    event_type="squad_member_role",
                    payload={
                        "nick": await _player_notify_name(target_id),
                        "action": "понижен до участника",
                    },
                    dedupe_suffix=f"demote:{target_id}:{int(time.time())}",
                )
            elif action == "transfer":
                if clan.get("member_role") not in ("creator", "owner"):
                    raise ValueError("only_creator_can_transfer")
                await db.transfer_ownership(clan_id, user_id, target_id)
            else:
                raise ValueError("invalid_action")
            return web.json_response({"success": True})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("squads_members_action failed: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def squads_requests_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            clan = await _require_squad_role(user_id, ("creator", "officer"))
            status = request.rel_url.query.get("status", "pending")
            requests = await db.get_join_requests(int(clan["id"]), status=status)
            return web.json_response(_serialize_datetime({"requests": requests}))
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))

    async def squads_request_respond_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            clan = await _require_squad_role(user_id, ("creator", "officer"))
            data = await request.json()
            request_id = int(data.get("request_id") or 0)
            action = str(data.get("action") or "").strip()
            if not request_id or action not in ("accept", "reject"):
                raise ValueError("invalid_action")
            req_row = await db.fetchrow("SELECT clan_id FROM clan_join_requests WHERE id = $1", request_id)
            if not req_row:
                raise ValueError("request_not_found")
            if int(req_row["clan_id"]) != int(clan["id"]):
                raise ValueError("no_permission")
            if action == "accept":
                req = await db.accept_join_request(request_id, user_id)
                await _enqueue_squad_creator_notification(
                    int(req["clan_id"]),
                    event_type="squad_new_member",
                    payload={
                        "nick": await _player_notify_name(int(req["user_id"])),
                        "squad_name": clan.get("name") or "сквад",
                    },
                    dedupe_suffix=f"{int(req['user_id'])}:{int(time.time())}",
                )
            else:
                req = await db.reject_join_request(request_id, user_id)
            return web.json_response({"success": True, "request": _serialize_datetime(req)})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("squads_request_respond failed: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def squads_shop_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            clan = await _require_user_squad(user_id)
            shop = await db.get_squad_shop_state(int(clan["id"]), user_id)
            return web.json_response(_serialize_datetime(shop))
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))

    async def squads_upgrade_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            clan = await _require_squad_role(user_id, ("creator", "officer"))
            data = await request.json()
            upgrade_type = str(data.get("upgrade_type") or data.get("type") or "member_slots")
            result = await db.buy_clan_upgrade(int(clan["id"]), user_id, upgrade_type)
            if upgrade_type == "boost":
                members = await db.get_clan_members(int(clan["id"]))
                await _enqueue_squad_member_notifications(
                    int(clan["id"]),
                    [int(m["user_id"]) for m in members],
                    category="squad_boost",
                    event_type="squad_boost",
                    payload={"squad_name": clan.get("name") or "Сквад"},
                    dedupe_suffix=f"boost:{result.get('level', 1)}",
                )
            return web.json_response(_serialize_datetime({"success": True, **result, "clan": _squad_public_payload(result.get("clan"))}))
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("squads_upgrade failed: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def squads_reward_buy_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            reward_id = str(data.get("reward_id") or "").strip()
            result = await db.buy_squad_personal_reward(user_id, reward_id)
            return web.json_response(_serialize_datetime({"success": True, **result}))
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("squads_reward_buy failed: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def squads_delete_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            clan = await _require_squad_role(user_id, ("creator",))
            members = await db.get_clan_members(int(clan["id"]))
            await db.delete_clan_by_owner(int(clan["id"]), user_id)
            await _enqueue_squad_member_notifications(
                int(clan["id"]),
                [int(m["user_id"]) for m in members],
                category="squad_disbanded",
                event_type="squad_disbanded",
                payload={"squad_name": clan.get("name") or "Сквад"},
                dedupe_suffix="disbanded",
            )
            return web.json_response({"success": True})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("squads_delete failed: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def squads_upload_image_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            clan = await _require_squad_role(user_id, ("creator",))
            reader = await request.multipart()
            field = await reader.next()
            if field is None or field.name != "file":
                return web.json_response({"error": "file_field_missing"}, status=400)
            kind = request.rel_url.query.get("kind", "avatar")
            if kind not in ("avatar", "banner"):
                return web.json_response({"error": "invalid_kind"}, status=400)
            content_type = field.headers.get("Content-Type", "")
            if content_type not in ("image/png", "image/jpeg", "image/webp"):
                return web.json_response({"error": "invalid_content_type"}, status=400)
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(content_type, "jpg")
            data = await field.read(decode=True)
            max_size = 6 * 1024 * 1024 if kind == "banner" else 4 * 1024 * 1024
            if len(data) > max_size:
                return web.json_response({"error": "file_too_large"}, status=400)

            image_b64 = base64.b64encode(data).decode("ascii")
            moderation_error = await _moderate_squad_payload(
                user_id,
                text=(
                    f"Загрузка изображения сквада\n"
                    f"Сквад: {clan.get('name')} [{clan.get('tag') or clan.get('id')}]\n"
                    f"Тип изображения: {kind}"
                ),
                image_b64=image_b64,
                image_mime=content_type,
            )
            if moderation_error is not None:
                return moderation_error

            filename = f"{int(clan['id'])}_{kind}_{uuid.uuid4().hex}.{ext}"
            filepath = SQUAD_UPLOADS_DIR / filename
            filepath.write_bytes(data)
            image_url = f"/uploads/squads/{filename}"
            field_name = "avatar_url" if kind == "avatar" else "banner_url"
            await db.update_clan_settings(int(clan["id"]), **{field_name: image_url})
            updated = await db.get_user_clan(user_id)
            return web.json_response({"success": True, "image_url": image_url, "clan": _squad_public_payload(updated)})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("squads_upload_image failed: %s", e, exc_info=True)
            return web.json_response({"error": "upload_failed", "message": str(e)}, status=500)

    async def squads_image_static_handler(request: web.Request) -> web.Response:
        filename = request.match_info["filename"]
        if ".." in filename or "/" in filename:
            raise web.HTTPForbidden()
        filepath = SQUAD_UPLOADS_DIR / filename
        if not filepath.exists() or not filepath.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(filepath, headers={"Cache-Control": "public, max-age=86400"})

    app.router.add_get("/api/squads/config", squads_config_handler)
    app.router.add_get("/api/squads/me", squads_me_handler)
    app.router.add_get("/api/mobile/squad/personal-cbrp-widget", mobile_squad_personal_cbrp_widget_handler)
    app.router.add_get("/api/mobile/squad/owner-overview-widget", mobile_squad_owner_overview_widget_handler)
    app.router.add_get("/api/mobile/squad/owner-cbrp-widget", mobile_squad_owner_cbrp_widget_handler)
    app.router.add_get("/api/squads/search", squads_search_handler)
    app.router.add_get("/api/squads/shop", squads_shop_handler)
    app.router.add_post("/api/squads/create", squads_create_handler)
    app.router.add_post("/api/squads/join", squads_join_handler)
    app.router.add_post("/api/squads/random-join", squads_random_join_handler)
    app.router.add_post("/api/squads/leave", squads_leave_handler)
    app.router.add_post("/api/squads/settings", squads_settings_handler)
    app.router.add_post("/api/squads/members/action", squads_members_action_handler)
    app.router.add_get("/api/squads/requests", squads_requests_handler)
    app.router.add_post("/api/squads/requests/respond", squads_request_respond_handler)
    app.router.add_post("/api/squads/upgrade", squads_upgrade_handler)
    app.router.add_post("/api/squads/rewards/buy", squads_reward_buy_handler)
    app.router.add_post("/api/squads/upload-image", squads_upload_image_handler)
    app.router.add_post("/api/squads/delete", squads_delete_handler)
    app.router.add_get("/api/squads/{clan_id}", squads_get_handler)
    app.router.add_get("/uploads/squads/{filename}", squads_image_static_handler)

    # ── Admin squads / clans ────────────────────────────────────────────────

    async def admin_squads_analytics_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            days = int(request.rel_url.query.get("days", "30"))
            data = await db.get_admin_squads_analytics(days=days)
            if isinstance(data, dict):
                base = _empty_admin_squads_analytics()
                base.update(data)
                data = base
            return web.json_response({"status": "ok", "data": _serialize_datetime(data)})
        except Exception as e:
            logging.getLogger(__name__).warning("admin_squads_analytics fallback: %s", e, exc_info=True)
            return web.json_response({"status": "ok", "data": _serialize_datetime(_empty_admin_squads_analytics(str(e)))})

    async def admin_squads_list_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            data = await db.search_admin_squads(
                query=request.rel_url.query.get("q") or None,
                filter_type=request.rel_url.query.get("filter", "all"),
                sort=request.rel_url.query.get("sort", "cbrp"),
                limit=int(request.rel_url.query.get("limit", "50")),
                offset=int(request.rel_url.query.get("offset", "0")),
            )
            return web.json_response({"status": "ok", "data": _serialize_datetime(data)})
        except Exception as e:
            logging.getLogger(__name__).error("admin_squads_list error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_squad_detail_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            clan = await db.resolve_clan_identifier(request.match_info["clan_id"])
            if not clan:
                return web.json_response({"error": "clan_not_found"}, status=404)
            data = await db.get_admin_squad_detail(int(clan["id"]))
            status = 404 if data.get("error") else 200
            return web.json_response({"status": "ok" if status == 200 else "error", "data": data}, status=status)
        except Exception as e:
            logging.getLogger(__name__).error("admin_squad_detail error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_squad_create_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            data = await request.json()
            owner_id = int(data.get("owner_id") or 0)
            name = str(data.get("name") or "").strip()
            tag = str(data.get("tag") or "").strip().upper()
            if not owner_id or not name or not tag:
                return web.json_response({"error": "owner_name_tag_required"}, status=400)
            clan = await db.create_clan(
                owner_id=owner_id,
                name=name,
                tag=tag,
                description=str(data.get("description") or ""),
                clan_type=str(data.get("type") or "open"),
                min_trophies=max(0, int(data.get("min_trophies") or 0)),
            )
            await db._log_clan_activity(int(clan["id"]), "admin_create", "Сквад создан администратором", user_id=user_id)
            return web.json_response({"status": "ok", "data": {"clan": _serialize_datetime(clan)}})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("admin_squad_create error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_squad_update_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            clan = await db.resolve_clan_identifier(request.match_info["clan_id"])
            if not clan:
                return web.json_response({"error": "clan_not_found"}, status=404)
            data = await request.json()
            fields = data.get("fields") if isinstance(data.get("fields"), dict) else data
            result = await db.admin_update_squad(
                user_id,
                int(clan["id"]),
                fields=fields,
                reason=data.get("reason") if isinstance(data, dict) else None,
            )
            status = 400 if result.get("error") else 200
            return web.json_response({"status": "ok" if status == 200 else "error", "data": result}, status=status)
        except Exception as e:
            logging.getLogger(__name__).error("admin_squad_update error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_squad_balance_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            clan = await db.resolve_clan_identifier(request.match_info["clan_id"])
            if not clan:
                return web.json_response({"error": "clan_not_found"}, status=404)
            data = await request.json()
            result = await db.admin_adjust_squad_balance(
                user_id,
                int(clan["id"]),
                resource=str(data.get("resource") or ""),
                amount=int(data.get("amount") or 0),
                reason=data.get("reason"),
            )
            status = 400 if result.get("error") else 200
            return web.json_response({"status": "ok" if status == 200 else "error", "data": result}, status=status)
        except Exception as e:
            logging.getLogger(__name__).error("admin_squad_balance error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_squad_member_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            clan = await db.resolve_clan_identifier(request.match_info["clan_id"])
            if not clan:
                return web.json_response({"error": "clan_not_found"}, status=404)
            data = await request.json()
            result = await db.admin_squad_member_action(
                user_id,
                int(clan["id"]),
                action=str(data.get("action") or ""),
                target_user_id=int(data.get("target_user_id") or data.get("user_id") or 0),
                personal_tokens=data.get("personal_tokens"),
            )
            status = 400 if result.get("error") else 200
            return web.json_response({"status": "ok" if status == 200 else "error", "data": result}, status=status)
        except Exception as e:
            logging.getLogger(__name__).error("admin_squad_member error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_squad_request_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            data = await request.json()
            request_id = int(data.get("request_id") or 0)
            action = str(data.get("action") or "").strip()
            if not request_id or action not in ("accept", "reject"):
                return web.json_response({"error": "invalid_action"}, status=400)
            if action == "accept":
                result = await db.accept_join_request(request_id, user_id)
            else:
                result = await db.reject_join_request(request_id, user_id)
            return web.json_response({"status": "ok", "data": {"request": _serialize_datetime(result)}})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("admin_squad_request error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_squad_upgrade_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            clan = await db.resolve_clan_identifier(request.match_info["clan_id"])
            if not clan:
                return web.json_response({"error": "clan_not_found"}, status=404)
            data = await request.json()
            upgrade_type = str(data.get("upgrade_type") or data.get("type") or "")
            mode = str(data.get("mode") or "set")
            if not upgrade_type:
                return web.json_response({"error": "upgrade_type_required"}, status=400)
            if mode == "buy":
                result = await db.buy_clan_upgrade(int(clan["id"]), int(clan["owner_id"]), upgrade_type)
            else:
                level = max(0, int(data.get("level") or 0))
                await db.execute(
                    """
                    INSERT INTO clan_upgrades (clan_id, upgrade_type, level)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (clan_id, upgrade_type) DO UPDATE SET level = EXCLUDED.level
                    """,
                    int(clan["id"]),
                    upgrade_type,
                    level,
                )
                await db._log_clan_activity(int(clan["id"]), "admin_upgrade", f"Админ установил {upgrade_type} = {level}", user_id=user_id)
                result = {"upgrade_type": upgrade_type, "level": level}
            return web.json_response({"status": "ok", "data": _serialize_datetime(result)})
        except ValueError as e:
            err = str(e)
            return web.json_response({"error": err}, status=_squad_error_status(err))
        except Exception as e:
            logging.getLogger(__name__).error("admin_squad_upgrade error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_squad_config_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            if request.method == "GET":
                return web.json_response({"status": "ok", "data": await db.get_squad_runtime_config()})
            data = await request.json()
            key = str(data.get("key") or "").strip()
            if key not in SQUAD_SETTINGS_DEFAULTS:
                return web.json_response({"error": "invalid_setting_key"}, status=400)
            await db.set_game_setting(key, data.get("value"), "Updated from admin squad panel")
            return web.json_response({"status": "ok", "data": await db.get_squad_runtime_config()})
        except Exception as e:
            logging.getLogger(__name__).error("admin_squad_config error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_squad_weekly_process_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            result = await db.process_weekly_squad_cbrp()
            return web.json_response({"status": "ok", "data": _serialize_datetime(result)})
        except Exception as e:
            logging.getLogger(__name__).error("admin_squad_weekly_process error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def admin_squad_delete_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            clan = await db.resolve_clan_identifier(request.match_info["clan_id"])
            if not clan:
                return web.json_response({"error": "clan_not_found"}, status=404)
            await db._log_clan_activity(int(clan["id"]), "admin_delete", "Сквад удален администратором", user_id=user_id)
            deleted = await db.delete_clan(int(clan["id"]))
            return web.json_response({"status": "ok", "data": {"deleted": deleted}})
        except Exception as e:
            logging.getLogger(__name__).error("admin_squad_delete error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_get("/api/admin/squads/analytics", admin_squads_analytics_handler)
    app.router.add_get("/api/admin/squads/list", admin_squads_list_handler)
    app.router.add_post("/api/admin/squads/create", admin_squad_create_handler)
    app.router.add_get("/api/admin/squads/config", admin_squad_config_handler)
    app.router.add_post("/api/admin/squads/config", admin_squad_config_handler)
    app.router.add_post("/api/admin/squads/process-weekly", admin_squad_weekly_process_handler)
    app.router.add_post("/api/admin/squads/requests/respond", admin_squad_request_handler)
    app.router.add_get("/api/admin/squads/{clan_id}", admin_squad_detail_handler)
    app.router.add_post("/api/admin/squads/{clan_id}/update", admin_squad_update_handler)
    app.router.add_post("/api/admin/squads/{clan_id}/balance", admin_squad_balance_handler)
    app.router.add_post("/api/admin/squads/{clan_id}/member", admin_squad_member_handler)
    app.router.add_post("/api/admin/squads/{clan_id}/upgrade", admin_squad_upgrade_handler)
    app.router.add_post("/api/admin/squads/{clan_id}/delete", admin_squad_delete_handler)

    # ── Image upload ─────────────────────────────────────────────────────────

    async def community_upload_image_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            reader = await request.multipart()
            field = await reader.next()
            if field is None or field.name != "file":
                return web.json_response({"error": "file_field_missing"}, status=400)
            content_type = field.headers.get("Content-Type", "")
            if content_type not in ("image/png", "image/jpeg", "image/webp"):
                return web.json_response({"error": "invalid_content_type"}, status=400)
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(content_type, "jpg")
            data = await field.read(decode=True)
            if len(data) > 5 * 1024 * 1024:
                return web.json_response({"error": "file_too_large"}, status=400)
            filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = COMMUNITY_UPLOADS_DIR / filename
            filepath.write_bytes(data)
            return web.json_response({"image_url": f"/uploads/community/{filename}"})
        except Exception as e:
            logging.getLogger(__name__).error("Ошибка загрузки изображения: %s", e, exc_info=True)
            return web.json_response({"error": "upload_failed", "message": str(e)}, status=500)

    async def community_image_static_handler(request: web.Request) -> web.Response:
        filename = request.match_info["filename"]
        if ".." in filename or "/" in filename:
            raise web.HTTPForbidden()
        filepath = COMMUNITY_UPLOADS_DIR / filename
        if not filepath.exists() or not filepath.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(filepath, headers={"Cache-Control": "public, max-age=86400"})

    app.router.add_post("/api/community/upload-image", community_upload_image_handler)
    app.router.add_get("/uploads/community/{filename}", community_image_static_handler)

    # ── News ──────────────────────────────────────────────────────────────────

    async def community_news_list_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        limit = int(request.rel_url.query.get("limit", 30))
        offset = int(request.rel_url.query.get("offset", 0))
        posts = await db.get_news_posts(limit=limit, offset=offset, user_id=user_id)
        return web.json_response({"posts": posts})

    async def community_news_create_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await db.is_admin(user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            data = await request.json()
            title = data.get("title", "").strip()
            content = data.get("content", "").strip()
            if not title or not content:
                return web.json_response({"error": "title_and_content_required"}, status=400)
            poll_data = data.get("poll")
            has_poll = isinstance(poll_data, dict)
            if has_poll:
                poll_question = str(poll_data.get("question") or "").strip()
                poll_options = [
                    option for option in (poll_data.get("options") or [])
                    if str((option or {}).get("text") if isinstance(option, dict) else option).strip()
                ]
                if not poll_question or len(poll_options) < 2:
                    return web.json_response({"error": "invalid_poll"}, status=400)
            result = await db.create_news_post(
                author_id=user_id,
                title=title,
                content=content,
                content_html=data.get("content_html", content),
                tags=data.get("tags", []),
                cover_image_url=data.get("cover_image_url"),
                post_type="poll" if has_poll else data.get("post_type", "news"),
            )
            if not result["success"]:
                return web.json_response(result, status=400)
            # Create poll if provided
            if has_poll and result.get("id"):
                poll_result = await db.create_poll(
                    post_id=result["id"],
                    question=poll_data.get("question", ""),
                    options=poll_data.get("options", []),
                    expires_at=poll_data.get("expires_at", ""),
                )
                if not poll_result.get("success"):
                    try:
                        await db.execute("DELETE FROM community_posts WHERE id = $1", result["id"])
                    except Exception:
                        pass
                    return web.json_response(poll_result, status=400)
                result["poll_id"] = poll_result.get("poll_id")
            return web.json_response(result)
        except Exception as e:
            logging.getLogger(__name__).error("Ошибка создания новости: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def community_news_like_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            post_id = int(data.get("post_id"))
            result = await db.toggle_post_like(post_id, user_id)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def community_poll_vote_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            poll_id = int(data.get("poll_id"))
            option_id = int(data.get("option_id"))
            result = await db.vote_poll(poll_id=poll_id, user_id=user_id, option_id=option_id)
            if result["success"]:
                results = await db.get_poll_results(poll_id=poll_id, user_id=user_id)
                return web.json_response(results)
            return web.json_response(result, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def community_poll_results_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            post_id = int(request.match_info["post_id"])
            poll = await db.fetchrow(
                "SELECT id FROM community_polls WHERE post_id = $1", post_id
            )
            if not poll:
                return web.json_response({"error": "poll_not_found"}, status=404)
            results = await db.get_poll_results(poll_id=poll["id"], user_id=user_id)
            return web.json_response(results)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_get("/api/community/news", community_news_list_handler)
    app.router.add_post("/api/community/news/create", community_news_create_handler)
    app.router.add_post("/api/community/news/like", community_news_like_handler)
    app.router.add_post("/api/community/poll/vote", community_poll_vote_handler)
    app.router.add_get("/api/community/poll/{post_id}", community_poll_results_handler)

    # ── Ideas & Bugs ──────────────────────────────────────────────────────────

    async def community_ideas_list_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        limit = int(request.rel_url.query.get("limit", 30))
        offset = int(request.rel_url.query.get("offset", 0))
        sort_by = request.rel_url.query.get("sort", "votes")
        ideas = await db.get_ideas(limit=limit, offset=offset, sort_by=sort_by, user_id=user_id)
        return web.json_response({"ideas": ideas})

    async def community_ideas_create_handler(request: web.Request) -> web.Response:
        from infrastructure.moderation import moderate_content, check_rate_limit
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            title = data.get("title", "").strip()
            description = data.get("description", "").strip()
            post_type = data.get("type", "idea")  # idea | bug
            if not title or not description:
                return web.json_response({"error": "title_and_description_required"}, status=400)
            if post_type not in ("idea", "bug"):
                return web.json_response({"error": "invalid_type"}, status=400)

            # rate limit
            rl = await check_rate_limit(db, user_id)
            if not rl["allowed"]:
                return web.json_response({
                    "error": "rate_limit_exceeded",
                    "message": f"Слишком много отправок. Попробуйте через {rl['retry_after_seconds'] // 60} мин.",
                    "retry_after_seconds": rl["retry_after_seconds"],
                }, status=429)

            # moderation
            category = "BUG" if post_type == "bug" else "IDEA"
            text = f"{title}\n\n{description}"
            mod = await moderate_content(text, category)
            await db.record_submission(user_id, category)

            if mod["decision"] == "reject":
                return web.json_response({
                    "error": "moderation_rejected",
                    "reason": mod.get("reason", "Контент не прошёл модерацию"),
                }, status=422)

            result = await db.create_idea(
                author_id=user_id,
                title=title,
                description=description,
                post_type=post_type,
                moderation_status="approved",
            )
            return web.json_response(result)
        except Exception as e:
            logging.getLogger(__name__).error("Ошибка создания идеи: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def community_ideas_vote_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            post_id = int(data.get("post_id"))
            vote_type = data.get("vote_type", "up")
            if vote_type not in ("up", "down"):
                return web.json_response({"error": "invalid_vote_type"}, status=400)
            result = await db.vote_idea(post_id=post_id, user_id=user_id, vote_type=vote_type)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def community_ideas_admin_approve_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await db.is_admin(user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            data = await request.json()
            post_id = int(data.get("post_id"))
            result = await db.admin_approve_idea(post_id)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def community_ideas_admin_status_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await db.is_admin(user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        try:
            data = await request.json()
            post_id = int(data.get("post_id"))
            status = data.get("status", "")
            result = await db.update_idea_status(post_id, status)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def community_bugs_list_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await db.is_admin(user_id):
            return web.json_response({"error": "admin_access_required"}, status=403)
        limit = int(request.rel_url.query.get("limit", 50))
        offset = int(request.rel_url.query.get("offset", 0))
        bugs = await db.get_bugs_for_admin(limit=limit, offset=offset)
        return web.json_response({"bugs": bugs})

    app.router.add_get("/api/community/ideas", community_ideas_list_handler)
    app.router.add_post("/api/community/ideas/create", community_ideas_create_handler)
    app.router.add_post("/api/community/ideas/vote", community_ideas_vote_handler)
    app.router.add_post("/api/community/ideas/admin/approve", community_ideas_admin_approve_handler)
    app.router.add_post("/api/community/ideas/admin/status", community_ideas_admin_status_handler)
    app.router.add_get("/api/community/bugs", community_bugs_list_handler)

    # ── Announcements ─────────────────────────────────────────────────────────

    async def community_announcements_list_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        limit = int(request.rel_url.query.get("limit", 20))
        offset = int(request.rel_url.query.get("offset", 0))
        announcements = await db.get_announcements(limit=limit, offset=offset, user_id=user_id)
        return web.json_response({"announcements": announcements})

    async def community_announcements_cost_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            from infrastructure.community_config import calc_announcement_price
            clan = await db.get_user_clan(user_id)
            has_boost = clan.get("has_boost", False) if clan else False
            price = calc_announcement_price(
                text=data.get("text", ""),
                has_image=bool(data.get("has_image", False)),
                duration_key=data.get("duration_key", "1d"),
                is_pinned=bool(data.get("is_pinned", False)),
                has_boost=has_boost,
            )
            # include min required pin price
            active_pin = await db.get_active_pin()
            from infrastructure.community_config import ANNOUNCE_PIN_BASE_COST, ANNOUNCE_PIN_OVERBID_STEP
            if active_pin:
                price["min_pin_price"] = active_pin["pin_price"] + ANNOUNCE_PIN_OVERBID_STEP
            else:
                price["min_pin_price"] = ANNOUNCE_PIN_BASE_COST
            return web.json_response(price)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def community_announcements_create_handler(request: web.Request) -> web.Response:
        from infrastructure.moderation import moderate_content, check_rate_limit
        from infrastructure.community_config import calc_announcement_price, ANNOUNCE_PIN_BASE_COST, ANNOUNCE_PIN_OVERBID_STEP
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            text = data.get("text", "").strip()
            if len(text) < 10:
                return web.json_response({"error": "text_too_short"}, status=400)

            # must be clan owner
            clan = await db.get_user_clan(user_id)
            if not clan or clan.get("member_role") not in ("creator", "owner"):
                return web.json_response({"error": "clan_owner_required"}, status=403)

            has_image = bool(data.get("image_url"))
            duration_key = data.get("duration_key", "1d")
            is_pinned = bool(data.get("is_pinned", False))
            image_url = data.get("image_url")

            # determine pin_price
            active_pin = await db.get_active_pin()
            if is_pinned:
                min_pin = (active_pin["pin_price"] + ANNOUNCE_PIN_OVERBID_STEP) if active_pin else ANNOUNCE_PIN_BASE_COST
                pin_price = int(data.get("pin_price", min_pin))
                if pin_price < min_pin:
                    return web.json_response({
                        "error": "pin_price_too_low",
                        "min_required": min_pin,
                    }, status=400)
            else:
                pin_price = 0

            # calculate cost
            price_info = calc_announcement_price(
                text=text,
                has_image=has_image,
                duration_key=duration_key,
                is_pinned=is_pinned,
                has_boost=clan.get("has_boost", False),
            )
            if is_pinned:
                extra_pin = pin_price - ANNOUNCE_PIN_BASE_COST
                if extra_pin > 0:
                    price_info["total"] += extra_pin
                    price_info["pin_extra"] = pin_price

            # rate limit
            rl = await check_rate_limit(db, user_id)
            if not rl["allowed"]:
                return web.json_response({
                    "error": "rate_limit_exceeded",
                    "message": f"Слишком много отправок. Попробуйте через {rl['retry_after_seconds'] // 60} мин.",
                }, status=429)

            # moderation (before charging gems)
            mod_text = text
            mod = await moderate_content(mod_text, "ANNOUNCEMENT", image_b64=None)
            await db.record_submission(user_id, "ANNOUNCEMENT")

            if mod["decision"] == "reject":
                return web.json_response({
                    "error": "moderation_rejected",
                    "reason": mod.get("reason", "Контент не прошёл модерацию"),
                }, status=422)

            result = await db.create_announcement(
                author_id=user_id,
                clan_id=clan["id"],
                content=text,
                image_url=image_url,
                duration_key=duration_key,
                is_pinned=is_pinned,
                gems_to_pay=price_info["total"],
                pin_price=pin_price,
            )
            return web.json_response(result, status=200 if result["success"] else 400)
        except Exception as e:
            logging.getLogger(__name__).error("Ошибка создания объявления: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def community_announcements_react_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await request.json()
            post_id = int(data.get("post_id"))
            vote_type = data.get("vote_type", "like")
            result = await db.react_announcement(post_id=post_id, user_id=user_id, vote_type=vote_type)
            return web.json_response(result)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def community_pin_info_handler(request: web.Request) -> web.Response:
        from infrastructure.community_config import ANNOUNCE_PIN_BASE_COST, ANNOUNCE_PIN_OVERBID_STEP
        await require_user_id(request)
        active_pin = await db.get_active_pin()
        if active_pin:
            min_price = active_pin["pin_price"] + ANNOUNCE_PIN_OVERBID_STEP
        else:
            min_price = ANNOUNCE_PIN_BASE_COST
        return web.json_response({
            "has_active_pin": active_pin is not None,
            "current_pin_price": active_pin["pin_price"] if active_pin else 0,
            "min_new_pin_price": min_price,
        })

    app.router.add_get("/api/community/announcements", community_announcements_list_handler)
    app.router.add_post("/api/community/announcements/create", community_announcements_create_handler)
    app.router.add_post("/api/community/announcements/cost", community_announcements_cost_handler)
    app.router.add_post("/api/community/announcements/react", community_announcements_react_handler)
    app.router.add_get("/api/community/announcements/pin-info", community_pin_info_handler)
    app.router.add_get("/api/recent-opponents", recent_opponents_handler)
    app.router.add_get("/api/battles/history", battle_history_handler)
    app.router.add_post("/api/friends/online-ping", friend_online_ping_handler)
    app.router.add_post("/api/friends/invite", friend_invite_handler)
    app.router.add_get("/api/friends/invite/status", friend_invite_status_handler)
    app.router.add_post("/api/friends/invite/respond", friend_invite_respond_handler)
    app.router.add_get("/api/friends/invite/pending", friend_invite_pending_handler)
    app.router.add_post("/api/friends/invite/cancel", friend_invite_cancel_handler)
    app.router.add_post("/api/friends/request", friend_request_send_handler)
    app.router.add_get("/api/friends/requests", friend_requests_list_handler)
    app.router.add_post("/api/friends/request/respond", friend_request_respond_handler)
    app.router.add_post("/api/friends/request/cancel", friend_request_cancel_handler)
    app.router.add_get("/api/friends/list", friend_list_handler)
    app.router.add_post("/api/friends/remove", friend_remove_handler)
    async def payment_config_handler(_: web.Request) -> web.Response:
        """Отдать конфигурацию платежей (курсы и коэффициенты)."""
        provider_order = [
            part.strip()
            for part in str(app.get("payment_provider_order", "yookassa,rustore,stars")).split(",")
            if part.strip()
        ]
        return web.json_response({
            "stars_rate_rub": app.get("stars_rate_rub", 1.5),
            "stars_markup": app.get("stars_markup", 1.2),
            "stars_test_mode": app.get("stars_test_mode", False),
            "payment_provider_order": provider_order,
            "providers": {
                "yookassa": {
                    "enabled": app.get("payment_service") is not None,
                    "label": "YooKassa",
                },
                "rustore": {
                    "enabled": True,
                    "configured": app.get("rustore_payment_service") is not None,
                    "label": "RuStore Pay",
                    "app_url": app.get("rustore_app_url"),
                    "console_app_id": app.get("rustore_console_app_id"),
                },
                "stars": {
                    "enabled": True,
                    "label": "Telegram Stars",
                },
            },
        })

    # Обработчик создания инвойса через Telegram Stars
    async def create_stars_invoice_handler(request: web.Request) -> web.Response:
        """Обработчик создания инвойса через Telegram Stars."""
        import logging
        logger = logging.getLogger(__name__)

        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            resolved = await _resolve_ruble_checkout_item(
                db,
                user_id=user_id,
                data=_normalize_checkout_item_data(data),
            )
            if "error" in resolved:
                return web.json_response(
                    {"error": resolved["error"], "message": resolved.get("message", "")},
                    status=resolved.get("status", 400),
                )

            amount_rub = float(resolved["amount_rub"])
            item_name = str(resolved["item_name"] or "Покупка в ExtraArena")
            description = item_name
            metadata = dict(resolved["metadata"])

            # Логируем metadata для отладки
            logger.info(
                f"Создание платежа Stars: user_id={user_id}, item_type={metadata.get('item_type')}, "
                f"item_name={item_name}, description={description}"
            )

            stars_rate_rub = float(app.get("stars_rate_rub", 1.5))
            stars_markup = float(app.get("stars_markup", 1.2))
            stars_test_mode = bool(app.get("stars_test_mode", False))
            is_admin_user = await _is_admin_user(db, user_id)
            stars_amount = max(1, math.ceil((amount_rub / stars_rate_rub) * stars_markup))
            if stars_test_mode and is_admin_user:
                logger.info(
                    "Stars test mode активен: пользователь %s (admin) платит фиксированные 1 ⭐",
                    user_id,
                )
                stars_amount = 1

            # Создаем уникальный invoice_payload для идентификации платежа
            import uuid
            invoice_payload = str(uuid.uuid4())

            # Сохраняем информацию о платеже в БД до отправки инвойса
            payment_id = f"stars_{invoice_payload}"
            db_result = await db.create_payment(
                user_id=user_id,
                payment_id=payment_id,
                amount=float(stars_amount),
                currency="XTR",  # Telegram Stars
                description=description,
                metadata={
                    **metadata,
                    "amount_rub": amount_rub,
                    "stars_rate_rub": stars_rate_rub,
                    "stars_markup": stars_markup,
                    "stars_amount": stars_amount,
                    "stars_test_mode": stars_test_mode,
                    "is_admin_test_purchase": stars_test_mode and is_admin_user,
                },
                status="pending"  # Платеж ожидает подтверждения
            )

            if not db_result.get("success"):
                logger.error("Не удалось сохранить платеж Stars %s в БД: %s", payment_id, db_result.get("error"))
                return web.json_response({
                    "success": False,
                    "error": "payment_record_not_saved",
                    "message": "Инвойс не был создан. Попробуйте еще раз.",
                }, status=500)

            # Отправляем инвойс через Telegram Bot API
            from aiogram import Bot
            bot = Bot(token=bot_token)

            try:
                # Формируем детали счета с пояснениями для пользователя
                detailed_message = (
                    "<b>🧾 Счет на оплату</b>\n"
                    f"Товар: {item_name}\n"
                    f"Сумма: {amount_rub:.2f} ₽ → {stars_amount} ⭐\n"
                    f"ID платежа: <code>{payment_id}</code>\n"
                    "После оплаты вернитесь в игру - награды выдаются автоматически."
                )
                await bot.send_message(user_id, detailed_message, parse_mode="HTML")

                # Формируем параметры для sendInvoice
                invoice_title = item_name[:32]  # Максимум 32 символа
                invoice_description = (
                    f"{item_name} • {amount_rub:.2f} ₽ • {stars_amount} ⭐\n"
                    "После оплаты вернитесь в ExtraArena."
                )[:255]

                # Создаем LabeledPrice для инвойса
                from aiogram.types import LabeledPrice
                prices = [LabeledPrice(label=invoice_title, amount=stars_amount)]

                # Отправляем инвойс
                sent_message = await bot.send_invoice(
                    chat_id=user_id,
                    title=invoice_title,
                    description=invoice_description,
                    payload=invoice_payload,
                    provider_token=None,  # Для Stars не нужен provider_token
                    currency="XTR",  # Telegram Stars
                    prices=prices,
                    start_parameter=invoice_payload[:64],  # Максимум 64 символа
                )

                logger.info(
                    f"Инвойс Stars отправлен пользователю {user_id}, invoice_payload={invoice_payload}, "
                    f"amount_rub={amount_rub}, stars={stars_amount}, "
                    f"item_type={metadata.get('item_type')}, item_name={item_name}, "
                    f"description={description}, payment_id={payment_id}"
                )

                return web.json_response({
                    "success": True,
                    "payment_id": payment_id,
                    "invoice_payload": invoice_payload,
                    "message_id": sent_message.message_id,
                    "stars_amount": stars_amount,
                    "amount_rub": amount_rub,
                    "stars_rate_rub": stars_rate_rub,
                    "stars_markup": stars_markup,
                    "stars_test_mode": stars_test_mode,
                    "is_admin_test_purchase": stars_test_mode and is_admin_user,
                })
            except Exception as e:
                logger.error(f"Ошибка отправки инвойса Stars пользователю {user_id}: {e}", exc_info=True)
                return web.json_response({
                    "success": False,
                    "error": "invoice_send_failed",
                    "message": "Не удалось отправить инвойс",
                }, status=500)
            finally:
                await bot.session.close()
        except Exception as e:
            logger.error(
                "Ошибка создания инвойса Stars для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": "Не удалось создать инвойс"}, status=500
            )

    async def payment_history_start_handler(request: web.Request) -> web.Response:
        """POST /api/payments/history/start — возвращает подписанный history_url."""
        user_id = await require_user_id(request)

        try:
            if not await db.has_any_purchase(user_id):
                return web.json_response({
                    "success": True,
                    "has_purchases": False,
                    "history_url": None,
                })

            secret = request.app["bot_token"]
            webapp_url = request.app.get("webapp_url", "https://t.me/your_bot")

            payload = {
                "user_id": user_id,
                "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
                "iat": int(datetime.now(timezone.utc).timestamp()),
                "jti": str(uuid.uuid4()),
            }

            payload_json = _stdlib_json.dumps(payload, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
            payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode()
            signature = hmac.new(secret.encode(), payload_b64.encode(), "sha256").hexdigest()
            token = f"{payload_b64}.{signature}"

            extra_shop_url = request.app.get("extra_shop_url", request.app.get("webapp_url", ""))
            history_url = f"{extra_shop_url.rstrip('/')}/extraShop?history={token}"

            return web.json_response({
                "success": True,
                "has_purchases": True,
                "history_url": history_url,
            })
        except Exception as e:
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def payment_history_handler(request: web.Request) -> web.Response:
        """GET /api/payments/history?token=... — возвращает историю покупок пользователя."""
        try:
            token = request.rel_url.query.get("token", "")
            if not token:
                return web.json_response({"error": "missing_token"}, status=400)

            secret = request.app["bot_token"]
            parts = token.rsplit(".", 1)
            if len(parts) != 2:
                return web.json_response({"error": "invalid_token_format"}, status=400)

            payload_b64, provided_sig = parts
            expected_sig = hmac.new(secret.encode(), payload_b64.encode(), "sha256").hexdigest()
            if not hmac.compare_digest(expected_sig, provided_sig):
                return web.json_response({"error": "invalid_signature"}, status=400)

            try:
                padded = payload_b64 + "=" * ((4 - len(payload_b64) % 4) % 4)
                payload_json = base64.urlsafe_b64decode(padded).decode()
                payload = _stdlib_json.loads(payload_json)
            except Exception:
                return web.json_response({"error": "invalid_token_payload"}, status=400)

            exp = int(payload.get("exp", 0))
            if time.time() > exp:
                return web.json_response({"error": "token_expired"}, status=400)

            user_id = int(payload.get("user_id", 0))
            if not user_id:
                return web.json_response({"error": "missing_user_id"}, status=400)

            history = await db.get_user_payment_history(user_id, limit=50)

            result = []
            for p in history:
                result.append({
                    "payment_id": p["payment_id"],
                    "amount": float(p["amount"]),
                    "currency": p["currency"],
                    "description": p["description"],
                    "metadata": p["metadata"] if isinstance(p["metadata"], dict) else
                                (_stdlib_json.loads(p["metadata"]) if isinstance(p["metadata"], str) else {}),
                    "status": p["status"],
                    "rewards_processed": p["rewards_processed"],
                    "created_at": p["created_at"].isoformat() if hasattr(p["created_at"], "isoformat") else str(p["created_at"]),
                })

            return web.json_response({"success": True, "history": result})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка получения истории покупок: %s", e, exc_info=True)
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def payment_modal_shown_handler(request: web.Request) -> web.Response:
        """POST /api/payments/modal-shown — помечает, что модалка показана в UI."""
        user_id = await require_user_id(request)

        try:
            data = await request.json()
            payment_id = str(data.get("payment_id", ""))
            if not payment_id:
                return web.json_response({"error": "missing_payment_id"}, status=400)

            marked = await db.mark_payment_modal_shown(payment_id, user_id)
            return web.json_response({"success": True, "marked": marked})
        except Exception as e:
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def checkout_session_status_handler(request: web.Request) -> web.Response:
        """GET /api/payments/checkout/session-status?jti=... — статус сессии и платежа."""
        user_id = await require_user_id(request)

        jti = request.rel_url.query.get("jti", "")
        if not jti:
            return web.json_response({"error": "missing_jti"}, status=400)

        try:
            session = await db.get_checkout_session(jti)
            if not session:
                return web.json_response({"error": "session_not_found"}, status=404)

            if int(session.get("user_id", 0)) != user_id:
                return web.json_response({"error": "access_denied"}, status=403)

            payment_id = session.get("payment_id")
            result: dict[str, Any] = {
                "success": True,
                "jti": jti,
                "session_status": session.get("status"),
                "payment_id": payment_id,
                "confirmation_url": session.get("confirmation_url"),
                "rewards_processed": False,
                "modal_shown": False,
                "payment_status": None,
            }

            if payment_id:
                payment_record = await db.get_payment_by_id(payment_id)
                if payment_record:
                    result["payment_status"] = payment_record.get("status")
                    result["rewards_processed"] = payment_record.get("rewards_processed", False)
                    if isinstance(payment_record.get("metadata"), dict):
                        result["modal_shown"] = payment_record["metadata"].get("modal_shown", False)

            return web.json_response(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка session-status: %s", e, exc_info=True)
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    async def recent_success_payments_handler(request: web.Request) -> web.Response:
        """GET /api/payments/recent-success — последние succeeded платежи без modal_shown."""
        user_id = await require_user_id(request)

        try:
            history = await db.get_user_payment_history(user_id, limit=20)
            result = []
            for p in history:
                if p["status"] != "succeeded":
                    continue
                meta = p.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        meta = _stdlib_json.loads(meta)
                    except Exception:
                        meta = {}
                if meta.get("modal_shown"):
                    continue
                result.append({
                    "payment_id": p["payment_id"],
                    "amount": float(p["amount"]),
                    "currency": p["currency"],
                    "description": p["description"],
                    "metadata": meta,
                    "rewards_processed": p["rewards_processed"],
                    "created_at": p["created_at"].isoformat() if hasattr(p["created_at"], "isoformat") else str(p["created_at"]),
                })

            return web.json_response({"success": True, "payments": result})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка recent-success: %s", e, exc_info=True)
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    app.router.add_post("/api/payments/create", create_payment_handler)
    app.router.add_post("/api/payments/rustore/create", create_rustore_payment_handler)
    app.router.add_post("/api/payments/rustore/attach", attach_rustore_payment_handler)
    app.router.add_post("/api/payments/rustore/complete", complete_rustore_payment_handler)
    app.router.add_get("/api/payments/status", get_payment_status_handler)
    app.router.add_get("/api/payments/config", payment_config_handler)
    app.router.add_post("/api/payments/webhook", yookassa_webhook_handler)
    app.router.add_post("/api/payments/webhook/", yookassa_webhook_handler)
    app.router.add_get("/api/payments/webhook/test", webhook_test_handler)
    app.router.add_post("/api/payments/webhook/debug", webhook_debug_handler)
    app.router.add_post("/api/payments/stars/create", create_stars_invoice_handler)
    app.router.add_post("/api/payments/history/start", payment_history_start_handler)
    app.router.add_get("/api/payments/history", payment_history_handler)
    app.router.add_post("/api/payments/modal-shown", payment_modal_shown_handler)
    app.router.add_get("/api/payments/checkout/session-status", checkout_session_status_handler)
    app.router.add_get("/api/payments/recent-success", recent_success_payments_handler)
    app.router.add_get("/api/mail", user_mail_handler)
    app.router.add_post("/api/mail/read", mark_mail_read_handler)
    app.router.add_get("/api/mail/unread-count", get_unread_mail_count_handler)

    async def shop_buy_handler(request: web.Request) -> web.Response:
        """Обработчик покупки товара за гемы."""
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            item_type = data.get("item_type")
            item_name = data.get("item_name", "Товар")

            if not item_type:
                return web.json_response({"error": "item_type_required"}, status=400)

            is_admin = await _is_admin_user(db, user_id)
            admin_case_tier: int | None = None
            if item_type.startswith("admin_case_tier_"):
                try:
                    admin_case_tier = int(item_type.split("_")[-1])
                except (ValueError, AttributeError):
                    return web.json_response({"error": "invalid_admin_case_tier"}, status=400)

            # Получаем профиль пользователя
            user_profile = await db.get_user_profile(user_id)
            if not user_profile:
                return web.json_response({"error": "user_not_found"}, status=404)

            current_gems = user_profile["gems"] if user_profile.get("gems") is not None else 0

            # Определяем цену из конфига
            gems_amount: int = 0
            if admin_case_tier is not None:
                gems_amount = 1  # админская тест-цена
            elif item_type == "case" or item_type in CASE_PACKS:
                pack = CASE_PACKS.get(item_type, CASE_PACKS["case_pack_1"])
                gems_amount = pack["gems"]
            elif item_type.startswith("shop_set_"):
                # Цена набора — из БД (рубли/монеты, гемы опционально)
                pass  # gems_amount остаётся 0, будет определён ниже по set_data
            else:
                gems_amount = SHOP_PRICES.get(item_type)
                if gems_amount is None:
                    return web.json_response({
                        "success": False,
                        "error": "unknown_item_type",
                        "message": f"Неизвестный тип товара: {item_type}"
                    }, status=400)

            if admin_case_tier is not None:
                if not is_admin:
                    return web.json_response({"error": "admin_only"}, status=403)
                if not (1 <= admin_case_tier <= 5):
                    return web.json_response({"error": "invalid_admin_case_tier"}, status=400)

                gems_amount = 1
                debit_row = await db.fetchrow(
                    """
                    UPDATE users
                    SET gems = COALESCE(gems, 0) - $1,
                        updated_at = NOW()
                    WHERE user_id = $2
                      AND COALESCE(gems, 0) >= $1
                    RETURNING gems
                    """,
                    gems_amount, user_id,
                )
                if not debit_row:
                    return web.json_response({
                        "success": False,
                        "error": "insufficient_gems",
                        "message": f"Недостаточно гемов! Нужно {gems_amount} 💎, у вас {current_gems} 💎"
                    }, status=400)
                await _track_economy_safe(
                    db, user_id=user_id, event_type="spend", resource="gems",
                    amount=gems_amount, source="admin_debug_shop",
                    metadata={"item_type": item_type, "admin_case_tier": admin_case_tier},
                )
            elif item_type.startswith("shop_set_"):
                # Цена и валюта определяются из БД в ветке shop_set_ ниже
                pass
            elif item_type == "case" or item_type in CASE_PACKS:
                # Атомарный UPDATE происходит ниже в ветке выдачи
                pass
            else:
                if gems_amount <= 0:
                    return web.json_response({"error": "invalid_gems_amount"}, status=400)

            # Выдаем товар в зависимости от типа
            if admin_case_tier is not None:
                if admin_case_tier == 1:
                    await db.increment_user_keys(user_id, 1)
                    await db.sync_user_key_cases(user_id)
                    updated_keys = await db.fetchval(
                        "SELECT COALESCE(keys, 0) FROM users WHERE user_id = $1",
                        user_id
                    )
                    return web.json_response({
                        "success": True,
                        "message": f"Добавлен тестовый кейс T1",
                        "remaining_gems": current_gems,
                        "updated_keys": updated_keys or 0,
                        "granted_case_tier": 1,
                    })
                else:
                    case_id = await db.get_admin_case_id(admin_case_tier)
                    result = await db.add_user_case(user_id, case_id, admin_case_tier)
                    if not result.get("success"):
                        return web.json_response(
                            {"error": "case_creation_failed", "details": result.get("error")},
                            status=500,
                        )
                    return web.json_response({
                        "success": True,
                        "message": f"Добавлен тестовый кейс T{admin_case_tier}",
                        "remaining_gems": current_gems,
                        "granted_case_tier": admin_case_tier,
                        "user_case_id": result.get("user_case_id"),
                    })

            if item_type == "case" or item_type in CASE_PACKS:
                pack = CASE_PACKS.get(item_type, CASE_PACKS["case_pack_1"])
                keys_amount = pack["keys"]
                gems_price = pack["gems"]

                row = await db.fetchrow(
                    """
                    UPDATE users
                    SET gems = COALESCE(gems, 0) - $1,
                        keys = COALESCE(keys, 0) + $2,
                        updated_at = NOW()
                    WHERE user_id = $3
                      AND COALESCE(gems, 0) >= $1
                    RETURNING gems, keys
                    """,
                    gems_price, keys_amount, user_id,
                )
                if not row:
                    return web.json_response({
                        "error": "insufficient_gems",
                        "message":
                            f"Недостаточно гемов! Нужно {gems_price} 💎, у вас {current_gems} 💎"
                    }, status=400)

                remaining_gems = row["gems"]
                updated_keys = row["keys"]

                eco_meta = {
                    "item_type": item_type,
                    "keys_added": keys_amount,
                    "cost_gems": gems_price,
                }
                await _track_economy_safe(
                    db, user_id=user_id, event_type="spend", resource="gems",
                    amount=gems_price, source="shop_case_pack",
                    metadata=eco_meta,
                )
                await _track_economy_safe(
                    db, user_id=user_id, event_type="earn", resource="keys",
                    amount=keys_amount, source="shop_case_pack",
                    metadata=eco_meta,
                )

                import logging
                logging.getLogger(__name__).info(
                    "Пользователь %s купил %s: +%d keys за %d gems",
                    user_id, item_type, keys_amount, gems_price,
                )

                return web.json_response({
                    "success": True,
                    "item_type": item_type,
                    "keys_added": keys_amount,
                    "gems_spent": gems_price,
                    "remaining_gems": remaining_gems,
                    "updated_keys": updated_keys,
                })

            elif item_type and item_type.startswith("case_tier_"):
                # Покупка кейса определенного тира за гемы (case_tier_1, case_tier_2, и т.д.)
                try:
                    case_tier = int(item_type.split("_")[-1])
                    if not (1 <= case_tier <= 5):
                        return web.json_response({
                            "success": False,
                            "error": "invalid_case_tier",
                            "message": f"Неверный тир кейса: {case_tier}"
                        }, status=400)

                    case_id = await db.get_admin_case_id(case_tier)
                    if not case_id:
                        return web.json_response({
                            "success": False,
                            "error": "case_not_found",
                            "message": f"Кейс тира {case_tier} не найден"
                        }, status=404)

                    result = await db.fetchrow(
                        """
                        WITH debit AS (
                            UPDATE users
                            SET gems = COALESCE(gems, 0) - $1,
                                updated_at = NOW()
                            WHERE user_id = $2
                              AND COALESCE(gems, 0) >= $1
                            RETURNING user_id, gems
                        ),
                        inserted_case AS (
                            INSERT INTO user_cases (user_id, case_id, tier, status)
                            SELECT user_id, $3, $4, 'pending'
                            FROM debit
                            RETURNING id
                        )
                        SELECT debit.gems AS remaining_gems,
                               inserted_case.id AS user_case_id
                        FROM debit
                        JOIN inserted_case ON TRUE
                        """,
                        gems_amount, user_id, case_id, case_tier,
                    )
                    if not result:
                        return web.json_response({
                            "success": False,
                            "error": "insufficient_gems",
                            "message": f"Недостаточно гемов! Нужно {gems_amount} 💎, у вас {current_gems} 💎"
                        }, status=400)

                    await db.track_economy_event(
                        user_id=user_id, event_type="spend", resource="gems",
                        amount=gems_amount, source="shop",
                        metadata={"item_type": item_type, "case_tier": case_tier},
                    )
                    await db.track_economy_event(
                        user_id=user_id, event_type="earn", resource="case",
                        amount=1, source="shop",
                        metadata={"item_type": item_type, "case_tier": case_tier, "cost_gems": gems_amount},
                    )

                    import logging
                    logging.getLogger(__name__).info(
                        f"Пользователь {user_id} купил кейс T{case_tier} за {gems_amount} гемов"
                    )

                    return web.json_response({
                        "success": True,
                        "message": f"Успешно куплено: {item_name}",
                        "remaining_gems": result["remaining_gems"],
                        "granted_case_tier": case_tier,
                        "user_case_id": result["user_case_id"]
                    })
                except (ValueError, IndexError) as e:
                    import logging
                    logging.getLogger(__name__).error(
                        f"Ошибка парсинга тира кейса из {item_type}: {e}"
                    )
                    return web.json_response({
                        "success": False,
                        "error": "invalid_case_tier",
                        "message": f"Ошибка обработки покупки кейса: {str(e)}"
                    }, status=400)

            elif item_type and item_type.startswith("coins_"):
                # Покупка монет за гемы (coins_300, coins_1400, coins_5000, coins_20000)
                try:
                    coins_amount = int(item_type.split("_")[1])
                    row = await db.fetchrow(
                        """
                        UPDATE users
                        SET gems = COALESCE(gems, 0) - $1,
                            coins = COALESCE(coins, 0) + $2,
                            updated_at = NOW()
                        WHERE user_id = $3
                          AND COALESCE(gems, 0) >= $1
                        RETURNING gems, coins
                        """,
                        gems_amount, coins_amount, user_id,
                    )
                    if not row:
                        return web.json_response({
                            "success": False,
                            "error": "insufficient_gems",
                            "message": f"Недостаточно гемов! Нужно {gems_amount} 💎, у вас {current_gems} 💎"
                        }, status=400)
                    await db.track_economy_event(
                        user_id=user_id, event_type="spend", resource="gems",
                        amount=gems_amount, source="shop",
                        metadata={"item_type": item_type, "coins_added": coins_amount},
                    )
                    await db.track_economy_event(
                        user_id=user_id, event_type="earn", resource="coins",
                        amount=coins_amount, source="shop",
                        metadata={"item_type": item_type, "cost_gems": gems_amount},
                    )
                    import logging
                    logging.getLogger(__name__).info(
                        f"Пользователь {user_id} купил {coins_amount} монет за {gems_amount} гемов"
                    )
                    return web.json_response({
                        "success": True,
                        "message": f"Успешно куплено: {item_name}",
                        "remaining_gems": row["gems"],
                        "updated_coins": row["coins"]
                    })
                except (ValueError, IndexError) as e:
                    import logging
                    logging.getLogger(__name__).error(
                        f"Ошибка парсинга количества монет из {item_type}: {e}"
                    )
                    return web.json_response({
                        "success": False,
                        "error": "invalid_coins_amount",
                        "message": f"Ошибка обработки покупки монет: {str(e)}"
                    }, status=400)

            elif item_type and item_type.startswith("keys_"):
                # Покупка кейсов за гемы (keys_1, keys_3, keys_10, keys_25, keys_50, keys_100)
                try:
                    keys_amount = int(item_type.split("_")[1])
                    if keys_amount <= 0:
                        return web.json_response({
                            "success": False,
                            "error": "invalid_keys_amount",
                            "message": "Количество кейсов должно быть больше 0"
                        }, status=400)

                    row = await db.fetchrow(
                        """
                        UPDATE users
                        SET gems = COALESCE(gems, 0) - $1,
                            keys = COALESCE(keys, 0) + $2,
                            updated_at = NOW()
                        WHERE user_id = $3
                          AND COALESCE(gems, 0) >= $1
                        RETURNING gems, keys
                        """,
                        gems_amount, keys_amount, user_id,
                    )
                    if not row:
                        return web.json_response({
                            "success": False,
                            "error": "insufficient_gems",
                            "message": f"Недостаточно гемов! Нужно {gems_amount} 💎, у вас {current_gems} 💎"
                        }, status=400)

                    await _track_economy_safe(
                        db, user_id=user_id, event_type="spend", resource="gems",
                        amount=gems_amount, source="shop",
                        metadata={"item_type": item_type, "keys_added": keys_amount},
                    )
                    await _track_economy_safe(
                        db, user_id=user_id, event_type="earn", resource="keys",
                        amount=keys_amount, source="shop",
                        metadata={"item_type": item_type, "cost_gems": gems_amount},
                    )

                    import logging
                    logging.getLogger(__name__).info(
                        f"Пользователь {user_id} купил {keys_amount} кейсов за {gems_amount} гемов"
                    )

                    return web.json_response({
                        "success": True,
                        "message": f"Успешно куплено: {item_name}",
                        "remaining_gems": row["gems"],
                        "updated_keys": row["keys"]
                    })
                except (ValueError, IndexError) as e:
                    import logging
                    logging.getLogger(__name__).error(
                        f"Ошибка парсинга количества кейсов из {item_type}: {e}"
                    )
                    return web.json_response({
                        "success": False,
                        "error": "invalid_keys_amount",
                        "message": f"Ошибка обработки покупки кейсов: {str(e)}"
                    }, status=400)

            elif item_type and item_type.startswith("shop_set_"):
                # Покупка набора из БД
                try:
                    set_id = int(item_type.split("_")[-1])
                    set_data = await db.get_shop_set(set_id)
                    if not set_data:
                        return web.json_response({
                            "success": False,
                            "error": "set_not_found",
                            "message": "Набор не найден"
                        }, status=404)

                    if not set_data.get("is_active"):
                        return web.json_response({
                            "success": False,
                            "error": "set_inactive",
                            "message": "Набор недоступен для покупки"
                        }, status=400)

                    # Проверяем валюту и списываем средства
                    set_currency = set_data.get("currency", "rubles")
                    set_price = float(set_data.get("price", 0))

                    if set_currency not in {"gems", "coins"}:
                        return web.json_response({
                            "success": False,
                            "error": "invalid_currency",
                            "message": "Набор можно купить только за рубли через платеж"
                        }, status=400)

                    result = await db.purchase_shop_set(user_id, set_id)
                    if result.get("error") == "insufficient_gems":
                        return web.json_response({
                            "success": False,
                            "error": "insufficient_gems",
                            "message": f"Недостаточно гемов! Нужно {set_price} 💎, у вас {current_gems} 💎"
                        }, status=400)
                    if result.get("error") == "insufficient_coins":
                        current_coins = user_profile.get("coins", 0)
                        return web.json_response({
                            "success": False,
                            "error": "insufficient_coins",
                            "message": f"Недостаточно монет! Нужно {set_price} 💰, у вас {current_coins} 💰"
                        }, status=400)
                    if not result.get("success"):
                        return web.json_response({
                            "success": False,
                            "error": "rewards_failed",
                            "message": result.get("error", "Ошибка выдачи наград")
                        }, status=500)

                    import logging
                    logging.getLogger(__name__).info(
                        f"Пользователь {user_id} купил набор {set_id} за {set_price} {set_currency}"
                    )

                    return web.json_response({
                        "success": True,
                        "message": f"Успешно куплено: {item_name}",
                        "remaining_gems": result.get("gems", current_gems) or 0,
                        "updated_coins": result.get("coins", user_profile.get("coins", 0)) or 0,
                        "granted_rewards": result.get("granted", [])
                    })
                except (ValueError, IndexError) as e:
                    import logging
                    logging.getLogger(__name__).error(
                        f"Ошибка парсинга ID набора из {item_type}: {e}"
                    )
                    return web.json_response({
                        "success": False,
                        "error": "invalid_set_id",
                        "message": f"Ошибка обработки покупки набора: {str(e)}"
                    }, status=400)

            return web.json_response({
                "success": True,
                "message": f"Успешно куплено: {item_name}",
                "remaining_gems": current_gems - gems_amount
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "Ошибка покупки товара за гемы для user_id %s: %s", user_id, e, exc_info=True
            )
            return web.json_response(
                {"error": "internal_server_error", "message": str(e)}, status=500
            )

    app.router.add_post("/api/shop/buy", shop_buy_handler)

    async def card_image_handler(request: web.Request) -> web.Response:
        """
        Возвращает изображение карты из локальной файловой системы.
        Изображение берется из DesignAssets/Cards/<card_id>.png
        """
        card_id = request.rel_url.query.get("card_id")
        file_id = request.rel_url.query.get("file_id")

        if file_id and not card_id:
            try:
                card_record = await db.fetchrow(
                    "SELECT id FROM cards WHERE image_file_id = $1 LIMIT 1",
                    file_id
                )
                if card_record:
                    card_id = str(card_record["id"])
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Не удалось найти card_id по file_id: {e}")

        if not card_id:
            return web.json_response({"error": "card_id required"}, status=400)

        try:
            card_id_int = int(card_id)
        except ValueError:
            return web.json_response({"error": "invalid_card_id"}, status=400)

        card_image_path = DESIGN_ASSETS_DIR / "Cards" / f"{card_id_int}.png"

        if not card_image_path.exists() or not card_image_path.is_file():
            import logging
            logging.getLogger(__name__).warning(f"Изображение карты не найдено: {card_image_path}")
            return web.json_response({"error": "card_image_not_found"}, status=404)

        try:
            with open(card_image_path, "rb") as f:
                image_data = f.read()
            content_type = "image/png"
            response = web.Response(
                body=image_data,
                content_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Access-Control-Allow-Origin": "*"
                }
            )
            return response
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка чтения изображения карты: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error"}, status=500)

    async def gems_payment_handler(request: web.Request) -> web.Response:
        """Создать платёж YooKassa за пакет гемов."""
        import logging
        logger = logging.getLogger(__name__)

        user_id = await require_user_id(request)
        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        payment_service = request.app.get("payment_service")
        if not payment_service:
            return web.json_response({
                "error": "payment_service_not_configured",
                "message": "Платежный сервис не настроен. Проверьте настройки YooKassa в .env файле."
            }, status=503)

        try:
            data = await request.json()
            package_type = data.get("package_type", "").strip()

            pkg = GEM_PACKAGES.get(package_type)
            if not pkg:
                return web.json_response({"error": "unknown_package_type"}, status=400)

            if pkg.get("one_time"):
                settings = await db.get_user_settings(user_id)
                if settings and settings.get("starter_pack_used"):
                    return web.json_response({
                        "error": "already_used",
                        "message": "Стартовый пакет уже куплен"
                    }, status=400)

            price_rub = float(pkg["price"])
            gems_count = int(pkg["gems"])
            item_name = f"{gems_count} гемов"
            if pkg.get("one_time"):
                item_name += " (стартовый)"

            metadata = {
                "user_id": user_id,
                "item_type": "gems_package",
                "package_type": package_type,
                "package_gems": gems_count,
                "amount_rub": price_rub,
                "item_name": item_name,
                "starter": package_type == "starter_once",
            }

            return_url = data.get("return_url", request.app.get("webapp_url", "https://t.me/your_bot"))

            payment_result = await asyncio.to_thread(
                payment_service.create_payment,
                amount=price_rub,
                currency="RUB",
                description=item_name,
                return_url=return_url,
                metadata=metadata,
            )

            if not payment_result.get("success"):
                err = payment_result.get("error", "unknown")
                logger.error("Ошибка создания платежа за гемы: %s", err)
                return web.json_response({
                    "success": False,
                    "error": err,
                    "message": f"Ошибка создания платежа: {err}"
                }, status=400)

            payment_id = payment_result.get("payment_id")

            db_result = await db.create_payment(
                user_id=user_id,
                payment_id=payment_id,
                amount=price_rub,
                currency="RUB",
                description=item_name,
                metadata=metadata,
            )
            if not db_result.get("success"):
                logger.error("Не удалось сохранить платёж %s в БД: %s", payment_id, db_result.get("error"))
                return web.json_response({
                    "success": False,
                    "error": "payment_record_not_saved",
                    "message": "Платеж не был создан. Попробуйте еще раз.",
                }, status=500)

            return web.json_response({
                "success": True,
                "payment_id": payment_id,
                "confirmation_url": payment_result.get("confirmation_url"),
                "package_type": package_type,
                "gems_amount": gems_count,
            })
        except Exception as e:
            logger.error("Ошибка создания платежа за гемы для user_id %s: %s", user_id, e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": "Не удалось создать платеж"}, status=500)

    async def particles_daily_handler(request: web.Request) -> web.Response:
        """Получить сегодняшнюю ротацию карт на частицы."""
        user_id = await require_user_id(request)

        from datetime import date, datetime, timedelta, timezone

        try:
            today = date.today()
            settings_record = await db.get_user_settings(user_id)
            rotation_date = settings_record.get("particles_rotation_date") if settings_record else None
            raw_cards = settings_record.get("particles_rotation_cards") if settings_record else None
            purchased_raw = settings_record.get("particles_purchased_today") if settings_record else None

            if isinstance(rotation_date, datetime):
                rotation_date = rotation_date.date()

            if isinstance(raw_cards, str):
                raw_cards = _stdlib_json.loads(raw_cards)
            if isinstance(purchased_raw, str):
                purchased_raw = _stdlib_json.loads(purchased_raw)
            purchased_today = purchased_raw if isinstance(purchased_raw, list) else []

            if rotation_date != today or not raw_cards or not isinstance(raw_cards, list) or len(raw_cards) == 0:
                common = await db.get_random_cards_by_rarities(["common"], limit=1)
                rare = await db.get_random_cards_by_rarities(["rare"], limit=1)
                epic = await db.get_random_cards_by_rarities(["epic"], limit=1)

                raw_cards = [
                    common[0]["id"] if common else None,
                    rare[0]["id"] if rare else None,
                    epic[0]["id"] if epic else None,
                ]
                raw_cards = [c for c in raw_cards if c is not None]

                await db.update_user_settings(
                    user_id,
                    particles_rotation_cards=_stdlib_json.dumps(raw_cards),
                    particles_rotation_date=today,
                    particles_purchased_today="[]",
                )

            if not raw_cards:
                return web.json_response({"cards": [], "rotation_date": today.isoformat(), "purchased_today": []})

            placeholders = ",".join(f"${i+1}" for i in range(len(raw_cards)))
            rows = await db.fetch(
                f"""
                SELECT c.id,
                       c.name,
                       c.rarity,
                       c.simplified_levelup,
                       COALESCE(uc.level, 1) AS level,
                       COALESCE(uc.particles, 0) AS current_particles
                FROM cards c
                LEFT JOIN user_cards uc
                  ON uc.card_id = c.id AND uc.user_id = ${len(raw_cards) + 1}
                WHERE c.id IN ({placeholders})
                """,
                *raw_cards,
                user_id,
            )

            now = datetime.now(timezone.utc)
            midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            next_rotation_ts = int(midnight.timestamp())

            rows_by_id = {int(row["id"]): dict(row) for row in rows}
            cards = []
            for card_id in raw_cards:
                row = rows_by_id.get(int(card_id))
                if not row:
                    continue
                rarity = row["rarity"]
                cost = PARTICLES_COSTS.get(rarity, PARTICLES_COSTS["common"])
                level = int(row.get("level") or 1)
                current_particles = int(row.get("current_particles") or 0)
                simplified_levelup = bool(row.get("simplified_levelup", False))
                max_level = db.get_card_max_level({"simplified_levelup": simplified_levelup})
                is_max_level = level >= max_level
                upgrade_cost = db.get_upgrade_cost(rarity, level, simplified_levelup)
                upgrade_particles_required = 0 if is_max_level else int(upgrade_cost.get("particles") or 0)
                progress_pct = 100 if is_max_level else (
                    min(100, round((current_particles / upgrade_particles_required) * 100))
                    if upgrade_particles_required > 0 else 0
                )
                cards.append({
                    "id": row["id"],
                    "name": row["name"],
                    "rarity": rarity,
                    "particles": cost["particles"],
                    "coins": cost["coins"],
                    "level": level,
                    "current_particles": current_particles,
                    "upgrade_particles_required": upgrade_particles_required,
                    "progress_pct": progress_pct,
                    "is_max_level": is_max_level,
                })
            cards = order_particles_for_shop(cards, today.isoformat())

            return web.json_response({
                "cards": cards,
                "rotation_date": today.isoformat(),
                "next_rotation_ts": next_rotation_ts,
                "purchased_today": purchased_today,
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка particles/daily: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def particles_buy_handler(request: web.Request) -> web.Response:
        """Купить частицы за монеты."""
        user_id = await require_user_id(request)
        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            card_id = int(data.get("card_id", 0))
            if card_id <= 0:
                return web.json_response({"error": "invalid_card_id"}, status=400)

            from datetime import date
            settings_record = await db.get_user_settings(user_id)
            raw_cards = settings_record.get("particles_rotation_cards") if settings_record else None
            rotation_date = settings_record.get("particles_rotation_date") if settings_record else None
            purchased_raw = settings_record.get("particles_purchased_today") if settings_record else None

            if isinstance(purchased_raw, str):
                purchased_raw = _stdlib_json.loads(purchased_raw)

            if isinstance(rotation_date, datetime):
                rotation_date = rotation_date.date()  # type: ignore[assignment]
            if isinstance(raw_cards, str):
                raw_cards = _stdlib_json.loads(raw_cards)

            if rotation_date != date.today() or not raw_cards or card_id not in raw_cards:
                return web.json_response({"error": "card_not_in_rotation"}, status=400)

            purchased = _stdlib_json.loads(purchased_raw) if isinstance(purchased_raw, str) else (purchased_raw or [])
            if card_id in purchased:
                return web.json_response({"error": "already_purchased", "message": "Вы уже купили частицы для этой карты сегодня"}, status=400)

            rarity_row = await db.fetchrow("SELECT rarity FROM cards WHERE id = $1", card_id)
            if not rarity_row:
                return web.json_response({"error": "card_not_found"}, status=404)

            rarity = rarity_row["rarity"]
            cost = PARTICLES_COSTS.get(rarity)
            if not cost:
                return web.json_response({"error": "unsupported_rarity"}, status=400)

            coins_cost = cost["coins"]
            particles_amount = cost["particles"]

            pool = getattr(db, "_pool", None)
            if pool:
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        locked_settings = await conn.fetchrow(
                            """
                            SELECT particles_purchased_today
                            FROM user_settings
                            WHERE user_id = $1
                            FOR UPDATE
                            """,
                            user_id,
                        )
                        locked_raw = locked_settings["particles_purchased_today"] if locked_settings else []
                        if isinstance(locked_raw, str):
                            locked_purchased = _stdlib_json.loads(locked_raw)
                        else:
                            locked_purchased = locked_raw or []
                        if card_id in locked_purchased:
                            return web.json_response({
                                "error": "already_purchased",
                                "message": "Вы уже купили частицы для этой карты сегодня",
                            }, status=400)

                        result = await conn.fetchrow(
                            """
                            UPDATE users
                            SET coins = COALESCE(coins, 0) - $1,
                                updated_at = NOW()
                            WHERE user_id = $2
                              AND COALESCE(coins, 0) >= $1
                            RETURNING coins
                            """,
                            coins_cost, user_id,
                        )
                        if not result:
                            return web.json_response({
                                "success": False,
                                "error": "insufficient_coins",
                                "message": f"Недостаточно монет! Нужно {coins_cost} 🪙",
                            }, status=400)

                        await conn.execute(
                            """
                            INSERT INTO user_cards (user_id, card_id, level, particles, obtained_at)
                            VALUES ($1, $2, 1, $3, NOW())
                            ON CONFLICT (user_id, card_id) DO UPDATE
                            SET particles = COALESCE(user_cards.particles, 0) + $3
                            """,
                            user_id, card_id, particles_amount,
                        )
                        locked_purchased.append(card_id)
                        await conn.execute(
                            """
                            INSERT INTO user_settings (user_id, particles_purchased_today)
                            VALUES ($1, $2::jsonb)
                            ON CONFLICT (user_id) DO UPDATE
                            SET particles_purchased_today = EXCLUDED.particles_purchased_today
                            """,
                            user_id,
                            _stdlib_json.dumps(locked_purchased),
                        )
                        particles_row = await conn.fetchrow(
                            "SELECT COALESCE(particles, 0) AS particles FROM user_cards WHERE user_id = $1 AND card_id = $2",
                            user_id, card_id,
                        )
                        return web.json_response({
                            "success": True,
                            "updated_coins": result["coins"],
                            "updated_particles": particles_row["particles"] if particles_row else particles_amount,
                            "card_id": card_id,
                            "message": f"Куплено {particles_amount} ✨ для карты",
                        })

            result = await db.fetchrow(
                "UPDATE users SET coins = coins - $1 WHERE user_id = $2 AND coins >= $1 RETURNING coins",
                coins_cost, user_id,
            )
            if not result:
                return web.json_response({
                    "success": False,
                    "error": "insufficient_coins",
                    "message": f"Недостаточно монет! Нужно {coins_cost} 🪙"
                }, status=400)

            updated_coins = result["coins"]

            await db.execute(
                """
                INSERT INTO user_cards (user_id, card_id, level, particles, obtained_at)
                VALUES ($1, $2, 1, $3, NOW())
                ON CONFLICT (user_id, card_id) DO UPDATE
                SET particles = COALESCE(user_cards.particles, 0) + $3
                """,
                user_id, card_id, particles_amount,
            )

            particles_row = await db.fetchrow(
                "SELECT COALESCE(particles, 0) as particles FROM user_cards WHERE user_id = $1 AND card_id = $2",
                user_id, card_id,
            )
            updated_particles = particles_row["particles"] if particles_row else particles_amount

            purchased = _stdlib_json.loads(purchased_raw) if isinstance(purchased_raw, str) else (purchased_raw or [])
            if card_id not in purchased:
                purchased.append(card_id)
            await db.update_user_settings(user_id, particles_purchased_today=_stdlib_json.dumps(purchased))

            return web.json_response({
                "success": True,
                "updated_coins": updated_coins,
                "updated_particles": updated_particles,
                "card_id": card_id,
                "message": f"Куплено {particles_amount} ✨ для карты",
            })
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка particles/buy: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": "Не удалось купить частицы"}, status=500)

    async def shop_sets_image_handler(request: web.Request) -> web.Response:
        """Прокси для получения изображения набора по Telegram file_id."""
        file_id = request.rel_url.query.get("file_id", "").strip()
        if not file_id:
            return web.json_response({"error": "file_id_required"}, status=400)

        import logging, aiohttp
        logger = logging.getLogger(__name__)
        token = app["bot_token"]

        try:
            from aiogram import Bot
            bot = Bot(token=token)
            tg_file = await bot.get_file(file_id)
            file_path = tg_file.file_path
            await bot.session.close()

            url = f"https://api.telegram.org/file/bot{token}/{file_path}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return web.json_response({"error": "file_not_found"}, status=404)
                    image_data = await resp.read()

            content_type = "image/png"
            if file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                content_type = "image/jpeg"
            elif file_path.endswith(".webp"):
                content_type = "image/webp"

            return web.Response(body=image_data, content_type=content_type)
        except Exception as e:
            logger.error("Ошибка загрузки изображения набора: %s", e, exc_info=True)
            return web.json_response({"error": "image_fetch_failed", "message": str(e)}, status=500)
        """
        Возвращает изображение карты из локальной файловой системы.
        Изображение берется из DesignAssets/Cards/<card_id>.png
        """
        card_id = request.rel_url.query.get("card_id")
        file_id = request.rel_url.query.get("file_id")  # Для обратной совместимости

        # Если передан file_id, пытаемся получить card_id из БД (для обратной совместимости)
        if file_id and not card_id:
            try:
                card_record = await db.fetchrow(
                    "SELECT id FROM cards WHERE image_file_id = $1 LIMIT 1",
                    file_id
                )
                if card_record:
                    card_id = str(card_record["id"])
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Не удалось найти card_id по file_id: {e}")

        if not card_id:
            return web.json_response({"error": "card_id required"}, status=400)

        try:
            card_id_int = int(card_id)
        except ValueError:
            return web.json_response({"error": "invalid_card_id"}, status=400)

        # Формируем путь к файлу изображения карты
        card_image_path = DESIGN_ASSETS_DIR / "Cards" / f"{card_id_int}.png"

        # Проверяем существование файла
        if not card_image_path.exists() or not card_image_path.is_file():
            import logging
            logging.getLogger(__name__).warning(f"Изображение карты не найдено: {card_image_path}")
            return web.json_response({"error": "card_image_not_found"}, status=404)

        try:
            # Читаем файл изображения
            with open(card_image_path, "rb") as f:
                image_data = f.read()

            # Определяем content-type по расширению файла
            content_type = "image/png"  # Все карты в формате PNG

            # Возвращаем изображение с правильными заголовками
            response = web.Response(
                body=image_data,
                content_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=86400",  # Кэшируем на 24 часа
                    "Access-Control-Allow-Origin": "*"  # Разрешаем CORS
                }
            )
            return response
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка чтения изображения карты: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error"}, status=500)

    app.router.add_get("/api/cards/image", card_image_handler)

    # API для управления наборами магазина (только для админа)
    async def shop_sets_list_handler(request: web.Request) -> web.Response:
        """Получить список всех наборов."""
        user_id = await require_user_id(request)

        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)

        try:
            active_only = request.rel_url.query.get("active_only", "false").lower() == "true"
            sets = await db.get_shop_sets(active_only=active_only)
            return web.json_response({"sets": sets})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения наборов: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_set_detail_handler(request: web.Request) -> web.Response:
        """Получить детали набора по ID."""
        user_id = await require_user_id(request)

        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)

        try:
            set_id = int(request.match_info.get("set_id", 0))
            if set_id <= 0:
                return web.json_response({"error": "invalid_set_id"}, status=400)
            set_data = await db.get_shop_set(set_id)
            if not set_data:
                return web.json_response({"error": "set_not_found"}, status=404)
            return web.json_response({"set": set_data})
        except (TypeError, ValueError) as e:
            return web.json_response({"error": "invalid_set_id", "message": str(e)}, status=400)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения набора: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_set_create_handler(request: web.Request) -> web.Response:
        """Создать новый набор."""
        user_id = await require_user_id(request)

        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            name = str(data.get("name") or "").strip()
            description_raw = data.get("description")
            image_raw = data.get("image_file_id")
            description = (
                str(description_raw).strip() if description_raw is not None else None
            ) or None
            image_file_id = (
                str(image_raw).strip() if image_raw is not None else None
            ) or None
            price = float(data.get("price", 0))
            currency = data.get("currency", "rubles")
            rewards = data.get("rewards", [])

            if not name:
                return web.json_response({"error": "name_required"}, status=400)
            if price < 0:
                return web.json_response({"error": "invalid_price"}, status=400)
            if currency not in ("rubles", "gems", "coins"):
                return web.json_response({"error": "invalid_currency"}, status=400)
            if not isinstance(rewards, list):
                return web.json_response({"error": "invalid_rewards"}, status=400)
            if price > 0:
                _, rewards_error = db._normalize_shop_set_rewards(rewards)
                if rewards_error:
                    return web.json_response({"error": rewards_error}, status=400)

            result = await db.create_shop_set(
                name=name,
                description=description,
                image_file_id=image_file_id,
                price=price,
                currency=currency,
                created_by=user_id,
                rewards=rewards
            )

            if result.get("success"):
                return web.json_response({"success": True, "set_id": result.get("set_id")})
            else:
                return web.json_response({"error": result.get("error", "unknown")}, status=500)
        except (TypeError, ValueError) as e:
            return web.json_response({"error": "invalid_body", "message": str(e)}, status=400)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка создания набора: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_set_update_handler(request: web.Request) -> web.Response:
        """Обновить набор."""
        user_id = await require_user_id(request)

        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            set_id = int(data.get("set_id", 0))
            if set_id <= 0:
                return web.json_response({"error": "invalid_set_id"}, status=400)

            update_kwargs: dict[str, Any] = {}
            if "name" in data:
                name = str(data.get("name") or "").strip()
                if not name:
                    return web.json_response({"error": "name_required"}, status=400)
                update_kwargs["name"] = name
            if "description" in data:
                description_raw = data.get("description")
                update_kwargs["description"] = (
                    str(description_raw).strip() if description_raw is not None else None
                ) or None
            if "image_file_id" in data:
                image_raw = data.get("image_file_id")
                update_kwargs["image_file_id"] = (
                    str(image_raw).strip() if image_raw is not None else None
                ) or None
            if "price" in data:
                price = float(data.get("price") or 0)
                if price < 0:
                    return web.json_response({"error": "invalid_price"}, status=400)
                update_kwargs["price"] = price
            if "currency" in data:
                currency = data.get("currency")
                if currency not in ("rubles", "gems", "coins"):
                    return web.json_response({"error": "invalid_currency"}, status=400)
                update_kwargs["currency"] = currency
            if "is_active" in data:
                update_kwargs["is_active"] = bool(data.get("is_active"))
            if "rewards" in data:
                rewards = data.get("rewards") or []
                if not isinstance(rewards, list):
                    return web.json_response({"error": "invalid_rewards"}, status=400)
                _, rewards_error = db._normalize_shop_set_rewards(rewards)
                if rewards_error:
                    return web.json_response({"error": rewards_error}, status=400)
                update_kwargs["rewards"] = rewards
            if not update_kwargs:
                return web.json_response({"error": "no_fields"}, status=400)

            result = await db.update_shop_set(
                set_id=set_id,
                **update_kwargs,
            )

            if result.get("success"):
                return web.json_response({"success": True})
            status = 404 if result.get("error") == "set_not_found" else 400
            return web.json_response({"error": result.get("error", "unknown")}, status=status)
        except (TypeError, ValueError) as e:
            return web.json_response({"error": "invalid_body", "message": str(e)}, status=400)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка обновления набора: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_set_delete_handler(request: web.Request) -> web.Response:
        """Удалить набор."""
        user_id = await require_user_id(request)

        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            set_id = int(data.get("set_id", 0))
            if set_id <= 0:
                return web.json_response({"error": "invalid_set_id"}, status=400)

            result = await db.delete_shop_set(set_id)
            if result.get("success"):
                return web.json_response({"success": True})
            else:
                return web.json_response({"error": result.get("error", "unknown")}, status=500)
        except (TypeError, ValueError) as e:
            return web.json_response({"error": "invalid_set_id", "message": str(e)}, status=400)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка удаления набора: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_sets_public_handler(request: web.Request) -> web.Response:
        """Получить список активных наборов для магазина (публичный endpoint)."""
        try:
            sets = await db.get_shop_sets(active_only=True)
            return web.json_response({"sets": sets})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения наборов: {e}", exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def shop_catalog_handler(request: web.Request) -> web.Response:
        """Получить серверный каталог товаров внутриигрового магазина."""
        try:
            products = await db.get_ruble_products(active_only=True, surface="shop")
            return web.json_response(build_shop_catalog(products))
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка получения каталога магазина: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    app.router.add_get("/api/shop/sets", shop_sets_public_handler)
    app.router.add_get("/api/shop/catalog", shop_catalog_handler)
    app.router.add_get("/api/admin/shop/sets", shop_sets_list_handler)
    app.router.add_get("/api/admin/shop/sets/{set_id}", shop_set_detail_handler)
    app.router.add_post("/api/admin/shop/sets/create", shop_set_create_handler)
    app.router.add_post("/api/admin/shop/sets/update", shop_set_update_handler)
    app.router.add_post("/api/admin/shop/sets/delete", shop_set_delete_handler)
    app.router.add_get("/api/shop/sets/image", shop_sets_image_handler)
    app.router.add_get("/api/shop/particles/daily", particles_daily_handler)
    app.router.add_post("/api/shop/particles/buy", particles_buy_handler)
    app.router.add_post("/api/payments/gems/create", gems_payment_handler)

    # ── Ruble products admin API ──

    RUBLE_PRODUCT_ITEM_TYPES = {"extrapass", "extrapass_ultra", "starter_boost", "gems_package", "shop_set"}

    def _product_error_status(error: str) -> int:
        if error in {"product_code_exists", "duplicate_product_code"} or "duplicate key" in error or "unique" in error.lower():
            return 409
        if error in {"shop_set_not_found", "product_not_found"}:
            return 404
        if error in {
            "invalid_item_type",
            "package_type_required",
            "invalid_package_type",
            "shop_set_id_required",
            "invalid_shop_set_id",
            "code_itemtype_name_required",
            "name_required",
            "invalid_price",
            "invalid_currency",
            "invalid_sort_order",
            "no_fields",
        }:
            return 400
        return 500

    async def _admin_ruble_product_options_payload() -> dict[str, Any]:
        try:
            sets = await db.get_shop_sets(active_only=False)
        except Exception as exc:
            logging.getLogger(__name__).warning("product options shop sets fallback: %s", exc, exc_info=True)
            sets = []
        return {
            "item_types": [
                {"value": "extrapass", "label": "ExtraPass", "description": "30-day premium pass", "defaults": {"code": "extrapass", "name": "ExtraPass", "price": 179, "show_in_game": True, "show_in_shop": True}},
                {"value": "extrapass_ultra", "label": "ExtraPass Ultra", "description": "30-day Ultra pass", "defaults": {"code": "extrapass_ultra", "name": "ExtraPass Ultra", "price": 349, "badge": "popular", "show_in_game": True, "show_in_shop": True}},
                {"value": "starter_boost", "label": "Starter Boost", "description": "Starter paid bundle", "defaults": {"code": "starter_boost", "name": "Starter Boost", "price": 499, "show_in_game": True, "show_in_shop": True}},
                {"value": "gems_package", "label": "Gems package", "description": "Real-money gem package", "defaults": {"code": "gems_100", "name": "100 gems", "price": 99, "show_in_game": False, "show_in_shop": True}},
                {"value": "shop_set", "label": "Shop set", "description": "DB-backed reward set", "defaults": {"code": "shop_set", "name": "Shop Set", "price": 0, "show_in_game": True, "show_in_shop": True}},
            ],
            "package_types": {
                "gems_package": [
                    {
                        "value": package_id,
                        "label": f"{package['gems']} gems" + (" · one-time" if package.get("one_time") else ""),
                        "price": package.get("price"),
                        "gems": package.get("gems"),
                        "badge": "one-time" if package.get("one_time") else ("discount" if package.get("discount_pct") else None),
                    }
                    for package_id, package in GEM_PACKAGES.items()
                ],
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
            ],
        }

    async def _normalize_admin_ruble_product_payload(
        data: dict[str, Any],
        *,
        require_identity: bool,
        existing: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        normalized: dict[str, Any] = {}
        existing = existing or {}

        if require_identity or "code" in data:
            code = str(data.get("code") or "").strip()
            if not code:
                return {}, "code_itemtype_name_required"
            normalized["code"] = code

        item_type = str(data.get("item_type") if "item_type" in data else existing.get("item_type") or "").strip()
        if require_identity or "item_type" in data or "package_type" in data or "shop_set_id" in data:
            if not item_type:
                return {}, "code_itemtype_name_required"
            if item_type not in RUBLE_PRODUCT_ITEM_TYPES:
                return {}, "invalid_item_type"
            if require_identity or "item_type" in data:
                normalized["item_type"] = item_type

            package_type = data.get("package_type") if "package_type" in data else existing.get("package_type")
            shop_set_id_raw = data.get("shop_set_id") if "shop_set_id" in data else existing.get("shop_set_id")
            if item_type == "gems_package":
                package_text = str(package_type or "").strip()
                if not package_text:
                    return {}, "package_type_required"
                if package_text not in GEM_PACKAGES:
                    return {}, "invalid_package_type"
                normalized["package_type"] = package_text
                normalized["shop_set_id"] = None
            elif item_type == "shop_set":
                if shop_set_id_raw in (None, ""):
                    return {}, "shop_set_id_required"
                try:
                    shop_set_id = int(shop_set_id_raw)
                except (TypeError, ValueError):
                    return {}, "invalid_shop_set_id"
                if shop_set_id <= 0:
                    return {}, "invalid_shop_set_id"
                if hasattr(db, "get_shop_set") and not await db.get_shop_set(shop_set_id):
                    return {}, "shop_set_not_found"
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
                return {}, "name_required" if "name" in data else "code_itemtype_name_required"
            normalized["name"] = name

        if require_identity or "price" in data:
            try:
                price = float(data.get("price", 0))
            except (TypeError, ValueError):
                return {}, "invalid_price"
            if price < 0:
                return {}, "invalid_price"
            normalized["price"] = price

        if require_identity or "currency" in data:
            currency = str(data.get("currency") or "rubles")
            if currency not in ("rubles", "gems", "coins"):
                return {}, "invalid_currency"
            normalized["currency"] = currency

        for field in ("description", "image_url", "badge"):
            if field in data:
                raw_value = data.get(field)
                normalized[field] = (str(raw_value).strip() if raw_value is not None else None) or None
        if "sort_order" in data:
            try:
                normalized["sort_order"] = int(data.get("sort_order") or 100)
            except (TypeError, ValueError):
                return {}, "invalid_sort_order"
        for field in ("show_in_game", "show_in_shop", "is_active"):
            if field in data:
                normalized[field] = bool(data[field])
        return normalized, None

    async def ruble_products_options_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)
        data = await _admin_ruble_product_options_payload()
        return web.json_response({"status": "ok", "data": _serialize_datetime(data)})

    async def ruble_products_list_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)
        try:
            active_only = request.rel_url.query.get("active_only", "false").lower() == "true"
            products = await db.get_ruble_products(active_only=active_only)
            return web.json_response({"products": products})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка получения ruble_products: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def ruble_product_detail_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)
        try:
            code = request.match_info.get("code", "").strip()
            if not code:
                return web.json_response({"error": "code_required"}, status=400)
            product = await db.get_ruble_product(code)
            if not product:
                return web.json_response({"error": "product_not_found"}, status=404)
            return web.json_response({"product": product})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка получения ruble_product: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def ruble_product_create_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)
        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)
        try:
            data = await request.json()
            create_kwargs, error = await _normalize_admin_ruble_product_payload(data, require_identity=True)
            if error:
                return web.json_response({"error": error}, status=_product_error_status(error))
            if hasattr(db, "get_ruble_product") and await db.get_ruble_product(str(create_kwargs["code"])):
                return web.json_response({"error": "product_code_exists"}, status=409)
            result = await db.create_ruble_product(**create_kwargs)
            if result.get("success"):
                return web.json_response({"success": True, "product_id": result.get("product_id")})
            error = str(result.get("error", "unknown"))
            return web.json_response({"error": error}, status=_product_error_status(error))
        except (TypeError, ValueError) as e:
            return web.json_response({"error": "invalid_body", "message": str(e)}, status=400)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка создания ruble_product: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def ruble_product_update_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)
        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)
        try:
            data = await request.json()
            code_or_id = data.get("code") or data.get("id")
            if not code_or_id:
                return web.json_response({"error": "code_or_id_required"}, status=400)
            try:
                code_or_id = int(code_or_id)
            except (ValueError, TypeError):
                code_or_id = str(code_or_id)
            existing = None
            if isinstance(code_or_id, str) and hasattr(db, "get_ruble_product"):
                existing = await db.get_ruble_product(code_or_id)
                if not existing:
                    return web.json_response({"error": "product_not_found"}, status=404)
            update_kwargs, error = await _normalize_admin_ruble_product_payload(
                data,
                require_identity=False,
                existing=existing,
            )
            if error:
                return web.json_response({"error": error}, status=_product_error_status(error))
            if not update_kwargs:
                return web.json_response({"error": "no_fields"}, status=400)
            result = await db.update_ruble_product(code_or_id, **update_kwargs)
            if result.get("success"):
                return web.json_response({"success": True})
            error = str(result.get("error", "unknown"))
            return web.json_response({"error": error}, status=_product_error_status(error))
        except (TypeError, ValueError) as e:
            return web.json_response({"error": "invalid_body", "message": str(e)}, status=400)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка обновления ruble_product: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def ruble_product_delete_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)
        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)
        try:
            data = await request.json()
            code_or_id = data.get("code") or data.get("id")
            if not code_or_id:
                return web.json_response({"error": "code_or_id_required"}, status=400)
            try:
                code_or_id = int(code_or_id)
            except (ValueError, TypeError):
                code_or_id = str(code_or_id)
            result = await db.delete_ruble_product(code_or_id)
            if result.get("success"):
                return web.json_response({"success": True})
            return web.json_response({"error": result.get("error", "unknown")}, status=500)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка удаления ruble_product: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    # ── Ruble products public API ──

    async def ruble_products_public_handler(request: web.Request) -> web.Response:
        try:
            surface = request.rel_url.query.get("surface", "shop")
            products = await db.get_ruble_products(active_only=True, surface=surface)
            return web.json_response({"products": products})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка получения публичных ruble_products: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    app.router.add_get("/api/admin/ruble-products", ruble_products_list_handler)
    app.router.add_get("/api/admin/ruble-products/options", ruble_products_options_handler)
    app.router.add_get("/api/admin/ruble-products/{code}", ruble_product_detail_handler)
    app.router.add_post("/api/admin/ruble-products/create", ruble_product_create_handler)
    app.router.add_post("/api/admin/ruble-products/update", ruble_product_update_handler)
    app.router.add_post("/api/admin/ruble-products/delete", ruble_product_delete_handler)
    app.router.add_get("/api/shop/ruble-products", ruble_products_public_handler)

    # ── Image upload for admin ──

    UPLOADS_DIR = Path(__file__).resolve().parents[1] / "extraShop" / "uploads" / "products"

    async def admin_upload_product_image(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            return web.json_response({"error": "admin_only"}, status=403)
        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)
        try:
            reader = await request.multipart()
            field = await reader.next()
            if not field or field.name != "file":
                return web.json_response({"error": "file_field_required"}, status=400)
            import uuid, os
            content_type = field.headers.get("Content-Type", "")
            if content_type not in ("image/png", "image/jpeg", "image/webp"):
                return web.json_response({"error": "invalid_image_type", "allowed": ["image/png", "image/jpeg", "image/webp"]}, status=400)
            ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
            ext = ext_map.get(content_type, ".png")
            data = b""
            while True:
                chunk = await field.read_chunk(65536)
                if not chunk:
                    break
                data += chunk
            if len(data) > 5 * 1024 * 1024:
                return web.json_response({"error": "file_too_large", "max_mb": 5}, status=400)
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            filename = str(uuid.uuid4()) + ext
            filepath = UPLOADS_DIR / filename
            filepath.write_bytes(data)
            image_url = f"/extraShop/uploads/products/{filename}"
            return web.json_response({"success": True, "image_url": image_url})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Ошибка загрузки картинки: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    async def uploads_static_handler(request: web.Request) -> web.Response:
        filename = request.match_info.get("filename", "").lstrip("/")
        if not filename or ".." in filename:
            raise web.HTTPForbidden()
        file_path = UPLOADS_DIR / filename
        if not file_path.exists() or not file_path.is_file():
            raise web.HTTPNotFound()
        content_type = "image/png"
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif filename.endswith(".webp"):
            content_type = "image/webp"
        return web.FileResponse(file_path, headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        })

    app.router.add_post("/api/admin/uploads/product-image", admin_upload_product_image)
    app.router.add_get("/extraShop/uploads/products/{filename}", uploads_static_handler)

    async def dice_status_handler(request: web.Request) -> web.Response:
        """Получить статус кубика (можно ли бросать, когда был последний бросок)."""
        user_id = await require_user_id(request)

        try:
            status = await db.get_dice_status(user_id)
            return web.json_response(_serialize_datetime(status))
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения статуса кубика: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def dice_roll_handler(request: web.Request) -> web.Response:
        """Бросить кубик."""
        user_id = await require_user_id(request)

        try:
            result = await db.roll_dice(user_id)
            return web.json_response(_serialize_datetime(result))
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка броска кубика: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def dice_notification_prompt_status_handler(request: web.Request) -> web.Response:
        """Проверить, показывалось ли уже предложение включить уведомления."""
        user_id = await require_user_id(request)

        try:
            prompt_shown = await db.get_dice_notification_prompt_status(user_id)
            return web.json_response({"prompt_shown": prompt_shown})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка проверки статуса предложения: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def dice_notification_prompt_mark_handler(request: web.Request) -> web.Response:
        """Отметить, что предложение включить уведомления было показано."""
        user_id = await require_user_id(request)

        try:
            await db.mark_dice_notification_prompt_shown(user_id)
            return web.json_response({"success": True})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка отметки предложения: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def generator_status_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            data = await db.get_generator_status(int(user_id))
            return web.json_response(data)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения статуса генератора: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def generator_claim_handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            body = None
        if body is not None:
            user_id = await require_user_id_from_payload(request, body)
        else:
            user_id = await require_user_id(request)
        try:
            data = await db.claim_generator_keys(int(user_id))
            if data.get("success") and data.get("keys_claimed", 0) > 0:
                await _track_economy_safe(db, user_id=int(user_id), event_type="earn",
                    resource="keys", amount=data["keys_claimed"], source="generator",
                    metadata={"generator_level": data.get("level")})
            return web.json_response(data)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка сбора ключей генератора: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def generator_upgrade_handler(request: web.Request) -> web.Response:
        user_id = await require_user_id(request)
        try:
            await request.json()
        except Exception:
            pass
        try:
            data = await db.upgrade_generator(int(user_id))
            if data.get("success"):
                cost_payload = data.get("cost") if isinstance(data.get("cost"), dict) else {}
                cost = data.get("amount_spent", cost_payload.get("gems", 0))
                await _track_economy_safe(db, user_id=int(user_id), event_type="spend",
                    resource="gems", amount=cost, source="generator_upgrade",
                    metadata={"old_level": data.get("old_level"), "new_level": data.get("new_level")})
            return web.json_response(data)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка улучшения генератора: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def season_current_handler(request: web.Request) -> web.Response:
        """GET /api/season/current — возвращает активный сезон."""
        try:
            record = await db.get_active_season() if hasattr(db, "get_active_season") else None
            if not record:
                return web.json_response({"error": "no_active_season"}, status=404)

            season = _normalize_extra_pass_season(record)
            end_date = dict(record).get("end_date")
            days_left = max(0, (end_date - datetime.now(timezone.utc)).days) if end_date else 0
            return web.json_response({**season, "season_id": season["id"], "days_left": days_left})
        except Exception as e:
            logging.getLogger(__name__).error("season_current error: %s", e, exc_info=True)
            return web.json_response({"error": "internal_error"}, status=500)

    async def welcome_status_handler(request: web.Request) -> web.Response:
        """Получить статус приветствия и данные о стартовой карте (работает даже если пользователя еще нет)."""
        user_id = await require_user_id(request)

        try:
            # Проверяем, существует ли пользователь
            user_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", user_id)

            if user_exists:
                welcome_status = await db.get_welcome_status(user_id)
                should_show = welcome_status["should_show"]
            else:
                # Пользователя нет - значит нужно показать приветствие
                should_show = True

            # Получаем данные о стартовой карте (ID 9)
            start_card = await db.fetchrow(
                "SELECT id, name, description, rarity, power, image_file_id FROM cards WHERE id = 9"
            )

            result = {
                "should_show": should_show,
                "welcome_shown": not should_show,
                "start_card": dict(start_card) if start_card else None
            }
            return web.json_response(_serialize_datetime(result))
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка получения статуса приветствия: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def welcome_mark_shown_handler(request: web.Request) -> web.Response:
        """Отметить, что приветствие было показано."""
        user_id = await require_user_id(request)

        try:
            await db.mark_welcome_shown(user_id)
            await db.track_onboarding_event(user_id, "welcome_completed", True)
            return web.json_response({"success": True})
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Ошибка отметки приветствия: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_get("/api/dice/status", dice_status_handler)
    app.router.add_post("/api/dice/roll", dice_roll_handler)
    app.router.add_get("/api/dice/notification-prompt-status", dice_notification_prompt_status_handler)
    app.router.add_post("/api/dice/notification-prompt-mark", dice_notification_prompt_mark_handler)
    app.router.add_get("/api/generator/status", generator_status_handler)
    app.router.add_post("/api/generator/claim", generator_claim_handler)
    app.router.add_post("/api/generator/upgrade", generator_upgrade_handler)
    async def welcome_create_user_handler(request: web.Request) -> web.Response:
        """Создать пользователя после завершения приветствия."""
        user_id = await require_user_id(request)

        # Если данных пользователя нет в initData, пытаемся получить через Bot API
        if not first_name_from_data:
            try:
                async with _create_ssl_disabled_session() as session:
                    url = f"https://api.telegram.org/bot{bot_token}/getChat"
                    async with session.get(url, params={"chat_id": user_id}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("ok"):
                                user_info = data.get("result", {})
                                first_name_from_data = user_info.get("first_name")
                                username = username or user_info.get("username")
                                last_name = user_info.get("last_name")
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Не удалось получить данные пользователя через Bot API: {e}")

        # Создаем пользователя (ensure_user автоматически выдаст карту и подарок для нового пользователя)
        await db.ensure_user(
            user_id=user_id,
            username=username,
            first_name=first_name_from_data,
            last_name=last_name,
        )

        await db.track_onboarding_event(user_id, "user_created", True)

        # Отмечаем, что приветствие было показано
        await db.mark_welcome_shown(user_id)

        # Загружаем профиль для ответа
        record = await db.get_user_profile(user_id)
        if not record:
            return web.json_response({"error": "failed_to_create_user"}, status=500)

        settings_record = await db.get_user_settings(user_id)
        settings_data = {}
        if settings_record:
            settings_data = {
                "notif_cases": settings_record["notif_cases"],
                "notif_daily_rewards": settings_record["notif_daily_rewards"],
                "notif_game_invites": settings_record["notif_game_invites"],
                "notif_friend_requests": settings_record["notif_friend_requests"],
                "notif_events": settings_record["notif_events"],
                "notif_news": settings_record["notif_news"],
                "notif_generator": settings_record.get("notif_generator", True),
                "notif_shop": settings_record.get("notif_shop", False),
                "notif_reminders": settings_record.get("notif_reminders", True),
                "notif_squad_member_role": settings_record.get("notif_squad_member_role", True),
                "notif_squad_new_member": settings_record.get("notif_squad_new_member", True),
                "notif_squad_disbanded": settings_record.get("notif_squad_disbanded", True),
                "notif_squad_boost": settings_record.get("notif_squad_boost", True),
                "notif_extra_arena_modifiers": settings_record.get("notif_extra_arena_modifiers", True),
                "ads_enabled": settings_record["ads_enabled"],
                "sound_music": settings_record["sound_music"],
                "sound_sfx": settings_record["sound_sfx"],
                "social_block_friend_requests": settings_record["social_block_friend_requests"],
            }

        title = record.get("title") or "Игрок"

        payload: dict[str, Any] = {
            "user_id": record["user_id"],
            "username": record.get("username"),
            "first_name": first_name_from_data or record.get("first_name"),
            "extra_pass": record.get("extra_pass", "inactive"),
            "trophies": record.get("trophies", 0),
            "max_trophies": record.get("max_trophies", 0),
            "league": record.get("league", 1),
            "keys": record.get("keys", 0),
            "gems": record.get("gems", 0),
            "coins": record.get("coins", 0),
            "squad_id": record.get("squad_id"),
            "status": record.get("status", "active"),
            "reg_date": record["reg_date"].isoformat() if record.get("reg_date") else None,
            "stars": record.get("stars", 0),
            "energy": record.get("energy", 5),
            "energy_cd": record.get("energy_cd").isoformat() if record.get("energy_cd") else None,
            "season": record.get("season", 0),
            "title": title,
            "img": record.get("img", ""),
            "custom_nickname": record.get("custom_nickname"),
            "nickname_changed": record.get("nickname_changed", False),
            "settings": settings_data,
            "should_show_welcome": False,
        }

        return web.json_response(payload)

    app.router.add_get("/api/season/current", season_current_handler)
    app.router.add_get("/api/welcome/status", welcome_status_handler)
    app.router.add_post("/api/welcome/mark-shown", welcome_mark_shown_handler)
    app.router.add_post("/api/welcome/create-user", welcome_create_user_handler)

    # Роуты extraShop
    async def extra_shop_index(request: web.Request) -> web.FileResponse:
        index_path = EXTRA_SHOP_DIR / "index.html"
        if not index_path.exists():
            raise web.HTTPNotFound(text="extraShop page not found")
        return web.FileResponse(index_path, headers={"Cache-Control": "no-store, must-revalidate"})

    app.router.add_get("/extraShop", extra_shop_index)
    app.router.add_get("/extraShop/", extra_shop_index)

    # Legal pages for extraShop
    async def extra_shop_oferta(_: web.Request) -> web.FileResponse:
        path = EXTRA_SHOP_DIR / "oferta.html"
        if not path.exists():
            raise web.HTTPNotFound(text="oferta.html not found")
        return web.FileResponse(path, headers={"Cache-Control": "no-store, must-revalidate"})

    async def extra_shop_privacy(_: web.Request) -> web.FileResponse:
        path = EXTRA_SHOP_DIR / "privacy.html"
        if not path.exists():
            raise web.HTTPNotFound(text="privacy.html not found")
        return web.FileResponse(path, headers={"Cache-Control": "no-store, must-revalidate"})

    async def extra_shop_refund(_: web.Request) -> web.FileResponse:
        path = EXTRA_SHOP_DIR / "refund.html"
        if not path.exists():
            raise web.HTTPNotFound(text="refund.html not found")
        return web.FileResponse(path, headers={"Cache-Control": "no-store, must-revalidate"})

    app.router.add_get("/extraShop/oferta", extra_shop_oferta)
    app.router.add_get("/extraShop/oferta/", extra_shop_oferta)
    app.router.add_get("/extraShop/privacy", extra_shop_privacy)
    app.router.add_get("/extraShop/privacy/", extra_shop_privacy)
    app.router.add_get("/extraShop/refund", extra_shop_refund)
    app.router.add_get("/extraShop/refund/", extra_shop_refund)
    app.router.add_get("/legal/offer", extra_shop_oferta)
    app.router.add_get("/legal/offer/", extra_shop_oferta)
    app.router.add_get("/legal/oferta", extra_shop_oferta)
    app.router.add_get("/legal/oferta/", extra_shop_oferta)
    app.router.add_get("/legal/privacy", extra_shop_privacy)
    app.router.add_get("/legal/privacy/", extra_shop_privacy)
    app.router.add_get("/legal/refund", extra_shop_refund)
    app.router.add_get("/legal/refund/", extra_shop_refund)
    app.router.add_get("/legal/refunds", extra_shop_refund)
    app.router.add_get("/legal/refunds/", extra_shop_refund)

    # Admin panel for shop_sets. The HTML shell is admin-gated too so the
    # endpoint map and client-side control surface are not exposed anonymously.
    async def extra_shop_admin(request: web.Request) -> web.FileResponse:
        user_id = await require_user_id(request)
        if not await _is_admin_user(db, user_id):
            raise web.HTTPForbidden(
                reason="admin_access_required",
                text='{"error":"admin_access_required"}',
                content_type="application/json",
            )
        path = EXTRA_SHOP_DIR / "admin.html"
        if not path.exists():
            raise web.HTTPNotFound(text="admin.html not found")
        response = web.FileResponse(
            path,
            headers={
                "Cache-Control": "no-store, must-revalidate",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": (
                    "default-src 'self'; "
                    "script-src 'self' 'unsafe-inline'; "
                    "style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data: blob: https:; "
                    "connect-src 'self'; "
                    "base-uri 'none'; "
                    "frame-ancestors 'self' https://web.telegram.org"
                ),
            },
        )
        response.set_cookie(
            ADMIN_SESSION_COOKIE_NAME,
            _make_admin_session_token(user_id),
            max_age=ADMIN_SESSION_MAX_AGE_SECONDS,
            httponly=True,
            secure=bool(request.secure or str(request.headers.get("X-Forwarded-Proto", "")).lower() == "https"),
            samesite="Lax",
            path="/",
        )
        return response

    app.router.add_get("/extraShop/admin", extra_shop_admin)
    app.router.add_get("/extraShop/admin/", extra_shop_admin)

    # POST /api/payments/checkout/public/start — публичный магазин без Telegram WebApp auth

    def _normalize_checkout_item_data(data: dict[str, Any]) -> dict[str, Any]:
        """Keep only product selectors from legacy client metadata, never prices or rewards."""
        normalized = dict(data or {})
        metadata = data.get("metadata") if isinstance(data, dict) else None
        if isinstance(metadata, dict):
            for key in ("product_code", "item_type", "package_type", "recipient_id", "ultra"):
                if not normalized.get(key) and metadata.get(key) is not None:
                    normalized[key] = metadata.get(key)
        return normalized

    async def _resolve_ruble_checkout_item(
        db: Database,
        *,
        user_id: int,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Валидация рублевого товара и сбор metadata.

        Сначала ищет товар в ruble_products (по product_code или item_type+package_type),
        затем fallback на старый hardcoded config.
        Возвращает {"item_type", "item_name", "amount_rub", "metadata"} при успехе,
        либо {"error": "код", "message": "текст", "status": http_code} при ошибке.
        """
        from infrastructure.shop_config import GEM_PACKAGES

        item_type = str(data.get("item_type", "") or "")
        package_type = str(data.get("package_type", "") or "")
        product_code = str(data.get("product_code", "") or "")
        recipient_id_raw = data.get("recipient_id")

        metadata: dict[str, Any] = {"user_id": user_id, "item_type": item_type}
        item_name: str = ""
        amount_rub: float = 0.0

        # 1. Try DB ruble_products
        db_product = None
        if product_code:
            db_product = await db.get_ruble_product(product_code)
        if not db_product and item_type:
            # Try matching by item_type + package_type
            all_products = await db.get_ruble_products(active_only=True)
            db_product = next((p for p in all_products if p["item_type"] == item_type and
                               (not package_type or p.get("package_type") == package_type)), None)

        if db_product and db_product.get("is_active"):
            amount_rub = float(db_product.get("price", 0))
            db_item_type = str(db_product.get("item_type") or item_type)
            db_package_type = db_product.get("package_type")
            shop_set_id = db_product.get("shop_set_id")
            if db_item_type == "shop_set" and shop_set_id:
                db_item_type = f"shop_set_{int(shop_set_id)}"

            item_type = db_item_type
            package_type = str(db_package_type or "")
            item_name = str(db_product.get("name", "") or item_type)
            metadata.update({
                "product_code": db_product.get("code"),
                "item_type": item_type,
                "package_type": package_type or None,
                "item_name": item_name,
                "amount_rub": amount_rub,
            })
            if db_product.get("image_url"):
                metadata["image_url"] = db_product["image_url"]
            if shop_set_id:
                metadata["shop_set_id"] = int(shop_set_id)
            if package_type:
                metadata["package_type"] = package_type
                # For gems_package, infer gems count from config or metadata
                if item_type == "gems_package":
                    pkg_cfg = GEM_PACKAGES.get(package_type)
                    if pkg_cfg:
                        metadata["package_gems"] = int(pkg_cfg["gems"])
                    metadata["starter"] = (package_type == "starter_once")

        # 2. Fallback: legacy hardcoded types
        elif item_type == "gems_package":
            if not package_type:
                return {"error": "missing_package_type", "message": "package_type required for gems_package", "status": 400}
            pkg = GEM_PACKAGES.get(package_type)
            if not pkg:
                return {"error": "unknown_package_type", "message": f"Неизвестный пакет: {package_type}", "status": 400}
            if pkg.get("one_time"):
                settings = await db.get_user_settings(user_id)
                if settings and settings.get("starter_pack_used"):
                    return {"error": "already_used", "message": "Стартовый пакет уже куплен", "status": 400}
            gems_count = int(pkg["gems"])
            amount_rub = float(pkg["price"])
            item_name = f"{gems_count} гемов"
            if pkg.get("one_time"):
                item_name += " (стартовый)"
            metadata.update({
                "package_type": package_type,
                "package_gems": gems_count,
                "amount_rub": amount_rub,
                "item_name": item_name,
                "starter": package_type == "starter_once",
            })

        elif item_type == "extrapass":
            amount_rub = 179.0
            item_name = "ExtraPass (30 дней)"
            metadata.update({"item_name": item_name, "amount_rub": amount_rub})

        elif item_type == "extrapass_ultra":
            amount_rub = 349.0
            item_name = "ExtraPass Ultra (30 дней)"
            metadata.update({"item_name": item_name, "amount_rub": amount_rub})

        elif item_type == "extrapass_gift":
            recipient_id = int(recipient_id_raw) if recipient_id_raw else 0
            if not recipient_id or recipient_id <= 0:
                return {"error": "missing_recipient", "message": "recipient_id required for gift", "status": 400}
            gift_ultra = str(data.get("ultra", "") or "").lower() in {"1", "true", "yes"}
            amount_rub = 349.0 if gift_ultra else 179.0
            item_name = ("Ultra " if gift_ultra else "") + "ExtraPass (подарок)"
            metadata.update({
                "item_name": item_name,
                "amount_rub": amount_rub,
                "recipient_id": recipient_id,
                "ultra": gift_ultra,
            })

        elif item_type == "starter_boost":
            amount_rub = 499.0
            item_name = "Starter Boost"
            metadata.update({"item_name": item_name, "amount_rub": amount_rub})

        elif item_type.startswith("shop_set_"):
            try:
                set_id = int(item_type.split("_")[-1])
            except (ValueError, IndexError):
                return {"error": "invalid_set_id", "status": 400}
            set_data = await db.get_shop_set(set_id)
            if not set_data:
                return {"error": "set_not_found", "status": 404}
            if not set_data.get("is_active"):
                return {"error": "set_inactive", "message": "Набор недоступен", "status": 400}
            if set_data.get("currency") != "rubles":
                return {"error": "set_currency_not_rubles", "message": "Набор не продаётся за рубли", "status": 400}
            amount_rub = float(set_data.get("price", 0))
            item_name = str(set_data.get("name", "") or f"Набор #{set_id}")
            metadata.update({
                "item_name": item_name,
                "amount_rub": amount_rub,
                "shop_set_id": set_id,
            })

        else:
            return {"error": "unknown_item_type", "message": f"Неизвестный тип товара: {item_type}", "status": 400}

        if amount_rub <= 0:
            return {"error": "invalid_amount", "status": 400}

        return {
            "item_type": item_type,
            "item_name": item_name,
            "amount_rub": amount_rub,
            "metadata": metadata,
        }

    async def checkout_public_start_handler(request: web.Request) -> web.Response:
        logger = logging.getLogger(__name__)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            telegram_id_raw = data.get("telegram_id") or data.get("recipient_id")
            if not telegram_id_raw:
                return web.json_response(
                    {"error": "missing_telegram_id", "message": "Укажите Telegram ID для зачисления"},
                    status=400,
                )

            try:
                user_id = int(str(telegram_id_raw).strip())
            except (ValueError, TypeError):
                return web.json_response(
                    {"error": "invalid_telegram_id", "message": "Telegram ID должен быть числом"},
                    status=400,
                )

            if user_id <= 0:
                return web.json_response(
                    {"error": "invalid_telegram_id", "message": "Telegram ID должен быть положительным числом"},
                    status=400,
                )

            user_exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", user_id)
            if not user_exists:
                return web.json_response({
                    "error": "user_not_found",
                    "message": "Сначала откройте игру ExtraArena в Telegram хотя бы один раз.",
                }, status=404)

            secret = request.app["bot_token"]

            resolved = await _resolve_ruble_checkout_item(db, user_id=user_id, data=data)
            if "error" in resolved:
                return web.json_response(
                    {"error": resolved["error"], "message": resolved.get("message", "")},
                    status=resolved.get("status", 400),
                )

            item_type = resolved["item_type"]
            item_name = resolved["item_name"]
            amount_rub = resolved["amount_rub"]
            metadata = resolved["metadata"]

            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(minutes=60)
            jti = str(uuid.uuid4())

            payload = {
                "user_id": user_id,
                "item_type": item_type,
                "item_name": item_name,
                "amount_rub": amount_rub,
                "metadata": metadata,
                "exp": int(expires_at.timestamp()),
                "iat": int(now.timestamp()),
                "jti": jti,
            }

            metadata["checkout_jti"] = jti

            session_result = await db.create_checkout_session(
                checkout_jti=jti,
                user_id=user_id,
                item_type=item_type,
                amount=amount_rub,
                metadata=metadata,
                expires_at=expires_at,
            )
            if not session_result.get("success"):
                logger.error("Не удалось создать checkout session %s: %s", jti, session_result.get("error"))
                return web.json_response(
                    {"error": "checkout_session_failed", "message": "Не удалось создать сессию оплаты"},
                    status=500,
                )

            payload_json = _stdlib_json.dumps(payload, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
            payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode()
            signature = hmac.new(secret.encode(), payload_b64.encode(), "sha256").hexdigest()

            token = f"{payload_b64}.{signature}"

            logger.info("Public checkout started: user=%s item=%s amount=%.2f token=%s", user_id, item_type, amount_rub, payload["jti"])

            extra_shop_url = request.app.get("extra_shop_url", request.app.get("webapp_url", ""))
            return web.json_response({
                "success": True,
                "checkout_url": f"{extra_shop_url.rstrip('/')}/extraShop?checkout={token}",
                "checkout_jti": payload["jti"],
                "token": token,
            })

        except Exception as e:
            logger.error("Ошибка checkout_public_start: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    # POST /api/payments/checkout/start — валидация товара и создание checkout-токена
    async def checkout_start_handler(request: web.Request) -> web.Response:
        logger = logging.getLogger(__name__)
        user_id = await require_user_id(request)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            secret = request.app["bot_token"]

            resolved = await _resolve_ruble_checkout_item(db, user_id=user_id, data=data)
            if "error" in resolved:
                return web.json_response(
                    {"error": resolved["error"], "message": resolved.get("message", "")},
                    status=resolved.get("status", 400),
                )

            item_type = resolved["item_type"]
            item_name = resolved["item_name"]
            amount_rub = resolved["amount_rub"]
            metadata = resolved["metadata"]

            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(minutes=60)
            jti = str(uuid.uuid4())

            payload = {
                "user_id": user_id,
                "item_type": item_type,
                "item_name": item_name,
                "amount_rub": amount_rub,
                "metadata": metadata,
                "exp": int(expires_at.timestamp()),
                "iat": int(now.timestamp()),
                "jti": jti,
            }

            metadata["checkout_jti"] = jti

            session_result = await db.create_checkout_session(
                checkout_jti=jti,
                user_id=user_id,
                item_type=item_type,
                amount=amount_rub,
                metadata=metadata,
                expires_at=expires_at,
            )
            if not session_result.get("success"):
                logger.error("Не удалось создать checkout session %s: %s", jti, session_result.get("error"))
                return web.json_response(
                    {"error": "checkout_session_failed", "message": "Не удалось создать сессию оплаты"},
                    status=500,
                )

            payload_json = _stdlib_json.dumps(payload, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
            payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode()
            signature = hmac.new(secret.encode(), payload_b64.encode(), "sha256").hexdigest()

            token = f"{payload_b64}.{signature}"

            logger.info("Checkout started: user=%s item=%s amount=%.2f token=%s", user_id, item_type, amount_rub, payload["jti"])

            extra_shop_url = request.app.get("extra_shop_url", request.app.get("webapp_url", ""))
            return web.json_response({
                "success": True,
                "checkout_url": f"{extra_shop_url.rstrip('/')}/extraShop?checkout={token}",
                "checkout_jti": jti,
            })

        except Exception as e:
            logger.error("Ошибка checkout_start: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": str(e)}, status=500)

    # POST /api/payments/checkout/create — проверка токена, поиск/создание YooKassa payment (идемпотентно)
    async def checkout_create_handler(request: web.Request) -> web.Response:
        logger = logging.getLogger(__name__)

        if request.method != "POST":
            return web.json_response({"error": "method_not_allowed"}, status=405)

        try:
            data = await request.json()
            token = str(data.get("token", "") or "")
            if not token:
                return web.json_response({"error": "missing_token"}, status=400)

            secret = request.app["bot_token"]
            parts = token.rsplit(".", 1)
            if len(parts) != 2:
                return web.json_response({"error": "invalid_token_format"}, status=400)

            payload_b64, provided_sig = parts
            expected_sig = hmac.new(secret.encode(), payload_b64.encode(), "sha256").hexdigest()
            if not hmac.compare_digest(expected_sig, provided_sig):
                return web.json_response({"error": "invalid_signature"}, status=400)

            try:
                padded = payload_b64 + "=" * ((4 - len(payload_b64) % 4) % 4)
                payload_json = base64.urlsafe_b64decode(padded).decode()
                payload = _stdlib_json.loads(payload_json)
            except Exception:
                return web.json_response({"error": "invalid_token_payload"}, status=400)

            exp = int(payload.get("exp", 0))
            if time.time() > exp:
                return web.json_response({"error": "token_expired", "message": "Checkout session expired, please restart purchase"}, status=400)

            jti = str(payload.get("jti", ""))
            if not jti:
                return web.json_response({"error": "missing_jti"}, status=400)

            logger.info("Checkout create requested: jti=%s", jti)

            session = await db.get_checkout_session(jti)
            if not session:
                return web.json_response({"error": "session_not_found", "message": "Сессия оплаты не найдена"}, status=404)

            if session.get("payment_id") and session.get("confirmation_url"):
                logger.info("Checkout session %s уже имеет payment %s, возвращаем существующий", jti, session["payment_id"])
                return web.json_response({
                    "success": True,
                    "confirmation_url": session["confirmation_url"],
                    "payment_id": session["payment_id"],
                })

            user_id = int(payload["user_id"])
            item_type = str(payload.get("item_type", ""))
            item_name = str(payload.get("item_name", ""))
            amount_rub = float(payload.get("amount_rub", 0))
            metadata = dict(payload.get("metadata", {}))
            metadata.setdefault("user_id", user_id)
            metadata.setdefault("item_type", item_type)
            metadata.setdefault("item_name", item_name)
            metadata.setdefault("amount_rub", amount_rub)
            metadata.setdefault("checkout_jti", jti)

            payment_service = request.app.get("payment_service")
            if not payment_service:
                return web.json_response({"error": "payment_service_not_configured"}, status=503)

            webapp_url = request.app.get("webapp_url", "https://t.me/your_bot")
            return_url = str(data.get("return_url") or webapp_url or "")

            logger.info(
                "Creating YooKassa payment for checkout: jti=%s user_id=%s item_type=%s amount=%.2f",
                jti, user_id, item_type, amount_rub,
            )
            payment_result = await asyncio.to_thread(
                payment_service.create_payment,
                amount=amount_rub,
                currency="RUB",
                description=item_name or f"Покупка {item_type}",
                return_url=return_url,
                metadata=metadata,
                idempotence_key=f"checkout:{jti}",
            )

            if not payment_result.get("success"):
                err = payment_result.get("error", "unknown")
                logger.error("YooKassa create_payment failed: %s", err)
                message = f"Ошибка создания платежа: {err}"
                if "api.yookassa.ru" in err or "SSLError" in err or "Max retries exceeded" in err:
                    message = "ЮKassa сейчас недоступна с сервера. Проверьте сеть/VPN на устройстве, где запущен сервер, и попробуйте ещё раз."
                return web.json_response({"success": False, "error": err, "message": message}, status=400)

            payment_id = payment_result.get("payment_id")
            confirmation_url = payment_result.get("confirmation_url")
            logger.info("YooKassa payment created for checkout: jti=%s payment_id=%s", jti, payment_id)

            db_result = await db.create_payment(
                user_id=user_id,
                payment_id=payment_id,
                amount=amount_rub,
                currency="RUB",
                description=item_name or f"Платеж {payment_id}",
                metadata=metadata,
            )
            if not db_result.get("success"):
                logger.error("Не удалось сохранить платеж %s в БД: %s", payment_id, db_result.get("error"))
                return web.json_response({
                    "success": False,
                    "error": "payment_record_not_saved",
                    "message": "Платеж не был создан. Попробуйте еще раз.",
                }, status=500)

            attach_result = await db.attach_checkout_payment(
                checkout_jti=jti,
                payment_id=payment_id,
                confirmation_url=confirmation_url,
            )
            if not attach_result.get("success"):
                existing = await db.get_checkout_session(jti)
                if existing and existing.get("confirmation_url"):
                    return web.json_response({
                        "success": True,
                        "confirmation_url": existing["confirmation_url"],
                        "payment_id": existing.get("payment_id"),
                    })
                logger.warning("Не удалось прикрепить payment к session %s: %s", jti, attach_result.get("error"))

            return web.json_response({
                "success": True,
                "confirmation_url": confirmation_url,
                "payment_id": payment_id,
            })

        except Exception as e:
            logger.error("Ошибка checkout_create: %s", e, exc_info=True)
            return web.json_response({"error": "internal_server_error", "message": "Не удалось создать платеж"}, status=500)

    app.router.add_post("/api/payments/checkout/public/start", checkout_public_start_handler)
    app.router.add_post("/api/payments/checkout/start", checkout_start_handler)
    app.router.add_post("/api/payments/checkout/create", checkout_create_handler)

    # Register ExtraID handlers
    register_extraid_handlers(
        app,
        extraid_db,
        _verify_init_data,
        _validate_auth_date,
        _extract_user_id_from_init_data,
        _verify_jwt_token_async,
        _display_id_generator,
        _mask_email,
        _nickname_valid,
        _get_ip_country,
        require_user_id,
    )

    async def _json_payload_from_existing_handler(handler, request: web.Request) -> tuple[int, dict[str, Any]]:
        response = await handler(request)
        status = getattr(response, "status", 200)
        raw_body = getattr(response, "body", None) or b"{}"
        if isinstance(raw_body, str):
            body_text = raw_body
        else:
            body_text = raw_body.decode("utf-8", errors="replace")
        try:
            payload = _stdlib_json.loads(body_text or "{}")
        except Exception:
            payload = {}
        return int(status), payload if isinstance(payload, dict) else {"data": payload}

    def _bootstrap_status(statuses: dict[str, int]) -> int:
        return 500 if any(status >= 500 for status in statuses.values()) else 200

    async def mobile_bootstrap_handler(request: web.Request) -> web.Response:
        profile_status, profile_payload = await _json_payload_from_existing_handler(profile_handler, request)
        runtime_status, runtime_payload = await _json_payload_from_existing_handler(runtime_status_handler, request)
        payload = {
            "profile": profile_payload if profile_status < 400 else None,
            "profile_error": profile_payload if profile_status >= 400 else None,
            "profile_status": profile_status,
            "runtime_status": runtime_payload,
            "runtime_status_code": runtime_status,
            "server_time": int(time.time()),
        }
        return web.json_response(_serialize_datetime(payload), status=_bootstrap_status({
            "profile": profile_status,
            "runtime_status": runtime_status,
        }))

    async def mobile_shop_bootstrap_handler(request: web.Request) -> web.Response:
        sets_status, sets_payload = await _json_payload_from_existing_handler(shop_sets_public_handler, request)
        shop_catalog_status, shop_catalog_payload = await _json_payload_from_existing_handler(shop_catalog_handler, request)
        particles_status, particles_payload = await _json_payload_from_existing_handler(particles_daily_handler, request)
        payload = {
            "shop_sets": sets_payload if sets_status < 400 else None,
            "shop_catalog": shop_catalog_payload if shop_catalog_status < 400 else None,
            "particles_daily": particles_payload if particles_status < 400 else None,
            "statuses": {
                "shop_sets": sets_status,
                "shop_catalog": shop_catalog_status,
                "particles_daily": particles_status,
            },
            "server_time": int(time.time()),
        }
        return web.json_response(_serialize_datetime(payload), status=_bootstrap_status(payload["statuses"]))

    async def mobile_shop_particles_widget_handler(request: web.Request) -> web.Response:
        particles_status, particles_payload = await _json_payload_from_existing_handler(particles_daily_handler, request)
        payload = {
            "particles_daily": particles_payload if particles_status < 400 else {"cards": []},
            "status": particles_status,
            "server_time": int(time.time()),
        }
        return web.json_response(_serialize_datetime(payload), status=particles_status if particles_status >= 400 else 200)

    async def mobile_collection_bootstrap_handler(request: web.Request) -> web.Response:
        collection_status, collection_payload = await _json_payload_from_existing_handler(collection_with_status_handler, request)
        presets_status, presets_payload = await _json_payload_from_existing_handler(deck_presets_list_handler, request)
        payload = {
            "collection": collection_payload if collection_status < 400 else None,
            "deck_presets": presets_payload if presets_status < 400 else None,
            "statuses": {
                "collection": collection_status,
                "deck_presets": presets_status,
            },
            "server_time": int(time.time()),
        }
        return web.json_response(_serialize_datetime(payload), status=_bootstrap_status(payload["statuses"]))

    async def mobile_battle_bootstrap_handler(request: web.Request) -> web.Response:
        presets_status, presets_payload = await _json_payload_from_existing_handler(deck_presets_list_handler, request)
        modes_status, modes_payload = await _json_payload_from_existing_handler(match_modes_handler, request)
        payload = {
            "deck_presets": presets_payload if presets_status < 400 else None,
            "match_modes": modes_payload if modes_status < 400 else None,
            "statuses": {
                "deck_presets": presets_status,
                "match_modes": modes_status,
            },
            "server_time": int(time.time()),
        }
        return web.json_response(_serialize_datetime(payload), status=_bootstrap_status(payload["statuses"]))

    async def mobile_squads_bootstrap_handler(request: web.Request) -> web.Response:
        squads_status, squads_payload = await _json_payload_from_existing_handler(squads_me_handler, request)
        shop_status, shop_payload = await _json_payload_from_existing_handler(squads_shop_handler, request)
        payload = {
            "squads_me": squads_payload if squads_status < 400 else None,
            "squads_shop": shop_payload if shop_status < 400 else None,
            "statuses": {
                "squads_me": squads_status,
                "squads_shop": shop_status,
            },
            "server_time": int(time.time()),
        }
        return web.json_response(_serialize_datetime(payload), status=_bootstrap_status(payload["statuses"]))

    app.router.add_get("/api/mobile/bootstrap", mobile_bootstrap_handler)
    app.router.add_get("/api/mobile/shop-bootstrap", mobile_shop_bootstrap_handler)
    app.router.add_get("/api/mobile/shop-particles-widget", mobile_shop_particles_widget_handler)
    app.router.add_get("/api/mobile/collection-bootstrap", mobile_collection_bootstrap_handler)
    app.router.add_get("/api/mobile/battle-bootstrap", mobile_battle_bootstrap_handler)
    app.router.add_get("/api/mobile/squads-bootstrap", mobile_squads_bootstrap_handler)

    app.router.add_get("/{path:.*}", static_handler)

    # ============================================
    # ФОНОВАЯ ЗАДАЧА: АВТО-ЗАВЕРШЕНИЕ ХОДА
    # ============================================

    async def match_timer_checker(app: web.Application) -> None:
        """
        Фоновая задача для автоматического завершения хода по истечении времени.
        Запускается раз в секунду и проверяет все активные матчи.
        """
        logger = logging.getLogger(__name__)
        logger.info("🕒 Match timer checker started")

        try:
            while True:
                await asyncio.sleep(1)  # Проверка каждую секунду
                _prune_finished_match_runtime()
                _prune_action_result_cache()

                # Получаем копию активных матчей для безопасной итерации
                matches_to_check = list(ACTIVE_MATCHES.items())

                for match_id, engine in matches_to_check:
                    try:
                        # Пропускаем завершённые матчи
                        if hasattr(engine, 'is_ended') and engine.is_ended:
                            continue

                        # Получаем текущего игрока для проверки таймера
                        current_player_id = engine.get_current_player_id() if hasattr(engine, 'get_current_player_id') else None

                        # Получаем состояние матча для текущего игрока
                        state = engine.get_full_state(viewer_id=current_player_id) if hasattr(engine, 'get_full_state') else {}

                        # Проверяем, не истекло ли время
                        time_remaining = state.get('turn_time_remaining', 99)

                        if time_remaining <= 0 and current_player_id is not None:
                            logger.warning(
                                "⏰ Auto-ending turn for match %s (player %s) - time expired",
                                match_id, current_player_id
                            )
                            await _handle_natural_turn_timeout(app, str(match_id), engine)

                    except Exception as exc:
                        logger.error(
                            "❌ Error checking timer for match %s: %s",
                            match_id, exc
                        )

        except asyncio.CancelledError:
            logger.info("🛑 Match timer checker stopped")
        except Exception as exc:
            logger.error("❌ Match timer checker fatal error: %s", exc, exc_info=True)

    async def _announcement_expiry_loop(app: web.Application) -> None:
        """Фоновая задача: помечать истёкшие объявления каждые 60 секунд."""
        logger = logging.getLogger(__name__)
        try:
            while True:
                await asyncio.sleep(60)
                try:
                    expired = await db.expire_announcements()
                    if expired:
                        logger.info("Expired %d announcements", expired)
                except Exception as exc:
                    logger.error("Error expiring announcements: %s", exc)
        except asyncio.CancelledError:
            pass

    async def _squad_weekly_cbrp_loop(app: web.Application) -> None:
        """Фоновая задача: закрывать недельные дельты трофеев для сквадов."""
        logger = logging.getLogger(__name__)
        try:
            while True:
                try:
                    result = await db.process_weekly_squad_cbrp()
                    if result.get("processed") and (result.get("closed") or result.get("awarded")):
                        logger.info(
                            "Squad weekly CBRP processed: closed=%s awarded=%s cbrp=%s period=%s",
                            result.get("closed"),
                            result.get("awarded"),
                            result.get("cbrp"),
                            result.get("period_key"),
                        )
                except Exception as exc:
                    logger.error("Error processing squad weekly CBRP: %s", exc, exc_info=True)
                await asyncio.sleep(300)
        except asyncio.CancelledError:
            pass

    async def start_background_tasks(app: web.Application) -> None:
        """Запуск фоновых задач при старте сервера."""
        app['match_timer_task'] = asyncio.create_task(match_timer_checker(app))
        app['announcement_expiry_task'] = asyncio.create_task(_announcement_expiry_loop(app))
        app['squad_weekly_cbrp_task'] = asyncio.create_task(_squad_weekly_cbrp_loop(app))

    async def cleanup_background_tasks(app: web.Application) -> None:
        """Остановка фоновых задач при остановке сервера."""
        if 'match_timer_task' in app:
            app['match_timer_task'].cancel()
            try:
                await app['match_timer_task']
            except asyncio.CancelledError:
                pass
        if 'announcement_expiry_task' in app:
            app['announcement_expiry_task'].cancel()
            try:
                await app['announcement_expiry_task']
            except asyncio.CancelledError:
                pass
        if 'squad_weekly_cbrp_task' in app:
            app['squad_weekly_cbrp_task'].cancel()
            try:
                await app['squad_weekly_cbrp_task']
            except asyncio.CancelledError:
                pass

    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    return app
