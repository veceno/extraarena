from __future__ import annotations

import hashlib
import importlib
import json
import time
import uuid
from base64 import b64encode
from io import BytesIO
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import jwt
import pytest
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from infrastructure.config import DatabaseSettings, get_settings
from infrastructure.database import Database
from infrastructure.case_config import (
    build_default_case_config,
    merge_case_config_patch,
    validate_case_config,
)
from web import mcp_admin_tools
from web import server as web_server


STRONG_TEST_JWT_SECRET = "test-mcp-jwt-secret-that-is-long-enough-2026"
STRONG_TEST_ADMIN_SECRET = "test-mcp-admin-secret-that-is-long-enough-2026"
STRONG_TEST_MCP_SECRET = "test-mcp-token-secret-that-is-long-enough-2026"
MCP_AUDIENCE = "extraarena:mcp-test"


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _set_base_production_env(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", STRONG_TEST_ADMIN_SECRET)
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://game.example")
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://ops.example")


def _mcp_settings(*, mcp_secret: str = STRONG_TEST_MCP_SECRET, jwt_secret: str = STRONG_TEST_JWT_SECRET):
    return SimpleNamespace(
        mcp_enabled=True,
        mcp_token_secret=mcp_secret,
        mcp_token_ttl_seconds=120,
        jwt_secret=jwt_secret,
    )


def _mcp_auth_module():
    return importlib.import_module("web.mcp_auth")


def test_mcp_config_defaults_to_enabled_with_dev_safe_values(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("MCP_ENABLED", raising=False)
    monkeypatch.delenv("MCP_TOKEN_SECRET", raising=False)
    monkeypatch.delenv("MCP_ENDPOINT_PATH", raising=False)
    monkeypatch.delenv("MCP_SESSION_PATH", raising=False)
    monkeypatch.delenv("MCP_TOKEN_TTL_SECONDS", raising=False)
    monkeypatch.delenv("MCP_ALLOWED_ORIGINS", raising=False)

    settings = get_settings()

    assert getattr(settings, "mcp_enabled", None) is True
    assert settings.mcp_token_secret != settings.jwt_secret
    assert settings.mcp_endpoint_path.startswith("/")
    assert settings.mcp_session_path.startswith("/api/admin/")
    assert settings.mcp_token_ttl_seconds > 0
    assert settings.mcp_allowed_origins == ("*",)


def test_production_mcp_enabled_requires_strong_separate_token_secret(monkeypatch):
    _set_base_production_env(monkeypatch)
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("MCP_ENDPOINT_PATH", "admin/mcp")
    monkeypatch.setenv("MCP_TOKEN_TTL_SECONDS", "120")
    monkeypatch.delenv("MCP_TOKEN_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="MCP_TOKEN_SECRET"):
        get_settings()

    monkeypatch.setenv("MCP_TOKEN_SECRET", "too-short")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="MCP_TOKEN_SECRET"):
        get_settings()

    monkeypatch.setenv("MCP_TOKEN_SECRET", STRONG_TEST_JWT_SECRET)
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="MCP_TOKEN_SECRET"):
        get_settings()

    monkeypatch.setenv("MCP_TOKEN_SECRET", STRONG_TEST_MCP_SECRET)
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.mcp_enabled is True
    assert settings.mcp_token_secret == STRONG_TEST_MCP_SECRET
    assert settings.mcp_endpoint_path == "/admin/mcp"
    assert settings.mcp_session_path == "/api/admin/mcp/session"
    assert settings.mcp_token_ttl_seconds == 120
    assert settings.mcp_allowed_origins == ("https://ops.example",)


def test_mcp_auth_mints_and_verifies_mcp_claims():
    mcp_auth = _mcp_auth_module()
    now = int(time.time())
    settings = _mcp_settings()

    token = mcp_auth.mint_mcp_token(
        admin_user_id=101,
        scopes=("mcp:admin", "users:read"),
        settings=settings,
        aud=MCP_AUDIENCE,
        issued_at=now,
        jti="test-jti",
    )

    claims = mcp_auth.verify_mcp_token(
        token,
        settings=settings,
        aud=MCP_AUDIENCE,
        required_scopes=("mcp:admin",),
    )

    assert claims["typ"] == "ea_mcp"
    assert claims["aud"] == MCP_AUDIENCE
    assert claims["sub"] == "101"
    assert claims["admin_user_id"] == 101
    assert claims["scopes"] == ["mcp:admin", "users:read"]
    assert claims["jti"] == "test-jti"
    assert claims["iat"] == now
    assert claims["exp"] == now + settings.mcp_token_ttl_seconds


def test_mcp_auth_rejects_non_mcp_tokens():
    mcp_auth = _mcp_auth_module()
    now = int(time.time())
    settings = _mcp_settings()
    base_claims = {
        "typ": "ea_mcp",
        "aud": MCP_AUDIENCE,
        "sub": "101",
        "admin_user_id": 101,
        "scopes": ["mcp:admin"],
        "jti": "test-jti",
        "iat": now,
        "exp": now + 60,
    }

    def raw_token(payload):
        return jwt.encode(payload, settings.mcp_token_secret, algorithm="HS256")

    wrong_typ = raw_token({**base_claims, "typ": "admin_session"})
    wrong_aud = raw_token({**base_claims, "aud": "extraarena:not-mcp"})
    expired = raw_token({**base_claims, "exp": now - 1})
    missing_scope = raw_token({**base_claims, "scopes": ["users:read"]})
    malformed = "not-a-jwt"

    for token in (wrong_typ, wrong_aud, expired, missing_scope, malformed):
        assert mcp_auth.verify_mcp_token(
            token,
            settings=settings,
            aud=MCP_AUDIENCE,
            required_scopes=("mcp:admin",),
        ) is None


def test_mcp_auth_rejects_extraarena_jwt_shape_even_with_same_secret():
    mcp_auth = _mcp_auth_module()
    now = int(time.time())
    settings = _mcp_settings(
        mcp_secret=STRONG_TEST_JWT_SECRET,
        jwt_secret=STRONG_TEST_JWT_SECRET,
    )
    extraarena_jwt = jwt.encode(
        {
            "user_id": 101,
            "session_id": str(uuid.uuid4()),
            "iat": now,
            "exp": now + 60,
        },
        settings.jwt_secret,
        algorithm="HS256",
    )

    assert mcp_auth.verify_mcp_token(
        extraarena_jwt,
        settings=settings,
        aud=MCP_AUDIENCE,
        required_scopes=("mcp:admin",),
    ) is None


class GatewayExtraIDDB:
    def __init__(self, session_id: str, user_id: int = 101):
        self.session_id = session_id
        self.user_id = user_id

    async def verify_session(self, session_uuid, token: str):
        if str(session_uuid) != self.session_id:
            return None
        return {"session_id": session_uuid, "user_id": self.user_id}


class GatewayMCPDB:
    def __init__(self):
        self.runtime_config = {
            "maintenance_mode": {"enabled": False},
            "feature_availability": {"shop": True},
            "disabled_card_ids": [],
        }
        self.runtime_config_updates = []
        self.case_config = build_default_case_config()
        self.case_config_updates = []
        self.created_shop_sets = []
        self.created_season_drafts = []
        self.updated_seasons = []
        self.replaced_reward_tracks = []
        self.executed_season_resets = []
        self.extra_pass_updates = []
        self.audit_calls = []
        self.rate_limit_calls = []
        self.confirmations = {}
        self.idempotency = {}
        self.fail_runtime_config = False
        self.cards = {46: {"id": 46, "name": "Warrior", "card_type": "warrior"}}
        self.shop_sets = {1: {"id": 1, "name": "Arena Set", "is_active": True}}
        self.ruble_products = {}
        self.cosmetics = {}
        self.next_cosmetic_id = 1
        self.reward_tracks = [{"id": 1, "track_type": "s1_free", "position": 1, "is_active": True}]

    async def is_admin(self, user_id):
        return int(user_id) == 101

    async def get_runtime_config(self):
        if self.fail_runtime_config:
            raise RuntimeError("secret_runtime_table leaked")
        return dict(self.runtime_config)

    async def set_runtime_config(self, **kwargs):
        self.runtime_config_updates.append(kwargs)
        for key, value in kwargs.items():
            if value is not None:
                self.runtime_config[key] = value
        return dict(self.runtime_config)

    async def get_case_config(self):
        import copy as _copy
        return _copy.deepcopy(self.case_config)

    async def set_case_config(self, *, patch=None):
        if not isinstance(patch, dict) or not patch:
            raise ValueError("empty_case_config_patch")
        current = self.case_config
        merged = merge_case_config_patch(current, patch)
        validate_case_config(merged)
        self.case_config_updates.append(patch)
        self.case_config = merged
        return merged

    async def get_match_mode_overrides(self):
        return [{"mode_id": "classic", "enabled": True}]

    async def get_ruble_products(self, active_only=False, surface=None):
        return list(self.ruble_products.values()) or [{"code": "starter", "name": "Starter", "is_active": True}]

    async def get_ruble_product(self, code):
        return self.ruble_products.get(str(code))

    async def get_shop_sets(self, active_only=False):
        return list(self.shop_sets.values())

    async def get_shop_set(self, set_id):
        return self.shop_sets.get(int(set_id))

    async def validate_shop_set_rewards(self, rewards):
        normalized = mcp_admin_tools._normalize_shop_set_rewards(rewards)
        for reward in normalized:
            if reward.get("type") == "card" and int(reward.get("card_id") or 0) not in self.cards:
                return [], "reward_card_not_found"
            if reward.get("type") == "particles" and int(reward.get("card_id") or 0) not in self.cards:
                return [], "reward_card_not_found"
            if reward.get("type") == "cosmetic":
                item = next(
                    (
                        row for row in self.cosmetics.values()
                        if row.get("slug") == reward.get("cosmetic_slug") and row.get("is_active", True)
                    ),
                    None,
                )
                if not item:
                    return [], "reward_cosmetic_not_found"
        return normalized, None

    async def list_cosmetic_items(self, *, active_only=False, item_type=None):
        rows = list(self.cosmetics.values())
        if active_only:
            rows = [row for row in rows if row.get("is_active", True)]
        if item_type:
            rows = [row for row in rows if row.get("item_type") == item_type]
        return rows

    async def get_cosmetic_item(self, identity):
        if isinstance(identity, int) or str(identity).isdigit():
            return self.cosmetics.get(int(identity))
        return next((row for row in self.cosmetics.values() if row.get("slug") == str(identity)), None)

    async def create_cosmetic_item(self, **kwargs):
        if any(row.get("slug") == kwargs.get("slug") for row in self.cosmetics.values()):
            return {"success": False, "error": "cosmetic_slug_exists"}
        cosmetic_id = self.next_cosmetic_id
        self.next_cosmetic_id += 1
        row = {"id": cosmetic_id, "is_active": True, "sort_order": 0, **kwargs}
        self.cosmetics[cosmetic_id] = row
        return {"success": True, "cosmetic": row}

    async def update_cosmetic_item(self, cosmetic_id, **kwargs):
        cosmetic_id = int(cosmetic_id)
        if cosmetic_id not in self.cosmetics:
            return {"success": False, "error": "cosmetic_not_found"}
        self.cosmetics[cosmetic_id].update(kwargs)
        return {"success": True, "cosmetic": self.cosmetics[cosmetic_id]}

    async def delete_cosmetic_item(self, cosmetic_id):
        cosmetic_id = int(cosmetic_id)
        if cosmetic_id not in self.cosmetics:
            return {"success": False, "error": "cosmetic_not_found"}
        self.cosmetics[cosmetic_id]["is_active"] = False
        return {"success": True, "cosmetic": self.cosmetics[cosmetic_id]}

    async def create_shop_set(
        self,
        *,
        name,
        description=None,
        image_file_id=None,
        price,
        currency="rubles",
        created_by,
        rewards=None,
    ):
        record = {
            "name": name,
            "description": description,
            "image_file_id": image_file_id,
            "price": price,
            "currency": currency,
            "created_by": created_by,
            "rewards": rewards or [],
        }
        self.created_shop_sets.append(record)
        return {"success": True, "set_id": len(self.created_shop_sets) + 100}

    async def get_seasons(self):
        return [{"id": 1, "slug": "s1", "status": "active"}]

    async def get_all_reward_tracks(self):
        return list(self.reward_tracks)

    async def get_reward_track_by_id(self, reward_id):
        return {
            "id": int(reward_id),
            "track_type": "bp_ultra",
            "position": 41,
            "reward_type": "coins",
            "reward_amount": 100,
            "reward_meta": {},
            "extra_pass_required": True,
        }

    async def get_season_reset_summaries(self):
        return {}

    async def get_season_by_id(self, season_id):
        return {
            "id": int(season_id),
            "slug": "s1",
            "name": "Season 1",
            "season_number": 1,
            "status": "active",
            "is_active": True,
            "max_stars": 45,
            "free_track_type": "bp_free",
            "pass_track_type": "bp_premium",
            "ultra_track_type": "bp_ultra",
            "pass_end_position": 40,
            "ultra_start_position": 41,
        }

    async def create_season_draft(self, preset_key="blank"):
        self.created_season_drafts.append(preset_key)
        return {"id": 2, "status": "draft", "preset_key": preset_key}

    async def update_season(self, season_id, **fields):
        self.updated_seasons.append((int(season_id), fields))
        return {"id": int(season_id), **fields}

    async def replace_reward_tracks(self, track_types, rows):
        self.replaced_reward_tracks.append((list(track_types), rows))
        return [{"id": index + 1, **row} for index, row in enumerate(rows)]

    async def get_card_info(self, card_id):
        return self.cards.get(int(card_id))

    async def preview_season_reset(self, season_id, sample_limit=200):
        return {
            "season_id": int(season_id),
            "already_completed": False,
            "summary": {"players": 3, "trophies_reduced": 120, "keys_granted": 1, "coins_granted": 200, "stars_reset": 9},
            "players": [],
            "players_limit": int(sample_limit),
            "players_truncated": False,
        }

    async def execute_season_reset(self, **kwargs):
        self.executed_season_resets.append(kwargs)
        return {
            "season_id": int(kwargs["season_id"]),
            "status": "completed",
            "processed_players": 3,
        }

    async def count_push_devices(self, platform="android"):
        return 2

    async def get_admin_analytics_overview(self, days=30):
        return {"days": days, "users": {"total": 3}}

    async def search_admin_players(self, **kwargs):
        return {"players": [{"user_id": 201, "username": "pilot"}], "total": 1}

    async def get_admin_player_detail(self, user_id):
        return {"profile": {"user_id": int(user_id)}, "balances": {"coins": 10}}

    async def process_weekly_squad_cbrp(self):
        return {"processed": False}

    async def refresh_due_rating_snapshots(self, scope="players"):
        return {"refreshed": []}

    async def expire_announcements(self):
        return 0

    async def admin_note_user(self, admin_user_id, target_user_id, note):
        return {"status": "ok", "action": "note", "target_user_id": int(target_user_id)}

    async def admin_adjust_resource(self, admin_user_id, target_user_id, resource, amount, reason=None):
        return {
            "status": "ok",
            "action": "grant",
            "target_user_id": int(target_user_id),
            "resource": resource,
            "amount": int(amount),
            "reason": reason,
        }

    async def admin_set_extra_pass(self, admin_user_id, target_user_id, mode, days=None, reason=None):
        self.extra_pass_updates.append({
            "admin_user_id": int(admin_user_id),
            "target_user_id": int(target_user_id),
            "mode": mode,
            "days": days,
            "reason": reason,
        })
        return {"status": "ok", "action": "set_extra_pass", "mode": mode}

    async def check_mcp_rate_limit(self, *, scope, subject, max_requests, window_seconds):
        self.rate_limit_calls.append({
            "scope": scope,
            "subject": subject,
            "max_requests": max_requests,
            "window_seconds": window_seconds,
        })
        return {"allowed": True, "count": 1}

    async def record_mcp_tool_call(self, **kwargs):
        self.audit_calls.append(kwargs)
        return {"id": len(self.audit_calls), **kwargs}

    async def create_mcp_confirmation(self, **kwargs):
        confirmation_id = f"confirm-{len(self.confirmations) + 1}"
        self.confirmations[confirmation_id] = {**kwargs, "confirmation_id": confirmation_id}
        return {
            "confirmation_id": confirmation_id,
            "expires_at": "2026-06-07T00:05:00+00:00",
            "args_digest": kwargs.get("args_digest"),
        }

    async def consume_mcp_confirmation(self, **kwargs):
        confirmation = next(
            (
                value
                for value in self.confirmations.values()
                if value.get("confirmation_token") == kwargs.get("confirmation_token")
            ),
            None,
        )
        if not confirmation:
            return {"success": False, "error": "confirmation_not_found"}
        if confirmation.get("args_digest") != kwargs.get("args_digest"):
            return {"success": False, "error": "confirmation_args_mismatch"}
        if confirmation.get("consumed"):
            return {"success": False, "error": "confirmation_already_used"}
        confirmation["consumed"] = True
        return {"success": True, "confirmation_id": confirmation.get("confirmation_id")}

    async def reserve_mcp_idempotency_key(self, **kwargs):
        key = (kwargs.get("admin_user_id"), kwargs.get("tool_name"), kwargs.get("idempotency_key"))
        existing = self.idempotency.get(key)
        if existing and existing.get("force_in_progress"):
            return {"status": "in_progress", "args_match": True, "reserved": False, "replayable": False}
        if existing and existing.get("args_digest") != kwargs.get("args_digest"):
            return {"status": "conflict", "error": "idempotency_key_conflict"}
        if existing and existing.get("response") is not None:
            return {"status": "replay", "response": existing["response"], "replayable": True, "reserved": False}
        if existing:
            return {"status": "in_progress", "args_match": True, "reserved": False, "replayable": False}
        self.idempotency[key] = {**kwargs, "response": None}
        return {"status": "in_progress", "reserved": True, "args_match": True, "replayable": False}

    async def complete_mcp_idempotency_key(self, **kwargs):
        key = (kwargs.get("admin_user_id"), kwargs.get("tool_name"), kwargs.get("idempotency_key"))
        self.idempotency.setdefault(key, {}).update({"response": kwargs.get("response")})
        return {"status": "stored"}


def _set_gateway_env(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", STRONG_TEST_ADMIN_SECRET)
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("MCP_TOKEN_SECRET", STRONG_TEST_MCP_SECRET)
    monkeypatch.setenv("MCP_ENDPOINT_PATH", "/ops/mcp")
    monkeypatch.delenv("MCP_SESSION_PATH", raising=False)
    monkeypatch.setenv("MCP_TOKEN_TTL_SECONDS", "120")
    monkeypatch.delenv("MCP_ALLOWED_ORIGINS", raising=False)
    get_settings.cache_clear()


def _extraarena_auth_token(session_id: str, user_id: int = 101) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "user_id": user_id,
            "session_id": session_id,
            "iat": now,
            "exp": now + 600,
        },
        get_settings().jwt_secret,
        algorithm="HS256",
    )


async def _gateway_client(monkeypatch, db=None, *, mcp_allowed_origins: str | None = None):
    _set_gateway_env(monkeypatch)
    if mcp_allowed_origins is not None:
        monkeypatch.setenv("MCP_ALLOWED_ORIGINS", mcp_allowed_origins)
        get_settings.cache_clear()
    session_id = str(uuid.uuid4())
    db = db or GatewayMCPDB()
    app = web_server.create_web_app(
        db,
        bot_token="bot-token",
        extraid_db=GatewayExtraIDDB(session_id),
        webapp_url="https://game.example",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, db, _extraarena_auth_token(session_id)


async def _mcp_token(client, admin_auth_token: str) -> str:
    response = await client.post(
        get_settings().mcp_session_path,
        headers={"Authorization": f"Bearer {admin_auth_token}"},
    )
    assert response.status == 200
    data = await response.json()
    assert data["endpoint"] == "/ops/mcp"
    return data["token"]


async def _mcp_call(client, token: str, method: str, params=None, *, request_id=1):
    return await client.post(
        "/ops/mcp",
        headers={"Authorization": f"Bearer {token}"},
        json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}},
    )


@pytest.mark.asyncio
async def test_mcp_admin_session_mints_token_for_configured_endpoint(monkeypatch):
    client, _db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        claims = _mcp_auth_module().verify_mcp_token(
            token,
            settings=get_settings(),
            required_scopes=("mcp:admin", "admin:runtime:read", "admin:economy:grant"),
        )

        assert claims is not None
        assert claims["admin_user_id"] == 101
        assert claims["typ"] == "ea_mcp"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_endpoint_rejects_query_cookie_and_regular_extraarena_jwt(monkeypatch):
    client, _db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        mcp_token = await _mcp_token(client, admin_auth_token)

        query_response = await client.post(
            f"/ops/mcp?_auth={mcp_token}",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        cookie_response = await client.post(
            "/ops/mcp",
            cookies={"ea_admin_session": "pretend-admin-cookie"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        bearer_with_cookie_response = await client.post(
            "/ops/mcp",
            headers={"Authorization": f"Bearer {mcp_token}", "Cookie": "ea_admin_session=pretend-admin-cookie"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        regular_jwt_response = await client.post(
            "/ops/mcp",
            headers={"Authorization": f"Bearer {admin_auth_token}"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )

        assert query_response.status == 401
        assert cookie_response.status == 401
        assert bearer_with_cookie_response.status == 401
        assert regular_jwt_response.status == 401
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_session_and_endpoint_enforce_origin_allowlist(monkeypatch):
    client, _db, admin_auth_token = await _gateway_client(
        monkeypatch,
        mcp_allowed_origins="https://ops.example",
    )
    try:
        bad_session = await client.post(
            get_settings().mcp_session_path,
            headers={
                "Authorization": f"Bearer {admin_auth_token}",
                "Origin": "https://evil.example",
            },
        )
        good_session = await client.post(
            get_settings().mcp_session_path,
            headers={
                "Authorization": f"Bearer {admin_auth_token}",
                "Origin": "https://ops.example",
            },
        )
        token = (await good_session.json())["token"]
        bad_tool = await client.post(
            "/ops/mcp",
            headers={
                "Authorization": f"Bearer {token}",
                "Origin": "https://evil.example",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )

        assert bad_session.status == 403
        assert good_session.status == 200
        assert bad_tool.status == 403
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_tools_are_allowlisted_and_exclude_raw_proxy_surfaces(monkeypatch):
    client, _db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(client, token, "tools/list")
        payload = await response.json()
        names = {tool["name"] for tool in payload["result"]["tools"]}

        assert response.status == 200
        assert "admin.runtime.config.read" in names
        assert "admin.players.resource.grant" in names
        assert not any("raw" in name or "sql" in name or "http" in name for name in names)
    finally:
        await client.close()


def test_mcp_mutating_capabilities_require_confirmation_and_idempotency():
    from web.admin_capabilities import ADMIN_CAPABILITIES

    mutating = [capability for capability in ADMIN_CAPABILITIES if capability.mutating]

    assert mutating
    assert [capability.id for capability in mutating if not capability.dry_run_required] == []
    assert [capability.id for capability in mutating if not capability.idempotency_required] == []
    for capability in mutating:
        properties = capability.input_schema.get("properties") or {}
        assert "dry_run" in properties
        assert "idempotency_key" in properties
        assert "confirmation_token" in properties


def test_admin_shop_schemas_expose_gift_sets_and_cosmetic_rewards():
    from web.admin_capabilities import ADMIN_CAPABILITIES
    from web.commerce_admin_mcp_specs import COMMERCE_ADMIN_CAPABILITY_SPECS

    capability_by_id = {capability.id: capability for capability in ADMIN_CAPABILITIES}
    create_set_rewards = capability_by_id["admin.shop.sets.create"].input_schema["properties"]["rewards"]["items"]
    create_product_item_type = capability_by_id["admin.shop.products.create"].input_schema["properties"]["item_type"]

    assert "cosmetic" in create_set_rewards["properties"]["type"]["enum"]
    assert "cosmetic_slug" in create_set_rewards["properties"]
    assert create_set_rewards["properties"]["cosmetic_slug"]["minLength"] == 1
    assert create_set_rewards["properties"]["auto_equip"]["type"] == "boolean"
    assert "gift_shop_set" in create_product_item_type["enum"]

    commerce_by_id = {spec["id"]: spec for spec in COMMERCE_ADMIN_CAPABILITY_SPECS}
    commerce_rewards = (
        commerce_by_id["admin.shop.sets.update"]["input_schema"]["properties"]["patch"]["properties"]["rewards"]["items"]
    )
    commerce_item_type = commerce_by_id["admin.shop.products.create"]["input_schema"]["properties"]["item_type"]

    assert "cosmetic" in commerce_rewards["properties"]["type"]["enum"]
    assert "cosmetic_slug" in commerce_rewards["properties"]
    assert "gift_shop_set" in commerce_item_type["enum"]


def test_mcp_admin_full_extraadmin_coverage_is_registered():
    from web.admin_capabilities import ADMIN_CAPABILITIES
    from web.mcp_admin_tools import ADAPTERS

    capability_by_id = {capability.id: capability for capability in ADMIN_CAPABILITIES}
    required = {
        "admin.analytics.section.read",
        "admin.analytics.dataset.export.read",
        "admin.players.analytics.read",
        "admin.players.ban",
        "admin.players.unban",
        "admin.players.warn",
        "admin.players.account.update",
        "admin.players.delete",
        "admin.shop.products.options.read",
        "admin.shop.products.detail.read",
        "admin.shop.products.create",
        "admin.shop.products.update",
        "admin.shop.products.delete",
        "admin.shop.sets.detail.read",
        "admin.shop.sets.update",
        "admin.shop.sets.delete",
        "admin.promocodes.read",
        "admin.promocodes.create",
        "admin.promocodes.update",
        "admin.promocodes.delete",
        "admin.match_modes.availability.set",
        "admin.push.app_update.broadcast",
        "admin.rewards.tracks.read",
        "admin.rewards.tracks.create",
        "admin.rewards.tracks.patch",
        "admin.rewards.tracks.delete",
        "admin.catalog.cards.read",
        "admin.catalog.cards.create",
        "admin.catalog.cards.collection.set",
        "admin.catalog.items.read",
        "admin.catalog.items.create",
        "admin.stars_test_mode.toggle",
        "admin.squads.read",
        "admin.squads.action.execute",
        "admin.cosmetics.read",
        "admin.cosmetics.detail.read",
        "admin.cosmetics.create",
        "admin.cosmetics.update",
        "admin.cosmetics.delete",
        "admin.uploads.cosmetic_image.create",
        "admin.uploads.product_image.create",
        "admin.configs.summary.read",
        "admin.runtime.tps.read",
        "admin.case_config.read",
        "admin.case_config.patch",
    }

    assert required <= set(capability_by_id)
    missing_adapters = [
        capability.adapter_function
        for capability in ADMIN_CAPABILITIES
        if capability.adapter_function not in ADAPTERS
    ]
    assert missing_adapters == []


def _test_png_base64(width: int, height: int) -> str:
    buf = BytesIO()
    Image.new("RGB", (width, height), (245, 0, 160)).save(buf, format="PNG")
    return b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.asyncio
async def test_mcp_cosmetic_upload_create_read_and_delete(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_admin_tools, "DESIGN_ASSETS_DIR", tmp_path / "DesignAssets")
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        upload_args = {
            "item_type": "avatar",
            "slug": "extra_cards_avatar",
            "content_type": "image/png",
            "base64": _test_png_base64(750, 750),
            "dry_run": True,
            "idempotency_key": "cosmetic-upload-avatar",
        }
        upload_dry = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.uploads.cosmetic_image.create", "arguments": upload_args},
        )
        upload_dry_payload = await upload_dry.json()
        upload_token = upload_dry_payload["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        upload_apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.uploads.cosmetic_image.create",
                "arguments": {
                    **upload_args,
                    "dry_run": False,
                    "confirmation_token": upload_token,
                },
            },
        )
        upload_payload = await upload_apply.json()
        upload_content = upload_payload["result"]["structuredContent"]

        assert upload_dry.status == 200
        assert upload_content["dry_run"] is False
        assert upload_content["dimensions"] == {"width": 750, "height": 750}
        assert upload_content["asset_path"].startswith("/DesignAssets/PlayerCosmetics/Avatars/Admin/")
        assert (tmp_path / upload_content["asset_path"].lstrip("/")).is_file()

        create_args = {
            "slug": "extra_cards_avatar",
            "item_type": "avatar",
            "class": "rare",
            "name": "Extra Cards Avatar",
            "asset_path": upload_content["asset_path"],
            "media_type": "image",
            "dry_run": True,
            "idempotency_key": "cosmetic-create-avatar",
        }
        create_dry = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.cosmetics.create", "arguments": create_args},
        )
        create_token = (await create_dry.json())["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        create_apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.cosmetics.create",
                "arguments": {
                    **create_args,
                    "dry_run": False,
                    "confirmation_token": create_token,
                },
            },
        )
        created = (await create_apply.json())["result"]["structuredContent"]["cosmetic"]

        assert created["slug"] == "extra_cards_avatar"
        assert created["asset_path"] == upload_content["asset_path"]

        read_response = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.cosmetics.read", "arguments": {"active_only": True}},
        )
        read_payload = await read_response.json()
        assert read_payload["result"]["structuredContent"]["items"][0]["slug"] == "extra_cards_avatar"

        delete_args = {
            "id": created["id"],
            "dry_run": True,
            "idempotency_key": "cosmetic-delete-avatar",
        }
        delete_dry = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.cosmetics.delete", "arguments": delete_args},
        )
        delete_token = (await delete_dry.json())["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        delete_apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.cosmetics.delete",
                "arguments": {
                    **delete_args,
                    "dry_run": False,
                    "confirmation_token": delete_token,
                },
            },
        )

        assert delete_apply.status == 200
        assert db.cosmetics[created["id"]]["is_active"] is False
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_rejects_title_cosmetic_image_fields(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    db.cosmetics[1] = {
        "id": 1,
        "slug": "avatar_to_title",
        "item_type": "avatar",
        "class": "rare",
        "name": "Avatar To Title",
        "asset_path": "/DesignAssets/PlayerCosmetics/Avatars/Admin/avatar_to_title.png",
        "media_type": "image",
        "is_active": True,
    }
    try:
        token = await _mcp_token(client, admin_auth_token)
        update_to_title = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.cosmetics.update",
                "arguments": {
                    "id": 1,
                    "item_type": "title",
                    "dry_run": True,
                    "idempotency_key": "avatar-to-title",
                },
            },
        )
        with_asset = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.cosmetics.create",
                "arguments": {
                    "slug": "title_with_asset",
                    "item_type": "title",
                    "class": "rare",
                    "name": "Title With Asset",
                    "asset_path": "/DesignAssets/PlayerCosmetics/Avatars/Admin/title.png",
                    "dry_run": True,
                    "idempotency_key": "title-with-asset",
                },
            },
        )
        with_media = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.cosmetics.create",
                "arguments": {
                    "slug": "title_with_image_media",
                    "item_type": "title",
                    "class": "rare",
                    "name": "Title With Image Media",
                    "media_type": "image",
                    "dry_run": True,
                    "idempotency_key": "title-with-image-media",
                },
            },
        )

        assert with_asset.status == 200
        assert with_media.status == 200
        update_preview = (await update_to_title.json())["result"]["structuredContent"]["cosmetic"]
        assert update_preview["asset_path"] is None
        assert update_preview["media_type"] == "text"
        assert db.cosmetics[1]["item_type"] == "avatar"
        assert (await with_asset.json())["error"]["message"] == "image_not_allowed_for_title"
        assert (await with_media.json())["error"]["message"] == "invalid_media_type"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_tool_call_reads_runtime_config_and_records_audit(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.runtime.config.read", "arguments": {}},
        )
        payload = await response.json()
        structured = payload["result"]["structuredContent"]

        assert response.status == 200
        assert structured["maintenance_mode"]["enabled"] is False
        assert db.rate_limit_calls
        assert db.rate_limit_calls[-1]["scope"] == "mcp:admin.runtime.config.read"
        assert db.rate_limit_calls[-1]["subject"] == "101"
        assert db.audit_calls[-1]["tool_name"] == "admin.runtime.config.read"
        assert db.audit_calls[-1]["status"] == "success"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_mutating_tool_requires_dry_run_confirmation_and_idempotency(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    patch = {"maintenance_mode": {"enabled": True}}
    try:
        token = await _mcp_token(client, admin_auth_token)
        no_dry_run = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.runtime.config.patch",
                "arguments": {
                    "patch": patch,
                    "dry_run": False,
                    "idempotency_key": "runtime-patch-1",
                    "reason": "maintenance window",
                },
            },
        )
        no_dry_run_payload = await no_dry_run.json()

        dry_run = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.runtime.config.patch",
                "arguments": {
                    "patch": patch,
                    "dry_run": True,
                    "idempotency_key": "runtime-patch-1",
                    "reason": "maintenance window",
                },
            },
        )
        dry_run_payload = await dry_run.json()
        confirmation_token = dry_run_payload["result"]["structuredContent"]["confirmation"]["confirmation_token"]

        apply_response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.runtime.config.patch",
                "arguments": {
                    "patch": patch,
                    "dry_run": False,
                    "idempotency_key": "runtime-patch-1",
                    "confirmation_token": confirmation_token,
                    "reason": "maintenance window",
                },
            },
        )
        apply_payload = await apply_response.json()

        assert no_dry_run_payload["error"]["message"] == "confirmation_required"
        assert dry_run_payload["result"]["structuredContent"]["dry_run"] is True
        assert apply_payload["result"]["structuredContent"]["dry_run"] is False
        assert db.runtime_config_updates == [{"maintenance_mode": {"enabled": True}}]
        assert db.audit_calls[-1]["confirmation_id"] == "confirm-1"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_case_config_read_returns_defaults_and_audits(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.case_config.read", "arguments": {}},
        )
        payload = await response.json()
        structured = payload["result"]["structuredContent"]

        assert response.status == 200
        # limited base = 150 (фикс бага 0-частиц для limited)
        assert structured["base_particles_by_rarity"]["limited"] == 150
        assert db.audit_calls[-1]["tool_name"] == "admin.case_config.read"
        assert db.audit_calls[-1]["status"] == "success"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_case_config_patch_dry_run_confirmation_and_idempotency(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    patch = {"base_particles_by_rarity": {"limited": 777}}
    try:
        token = await _mcp_token(client, admin_auth_token)
        no_dry_run = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.case_config.patch",
                "arguments": {
                    "patch": patch,
                    "dry_run": False,
                    "idempotency_key": "case-patch-1",
                    "reason": "limited base tuning",
                },
            },
        )
        no_dry_run_payload = await no_dry_run.json()

        dry_run = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.case_config.patch",
                "arguments": {
                    "patch": patch,
                    "dry_run": True,
                    "idempotency_key": "case-patch-1",
                    "reason": "limited base tuning",
                },
            },
        )
        dry_run_payload = await dry_run.json()
        confirmation_token = dry_run_payload["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        assert dry_run_payload["result"]["structuredContent"]["dry_run"] is True
        # dry-run не пишет
        assert db.case_config_updates == []

        apply_response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.case_config.patch",
                "arguments": {
                    "patch": patch,
                    "dry_run": False,
                    "idempotency_key": "case-patch-1",
                    "confirmation_token": confirmation_token,
                    "reason": "limited base tuning",
                },
            },
        )
        apply_payload = await apply_response.json()

        assert no_dry_run_payload["error"]["message"] == "confirmation_required"
        assert apply_payload["result"]["structuredContent"]["dry_run"] is False
        bpr = apply_payload["result"]["structuredContent"]["applied"]["base_particles_by_rarity"]
        assert bpr["limited"] == 777
        # КРИТИЧНО: остальные редкости сохранены (deep-merge), не обнулены
        assert bpr["common"] == 2
        assert bpr["divine"] == 100
        assert db.case_config_updates == [patch]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_case_config_patch_rejects_invalid_sum(monkeypatch):
    client, _db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.case_config.patch",
                "arguments": {
                    "patch": {"tier_rarity_probabilities": {1: {"common": 0.1, "rare": 0.1}}},
                    "dry_run": True,
                    "idempotency_key": "case-bad-sum",
                    "reason": "test",
                },
            },
        )
        payload = await response.json()
        assert payload["error"]["message"] == "invalid_tier_rarity_sum"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_case_config_patch_rejects_unknown_field(monkeypatch):
    client, _db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.case_config.patch",
                "arguments": {
                    "patch": {"bogus_field": 1},
                    "dry_run": True,
                    "idempotency_key": "case-unknown",
                    "reason": "test",
                },
            },
        )
        payload = await response.json()
        # Schema (additionalProperties=False) rejects unknown patch keys first;
        # adapter's unsupported_case_config_field is a secondary defense.
        assert payload["error"]["message"] == "unexpected_argument"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_case_config_patch_rejects_empty_patch(monkeypatch):
    client, _db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.case_config.patch",
                "arguments": {
                    "patch": {},
                    "dry_run": True,
                    "idempotency_key": "case-empty",
                    "reason": "test",
                },
            },
        )
        payload = await response.json()
        assert payload["error"]["message"] == "empty_case_config_patch"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_case_config_patch_partial_tier_preserves_others(monkeypatch):
    """Partial tier_upgrade_chances патч сохраняет остальные тиры."""
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    patch = {"tier_upgrade_chances": {2: 0.5}}
    try:
        token = await _mcp_token(client, admin_auth_token)
        dry_run = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.case_config.patch",
                "arguments": {
                    "patch": patch,
                    "dry_run": True,
                    "idempotency_key": "case-tier",
                    "reason": "tier tuning",
                },
            },
        )
        dry_run_payload = await dry_run.json()
        confirmation_token = dry_run_payload["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.case_config.patch",
                "arguments": {
                    "patch": patch,
                    "dry_run": False,
                    "idempotency_key": "case-tier",
                    "confirmation_token": confirmation_token,
                    "reason": "tier tuning",
                },
            },
        )
        apply_payload = await apply.json()
        tuc = apply_payload["result"]["structuredContent"]["applied"]["tier_upgrade_chances"]
        # JSON round-trip stringifies int tier keys.
        assert tuc["2"] == 0.5
        assert tuc["1"] == 0.25  # дефолт сохранён (deep-merge)
        assert tuc["3"] == 0.15
        assert tuc["4"] == 0.10
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_can_create_shop_set_with_confirmation_and_idempotency(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    arguments = {
        "name": "MCP Test Set",
        "description": "Created from MCP regression test",
        "price": 0,
        "currency": "rubles",
        "rewards": [{"type": "coins", "amount": 25}],
        "dry_run": True,
        "idempotency_key": "shop-set-create-1",
    }
    try:
        token = await _mcp_token(client, admin_auth_token)
        list_response = await _mcp_call(client, token, "tools/list")
        names = {tool["name"] for tool in (await list_response.json())["result"]["tools"]}
        assert "admin.shop.sets.create" in names

        dry_run = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.shop.sets.create", "arguments": arguments},
        )
        dry_run_payload = await dry_run.json()
        confirmation_token = dry_run_payload["result"]["structuredContent"]["confirmation"]["confirmation_token"]

        apply_response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.shop.sets.create",
                "arguments": {
                    **arguments,
                    "dry_run": False,
                    "confirmation_token": confirmation_token,
                },
            },
        )
        apply_payload = await apply_response.json()

        assert dry_run_payload["result"]["structuredContent"]["dry_run"] is True
        assert apply_payload["result"]["structuredContent"]["dry_run"] is False
        assert apply_payload["result"]["structuredContent"]["set_id"] == 101
        assert db.created_shop_sets == [
            {
                "name": "MCP Test Set",
                "description": "Created from MCP regression test",
                "image_file_id": None,
                "price": 0.0,
                "currency": "rubles",
                "created_by": 101,
                "rewards": [{"type": "coins", "amount": 25}],
            }
        ]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_shop_set_create_dry_run_accepts_cosmetic_reward(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    db.cosmetics[1] = {
        "id": 1,
        "slug": "avatar_frame_gold",
        "item_type": "avatar",
        "class": "rare",
        "name": "Avatar Frame Gold",
        "asset_path": "/DesignAssets/PlayerCosmetics/Avatars/Admin/avatar_frame_gold.png",
        "media_type": "image",
        "is_active": True,
    }
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.shop.sets.create",
                "arguments": {
                    "name": "Cosmetic MCP Set",
                    "price": 0,
                    "currency": "rubles",
                    "rewards": [
                        {
                            "type": "cosmetic",
                            "cosmetic_slug": "avatar_frame_gold",
                            "auto_equip": True,
                        }
                    ],
                    "dry_run": True,
                    "idempotency_key": "shop-set-cosmetic",
                },
            },
        )
        payload = await response.json()

        assert response.status == 200
        assert payload["result"]["structuredContent"]["dry_run"] is True
        assert payload["result"]["structuredContent"]["set"]["rewards"] == [
            {
                "type": "cosmetic",
                "cosmetic_slug": "avatar_frame_gold",
                "auto_equip": True,
            }
        ]
        assert db.created_shop_sets == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_shop_set_create_rejects_missing_cosmetic_reward(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.shop.sets.create",
                "arguments": {
                    "name": "Missing Cosmetic MCP Set",
                    "price": 0,
                    "currency": "rubles",
                    "rewards": [{"type": "cosmetic", "cosmetic_slug": "missing_cosmetic"}],
                    "dry_run": True,
                    "idempotency_key": "shop-set-missing-cosmetic",
                },
            },
        )
        payload = await response.json()

        assert response.status == 200
        assert payload["error"]["message"] == "reward_cosmetic_not_found"
        assert db.created_shop_sets == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_shop_set_read_normalizes_legacy_string_rewards(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    db.shop_sets[42] = {
        "id": 42,
        "name": "Legacy Rewards Set",
        "is_active": True,
        "rewards": json.dumps([{"type": "cosmetic", "cosmetic_slug": "bg_old"}]),
    }
    try:
        token = await _mcp_token(client, admin_auth_token)
        list_response = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.shop.sets.read", "arguments": {"active_only": False}},
        )
        detail_response = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.shop.sets.detail.read", "arguments": {"set_id": 42}},
        )
        listed_sets = (await list_response.json())["result"]["structuredContent"]["sets"]
        detail_set = (await detail_response.json())["result"]["structuredContent"]["set"]

        assert next(item for item in listed_sets if item["id"] == 42)["rewards"] == [
            {"type": "cosmetic", "cosmetic_slug": "bg_old"}
        ]
        assert detail_set["rewards"] == [{"type": "cosmetic", "cosmetic_slug": "bg_old"}]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_shop_set_create_rejects_empty_cosmetic_slug(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.shop.sets.create",
                "arguments": {
                    "name": "Bad Cosmetic MCP Set",
                    "price": 0,
                    "currency": "rubles",
                    "rewards": [{"type": "cosmetic", "cosmetic_slug": ""}],
                    "dry_run": True,
                    "idempotency_key": "shop-set-bad-cosmetic",
                },
            },
        )
        payload = await response.json()

        assert "cosmetic_slug" in payload["error"]["message"]
        assert db.created_shop_sets == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_gift_shop_set_product_requires_set_id_and_zero_price(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    db.shop_sets[77] = {
        "id": 77,
        "name": "Giftable Set",
        "price": 0,
        "currency": "rubles",
        "is_active": True,
    }
    try:
        token = await _mcp_token(client, admin_auth_token)
        missing_set = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.shop.products.create",
                "arguments": {
                    "code": "gift_missing_set",
                    "item_type": "gift_shop_set",
                    "name": "Gift Missing Set",
                    "price": 0,
                    "dry_run": True,
                    "idempotency_key": "gift-missing-set",
                },
            },
        )
        paid_gift = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.shop.products.create",
                "arguments": {
                    "code": "gift_paid",
                    "item_type": "gift_shop_set",
                    "name": "Paid Gift",
                    "price": 1,
                    "shop_set_id": 77,
                    "dry_run": True,
                    "idempotency_key": "gift-paid",
                },
            },
        )
        valid_gift = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.shop.products.create",
                "arguments": {
                    "code": "gift_set_77",
                    "item_type": "gift_shop_set",
                    "name": "Gift Set 77",
                    "price": 0,
                    "shop_set_id": 77,
                    "dry_run": True,
                    "idempotency_key": "gift-set-77",
                },
            },
        )

        missing_payload = await missing_set.json()
        paid_payload = await paid_gift.json()
        valid_payload = await valid_gift.json()

        assert missing_payload["error"]["message"] == "shop_set_id_required"
        assert paid_payload["error"]["message"] == "gift_shop_set_price_must_be_zero"
        assert valid_gift.status == 200
        assert valid_payload["result"]["structuredContent"]["product"] == {
            "code": "gift_set_77",
            "item_type": "gift_shop_set",
            "shop_set_id": 77,
            "package_type": None,
            "name": "Gift Set 77",
            "price": 0.0,
            "currency": "rubles",
        }
        assert db.created_shop_sets == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_extrapass_admin_tools_cover_draft_update_rewards_and_reset(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    try:
        token = await _mcp_token(client, admin_auth_token)
        list_response = await _mcp_call(client, token, "tools/list")
        names = {tool["name"] for tool in (await list_response.json())["result"]["tools"]}

        assert {
            "admin.extrapass.seasons.draft.create",
            "admin.extrapass.seasons.patch",
            "admin.extrapass.rewards.import",
            "admin.extrapass.reset.preview",
            "admin.extrapass.reset.execute",
            "admin.extrapass.players.entitlement.set",
        }.issubset(names)

        draft_dry = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.extrapass.seasons.draft.create",
                "arguments": {
                    "preset_key": "blank",
                    "dry_run": True,
                    "idempotency_key": "ep-draft-create",
                },
            },
        )
        draft_token = (await draft_dry.json())["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        draft_apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.extrapass.seasons.draft.create",
                "arguments": {
                    "preset_key": "blank",
                    "dry_run": False,
                    "idempotency_key": "ep-draft-create",
                    "confirmation_token": draft_token,
                },
            },
        )

        patch_dry = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.extrapass.seasons.patch",
                "arguments": {
                    "season_id": 1,
                    "patch": {"status": "scheduled", "name": "Next Season"},
                    "dry_run": True,
                    "idempotency_key": "ep-season-patch",
                },
            },
        )
        patch_token = (await patch_dry.json())["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        patch_apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.extrapass.seasons.patch",
                "arguments": {
                    "season_id": 1,
                    "patch": {"status": "scheduled", "name": "Next Season"},
                    "dry_run": False,
                    "idempotency_key": "ep-season-patch",
                    "confirmation_token": patch_token,
                },
            },
        )

        rewards_dry = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.extrapass.rewards.import",
                "arguments": {
                    "season_id": 1,
                    "replace": True,
                    "tracks": {"free": [{"position": 1, "reward_type": "coins", "reward_amount": 100}]},
                    "dry_run": True,
                    "idempotency_key": "ep-rewards-import",
                },
            },
        )
        rewards_token = (await rewards_dry.json())["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        rewards_apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.extrapass.rewards.import",
                "arguments": {
                    "season_id": 1,
                    "replace": True,
                    "tracks": {"free": [{"position": 1, "reward_type": "coins", "reward_amount": 100}]},
                    "dry_run": False,
                    "idempotency_key": "ep-rewards-import",
                    "confirmation_token": rewards_token,
                },
            },
        )

        reset_preview = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.extrapass.reset.preview", "arguments": {"season_id": 1, "sample_limit": 10}},
        )
        reset_dry = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.extrapass.reset.execute",
                "arguments": {
                    "season_id": 1,
                    "confirm_season_id": 1,
                    "reason": "test reset",
                    "dry_run": True,
                    "idempotency_key": "ep-reset-execute",
                },
            },
        )
        reset_token = (await reset_dry.json())["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        reset_apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.extrapass.reset.execute",
                "arguments": {
                    "season_id": 1,
                    "confirm_season_id": 1,
                    "reason": "test reset",
                    "dry_run": False,
                    "idempotency_key": "ep-reset-execute",
                    "confirmation_token": reset_token,
                },
            },
        )

        entitlement_dry = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.extrapass.players.entitlement.set",
                "arguments": {
                    "user_id": 201,
                    "mode": "ultra",
                    "reason": "test entitlement",
                    "dry_run": True,
                    "idempotency_key": "ep-entitlement-set",
                },
            },
        )
        entitlement_token = (await entitlement_dry.json())["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        entitlement_apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.extrapass.players.entitlement.set",
                "arguments": {
                    "user_id": 201,
                    "mode": "ultra",
                    "reason": "test entitlement",
                    "dry_run": False,
                    "idempotency_key": "ep-entitlement-set",
                    "confirmation_token": entitlement_token,
                },
            },
        )

        assert (await draft_apply.json())["result"]["structuredContent"]["season"]["id"] == 2
        assert (await patch_apply.json())["result"]["structuredContent"]["season"]["name"] == "Next Season"
        assert (await rewards_apply.json())["result"]["structuredContent"]["imported"] == 1
        assert (await reset_preview.json())["result"]["structuredContent"]["summary"]["players"] == 3
        assert (await reset_apply.json())["result"]["structuredContent"]["reset"]["status"] == "completed"
        assert (await entitlement_apply.json())["result"]["structuredContent"]["result"]["mode"] == "ultra"
        assert db.created_season_drafts == ["blank"]
        assert db.updated_seasons == [(1, {"status": "scheduled", "name": "Next Season", "admin_user_id": 101})]
        assert db.replaced_reward_tracks[0][0] == ["bp_free", "bp_premium", "bp_ultra"]
        assert db.executed_season_resets == [
            {
                "season_id": 1,
                "previous_season_id": None,
                "trigger": "admin",
                "admin_user_id": 101,
                "reason": "test reset",
                "require_active": True,
            }
        ]
        assert db.extra_pass_updates == [
            {
                "admin_user_id": 101,
                "target_user_id": 201,
                "mode": "ultra",
                "days": None,
                "reason": "test entitlement",
            }
        ]
    finally:
        await client.close()


def test_mcp_extrapass_reward_import_accepts_specific_card_and_rejects_bad_card_config():
    season = {
        "free_track_type": "bp_free",
        "pass_track_type": "bp_premium",
        "ultra_track_type": "bp_ultra",
        "max_stars": 45,
        "pass_end_position": 40,
        "ultra_start_position": 41,
    }

    rows = mcp_admin_tools._normalize_extrapass_reward_import_payload(
        {"ultra": [{"position": 41, "reward_type": "specific_card", "reward_amount": 1, "reward_meta": {"card_id": 46}}]},
        season,
    )

    assert rows == [
        {
            "track_type": "bp_ultra",
            "position": 41,
            "reward_type": "specific_card",
            "reward_amount": 1,
            "reward_meta": {"card_id": 46},
            "extra_pass_required": True,
        }
    ]
    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="specific_card_id_required"):
        mcp_admin_tools._normalize_extrapass_reward_import_payload(
            {"ultra": [{"position": 41, "reward_type": "specific_card", "reward_amount": 1}]},
            season,
        )


def test_mcp_extrapass_reward_import_accepts_special_reward_types_and_aliases():
    season = {
        "free_track_type": "bp_free",
        "pass_track_type": "bp_premium",
        "ultra_track_type": "bp_ultra",
        "max_stars": 45,
        "pass_end_position": 40,
        "ultra_start_position": 41,
    }

    rows = mcp_admin_tools._normalize_extrapass_reward_import_payload(
        {
            "free": [{"position": 1, "reward_type": "particles", "reward_amount": 25, "card_id": 46}],
            "premium": [{"position": 2, "reward_type": "cosmetic", "cosmetic_slug": "avatar_gold", "auto_equip": True}],
            "ultra": [{"position": 41, "reward_type": "guaranteed_card", "reward_amount": 99, "reward_meta": {"card_id": 46}}],
        },
        season,
    )

    assert rows[0]["reward_type"] == "particles"
    assert rows[0]["reward_meta"] == {"card_id": 46}
    assert rows[1]["reward_type"] == "cosmetic"
    assert rows[1]["reward_amount"] == 1
    assert rows[1]["reward_meta"] == {"cosmetic_slug": "avatar_gold", "auto_equip": True}
    assert rows[2]["reward_type"] == "specific_card"
    assert rows[2]["reward_amount"] == 1


def test_mcp_extrapass_reward_import_validates_lane_bounds_and_derives_access():
    season = {
        "free_track_type": "bp_free",
        "pass_track_type": "bp_premium",
        "ultra_track_type": "bp_ultra",
        "max_stars": 45,
        "pass_end_position": 40,
        "ultra_start_position": 41,
    }

    rows = mcp_admin_tools._normalize_extrapass_reward_import_payload(
        {
            "free": [{"position": 45, "reward_type": "coins", "reward_amount": 100, "extra_pass_required": True}],
            "premium": [{"position": 40, "reward_type": "gems", "reward_amount": 10, "extra_pass_required": False}],
            "ultra": [{"position": 41, "reward_type": "keys", "reward_amount": 1, "extra_pass_required": False}],
        },
        season,
    )

    assert [row["extra_pass_required"] for row in rows] == [False, True, True]

    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="position_out_of_track_scope"):
        mcp_admin_tools._normalize_extrapass_reward_import_payload(
            {"premium": [{"position": 41, "reward_type": "gems", "reward_amount": 10}]},
            season,
        )
    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="position_out_of_track_scope"):
        mcp_admin_tools._normalize_extrapass_reward_import_payload(
            {"ultra": [{"position": 40, "reward_type": "keys", "reward_amount": 1}]},
            season,
        )


def test_mcp_extrapass_reward_import_validates_lane_bounds_and_derives_pass_access():
    season = {
        "free_track_type": "bp_free",
        "pass_track_type": "bp_premium",
        "ultra_track_type": "bp_ultra",
        "max_stars": 45,
        "pass_end_position": 40,
        "ultra_start_position": 41,
    }

    rows = mcp_admin_tools._normalize_extrapass_reward_import_payload(
        {"premium": [{"position": 2, "reward_type": "coins", "reward_amount": 100, "extra_pass_required": False}]},
        season,
    )

    assert rows[0]["extra_pass_required"] is True
    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="position_out_of_track_scope:ultra:1"):
        mcp_admin_tools._normalize_extrapass_reward_import_payload(
            {"ultra": [{"position": 1, "reward_type": "coins", "reward_amount": 100}]},
            season,
        )
    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="position_out_of_track_scope:premium:41"):
        mcp_admin_tools._normalize_extrapass_reward_import_payload(
            {"premium": [{"position": 41, "reward_type": "coins", "reward_amount": 100}]},
            season,
        )


@pytest.mark.asyncio
async def test_mcp_seasons_reward_tracks_read_filters_inactive_rewards():
    db = GatewayMCPDB()
    db.reward_tracks = [
        {"id": 1, "track_type": "s1_free", "position": 1, "is_active": True},
        {"id": 2, "track_type": "s1_free", "position": 2, "is_active": False},
    ]

    payload = await mcp_admin_tools.adapter_read_seasons_reward_tracks(
        {"db": db},
        101,
        {"include_inactive_rewards": False},
    )

    assert payload["reward_tracks"] == [{"id": 1, "track_type": "s1_free", "position": 1, "is_active": True}]


@pytest.mark.asyncio
async def test_mcp_extrapass_reward_import_validates_specific_card_exists_and_is_warrior():
    db = GatewayMCPDB()
    db.cards = {}
    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="specific_card_not_found"):
        await mcp_admin_tools.adapter_import_extrapass_rewards(
            {"db": db},
            101,
            {
                "season_id": 1,
                "tracks": {"ultra": [{"position": 41, "reward_type": "specific_card", "reward_amount": 1, "reward_meta": {"card_id": 46}}]},
                "dry_run": True,
            },
        )

    db.cards = {46: {"id": 46, "name": "Hero", "card_type": "hero"}}
    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="specific_card_must_be_warrior"):
        await mcp_admin_tools.adapter_import_extrapass_rewards(
            {"db": db},
            101,
            {
                "season_id": 1,
                "tracks": {"ultra": [{"position": 41, "reward_type": "specific_card", "reward_amount": 1, "reward_meta": {"card_id": 46}}]},
                "dry_run": True,
            },
        )


@pytest.mark.asyncio
async def test_mcp_extrapass_reward_import_validates_and_replaces_special_reward_types():
    db = GatewayMCPDB()
    db.cosmetics[1] = {"id": 1, "slug": "avatar_gold", "item_type": "avatar", "is_active": True}

    payload = await mcp_admin_tools.adapter_import_extrapass_rewards(
        {"db": db},
        101,
        {
            "season_id": 1,
            "tracks": {
                "free": [{"position": 1, "reward_type": "particles", "reward_amount": 25, "reward_meta": {"card_id": 46}}],
                "premium": [{"position": 2, "reward_type": "cosmetic", "reward_amount": 0, "reward_meta": {"cosmetic_slug": "avatar_gold"}}],
                "ultra": [{"position": 41, "reward_type": "guaranteed_card", "reward_amount": 99, "reward_meta": {"card_id": 46}}],
            },
            "replace": True,
        },
    )

    assert payload["imported"] == 3
    assert db.replaced_reward_tracks[0][0] == ["bp_free", "bp_premium", "bp_ultra"]
    rows = db.replaced_reward_tracks[0][1]
    assert [row["reward_type"] for row in rows] == ["particles", "cosmetic", "specific_card"]
    assert rows[1]["reward_amount"] == 1
    assert rows[2]["reward_amount"] == 1

    db.cosmetics.clear()
    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="reward_cosmetic_not_found"):
        await mcp_admin_tools.adapter_import_extrapass_rewards(
            {"db": db},
            101,
            {
                "season_id": 1,
                "tracks": {"premium": [{"position": 2, "reward_type": "cosmetic", "reward_amount": 1, "reward_meta": {"cosmetic_slug": "avatar_gold"}}]},
                "dry_run": True,
            },
        )


@pytest.mark.asyncio
async def test_mcp_seasons_reward_tracks_read_can_include_inactive_rewards():
    db = GatewayMCPDB()
    db.reward_tracks = [
        {"id": 1, "track_type": "bp_free", "position": 1, "is_active": True},
        {"id": 2, "track_type": "bp_free", "position": 2, "is_active": False},
        {"id": 3, "track_type": "bp_premium", "position": 3},
    ]

    filtered = await mcp_admin_tools.adapter_read_seasons_reward_tracks(
        {"db": db},
        101,
        {"include_inactive_rewards": False, "include_reset_summaries": False},
    )
    unfiltered = await mcp_admin_tools.adapter_read_seasons_reward_tracks(
        {"db": db},
        101,
        {"include_inactive_rewards": True, "include_reset_summaries": False},
    )

    assert [row["id"] for row in filtered["reward_tracks"]] == [1, 3]
    assert [row["id"] for row in unfiltered["reward_tracks"]] == [1, 2, 3]


@pytest.mark.asyncio
async def test_mcp_reward_track_create_and_patch_validate_specific_card_with_db():
    db = GatewayMCPDB()
    db.cards = {}

    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="specific_card_not_found"):
        await mcp_admin_tools.adapter_create_reward_track(
            {"db": db},
            101,
            {
                "track_type": "bp_ultra",
                "position": 41,
                "reward_type": "specific_card",
                "reward_amount": 1,
                "reward_meta": {"card_id": 46},
                "dry_run": True,
            },
        )

    db.cards = {46: {"id": 46, "name": "Hero", "card_type": "hero"}}
    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="specific_card_must_be_warrior"):
        await mcp_admin_tools.adapter_patch_reward_track(
            {"db": db},
            101,
            {
                "id": 1,
                "patch": {"reward_type": "specific_card", "reward_amount": 1, "reward_meta": {"card_id": 46}},
                "dry_run": True,
            },
        )


@pytest.mark.asyncio
async def test_mcp_completed_idempotency_replays_before_reconsuming_confirmation(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    patch = {"maintenance_mode": {"enabled": True}}
    arguments = {
        "patch": patch,
        "dry_run": True,
        "idempotency_key": "runtime-patch-replay",
        "reason": "maintenance window",
    }
    try:
        token = await _mcp_token(client, admin_auth_token)
        dry_run = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.runtime.config.patch", "arguments": arguments},
        )
        confirmation_token = (await dry_run.json())["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        apply_arguments = {**arguments, "dry_run": False, "confirmation_token": confirmation_token}
        first_apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.runtime.config.patch", "arguments": apply_arguments},
        )
        second_apply = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.runtime.config.patch", "arguments": apply_arguments},
        )
        first_payload = await first_apply.json()
        second_payload = await second_apply.json()

        assert first_payload["result"]["structuredContent"]["dry_run"] is False
        assert second_payload["result"]["structuredContent"] == first_payload["result"]["structuredContent"]
        assert db.runtime_config_updates == [{"maintenance_mode": {"enabled": True}}]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_dry_run_does_not_complete_existing_idempotency_key(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    key = (101, "admin.runtime.config.patch", "runtime-patch-dry-run")
    db.idempotency[key] = {
        "args_digest": "existing-digest",
        "response": {"dry_run": False, "applied": {"maintenance_mode": {"enabled": True}}},
    }
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.runtime.config.patch",
                "arguments": {
                    "patch": {"maintenance_mode": {"enabled": False}},
                    "dry_run": True,
                    "idempotency_key": "runtime-patch-dry-run",
                    "reason": "maintenance window",
                },
            },
        )
        payload = await response.json()

        assert payload["result"]["structuredContent"]["dry_run"] is True
        assert db.idempotency[key]["response"] == {
            "dry_run": False,
            "applied": {"maintenance_mode": {"enabled": True}},
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_mutating_tool_blocks_active_idempotency_key(monkeypatch):
    client, db, admin_auth_token = await _gateway_client(monkeypatch)
    patch = {"maintenance_mode": {"enabled": True}}
    try:
        token = await _mcp_token(client, admin_auth_token)
        dry_run = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.runtime.config.patch",
                "arguments": {
                    "patch": patch,
                    "dry_run": True,
                    "idempotency_key": "runtime-patch-busy",
                    "reason": "maintenance window",
                },
            },
        )
        confirmation_token = (await dry_run.json())["result"]["structuredContent"]["confirmation"]["confirmation_token"]
        db.idempotency[(101, "admin.runtime.config.patch", "runtime-patch-busy")] = {"force_in_progress": True}

        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {
                "name": "admin.runtime.config.patch",
                "arguments": {
                    "patch": patch,
                    "dry_run": False,
                    "idempotency_key": "runtime-patch-busy",
                    "confirmation_token": confirmation_token,
                    "reason": "maintenance window",
                },
            },
        )
        payload = await response.json()

        assert payload["error"]["message"] == "idempotency_key_in_progress"
        assert db.runtime_config_updates == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_mcp_tool_errors_are_sanitized(monkeypatch):
    db = GatewayMCPDB()
    db.fail_runtime_config = True
    client, _db, admin_auth_token = await _gateway_client(monkeypatch, db=db)
    try:
        token = await _mcp_token(client, admin_auth_token)
        response = await _mcp_call(
            client,
            token,
            "tools/call",
            {"name": "admin.runtime.config.read", "arguments": {}},
        )
        payload = await response.json()
        encoded = json.dumps(payload)

        assert response.status == 200
        assert payload["error"]["message"] == "tool_execution_failed"
        assert "secret_runtime_table" not in encoded
    finally:
        await client.close()


def _db_settings() -> DatabaseSettings:
    return DatabaseSettings("localhost", 5432, "user", "pass", "db")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scoped_idempotency_hash(admin_user_id: int, tool_name: str, idempotency_key: str) -> str:
    return _sha256(f"{int(admin_user_id)}:{tool_name}:{idempotency_key}")


def _summary(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256(payload)


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            out.extend(_flatten_strings(key))
            out.extend(_flatten_strings(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return [str(value)]


class FakeMCPTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeMCPAcquire:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeMCPPool:
    def __init__(self, db):
        self.db = db

    def acquire(self):
        return FakeMCPAcquire(self.db)


class FakeMCPDatabase(Database):
    def __init__(self):
        super().__init__(_db_settings())
        self._pool = FakeMCPPool(self)
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_results: list[Any] = []
        self.fetchval_results: list[Any] = []

    def transaction(self):
        return FakeMCPTransaction()

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.fetch_calls.append((query, args))
        return []

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.fetchrow_calls.append((query, args))
        if self.fetchrow_results:
            return self.fetchrow_results.pop(0)
        return None

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append((query, args))
        if self.fetchval_results:
            return self.fetchval_results.pop(0)
        return None


@pytest.mark.asyncio
async def test_mcp_schema_helpers_create_audit_confirmation_idempotency_and_rate_tables():
    db = FakeMCPDatabase()

    changed = await db._ensure_mcp_admin_tables()

    assert changed is True
    sql = "\n".join(query for query, _ in db.executed)
    for table in (
        "mcp_tool_calls",
        "mcp_confirmations",
        "mcp_idempotency_keys",
        "mcp_rate_limits",
    ):
        assert f"CREATE TABLE {table}" in sql
    assert "admin_user_id BIGINT" in sql
    assert "tool_name TEXT" in sql
    assert "args_digest TEXT" in sql
    assert "jti_hash TEXT" in sql
    assert "args_summary JSONB" in sql
    assert "result_summary JSONB" in sql
    assert "duration_ms INTEGER" in sql


@pytest.mark.asyncio
async def test_record_mcp_tool_call_stores_sanitized_summaries_and_hashed_jti():
    db = FakeMCPDatabase()
    db.fetchrow_results.append({"id": 987})
    args = {
        "target_user_id": 77,
        "reason": "spam",
        "auth_token": "raw-admin-token",
        "nested": {"password": "raw-password"},
    }
    result = {"success": True, "authorization": "Bearer raw-result-token"}

    saved = await db.record_mcp_tool_call(
        admin_user_id=42,
        tool_name="admin.ban_user",
        args=args,
        result=result,
        status="success",
        error=None,
        duration_ms=37,
        jti="raw-jti",
    )

    query, params = db.fetchrow_calls[-1]
    args_summary = _summary(params[3])
    result_summary = _summary(params[4])
    assert "INSERT INTO mcp_tool_calls" in query
    assert params[:3] == (
        42,
        "admin.ban_user",
        _canonical_digest(
            {
                "target_user_id": 77,
                "reason": "spam",
                "auth_token": "[REDACTED]",
                "nested": {"password": "[REDACTED]"},
            }
        ),
    )
    assert args_summary["auth_token"] == "[REDACTED]"
    assert args_summary["nested"]["password"] == "[REDACTED]"
    assert result_summary["authorization"] == "[REDACTED]"
    assert params[5:9] == ("success", None, 37, _sha256("raw-jti"))
    assert saved["id"] == 987
    assert saved["jti_hash"] == _sha256("raw-jti")
    saved_strings = _flatten_strings(params)
    assert "raw-admin-token" not in saved_strings
    assert "raw-password" not in saved_strings
    assert "raw-result-token" not in saved_strings
    assert "raw-jti" not in saved_strings


@pytest.mark.asyncio
async def test_mcp_confirmation_helpers_hash_tokens_and_mark_consumption():
    db = FakeMCPDatabase()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    db.fetchrow_results.extend(
        [
            {
                "confirmation_id": "confirm-1",
                "admin_user_id": 42,
                "tool_name": "admin.delete_post",
                "args_digest": "digest-1",
                "status": "pending",
                "expires_at": expires_at,
            },
            {"confirmation_id": "confirm-1", "status": "confirmed"},
            {"confirmation_id": "confirm-1", "status": "confirmed"},
        ]
    )

    created = await db.create_mcp_confirmation(
        admin_user_id=42,
        tool_name="admin.delete_post",
        args_digest="digest-1",
        confirmation_token="raw-confirmation-token",
        expires_at=expires_at,
        jti="raw-jti",
    )
    consumed = await db.consume_mcp_confirmation(
        confirmation_token="raw-confirmation-token",
        admin_user_id=42,
        tool_name="admin.delete_post",
        args_digest="digest-1",
    )
    loaded = await db.get_mcp_confirmation("confirm-1")

    assert created["confirmation_id"] == "confirm-1"
    assert consumed["status"] == "confirmed"
    assert loaded["status"] == "confirmed"
    all_params = _flatten_strings([params for _, params in db.fetchrow_calls])
    assert _sha256("raw-confirmation-token") in all_params
    assert _sha256("raw-jti") in all_params
    assert "raw-confirmation-token" not in all_params
    assert "raw-jti" not in all_params


@pytest.mark.asyncio
async def test_mcp_idempotency_helpers_hash_keys_and_store_sanitized_responses():
    db = FakeMCPDatabase()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
    db.fetchrow_results.extend(
        [
            None,
            {
                "key_hash": _scoped_idempotency_hash(42, "admin.ban_user", "raw-idempotency-key"),
                "admin_user_id": 42,
                "tool_name": "admin.ban_user",
                "args_digest": "digest-1",
                "status": "in_progress",
                "response_summary": None,
                "error": None,
                "expires_at": expires_at,
            },
            {
                "key_hash": _scoped_idempotency_hash(42, "admin.ban_user", "raw-idempotency-key"),
                "status": "success",
                "response_summary": {"ok": True},
                "error": None,
            },
            {
                "key_hash": _scoped_idempotency_hash(42, "admin.ban_user", "raw-idempotency-key"),
                "status": "success",
                "response_summary": {"ok": True},
                "error": None,
            },
        ]
    )

    reserved = await db.reserve_mcp_idempotency_key(
        idempotency_key="raw-idempotency-key",
        admin_user_id=42,
        tool_name="admin.ban_user",
        args_digest="digest-1",
        expires_at=expires_at,
        jti="raw-jti",
    )
    completed = await db.complete_mcp_idempotency_key(
        idempotency_key="raw-idempotency-key",
        admin_user_id=42,
        tool_name="admin.ban_user",
        status="success",
        response={"ok": True, "access_token": "raw-response-token"},
        error=None,
    )
    loaded = await db.get_mcp_idempotency_key(
        idempotency_key="raw-idempotency-key",
        admin_user_id=42,
        tool_name="admin.ban_user",
    )

    assert reserved["status"] == "in_progress"
    assert reserved["reserved"] is True
    assert completed["status"] == "success"
    assert loaded["response_summary"] == {"ok": True}
    all_params = _flatten_strings([params for _, params in db.fetchrow_calls])
    assert _scoped_idempotency_hash(42, "admin.ban_user", "raw-idempotency-key") in all_params
    assert _sha256("raw-idempotency-key") not in all_params
    assert _sha256("raw-jti") in all_params
    assert "raw-idempotency-key" not in all_params
    assert "raw-jti" not in all_params
    assert "raw-response-token" not in all_params


@pytest.mark.asyncio
async def test_mcp_idempotency_helper_marks_active_existing_key_unreserved():
    db = FakeMCPDatabase()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
    db.fetchrow_results.append(
        {
            "key_hash": _scoped_idempotency_hash(42, "admin.ban_user", "busy-idempotency-key"),
            "admin_user_id": 42,
            "tool_name": "admin.ban_user",
            "args_digest": "digest-1",
            "status": "in_progress",
            "response_summary": None,
            "error": None,
            "expires_at": expires_at,
        }
    )

    reserved = await db.reserve_mcp_idempotency_key(
        idempotency_key="busy-idempotency-key",
        admin_user_id=42,
        tool_name="admin.ban_user",
        args_digest="digest-1",
        expires_at=expires_at,
        jti="raw-jti",
    )

    assert reserved["reserved"] is False
    assert reserved["args_match"] is True
    assert reserved["replayable"] is False
    assert reserved["status"] == "in_progress"


@pytest.mark.asyncio
async def test_mcp_rate_helper_hashes_subject_and_returns_limit_state():
    db = FakeMCPDatabase()
    db.fetchrow_results.append(
        {
            "allowed": False,
            "count": 4,
            "reset_at": datetime(2026, 6, 7, tzinfo=timezone.utc),
        }
    )

    result = await db.check_mcp_rate_limit(
        scope="admin-tool",
        subject="Bearer raw-admin-token",
        max_requests=3,
        window_seconds=60,
    )

    assert result["allowed"] is False
    assert result["count"] == 4
    assert result["key_hash"] == _sha256("Bearer raw-admin-token")
    all_params = _flatten_strings([params for _, params in db.fetchrow_calls])
    assert _sha256("Bearer raw-admin-token") in all_params
    assert "Bearer raw-admin-token" not in all_params
