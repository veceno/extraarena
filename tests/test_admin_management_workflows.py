import json
import time
import uuid
from io import BytesIO

import jwt
import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from bot.constants import ADMIN_ID
from infrastructure.config import get_settings
from infrastructure.case_config import (
    build_default_case_config,
    merge_case_config_patch,
    validate_case_config,
)
from web import mcp_admin_tools
from web import server as web_server


STRONG_TEST_JWT_SECRET = "test-admin-workflow-jwt-secret-that-is-long-enough-2026"


@pytest.fixture(autouse=True)
def _strong_jwt_secret(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    monkeypatch.setenv("JWT_SECRET", STRONG_TEST_JWT_SECRET)
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-admin-workflow-session-secret-that-is-long-enough-2026")
    monkeypatch.setenv("MCP_TOKEN_SECRET", "test-admin-workflow-mcp-secret-that-is-long-enough-2026")
    monkeypatch.setenv("MCP_ALLOWED_ORIGINS", "https://example.test")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class WorkflowExtraIDDB:
    def __init__(self, session_id: str):
        self.session_id = session_id

    async def verify_session(self, session_uuid, token: str):
        if str(session_uuid) != self.session_id:
            return None
        return {"session_id": session_uuid, "user_id": ADMIN_ID}


class AdminWorkflowDB:
    def __init__(self):
        self.case_config = build_default_case_config()
        self.sets = {}
        self.next_set_id = 1
        self.products = {}
        self.next_product_id = 1
        self.seasons = [
            {
                "id": 1,
                "slug": "season-1",
                "name": "Season 1",
                "season_number": 1,
                "status": "active",
                "is_active": True,
                "start_date": "2026-05-01T00:00:00+00:00",
                "end_date": "2026-06-01T00:00:00+00:00",
                "free_track_type": "s1_free",
                "pass_track_type": "s1_pass",
                "ultra_track_type": "s1_ultra",
                "max_stars": 45,
                "stage_cost_min": 3,
                "stage_cost_growth": 0.07,
                "stage_cost_exponent": 1.5,
                "stage_cost_cap": 25,
                "pass_end_position": 40,
                "ultra_start_position": 41,
            }
        ]
        self.reward_tracks = []
        self.next_reward_id = 1
        self.extra_pass_updates = []
        self.promocodes = []
        self.cosmetics = {}
        self.next_cosmetic_id = 1
        self.account_updates = []
        self.resource_adjustments = []
        self.reset_completed = False
        self.reset_execute_kwargs = []
        self.reset_preview = {
            "season_id": 1,
            "already_completed": False,
            "summary": {
                "players": 2,
                "trophies_reduced": 299,
                "keys_granted": 2,
                "coins_granted": 400,
                "stars_reset": 17,
            },
            "players": [
                {
                    "user_id": 1001,
                    "old_trophies": 799,
                    "new_trophies": 600,
                    "old_stars": 8,
                    "excess_trophies": 199,
                    "granted_keys": 1,
                    "granted_coins": 200,
                },
                {
                    "user_id": 1002,
                    "old_trophies": 1499,
                    "new_trophies": 1200,
                    "old_stars": 9,
                    "excess_trophies": 299,
                    "granted_keys": 2,
                    "granted_coins": 400,
                },
            ],
        }

    async def is_admin(self, user_id):
        return False

    async def get_runtime_config(self):
        return {
            "maintenance_mode": {"enabled": False},
            "feature_availability": {},
            "disabled_card_ids": [],
        }

    async def get_case_config(self):
        import copy as _copy
        return _copy.deepcopy(self.case_config)

    async def set_case_config(self, *, patch=None):
        if not isinstance(patch, dict) or not patch:
            raise ValueError("empty_case_config_patch")
        merged = merge_case_config_patch(self.case_config, patch)
        validate_case_config(merged)
        self.case_config = merged
        return merged

    async def get_match_mode_overrides(self):
        return []

    async def get_disabled_card_ids(self):
        return []

    async def process_weekly_squad_cbrp(self):
        return {"processed": False}

    async def expire_announcements(self):
        return 0

    def _normalize_shop_set_rewards(self, rewards):
        return rewards, None

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

    async def get_card_info(self, card_id):
        card_id = int(card_id)
        if card_id == 46:
            return {"id": 46, "name": "Reward Card", "card_type": "warrior"}
        return None

    async def create_cosmetic_item(self, **kwargs):
        if any(row["slug"] == kwargs["slug"] for row in self.cosmetics.values()):
            return {"success": False, "error": "cosmetic_slug_exists"}
        cosmetic_id = self.next_cosmetic_id
        self.next_cosmetic_id += 1
        row = {
            "id": cosmetic_id,
            "is_active": True,
            "sort_order": 0,
            **kwargs,
        }
        self.cosmetics[cosmetic_id] = row
        return {"success": True, "cosmetic": row}

    async def update_cosmetic_item(self, cosmetic_id, **kwargs):
        cosmetic_id = int(cosmetic_id)
        if cosmetic_id not in self.cosmetics:
            return {"success": False, "error": "cosmetic_not_found"}
        if "slug" in kwargs and any(
            row_id != cosmetic_id and row["slug"] == kwargs["slug"]
            for row_id, row in self.cosmetics.items()
        ):
            return {"success": False, "error": "cosmetic_slug_exists"}
        self.cosmetics[cosmetic_id].update(kwargs)
        return {"success": True, "cosmetic": self.cosmetics[cosmetic_id]}

    async def delete_cosmetic_item(self, cosmetic_id):
        cosmetic_id = int(cosmetic_id)
        if cosmetic_id not in self.cosmetics:
            return {"success": False, "error": "cosmetic_not_found"}
        self.cosmetics[cosmetic_id]["is_active"] = False
        return {"success": True, "cosmetic": self.cosmetics[cosmetic_id]}

    async def get_shop_sets(self, active_only=True):
        rows = list(self.sets.values())
        if active_only:
            rows = [row for row in rows if row.get("is_active", True)]
        return rows

    async def get_shop_set(self, set_id):
        return self.sets.get(int(set_id))

    async def create_shop_set(self, **kwargs):
        set_id = self.next_set_id
        self.next_set_id += 1
        self.sets[set_id] = {
            "id": set_id,
            "is_active": True,
            **kwargs,
        }
        return {"success": True, "set_id": set_id}

    async def update_shop_set(self, set_id, **kwargs):
        set_id = int(set_id)
        if set_id not in self.sets:
            return {"success": False, "error": "set_not_found"}
        self.sets[set_id].update(kwargs)
        return {"success": True}

    async def delete_shop_set(self, set_id):
        set_id = int(set_id)
        if set_id not in self.sets:
            return {"success": False, "error": "set_not_found"}
        self.sets[set_id]["is_active"] = False
        return {"success": True}

    async def get_ruble_products(self, active_only=True, surface=None):
        rows = list(self.products.values())
        if active_only:
            rows = [row for row in rows if row.get("is_active", True)]
        return rows

    async def get_ruble_product(self, code):
        return self.products.get(str(code))

    async def create_ruble_product(self, **kwargs):
        product_id = self.next_product_id
        self.next_product_id += 1
        row = {
            "id": product_id,
            "is_active": True,
            "show_in_game": False,
            "show_in_shop": True,
            **kwargs,
        }
        self.products[str(row["code"])] = row
        return {"success": True, "product_id": product_id}

    async def update_ruble_product(self, code_or_id, **kwargs):
        key = None
        if isinstance(code_or_id, int):
            key = next((code for code, row in self.products.items() if row["id"] == code_or_id), None)
        else:
            key = str(code_or_id)
        if key not in self.products:
            return {"success": False, "error": "product_not_found"}
        row = self.products[key]
        row.update(kwargs)
        if "code" in kwargs and kwargs["code"] != key:
            self.products[str(kwargs["code"])] = row
            del self.products[key]
        return {"success": True}

    async def delete_ruble_product(self, code_or_id):
        key = str(code_or_id)
        if key not in self.products:
            return {"success": False, "error": "product_not_found"}
        self.products[key]["is_active"] = False
        return {"success": True}

    async def get_seasons(self):
        return self.seasons

    async def get_all_reward_tracks(self):
        return list(self.reward_tracks)

    async def get_promocodes_list(self):
        return list(self.promocodes)

    async def create_promocode(self, **kwargs):
        self.promocodes.append(kwargs)
        return {"success": True, "code": kwargs["code"]}

    async def create_season_draft(self, preset_key="blank"):
        season_id = len(self.seasons) + 1
        season = {
            "id": season_id,
            "slug": f"season-{season_id}",
            "name": f"Season {season_id}",
            "season_number": season_id,
            "status": "draft",
            "is_active": False,
            "start_date": None,
            "end_date": None,
            "free_track_type": f"s{season_id}_free",
            "pass_track_type": f"s{season_id}_pass",
            "ultra_track_type": f"s{season_id}_ultra",
            "max_stars": 45,
            "stage_cost_min": 3,
            "stage_cost_growth": 0.07,
            "stage_cost_exponent": 1.5,
            "stage_cost_cap": 25,
            "pass_end_position": 40,
            "ultra_start_position": 41,
            "preset_key": preset_key,
        }
        self.seasons.append(season)
        return season

    async def update_season(self, season_id, **kwargs):
        kwargs.pop("admin_user_id", None)
        season = await self.get_season_by_id(season_id)
        if not season:
            return {"error": "season_not_found"}
        season.update(kwargs)
        return season

    async def get_season_by_id(self, season_id):
        return next((season for season in self.seasons if int(season["id"]) == int(season_id)), None)

    async def get_season_reset_summaries(self):
        if not self.reset_completed:
            return {}
        return {
            1: {
                "id": 1,
                "season_id": 1,
                "status": "completed",
                "trigger": "admin",
                "processed_players": 2,
                "total_trophies_reduced": 299,
                "total_keys_granted": 2,
                "total_coins_granted": 400,
                "completed_at": "2026-06-07T00:00:00+00:00",
            }
        }

    async def preview_season_reset(self, season_id):
        preview = dict(self.reset_preview)
        preview["season_id"] = int(season_id)
        preview["already_completed"] = self.reset_completed and int(season_id) == 1
        return preview

    async def execute_season_reset(self, **kwargs):
        self.reset_execute_kwargs.append(kwargs)
        if self.reset_completed:
            return {"error": "season_reset_already_completed"}
        self.reset_completed = True
        return {
            "id": 1,
            "season_id": int(kwargs["season_id"]),
            "previous_season_id": kwargs.get("previous_season_id"),
            "trigger": kwargs.get("trigger"),
            "admin_user_id": kwargs.get("admin_user_id"),
            "status": "completed",
            "processed_players": 2,
            "total_trophies_reduced": 299,
            "total_keys_granted": 2,
            "total_coins_granted": 400,
            "completed_at": "2026-06-07T00:00:00+00:00",
        }

    async def clear_reward_tracks(self, track_types):
        track_types = set(track_types)
        self.reward_tracks = [row for row in self.reward_tracks if row.get("track_type") not in track_types]

    async def create_reward_track(self, **kwargs):
        row = {
            "id": self.next_reward_id,
            "is_active": True,
            **kwargs,
        }
        self.next_reward_id += 1
        self.reward_tracks.append(row)
        return row

    async def update_reward_track(self, reward_id, **fields):
        row = next((row for row in self.reward_tracks if int(row["id"]) == int(reward_id)), None)
        if not row:
            return {"error": "reward_not_found"}
        row.update(fields)
        return row

    async def delete_reward_track(self, reward_id):
        row = next((row for row in self.reward_tracks if int(row["id"]) == int(reward_id)), None)
        if not row:
            return {"error": "reward_not_found"}
        row["is_active"] = False
        return {"success": True}

    async def admin_set_extra_pass(self, admin_id, target_user_id, mode, days=None, reason=None):
        update = {
            "admin_id": admin_id,
            "target_user_id": target_user_id,
            "mode": mode,
            "days": days,
            "reason": reason,
        }
        self.extra_pass_updates.append(update)
        return {"status": "ok", "action": "set_extra_pass", "mode": mode}

    async def search_admin_players(self, **kwargs):
        return {"players": [], "total": 0, "filters": kwargs}

    async def admin_ban_user(self, admin_id, target_user_id, reason=None, until=None):
        if int(target_user_id) == 404:
            return {"error": "user_not_found"}
        return {"success": True, "action": "ban"}

    async def admin_update_user_account(self, admin_id, target_user_id, fields=None, reason=None):
        self.account_updates.append(
            {
                "admin_id": admin_id,
                "target_user_id": target_user_id,
                "fields": fields,
                "reason": reason,
            }
        )
        return {"success": True, "action": "update"}

    async def admin_adjust_resource(self, admin_id, target_user_id, resource, amount, reason=None):
        self.resource_adjustments.append(
            {
                "admin_id": admin_id,
                "target_user_id": target_user_id,
                "resource": resource,
                "amount": amount,
                "reason": reason,
            }
        )
        return {"success": True, "action": "resource"}


@pytest.mark.asyncio
async def test_mcp_ruble_product_adapter_validates_codes_and_gift_shop_sets():
    db = AdminWorkflowDB()
    db.sets[12] = {"id": 12, "name": "Gift Set", "is_active": True}

    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="invalid_code"):
        await mcp_admin_tools._normalize_ruble_product_payload(
            db,
            {
                "code": 'bad"code',
                "item_type": "gems_package",
                "package_type": "gems_100",
                "name": "Bad Code",
                "price": 99,
            },
            require_identity=True,
        )

    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="shop_set_id_required"):
        await mcp_admin_tools._normalize_ruble_product_payload(
            db,
            {
                "code": "gift_missing_set",
                "item_type": "gift_shop_set",
                "name": "Gift Missing Set",
                "price": 0,
            },
            require_identity=True,
        )

    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="gift_shop_set_price_must_be_zero"):
        await mcp_admin_tools._normalize_ruble_product_payload(
            db,
            {
                "code": "gift_paid",
                "item_type": "gift_shop_set",
                "shop_set_id": 12,
                "name": "Gift Paid",
                "price": 1,
            },
            require_identity=True,
        )

    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="invalid_currency"):
        await mcp_admin_tools._normalize_ruble_product_payload(
            db,
            {
                "code": "gift_gems",
                "item_type": "gift_shop_set",
                "shop_set_id": 12,
                "name": "Gift Gems",
                "price": 0,
                "currency": "gems",
            },
            require_identity=True,
        )

    normalized = await mcp_admin_tools._normalize_ruble_product_payload(
        db,
        {
            "code": "gift_set_12",
            "item_type": "gift_shop_set",
            "shop_set_id": 12,
            "name": "Gift Set 12",
            "price": 0,
        },
        require_identity=True,
    )

    assert normalized == {
        "code": "gift_set_12",
        "item_type": "gift_shop_set",
        "shop_set_id": 12,
        "package_type": None,
        "name": "Gift Set 12",
        "price": 0.0,
        "currency": "rubles",
    }


@pytest.mark.asyncio
async def test_mcp_ruble_product_update_by_id_preserves_gift_invariants():
    db = AdminWorkflowDB()
    db.sets[12] = {"id": 12, "name": "Gift Set", "is_active": True}
    db.products["paid_set"] = {
        "id": 7,
        "code": "paid_set",
        "item_type": "shop_set",
        "shop_set_id": 12,
        "name": "Paid Set",
        "price": 199.0,
        "currency": "rubles",
        "is_active": True,
    }
    app = {"db": db}

    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="gift_shop_set_price_must_be_zero"):
        await mcp_admin_tools.adapter_update_ruble_product(
            app,
            1,
            {
                "id": 7,
                "item_type": "gift_shop_set",
                "shop_set_id": 12,
                "dry_run": True,
            },
        )

    with pytest.raises(mcp_admin_tools.MCPToolInputError, match="invalid_currency"):
        await mcp_admin_tools.adapter_update_ruble_product(
            app,
            1,
            {
                "id": 7,
                "item_type": "gift_shop_set",
                "shop_set_id": 12,
                "price": 0,
                "currency": "gems",
                "dry_run": True,
            },
        )

    assert db.products["paid_set"]["item_type"] == "shop_set"


def _admin_token(session_id: str) -> str:
    return _token_for_user(session_id, ADMIN_ID)


def _token_for_user(session_id: str, user_id: int) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "user_id": int(user_id),
            "session_id": session_id,
            "iat": now,
            "exp": now + 600,
        },
        get_settings().jwt_secret,
        algorithm="HS256",
    )


async def _client(db):
    session_id = str(uuid.uuid4())
    app = web_server.create_web_app(
        db,
        bot_token="bot-token",
        extraid_db=WorkflowExtraIDDB(session_id),
        webapp_url="https://game.example",
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, _admin_token(session_id)


@pytest.mark.asyncio
async def test_admin_extra_shop_product_and_set_management_workflow(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        create_set = await client.post(
            "/api/admin/shop/sets/create",
            headers=headers,
            json={
                "name": "Smoke Set",
                "description": "integration smoke",
                "price": 199,
                "currency": "rubles",
                "rewards": [{"type": "gems", "amount": 100}],
            },
        )
        body = await create_set.json()
        assert create_set.status == 200
        set_id = body["set_id"]

        update_set = await client.post(
            "/api/admin/shop/sets/update",
            headers=headers,
            json={"set_id": set_id, "name": "Smoke Set v2", "price": 249},
        )
        assert update_set.status == 200
        assert db.sets[set_id]["name"] == "Smoke Set v2"

        create_product = await client.post(
            "/api/admin/ruble-products/create",
            headers=headers,
            json={
                "code": "smoke_pack",
                "item_type": "shop_set",
                "name": "Smoke Pack",
                "price": 249,
                "currency": "rubles",
                "shop_set_id": set_id,
                "show_in_game": True,
                "show_in_shop": True,
            },
        )
        assert create_product.status == 200
        assert db.products["smoke_pack"]["show_in_game"] is True

        update_product = await client.post(
            "/api/admin/ruble-products/update",
            headers=headers,
            json={"code": "smoke_pack", "name": "Smoke Pack v2", "is_active": True},
        )
        assert update_product.status == 200
        assert db.products["smoke_pack"]["name"] == "Smoke Pack v2"

        delete_product = await client.post(
            "/api/admin/ruble-products/delete",
            headers=headers,
            json={"code": "smoke_pack"},
        )
        assert delete_product.status == 200
        assert db.products["smoke_pack"]["is_active"] is False

        delete_set = await client.post(
            "/api/admin/shop/sets/delete",
            headers=headers,
            json={"set_id": set_id},
        )
        assert delete_set.status == 200
        assert db.sets[set_id]["is_active"] is False
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_extra_pass_season_rewards_and_player_pass_workflow(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        seasons = await client.get("/api/admin/seasons", headers=headers)
        assert seasons.status == 200
        assert (await seasons.json())["data"]["active"]["id"] == 1

        draft_response = await client.post(
            "/api/admin/seasons/create-draft",
            headers=headers,
            json={"preset_key": "blank"},
        )
        draft_body = await draft_response.json()
        assert draft_response.status == 200
        draft_id = draft_body["data"]["season"]["id"]

        update_response = await client.post(
            f"/api/admin/seasons/{draft_id}",
            headers=headers,
            json={
                "name": "Smoke Season",
                "status": "scheduled",
                "start_date": "2026-06-01T00:00:00+00:00",
                "end_date": "2026-07-01T00:00:00+00:00",
                "stage_cost_min": 4,
                "stage_cost_growth": 0.2,
                "stage_cost_exponent": 1.2,
                "stage_cost_cap": 30,
            },
        )
        assert update_response.status == 200
        updated_season = await db.get_season_by_id(draft_id)
        assert updated_season["status"] == "scheduled"
        assert updated_season["stage_cost_min"] == 4
        assert updated_season["stage_cost_cap"] == 30

        empty_import_response = await client.post(
            f"/api/admin/seasons/{draft_id}/rewards/import",
            headers=headers,
            json={"replace": True, "tracks": {"free": [], "premium": [], "ultra": []}},
        )
        empty_import_body = await empty_import_response.json()
        assert empty_import_response.status == 400
        assert empty_import_body["error"] == "empty_reward_tracks"

        import_response = await client.post(
            f"/api/admin/seasons/{draft_id}/rewards/import",
            headers=headers,
            json={
                "replace": True,
                "tracks": {
                    "free": [{"position": 1, "reward_type": "coins", "reward_amount": 100}],
                    "premium": [{"position": 2, "type": "gems", "amount": 25}],
                    "ultra": [{"position": 41, "reward_type": "card", "reward_amount": 1}],
                },
            },
        )
        import_body = await import_response.json()
        assert import_response.status == 200
        assert import_body["data"]["imported"] == 3
        returned_draft = next(season for season in import_body["data"]["seasons"] if int(season["id"]) == int(draft_id))
        assert returned_draft["progression_preview"][0]["required_stars"] == 5

        tier_response = await client.post(
            "/api/admin/rewards/tracks/create",
            headers=headers,
            json={
                "track_type": f"s{draft_id}_pass",
                "position": 3,
                "reward_type": "keys",
                "reward_amount": 1,
                "extra_pass_required": True,
            },
        )
        tier_body = await tier_response.json()
        assert tier_response.status == 200
        tier_id = tier_body["tier"]["id"]

        cosmetic_create = await client.post(
            "/api/admin/cosmetics/create",
            headers=headers,
            json={
                "slug": "avatar_gold",
                "item_type": "avatar",
                "class": "gold",
                "name": "Gold Avatar",
                "asset_path": "/DesignAssets/PlayerCosmetics/Avatars/Admin/avatar_gold.png",
            },
        )
        assert cosmetic_create.status == 200

        particles_reward = await client.post(
            "/api/admin/rewards/tracks/create",
            headers=headers,
            json={
                "track_type": f"s{draft_id}_pass",
                "position": 3,
                "reward_type": "particles",
                "reward_amount": 25,
                "reward_meta": {"card_id": 46},
                "extra_pass_required": True,
            },
        )
        particles_body = await particles_reward.json()
        assert particles_reward.status == 200
        assert particles_body["tier"]["reward_type"] == "particles"

        cosmetic_reward = await client.post(
            "/api/admin/rewards/tracks/create",
            headers=headers,
            json={
                "track_type": f"s{draft_id}_pass",
                "position": 4,
                "reward_type": "cosmetic",
                "reward_amount": 1,
                "reward_meta": {"cosmetic_slug": "avatar_gold", "auto_equip": True},
                "extra_pass_required": True,
            },
        )
        cosmetic_body = await cosmetic_reward.json()
        assert cosmetic_reward.status == 200
        assert cosmetic_body["tier"]["reward_type"] == "cosmetic"

        invalid_particles = await client.post(
            "/api/admin/rewards/tracks/create",
            headers=headers,
            json={
                "track_type": f"s{draft_id}_pass",
                "position": 5,
                "reward_type": "particles",
                "reward_amount": 1,
                "extra_pass_required": True,
            },
        )
        assert invalid_particles.status == 400
        assert (await invalid_particles.json())["error"] == "reward_card_id_required"

        out_of_scope = await client.post(
            "/api/admin/rewards/tracks/create",
            headers=headers,
            json={
                "track_type": f"s{draft_id}_ultra",
                "position": 1,
                "reward_type": "gems",
                "reward_amount": 1,
                "extra_pass_required": True,
            },
        )
        assert out_of_scope.status == 400
        assert (await out_of_scope.json())["error"] == "position_out_of_track_scope"

        update_tier = await client.post(
            "/api/admin/rewards/tracks/update",
            headers=headers,
            json={"id": tier_id, "reward_amount": 2},
        )
        assert update_tier.status == 200

        delete_tier = await client.post(
            "/api/admin/rewards/tracks/delete",
            headers=headers,
            json={"id": tier_id},
        )
        assert delete_tier.status == 200

        pass_response = await client.post(
            "/api/admin/players/4242/extra-pass",
            headers=headers,
            json={"mode": "ultra", "reason": "smoke"},
        )
        assert pass_response.status == 200
        assert db.extra_pass_updates[-1]["mode"] == "ultra"
        assert db.extra_pass_updates[-1]["days"] is None
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_season_reset_preview_execute_and_idempotency(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        unauthenticated_preview = await client.get("/api/admin/seasons/1/reset-preview")
        assert unauthenticated_preview.status in {401, 403}
        unauthenticated_execute = await client.post(
            "/api/admin/seasons/1/reset",
            json={"confirm": True},
        )
        assert unauthenticated_execute.status in {401, 403}

        token_payload = jwt.decode(token, options={"verify_signature": False})
        non_admin_headers = {"Authorization": f"Bearer {_token_for_user(token_payload['session_id'], 123456)}"}
        non_admin_preview = await client.get("/api/admin/seasons/1/reset-preview", headers=non_admin_headers)
        assert non_admin_preview.status == 403

        preview = await client.get("/api/admin/seasons/1/reset-preview", headers=headers)
        preview_body = await preview.json()
        assert preview.status == 200
        assert preview_body["data"]["summary"]["players"] == 2
        assert preview_body["data"]["summary"]["keys_granted"] == 2
        assert preview_body["data"]["players_limit"] == 200
        assert preview_body["data"]["players_truncated"] is False

        missing_confirm = await client.post(
            "/api/admin/seasons/1/reset",
            headers=headers,
            json={"reason": "smoke"},
        )
        assert missing_confirm.status == 400
        assert (await missing_confirm.json())["error"] == "confirm_required"

        string_confirm = await client.post(
            "/api/admin/seasons/1/reset",
            headers=headers,
            json={"confirm": "false"},
        )
        assert string_confirm.status == 400
        assert (await string_confirm.json())["error"] == "confirm_required"

        non_object_payload = await client.post(
            "/api/admin/seasons/1/reset",
            headers=headers,
            json=[],
        )
        assert non_object_payload.status == 400
        assert (await non_object_payload.json())["error"] == "invalid_reset_payload"

        mismatched_season = await client.post(
            "/api/admin/seasons/1/reset",
            headers=headers,
            json={"confirm": True, "confirm_season_id": 2},
        )
        assert mismatched_season.status == 400
        assert (await mismatched_season.json())["error"] == "confirm_season_mismatch"

        long_reason = await client.post(
            "/api/admin/seasons/1/reset",
            headers=headers,
            json={"confirm": True, "reason": "x" * 501},
        )
        assert long_reason.status == 400
        assert (await long_reason.json())["error"] == "reason_too_long"

        missing_season = await client.post(
            "/api/admin/seasons/999/reset",
            headers=headers,
            json={"confirm": True},
        )
        assert missing_season.status == 404
        assert (await missing_season.json())["error"] == "season_not_found"

        non_active = await client.post(
            "/api/admin/seasons/create-draft",
            headers=headers,
            json={"preset_key": "blank"},
        )
        draft_id = (await non_active.json())["data"]["season"]["id"]
        inactive_preview = await client.get(
            f"/api/admin/seasons/{draft_id}/reset-preview",
            headers=headers,
        )
        assert inactive_preview.status == 400
        assert (await inactive_preview.json())["error"] == "season_reset_requires_active_season"

        inactive_execute = await client.post(
            f"/api/admin/seasons/{draft_id}/reset",
            headers=headers,
            json={"confirm": True},
        )
        assert inactive_execute.status == 400
        assert (await inactive_execute.json())["error"] == "season_reset_requires_active_season"

        execute = await client.post(
            "/api/admin/seasons/1/reset",
            headers=headers,
            json={"confirm": True, "reason": "smoke reset"},
        )
        execute_body = await execute.json()
        assert execute.status == 200
        assert execute_body["data"]["reset"]["processed_players"] == 2
        assert execute_body["data"]["seasons"][0]["reset"]["status"] == "completed"
        assert db.reset_execute_kwargs[-1]["require_active"] is True
        assert db.reset_execute_kwargs[-1]["admin_user_id"] == ADMIN_ID

        duplicate = await client.post(
            "/api/admin/seasons/1/reset",
            headers=headers,
            json={"confirm": True},
        )
        assert duplicate.status == 409
        assert (await duplicate.json())["error"] == "season_reset_already_completed"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_season_reset_preview_truncates_player_sample(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    db.reset_preview = {
        "season_id": 1,
        "already_completed": False,
        "summary": {
            "players": 250,
            "trophies_reduced": 25000,
            "keys_granted": 250,
            "coins_granted": 50000,
            "stars_reset": 1000,
        },
        "players": [
            {
                "user_id": 10_000 + idx,
                "old_trophies": 799,
                "new_trophies": 600,
                "old_stars": 4,
                "excess_trophies": 199,
                "granted_keys": 1,
                "granted_coins": 200,
            }
            for idx in range(250)
        ],
    }
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        preview = await client.get("/api/admin/seasons/1/reset-preview", headers=headers)
        body = await preview.json()

        assert preview.status == 200
        assert body["data"]["summary"]["players"] == 250
        assert len(body["data"]["players"]) == 200
        assert body["data"]["players_limit"] == 200
        assert body["data"]["players_truncated"] is True
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_product_options_and_validation(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        options_response = await client.get("/api/admin/ruble-products/options", headers=headers)
        options_body = await options_response.json()
        assert options_response.status == 200
        item_types = {item["value"] for item in options_body["data"]["item_types"]}
        assert {"extrapass", "extrapass_ultra", "starter_boost", "squad_boost", "gems_package", "shop_set"} <= item_types
        assert "starter_once" in {item["value"] for item in options_body["data"]["package_types"]["gems_package"]}

        invalid_type = await client.post(
            "/api/admin/ruble-products/create",
            headers=headers,
            json={"code": "bad", "item_type": "manual_bad", "name": "Bad", "price": 1},
        )
        assert invalid_type.status == 400
        assert (await invalid_type.json())["error"] == "invalid_item_type"

        missing_package = await client.post(
            "/api/admin/ruble-products/create",
            headers=headers,
            json={"code": "gems_missing", "item_type": "gems_package", "name": "Gems", "price": 49},
        )
        assert missing_package.status == 400
        assert (await missing_package.json())["error"] == "package_type_required"

        create_product = await client.post(
            "/api/admin/ruble-products/create",
            headers=headers,
            json={
                "code": "gems_smoke",
                "item_type": "gems_package",
                "package_type": "gems_100",
                "name": "100 gems",
                "price": 99,
                "rustore_product_id": "ea.gems.100",
                "image_url": "/extraShop/uploads/products/gems_100.webp",
            },
        )
        assert create_product.status == 200
        assert db.products["gems_smoke"]["metadata"]["rustore_product_id"] == "ea.gems.100"
        assert db.products["gems_smoke"]["image_url"] == "/extraShop/uploads/products/gems_100.webp"

        update_rustore_id = await client.post(
            "/api/admin/ruble-products/update",
            headers=headers,
            json={
                "code": "gems_smoke",
                "rustore_product_id": "ea.gems.100.v2",
                "image_url": "https://cdn.example/products/gems-100.jpg",
            },
        )
        assert update_rustore_id.status == 200
        assert db.products["gems_smoke"]["metadata"]["rustore_product_id"] == "ea.gems.100.v2"
        assert db.products["gems_smoke"]["image_url"] == "https://cdn.example/products/gems-100.jpg"

        for unsafe_url in (
            "javascript:alert(1)",
            "data:image/svg+xml,<svg>",
            "http://cdn.example/products/gems.jpg",
            "/extraShop/uploads/products/../secret.png",
            "/extraShop/uploads/products/nested/gems.png",
            "/extraShop/uploads/products/bad.svg",
            'https://cdn.example/products/gems.jpg" onerror="alert(1)',
        ):
            invalid_image = await client.post(
                "/api/admin/ruble-products/update",
                headers=headers,
                json={"code": "gems_smoke", "image_url": unsafe_url},
            )
            assert invalid_image.status == 400
            assert (await invalid_image.json())["error"] == "invalid_image_url"

        duplicate = await client.post(
            "/api/admin/ruble-products/create",
            headers=headers,
            json={
                "code": "gems_smoke",
                "item_type": "gems_package",
                "package_type": "gems_100",
                "name": "100 gems again",
                "price": 99,
            },
        )
        assert duplicate.status == 409
        assert (await duplicate.json())["error"] == "product_code_exists"

        db.products["legacy_bad_image"] = {
            "id": 999,
            "code": "legacy_bad_image",
            "item_type": "extrapass",
            "package_type": None,
            "name": "Legacy Bad Image",
            "price": 179,
            "currency": "rubles",
            "is_active": True,
            "show_in_shop": True,
            "image_url": "javascript:alert(1)",
        }
        public_response = await client.get("/api/shop/ruble-products")
        public_body = await public_response.json()
        public_legacy = next(product for product in public_body["products"] if product["code"] == "legacy_bad_image")
        assert public_response.status == 200
        assert public_legacy["image_url"] is None
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_configs_summary_returns_partial_payload_when_blocks_fail(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = await client.get("/api/admin/configs", headers=headers)
        body = await response.json()

        assert response.status == 200
        assert body["status"] == "ok"
        assert body["data"]["runtime_config"]["maintenance_mode"]["enabled"] is False
        assert body["data"]["promocodes_count"] == 0
        assert "cards" in body["data"]["errors"]
        assert "squads" in body["data"]["errors"]
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_case_config_http_read_and_patch_live(monkeypatch):
    """POST /api/admin/case-config применяет partial-патч в реальном времени.

    КРИТИЧНО: partial base_particles патч сохраняет остальные редкости (deep-merge),
    иначе fix limited-base был бы потерян при следующей правке админкой.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        # GET — дефолты заполнены
        read = await client.get("/api/admin/case-config", headers=headers)
        read_body = await read.json()
        assert read.status == 200
        assert read_body["status"] == "ok"
        assert read_body["data"]["base_particles_by_rarity"]["limited"] == 150

        # POST partial base_particles patch
        patch = await client.post(
            "/api/admin/case-config",
            headers=headers,
            json={"patch": {"base_particles_by_rarity": {"limited": 777}}},
        )
        patch_body = await patch.json()
        assert patch.status == 200
        bpr = patch_body["data"]["base_particles_by_rarity"]
        assert bpr["limited"] == 777
        # остальные редкости сохранены (deep-merge, а не shallow replace)
        assert bpr["common"] == 2
        assert bpr["divine"] == 100

        # Невалидный патч → 400
        bad = await client.post(
            "/api/admin/case-config",
            headers=headers,
            json={"patch": {"tier_rarity_probabilities": {1: {"common": 0.1, "rare": 0.1}}}},
        )
        assert bad.status == 400
        bad_body = await bad.json()
        assert "invalid_tier_rarity_sum" in bad_body["error"]
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_squads_analytics_returns_empty_payload_when_backend_missing(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = await client.get("/api/admin/squads/analytics", headers=headers)
        body = await response.json()

        assert response.status == 200
        assert body["status"] == "ok"
        assert body["data"]["summary"]["total_squads"] == 0
        assert "analytics" in body["data"]["errors"]
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_season_update_normalizes_current_season_and_validation(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        invalid_dates = await client.post(
            "/api/admin/seasons/1",
            headers=headers,
            json={
                "start_date": "2026-07-01T00:00:00+00:00",
                "end_date": "2026-06-01T00:00:00+00:00",
            },
        )
        assert invalid_dates.status == 400
        assert (await invalid_dates.json())["error"] == "season_dates_invalid"

        invalid_status = await client.post(
            "/api/admin/seasons/1",
            headers=headers,
            json={"status": "publishing"},
        )
        assert invalid_status.status == 400
        assert (await invalid_status.json())["error"] == "invalid_season_status"

        update_response = await client.post(
            "/api/admin/seasons/1",
            headers=headers,
            json={
                "status": "active",
                "is_active": False,
                "start_date": "",
                "end_date": "",
                "max_stars": 45,
                "stage_cost_min": 4,
                "stage_cost_growth": 0.1,
                "stage_cost_exponent": 1.0,
                "stage_cost_cap": 20,
                "pass_end_position": 40,
                "ultra_start_position": 41,
            },
        )
        body = await update_response.json()
        assert update_response.status == 200
        assert body["data"]["season"]["is_active"] is True
        assert body["data"]["season"]["stage_cost_min"] == 4
        assert body["data"]["season"]["stage_cost_cap"] == 20
        assert body["data"]["season"]["start_date"] is None
        assert body["data"]["season"]["end_date"] is None
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_season_update_rejects_duplicate_track_types(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = await client.post(
            "/api/admin/seasons/1",
            headers=headers,
            json={
                "free_track_type": "same_track",
                "pass_track_type": "same_track",
                "ultra_track_type": "s1_ultra",
            },
        )

        assert response.status == 400
        assert (await response.json())["error"] == "duplicate_season_track_types"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_season_update_rejects_track_type_reused_by_another_season(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    db.seasons.append(
        {
            "id": 2,
            "slug": "season-2",
            "name": "Season 2",
            "season_number": 2,
            "status": "draft",
            "is_active": False,
            "start_date": None,
            "end_date": None,
            "free_track_type": "s2_free",
            "pass_track_type": "s2_pass",
            "ultra_track_type": "s2_ultra",
            "max_stars": 45,
            "stage_cost_min": 3,
            "stage_cost_growth": 0.07,
            "stage_cost_exponent": 1.5,
            "stage_cost_cap": 25,
            "pass_end_position": 40,
            "ultra_start_position": 41,
        }
    )
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = await client.post(
            "/api/admin/seasons/1",
            headers=headers,
            json={"pass_track_type": "s2_pass"},
        )

        assert response.status == 400
        assert (await response.json())["error"] == "season_track_type_reused"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_season_partial_update_validates_against_existing_season(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        invalid_start = await client.post(
            "/api/admin/seasons/1",
            headers=headers,
            json={"start_date": "2026-06-02T00:00:00+00:00"},
        )

        assert invalid_start.status == 400
        assert (await invalid_start.json())["error"] == "season_dates_invalid"

        invalid_positions = await client.post(
            "/api/admin/seasons/1",
            headers=headers,
            json={"max_stars": 30},
        )

        assert invalid_positions.status == 400
        assert (await invalid_positions.json())["error"] == "invalid_season_positions"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_endpoint_input_validation_and_player_error_mapping(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        invalid_limit = await client.get("/api/admin/players/list?limit=nope", headers=headers)
        assert invalid_limit.status == 400
        assert (await invalid_limit.json())["error"] == "invalid_limit"

        invalid_days = await client.get("/api/admin/players/activity?days=abc", headers=headers)
        assert invalid_days.status == 400
        assert (await invalid_days.json())["error"] == "invalid_days"

        not_found = await client.post(
            "/api/admin/players/404/ban",
            headers=headers,
            json={"reason": "missing user"},
        )
        not_found_body = await not_found.json()
        assert not_found.status == 404
        assert not_found_body["error"] == "user_not_found"

        invalid_amount = await client.post(
            "/api/admin/players/4242/resource",
            headers=headers,
            json={"resource": "gems", "amount": "nan"},
        )
        assert invalid_amount.status == 400
        assert (await invalid_amount.json())["error"] == "invalid_amount"

        missing_fields = await client.post(
            "/api/admin/players/4242/update",
            headers=headers,
            json={"status": "banned"},
        )
        assert missing_fields.status == 400
        assert (await missing_fields.json())["error"] == "fields_required"

        self_ban = await client.post(
            f"/api/admin/players/{ADMIN_ID}/update",
            headers=headers,
            json={"fields": {"status": "banned"}},
        )
        assert self_ban.status == 400
        assert (await self_ban.json())["error"] == "self_ban_requires_confirm"

        invalid_extra_pass_days = await client.post(
            "/api/admin/players/4242/extra-pass",
            headers=headers,
            json={"mode": "grant", "days": "abc"},
        )
        assert invalid_extra_pass_days.status == 400
        assert (await invalid_extra_pass_days.json())["error"] == "invalid_days"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_promocode_rewards_are_validated(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        invalid_payloads = [
            {"code": "NEG", "reward_gems": -1},
            {"code": "TXT", "reward_coins": "ten"},
            {"code": "HUGE", "reward_keys": 1_000_000_001},
            {"code": "EMPTY"},
        ]
        for payload in invalid_payloads:
            response = await client.post(
                "/api/admin/promocodes/create",
                headers=headers,
                json=payload,
            )
            assert response.status == 400
            assert (await response.json())["error"]

        valid = await client.post(
            "/api/admin/promocodes/create",
            headers=headers,
            json={"code": "GOOD", "reward_gems": "25", "reward_extrapass": False},
        )
        body = await valid.json()
        assert valid.status == 200
        assert body["success"] is True
        assert db.promocodes[-1]["reward_gems"] == 25
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_missing_shop_product_and_reward_targets_map_to_404(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        missing_set = await client.post(
            "/api/admin/shop/sets/delete",
            headers=headers,
            json={"set_id": 999},
        )
        assert missing_set.status == 404
        assert (await missing_set.json())["error"] == "set_not_found"

        missing_product = await client.post(
            "/api/admin/ruble-products/delete",
            headers=headers,
            json={"code": "missing"},
        )
        assert missing_product.status == 404
        assert (await missing_product.json())["error"] == "product_not_found"

        missing_reward_update = await client.post(
            "/api/admin/rewards/tracks/update",
            headers=headers,
            json={"id": 999, "reward_amount": 2},
        )
        assert missing_reward_update.status == 404
        assert (await missing_reward_update.json())["error"] == "reward_not_found"

        missing_reward_delete = await client.post(
            "/api/admin/rewards/tracks/delete",
            headers=headers,
            json={"id": 999},
        )
        assert missing_reward_delete.status == 404
        assert (await missing_reward_delete.json())["error"] == "reward_not_found"
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_admin_product_image_upload_rejects_mismatched_and_oversized_payloads(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        mismatched = FormData()
        mismatched.add_field(
            "file",
            b"not actually a png",
            filename="bad.png",
            content_type="image/png",
        )
        mismatch_response = await client.post(
            "/api/admin/uploads/product-image",
            headers=headers,
            data=mismatched,
        )
        assert mismatch_response.status == 400
        assert (await mismatch_response.json())["error"] == "invalid_image_signature"

        oversized = FormData()
        oversized.add_field(
            "file",
            b"\x89PNG\r\n\x1a\n" + (b"x" * (5 * 1024 * 1024 + 1)),
            filename="too-large.png",
            content_type="image/png",
        )
        oversized_response = await client.post(
            "/api/admin/uploads/product-image",
            headers=headers,
            data=oversized,
        )
        assert oversized_response.status == 400
        assert (await oversized_response.json())["error"] == "file_too_large"
    finally:
        await client.close()
        get_settings.cache_clear()


def _png_bytes(width: int, height: int) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), (245, 0, 160)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_admin_cosmetic_upload_create_list_and_delete_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    monkeypatch.setattr(web_server, "DESIGN_ASSETS_DIR", tmp_path / "DesignAssets")
    db = AdminWorkflowDB()
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        avatar = FormData()
        avatar.add_field("item_type", "avatar")
        avatar.add_field("slug", "extra_cards_avatar")
        avatar.add_field(
            "file",
            _png_bytes(750, 750),
            filename="avatar.png",
            content_type="image/png",
        )
        upload_response = await client.post(
            "/api/admin/cosmetics/upload-image",
            headers=headers,
            data=avatar,
        )
        upload_payload = await upload_response.json()

        assert upload_response.status == 200
        assert upload_payload["success"] is True
        assert upload_payload["dimensions"] == {"width": 750, "height": 750}
        assert upload_payload["asset_path"].startswith("/DesignAssets/PlayerCosmetics/Avatars/Admin/")
        assert (tmp_path / upload_payload["asset_path"].lstrip("/")).is_file()

        file_first = FormData()
        file_first.add_field(
            "file",
            _png_bytes(750, 750),
            filename="avatar-second.png",
            content_type="image/png",
        )
        file_first.add_field("item_type", "avatar")
        file_first.add_field("slug", "extra_cards_avatar_second")
        file_first_response = await client.post(
            "/api/admin/cosmetics/upload-image",
            headers=headers,
            data=file_first,
        )
        file_first_payload = await file_first_response.json()

        assert file_first_response.status == 200
        assert file_first_payload["asset_path"].startswith("/DesignAssets/PlayerCosmetics/Avatars/Admin/")

        create_response = await client.post(
            "/api/admin/cosmetics/create",
            headers=headers,
            json={
                "slug": "extra_cards_avatar",
                "item_type": "avatar",
                "class": "rare",
                "name": "Extra Cards Avatar",
                "asset_path": upload_payload["asset_path"],
                "media_type": "image",
                "sort_order": 77,
            },
        )
        create_payload = await create_response.json()

        assert create_response.status == 200
        assert create_payload["success"] is True
        assert create_payload["cosmetic"]["slug"] == "extra_cards_avatar"

        update_response = await client.post(
            "/api/admin/cosmetics/update",
            headers=headers,
            json={
                "id": create_payload["cosmetic"]["id"],
                "asset_path": file_first_payload["asset_path"],
            },
        )
        update_payload = await update_response.json()

        assert update_response.status == 200
        assert update_payload["cosmetic"]["asset_path"] == file_first_payload["asset_path"]

        title_update_response = await client.post(
            "/api/admin/cosmetics/update",
            headers=headers,
            json={
                "id": create_payload["cosmetic"]["id"],
                "item_type": "title",
            },
        )
        title_update_payload = await title_update_response.json()

        assert title_update_response.status == 200
        assert title_update_payload["cosmetic"]["item_type"] == "title"
        assert title_update_payload["cosmetic"]["media_type"] == "text"
        assert title_update_payload["cosmetic"]["asset_path"] is None

        title_response = await client.post(
            "/api/admin/cosmetics/create",
            headers=headers,
            json={
                "slug": "title_extra_old",
                "item_type": "title",
                "class": "epic",
                "name": "Экстра олд",
                "sort_order": 78,
            },
        )
        assert title_response.status == 200

        invalid_title_response = await client.post(
            "/api/admin/cosmetics/create",
            headers=headers,
            json={
                "slug": "title_bad_image",
                "item_type": "title",
                "class": "epic",
                "name": "Bad image title",
                "media_type": "image",
            },
        )
        assert invalid_title_response.status == 400
        assert (await invalid_title_response.json())["error"] == "invalid_media_type"

        list_response = await client.get("/api/admin/cosmetics", headers=headers)
        list_payload = await list_response.json()
        assert list_response.status == 200
        assert [item["slug"] for item in list_payload["items"]] == [
            "extra_cards_avatar",
            "title_extra_old",
        ]

        duplicate_response = await client.post(
            "/api/admin/cosmetics/create",
            headers=headers,
            json={
                "slug": "extra_cards_avatar",
                "item_type": "avatar",
                "class": "rare",
                "name": "Duplicate",
                "asset_path": upload_payload["asset_path"],
            },
        )
        assert duplicate_response.status == 409
        assert (await duplicate_response.json())["error"] == "cosmetic_slug_exists"

        delete_response = await client.post(
            "/api/admin/cosmetics/delete",
            headers=headers,
            json={"id": create_payload["cosmetic"]["id"]},
        )
        assert delete_response.status == 200
        assert db.cosmetics[create_payload["cosmetic"]["id"]]["is_active"] is False
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_shop_set_list_responses_normalize_json_string_rewards(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()
    db = AdminWorkflowDB()
    db.sets[1] = {
        "id": 1,
        "name": "String rewards",
        "description": None,
        "price": 0,
        "currency": "rubles",
        "is_active": True,
        "rewards": json.dumps([
            {"type": "cosmetic", "cosmetic_slug": "avatar_live"},
            {"type": "gems", "amount": 50},
        ]),
    }
    client, token = await _client(db)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        admin_response = await client.get("/api/admin/shop/sets", headers=headers)
        public_response = await client.get("/api/shop/sets")

        assert admin_response.status == 200
        assert public_response.status == 200
        assert (await admin_response.json())["sets"][0]["rewards"] == [
            {"type": "cosmetic", "cosmetic_slug": "avatar_live"},
            {"type": "gems", "amount": 50},
        ]
        assert (await public_response.json())["sets"][0]["rewards"] == [
            {"type": "cosmetic", "cosmetic_slug": "avatar_live"},
            {"type": "gems", "amount": 50},
        ]
    finally:
        await client.close()
        get_settings.cache_clear()
