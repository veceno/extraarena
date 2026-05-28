import time
import uuid

import jwt
import pytest
from aiohttp.test_utils import TestClient, TestServer

from bot.constants import ADMIN_ID
from infrastructure.config import get_settings
from web import server as web_server


class WorkflowExtraIDDB:
    def __init__(self, session_id: str):
        self.session_id = session_id

    async def verify_session(self, session_uuid, token: str):
        if str(session_uuid) != self.session_id:
            return None
        return {"session_id": session_uuid, "user_id": ADMIN_ID}


class AdminWorkflowDB:
    def __init__(self):
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
                "pass_end_position": 40,
                "ultra_start_position": 41,
            }
        ]
        self.reward_tracks = []
        self.next_reward_id = 1
        self.extra_pass_updates = []

    async def is_admin(self, user_id):
        return False

    async def get_runtime_config(self):
        return {
            "maintenance_mode": {"enabled": False},
            "feature_availability": {},
            "disabled_card_ids": [],
        }

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
            "pass_end_position": 40,
            "ultra_start_position": 41,
            "preset_key": preset_key,
        }
        self.seasons.append(season)
        return season

    async def update_season(self, season_id, **kwargs):
        season = await self.get_season_by_id(season_id)
        if not season:
            return {"error": "season_not_found"}
        season.update(kwargs)
        return season

    async def get_season_by_id(self, season_id):
        return next((season for season in self.seasons if int(season["id"]) == int(season_id)), None)

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
        if row:
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


def _admin_token(session_id: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "user_id": ADMIN_ID,
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
            },
        )
        assert update_response.status == 200
        assert (await db.get_season_by_id(draft_id))["status"] == "scheduled"

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
            json={"mode": "ultra", "days": 30, "reason": "smoke"},
        )
        assert pass_response.status == 200
        assert db.extra_pass_updates[-1]["mode"] == "ultra"
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
        assert {"extrapass", "extrapass_ultra", "starter_boost", "gems_package", "shop_set"} <= item_types
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
            },
        )
        assert create_product.status == 200

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
        assert "promocodes" in body["data"]["errors"]
        assert "cards" in body["data"]["errors"]
        assert "squads" in body["data"]["errors"]
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
                "pass_end_position": 40,
                "ultra_start_position": 41,
            },
        )
        body = await update_response.json()
        assert update_response.status == 200
        assert body["data"]["season"]["is_active"] is True
        assert body["data"]["season"]["start_date"] is None
        assert body["data"]["season"]["end_date"] is None
    finally:
        await client.close()
        get_settings.cache_clear()
