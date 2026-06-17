from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from infrastructure.config import get_settings
from infrastructure.database import Database
from web import server as web_server


class SquadInvariantDB:
    def __init__(self, role: str = "officer"):
        self.role = role
        self.settings_updates = []

    async def get_runtime_config(self):
        return {
            "maintenance_mode": {"enabled": False},
            "feature_availability": {"squads": True},
            "disabled_card_ids": [],
        }

    async def is_admin(self, user_id):
        return False

    async def get_match_mode_overrides(self):
        return []

    async def is_match_mode_enabled(self, mode_id):
        return True

    async def get_disabled_card_ids(self):
        return []

    async def get_user_clan(self, user_id):
        return {
            "id": 10,
            "name": "Audit Squad",
            "tag": "AUD",
            "member_role": self.role,
            "description": "",
            "type": "open",
            "min_trophies": 0,
        }

    async def update_clan_settings(self, clan_id, **fields):
        self.settings_updates.append((clan_id, fields))
        return True


@pytest.mark.asyncio
async def test_officer_cannot_update_squad_settings(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db = SquadInvariantDB(role="officer")
    app = web_server.create_web_app(db, bot_token="bot-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/squads/settings?user_id=42",
            json={"name": "New Squad Name"},
        )
        body = await response.json()

        assert response.status == 403
        assert body["error"] == "no_permission"
        assert db.settings_updates == []
    finally:
        await client.close()
        get_settings.cache_clear()


def test_squad_membership_schema_enforces_single_squad_per_user():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")

    assert "clan_members_user_id_unique_idx" in source
    assert "ON clan_members (user_id)" in source


def test_squad_join_and_accept_paths_are_transactional():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    clan_join_block = source.split("async def clan_join", 1)[1].split("async def random_join_squad", 1)[0]
    accept_block = source.split("async def accept_join_request", 1)[1].split("async def reject_join_request", 1)[0]

    assert "async with conn.transaction()" in clan_join_block
    assert "INSERT INTO clan_members" in clan_join_block
    assert "RETURNING members_count" in clan_join_block
    assert "FOR UPDATE" in accept_block
    assert "status = 'failed'" in accept_block
    assert "status = 'rejected'" in accept_block


def test_squad_cbrp_awards_can_be_bound_to_expected_clan():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    award_block = source.split("async def award_squad_cbrp", 1)[1].split("async def get_squad_cbrp_events", 1)[0]
    weekly_block = source.split("async def process_weekly_squad_cbrp", 1)[1].split("async def award_squad_seasonal_cbrp", 1)[0]

    assert "clan_id: Optional[int] = None" in award_block
    assert "AND cm.clan_id = $" in award_block
    assert "clan_id=int(row[\"clan_id\"])" in weekly_block


@pytest.mark.asyncio
async def test_squad_cbrp_owner_tax_without_boosts_does_not_shadow_math():
    class FakeTransaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        def transaction(self):
            return FakeTransaction()

        async def fetchrow(self, query, *args):
            if "owner_tax_remainder_micros" in query:
                return {"owner_tax_remainder_micros": 0}
            if "INSERT INTO squad_cbrp_events" in query:
                return {"id": 99}
            raise AssertionError(f"unexpected fetchrow query: {query}")

        async def execute(self, query, *args):
            return "UPDATE 1"

    class FakeAcquire:
        def __init__(self, conn):
            self.conn = conn

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakePool:
        def __init__(self):
            self.conn = FakeConnection()

        def acquire(self):
            return FakeAcquire(self.conn)

    class AwardDB:
        award_squad_cbrp = Database.award_squad_cbrp
        _resolve_squad_reward = Database._resolve_squad_reward

        def __init__(self):
            self._pool = FakePool()

        async def fetchrow(self, query, *args):
            return {
                "clan_id": 10,
                "role": "member",
                "owner_id": 100,
                "has_boost": False,
            }

        async def get_squad_runtime_config(self):
            return {
                "squad_rewards": {},
                "squad_creator_passive_tax_pct": 0.15,
                "squad_clan_boost_token_multiplier": 1.0,
            }

        async def get_clan_upgrades(self, clan_id):
            return {"cbrp_boost": 0}

    result = await AwardDB().award_squad_cbrp(
        200,
        "weekly_trophy_delta",
        clan_id=10,
        source_id="regression-owner-tax",
        personal_tokens=10,
    )

    assert result["awarded"] is True
    assert result["owner_tax_tokens"] == 1
    assert result["personal_tokens"] == 9


def test_squad_upgrade_level_is_recomputed_under_lock():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    upgrade_block = source.split("async def buy_clan_upgrade", 1)[1].split("async def buy_slot_upgrade", 1)[0]

    assert "FOR UPDATE" in upgrade_block
    assert "SELECT level FROM clan_upgrades" in upgrade_block
    assert "next_level = current_level + 1" in upgrade_block.split("async with conn.transaction()", 1)[1]


def test_squad_and_announcement_api_blocks_unmoderated_external_images():
    source = Path("web/server.py").read_text(encoding="utf-8")
    settings_block = source.split("async def squads_settings_handler", 1)[1].split("async def squads_members_action_handler", 1)[0]
    upload_block = source.split("async def community_upload_image_handler", 1)[1].split("async def community_image_static_handler", 1)[0]
    announce_block = source.split("async def community_announcements_create_handler", 1)[1].split("async def community_announcements_react_handler", 1)[0]

    assert '_require_squad_role(user_id, ("creator",))' in settings_block
    assert "_validate_local_upload_url" in settings_block
    assert "moderate_content" in upload_block
    assert "image_b64=base64.b64encode(data).decode(\"ascii\")" in upload_block
    assert "invalid_image_url" in announce_block
    assert "pin_price=pin_price" in announce_block


def test_key_case_and_battle_cbrp_sources_are_stable_and_include_bot_matches():
    source = Path("web/server.py").read_text(encoding="utf-8")
    battle_block = source.split("Failed to award squad CBRP for battle", 1)[0].rsplit("eligible_mode", 1)[1]
    claim_helper_block = source.split("async def _claim_case_key_opening", 1)[1].split("async def case_roll_from_keys_handler", 1)[0]
    case_block = source.split("async def case_open_from_keys_handler", 1)[1].split("async def debug_add_key_handler", 1)[0]

    assert "p1_status == ReplacementStatus.ACTIVE" in battle_block
    assert "p2_status == ReplacementStatus.ACTIVE" in battle_block
    assert "not p1_is_bot" not in battle_block
    assert "not p2_is_bot" not in battle_block
    assert "opening_id = roll_token or" in case_block
    assert "response_payload, status = await _claim_case_key_opening(user_id, opening_token)" in case_block
    assert 'opening_id = str(opening.get("opening_id") or opening_token)' in claim_helper_block
    assert 'event_source = f"case_open_key:{user_id}:{opening_id}"' in claim_helper_block
    assert "uuid.uuid4().hex" not in claim_helper_block.split('event_source = f"case_open_key:{user_id}:{opening_id}"', 1)[0]


def test_community_moderation_has_no_hardcoded_key_and_rate_limit_uses_window():
    config_source = Path("infrastructure/community_config.py").read_text(encoding="utf-8")
    moderation_source = Path("infrastructure/moderation.py").read_text(encoding="utf-8")

    assert 'MODERATION_API_KEY: str = os.environ.get("POLZA_AI_KEY", "")' in config_source
    assert "pza_" not in config_source
    assert "if not MODERATION_API_KEY" in moderation_source
    assert "NOW() - ($2 || ' minutes')::INTERVAL" in moderation_source
