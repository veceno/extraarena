import io
from pathlib import Path

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from infrastructure.config import get_settings
from infrastructure.database import Database
from web import server as web_server


class SquadInvariantDB:
    _public_squad_upgrade_catalog = staticmethod(Database._public_squad_upgrade_catalog)

    def __init__(self, role: str = "officer", has_boost: bool = False, chat_link: str | None = None):
        self.role = role
        self.has_boost = has_boost
        self.chat_link = chat_link
        self.settings_updates = []
        self.upgrade_calls = []

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
            "has_boost": self.has_boost,
            "chat_link": self.chat_link,
        }

    async def update_clan_settings(self, clan_id, **fields):
        self.settings_updates.append((clan_id, fields))
        return True

    async def get_clan_upgrades(self, clan_id):
        return {}

    async def get_squad_runtime_config(self):
        return {
            "squad_upgrades": {
                "boost": {"title": "BOOST"},
                "customization": {"title": "Кастомизация"},
                "member_slots": {"title": "Слоты участников"},
            },
            "squad_personal_rewards": [],
        }

    async def get_squad_shop_state(self, clan_id, user_id):
        return await Database.get_squad_shop_state(self, clan_id, user_id)

    async def fetchval(self, query, *args):
        if "has_boost" in query:
            return True
        return None

    async def fetch(self, query, *args):
        if "squad_shop_purchases" in query:
            return []
        return []

    async def buy_clan_upgrade(self, clan_id, actor_id, upgrade_type):
        self.upgrade_calls.append((clan_id, actor_id, upgrade_type))
        return {"upgrade_type": upgrade_type, "level": 1, "clan": await self.get_user_clan(actor_id)}

    async def count_recent_submissions(self, user_id, *, minutes):
        return 0

    async def record_submission(self, user_id, category):
        return None

    async def process_weekly_squad_cbrp(self):
        return {"processed": True, "awarded": 0}

    async def refresh_due_rating_snapshots(self, *, scope):
        return {"refreshed": 0}


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


@pytest.mark.asyncio
async def test_squad_avatar_url_update_does_not_require_customization_upgrade(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    async def approve_moderation(*args, **kwargs):
        return {"decision": "approve", "reason": ""}
    monkeypatch.setattr("infrastructure.moderation.moderate_content", approve_moderation)
    get_settings.cache_clear()
    db = SquadInvariantDB(role="creator")
    app = web_server.create_web_app(db, bot_token="bot-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/squads/settings?user_id=42",
            json={"avatar_url": ""},
        )
        body = await response.json()

        assert response.status == 200
        assert body["success"] is True
        assert db.settings_updates == [(10, {"avatar_url": None})]
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_squad_banner_url_update_requires_active_boost(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    async def approve_moderation(*args, **kwargs):
        return {"decision": "approve", "reason": ""}
    monkeypatch.setattr("infrastructure.moderation.moderate_content", approve_moderation)
    get_settings.cache_clear()
    db = SquadInvariantDB(role="creator", has_boost=False)
    app = web_server.create_web_app(db, bot_token="bot-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/squads/settings?user_id=42",
            json={"banner_url": "/uploads/squads/10_banner_allowed.webp"},
        )
        body = await response.json()

        assert response.status == 403
        assert body["error"] == "boost_required"
        assert db.settings_updates == []
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_squad_banner_direct_upload_requires_active_boost(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db = SquadInvariantDB(role="creator", has_boost=False)
    app = web_server.create_web_app(db, bot_token="bot-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        form = FormData()
        form.add_field(
            "file",
            io.BytesIO(b"fake-webp"),
            filename="banner.webp",
            content_type="image/webp",
        )
        response = await client.post(
            "/api/squads/upload-image?user_id=42&kind=banner",
            data=form,
        )
        body = await response.json()

        assert response.status == 403
        assert body["error"] == "boost_required"
        assert db.settings_updates == []
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("upgrade_type", ["boost", "customization"])
async def test_squad_upgrade_route_rejects_removed_upgrades_without_buy_call(monkeypatch, upgrade_type):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db = SquadInvariantDB(role="officer")
    app = web_server.create_web_app(db, bot_token="bot-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/squads/upgrade?user_id=42",
            json={"upgrade_type": upgrade_type},
        )
        body = await response.json()

        assert response.status == 404
        assert body["error"] == "unknown_upgrade"
        assert db.upgrade_calls == []
    finally:
        await client.close()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_squad_shop_state_filters_removed_upgrades_from_legacy_runtime_catalog():
    db = SquadInvariantDB(role="creator")

    state = await db.get_squad_shop_state(10, 42)

    assert "boost" not in state["upgrade_catalog"]
    assert "customization" not in state["upgrade_catalog"]
    assert state["upgrades"]["boost"] == 1
    assert "member_slots" in state["upgrade_catalog"]


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


@pytest.mark.asyncio
async def test_squad_clan_boost_multiplies_cbrp_awards():
    class FakeTransaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeConnection:
        async def fetchrow(self, query, *args):
            if "INSERT INTO squad_cbrp_events" in query:
                return {"id": 100}
            raise AssertionError(f"unexpected fetchrow query: {query}")

        async def execute(self, query, *args):
            return "UPDATE 1"

        def transaction(self):
            return FakeTransaction()

    class FakeAcquire:
        def __init__(self):
            self.conn = FakeConnection()

        async def __aenter__(self):
            return self.conn

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakePool:
        def acquire(self):
            return FakeAcquire()

    class AwardDB:
        award_squad_cbrp = Database.award_squad_cbrp
        _resolve_squad_reward = Database._resolve_squad_reward

        def __init__(self):
            self._pool = FakePool()

        async def fetchrow(self, query, *args):
            return {
                "clan_id": 10,
                "role": "member",
                "owner_id": 200,
                "has_boost": True,
            }

        async def get_squad_runtime_config(self):
            return {
                "squad_rewards": {},
                "squad_creator_passive_tax_pct": 0,
                "squad_clan_boost_cbrp_multiplier": 1.2,
                "squad_clan_boost_token_multiplier": 1.0,
            }

        async def get_clan_upgrades(self, clan_id):
            return {"cbrp_boost": 0}

    result = await AwardDB().award_squad_cbrp(
        200,
        "weekly_trophy_delta",
        clan_id=10,
        source_id="regression-clan-boost-cbrp",
        cbrp=10,
        personal_tokens=0,
        treasury_tokens=0,
    )

    assert result["awarded"] is True
    assert result["cbrp"] == 12


def test_squad_upgrade_level_is_recomputed_under_lock():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    upgrade_block = source.split("async def buy_clan_upgrade", 1)[1].split("async def buy_slot_upgrade", 1)[0]

    assert "FOR UPDATE" in upgrade_block
    assert "SELECT level FROM clan_upgrades" in upgrade_block
    assert "next_level = current_level + 1" in upgrade_block.split("async with conn.transaction()", 1)[1]


def test_squad_boost_activation_grants_member_slots():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    activation_block = source.split("async def activate_clan_boost_from_purchase", 1)[1].split("async def buy_clan_upgrade", 1)[0]

    assert '"squad_clan_boost_member_slots": 5' in source
    assert "boost_member_slots_applied INTEGER NOT NULL DEFAULT 0" in source
    assert 'member_slots_added = max(0, int(config.get("squad_clan_boost_member_slots") or 0))' in activation_block
    assert "max_members = max_members + $3" in activation_block
    assert "boost_member_slots_applied = $3" in activation_block
    assert '"member_slots_added": member_slots_added' in activation_block


def test_public_squad_upgrade_catalog_filters_removed_upgrades():
    catalog = Database._public_squad_upgrade_catalog(
        {
            "squad_upgrades": {
                "boost": {"title": "BOOST"},
                "customization": {"title": "Кастомизация"},
                "member_slots": {"title": "Слоты участников"},
                "cbrp_boost": {"title": "Буст CBRP"},
            }
        }
    )

    assert "boost" not in catalog
    assert "customization" not in catalog
    assert "member_slots" in catalog
    assert "cbrp_boost" in catalog


@pytest.mark.parametrize("upgrade_type", ["boost", "customization"])
@pytest.mark.asyncio
async def test_direct_removed_squad_upgrade_purchase_is_rejected_before_runtime_catalog_lookup(upgrade_type):
    class RemovedUpgradeRejectedDB:
        buy_clan_upgrade = Database.buy_clan_upgrade

        async def get_squad_runtime_config(self):
            raise AssertionError(f"{upgrade_type} should be rejected before runtime config lookup")

    with pytest.raises(ValueError, match="unknown_upgrade"):
        await RemovedUpgradeRejectedDB().buy_clan_upgrade(10, 42, upgrade_type)


def test_weekly_squad_cbrp_creates_visible_notice_after_award_attempt():
    source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    weekly_block = source.split("async def process_weekly_squad_cbrp", 1)[1].split("async def award_squad_seasonal_cbrp", 1)[0]
    notice_block = source.split("async def _create_squad_weekly_tokens_notice", 1)[1].split("async def award_squad_seasonal_cbrp", 1)[0]

    assert "_create_squad_weekly_tokens_notice" in weekly_block
    assert 'result.get("awarded") or result.get("reason") == "duplicate"' in weekly_block
    assert 'category="squad_weekly_tokens"' in notice_block
    assert 'event_type="squad_weekly_tokens"' in notice_block
    assert '"event_type"' not in notice_block.split("attachments={", 1)[1].split("}", 1)[0]


def test_squad_and_announcement_api_blocks_unmoderated_external_images():
    source = Path("web/server.py").read_text(encoding="utf-8")
    settings_block = source.split("async def squads_settings_handler", 1)[1].split("async def squads_members_action_handler", 1)[0]
    upload_block = source.split("async def community_upload_image_handler", 1)[1].split("async def community_image_static_handler", 1)[0]
    announce_block = source.split("async def community_announcements_create_handler", 1)[1].split("async def community_announcements_react_handler", 1)[0]

    assert '_require_squad_role(user_id, ("creator",))' in settings_block
    assert "_validate_local_upload_url" in settings_block
    assert "_require_squad_customization_unlocked" not in settings_block
    assert "customization_required" not in source
    assert "moderate_content" in upload_block
    assert "image_b64=base64.b64encode(data).decode(\"ascii\")" in upload_block
    assert "invalid_image_url" in announce_block
    assert "pin_price=pin_price" in announce_block
    assert 'raise ValueError("boost_required")' in settings_block


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


# ---------- chat_link ----------


ALLOWED_CHAT_LINK_HOSTS = [
    ("https://t.me/aud_squad", "https://t.me/aud_squad"),
    ("https://telegram.me/aud_squad", "https://telegram.me/aud_squad"),
    ("https://telegram.org/aud_squad", "https://telegram.org/aud_squad"),
    ("https://vk.me/aud_squad", "https://vk.me/aud_squad"),
    ("https://vk.com/aud_squad", "https://vk.com/aud_squad"),
    ("https://max.ru/aud_squad", "https://max.ru/aud_squad"),
    ("https://discord.gg/invite", "https://discord.gg/invite"),
    ("https://discord.com/channels/1", "https://discord.com/channels/1"),
    ("https://ok.ru/aud_squad", "https://ok.ru/aud_squad"),
    ("https://join.t.me/aud_squad", "https://join.t.me/aud_squad"),
]


async def _owner_client(monkeypatch, *, chat_link=None):
    async def approve_moderation(*args, **kwargs):
        return {"decision": "approve", "reason": ""}
    monkeypatch.setattr("infrastructure.moderation.moderate_content", approve_moderation)
    db = SquadInvariantDB(role="creator", chat_link=chat_link)
    app = web_server.create_web_app(db, bot_token="bot-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    return db, client


async def _close_client(client):
    await client.close()
    get_settings.cache_clear()


@pytest.mark.parametrize("submitted,normalized", ALLOWED_CHAT_LINK_HOSTS)
@pytest.mark.asyncio
async def test_owner_can_set_valid_chat_link(monkeypatch, submitted, normalized):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db, client = await _owner_client(monkeypatch)
    try:
        response = await client.post(
            "/api/squads/settings?user_id=42",
            json={"chat_link": submitted},
        )
        body = await response.json()

        assert response.status == 200, body
        assert body["success"] is True
        last = db.settings_updates[-1]
        assert last[1].get("chat_link") == normalized
    finally:
        await _close_client(client)


@pytest.mark.asyncio
async def test_owner_can_reset_chat_link_to_none(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db, client = await _owner_client(monkeypatch, chat_link="https://t.me/old")
    try:
        response = await client.post(
            "/api/squads/settings?user_id=42",
            json={"chat_link": ""},
        )
        body = await response.json()

        assert response.status == 200, body
        last = db.settings_updates[-1]
        assert last[1].get("chat_link") is None
    finally:
        await _close_client(client)


@pytest.mark.parametrize("role", ["officer", "member"])
@pytest.mark.asyncio
async def test_non_owner_cannot_change_chat_link(monkeypatch, role):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db = SquadInvariantDB(role=role)
    app = web_server.create_web_app(db, bot_token="bot-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/squads/settings?user_id=42",
            json={"chat_link": "https://t.me/aud_squad"},
        )
        body = await response.json()

        assert response.status == 403
        assert body["error"] == "no_permission"
        assert db.settings_updates == []
    finally:
        await _close_client(client)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/chat",
        "https://youtube.com/watch?v=x",
        "https://google.com",
        "ftp://t.me/aud_squad",
        "javascript:alert(1)",
        "http://t.me/aud_squad",
    ],
)
@pytest.mark.asyncio
async def test_invalid_chat_link_domain_rejected(monkeypatch, url):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db, client = await _owner_client(monkeypatch)
    try:
        response = await client.post(
            "/api/squads/settings?user_id=42",
            json={"chat_link": url},
        )
        body = await response.json()

        assert response.status == 400, body
        assert body["error"] in ("invalid_chat_link", "chat_link_too_long")
        assert db.settings_updates == []
    finally:
        await _close_client(client)


@pytest.mark.asyncio
async def test_chat_link_without_scheme_auto_prefixes_https(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    db, client = await _owner_client(monkeypatch)
    try:
        response = await client.post(
            "/api/squads/settings?user_id=42",
            json={"chat_link": "t.me/aud_squad"},
        )
        body = await response.json()

        assert response.status == 200, body
        last = db.settings_updates[-1]
        assert last[1].get("chat_link") == "https://t.me/aud_squad"
    finally:
        await _close_client(client)


@pytest.mark.asyncio
async def test_chat_link_too_long_rejected(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    get_settings.cache_clear()
    long_url = "https://t.me/" + ("a" * 250)
    db, client = await _owner_client(monkeypatch)
    try:
        response = await client.post(
            "/api/squads/settings?user_id=42",
            json={"chat_link": long_url},
        )
        body = await response.json()

        assert response.status == 400, body
        assert body["error"] == "chat_link_too_long"
        assert db.settings_updates == []
    finally:
        await _close_client(client)


def test_squad_chat_link_is_supported_in_settings_handler_and_database():
    settings_source = Path("web/server.py").read_text(encoding="utf-8")
    db_source = Path("infrastructure/database.py").read_text(encoding="utf-8")
    settings_block = settings_source.split("async def squads_settings_handler", 1)[1].split("async def squads_members_action_handler", 1)[0]
    update_block = db_source.split("async def update_clan_settings", 1)[1].split("async def delete_clan", 1)[0]

    assert "CLAN_CHAT_LINK_URL_RE" in settings_block
    assert "invalid_chat_link" in settings_block
    assert "chat_link_too_long" in settings_block
    assert "c.chat_link" in db_source
    assert "chat_link" in update_block
