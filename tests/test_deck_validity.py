from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer
import pytest

from infrastructure.config import DatabaseSettings
from infrastructure.database import DECK_SIZE, Database
from web import server as web_server


USER_ID = 12345
PLAYABLE_DECK = [1, 36, 37, 38, 39, 40, 41, 42, 46]


def _settings() -> DatabaseSettings:
    return DatabaseSettings(
        host="localhost",
        port=5432,
        user="test",
        password="test",
        database="test",
    )


class FakeDeckValidityDB(Database):
    def __init__(
        self,
        *,
        preset_slots: list[int | None] | None = None,
        owned_ids: set[int] | None = None,
        card_types: dict[int, str] | None = None,
        existing_preset: bool = True,
    ):
        super().__init__(_settings())
        self._pool = object()
        self.preset_slots = preset_slots
        self.owned_ids = owned_ids if owned_ids is not None else set(PLAYABLE_DECK)
        self.card_types = card_types if card_types is not None else {
            PLAYABLE_DECK[0]: "hero",
            **{card_id: "warrior" for card_id in PLAYABLE_DECK[1:]},
        }
        self.existing_preset = existing_preset
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, query, *args):
        if "FROM deck_presets" in query:
            if self.preset_slots is None:
                return None
            return {
                f"card_slot_{idx + 1}": card_id
                for idx, card_id in enumerate(self.preset_slots)
            }
        if "FROM users" in query:
            return {"extra_pass": "inactive", "extra_pass_expires_at": None}
        return None

    async def fetch(self, query, *args):
        if "FROM deck_presets dp" in query:
            if self.preset_slots is None:
                return []
            row = {
                "id": 10,
                "preset_number": 1,
                "preset_name": "Preset",
                "used_by_bot": False,
                "updated_at": None,
            }
            row.update({
                f"card_slot_{idx + 1}": card_id
                for idx, card_id in enumerate(self.preset_slots)
            })
            return [row]
        if "FROM cards" in query:
            requested = [int(card_id) for card_id in (args[0] or [])]
            return [
                {"id": card_id, "card_type": self.card_types[card_id]}
                for card_id in requested
                if card_id in self.card_types
            ]
        if "FROM user_cards" in query:
            requested = [int(card_id) for card_id in (args[1] or [])]
            return [
                {"card_id": card_id}
                for card_id in requested
                if card_id in self.owned_ids
            ]
        return []

    async def fetchval(self, query, *args):
        if "SELECT id FROM deck_presets" in query:
            return 10 if self.existing_preset else None
        if "SELECT COUNT(*) FROM deck_presets" in query:
            return 1
        return None

    async def execute(self, query, *args):
        self.executed.append((query, args))
        return "OK"


@pytest.mark.asyncio
async def test_save_deck_preset_allows_incomplete_draft_without_marking_playable():
    db = FakeDeckValidityDB()

    result = await db.save_deck_preset(
        USER_ID,
        preset_number=1,
        preset_name="Draft",
        card_slots=[None, 36, None, None, None, None, None, None, None],
    )

    assert result == {"success": True, "playable": False}
    assert any("UPDATE deck_presets" in query for query, _ in db.executed)


@pytest.mark.asyncio
async def test_set_primary_deck_rejects_incomplete_preset():
    db = FakeDeckValidityDB(preset_slots=[1, 36, 37, 38, 39, 40, 41, 42, None])

    result = await db.set_primary_deck(USER_ID, 1)

    assert result["success"] is False
    assert result["error"] == "deck_incomplete"
    assert not any("UPDATE users SET primary_deck" in query for query, _ in db.executed)


@pytest.mark.asyncio
async def test_set_primary_deck_rejects_unowned_or_stale_cards():
    stale_card = PLAYABLE_DECK[-1]
    db = FakeDeckValidityDB(
        preset_slots=PLAYABLE_DECK,
        owned_ids=set(PLAYABLE_DECK) - {stale_card},
    )

    result = await db.set_primary_deck(USER_ID, 1)

    assert result["success"] is False
    assert result["error"] == "unowned_cards"
    assert result["card_ids"] == [stale_card]
    assert not any("UPDATE users SET primary_deck" in query for query, _ in db.executed)


@pytest.mark.asyncio
async def test_set_primary_deck_rejects_wrong_slot_types():
    db = FakeDeckValidityDB(
        preset_slots=PLAYABLE_DECK,
        card_types={card_id: "warrior" for card_id in PLAYABLE_DECK},
    )

    result = await db.set_primary_deck(USER_ID, 1)

    assert result["success"] is False
    assert result["error"] == "slot_0_must_be_hero"
    assert not any("UPDATE users SET primary_deck" in query for query, _ in db.executed)


@pytest.mark.asyncio
async def test_get_user_deck_presets_annotates_playable_and_stale_state():
    stale_card = PLAYABLE_DECK[-1]
    db = FakeDeckValidityDB(
        preset_slots=PLAYABLE_DECK,
        owned_ids=set(PLAYABLE_DECK) - {stale_card},
    )

    presets = await db.get_user_deck_presets(USER_ID)

    assert presets[0]["filled_count"] == DECK_SIZE
    assert presets[0]["owned_valid_count"] == DECK_SIZE - 1
    assert presets[0]["stale_card_ids"] == [stale_card]
    assert presets[0]["is_playable"] is False


class FakeDeckRouteDB:
    def __init__(self, save_result: dict | None = None, primary_result: dict | None = None):
        self.save_result = save_result or {"success": True, "playable": False}
        self.primary_result = primary_result or {"success": False, "error": "deck_incomplete"}
        self.marked_tasks: list[tuple[int, str, bool]] = []

    async def save_deck_preset(self, **kwargs):
        return dict(self.save_result)

    async def set_primary_deck(self, user_id, preset_number):
        return dict(self.primary_result)

    async def mark_newbie_path_task(self, user_id, task_id, *, claimed=False):
        self.marked_tasks.append((int(user_id), task_id, bool(claimed)))
        return {"newbie_path_progress": {}}


async def _deck_route_client(monkeypatch, db) -> TestClient:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("WEBAPP_HOST", "127.0.0.1")
    monkeypatch.setenv("BOT_TOKEN", "bot-token")
    web_server.get_settings.cache_clear()
    app = web_server.create_web_app(db, bot_token="bot-token")
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_save_deck_route_marks_newbie_task_only_for_playable_deck(monkeypatch):
    draft_db = FakeDeckRouteDB(save_result={"success": True, "playable": False})
    draft_client = await _deck_route_client(monkeypatch, draft_db)
    try:
        response = await draft_client.post(
            f"/api/deck/presets/save?user_id={USER_ID}",
            json={"preset_number": 1, "preset_name": "Draft", "card_slots": [None] * DECK_SIZE},
        )
        assert response.status == 200
        assert draft_db.marked_tasks == []
    finally:
        await draft_client.close()

    playable_db = FakeDeckRouteDB(save_result={"success": True, "playable": True})
    playable_client = await _deck_route_client(monkeypatch, playable_db)
    try:
        response = await playable_client.post(
            f"/api/deck/presets/save?user_id={USER_ID}",
            json={"preset_number": 1, "preset_name": "Ready", "card_slots": PLAYABLE_DECK},
        )
        assert response.status == 200
        assert playable_db.marked_tasks == [(USER_ID, "save_first_deck", False)]
    finally:
        await playable_client.close()


@pytest.mark.asyncio
async def test_set_primary_route_surfaces_json_failure_as_bad_request(monkeypatch):
    db = FakeDeckRouteDB(primary_result={"success": False, "error": "deck_incomplete"})
    client = await _deck_route_client(monkeypatch, db)
    try:
        response = await client.post(
            f"/api/deck/presets/set-primary?user_id={USER_ID}",
            json={"preset_number": 1},
        )
        payload = await response.json()
    finally:
        await client.close()

    assert response.status == 400
    assert payload == {"success": False, "error": "deck_incomplete"}
