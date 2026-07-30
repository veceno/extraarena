import pytest

from infrastructure.config import DatabaseSettings
from infrastructure.database import DECK_SIZE, STARTER_DECK_CARD_IDS, Database


def _settings() -> DatabaseSettings:
    return DatabaseSettings(
        host="localhost",
        port=5432,
        user="test",
        password="test",
        database="test",
    )


class FakeStarterGrantDB(Database):
    def __init__(self):
        super().__init__(_settings())
        self._pool = object()
        self.fetch_calls = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        return [{"card_id": 1}, {"card_id": 36}, {"card_id": 37}, {"card_id": 46}]


@pytest.mark.asyncio
async def test_grant_start_cards_adds_all_start_rarity_cards():
    db = FakeStarterGrantDB()

    result = await db.grant_start_cards(123)

    assert result == {"success": True, "added": 4}
    query, args = db.fetch_calls[0]
    assert args == (123, STARTER_DECK_CARD_IDS)
    assert "FROM cards" in query
    assert "WHERE rarity = 'start'" in query
    assert "id = ANY($2::bigint[])" in query
    assert "ON CONFLICT (user_id, card_id) DO NOTHING" in query


class FakeEnsureUserDB(Database):
    def __init__(self):
        super().__init__(_settings())
        self._pool = object()
        self.executed = []
        self.start_cards_user_id = None
        self.starter_deck_user_id = None

    async def fetchval(self, query, *args):
        if "SELECT 1 FROM users" in query:
            return None
        if "SELECT 1 FROM deck_presets" in query:
            return None
        return None

    async def execute(self, query, *args):
        self.executed.append((query, args))
        return "OK"

    async def fetchrow(self, query, *args):
        if "INSERT INTO users" in query and "RETURNING user_id" in query:
            return {"user_id": args[0]}
        return None

    async def grant_start_cards(self, user_id):
        self.start_cards_user_id = user_id
        return {"success": True, "added": len(STARTER_DECK_CARD_IDS)}

    async def _ensure_user_has_starter_deck(self, user_id):
        self.starter_deck_user_id = user_id
        return {"success": True, "deck_ids": STARTER_DECK_CARD_IDS}

    async def add_card_to_user(self, user_id, card_id):
        raise AssertionError("ensure_user must grant start rarity cards, not a hard-coded card")

    async def grant_starter_cosmetics(self, user_id):
        return None


@pytest.mark.asyncio
async def test_new_user_registration_grants_start_rarity_cards():
    db = FakeEnsureUserDB()

    created = await db.ensure_user(
        user_id=987,
        username="player",
        first_name="Player",
        last_name=None,
    )

    assert created is True
    assert db.start_cards_user_id == 987
    assert db.starter_deck_user_id == 987


class FakeStarterDeckDB(Database):
    def __init__(self, *, existing_complete=False):
        super().__init__(_settings())
        self._pool = object()
        self.existing_complete = existing_complete
        self.executed = []
        self.fetch_calls = []
        self.saved_presets = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        if "FROM deck_presets" in query:
            if self.existing_complete:
                return [
                    {
                        "preset_number": 2,
                        **{f"card_slot_{idx + 1}": card_id for idx, card_id in enumerate(STARTER_DECK_CARD_IDS)},
                    }
                ]
            return []
        if "FROM cards" in query:
            return [
                {"id": 1, "card_type": "hero"},
                *[{"id": card_id, "card_type": "warrior"} for card_id in STARTER_DECK_CARD_IDS[1:]],
            ]
        if "FROM user_cards" in query:
            return [{"card_id": card_id} for card_id in STARTER_DECK_CARD_IDS]
        return []

    async def fetchval(self, query, *args):
        if "SELECT id FROM deck_presets" in query:
            return 10
        return None

    async def fetchrow(self, query, *args):
        return {"extra_pass": "inactive", "extra_pass_expires_at": None}

    async def execute(self, query, *args):
        self.executed.append((query, args))
        return "OK"

    async def grant_start_cards(self, user_id):
        self.executed.append(("grant_start_cards", (user_id,)))
        return {"success": True, "added": len(STARTER_DECK_CARD_IDS)}


@pytest.mark.asyncio
async def test_starter_deck_helper_creates_legal_nine_card_primary_preset():
    db = FakeStarterDeckDB()

    result = await db._ensure_user_has_starter_deck(555)

    assert result["success"] is True
    assert result["deck_ids"] == STARTER_DECK_CARD_IDS
    assert len(result["deck_ids"]) == DECK_SIZE
    assert any("card_slot_1 = $2" in query for query, _ in db.executed)
    assert any("primary_deck = 1" in query for query, _ in db.executed)


@pytest.mark.asyncio
async def test_starter_deck_helper_does_not_overwrite_existing_complete_deck():
    db = FakeStarterDeckDB(existing_complete=True)

    result = await db._ensure_user_has_starter_deck(555)

    assert result == {"success": True, "skipped": "complete_deck_exists"}
    assert not any("card_slot_1 = $2" in query for query, _ in db.executed)
